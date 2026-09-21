from __future__ import annotations

import hashlib
import threading
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.auth.human import (
    AcceptedInvitation,
    AuthenticatedPrincipal,
    InvitationAuthorizationError,
    InvitationConflictError,
    InvitationInfrastructureError,
    InvalidInvitationError,
    InvalidInvitationPasswordError,
    accept_invitation,
    create_invitation,
)
from app.auth.human import invitation_service, service as human_service
from app.auth.human.passwords import hash_password, verify_password
from app.database import Base, engine
from app.domain.enums import MembershipRole
from app.models import Company, CompanyInvitation, CompanyMembership, User

NOW = datetime(2026, 9, 16, 9, 30, tzinfo=timezone.utc)
RAW_TOKEN = "A" * 43
CREATOR_HASH = "$argon2id$creator-password-hash"


@pytest.fixture
def postgres_invitation_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[Engine, None, None]:
    assert engine.dialect.name == "postgresql"
    schema_name = f"human_invitation_{uuid4().hex}"
    quoted_schema_name = engine.dialect.identifier_preparer.quote(schema_name)
    database_url = engine.url.render_as_string(hide_password=False)

    with engine.begin() as connection:
        connection.execute(sa.text(f"CREATE SCHEMA {quoted_schema_name}"))

    isolated_engine = sa.create_engine(
        database_url,
        connect_args={"options": f"-csearch_path={schema_name}"},
        poolclass=NullPool,
    )
    Base.metadata.create_all(
        isolated_engine,
        tables=[
            Company.__table__,
            User.__table__,
            CompanyMembership.__table__,
            CompanyInvitation.__table__,
        ],
    )
    monkeypatch.setattr(
        invitation_service,
        "SessionLocal",
        sessionmaker(bind=isolated_engine),
    )
    try:
        yield isolated_engine
    finally:
        isolated_engine.dispose()
        with engine.begin() as connection:
            connection.execute(
                sa.text(f"DROP SCHEMA {quoted_schema_name} CASCADE")
            )


def _seed_creator(
    database_engine: Engine,
    *,
    role: MembershipRole = MembershipRole.COMPANY_ADMIN,
    user_active: bool = True,
    membership_active: bool = True,
    membership_company_id: int | None = None,
) -> tuple[int, int]:
    with Session(database_engine) as session:
        company = Company(name="Invitation Company")
        session.add(company)
        session.flush()
        creator = User(
            normalized_email=f"creator-{uuid4().hex}@example.com",
            password_hash=CREATOR_HASH,
            is_active=user_active,
        )
        session.add(creator)
        session.flush()
        target_company_id = membership_company_id or company.id
        if membership_company_id is not None:
            session.add(Company(id=membership_company_id, name="Other Company"))
            session.flush()
        membership = CompanyMembership(
            user_id=creator.id,
            company_id=target_company_id,
            role=role.value,
            is_active=membership_active,
        )
        session.add(membership)
        session.commit()
        return company.id, creator.id


def _create_invitation(
    database_engine: Engine,
    *,
    email: str = " Invited@Example.COM ",
    role: MembershipRole = MembershipRole.SALESPERSON,
):
    company_id, creator_id = _seed_creator(database_engine)
    created = create_invitation(
        company_id=company_id,
        created_by_user_id=creator_id,
        email=email,
        role=role,
    )
    return created, company_id, creator_id


def test_normal_invitation_creation_persists_only_digest_and_no_placeholder_user(
    monkeypatch,
    postgres_invitation_engine,
):
    company_id, creator_id = _seed_creator(postgres_invitation_engine)
    monkeypatch.setattr(invitation_service, "_utc_now", lambda: NOW)
    monkeypatch.setattr(
        invitation_service,
        "_generate_invitation_token",
        lambda: RAW_TOKEN,
    )

    created = create_invitation(
        company_id=company_id,
        created_by_user_id=creator_id,
        email=" Invited@Example.COM ",
        role=MembershipRole.COMPANY_ADMIN,
    )

    assert created.invitation_id > 0
    assert created.company_id == company_id
    assert created.normalized_email == "invited@example.com"
    assert created.role is MembershipRole.COMPANY_ADMIN
    assert created.token == RAW_TOKEN
    assert created.expires_at == NOW + timedelta(hours=24)
    with Session(postgres_invitation_engine) as session:
        invitation = session.get(CompanyInvitation, created.invitation_id)
        assert invitation is not None
        assert invitation.created_by_user_id == creator_id
        assert invitation.token_digest == hashlib.sha256(
            RAW_TOKEN.encode("ascii")
        ).digest()
        assert invitation.token_digest != RAW_TOKEN.encode("ascii")
        assert invitation.created_at == NOW
        assert invitation.expires_at == created.expires_at
        assert invitation.consumed_at is None
        assert invitation.revoked_at is None
        assert session.scalar(
            sa.select(User).where(
                User.normalized_email == "invited@example.com"
            )
        ) is None


def test_new_user_acceptance_uses_argon2id_and_trusted_invitation_data(
    postgres_invitation_engine,
):
    created, company_id, _ = _create_invitation(
        postgres_invitation_engine,
        email=" New.User@Example.COM ",
        role=MembershipRole.COMPANY_ADMIN,
    )

    accepted = accept_invitation(
        token=created.token,
        password="new user password",
    )

    assert accepted == AcceptedInvitation(
        user_id=accepted.user_id,
        company_id=company_id,
        membership_id=accepted.membership_id,
        role=MembershipRole.COMPANY_ADMIN,
    )
    with Session(postgres_invitation_engine) as session:
        user = session.get(User, accepted.user_id)
        membership = session.get(CompanyMembership, accepted.membership_id)
        invitation = session.get(CompanyInvitation, created.invitation_id)
        assert user is not None
        assert user.normalized_email == "new.user@example.com"
        assert user.password_hash.startswith("$argon2id$")
        assert verify_password("new user password", user.password_hash) is True
        assert user.password_changed_at is not None
        assert membership is not None
        assert membership.user_id == user.id
        assert membership.company_id == company_id
        assert membership.role == MembershipRole.COMPANY_ADMIN.value
        assert invitation is not None
        assert invitation.consumed_at is not None


def test_existing_user_acceptance_preserves_password_hash_exactly(
    postgres_invitation_engine,
):
    created, company_id, _ = _create_invitation(
        postgres_invitation_engine,
        email="existing@example.com",
    )
    original_hash = "$argon2id$existing-byte-for-byte-hash"
    with Session(postgres_invitation_engine) as session:
        user = User(
            normalized_email="existing@example.com",
            password_hash=original_hash,
        )
        session.add(user)
        session.commit()
        user_id = user.id

    accepted = accept_invitation(token=created.token, password="   ")

    assert accepted.user_id == user_id
    assert accepted.company_id == company_id
    with Session(postgres_invitation_engine) as session:
        user = session.get(User, user_id)
        assert user is not None
        assert user.password_hash == original_hash


def test_existing_user_accepts_second_company_and_authenticates_into_each(
    monkeypatch,
    postgres_invitation_engine,
):
    password = "existing user password"
    created, invited_company_id, _ = _create_invitation(
        postgres_invitation_engine,
        email="multi-company@example.com",
        role=MembershipRole.SALESPERSON,
    )
    with Session(postgres_invitation_engine) as session:
        existing_company = Company(name="Existing Company")
        user = User(
            normalized_email="multi-company@example.com",
            password_hash=hash_password(password),
        )
        session.add_all([existing_company, user])
        session.flush()
        existing_membership = CompanyMembership(
            user_id=user.id,
            company_id=existing_company.id,
            role=MembershipRole.COMPANY_ADMIN.value,
        )
        session.add(existing_membership)
        session.flush()
        user_id = user.id
        existing_company_id = existing_company.id
        existing_membership_id = existing_membership.id
        session.commit()

    accepted = accept_invitation(token=created.token, password="ignored")
    monkeypatch.setattr(
        human_service,
        "SessionLocal",
        sessionmaker(bind=postgres_invitation_engine),
    )

    existing_principal = human_service.authenticate_credentials(
        email=" MULTI-COMPANY@example.com ",
        password=password,
        company_id=existing_company_id,
    )
    invited_principal = human_service.authenticate_credentials(
        email="multi-company@example.com",
        password=password,
        company_id=invited_company_id,
    )

    assert existing_principal == AuthenticatedPrincipal(
        user_id=user_id,
        membership_id=existing_membership_id,
        company_id=existing_company_id,
        role=MembershipRole.COMPANY_ADMIN,
        password_version=1,
    )
    assert invited_principal == AuthenticatedPrincipal(
        user_id=user_id,
        membership_id=accepted.membership_id,
        company_id=invited_company_id,
        role=MembershipRole.SALESPERSON,
        password_version=1,
    )


@pytest.mark.parametrize("token", [None, "", " ", "short", "!" * 43])
def test_malformed_tokens_share_generic_invalid_error(token):
    with pytest.raises(InvalidInvitationError) as exc_info:
        accept_invitation(token=token, password="password")
    assert str(exc_info.value) == "Invalid invitation"
    assert exc_info.value.__cause__ is None


def test_unknown_token_is_generic_invalid(postgres_invitation_engine):
    with pytest.raises(InvalidInvitationError, match="^Invalid invitation$"):
        accept_invitation(token="B" * 43, password="password")


def test_expiry_boundary_is_inclusive(monkeypatch, postgres_invitation_engine):
    monkeypatch.setattr(invitation_service, "_utc_now", lambda: NOW)
    created, _, _ = _create_invitation(postgres_invitation_engine)
    monkeypatch.setattr(
        invitation_service,
        "_utc_now",
        lambda: NOW + timedelta(hours=24),
    )

    with pytest.raises(InvalidInvitationError, match="^Invalid invitation$"):
        accept_invitation(token=created.token, password="password")


def test_revoked_invitation_is_generic_invalid(postgres_invitation_engine):
    created, _, _ = _create_invitation(postgres_invitation_engine)
    with Session(postgres_invitation_engine) as session:
        invitation = session.get(CompanyInvitation, created.invitation_id)
        assert invitation is not None
        invitation.revoked_at = datetime.now(timezone.utc)
        session.commit()

    with pytest.raises(InvalidInvitationError, match="^Invalid invitation$"):
        accept_invitation(token=created.token, password="password")


def test_consumed_invitation_and_replay_are_generic_invalid(
    postgres_invitation_engine,
):
    created, _, _ = _create_invitation(postgres_invitation_engine)
    accept_invitation(token=created.token, password="password")

    with pytest.raises(InvalidInvitationError, match="^Invalid invitation$"):
        accept_invitation(token=created.token, password="password")


def test_reinvitation_revokes_previous_token(postgres_invitation_engine):
    company_id, creator_id = _seed_creator(postgres_invitation_engine)
    first = create_invitation(
        company_id=company_id,
        created_by_user_id=creator_id,
        email="repeat@example.com",
        role=MembershipRole.SALESPERSON,
    )
    second = create_invitation(
        company_id=company_id,
        created_by_user_id=creator_id,
        email=" REPEAT@example.com ",
        role=MembershipRole.COMPANY_ADMIN,
    )

    with pytest.raises(InvalidInvitationError, match="^Invalid invitation$"):
        accept_invitation(token=first.token, password="password")
    accepted = accept_invitation(token=second.token, password="password")
    assert accepted.role is MembershipRole.COMPANY_ADMIN


@pytest.mark.parametrize(
    ("role", "user_active", "membership_active", "wrong_company"),
    [
        (MembershipRole.SALESPERSON, True, True, False),
        (MembershipRole.COMPANY_ADMIN, False, True, False),
        (MembershipRole.COMPANY_ADMIN, True, False, False),
        (MembershipRole.COMPANY_ADMIN, True, True, True),
    ],
)
def test_creator_authorization_is_independently_enforced(
    postgres_invitation_engine,
    role,
    user_active,
    membership_active,
    wrong_company,
):
    membership_company_id = 1000 if wrong_company else None
    company_id, creator_id = _seed_creator(
        postgres_invitation_engine,
        role=role,
        user_active=user_active,
        membership_active=membership_active,
        membership_company_id=membership_company_id,
    )

    with pytest.raises(
        InvitationAuthorizationError,
        match="^Invitation operation not permitted$",
    ):
        create_invitation(
            company_id=company_id,
            created_by_user_id=creator_id,
            email="unauthorized@example.com",
            role=MembershipRole.SALESPERSON,
        )

    with Session(postgres_invitation_engine) as session:
        assert session.scalar(sa.select(sa.func.count(CompanyInvitation.id))) == 0


def test_active_existing_membership_blocks_invitation_without_role_change(
    postgres_invitation_engine,
):
    company_id, creator_id = _seed_creator(postgres_invitation_engine)
    with Session(postgres_invitation_engine) as session:
        invited_user = User(
            normalized_email="member@example.com",
            password_hash="$argon2id$member",
        )
        session.add(invited_user)
        session.flush()
        membership = CompanyMembership(
            user_id=invited_user.id,
            company_id=company_id,
            role=MembershipRole.SALESPERSON.value,
        )
        session.add(membership)
        session.commit()
        membership_id = membership.id

    with pytest.raises(InvitationConflictError):
        create_invitation(
            company_id=company_id,
            created_by_user_id=creator_id,
            email="MEMBER@example.com",
            role=MembershipRole.COMPANY_ADMIN,
        )

    with Session(postgres_invitation_engine) as session:
        membership = session.get(CompanyMembership, membership_id)
        assert membership is not None
        assert membership.role == MembershipRole.SALESPERSON.value


@pytest.mark.parametrize("inactive_state", ["user", "membership"])
def test_inactive_identity_state_is_never_reactivated(
    postgres_invitation_engine,
    inactive_state,
):
    created, company_id, _ = _create_invitation(
        postgres_invitation_engine,
        email=f"inactive-{inactive_state}@example.com",
    )
    with Session(postgres_invitation_engine) as session:
        user = User(
            normalized_email=f"inactive-{inactive_state}@example.com",
            password_hash="$argon2id$inactive",
            is_active=inactive_state != "user",
        )
        session.add(user)
        session.flush()
        membership = None
        if inactive_state == "membership":
            membership = CompanyMembership(
                user_id=user.id,
                company_id=company_id,
                role=MembershipRole.COMPANY_ADMIN.value,
                is_active=False,
                deactivated_at=datetime.now(timezone.utc),
            )
            session.add(membership)
        session.commit()
        user_id = user.id
        membership_id = membership.id if membership is not None else None

    with pytest.raises(InvitationConflictError):
        accept_invitation(token=created.token, password="password")

    with Session(postgres_invitation_engine) as session:
        user = session.get(User, user_id)
        assert user is not None
        assert user.is_active is (inactive_state != "user")
        if membership_id is not None:
            membership = session.get(CompanyMembership, membership_id)
            assert membership is not None
            assert membership.is_active is False
            assert membership.role == MembershipRole.COMPANY_ADMIN.value
        invitation = session.get(CompanyInvitation, created.invitation_id)
        assert invitation is not None
        assert invitation.consumed_at is None


def test_new_user_blank_password_uses_existing_password_policy(
    postgres_invitation_engine,
):
    created, _, _ = _create_invitation(postgres_invitation_engine)

    with pytest.raises(
        InvalidInvitationPasswordError,
        match="^Invitation password is invalid$",
    ):
        accept_invitation(token=created.token, password="   ")


@pytest.mark.parametrize("tampered_field", ["company_id", "email", "role"])
def test_acceptance_rejects_caller_supplied_identity_tampering(
    postgres_invitation_engine,
    tampered_field,
):
    created, _, _ = _create_invitation(postgres_invitation_engine)
    kwargs = {
        "token": created.token,
        "password": "password",
        tampered_field: "attacker-controlled",
    }

    with pytest.raises(TypeError):
        accept_invitation(**kwargs)


@pytest.mark.parametrize("operation", ["create", "accept"])
def test_unexpected_backend_failures_are_sanitized(
    monkeypatch,
    postgres_invitation_engine,
    operation,
):
    secret = "postgresql://private/invitation-token-password-detail"
    if operation == "create":
        company_id, creator_id = _seed_creator(postgres_invitation_engine)

        def fail(*args, **kwargs):
            raise RuntimeError(secret)

        monkeypatch.setattr(
            invitation_service,
            "acquire_invitation_identity_lock",
            fail,
        )
        invoke = lambda: create_invitation(
            company_id=company_id,
            created_by_user_id=creator_id,
            email="failure@example.com",
            role=MembershipRole.SALESPERSON,
        )
    else:
        created, _, _ = _create_invitation(postgres_invitation_engine)

        def fail(*args, **kwargs):
            raise RuntimeError(secret)

        monkeypatch.setattr(
            invitation_service,
            "find_invitation_by_token_digest_for_update",
            fail,
        )
        invoke = lambda: accept_invitation(
            token=created.token,
            password="password",
        )

    with pytest.raises(InvitationInfrastructureError) as exc_info:
        invoke()
    assert str(exc_info.value) == "Invitation operation unavailable"
    assert secret not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_failure_after_user_and_membership_creation_rolls_back_everything(
    monkeypatch,
    postgres_invitation_engine,
):
    created, _, _ = _create_invitation(
        postgres_invitation_engine,
        email="rollback@example.com",
    )

    class FailOnConsumptionSession(Session):
        def flush(self, objects=None):
            if any(
                isinstance(item, CompanyInvitation)
                and item.consumed_at is not None
                for item in self.dirty
            ):
                raise RuntimeError("private failure after partial writes")
            return super().flush(objects)

    monkeypatch.setattr(
        invitation_service,
        "SessionLocal",
        sessionmaker(
            bind=postgres_invitation_engine,
            class_=FailOnConsumptionSession,
        ),
    )

    with pytest.raises(
        InvitationInfrastructureError,
        match="^Invitation operation unavailable$",
    ):
        accept_invitation(token=created.token, password="password")

    with Session(postgres_invitation_engine) as session:
        assert session.scalar(
            sa.select(User).where(User.normalized_email == "rollback@example.com")
        ) is None
        assert session.scalar(
            sa.select(CompanyMembership).join(User).where(
                User.normalized_email == "rollback@example.com"
            )
        ) is None
        invitation = session.get(CompanyInvitation, created.invitation_id)
        assert invitation is not None
        assert invitation.consumed_at is None


def test_concurrent_acceptance_of_same_token_allows_exactly_one_success(
    postgres_invitation_engine,
):
    created, _, _ = _create_invitation(
        postgres_invitation_engine,
        email="concurrent-accept@example.com",
    )
    barrier = threading.Barrier(2)
    successes = []
    errors = []
    result_lock = threading.Lock()

    def accept():
        barrier.wait(timeout=5)
        try:
            result = accept_invitation(
                token=created.token,
                password="password",
            )
            with result_lock:
                successes.append(result)
        except Exception as exc:
            with result_lock:
                errors.append(exc)

    threads = [threading.Thread(target=accept) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert all(thread.is_alive() is False for thread in threads)
    assert len(successes) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], InvalidInvitationError)
    with Session(postgres_invitation_engine) as session:
        assert session.scalar(sa.select(sa.func.count(User.id))) == 2
        assert session.scalar(
            sa.select(sa.func.count(CompanyMembership.id))
        ) == 2


def test_concurrent_reinvitation_leaves_exactly_one_usable_token(
    postgres_invitation_engine,
):
    company_id, creator_id = _seed_creator(postgres_invitation_engine)
    barrier = threading.Barrier(2)
    results = []
    errors = []
    result_lock = threading.Lock()

    def create(role):
        barrier.wait(timeout=5)
        try:
            result = create_invitation(
                company_id=company_id,
                created_by_user_id=creator_id,
                email="concurrent-create@example.com",
                role=role,
            )
            with result_lock:
                results.append(result)
        except Exception as exc:
            with result_lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=create, args=(MembershipRole.SALESPERSON,)),
        threading.Thread(target=create, args=(MembershipRole.COMPANY_ADMIN,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert all(thread.is_alive() is False for thread in threads)
    assert errors == []
    assert len(results) == 2
    with Session(postgres_invitation_engine) as session:
        invitations = list(
            session.scalars(
                sa.select(CompanyInvitation).order_by(CompanyInvitation.id)
            )
        )
        usable = [
            invitation
            for invitation in invitations
            if invitation.consumed_at is None
            and invitation.revoked_at is None
            and datetime.now(timezone.utc) < invitation.expires_at
        ]
        assert len(invitations) == 2
        assert len(usable) == 1
        usable_digest = usable[0].token_digest

    usable_result = next(
        result
        for result in results
        if hashlib.sha256(result.token.encode("ascii")).digest() == usable_digest
    )
    replaced_result = next(result for result in results if result != usable_result)
    with pytest.raises(InvalidInvitationError):
        accept_invitation(token=replaced_result.token, password="password")
    assert accept_invitation(
        token=usable_result.token,
        password="password",
    ).company_id == company_id


@pytest.mark.parametrize("first_operation", ["accept", "create"])
def test_concurrent_reinvite_and_accept_share_identity_serialization(
    monkeypatch,
    postgres_invitation_engine,
    first_operation,
):
    old_invitation, company_id, creator_id = _create_invitation(
        postgres_invitation_engine,
        email="create-accept-race@example.com",
        role=MembershipRole.SALESPERSON,
    )
    with Session(postgres_invitation_engine) as session:
        invited_user = User(
            normalized_email="create-accept-race@example.com",
            password_hash="$argon2id$existing",
        )
        session.add(invited_user)
        session.commit()
        invited_user_id = invited_user.id

    first_acquired_identity_lock = threading.Event()
    second_attempted_identity_lock = threading.Event()
    release_first = threading.Event()
    results = {}
    errors = {}
    result_lock = threading.Lock()
    real_acquire_identity_lock = (
        invitation_service.acquire_invitation_identity_lock
    )

    def coordinated_identity_lock(session, *, lock_key):
        operation = threading.current_thread().name
        if operation == first_operation:
            real_acquire_identity_lock(session, lock_key=lock_key)
            first_acquired_identity_lock.set()
            assert release_first.wait(timeout=10)
            return
        second_attempted_identity_lock.set()
        real_acquire_identity_lock(session, lock_key=lock_key)

    monkeypatch.setattr(
        invitation_service,
        "acquire_invitation_identity_lock",
        coordinated_identity_lock,
    )

    def invoke(operation):
        try:
            if operation == "accept":
                result = accept_invitation(
                    token=old_invitation.token,
                    password="ignored",
                )
            else:
                result = create_invitation(
                    company_id=company_id,
                    created_by_user_id=creator_id,
                    email="CREATE-ACCEPT-RACE@example.com",
                    role=MembershipRole.COMPANY_ADMIN,
                )
            with result_lock:
                results[operation] = result
        except Exception as exc:
            with result_lock:
                errors[operation] = exc

    second_operation = "create" if first_operation == "accept" else "accept"
    first_thread = threading.Thread(
        target=invoke,
        args=(first_operation,),
        name=first_operation,
    )
    second_thread = threading.Thread(
        target=invoke,
        args=(second_operation,),
        name=second_operation,
    )
    first_thread.start()
    assert first_acquired_identity_lock.wait(timeout=10)
    second_thread.start()
    assert second_attempted_identity_lock.wait(timeout=10)
    release_first.set()
    first_thread.join(timeout=30)
    second_thread.join(timeout=30)

    assert first_thread.is_alive() is False
    assert second_thread.is_alive() is False
    with Session(postgres_invitation_engine) as session:
        invitations = list(
            session.scalars(
                sa.select(CompanyInvitation).order_by(CompanyInvitation.id)
            )
        )
        memberships = list(
            session.scalars(
                sa.select(CompanyMembership).where(
                    CompanyMembership.user_id == invited_user_id,
                    CompanyMembership.company_id == company_id,
                )
            )
        )
        usable_invitations = [
            invitation
            for invitation in invitations
            if invitation.consumed_at is None
            and invitation.revoked_at is None
            and datetime.now(timezone.utc) < invitation.expires_at
        ]

        assert len(memberships) <= 1
        assert not (memberships and usable_invitations)
        if first_operation == "accept":
            assert "accept" in results
            assert isinstance(errors.get("create"), InvitationConflictError)
            assert str(errors["create"]) == (
                "Invitation conflicts with existing identity state"
            )
            assert len(invitations) == 1
            assert invitations[0].consumed_at is not None
            assert invitations[0].revoked_at is None
            assert len(memberships) == 1
            assert memberships[0].role == MembershipRole.SALESPERSON.value
            assert usable_invitations == []
        else:
            assert "create" in results
            assert isinstance(errors.get("accept"), InvalidInvitationError)
            assert str(errors["accept"]) == "Invalid invitation"
            assert len(invitations) == 2
            assert invitations[0].consumed_at is None
            assert invitations[0].revoked_at is not None
            assert memberships == []
            assert len(usable_invitations) == 1
            assert usable_invitations[0].role == (
                MembershipRole.COMPANY_ADMIN.value
            )


def test_concurrent_membership_creation_prevents_duplicate_or_role_mutation(
    monkeypatch,
    postgres_invitation_engine,
):
    created, company_id, _ = _create_invitation(
        postgres_invitation_engine,
        email="membership-race@example.com",
        role=MembershipRole.SALESPERSON,
    )
    with Session(postgres_invitation_engine) as session:
        user = User(
            normalized_email="membership-race@example.com",
            password_hash="$argon2id$existing",
        )
        session.add(user)
        session.commit()
        user_id = user.id

    external_inserted = threading.Event()
    acceptance_user_lock_attempted = threading.Event()
    release_external = threading.Event()
    external_errors = []
    acceptance_errors = []

    def external_create():
        try:
            with Session(postgres_invitation_engine) as session:
                session.add(
                    CompanyMembership(
                        user_id=user_id,
                        company_id=company_id,
                        role=MembershipRole.COMPANY_ADMIN.value,
                    )
                )
                session.flush()
                external_inserted.set()
                assert release_external.wait(timeout=10)
                session.commit()
        except Exception as exc:
            external_errors.append(exc)

    real_find_user = (
        invitation_service.find_user_by_normalized_email_for_update
    )

    def recording_find_user(*args, **kwargs):
        acceptance_user_lock_attempted.set()
        return real_find_user(*args, **kwargs)

    monkeypatch.setattr(
        invitation_service,
        "find_user_by_normalized_email_for_update",
        recording_find_user,
    )

    def accept():
        try:
            accept_invitation(token=created.token, password="ignored")
        except Exception as exc:
            acceptance_errors.append(exc)

    external_thread = threading.Thread(target=external_create)
    external_thread.start()
    assert external_inserted.wait(timeout=10)
    acceptance_thread = threading.Thread(target=accept)
    acceptance_thread.start()
    assert acceptance_user_lock_attempted.wait(timeout=10)
    release_external.set()
    external_thread.join(timeout=30)
    acceptance_thread.join(timeout=30)

    assert external_thread.is_alive() is False
    assert acceptance_thread.is_alive() is False
    assert external_errors == []
    assert len(acceptance_errors) == 1
    assert isinstance(acceptance_errors[0], InvitationConflictError)
    with Session(postgres_invitation_engine) as session:
        memberships = list(
            session.scalars(
                sa.select(CompanyMembership).where(
                    CompanyMembership.user_id == user_id,
                    CompanyMembership.company_id == company_id,
                )
            )
        )
        assert len(memberships) == 1
        assert memberships[0].role == MembershipRole.COMPANY_ADMIN.value
        invitation = session.get(CompanyInvitation, created.invitation_id)
        assert invitation is not None
        assert invitation.consumed_at is None
