from collections.abc import Generator
from dataclasses import dataclass
from uuid import uuid4

import pytest
import sqlalchemy as sa
from argon2 import PasswordHasher, Type
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.auth.contracts import TrustedTenantContext
from app.auth.human import (
    AuthInfrastructureError,
    AuthenticatedPrincipal,
    InvalidCredentialsError,
)
from app.auth.human import service as human_service
from app.auth.human.passwords import hash_password, verify_password
from app.database import Base, engine
from app.domain.enums import MembershipRole
from app.models import Company, CompanyMembership, User

PASSWORD = "correct horse battery staple"
PASSWORD_HASH = hash_password(PASSWORD)
UNSUPPORTED_PASSWORD_HASH = PasswordHasher(type=Type.I).hash(PASSWORD)


@dataclass
class StubUser:
    id: int = 101
    normalized_email: str = "user@example.com"
    password_hash: str = PASSWORD_HASH
    is_active: bool = True
    password_version: int = 1


@dataclass
class StubMembership:
    id: int = 202
    user_id: int = 101
    company_id: int = 303
    role: str = MembershipRole.SALESPERSON.value
    is_active: bool = True


class RecordingSessionContext:
    def __init__(self):
        self.session = object()
        self.closed = False

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True
        return False


@pytest.fixture
def postgres_human_auth_engine() -> Generator[Engine, None, None]:
    assert engine.dialect.name == "postgresql"
    schema_name = f"human_auth_{uuid4().hex}"
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
        ],
    )
    try:
        yield isolated_engine
    finally:
        isolated_engine.dispose()
        with engine.begin() as connection:
            connection.execute(sa.text(f"DROP SCHEMA {quoted_schema_name} CASCADE"))


def _install_backend(
    monkeypatch,
    *,
    user=None,
    memberships=(),
    error=None,
):
    contexts = []
    lookups = []
    membership_lookups = []

    def session_factory():
        context = RecordingSessionContext()
        contexts.append(context)
        return context

    def find_user(session, *, normalized_email):
        lookups.append(normalized_email)
        if error is not None:
            raise error
        return user

    def find_membership(session, *, user_id, company_id):
        assert user is not None
        assert user_id == user.id
        membership_lookups.append((user_id, company_id))
        return next(
            (
                membership
                for membership in memberships
                if membership.user_id == user_id
                and membership.company_id == company_id
                and membership.is_active is True
            ),
            None,
        )

    monkeypatch.setattr(human_service, "SessionLocal", session_factory)
    monkeypatch.setattr(
        human_service,
        "find_user_by_normalized_email",
        find_user,
    )
    monkeypatch.setattr(
        human_service,
        "find_active_membership_for_user_and_company",
        find_membership,
    )
    return contexts, lookups, membership_lookups


def test_email_is_normalized_and_persisted_identity_builds_principal(monkeypatch):
    user = StubUser(id=711, password_version=7)
    membership = StubMembership(
        id=822,
        user_id=user.id,
        company_id=933,
        role=MembershipRole.COMPANY_ADMIN.value,
    )
    contexts, lookups, membership_lookups = _install_backend(
        monkeypatch,
        user=user,
        memberships=[membership],
    )

    principal = human_service.authenticate_credentials(
        email="  USER@Example.COM\t",
        password=PASSWORD,
        company_id=membership.company_id,
    )

    assert lookups == ["user@example.com"]
    assert membership_lookups == [(user.id, membership.company_id)]
    assert principal == AuthenticatedPrincipal(
        user_id=711,
        membership_id=822,
        company_id=933,
        role=MembershipRole.COMPANY_ADMIN,
        password_version=7,
    )
    assert not isinstance(principal, TrustedTenantContext)
    assert contexts[0].closed is True


@pytest.mark.parametrize(
    ("user", "password", "memberships", "company_id"),
    [
        (StubUser(), "wrong password", [StubMembership()], 303),
        (StubUser(), "wrong password", [StubMembership()], 999),
        (None, PASSWORD, [], 303),
        (StubUser(is_active=False), PASSWORD, [StubMembership()], 303),
        (StubUser(), PASSWORD, [], 303),
        (StubUser(), PASSWORD, [StubMembership(is_active=False)], 303),
        (StubUser(), PASSWORD, [StubMembership(company_id=404)], 303),
    ],
)
def test_ordinary_authentication_rejections_are_indistinguishable(
    monkeypatch,
    user,
    password,
    memberships,
    company_id,
):
    contexts, _, membership_lookups = _install_backend(
        monkeypatch,
        user=user,
        memberships=memberships,
    )

    with pytest.raises(InvalidCredentialsError) as exc_info:
        human_service.authenticate_credentials(
            email="user@example.com",
            password=password,
            company_id=company_id,
        )

    assert type(exc_info.value) is InvalidCredentialsError
    assert str(exc_info.value) == "Invalid credentials"
    assert contexts[0].closed is True
    if user is None or user.is_active is not True or password != PASSWORD:
        assert membership_lookups == []


def test_company_id_is_required_and_single_membership_is_not_inferred(
    monkeypatch,
):
    def fail_if_called():
        raise AssertionError("missing company_id must not open a session")

    monkeypatch.setattr(human_service, "SessionLocal", fail_if_called)

    with pytest.raises(TypeError, match="company_id"):
        human_service.authenticate_credentials(
            email="user@example.com",
            password=PASSWORD,
        )


@pytest.mark.parametrize(
    "company_id",
    [True, False, 0, -1, "303", 303.0, None, object()],
)
def test_invalid_company_id_fails_closed_without_database_access(
    monkeypatch,
    company_id,
):
    def fail_if_called():
        raise AssertionError("invalid company_id must not open a session")

    monkeypatch.setattr(human_service, "SessionLocal", fail_if_called)

    with pytest.raises(InvalidCredentialsError, match="^Invalid credentials$"):
        human_service.authenticate_credentials(
            email="user@example.com",
            password=PASSWORD,
            company_id=company_id,
        )


def test_unknown_email_runs_argon2_equivalent_dummy_verification(monkeypatch):
    _install_backend(monkeypatch, user=None)
    verification_calls = []

    def record_verification(password, password_hash):
        verification_calls.append((password, password_hash))
        return False

    monkeypatch.setattr(human_service, "verify_password", record_verification)

    with pytest.raises(InvalidCredentialsError):
        human_service.authenticate_credentials(
            email="missing@example.com",
            password="supplied password",
            company_id=303,
        )

    assert verification_calls == [
        ("supplied password", human_service._DUMMY_PASSWORD_HASH)
    ]
    assert verification_calls[0][1].startswith("$argon2id$")


def test_dummy_hash_is_a_valid_same_profile_argon2id_hash():
    assert verify_password(
        "fixed non-secret dummy verification value",
        human_service._DUMMY_PASSWORD_HASH,
    ) is True


def test_invalid_persisted_membership_role_is_sanitized_infrastructure_error(
    monkeypatch,
):
    user = StubUser()
    _install_backend(
        monkeypatch,
        user=user,
        memberships=[StubMembership(user_id=user.id, role="owner")],
    )

    with pytest.raises(AuthInfrastructureError) as exc_info:
        human_service.authenticate_credentials(
            email=user.normalized_email,
            password=PASSWORD,
            company_id=303,
        )

    assert str(exc_info.value) == "Human authentication unavailable"
    assert "owner" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_database_failure_is_translated_without_leaking_details(monkeypatch):
    contexts, _, _ = _install_backend(
        monkeypatch,
        error=RuntimeError("postgresql://secret internal details"),
    )

    with pytest.raises(AuthInfrastructureError) as exc_info:
        human_service.authenticate_credentials(
            email="user@example.com",
            password=PASSWORD,
            company_id=303,
        )

    assert str(exc_info.value) == "Human authentication unavailable"
    assert "postgresql" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert contexts[0].closed is True


@pytest.mark.parametrize(
    "password_hash",
    [
        None,
        "not-an-argon2-hash",
        "$argon2id$malformed",
        UNSUPPORTED_PASSWORD_HASH,
    ],
)
def test_unusable_persisted_password_hash_runs_dummy_work_and_is_invalid(
    monkeypatch,
    password_hash,
):
    from app.auth.human import passwords as password_service

    user = StubUser(password_hash=password_hash)
    _install_backend(monkeypatch, user=user, memberships=[StubMembership()])
    real_hasher = password_service._PASSWORD_HASHER
    verification_hashes = []

    class RecordingHasher:
        def verify(self, candidate_hash, supplied_password):
            verification_hashes.append(candidate_hash)
            return real_hasher.verify(candidate_hash, supplied_password)

    monkeypatch.setattr(password_service, "_PASSWORD_HASHER", RecordingHasher())

    with pytest.raises(InvalidCredentialsError) as exc_info:
        human_service.authenticate_credentials(
            email=user.normalized_email,
            password=PASSWORD,
            company_id=303,
        )

    assert str(exc_info.value) == "Invalid credentials"
    assert password_service._DUMMY_PASSWORD_HASH in verification_hashes
    assert repr(user.password_hash) not in str(exc_info.value)


def test_real_postgresql_lookup_selects_each_active_company_membership(
    monkeypatch,
    postgres_human_auth_engine,
):
    connection = postgres_human_auth_engine.connect()
    transaction = connection.begin()
    setup_session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        marker = uuid4().hex
        company_a = Company(name=f"Human Auth A {marker}")
        company_b = Company(name=f"Human Auth B {marker}")
        inactive_company = Company(name=f"Human Auth Inactive {marker}")
        user = User(
            normalized_email=f"  Auth-{marker}@EXAMPLE.COM ",
            password_hash=hash_password(PASSWORD),
        )
        setup_session.add_all([company_a, company_b, inactive_company, user])
        setup_session.flush()
        membership_a = CompanyMembership(
            user_id=user.id,
            company_id=company_a.id,
            role=MembershipRole.SALESPERSON.value,
        )
        membership_b = CompanyMembership(
            user_id=user.id,
            company_id=company_b.id,
            role=MembershipRole.COMPANY_ADMIN.value,
        )
        inactive_membership = CompanyMembership(
            user_id=user.id,
            company_id=inactive_company.id,
            role=MembershipRole.COMPANY_ADMIN.value,
            is_active=False,
        )
        setup_session.add_all(
            [membership_a, membership_b, inactive_membership]
        )
        setup_session.flush()
        auth_session_factory = sessionmaker(
            bind=connection,
            join_transaction_mode="create_savepoint",
        )
        monkeypatch.setattr(
            human_service,
            "SessionLocal",
            auth_session_factory,
        )

        for company, membership, role in (
            (company_a, membership_a, MembershipRole.SALESPERSON),
            (company_b, membership_b, MembershipRole.COMPANY_ADMIN),
        ):
            principal = human_service.authenticate_credentials(
                email=f" AUTH-{marker}@example.com\t",
                password=PASSWORD,
                company_id=company.id,
            )
            assert principal == AuthenticatedPrincipal(
                user_id=user.id,
                membership_id=membership.id,
                company_id=company.id,
                role=role,
                password_version=user.password_version,
            )

        for unavailable_company_id in (
            inactive_company.id,
            max(company_a.id, company_b.id, inactive_company.id) + 1,
        ):
            with pytest.raises(
                InvalidCredentialsError,
                match="^Invalid credentials$",
            ):
                human_service.authenticate_credentials(
                    email=f" AUTH-{marker}@example.com\t",
                    password=PASSWORD,
                    company_id=unavailable_company_id,
                )
    finally:
        setup_session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()
