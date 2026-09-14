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

    def session_factory():
        context = RecordingSessionContext()
        contexts.append(context)
        return context

    def find_user(session, *, normalized_email):
        lookups.append(normalized_email)
        if error is not None:
            raise error
        return user

    def find_memberships(session, *, user_id):
        assert user is not None
        assert user_id == user.id
        return list(memberships)

    monkeypatch.setattr(human_service, "SessionLocal", session_factory)
    monkeypatch.setattr(
        human_service,
        "find_user_by_normalized_email",
        find_user,
    )
    monkeypatch.setattr(
        human_service,
        "find_active_memberships_for_user",
        find_memberships,
    )
    return contexts, lookups


def test_email_is_normalized_and_persisted_identity_builds_principal(monkeypatch):
    user = StubUser(id=711)
    membership = StubMembership(
        id=822,
        user_id=user.id,
        company_id=933,
        role=MembershipRole.COMPANY_ADMIN.value,
    )
    contexts, lookups = _install_backend(
        monkeypatch,
        user=user,
        memberships=[membership],
    )

    principal = human_service.authenticate_credentials(
        email="  USER@Example.COM\t",
        password=PASSWORD,
    )

    assert lookups == ["user@example.com"]
    assert principal == AuthenticatedPrincipal(
        user_id=711,
        membership_id=822,
        company_id=933,
        role=MembershipRole.COMPANY_ADMIN,
    )
    assert not isinstance(principal, TrustedTenantContext)
    assert contexts[0].closed is True


@pytest.mark.parametrize(
    ("user", "password", "memberships"),
    [
        (StubUser(), "wrong password", [StubMembership()]),
        (None, PASSWORD, []),
        (StubUser(is_active=False), PASSWORD, [StubMembership()]),
        (StubUser(), PASSWORD, []),
        (StubUser(), PASSWORD, [StubMembership(is_active=False)]),
        (
            StubUser(),
            PASSWORD,
            [StubMembership(id=1), StubMembership(id=2, company_id=404)],
        ),
    ],
)
def test_ordinary_authentication_rejections_are_indistinguishable(
    monkeypatch,
    user,
    password,
    memberships,
):
    contexts, _ = _install_backend(
        monkeypatch,
        user=user,
        memberships=memberships,
    )

    with pytest.raises(InvalidCredentialsError) as exc_info:
        human_service.authenticate_credentials(
            email="user@example.com",
            password=password,
        )

    assert type(exc_info.value) is InvalidCredentialsError
    assert str(exc_info.value) == "Invalid credentials"
    assert contexts[0].closed is True


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
        )

    assert str(exc_info.value) == "Human authentication unavailable"
    assert "owner" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_database_failure_is_translated_without_leaking_details(monkeypatch):
    contexts, _ = _install_backend(
        monkeypatch,
        error=RuntimeError("postgresql://secret internal details"),
    )

    with pytest.raises(AuthInfrastructureError) as exc_info:
        human_service.authenticate_credentials(
            email="user@example.com",
            password=PASSWORD,
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
        )

    assert str(exc_info.value) == "Invalid credentials"
    assert password_service._DUMMY_PASSWORD_HASH in verification_hashes
    assert repr(user.password_hash) not in str(exc_info.value)


def test_real_postgresql_lookup_resolves_only_one_active_membership(
    monkeypatch,
    postgres_human_auth_engine,
):
    connection = postgres_human_auth_engine.connect()
    transaction = connection.begin()
    setup_session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        marker = uuid4().hex
        active_company = Company(name=f"Human Auth Active {marker}")
        inactive_company = Company(name=f"Human Auth Inactive {marker}")
        user = User(
            normalized_email=f"  Auth-{marker}@EXAMPLE.COM ",
            password_hash=hash_password(PASSWORD),
        )
        setup_session.add_all([active_company, inactive_company, user])
        setup_session.flush()
        active_membership = CompanyMembership(
            user_id=user.id,
            company_id=active_company.id,
            role=MembershipRole.SALESPERSON.value,
        )
        inactive_membership = CompanyMembership(
            user_id=user.id,
            company_id=inactive_company.id,
            role=MembershipRole.COMPANY_ADMIN.value,
            is_active=False,
        )
        setup_session.add_all([active_membership, inactive_membership])
        setup_session.flush()
        expected = AuthenticatedPrincipal(
            user_id=user.id,
            membership_id=active_membership.id,
            company_id=active_company.id,
            role=MembershipRole.SALESPERSON,
        )
        auth_session_factory = sessionmaker(
            bind=connection,
            join_transaction_mode="create_savepoint",
        )
        monkeypatch.setattr(
            human_service,
            "SessionLocal",
            auth_session_factory,
        )

        principal = human_service.authenticate_credentials(
            email=f" AUTH-{marker}@example.com\t",
            password=PASSWORD,
        )

        assert principal == expected
    finally:
        setup_session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()
