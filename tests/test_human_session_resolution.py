from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Event, Thread, current_thread
from uuid import uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.auth.human import (
    AuthInfrastructureError,
    AuthenticatedUserContext,
    InactiveMembershipError,
    InactiveUserError,
    InvalidSessionError,
    SessionExpiredError,
    SessionRevokedError,
)
from app.auth.human import service as human_service
from app.auth.human.sessions import digest_session_token, generate_session_token
from app.database import Base, engine
from app.domain.enums import MembershipRole
from app.models import Company, CompanyMembership, User, UserSession

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
RAW_TOKEN = "opaque-session-token"
TOKEN_DIGEST = bytes(range(32))


@dataclass
class StubSession:
    id: int = 404
    user_id: int = 101
    membership_id: int = 202
    company_id: int = 303
    expires_at: datetime = NOW + timedelta(hours=8)
    last_activity_at: datetime = NOW - timedelta(minutes=30)
    revoked_at: datetime | None = None


@dataclass
class StubUser:
    id: int = 101
    is_active: bool = True


@dataclass
class StubMembership:
    id: int = 202
    user_id: int = 101
    company_id: int = 303
    role: str = MembershipRole.COMPANY_ADMIN.value
    is_active: bool = True


class RecordingTransaction:
    def __init__(self):
        self.session = object()
        self.committed = False
        self.rolled_back = False

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc_value, traceback):
        self.committed = exc_type is None
        self.rolled_back = exc_type is not None
        return False


class RecordingSessionFactory:
    def __init__(self):
        self.transactions = []

    def begin(self):
        transaction = RecordingTransaction()
        self.transactions.append(transaction)
        return transaction


@pytest.fixture
def postgres_resolution_engine() -> Generator[Engine, None, None]:
    assert engine.dialect.name == "postgresql"
    schema_name = f"session_resolution_{uuid4().hex}"
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
            UserSession.__table__,
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
    user_session=None,
    user=None,
    membership=None,
    lookup_error=None,
):
    session_factory = RecordingSessionFactory()
    calls = []
    refreshes = []

    def derive_digest(*, token):
        calls.append(("digest", token))
        return TOKEN_DIGEST

    def find_session(session, *, token_digest):
        calls.append(("session", token_digest))
        if lookup_error is not None:
            raise lookup_error
        return user_session

    def find_user(session, *, user_id):
        calls.append(("user", user_id))
        return user

    def find_membership(session, *, membership_id):
        calls.append(("membership", membership_id))
        return membership

    def refresh(session, *, user_session, last_activity_at):
        calls.append(("refresh", last_activity_at))
        refreshes.append(last_activity_at)
        user_session.last_activity_at = last_activity_at

    monkeypatch.setattr(human_service, "SessionLocal", session_factory)
    monkeypatch.setattr(human_service, "digest_session_token", derive_digest)
    monkeypatch.setattr(
        human_service,
        "find_user_session_by_token_digest_for_update",
        find_session,
    )
    monkeypatch.setattr(human_service, "find_user_by_id", find_user)
    monkeypatch.setattr(human_service, "find_membership_by_id", find_membership)
    monkeypatch.setattr(human_service, "refresh_user_session_activity", refresh)
    monkeypatch.setattr(human_service, "_utc_now", lambda: NOW)
    return session_factory, calls, refreshes


def test_successful_resolution_refreshes_activity_and_returns_persisted_context(
    monkeypatch,
):
    user_session = StubSession()
    original_expiry = user_session.expires_at
    user = StubUser()
    membership = StubMembership()
    session_factory, calls, refreshes = _install_backend(
        monkeypatch,
        user_session=user_session,
        user=user,
        membership=membership,
    )

    context = human_service.resolve_session(token=RAW_TOKEN)

    assert context == AuthenticatedUserContext(
        user_id=user.id,
        membership_id=membership.id,
        company_id=membership.company_id,
        role=MembershipRole.COMPANY_ADMIN,
        session_id=user_session.id,
    )
    assert calls == [
        ("digest", RAW_TOKEN),
        ("session", TOKEN_DIGEST),
        ("user", user.id),
        ("membership", membership.id),
        ("refresh", NOW),
    ]
    assert refreshes == [NOW]
    assert user_session.last_activity_at == NOW
    assert user_session.expires_at == original_expiry
    assert session_factory.transactions[0].committed is True


def test_absolute_expiry_is_rejected_without_activity_refresh(monkeypatch):
    user_session = StubSession(expires_at=NOW - timedelta(microseconds=1))
    session_factory, _, refreshes = _install_backend(
        monkeypatch,
        user_session=user_session,
    )

    with pytest.raises(SessionExpiredError, match="^Invalid session$"):
        human_service.resolve_session(token=RAW_TOKEN)

    assert refreshes == []
    assert session_factory.transactions[0].rolled_back is True


def test_absolute_expiry_boundary_is_inclusive(monkeypatch):
    user_session = StubSession(expires_at=NOW)
    _, _, refreshes = _install_backend(monkeypatch, user_session=user_session)

    with pytest.raises(SessionExpiredError, match="^Invalid session$"):
        human_service.resolve_session(token=RAW_TOKEN)

    assert refreshes == []


def test_idle_expiry_is_rejected_without_activity_refresh(monkeypatch):
    user_session = StubSession(
        last_activity_at=NOW - human_service.SESSION_IDLE_TIMEOUT - timedelta(seconds=1)
    )
    _, _, refreshes = _install_backend(monkeypatch, user_session=user_session)

    with pytest.raises(SessionExpiredError, match="^Invalid session$"):
        human_service.resolve_session(token=RAW_TOKEN)

    assert refreshes == []


def test_idle_expiry_boundary_is_inclusive(monkeypatch):
    user_session = StubSession(
        last_activity_at=NOW - human_service.SESSION_IDLE_TIMEOUT
    )
    _, _, refreshes = _install_backend(monkeypatch, user_session=user_session)

    with pytest.raises(SessionExpiredError, match="^Invalid session$"):
        human_service.resolve_session(token=RAW_TOKEN)

    assert refreshes == []


def test_revoked_session_is_rejected_before_expiry(monkeypatch):
    user_session = StubSession(revoked_at=NOW, expires_at=NOW)
    _, calls, refreshes = _install_backend(monkeypatch, user_session=user_session)

    with pytest.raises(SessionRevokedError, match="^Invalid session$"):
        human_service.resolve_session(token=RAW_TOKEN)

    assert [name for name, _ in calls] == ["digest", "session"]
    assert refreshes == []


def test_inactive_user_is_rejected_without_activity_refresh(monkeypatch):
    user_session = StubSession()
    _, _, refreshes = _install_backend(
        monkeypatch,
        user_session=user_session,
        user=StubUser(is_active=False),
    )

    with pytest.raises(InactiveUserError, match="^Invalid session$"):
        human_service.resolve_session(token=RAW_TOKEN)

    assert refreshes == []


def test_inactive_membership_is_rejected_without_activity_refresh(monkeypatch):
    user_session = StubSession()
    _, _, refreshes = _install_backend(
        monkeypatch,
        user_session=user_session,
        user=StubUser(),
        membership=StubMembership(is_active=False),
    )

    with pytest.raises(InactiveMembershipError, match="^Invalid session$"):
        human_service.resolve_session(token=RAW_TOKEN)

    assert refreshes == []


@pytest.mark.parametrize("token", [None, "", " \t\n ", "tökén"])
def test_malformed_token_is_generic_and_does_not_open_transaction(
    monkeypatch,
    token,
):
    session_factory = RecordingSessionFactory()
    monkeypatch.setattr(human_service, "SessionLocal", session_factory)

    with pytest.raises(InvalidSessionError, match="^Invalid session$"):
        human_service.resolve_session(token=token)

    assert session_factory.transactions == []


def test_unknown_token_is_generic_and_does_not_refresh(monkeypatch):
    session_factory, _, refreshes = _install_backend(
        monkeypatch,
        user_session=None,
    )

    with pytest.raises(InvalidSessionError, match="^Invalid session$"):
        human_service.resolve_session(token=RAW_TOKEN)

    assert refreshes == []
    assert session_factory.transactions[0].rolled_back is True


def test_invalid_persisted_role_is_sanitized_infrastructure_error(monkeypatch):
    _, _, refreshes = _install_backend(
        monkeypatch,
        user_session=StubSession(),
        user=StubUser(),
        membership=StubMembership(role="owner"),
    )

    with pytest.raises(AuthInfrastructureError) as exc_info:
        human_service.resolve_session(token=RAW_TOKEN)

    assert str(exc_info.value) == "Human authentication unavailable"
    assert exc_info.value.__cause__ is None
    assert refreshes == []


@pytest.mark.parametrize(
    ("user_session", "user", "membership"),
    [
        (StubSession(user_id=999), StubUser(), StubMembership()),
        (StubSession(membership_id=999), StubUser(), StubMembership()),
        (StubSession(), StubUser(), StubMembership(user_id=999)),
        (StubSession(company_id=999), StubUser(), StubMembership()),
    ],
)
def test_inconsistent_persisted_identity_is_sanitized_and_not_refreshed(
    monkeypatch,
    user_session,
    user,
    membership,
):
    _, _, refreshes = _install_backend(
        monkeypatch,
        user_session=user_session,
        user=user,
        membership=membership,
    )

    with pytest.raises(AuthInfrastructureError) as exc_info:
        human_service.resolve_session(token=RAW_TOKEN)

    assert str(exc_info.value) == "Human authentication unavailable"
    assert exc_info.value.__cause__ is None
    assert refreshes == []


@pytest.mark.parametrize(
    ("user", "membership"),
    [
        (None, StubMembership()),
        (StubUser(), None),
    ],
)
def test_missing_trusted_identity_is_sanitized_and_not_refreshed(
    monkeypatch,
    user,
    membership,
):
    _, _, refreshes = _install_backend(
        monkeypatch,
        user_session=StubSession(),
        user=user,
        membership=membership,
    )

    with pytest.raises(AuthInfrastructureError) as exc_info:
        human_service.resolve_session(token=RAW_TOKEN)

    assert str(exc_info.value) == "Human authentication unavailable"
    assert exc_info.value.__cause__ is None
    assert refreshes == []


def test_backend_failure_is_sanitized_without_token_or_database_details(monkeypatch):
    secret_detail = f"postgresql://private/{RAW_TOKEN}"
    session_factory, _, refreshes = _install_backend(
        monkeypatch,
        lookup_error=RuntimeError(secret_detail),
    )

    with pytest.raises(AuthInfrastructureError) as exc_info:
        human_service.resolve_session(token=RAW_TOKEN)

    assert str(exc_info.value) == "Human authentication unavailable"
    assert secret_detail not in str(exc_info.value)
    assert RAW_TOKEN not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert refreshes == []
    assert session_factory.transactions[0].rolled_back is True


def test_concurrent_resolutions_serialize_activity_refreshes_with_row_lock(
    monkeypatch,
    postgres_resolution_engine,
):
    initial_activity = datetime.now(timezone.utc)
    first_now = initial_activity + timedelta(minutes=5)
    second_now = initial_activity + timedelta(minutes=10)
    raw_token = generate_session_token()
    token_digest = digest_session_token(token=raw_token)

    with Session(postgres_resolution_engine) as setup_session:
        company = Company(name="Concurrent Session Company")
        user = User(
            normalized_email=f"concurrent-{uuid4().hex}@example.com",
            password_hash="$argon2id$opaque",
        )
        setup_session.add_all([company, user])
        setup_session.flush()
        membership = CompanyMembership(
            user_id=user.id,
            company_id=company.id,
            role=MembershipRole.SALESPERSON.value,
        )
        setup_session.add(membership)
        setup_session.flush()
        user_session = UserSession(
            user_id=user.id,
            membership_id=membership.id,
            company_id=company.id,
            token_digest=token_digest,
            expires_at=initial_activity + timedelta(hours=8),
            last_activity_at=initial_activity,
        )
        setup_session.add(user_session)
        setup_session.commit()
        expected_session_id = user_session.id
        expected_user_id = user.id
        expected_membership_id = membership.id
        expected_company_id = company.id

    monkeypatch.setattr(
        human_service,
        "SessionLocal",
        sessionmaker(bind=postgres_resolution_engine),
    )
    original_find_user = human_service.find_user_by_id
    first_holds_session_lock = Event()
    release_first_resolution = Event()
    second_reached_user_lookup = Event()
    second_started = Event()
    contexts = {}
    errors = {}

    def blocking_find_user(session, *, user_id):
        if current_thread().name == "first-resolution":
            first_holds_session_lock.set()
            if not release_first_resolution.wait(timeout=5):
                raise RuntimeError("test timed out releasing first resolution")
        else:
            second_reached_user_lookup.set()
        return original_find_user(session, user_id=user_id)

    def frozen_clock():
        if current_thread().name == "first-resolution":
            return first_now
        return second_now

    def resolve(name):
        if name == "second":
            second_started.set()
        try:
            contexts[name] = human_service.resolve_session(token=raw_token)
        except Exception as exc:
            errors[name] = exc

    monkeypatch.setattr(human_service, "find_user_by_id", blocking_find_user)
    monkeypatch.setattr(human_service, "_utc_now", frozen_clock)

    first = Thread(target=resolve, args=("first",), name="first-resolution")
    second = Thread(target=resolve, args=("second",), name="second-resolution")
    first.start()
    assert first_holds_session_lock.wait(timeout=5)
    second.start()
    assert second_started.wait(timeout=5)
    second_was_blocked = not second_reached_user_lookup.wait(timeout=0.25)
    release_first_resolution.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert second_was_blocked is True
    assert first.is_alive() is False
    assert second.is_alive() is False
    assert errors == {}
    expected_context = AuthenticatedUserContext(
        user_id=expected_user_id,
        membership_id=expected_membership_id,
        company_id=expected_company_id,
        role=MembershipRole.SALESPERSON,
        session_id=expected_session_id,
    )
    assert contexts == {"first": expected_context, "second": expected_context}
    assert second_reached_user_lookup.is_set()

    with Session(postgres_resolution_engine) as verification_session:
        persisted = verification_session.get(UserSession, expected_session_id)
        assert persisted is not None
        assert persisted.last_activity_at == second_now
        assert persisted.expires_at == initial_activity + timedelta(hours=8)
