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
    InvalidSessionError,
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
    token_digest: bytes = TOKEN_DIGEST
    expires_at: datetime = NOW + timedelta(hours=8)
    last_activity_at: datetime = NOW - timedelta(minutes=30)
    revoked_at: datetime | None = None


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


@dataclass(frozen=True)
class PersistedSessionIdentity:
    token: str
    session_id: int
    user_id: int
    membership_id: int
    company_id: int
    expires_at: datetime
    last_activity_at: datetime
    token_digest: bytes


@pytest.fixture
def postgres_revocation_engine() -> Generator[Engine, None, None]:
    assert engine.dialect.name == "postgresql"
    schema_name = f"session_revocation_{uuid4().hex}"
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
    lookup_error=None,
    write_error=None,
):
    session_factory = RecordingSessionFactory()
    calls = []
    writes = []

    def derive_digest(*, token):
        calls.append(("digest", token))
        return TOKEN_DIGEST

    def find_session(session, *, token_digest):
        calls.append(("session", token_digest))
        if lookup_error is not None:
            raise lookup_error
        return user_session

    def revoke(session, *, user_session, revoked_at):
        calls.append(("revoke", revoked_at))
        if write_error is not None:
            raise write_error
        writes.append(revoked_at)
        user_session.revoked_at = revoked_at

    monkeypatch.setattr(human_service, "SessionLocal", session_factory)
    monkeypatch.setattr(human_service, "digest_session_token", derive_digest)
    monkeypatch.setattr(
        human_service,
        "find_user_session_by_token_digest_for_update",
        find_session,
    )
    monkeypatch.setattr(human_service, "revoke_user_session", revoke)
    monkeypatch.setattr(human_service, "_utc_now", lambda: NOW)
    return session_factory, calls, writes


def _create_persisted_session(
    database_engine: Engine,
    *,
    initial_activity: datetime,
) -> PersistedSessionIdentity:
    raw_token = generate_session_token()
    token_digest = digest_session_token(token=raw_token)
    expires_at = initial_activity + human_service.SESSION_ABSOLUTE_LIFETIME

    with Session(database_engine) as setup_session:
        company = Company(name=f"Revocation Company {uuid4().hex}")
        user = User(
            normalized_email=f"revoke-{uuid4().hex}@example.com",
            password_hash="$argon2id$opaque",
        )
        setup_session.add_all([company, user])
        setup_session.flush()
        membership = CompanyMembership(
            user_id=user.id,
            company_id=company.id,
            role=MembershipRole.COMPANY_ADMIN.value,
        )
        setup_session.add(membership)
        setup_session.flush()
        user_session = UserSession(
            user_id=user.id,
            membership_id=membership.id,
            company_id=company.id,
            token_digest=token_digest,
            expires_at=expires_at,
            last_activity_at=initial_activity,
        )
        setup_session.add(user_session)
        setup_session.commit()
        return PersistedSessionIdentity(
            token=raw_token,
            session_id=user_session.id,
            user_id=user.id,
            membership_id=membership.id,
            company_id=company.id,
            expires_at=expires_at,
            last_activity_at=initial_activity,
            token_digest=token_digest,
        )


def test_successful_revocation_uses_frozen_time_and_commits(monkeypatch):
    user_session = StubSession()
    session_factory, calls, writes = _install_backend(
        monkeypatch,
        user_session=user_session,
    )

    result = human_service.revoke_session(token=RAW_TOKEN)

    assert result is None
    assert calls == [
        ("digest", RAW_TOKEN),
        ("session", TOKEN_DIGEST),
        ("revoke", NOW),
    ]
    assert writes == [NOW]
    assert user_session.revoked_at == NOW
    assert NOW.tzinfo is timezone.utc
    assert session_factory.transactions[0].committed is True


def test_already_revoked_session_is_a_true_no_op(monkeypatch):
    original_revoked_at = NOW - timedelta(hours=1)
    user_session = StubSession(revoked_at=original_revoked_at)
    session_factory, calls, writes = _install_backend(
        monkeypatch,
        user_session=user_session,
    )

    def fail_if_clock_is_called():
        raise AssertionError("idempotent revocation must not request a new timestamp")

    monkeypatch.setattr(human_service, "_utc_now", fail_if_clock_is_called)

    result = human_service.revoke_session(token=RAW_TOKEN)

    assert result is None
    assert calls == [("digest", RAW_TOKEN), ("session", TOKEN_DIGEST)]
    assert writes == []
    assert user_session.revoked_at is original_revoked_at
    assert session_factory.transactions[0].committed is True


@pytest.mark.parametrize("token", [None, "", " \t\n ", "tökén"])
def test_malformed_token_is_generic_and_does_not_open_transaction(
    monkeypatch,
    token,
):
    session_factory = RecordingSessionFactory()
    monkeypatch.setattr(human_service, "SessionLocal", session_factory)

    with pytest.raises(InvalidSessionError, match="^Invalid session$"):
        human_service.revoke_session(token=token)

    assert session_factory.transactions == []


def test_unknown_token_is_generic_and_rolls_back(monkeypatch):
    session_factory, _, writes = _install_backend(
        monkeypatch,
        user_session=None,
    )

    with pytest.raises(InvalidSessionError, match="^Invalid session$"):
        human_service.revoke_session(token=RAW_TOKEN)

    assert writes == []
    assert session_factory.transactions[0].rolled_back is True


def test_digest_backend_failure_is_sanitized_before_transaction(monkeypatch):
    secret_detail = f"digest backend detail containing {RAW_TOKEN}"
    session_factory = RecordingSessionFactory()
    monkeypatch.setattr(human_service, "SessionLocal", session_factory)

    def fail_digest(*, token):
        raise RuntimeError(secret_detail)

    monkeypatch.setattr(human_service, "digest_session_token", fail_digest)

    with pytest.raises(AuthInfrastructureError) as exc_info:
        human_service.revoke_session(token=RAW_TOKEN)

    assert str(exc_info.value) == "Human authentication unavailable"
    assert secret_detail not in str(exc_info.value)
    assert RAW_TOKEN not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert session_factory.transactions == []


@pytest.mark.parametrize(
    "user_session",
    [
        StubSession(expires_at=NOW - timedelta(seconds=1)),
        StubSession(
            last_activity_at=(
                NOW - human_service.SESSION_IDLE_TIMEOUT - timedelta(seconds=1)
            )
        ),
    ],
    ids=["absolute-expired", "idle-expired"],
)
def test_expired_sessions_can_still_be_revoked(monkeypatch, user_session):
    session_factory, _, writes = _install_backend(
        monkeypatch,
        user_session=user_session,
    )

    human_service.revoke_session(token=RAW_TOKEN)

    assert writes == [NOW]
    assert user_session.revoked_at == NOW
    assert session_factory.transactions[0].committed is True


def test_revocation_does_not_query_user_or_membership(monkeypatch):
    user_session = StubSession()
    _, _, writes = _install_backend(monkeypatch, user_session=user_session)

    def unexpected_identity_query(*args, **kwargs):
        raise AssertionError("revocation must not query user or membership state")

    monkeypatch.setattr(human_service, "find_user_by_id", unexpected_identity_query)
    monkeypatch.setattr(
        human_service,
        "find_membership_by_id",
        unexpected_identity_query,
    )

    human_service.revoke_session(token=RAW_TOKEN)

    assert writes == [NOW]


@pytest.mark.parametrize("failure_stage", ["lookup", "write"])
def test_backend_failure_is_sanitized_and_rolls_back(monkeypatch, failure_stage):
    secret_detail = f"postgresql://private/{RAW_TOKEN}"
    error = RuntimeError(secret_detail)
    session_factory, _, _ = _install_backend(
        monkeypatch,
        user_session=StubSession(),
        lookup_error=error if failure_stage == "lookup" else None,
        write_error=error if failure_stage == "write" else None,
    )

    with pytest.raises(AuthInfrastructureError) as exc_info:
        human_service.revoke_session(token=RAW_TOKEN)

    assert str(exc_info.value) == "Human authentication unavailable"
    assert RAW_TOKEN not in str(exc_info.value)
    assert "postgresql" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert session_factory.transactions[0].rolled_back is True


def test_postgresql_revocation_changes_only_revocation_state(
    monkeypatch,
    postgres_revocation_engine,
):
    initial_activity = datetime.now(timezone.utc)
    revoked_at = initial_activity + timedelta(minutes=15)
    identity = _create_persisted_session(
        postgres_revocation_engine,
        initial_activity=initial_activity,
    )
    monkeypatch.setattr(
        human_service,
        "SessionLocal",
        sessionmaker(bind=postgres_revocation_engine),
    )
    monkeypatch.setattr(human_service, "_utc_now", lambda: revoked_at)

    result = human_service.revoke_session(token=identity.token)

    assert result is None
    with Session(postgres_revocation_engine) as verification_session:
        persisted = verification_session.get(UserSession, identity.session_id)
        assert persisted is not None
        assert persisted.revoked_at == revoked_at
        assert persisted.revoked_at.tzinfo is not None
        assert persisted.revoked_at.utcoffset() == timedelta(0)
        assert persisted.expires_at == identity.expires_at
        assert persisted.last_activity_at == identity.last_activity_at
        assert persisted.user_id == identity.user_id
        assert persisted.membership_id == identity.membership_id
        assert persisted.company_id == identity.company_id
        assert persisted.token_digest == identity.token_digest


def test_resolve_first_serializes_before_revocation(
    monkeypatch,
    postgres_revocation_engine,
):
    initial_activity = datetime.now(timezone.utc)
    resolve_now = initial_activity + timedelta(minutes=5)
    revoke_now = initial_activity + timedelta(minutes=10)
    identity = _create_persisted_session(
        postgres_revocation_engine,
        initial_activity=initial_activity,
    )
    monkeypatch.setattr(
        human_service,
        "SessionLocal",
        sessionmaker(bind=postgres_revocation_engine),
    )
    original_locked_lookup = (
        human_service.find_user_session_by_token_digest_for_update
    )
    resolve_holds_lock = Event()
    release_resolve = Event()
    revoke_attempting_lock = Event()
    revoke_acquired_lock = Event()
    outcomes = {}
    errors = {}

    def controlled_locked_lookup(session, *, token_digest):
        thread_name = current_thread().name
        if thread_name == "revoke-second":
            revoke_attempting_lock.set()
        persisted = original_locked_lookup(session, token_digest=token_digest)
        if thread_name == "resolve-first":
            resolve_holds_lock.set()
            if not release_resolve.wait(timeout=5):
                raise RuntimeError("test timed out releasing resolve transaction")
        elif thread_name == "revoke-second":
            revoke_acquired_lock.set()
        return persisted

    def frozen_clock():
        if current_thread().name == "resolve-first":
            return resolve_now
        return revoke_now

    def run_resolve():
        try:
            outcomes["resolve"] = human_service.resolve_session(token=identity.token)
        except Exception as exc:
            errors["resolve"] = exc

    def run_revoke():
        try:
            outcomes["revoke"] = human_service.revoke_session(token=identity.token)
        except Exception as exc:
            errors["revoke"] = exc

    monkeypatch.setattr(
        human_service,
        "find_user_session_by_token_digest_for_update",
        controlled_locked_lookup,
    )
    monkeypatch.setattr(human_service, "_utc_now", frozen_clock)

    resolve_thread = Thread(target=run_resolve, name="resolve-first")
    revoke_thread = Thread(target=run_revoke, name="revoke-second")
    resolve_thread.start()
    assert resolve_holds_lock.wait(timeout=5)
    revoke_thread.start()
    assert revoke_attempting_lock.wait(timeout=5)
    revoke_was_blocked = not revoke_acquired_lock.wait(timeout=0.25)
    release_resolve.set()
    resolve_thread.join(timeout=5)
    revoke_thread.join(timeout=5)

    assert revoke_was_blocked is True
    assert resolve_thread.is_alive() is False
    assert revoke_thread.is_alive() is False
    assert errors == {}
    assert outcomes == {
        "resolve": AuthenticatedUserContext(
            user_id=identity.user_id,
            membership_id=identity.membership_id,
            company_id=identity.company_id,
            role=MembershipRole.COMPANY_ADMIN,
            session_id=identity.session_id,
        ),
        "revoke": None,
    }
    assert revoke_acquired_lock.is_set()

    with Session(postgres_revocation_engine) as verification_session:
        persisted = verification_session.get(UserSession, identity.session_id)
        assert persisted is not None
        assert persisted.revoked_at == revoke_now
        assert persisted.last_activity_at == resolve_now
        assert persisted.expires_at == identity.expires_at

    with pytest.raises(SessionRevokedError, match="^Invalid session$"):
        human_service.resolve_session(token=identity.token)


def test_revoke_first_serializes_before_resolution(
    monkeypatch,
    postgres_revocation_engine,
):
    initial_activity = datetime.now(timezone.utc)
    revoke_now = initial_activity + timedelta(minutes=5)
    identity = _create_persisted_session(
        postgres_revocation_engine,
        initial_activity=initial_activity,
    )
    monkeypatch.setattr(
        human_service,
        "SessionLocal",
        sessionmaker(bind=postgres_revocation_engine),
    )
    original_locked_lookup = (
        human_service.find_user_session_by_token_digest_for_update
    )
    revoke_holds_lock = Event()
    release_revoke = Event()
    resolve_attempting_lock = Event()
    resolve_acquired_lock = Event()
    outcomes = {}
    errors = {}

    def controlled_locked_lookup(session, *, token_digest):
        thread_name = current_thread().name
        if thread_name == "resolve-second":
            resolve_attempting_lock.set()
        persisted = original_locked_lookup(session, token_digest=token_digest)
        if thread_name == "revoke-first":
            revoke_holds_lock.set()
            if not release_revoke.wait(timeout=5):
                raise RuntimeError("test timed out releasing revoke transaction")
        elif thread_name == "resolve-second":
            resolve_acquired_lock.set()
        return persisted

    def run_revoke():
        try:
            outcomes["revoke"] = human_service.revoke_session(token=identity.token)
        except Exception as exc:
            errors["revoke"] = exc

    def run_resolve():
        try:
            human_service.resolve_session(token=identity.token)
        except Exception as exc:
            errors["resolve"] = exc

    monkeypatch.setattr(
        human_service,
        "find_user_session_by_token_digest_for_update",
        controlled_locked_lookup,
    )
    monkeypatch.setattr(human_service, "_utc_now", lambda: revoke_now)

    revoke_thread = Thread(target=run_revoke, name="revoke-first")
    resolve_thread = Thread(target=run_resolve, name="resolve-second")
    revoke_thread.start()
    assert revoke_holds_lock.wait(timeout=5)
    resolve_thread.start()
    assert resolve_attempting_lock.wait(timeout=5)
    resolve_was_blocked = not resolve_acquired_lock.wait(timeout=0.25)
    release_revoke.set()
    revoke_thread.join(timeout=5)
    resolve_thread.join(timeout=5)

    assert resolve_was_blocked is True
    assert revoke_thread.is_alive() is False
    assert resolve_thread.is_alive() is False
    assert outcomes == {"revoke": None}
    assert "revoke" not in errors
    assert isinstance(errors.get("resolve"), SessionRevokedError)
    assert str(errors["resolve"]) == "Invalid session"
    assert resolve_acquired_lock.is_set()

    with Session(postgres_revocation_engine) as verification_session:
        persisted = verification_session.get(UserSession, identity.session_id)
        assert persisted is not None
        assert persisted.revoked_at == revoke_now
        assert persisted.last_activity_at == identity.last_activity_at
        assert persisted.expires_at == identity.expires_at
