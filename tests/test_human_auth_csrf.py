import hashlib
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
    InvalidCsrfTokenError,
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
SESSION_TOKEN = "opaque-session-token"
SESSION_TOKEN_DIGEST = bytes(range(32))
CSRF_TOKEN = "opaque-csrf-token"
CSRF_TOKEN_DIGEST = hashlib.sha256(CSRF_TOKEN.encode("ascii")).digest()


@dataclass
class StubSession:
    id: int = 404
    user_id: int = 101
    membership_id: int = 202
    company_id: int = 303
    token_digest: bytes = SESSION_TOKEN_DIGEST
    expires_at: datetime = NOW + timedelta(hours=8)
    last_activity_at: datetime = NOW - timedelta(minutes=30)
    revoked_at: datetime | None = None
    csrf_token_digest: bytes | None = None


class RecordingTransaction:
    def __init__(self, *, commit_error=None):
        self.session = object()
        self.commit_error = commit_error
        self.committed = False
        self.rolled_back = False

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.rolled_back = True
            return False
        if self.commit_error is not None:
            raise self.commit_error
        self.committed = True
        return False


class RecordingSessionFactory:
    def __init__(self, *, commit_error=None):
        self.commit_error = commit_error
        self.transactions = []

    def begin(self):
        transaction = RecordingTransaction(commit_error=self.commit_error)
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
def postgres_csrf_engine() -> Generator[Engine, None, None]:
    assert engine.dialect.name == "postgresql"
    schema_name = f"human_csrf_{uuid4().hex}"
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
    commit_error=None,
):
    session_factory = RecordingSessionFactory(commit_error=commit_error)
    calls = []
    writes = []

    def derive_session_digest(*, token):
        calls.append(("session-digest", token))
        return SESSION_TOKEN_DIGEST

    def find_session(session, *, token_digest):
        calls.append(("session", token_digest))
        if lookup_error is not None:
            raise lookup_error
        return user_session

    def replace_digest(session, *, user_session, csrf_token_digest):
        calls.append(("replace", csrf_token_digest))
        if write_error is not None:
            raise write_error
        writes.append(csrf_token_digest)
        user_session.csrf_token_digest = csrf_token_digest

    monkeypatch.setattr(human_service, "SessionLocal", session_factory)
    monkeypatch.setattr(
        human_service,
        "digest_session_token",
        derive_session_digest,
    )
    monkeypatch.setattr(
        human_service,
        "find_user_session_by_token_digest_for_update",
        find_session,
    )
    monkeypatch.setattr(
        human_service,
        "replace_user_session_csrf_digest",
        replace_digest,
    )
    monkeypatch.setattr(human_service, "_utc_now", lambda: NOW)
    monkeypatch.setattr(human_service, "_generate_csrf_token", lambda: CSRF_TOKEN)
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
        company = Company(name=f"CSRF Company {uuid4().hex}")
        user = User(
            normalized_email=f"csrf-{uuid4().hex}@example.com",
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


def _lifecycle_fields(user_session):
    return (
        user_session.expires_at,
        user_session.last_activity_at,
        user_session.user_id,
        user_session.membership_id,
        user_session.company_id,
        user_session.token_digest,
    )


def test_issue_returns_raw_token_after_persisting_only_its_digest(monkeypatch):
    user_session = StubSession()
    original_fields = _lifecycle_fields(user_session)
    session_factory, calls, writes = _install_backend(
        monkeypatch,
        user_session=user_session,
    )

    issued_token = human_service.issue_csrf_token(session_token=SESSION_TOKEN)

    assert issued_token == CSRF_TOKEN
    assert calls == [
        ("session-digest", SESSION_TOKEN),
        ("session", SESSION_TOKEN_DIGEST),
        ("replace", CSRF_TOKEN_DIGEST),
    ]
    assert writes == [CSRF_TOKEN_DIGEST]
    assert len(user_session.csrf_token_digest) == 32
    assert user_session.csrf_token_digest == CSRF_TOKEN_DIGEST
    assert user_session.csrf_token_digest != CSRF_TOKEN.encode("ascii")
    assert CSRF_TOKEN not in repr(user_session.csrf_token_digest)
    assert _lifecycle_fields(user_session) == original_fields
    assert session_factory.transactions[0].committed is True


def test_repeated_issuance_rotates_and_invalidates_previous_token(monkeypatch):
    first_token = "first-csrf-token"
    second_token = "second-csrf-token"
    generated_tokens = iter([first_token, second_token])
    user_session = StubSession()
    _install_backend(monkeypatch, user_session=user_session)
    monkeypatch.setattr(
        human_service,
        "_generate_csrf_token",
        lambda: next(generated_tokens),
    )

    assert human_service.issue_csrf_token(session_token=SESSION_TOKEN) == first_token
    first_digest = user_session.csrf_token_digest
    assert human_service.issue_csrf_token(session_token=SESSION_TOKEN) == second_token

    assert user_session.csrf_token_digest != first_digest
    assert user_session.csrf_token_digest == hashlib.sha256(
        second_token.encode("ascii")
    ).digest()
    with pytest.raises(InvalidCsrfTokenError, match="^Invalid CSRF token$"):
        human_service.verify_csrf_token(
            session_token=SESSION_TOKEN,
            csrf_token=first_token,
        )
    assert (
        human_service.verify_csrf_token(
            session_token=SESSION_TOKEN,
            csrf_token=second_token,
        )
        is None
    )


def test_successful_verification_returns_none_without_rotation(monkeypatch):
    user_session = StubSession(csrf_token_digest=CSRF_TOKEN_DIGEST)
    original_fields = _lifecycle_fields(user_session)
    session_factory, _, writes = _install_backend(
        monkeypatch,
        user_session=user_session,
    )

    result = human_service.verify_csrf_token(
        session_token=SESSION_TOKEN,
        csrf_token=CSRF_TOKEN,
    )

    assert result is None
    assert writes == []
    assert user_session.csrf_token_digest == CSRF_TOKEN_DIGEST
    assert _lifecycle_fields(user_session) == original_fields
    assert session_factory.transactions[0].committed is True


@pytest.mark.parametrize(
    "stored_digest",
    [
        CSRF_TOKEN_DIGEST,
        None,
        b"malformed",
    ],
    ids=["wrong-token", "missing-stored-digest", "malformed-stored-digest"],
)
def test_invalid_or_missing_csrf_digest_is_generic(monkeypatch, stored_digest):
    user_session = StubSession(csrf_token_digest=stored_digest)
    _install_backend(monkeypatch, user_session=user_session)

    supplied_token = "wrong-csrf-token" if stored_digest else CSRF_TOKEN
    with pytest.raises(InvalidCsrfTokenError, match="^Invalid CSRF token$"):
        human_service.verify_csrf_token(
            session_token=SESSION_TOKEN,
            csrf_token=supplied_token,
        )


def test_missing_stored_digest_still_uses_constant_time_comparison(monkeypatch):
    user_session = StubSession(csrf_token_digest=None)
    _, _, writes = _install_backend(monkeypatch, user_session=user_session)
    comparisons = []

    def compare_digest(stored_digest, supplied_digest):
        comparisons.append((stored_digest, supplied_digest))
        return True

    monkeypatch.setattr(human_service.hmac, "compare_digest", compare_digest)

    with pytest.raises(InvalidCsrfTokenError, match="^Invalid CSRF token$"):
        human_service.verify_csrf_token(
            session_token=SESSION_TOKEN,
            csrf_token=CSRF_TOKEN,
        )

    assert comparisons == [(bytes(32), CSRF_TOKEN_DIGEST)]
    assert user_session.csrf_token_digest is None
    assert writes == []


@pytest.mark.parametrize("csrf_token", [None, "", " \t\n ", "tökén"])
def test_malformed_csrf_token_is_generic_without_database_access(
    monkeypatch,
    csrf_token,
):
    session_factory, _, _ = _install_backend(
        monkeypatch,
        user_session=StubSession(csrf_token_digest=CSRF_TOKEN_DIGEST),
    )

    with pytest.raises(InvalidCsrfTokenError, match="^Invalid CSRF token$"):
        human_service.verify_csrf_token(
            session_token=SESSION_TOKEN,
            csrf_token=csrf_token,
        )

    assert session_factory.transactions == []


@pytest.mark.parametrize("session_token", [None, "", " \t\n ", "tökén"])
@pytest.mark.parametrize("operation", ["issue", "verify"])
def test_malformed_session_token_preserves_invalid_session_behavior(
    monkeypatch,
    session_token,
    operation,
):
    session_factory = RecordingSessionFactory()
    monkeypatch.setattr(human_service, "SessionLocal", session_factory)

    with pytest.raises(InvalidSessionError, match="^Invalid session$"):
        if operation == "issue":
            human_service.issue_csrf_token(session_token=session_token)
        else:
            human_service.verify_csrf_token(
                session_token=session_token,
                csrf_token=CSRF_TOKEN,
            )

    assert session_factory.transactions == []


@pytest.mark.parametrize("operation", ["issue", "verify"])
def test_unknown_session_is_generic_and_rolls_back(monkeypatch, operation):
    session_factory, _, writes = _install_backend(
        monkeypatch,
        user_session=None,
    )

    with pytest.raises(InvalidSessionError, match="^Invalid session$"):
        if operation == "issue":
            human_service.issue_csrf_token(session_token=SESSION_TOKEN)
        else:
            human_service.verify_csrf_token(
                session_token=SESSION_TOKEN,
                csrf_token=CSRF_TOKEN,
            )

    assert writes == []
    assert session_factory.transactions[0].rolled_back is True


@pytest.mark.parametrize("operation", ["issue", "verify"])
@pytest.mark.parametrize(
    ("user_session", "expected_error"),
    [
        (StubSession(revoked_at=NOW), SessionRevokedError),
        (
            StubSession(expires_at=NOW),
            SessionExpiredError,
        ),
        (
            StubSession(last_activity_at=NOW - human_service.SESSION_IDLE_TIMEOUT),
            SessionExpiredError,
        ),
    ],
    ids=["revoked", "absolute-expired", "idle-expired"],
)
def test_ineligible_session_cannot_issue_or_verify_csrf(
    monkeypatch,
    user_session,
    expected_error,
    operation,
):
    user_session.csrf_token_digest = CSRF_TOKEN_DIGEST
    session_factory, _, writes = _install_backend(
        monkeypatch,
        user_session=user_session,
    )

    with pytest.raises(expected_error, match="^Invalid session$"):
        if operation == "issue":
            human_service.issue_csrf_token(session_token=SESSION_TOKEN)
        else:
            human_service.verify_csrf_token(
                session_token=SESSION_TOKEN,
                csrf_token=CSRF_TOKEN,
            )

    assert writes == []
    assert session_factory.transactions[0].rolled_back is True


def test_issue_and_verify_do_not_query_user_or_membership(monkeypatch):
    user_session = StubSession(csrf_token_digest=CSRF_TOKEN_DIGEST)
    _install_backend(monkeypatch, user_session=user_session)

    def unexpected_identity_query(*args, **kwargs):
        raise AssertionError("CSRF operations must not revalidate identity")

    monkeypatch.setattr(human_service, "find_user_by_id", unexpected_identity_query)
    monkeypatch.setattr(
        human_service,
        "find_membership_by_id",
        unexpected_identity_query,
    )

    human_service.issue_csrf_token(session_token=SESSION_TOKEN)
    human_service.verify_csrf_token(
        session_token=SESSION_TOKEN,
        csrf_token=CSRF_TOKEN,
    )


def test_verification_uses_constant_time_digest_comparison(monkeypatch):
    user_session = StubSession(csrf_token_digest=CSRF_TOKEN_DIGEST)
    _install_backend(monkeypatch, user_session=user_session)
    comparisons = []

    def compare_digest(stored_digest, supplied_digest):
        comparisons.append((stored_digest, supplied_digest))
        return True

    monkeypatch.setattr(human_service.hmac, "compare_digest", compare_digest)

    human_service.verify_csrf_token(
        session_token=SESSION_TOKEN,
        csrf_token=CSRF_TOKEN,
    )

    assert comparisons == [(CSRF_TOKEN_DIGEST, CSRF_TOKEN_DIGEST)]


@pytest.mark.parametrize(
    ("operation", "failure_stage"),
    [
        ("issue", "lookup"),
        ("issue", "write"),
        ("verify", "lookup"),
    ],
)
def test_database_failures_are_sanitized_and_rolled_back(
    monkeypatch,
    operation,
    failure_stage,
):
    secret_detail = f"postgresql://private/{SESSION_TOKEN}/{CSRF_TOKEN}"
    error = RuntimeError(secret_detail)
    session_factory, _, _ = _install_backend(
        monkeypatch,
        user_session=StubSession(csrf_token_digest=CSRF_TOKEN_DIGEST),
        lookup_error=error if failure_stage == "lookup" else None,
        write_error=error if failure_stage == "write" else None,
    )

    with pytest.raises(AuthInfrastructureError) as exc_info:
        if operation == "issue":
            human_service.issue_csrf_token(session_token=SESSION_TOKEN)
        else:
            human_service.verify_csrf_token(
                session_token=SESSION_TOKEN,
                csrf_token=CSRF_TOKEN,
            )

    assert str(exc_info.value) == "Human authentication unavailable"
    assert SESSION_TOKEN not in str(exc_info.value)
    assert CSRF_TOKEN not in str(exc_info.value)
    assert "postgresql" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert session_factory.transactions[0].rolled_back is True


@pytest.mark.parametrize(
    ("operation", "failure_stage"),
    [
        ("issue", "session-digest"),
        ("issue", "generation"),
        ("verify", "csrf-digest"),
        ("verify", "comparison"),
    ],
)
def test_internal_failures_are_sanitized(
    monkeypatch,
    operation,
    failure_stage,
):
    secret_detail = f"internal detail containing {SESSION_TOKEN}/{CSRF_TOKEN}"
    error = RuntimeError(secret_detail)
    session_factory, _, _ = _install_backend(
        monkeypatch,
        user_session=StubSession(csrf_token_digest=CSRF_TOKEN_DIGEST),
    )

    def fail(*args, **kwargs):
        raise error

    if failure_stage == "session-digest":
        monkeypatch.setattr(human_service, "digest_session_token", fail)
    elif failure_stage == "generation":
        monkeypatch.setattr(human_service, "_generate_csrf_token", fail)
    elif failure_stage == "csrf-digest":
        monkeypatch.setattr(human_service, "_digest_csrf_token", fail)
    else:
        monkeypatch.setattr(human_service.hmac, "compare_digest", fail)

    with pytest.raises(AuthInfrastructureError) as exc_info:
        if operation == "issue":
            human_service.issue_csrf_token(session_token=SESSION_TOKEN)
        else:
            human_service.verify_csrf_token(
                session_token=SESSION_TOKEN,
                csrf_token=CSRF_TOKEN,
            )

    assert str(exc_info.value) == "Human authentication unavailable"
    assert SESSION_TOKEN not in str(exc_info.value)
    assert CSRF_TOKEN not in str(exc_info.value)
    assert secret_detail not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    if failure_stage in {"session-digest", "csrf-digest"}:
        assert session_factory.transactions == []
    else:
        assert session_factory.transactions[0].rolled_back is True


def test_issued_token_is_not_returned_when_commit_fails(monkeypatch):
    secret_detail = f"commit failed for {SESSION_TOKEN}/{CSRF_TOKEN}"
    session_factory, _, _ = _install_backend(
        monkeypatch,
        user_session=StubSession(),
        commit_error=RuntimeError(secret_detail),
    )

    with pytest.raises(AuthInfrastructureError) as exc_info:
        human_service.issue_csrf_token(session_token=SESSION_TOKEN)

    assert str(exc_info.value) == "Human authentication unavailable"
    assert SESSION_TOKEN not in str(exc_info.value)
    assert CSRF_TOKEN not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert session_factory.transactions[0].committed is False


def test_postgresql_issuance_and_rotation_persist_only_current_digest(
    monkeypatch,
    postgres_csrf_engine,
):
    initial_activity = datetime.now(timezone.utc)
    identity = _create_persisted_session(
        postgres_csrf_engine,
        initial_activity=initial_activity,
    )
    first_token = "postgres-first-csrf-token"
    second_token = "postgres-second-csrf-token"
    generated_tokens = iter([first_token, second_token])
    monkeypatch.setattr(
        human_service,
        "SessionLocal",
        sessionmaker(bind=postgres_csrf_engine),
    )
    monkeypatch.setattr(
        human_service,
        "_generate_csrf_token",
        lambda: next(generated_tokens),
    )
    monkeypatch.setattr(
        human_service,
        "_utc_now",
        lambda: initial_activity + timedelta(minutes=5),
    )

    assert human_service.issue_csrf_token(session_token=identity.token) == first_token
    with Session(postgres_csrf_engine) as verification_session:
        persisted = verification_session.get(UserSession, identity.session_id)
        assert persisted is not None
        assert persisted.csrf_token_digest == hashlib.sha256(
            first_token.encode("ascii")
        ).digest()
        assert len(persisted.csrf_token_digest) == 32
        assert persisted.csrf_token_digest != first_token.encode("ascii")

    assert human_service.issue_csrf_token(session_token=identity.token) == second_token
    with Session(postgres_csrf_engine) as verification_session:
        persisted = verification_session.get(UserSession, identity.session_id)
        assert persisted is not None
        assert persisted.csrf_token_digest == hashlib.sha256(
            second_token.encode("ascii")
        ).digest()
        assert persisted.expires_at == identity.expires_at
        assert persisted.last_activity_at == identity.last_activity_at
        assert persisted.user_id == identity.user_id
        assert persisted.membership_id == identity.membership_id
        assert persisted.company_id == identity.company_id
        assert persisted.token_digest == identity.token_digest

    with pytest.raises(InvalidCsrfTokenError, match="^Invalid CSRF token$"):
        human_service.verify_csrf_token(
            session_token=identity.token,
            csrf_token=first_token,
        )
    assert (
        human_service.verify_csrf_token(
            session_token=identity.token,
            csrf_token=second_token,
        )
        is None
    )


def test_concurrent_issuance_is_serialized_and_last_commit_wins(
    monkeypatch,
    postgres_csrf_engine,
):
    initial_activity = datetime.now(timezone.utc)
    identity = _create_persisted_session(
        postgres_csrf_engine,
        initial_activity=initial_activity,
    )
    first_token = "concurrent-first-csrf-token"
    second_token = "concurrent-second-csrf-token"
    monkeypatch.setattr(
        human_service,
        "SessionLocal",
        sessionmaker(bind=postgres_csrf_engine),
    )
    original_locked_lookup = (
        human_service.find_user_session_by_token_digest_for_update
    )
    first_holds_lock = Event()
    release_first = Event()
    second_attempting_lock = Event()
    second_acquired_lock = Event()
    issued_tokens = {}
    errors = {}

    def controlled_locked_lookup(session, *, token_digest):
        thread_name = current_thread().name
        if thread_name == "csrf-second":
            second_attempting_lock.set()
        persisted = original_locked_lookup(session, token_digest=token_digest)
        if thread_name == "csrf-first":
            first_holds_lock.set()
            if not release_first.wait(timeout=5):
                raise RuntimeError("test timed out releasing first issuance")
        elif thread_name == "csrf-second":
            second_acquired_lock.set()
        return persisted

    def generate_for_thread():
        if current_thread().name == "csrf-first":
            return first_token
        return second_token

    def issue(name):
        try:
            issued_tokens[name] = human_service.issue_csrf_token(
                session_token=identity.token
            )
        except Exception as exc:
            errors[name] = exc

    monkeypatch.setattr(
        human_service,
        "find_user_session_by_token_digest_for_update",
        controlled_locked_lookup,
    )
    monkeypatch.setattr(human_service, "_generate_csrf_token", generate_for_thread)
    monkeypatch.setattr(
        human_service,
        "_utc_now",
        lambda: initial_activity + timedelta(minutes=5),
    )

    first = Thread(target=issue, args=("first",), name="csrf-first")
    second = Thread(target=issue, args=("second",), name="csrf-second")
    first.start()
    assert first_holds_lock.wait(timeout=5)
    second.start()
    assert second_attempting_lock.wait(timeout=5)
    second_was_blocked = not second_acquired_lock.wait(timeout=0.25)
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert second_was_blocked is True
    assert first.is_alive() is False
    assert second.is_alive() is False
    assert errors == {}
    assert issued_tokens == {"first": first_token, "second": second_token}
    assert second_acquired_lock.is_set()

    with Session(postgres_csrf_engine) as verification_session:
        persisted = verification_session.get(UserSession, identity.session_id)
        assert persisted is not None
        assert persisted.csrf_token_digest == hashlib.sha256(
            second_token.encode("ascii")
        ).digest()
        assert persisted.expires_at == identity.expires_at
        assert persisted.last_activity_at == identity.last_activity_at

    with pytest.raises(InvalidCsrfTokenError, match="^Invalid CSRF token$"):
        human_service.verify_csrf_token(
            session_token=identity.token,
            csrf_token=first_token,
        )
    assert (
        human_service.verify_csrf_token(
            session_token=identity.token,
            csrf_token=second_token,
        )
        is None
    )
