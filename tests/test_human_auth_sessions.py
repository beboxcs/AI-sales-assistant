from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.auth.human import (
    AuthInfrastructureError,
    AuthenticatedPrincipal,
    AuthenticatedUserContext,
    CreatedSession,
)
from app.auth.human import service as human_service
from app.auth.human.sessions import (
    SESSION_TOKEN_ENTROPY_BYTES,
    digest_session_token,
    generate_session_token,
)
from app.database import Base, engine
from app.domain.enums import MembershipRole
from app.models import Company, CompanyMembership, User, UserSession

NOW = datetime(2026, 9, 12, 9, 30, tzinfo=timezone.utc)
RAW_TOKEN = "opaque-session-token"
TOKEN_DIGEST = digest_session_token(token=RAW_TOKEN)
PRINCIPAL = AuthenticatedPrincipal(
    user_id=101,
    membership_id=202,
    company_id=303,
    role=MembershipRole.COMPANY_ADMIN,
)


@dataclass
class StubPersistedSession:
    id: int


class RecordingTransactionContext:
    def __init__(self):
        self.session = object()
        self.exited = False

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc_value, traceback):
        self.exited = True
        return False


class RecordingSessionFactory:
    def __init__(self):
        self.contexts = []

    def begin(self):
        context = RecordingTransactionContext()
        self.contexts.append(context)
        return context


@pytest.fixture
def postgres_session_engine() -> Generator[Engine, None, None]:
    assert engine.dialect.name == "postgresql"
    schema_name = f"human_session_{uuid4().hex}"
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


def test_create_session_owns_expiry_and_returns_persisted_context(monkeypatch):
    session_factory = RecordingSessionFactory()
    persisted_values = {}

    def persist(session, **values):
        persisted_values.update(values)
        return StubPersistedSession(id=404)

    monkeypatch.setattr(human_service, "SessionLocal", session_factory)
    monkeypatch.setattr(human_service, "_utc_now", lambda: NOW)
    monkeypatch.setattr(
        human_service,
        "generate_session_token",
        lambda: RAW_TOKEN,
    )
    monkeypatch.setattr(
        human_service,
        "digest_session_token",
        lambda *, token: TOKEN_DIGEST,
    )
    monkeypatch.setattr(human_service, "persist_user_session", persist)

    created = human_service.create_session(principal=PRINCIPAL)

    expected_expiry = NOW + timedelta(hours=8)
    assert created == CreatedSession(
        token=RAW_TOKEN,
        expires_at=expected_expiry,
        context=AuthenticatedUserContext(
            user_id=PRINCIPAL.user_id,
            membership_id=PRINCIPAL.membership_id,
            company_id=PRINCIPAL.company_id,
            role=PRINCIPAL.role,
            session_id=404,
        ),
    )
    assert created.expires_at.tzinfo is timezone.utc
    assert persisted_values == {
        "user_id": PRINCIPAL.user_id,
        "membership_id": PRINCIPAL.membership_id,
        "company_id": PRINCIPAL.company_id,
        "token_digest": TOKEN_DIGEST,
        "expires_at": expected_expiry,
        "last_activity_at": NOW,
    }
    assert persisted_values["expires_at"] is created.expires_at
    assert session_factory.contexts[0].exited is True


def test_session_tokens_are_independent_high_entropy_bearer_credentials():
    first_token = generate_session_token()
    second_token = generate_session_token()

    assert SESSION_TOKEN_ENTROPY_BYTES * 8 >= 256
    assert first_token != second_token
    assert digest_session_token(token=first_token) != digest_session_token(
        token=second_token
    )


def test_token_digest_is_deterministic_non_plaintext_lookup_material():
    first_digest = digest_session_token(token=RAW_TOKEN)
    second_digest = digest_session_token(token=RAW_TOKEN)

    assert first_digest == second_digest
    assert len(first_digest) == 32
    assert first_digest != RAW_TOKEN.encode("ascii")
    assert RAW_TOKEN.encode("ascii") not in first_digest


def test_persistence_failure_maps_to_safe_infrastructure_error(monkeypatch):
    session_factory = RecordingSessionFactory()

    def fail_persistence(session, **values):
        raise RuntimeError(
            f"postgresql://private details containing {RAW_TOKEN}"
        )

    monkeypatch.setattr(human_service, "SessionLocal", session_factory)
    monkeypatch.setattr(human_service, "_utc_now", lambda: NOW)
    monkeypatch.setattr(
        human_service,
        "generate_session_token",
        lambda: RAW_TOKEN,
    )
    monkeypatch.setattr(human_service, "persist_user_session", fail_persistence)

    with pytest.raises(AuthInfrastructureError) as exc_info:
        human_service.create_session(principal=PRINCIPAL)

    assert str(exc_info.value) == "Human authentication unavailable"
    assert "postgresql" not in str(exc_info.value)
    assert RAW_TOKEN not in str(exc_info.value)
    assert session_factory.contexts[0].exited is True


def test_postgresql_session_row_persists_authoritative_identity_and_expiry(
    monkeypatch,
    postgres_session_engine,
):
    frozen_now = datetime.now(timezone.utc)

    with Session(postgres_session_engine) as setup_session:
        company = Company(name="Session Company")
        user = User(
            normalized_email=f"session-{uuid4().hex}@example.com",
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
        principal = AuthenticatedPrincipal(
            user_id=user.id,
            membership_id=membership.id,
            company_id=company.id,
            role=MembershipRole.SALESPERSON,
        )
        setup_session.commit()

    monkeypatch.setattr(human_service, "_utc_now", lambda: frozen_now)
    monkeypatch.setattr(
        human_service,
        "generate_session_token",
        lambda: RAW_TOKEN,
    )
    monkeypatch.setattr(
        human_service,
        "SessionLocal",
        sessionmaker(bind=postgres_session_engine),
    )

    created = human_service.create_session(principal=principal)
    expected_expiry = frozen_now + human_service.SESSION_ABSOLUTE_LIFETIME

    with Session(postgres_session_engine) as verification_session:
        persisted = verification_session.scalar(sa.select(UserSession))
        assert persisted is not None
        assert persisted.id == created.context.session_id
        assert persisted.user_id == principal.user_id
        assert persisted.membership_id == principal.membership_id
        assert persisted.company_id == principal.company_id
        assert persisted.expires_at == created.expires_at
        assert human_service.SESSION_ABSOLUTE_LIFETIME == timedelta(hours=8)
        assert persisted.expires_at == expected_expiry
        assert persisted.expires_at > persisted.created_at
        assert persisted.last_activity_at == frozen_now
        assert persisted.last_activity_at.tzinfo is not None
        assert persisted.last_activity_at.utcoffset() == timedelta(0)
        assert persisted.expires_at.tzinfo is not None
        assert persisted.expires_at.utcoffset() == timedelta(0)
        assert persisted.token_digest == digest_session_token(token=created.token)
        assert persisted.token_digest != created.token.encode("ascii")
        assert persisted.revoked_at is None
