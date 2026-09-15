from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from app.database import engine as application_engine
from app.domain.enums import MembershipRole

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
PREVIOUS_REVISION = "f4a7d9c2e1b6"
CSRF_REVISION = "9d7c2a5e4f1b"


@dataclass(frozen=True)
class MigrationDatabase:
    config: Config
    engine: Engine


@pytest.fixture(autouse=True)
def preserve_existing_logger_states() -> Iterator[None]:
    logger_states = {
        logger: logger.disabled
        for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    }
    try:
        yield
    finally:
        for logger, disabled in logger_states.items():
            logger.disabled = disabled


@pytest.fixture
def migration_database(monkeypatch: pytest.MonkeyPatch) -> Iterator[MigrationDatabase]:
    assert application_engine.dialect.name == "postgresql"
    schema_name = f"session_csrf_migration_{uuid4().hex}"
    quoted_schema_name = application_engine.dialect.identifier_preparer.quote(
        schema_name
    )
    database_url = application_engine.url.render_as_string(hide_password=False)

    with application_engine.begin() as connection:
        connection.execute(sa.text(f"CREATE SCHEMA {quoted_schema_name}"))

    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("PGOPTIONS", f"-csearch_path={schema_name}")
    isolated_engine = sa.create_engine(
        database_url,
        connect_args={"options": f"-csearch_path={schema_name}"},
        poolclass=NullPool,
    )
    try:
        yield MigrationDatabase(config=_alembic_config(), engine=isolated_engine)
    finally:
        isolated_engine.dispose()
        with application_engine.begin() as connection:
            connection.execute(sa.text(f"DROP SCHEMA {quoted_schema_name} CASCADE"))


def test_upgrade_adds_only_nullable_bytea_csrf_digest(
    migration_database: MigrationDatabase,
) -> None:
    command.upgrade(migration_database.config, PREVIOUS_REVISION)
    created_at = datetime.now(timezone.utc) - timedelta(hours=1)
    expires_at = created_at + timedelta(hours=8)
    _insert_identity_and_session(
        migration_database.engine,
        created_at=created_at,
        expires_at=expires_at,
    )
    before_inspector = sa.inspect(migration_database.engine)
    before_columns = {
        column["name"] for column in before_inspector.get_columns("user_sessions")
    }
    before_constraints = {
        constraint["name"]
        for constraint in before_inspector.get_check_constraints("user_sessions")
    }
    before_indexes = {
        index["name"] for index in before_inspector.get_indexes("user_sessions")
    }

    command.upgrade(migration_database.config, CSRF_REVISION)

    inspector = sa.inspect(migration_database.engine)
    columns = {
        column["name"]: column for column in inspector.get_columns("user_sessions")
    }
    assert set(columns) - before_columns == {"csrf_token_digest"}
    assert str(columns["csrf_token_digest"]["type"]) == "BYTEA"
    assert columns["csrf_token_digest"]["nullable"] is True
    with migration_database.engine.connect() as connection:
        assert connection.scalar(
            sa.text("SELECT csrf_token_digest FROM user_sessions WHERE id = 1000")
        ) is None

    csrf_digest = bytes(range(32))
    with migration_database.engine.begin() as connection:
        connection.execute(
            sa.text(
                "UPDATE user_sessions SET csrf_token_digest = :digest WHERE id = 1000"
            ),
            {"digest": csrf_digest},
        )
    with migration_database.engine.connect() as connection:
        persisted_digest = connection.scalar(
            sa.text("SELECT csrf_token_digest FROM user_sessions WHERE id = 1000")
        )
    assert bytes(persisted_digest) == csrf_digest
    assert {
        constraint["name"]
        for constraint in inspector.get_check_constraints("user_sessions")
    } == before_constraints
    assert {
        index["name"] for index in inspector.get_indexes("user_sessions")
    } == before_indexes
    assert _current_revision(migration_database.engine) == CSRF_REVISION


def test_downgrade_removes_only_csrf_digest(
    migration_database: MigrationDatabase,
) -> None:
    command.upgrade(migration_database.config, PREVIOUS_REVISION)
    created_at = datetime.now(timezone.utc) - timedelta(hours=1)
    expires_at = created_at + timedelta(hours=8)
    _insert_identity_and_session(
        migration_database.engine,
        created_at=created_at,
        expires_at=expires_at,
    )
    before_inspector = sa.inspect(migration_database.engine)
    before_columns = {
        column["name"] for column in before_inspector.get_columns("user_sessions")
    }
    before_constraints = {
        constraint["name"]
        for constraint in before_inspector.get_check_constraints("user_sessions")
    }
    before_indexes = {
        index["name"] for index in before_inspector.get_indexes("user_sessions")
    }
    command.upgrade(migration_database.config, CSRF_REVISION)
    with migration_database.engine.begin() as connection:
        connection.execute(
            sa.text(
                "UPDATE user_sessions SET csrf_token_digest = :digest WHERE id = 1000"
            ),
            {"digest": bytes(range(32))},
        )

    command.downgrade(migration_database.config, PREVIOUS_REVISION)

    inspector = sa.inspect(migration_database.engine)
    after_columns = {
        column["name"] for column in inspector.get_columns("user_sessions")
    }
    assert after_columns == before_columns
    assert "csrf_token_digest" not in after_columns
    assert {
        constraint["name"]
        for constraint in inspector.get_check_constraints("user_sessions")
    } == before_constraints
    assert {
        index["name"] for index in inspector.get_indexes("user_sessions")
    } == before_indexes
    with migration_database.engine.connect() as connection:
        row = connection.execute(
            sa.text(
                "SELECT user_id, membership_id, company_id, token_digest, "
                "expires_at, last_activity_at FROM user_sessions WHERE id = 1000"
            )
        ).one()
    assert row.user_id == 1
    assert row.membership_id == 100
    assert row.company_id == 10
    assert bytes(row.token_digest) == bytes(32)
    assert row.expires_at == expires_at
    assert row.last_activity_at == created_at
    assert _current_revision(migration_database.engine) == PREVIOUS_REVISION


def _insert_identity_and_session(
    database_engine: Engine,
    *,
    created_at: datetime,
    expires_at: datetime,
) -> None:
    with database_engine.begin() as connection:
        connection.execute(
            sa.text("INSERT INTO companies (id, name) VALUES (10, 'CSRF Session Co')")
        )
        connection.execute(
            sa.text(
                "INSERT INTO users (id, normalized_email, password_hash) VALUES "
                "(1, 'csrf-session@example.com', '$argon2id$opaque')"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO company_memberships "
                "(id, user_id, company_id, role) VALUES (100, 1, 10, :role)"
            ),
            {"role": MembershipRole.SALESPERSON.value},
        )
        connection.execute(
            sa.text(
                "INSERT INTO user_sessions "
                "(id, user_id, membership_id, company_id, token_digest, "
                "expires_at, last_activity_at, created_at) VALUES "
                "(1000, 1, 100, 10, :digest, :expires_at, "
                ":last_activity_at, :created_at)"
            ),
            {
                "digest": bytes(32),
                "expires_at": expires_at,
                "last_activity_at": created_at,
                "created_at": created_at,
            },
        )


def _alembic_config() -> Config:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    return config


def _current_revision(database_engine: Engine) -> str | None:
    with database_engine.connect() as connection:
        return connection.scalar(sa.text("SELECT version_num FROM alembic_version"))
