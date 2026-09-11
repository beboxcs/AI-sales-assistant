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
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import NullPool

from app.database import engine as application_engine
from app.domain.enums import MembershipRole

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
PREVIOUS_REVISION = "b7c9d2e4f6a8"
SESSION_REVISION = "c8e1f4a6b2d9"


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
    schema_name = f"user_session_migration_{uuid4().hex}"
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
            connection.execute(
                sa.text(f"DROP SCHEMA {quoted_schema_name} CASCADE")
            )


def test_upgrade_creates_secure_tenant_bound_session_schema(
    migration_database: MigrationDatabase,
) -> None:
    command.upgrade(migration_database.config, SESSION_REVISION)

    inspector = sa.inspect(migration_database.engine)
    columns = {
        column["name"]: column for column in inspector.get_columns("user_sessions")
    }
    assert list(columns) == [
        "id",
        "user_id",
        "membership_id",
        "company_id",
        "token_digest",
        "expires_at",
        "revoked_at",
        "created_at",
        "updated_at",
    ]
    assert isinstance(columns["token_digest"]["type"], sa.LargeBinary)
    assert columns["revoked_at"]["nullable"] is True
    assert columns["expires_at"]["type"].timezone is True
    assert columns["created_at"]["type"].timezone is True
    assert columns["updated_at"]["type"].timezone is True
    for column_name in set(columns) - {"revoked_at"}:
        assert columns[column_name]["nullable"] is False

    unique_constraints = {
        constraint["name"]: constraint
        for constraint in inspector.get_unique_constraints("user_sessions")
    }
    assert unique_constraints["uq_user_sessions_token_digest"][
        "column_names"
    ] == ["token_digest"]
    membership_uniques = {
        constraint["name"]: constraint
        for constraint in inspector.get_unique_constraints("company_memberships")
    }
    assert membership_uniques["uq_company_memberships_id_user_company"][
        "column_names"
    ] == ["id", "user_id", "company_id"]

    foreign_keys = {
        foreign_key["name"]: foreign_key
        for foreign_key in inspector.get_foreign_keys("user_sessions")
    }
    identity_foreign_key = foreign_keys["fk_user_sessions_membership_identity"]
    assert identity_foreign_key["constrained_columns"] == [
        "membership_id",
        "user_id",
        "company_id",
    ]
    assert identity_foreign_key["referred_columns"] == [
        "id",
        "user_id",
        "company_id",
    ]
    assert identity_foreign_key["options"]["ondelete"] == "RESTRICT"

    check_names = {
        constraint["name"]
        for constraint in inspector.get_check_constraints("user_sessions")
    }
    assert check_names == {
        "ck_user_sessions_expiration_after_creation",
        "ck_user_sessions_token_digest_sha256_length",
    }
    indexes = {
        index["name"]: index
        for index in inspector.get_indexes("user_sessions")
    }
    assert indexes["ix_user_sessions_company_id"]["column_names"] == [
        "company_id"
    ]


def test_session_constraints_reject_invalid_digest_and_mixed_identity(
    migration_database: MigrationDatabase,
) -> None:
    command.upgrade(migration_database.config, SESSION_REVISION)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=8)
    with migration_database.engine.begin() as connection:
        connection.execute(
            sa.text(
                "INSERT INTO companies (id, name) VALUES "
                "(10, 'Company Ten'), (20, 'Company Twenty')"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO users (id, normalized_email, password_hash) VALUES "
                "(1, 'one@example.com', '$argon2id$opaque')"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO company_memberships "
                "(id, user_id, company_id, role) VALUES "
                "(100, 1, 10, :role)"
            ),
            {"role": MembershipRole.SALESPERSON.value},
        )
        connection.execute(
            sa.text(
                "INSERT INTO user_sessions "
                "(id, user_id, membership_id, company_id, token_digest, expires_at) "
                "VALUES (1000, 1, 100, 10, :digest, :expires_at)"
            ),
            {"digest": bytes(32), "expires_at": expires_at},
        )

    with pytest.raises(
        IntegrityError,
        match="ck_user_sessions_token_digest_sha256_length",
    ):
        with migration_database.engine.begin() as connection:
            connection.execute(
                sa.text(
                    "INSERT INTO user_sessions "
                    "(id, user_id, membership_id, company_id, token_digest, "
                    "expires_at) "
                    "VALUES (1001, 1, 100, 10, :digest, :expires_at)"
                ),
                {"digest": bytes(31), "expires_at": expires_at},
            )

    with pytest.raises(
        IntegrityError,
        match="fk_user_sessions_membership_identity",
    ):
        with migration_database.engine.begin() as connection:
            connection.execute(
                sa.text(
                    "INSERT INTO user_sessions "
                    "(id, user_id, membership_id, company_id, token_digest, "
                    "expires_at) "
                    "VALUES (1002, 1, 100, 20, :digest, :expires_at)"
                ),
                {"digest": bytes([1]) * 32, "expires_at": expires_at},
            )


def test_downgrade_removes_only_session_schema(
    migration_database: MigrationDatabase,
) -> None:
    command.upgrade(migration_database.config, SESSION_REVISION)
    with migration_database.engine.begin() as connection:
        connection.execute(
            sa.text("INSERT INTO companies (id, name) VALUES (10, 'Preserved Co')")
        )
        connection.execute(
            sa.text(
                "INSERT INTO users (id, normalized_email, password_hash) VALUES "
                "(1, 'preserved@example.com', '$argon2id$opaque')"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO company_memberships "
                "(id, user_id, company_id, role) VALUES "
                "(100, 1, 10, :role)"
            ),
            {"role": MembershipRole.COMPANY_ADMIN.value},
        )

    command.downgrade(migration_database.config, PREVIOUS_REVISION)

    inspector = sa.inspect(migration_database.engine)
    assert "user_sessions" not in inspector.get_table_names()
    assert {"users", "company_memberships"}.issubset(inspector.get_table_names())
    assert "uq_company_memberships_id_user_company" not in {
        constraint["name"]
        for constraint in inspector.get_unique_constraints("company_memberships")
    }
    with migration_database.engine.connect() as connection:
        assert connection.execute(
            sa.text("SELECT id, company_id FROM company_memberships")
        ).one() == (100, 10)
    assert _current_revision(migration_database.engine) == PREVIOUS_REVISION


def _alembic_config() -> Config:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    return config


def _current_revision(engine: Engine) -> str | None:
    with engine.connect() as connection:
        return connection.scalar(sa.text("SELECT version_num FROM alembic_version"))
