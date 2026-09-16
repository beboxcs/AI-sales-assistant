from __future__ import annotations

import logging
from collections.abc import Generator
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
PREVIOUS_REVISION = "9d7c2a5e4f1b"
INVITATION_REVISION = "e6b2c9d4a7f1"


@dataclass(frozen=True)
class MigrationDatabase:
    config: Config
    engine: Engine


@pytest.fixture(autouse=True)
def preserve_existing_logger_states() -> Generator[None, None, None]:
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
def migration_database(monkeypatch: pytest.MonkeyPatch):
    assert application_engine.dialect.name == "postgresql"
    schema_name = f"company_invitation_migration_{uuid4().hex}"
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


def test_upgrade_creates_invitation_schema_and_preserves_existing_auth_data(
    migration_database: MigrationDatabase,
):
    command.upgrade(migration_database.config, PREVIOUS_REVISION)
    with migration_database.engine.begin() as connection:
        _insert_company_creator(connection)

    command.upgrade(migration_database.config, INVITATION_REVISION)

    inspector = sa.inspect(migration_database.engine)
    columns = {
        column["name"]: column
        for column in inspector.get_columns("company_invitations")
    }
    assert list(columns) == [
        "id",
        "company_id",
        "normalized_email",
        "role",
        "token_digest",
        "created_by_user_id",
        "created_at",
        "expires_at",
        "consumed_at",
        "revoked_at",
    ]
    assert columns["normalized_email"]["type"].length == 320
    assert columns["role"]["type"].length == 32
    assert isinstance(columns["token_digest"]["type"], sa.LargeBinary)
    for timestamp_name in (
        "created_at",
        "expires_at",
        "consumed_at",
        "revoked_at",
    ):
        assert columns[timestamp_name]["type"].timezone is True
    assert columns["consumed_at"]["nullable"] is True
    assert columns["revoked_at"]["nullable"] is True
    for name in set(columns) - {"consumed_at", "revoked_at"}:
        assert columns[name]["nullable"] is False

    unique_constraints = {
        constraint["name"]: constraint
        for constraint in inspector.get_unique_constraints(
            "company_invitations"
        )
    }
    assert unique_constraints["uq_company_invitations_token_digest"][
        "column_names"
    ] == ["token_digest"]
    indexes = {
        index["name"]: index
        for index in inspector.get_indexes("company_invitations")
    }
    assert indexes["ix_company_invitations_company_email"][
        "column_names"
    ] == ["company_id", "normalized_email"]
    assert indexes["ix_company_invitations_created_by_user_id"][
        "column_names"
    ] == ["created_by_user_id"]
    assert indexes["uq_company_invitations_token_digest"]["unique"] is True

    assert {
        constraint["name"]
        for constraint in inspector.get_check_constraints(
            "company_invitations"
        )
    } == {
        "ck_company_invitations_email_not_blank",
        "ck_company_invitations_role_allowed",
        "ck_company_invitations_token_digest_sha256_length",
        "ck_company_invitations_expiration_after_creation",
        "ck_company_invitations_single_terminal_state",
        "ck_company_invitations_consumed_after_creation",
        "ck_company_invitations_revoked_after_creation",
    }
    foreign_keys = {
        foreign_key["name"]: foreign_key
        for foreign_key in inspector.get_foreign_keys("company_invitations")
    }
    assert foreign_keys["fk_company_invitations_company"][
        "referred_table"
    ] == "companies"
    assert foreign_keys["fk_company_invitations_creator"][
        "referred_table"
    ] == "users"
    assert all(
        foreign_key["options"]["ondelete"] == "RESTRICT"
        for foreign_key in foreign_keys.values()
    )

    with migration_database.engine.connect() as connection:
        assert connection.execute(
            sa.text("SELECT normalized_email FROM users WHERE id = 20")
        ).scalar_one() == "creator@example.com"
        assert connection.execute(
            sa.text(
                "SELECT role FROM company_memberships WHERE id = 30"
            )
        ).scalar_one() == MembershipRole.COMPANY_ADMIN.value
    assert _current_revision(migration_database.engine) == INVITATION_REVISION


def test_postgresql_invitation_constraints_are_enforced(
    migration_database: MigrationDatabase,
):
    command.upgrade(migration_database.config, INVITATION_REVISION)
    created_at = datetime.now(timezone.utc)
    with migration_database.engine.begin() as connection:
        _insert_company_creator(connection)
        _insert_invitation(
            connection,
            invitation_id=100,
            token_digest=bytes(32),
            created_at=created_at,
            expires_at=created_at + timedelta(hours=24),
        )

    invalid_rows = [
        (
            "uq_company_invitations_token_digest",
            {"invitation_id": 101, "token_digest": bytes(32)},
        ),
        (
            "ck_company_invitations_token_digest_sha256_length",
            {"invitation_id": 102, "token_digest": bytes(31)},
        ),
        (
            "ck_company_invitations_role_allowed",
            {"invitation_id": 103, "token_digest": bytes([3]) * 32, "role": "owner"},
        ),
        (
            "ck_company_invitations_email_not_blank",
            {"invitation_id": 104, "token_digest": bytes([4]) * 32, "email": "   "},
        ),
        (
            "ck_company_invitations_expiration_after_creation",
            {
                "invitation_id": 105,
                "token_digest": bytes([5]) * 32,
                "expires_at": created_at,
            },
        ),
        (
            "ck_company_invitations_single_terminal_state",
            {
                "invitation_id": 106,
                "token_digest": bytes([6]) * 32,
                "consumed_at": created_at + timedelta(minutes=1),
                "revoked_at": created_at + timedelta(minutes=1),
            },
        ),
    ]
    for constraint_name, overrides in invalid_rows:
        values = {
            "invitation_id": overrides["invitation_id"],
            "token_digest": overrides["token_digest"],
            "created_at": created_at,
            "expires_at": created_at + timedelta(hours=24),
            "role": MembershipRole.SALESPERSON.value,
            "email": "invited@example.com",
            "consumed_at": None,
            "revoked_at": None,
        }
        values.update(overrides)
        with pytest.raises(IntegrityError, match=constraint_name):
            with migration_database.engine.begin() as connection:
                _insert_invitation(connection, **values)


def test_downgrade_removes_only_invitation_table(
    migration_database: MigrationDatabase,
):
    command.upgrade(migration_database.config, INVITATION_REVISION)
    with migration_database.engine.begin() as connection:
        _insert_company_creator(connection)

    command.downgrade(migration_database.config, PREVIOUS_REVISION)

    inspector = sa.inspect(migration_database.engine)
    assert "company_invitations" not in inspector.get_table_names()
    assert {"companies", "users", "company_memberships", "user_sessions"}.issubset(
        inspector.get_table_names()
    )
    with migration_database.engine.connect() as connection:
        assert connection.scalar(sa.text("SELECT count(*) FROM companies")) == 1
        assert connection.scalar(sa.text("SELECT count(*) FROM users")) == 1
        assert connection.scalar(
            sa.text("SELECT count(*) FROM company_memberships")
        ) == 1
    assert _current_revision(migration_database.engine) == PREVIOUS_REVISION


def test_alembic_check_reports_no_invitation_model_drift(
    migration_database: MigrationDatabase,
):
    command.upgrade(migration_database.config, "head")
    command.check(migration_database.config)


def _insert_company_creator(connection) -> None:
    connection.execute(
        sa.text("INSERT INTO companies (id, name) VALUES (10, 'Invite Co')")
    )
    connection.execute(
        sa.text(
            "INSERT INTO users (id, normalized_email, password_hash) VALUES "
            "(20, 'creator@example.com', '$argon2id$creator')"
        )
    )
    connection.execute(
        sa.text(
            "INSERT INTO company_memberships "
            "(id, user_id, company_id, role) VALUES "
            "(30, 20, 10, :role)"
        ),
        {"role": MembershipRole.COMPANY_ADMIN.value},
    )


def _insert_invitation(
    connection,
    *,
    invitation_id: int,
    token_digest: bytes,
    created_at: datetime,
    expires_at: datetime,
    role: str = MembershipRole.SALESPERSON.value,
    email: str = "invited@example.com",
    consumed_at: datetime | None = None,
    revoked_at: datetime | None = None,
) -> None:
    connection.execute(
        sa.text(
            "INSERT INTO company_invitations "
            "(id, company_id, normalized_email, role, token_digest, "
            "created_by_user_id, created_at, expires_at, consumed_at, revoked_at) "
            "VALUES (:id, 10, :email, :role, :token_digest, 20, :created_at, "
            ":expires_at, :consumed_at, :revoked_at)"
        ),
        {
            "id": invitation_id,
            "email": email,
            "role": role,
            "token_digest": token_digest,
            "created_at": created_at,
            "expires_at": expires_at,
            "consumed_at": consumed_at,
            "revoked_at": revoked_at,
        },
    )


def _alembic_config() -> Config:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    return config


def _current_revision(database_engine: Engine) -> str | None:
    with database_engine.connect() as connection:
        return connection.scalar(sa.text("SELECT version_num FROM alembic_version"))
