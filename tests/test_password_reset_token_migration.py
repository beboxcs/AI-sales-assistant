from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import NullPool

from app.database import engine

PARENT = "e6b2c9d4a7f1"
REVISION = "a7d3e9f2b6c4"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def preserve_logger_states():
    states = {
        logger: logger.disabled
        for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    }
    try:
        yield
    finally:
        for logger, disabled in states.items():
            logger.disabled = disabled


@pytest.fixture
def migration_db(monkeypatch):
    assert engine.dialect.name == "postgresql"
    schema = f"password_reset_migration_{uuid4().hex}"
    quoted = engine.dialect.identifier_preparer.quote(schema)
    with engine.begin() as connection:
        connection.execute(sa.text(f"CREATE SCHEMA {quoted}"))
    monkeypatch.setenv("DATABASE_URL", engine.url.render_as_string(hide_password=False))
    monkeypatch.setenv("PGOPTIONS", f"-csearch_path={schema}")
    isolated = sa.create_engine(
        engine.url,
        connect_args={"options": f"-csearch_path={schema}"},
        poolclass=NullPool,
    )
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    try:
        command.upgrade(config, PARENT)
        with isolated.begin() as connection:
            connection.execute(sa.text(
                "INSERT INTO users (id, normalized_email, password_hash) "
                "VALUES (1, 'existing@example.com', 'old-password-hash'), "
                "(2, 'second@example.com', 'second-password-hash')"
            ))
            connection.execute(sa.text(
                "INSERT INTO companies (id, name) VALUES (1, 'Existing')"
            ))
            connection.execute(sa.text(
                "INSERT INTO api_credentials (key_id, company_id, secret_hash, name) "
                "VALUES ('existing-b1', 1, :digest, 'Existing B1')"
            ), {"digest": b"a" * 32})
        yield config, isolated
    finally:
        isolated.dispose()
        with engine.begin() as connection:
            connection.execute(sa.text(f"DROP SCHEMA {quoted} CASCADE"))


def insert_token(connection, **changes):
    now = datetime.now(timezone.utc)
    values = dict(
        user_id=1, digest=b"t" * 32, created=now,
        expires=now + timedelta(minutes=30), consumed=None, revoked=None,
    )
    values.update(changes)
    connection.execute(sa.text(
        "INSERT INTO password_reset_tokens "
        "(user_id, token_digest, created_at, expires_at, consumed_at, revoked_at) "
        "VALUES (:user_id, :digest, :created, :expires, :consumed, :revoked)"
    ), values)


def test_upgrade_backfills_defaults_and_matches_schema_conventions(migration_db):
    config, database = migration_db
    script = ScriptDirectory.from_config(config)
    assert script.get_current_head() == REVISION
    assert script.get_revision(REVISION).down_revision == PARENT
    with database.connect() as connection:
        b1_before = tuple(connection.execute(sa.text("SELECT * FROM api_credentials")).one())
    command.upgrade(config, REVISION)
    inspector = sa.inspect(database)
    users = {column["name"]: column for column in inspector.get_columns("users")}
    assert not users["password_version"]["nullable"]
    assert users["password_version"]["default"] == "1"
    assert "ck_users_password_version_positive" in {
        c["name"] for c in inspector.get_check_constraints("users")
    }
    columns = {
        column["name"]: column
        for column in inspector.get_columns("password_reset_tokens")
    }
    assert set(columns) == {
        "id", "user_id", "token_digest", "created_at", "expires_at",
        "consumed_at", "revoked_at",
    }
    for name in ("created_at", "expires_at", "consumed_at", "revoked_at"):
        assert columns[name]["type"].timezone
    for name in ("id", "user_id", "token_digest", "created_at", "expires_at"):
        assert not columns[name]["nullable"]
    assert columns["consumed_at"]["nullable"] and columns["revoked_at"]["nullable"]
    assert inspector.get_pk_constraint("password_reset_tokens")["constrained_columns"] == ["id"]
    foreign_key = inspector.get_foreign_keys("password_reset_tokens")[0]
    assert foreign_key["referred_table"] == "users"
    assert foreign_key["constrained_columns"] == ["user_id"]
    assert foreign_key["options"]["ondelete"] == "RESTRICT"
    assert any(
        c["column_names"] == ["token_digest"]
        for c in inspector.get_unique_constraints("password_reset_tokens")
    )
    indexes = {
        index["name"]: index
        for index in inspector.get_indexes("password_reset_tokens")
    }
    assert indexes["ix_password_reset_tokens_user_id"]["column_names"] == ["user_id"]
    unique = indexes["uq_password_reset_tokens_open_user"]
    assert unique["unique"]
    assert "now" not in unique["dialect_options"]["postgresql_where"].lower()
    with database.begin() as connection:
        assert list(connection.execute(sa.text(
            "SELECT password_version FROM users ORDER BY id"
        )).scalars()) == [1, 1]
        assert tuple(connection.execute(sa.text("SELECT * FROM api_credentials")).one()) == b1_before
        connection.execute(sa.text(
            "INSERT INTO users (id, normalized_email, password_hash) "
            "VALUES (3, 'new@example.com', 'hash')"
        ))
        assert connection.scalar(sa.text(
            "SELECT password_version FROM users WHERE id = 3"
        )) == 1


@pytest.mark.parametrize("version", [0, -1, None])
def test_password_version_positive_not_null_constraints(migration_db, version):
    config, database = migration_db
    command.upgrade(config, REVISION)
    with pytest.raises(IntegrityError):
        with database.begin() as connection:
            connection.execute(sa.text(
                "UPDATE users SET password_version = :version WHERE id = 1"
            ), {"version": version})


@pytest.mark.parametrize("case", [
    "short_digest", "null_digest", "null_user", "unknown_user", "expiry",
    "both_terminal", "consumed_before", "revoked_before", "duplicate_digest",
    "duplicate_open", "delete_user",
])
def test_postgresql_reset_token_constraints(migration_db, case):
    config, database = migration_db
    command.upgrade(config, REVISION)
    now = datetime.now(timezone.utc)
    if case in {"duplicate_digest", "duplicate_open", "delete_user"}:
        with database.begin() as connection:
            insert_token(connection)
    values = {
        "short_digest": {"digest": b"short"},
        "null_digest": {"digest": None},
        "null_user": {"user_id": None},
        "unknown_user": {"user_id": 999},
        "expiry": {"created": now, "expires": now},
        "both_terminal": {"created": now, "consumed": now, "revoked": now},
        "consumed_before": {"created": now, "consumed": now - timedelta(seconds=1)},
        "revoked_before": {"created": now, "revoked": now - timedelta(seconds=1)},
        "duplicate_digest": {"user_id": 2},
        "duplicate_open": {"digest": b"u" * 32},
    }
    with pytest.raises(IntegrityError):
        with database.begin() as connection:
            if case == "delete_user":
                connection.execute(sa.text("DELETE FROM users WHERE id = 1"))
            else:
                insert_token(connection, **values[case])


def test_terminal_history_allows_replacement_but_expired_open_does_not(migration_db):
    config, database = migration_db
    command.upgrade(config, REVISION)
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    with database.begin() as connection:
        insert_token(connection, created=past, expires=past + timedelta(minutes=30))
    with pytest.raises(IntegrityError):
        with database.begin() as connection:
            insert_token(connection, digest=b"u" * 32)
    with database.begin() as connection:
        connection.execute(sa.text("UPDATE password_reset_tokens SET revoked_at = now()"))
        insert_token(connection, digest=b"u" * 32)


def test_downgrade_removes_only_b27a_additions(migration_db):
    config, database = migration_db
    before = set(sa.inspect(database).get_table_names())
    command.upgrade(config, REVISION)
    with database.begin() as connection:
        insert_token(connection)
    command.downgrade(config, PARENT)
    inspector = sa.inspect(database)
    assert set(inspector.get_table_names()) == before
    assert "password_version" not in {
        column["name"] for column in inspector.get_columns("users")
    }
    with database.connect() as connection:
        assert connection.scalar(sa.text("SELECT version_num FROM alembic_version")) == PARENT
        assert connection.scalar(sa.text(
            "SELECT password_hash FROM users WHERE id = 1"
        )) == "old-password-hash"
        assert connection.scalar(sa.text("SELECT count(*) FROM users")) == 2
        assert connection.scalar(sa.text("SELECT count(*) FROM api_credentials")) == 1


def test_alembic_check_has_no_model_drift(migration_db):
    config, database = migration_db
    command.upgrade(config, "head")
    command.check(config)
