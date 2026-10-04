from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.pool import NullPool

from app.database import engine as application_engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
BASE_REVISION = "3ca60b9db9e3"
HEAD_REVISION = "84c1e9b72f0a"
RECONCILIATION_ERROR = (
    "Cannot add compounds.compound_key because legacy compound rows exist. "
    "Authoritative compound_key reconciliation is required before upgrading."
)


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

    schema_name = f"compound_key_migration_{uuid4().hex}"
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


def test_empty_upgrade_succeeds_with_final_schema(
    migration_database: MigrationDatabase,
) -> None:
    command.upgrade(migration_database.config, BASE_REVISION)

    with migration_database.engine.connect() as connection:
        assert connection.scalar(sa.text("SELECT count(*) FROM compounds")) == 0

    command.upgrade(migration_database.config, HEAD_REVISION)

    inspector = sa.inspect(migration_database.engine)
    columns = {
        column["name"]: column
        for column in inspector.get_columns("compounds")
    }
    unique_constraints = {
        constraint["name"]: constraint
        for constraint in inspector.get_unique_constraints("compounds")
    }

    assert columns["compound_key"]["nullable"] is False
    assert unique_constraints["uq_compounds_company_compound_key"][
        "column_names"
    ] == ["company_id", "compound_key"]
    assert _current_revision(migration_database.engine) == HEAD_REVISION


def test_nonempty_upgrade_fails_clearly_and_atomically(
    migration_database: MigrationDatabase,
) -> None:
    command.upgrade(migration_database.config, BASE_REVISION)
    with migration_database.engine.begin() as connection:
        connection.execute(
            sa.text(
                "INSERT INTO companies (id, name) "
                "VALUES (101, 'Legacy Company')"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO compounds (id, company_id, name, location) "
                "VALUES (202, 101, 'Legacy Compound', 'Legacy Location')"
            )
        )

    with pytest.raises(DBAPIError, match=re.escape(RECONCILIATION_ERROR)):
        command.upgrade(migration_database.config, HEAD_REVISION)

    assert _current_revision(migration_database.engine) == BASE_REVISION
    assert "compound_key" not in {
        column["name"]
        for column in sa.inspect(migration_database.engine).get_columns(
            "compounds"
        )
    }

    with migration_database.engine.connect() as connection:
        legacy_rows = connection.execute(
            sa.text(
                "SELECT id, company_id, name, location "
                "FROM compounds ORDER BY id"
            )
        ).all()

    assert legacy_rows == [
        (202, 101, "Legacy Compound", "Legacy Location")
    ]
    assert "uq_compounds_company_compound_key" not in {
        constraint["name"]
        for constraint in sa.inspect(
            migration_database.engine
        ).get_unique_constraints("compounds")
    }


def test_final_not_null_and_tenant_scoped_unique_constraints_work(
    migration_database: MigrationDatabase,
) -> None:
    command.upgrade(migration_database.config, HEAD_REVISION)

    with migration_database.engine.begin() as connection:
        connection.execute(
            sa.text(
                "INSERT INTO companies (id, name) VALUES "
                "(1, 'Company One'), (2, 'Company Two')"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO compounds "
                "(id, company_id, compound_key, name, location) VALUES "
                "(1, 1, 'shared-key', 'First', 'Cairo'), "
                "(2, 2, 'shared-key', 'Second', 'Giza')"
            )
        )

    with migration_database.engine.connect() as connection:
        assert connection.scalar(
            sa.text(
                "SELECT count(*) FROM compounds "
                "WHERE compound_key = 'shared-key'"
            )
        ) == 2

    with pytest.raises(IntegrityError, match="uq_compounds_company_compound_key"):
        with migration_database.engine.begin() as connection:
            connection.execute(
                sa.text(
                    "INSERT INTO compounds "
                    "(id, company_id, compound_key, name, location) "
                    "VALUES (3, 1, 'shared-key', 'Duplicate', 'Alexandria')"
                )
            )

    with pytest.raises(IntegrityError, match="compound_key"):
        with migration_database.engine.begin() as connection:
            connection.execute(
                sa.text(
                    "INSERT INTO compounds "
                    "(id, company_id, name, location) "
                    "VALUES (4, 1, 'Missing Key', 'Mansoura')"
                )
            )


def test_downgrade_removes_compound_key_schema(
    migration_database: MigrationDatabase,
) -> None:
    command.upgrade(migration_database.config, HEAD_REVISION)
    with migration_database.engine.begin() as connection:
        connection.execute(
            sa.text("INSERT INTO companies (id, name) VALUES (1, 'Company')")
        )
        connection.execute(
            sa.text(
                "INSERT INTO compounds "
                "(id, company_id, compound_key, name, location) "
                "VALUES (1, 1, 'canonical-key', 'Compound', 'Cairo')"
            )
        )

    command.downgrade(migration_database.config, BASE_REVISION)

    inspector = sa.inspect(migration_database.engine)
    assert "compound_key" not in {
        column["name"] for column in inspector.get_columns("compounds")
    }
    assert "uq_compounds_company_compound_key" not in {
        constraint["name"]
        for constraint in inspector.get_unique_constraints("compounds")
    }
    assert _current_revision(migration_database.engine) == BASE_REVISION

    with migration_database.engine.connect() as connection:
        assert connection.execute(
            sa.text(
                "SELECT id, company_id, name, location FROM compounds"
            )
        ).one() == (1, 1, "Compound", "Cairo")


def test_offline_sql_contains_guard_before_add_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = StringIO()
    database_url = application_engine.url.render_as_string(hide_password=False)
    monkeypatch.setenv("DATABASE_URL", database_url)

    command.upgrade(_alembic_config(output_buffer=output), HEAD_REVISION, sql=True)

    migration_sql = output.getvalue()
    lock_position = migration_sql.index(
        "LOCK TABLE compounds IN ACCESS EXCLUSIVE MODE"
    )
    guard_position = migration_sql.index(
        "IF EXISTS (SELECT 1 FROM compounds)"
    )
    add_column_position = migration_sql.index(
        "ALTER TABLE compounds ADD COLUMN compound_key VARCHAR NOT NULL"
    )

    assert RECONCILIATION_ERROR in migration_sql
    assert lock_position < guard_position < add_column_position


def _alembic_config(*, output_buffer: StringIO | None = None) -> Config:
    return Config(str(ALEMBIC_INI), output_buffer=output_buffer)


def _current_revision(engine: Engine) -> str:
    with engine.connect() as connection:
        revision = connection.scalar(
            sa.text("SELECT version_num FROM alembic_version")
        )

    assert isinstance(revision, str)
    return revision
