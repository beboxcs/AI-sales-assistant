from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.auth import service as auth_service
from app.auth.api_keys import generate_api_key, hash_api_key_secret, parse_api_key
from app.auth.contracts import TrustedTenantContext
from app.auth.repository import find_api_credential_by_key_id
from app.database import engine as application_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
PREVIOUS_REVISION = "84c1e9b72f0a"
AUTH_REVISION = "d4e8f2a7c1b3"


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

    schema_name = f"api_credentials_migration_{uuid4().hex}"
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


def test_upgrade_preserves_existing_tenant_inventory_and_creates_exact_schema(
    migration_database: MigrationDatabase,
) -> None:
    command.upgrade(migration_database.config, PREVIOUS_REVISION)
    _insert_representative_inventory(migration_database.engine)

    command.upgrade(migration_database.config, AUTH_REVISION)

    inspector = sa.inspect(migration_database.engine)
    columns = {
        column["name"]: column
        for column in inspector.get_columns("api_credentials")
    }
    assert list(columns) == [
        "id",
        "key_id",
        "company_id",
        "secret_hash",
        "name",
        "is_active",
        "created_at",
        "expires_at",
        "revoked_at",
        "updated_at",
    ]
    assert isinstance(columns["id"]["type"], sa.Integer)
    assert isinstance(columns["company_id"]["type"], sa.Integer)
    assert isinstance(columns["key_id"]["type"], sa.String)
    assert columns["key_id"]["type"].length == 64
    assert isinstance(columns["secret_hash"]["type"], sa.LargeBinary)
    assert isinstance(columns["name"]["type"], sa.String)
    assert columns["name"]["type"].length == 255
    assert isinstance(columns["is_active"]["type"], sa.Boolean)
    for column_name in ("created_at", "expires_at", "revoked_at", "updated_at"):
        assert isinstance(columns[column_name]["type"], sa.DateTime)
        assert columns[column_name]["type"].timezone is True

    assert columns["expires_at"]["nullable"] is True
    assert columns["revoked_at"]["nullable"] is True
    for column_name in set(columns) - {"expires_at", "revoked_at"}:
        assert columns[column_name]["nullable"] is False
    assert "true" in columns["is_active"]["default"].lower()
    assert "now()" in columns["created_at"]["default"].lower()
    assert "now()" in columns["updated_at"]["default"].lower()

    unique_constraints = {
        constraint["name"]: constraint
        for constraint in inspector.get_unique_constraints("api_credentials")
    }
    assert unique_constraints["uq_api_credentials_key_id"]["column_names"] == [
        "key_id"
    ]
    indexes = {
        index["name"]: index
        for index in inspector.get_indexes("api_credentials")
    }
    assert indexes["ix_api_credentials_company_id"]["column_names"] == [
        "company_id"
    ]
    foreign_keys = {
        foreign_key["name"]: foreign_key
        for foreign_key in inspector.get_foreign_keys("api_credentials")
    }
    company_fk = foreign_keys["fk_api_credentials_company"]
    assert company_fk["constrained_columns"] == ["company_id"]
    assert company_fk["referred_table"] == "companies"
    assert company_fk["referred_columns"] == ["id"]
    assert company_fk["options"]["ondelete"] == "RESTRICT"
    assert {
        constraint["name"]
        for constraint in inspector.get_check_constraints("api_credentials")
    } == {
        "ck_api_credentials_key_id_not_blank",
        "ck_api_credentials_name_not_blank",
        "ck_api_credentials_revoked_inactive",
        "ck_api_credentials_secret_hash_sha256_length",
    }

    with migration_database.engine.connect() as connection:
        assert connection.execute(
            sa.text(
                "SELECT id, company_id, compound_key, name "
                "FROM compounds ORDER BY id"
            )
        ).all() == [(202, 101, "legacy-compound", "Legacy Compound")]
        assert connection.execute(
            sa.text(
                "SELECT id, company_id, scope_key "
                "FROM inventory_scopes ORDER BY id"
            )
        ).all() == [(303, 101, "legacy-scope")]
    assert _current_revision(migration_database.engine) == AUTH_REVISION


def test_postgresql_constraints_and_tenant_relationships(
    migration_database: MigrationDatabase,
) -> None:
    command.upgrade(migration_database.config, AUTH_REVISION)
    with migration_database.engine.begin() as connection:
        connection.execute(
            sa.text(
                "INSERT INTO companies (id, name) VALUES "
                "(1, 'Company One'), (2, 'Company Two')"
            )
        )
        _insert_credential(connection, 11, "A" * 22, 1)
        _insert_credential(connection, 12, "B" * 22, 1)
        _insert_credential(connection, 21, "C" * 22, 2)

    with migration_database.engine.connect() as connection:
        rows = connection.execute(
            sa.text(
                "SELECT id, company_id FROM api_credentials "
                "ORDER BY company_id, id"
            )
        ).all()
    assert rows == [(11, 1), (12, 1), (21, 2)]

    invalid_inserts = [
        (
            "uq_api_credentials_key_id",
            {
                "id": 31,
                "key_id": "A" * 22,
                "company_id": 2,
                "secret_hash": bytes(32),
                "name": "Duplicate key",
                "is_active": True,
                "revoked_at": None,
            },
        ),
        (
            "ck_api_credentials_secret_hash_sha256_length",
            {
                "id": 32,
                "key_id": "D" * 22,
                "company_id": 1,
                "secret_hash": bytes(31),
                "name": "Short digest",
                "is_active": True,
                "revoked_at": None,
            },
        ),
        (
            "ck_api_credentials_secret_hash_sha256_length",
            {
                "id": 37,
                "key_id": "H" * 22,
                "company_id": 1,
                "secret_hash": bytes(33),
                "name": "Long digest",
                "is_active": True,
                "revoked_at": None,
            },
        ),
        (
            "fk_api_credentials_company",
            {
                "id": 33,
                "key_id": "E" * 22,
                "company_id": 999,
                "secret_hash": bytes(32),
                "name": "Unknown company",
                "is_active": True,
                "revoked_at": None,
            },
        ),
        (
            "ck_api_credentials_revoked_inactive",
            {
                "id": 36,
                "key_id": "G" * 22,
                "company_id": 1,
                "secret_hash": bytes(32),
                "name": "Invalid revocation",
                "is_active": True,
                "revoked_at": "2026-09-08T00:00:00+00:00",
            },
        ),
    ]
    for constraint_name, parameters in invalid_inserts:
        with pytest.raises(IntegrityError, match=constraint_name):
            with migration_database.engine.begin() as connection:
                connection.execute(
                    sa.text(
                        "INSERT INTO api_credentials "
                        "(id, key_id, company_id, secret_hash, name, "
                        "is_active, revoked_at) VALUES "
                        "(:id, :key_id, :company_id, :secret_hash, :name, "
                        ":is_active, :revoked_at)"
                    ),
                    parameters,
                )

    whitespace_only_values = ("", "   ", "\t", "\n", " \t\n ")
    for whitespace_only in whitespace_only_values:
        with pytest.raises(
            IntegrityError,
            match="ck_api_credentials_key_id_not_blank",
        ):
            with migration_database.engine.begin() as connection:
                connection.execute(
                    sa.text(
                        "INSERT INTO api_credentials "
                        "(id, key_id, company_id, secret_hash, name) VALUES "
                        "(34, :key_id, 1, :secret_hash, 'Blank key')"
                    ),
                    {"key_id": whitespace_only, "secret_hash": bytes(32)},
                )

        with pytest.raises(
            IntegrityError,
            match="ck_api_credentials_name_not_blank",
        ):
            with migration_database.engine.begin() as connection:
                connection.execute(
                    sa.text(
                        "INSERT INTO api_credentials "
                        "(id, key_id, company_id, secret_hash, name) VALUES "
                        "(35, :key_id, 1, :secret_hash, :name)"
                    ),
                    {
                        "key_id": "F" * 22,
                        "secret_hash": bytes(32),
                        "name": whitespace_only,
                    },
                )

    with pytest.raises(IntegrityError, match="fk_api_credentials_company"):
        with migration_database.engine.begin() as connection:
            connection.execute(sa.text("DELETE FROM companies WHERE id = 1"))


def test_secret_persistence_repository_and_real_authentication(
    migration_database: MigrationDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command.upgrade(migration_database.config, AUTH_REVISION)
    raw_key = generate_api_key()
    key_id, secret = parse_api_key(raw_key)
    pepper = "never-persist-this-pepper"
    digest = hash_api_key_secret(secret=secret, pepper=pepper)

    with migration_database.engine.begin() as connection:
        connection.execute(
            sa.text("INSERT INTO companies (id, name) VALUES (7, 'Tenant Seven')")
        )
        connection.execute(
            sa.text(
                "INSERT INTO api_credentials "
                "(id, key_id, company_id, secret_hash, name) VALUES "
                "(70, :key_id, 7, :secret_hash, 'Primary integration key')"
            ),
            {"key_id": key_id, "secret_hash": digest},
        )

    with Session(migration_database.engine) as session:
        credential = find_api_credential_by_key_id(session, key_id=key_id)
        assert credential is not None
        assert credential.id == 70
        assert credential.company_id == 7
        assert isinstance(credential.secret_hash, bytes)
        assert len(credential.secret_hash) == 32
        assert credential.secret_hash == digest
        assert credential.secret_hash != secret.encode("ascii")
        assert credential.is_active is True
        assert credential.created_at is not None
        assert credential.updated_at is not None

    with migration_database.engine.connect() as connection:
        persisted = connection.execute(
            sa.text("SELECT * FROM api_credentials WHERE id = 70")
        ).mappings().one()
    persisted_text = repr(dict(persisted))
    assert raw_key not in persisted_text
    assert secret not in persisted_text
    assert pepper not in persisted_text
    assert "pepper" not in persisted

    monkeypatch.setattr(
        auth_service,
        "SessionLocal",
        sessionmaker(bind=migration_database.engine),
    )
    assert auth_service.authenticate_api_key(raw_key, pepper=pepper) == (
        TrustedTenantContext(company_id=7, credential_id=70)
    )


def test_downgrade_removes_only_auth_schema_and_returns_to_previous_revision(
    migration_database: MigrationDatabase,
) -> None:
    command.upgrade(migration_database.config, PREVIOUS_REVISION)
    _insert_representative_inventory(migration_database.engine)
    command.upgrade(migration_database.config, AUTH_REVISION)
    with migration_database.engine.begin() as connection:
        _insert_credential(connection, 404, "Z" * 22, 101)

    command.downgrade(migration_database.config, PREVIOUS_REVISION)

    inspector = sa.inspect(migration_database.engine)
    assert "api_credentials" not in inspector.get_table_names()
    with migration_database.engine.connect() as connection:
        assert connection.execute(
            sa.text("SELECT id, name FROM companies ORDER BY id")
        ).all() == [(101, "Legacy Company")]
        assert connection.execute(
            sa.text("SELECT id, compound_key FROM compounds ORDER BY id")
        ).all() == [(202, "legacy-compound")]
        assert connection.execute(
            sa.text("SELECT id, scope_key FROM inventory_scopes ORDER BY id")
        ).all() == [(303, "legacy-scope")]
    assert _current_revision(migration_database.engine) == PREVIOUS_REVISION


def test_alembic_check_reports_no_model_drift(
    migration_database: MigrationDatabase,
) -> None:
    command.upgrade(migration_database.config, "head")

    command.check(migration_database.config)


def _insert_representative_inventory(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "INSERT INTO companies (id, name) VALUES (101, 'Legacy Company')"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO compounds "
                "(id, company_id, compound_key, name, location) VALUES "
                "(202, 101, 'legacy-compound', 'Legacy Compound', 'Cairo')"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO inventory_scopes "
                "(id, company_id, compound_id, scope_key, source_identifier) VALUES "
                "(303, 101, 202, 'legacy-scope', 'legacy-source')"
            )
        )


def _insert_credential(
    connection,
    credential_id: int,
    key_id: str,
    company_id: int,
) -> None:
    connection.execute(
        sa.text(
            "INSERT INTO api_credentials "
            "(id, key_id, company_id, secret_hash, name) VALUES "
            "(:id, :key_id, :company_id, :secret_hash, :name)"
        ),
        {
            "id": credential_id,
            "key_id": key_id,
            "company_id": company_id,
            "secret_hash": bytes([credential_id % 256]) * 32,
            "name": f"Credential {credential_id}",
        },
    )


def _alembic_config() -> Config:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    return config


def _current_revision(engine: Engine) -> str | None:
    with engine.connect() as connection:
        return connection.scalar(sa.text("SELECT version_num FROM alembic_version"))
