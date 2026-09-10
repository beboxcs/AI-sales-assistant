from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from app.database import engine as application_engine
from app.domain.enums import MembershipRole
from app.domain.users import normalize_email
from app.models import Company, CompanyMembership, User

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
PREVIOUS_REVISION = "d4e8f2a7c1b3"
MEMBERSHIP_REVISION = "b7c9d2e4f6a8"
PASSWORD_HASH = "$argon2id$v=19$m=65536,t=3,p=4$opaque-salt$opaque-digest"


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

    schema_name = f"user_membership_migration_{uuid4().hex}"
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


def test_upgrade_preserves_b1_data_and_creates_exact_schema(
    migration_database: MigrationDatabase,
) -> None:
    command.upgrade(migration_database.config, PREVIOUS_REVISION)
    with migration_database.engine.begin() as connection:
        connection.execute(
            sa.text("INSERT INTO companies (id, name) VALUES (101, 'Existing Co')")
        )
        connection.execute(
            sa.text(
                "INSERT INTO api_credentials "
                "(id, key_id, company_id, secret_hash, name) VALUES "
                "(201, :key_id, 101, :secret_hash, 'Existing key')"
            ),
            {"key_id": "K" * 22, "secret_hash": bytes(32)},
        )

    command.upgrade(migration_database.config, MEMBERSHIP_REVISION)

    inspector = sa.inspect(migration_database.engine)
    assert {"users", "company_memberships"}.issubset(inspector.get_table_names())

    user_columns = {
        column["name"]: column for column in inspector.get_columns("users")
    }
    assert list(user_columns) == [
        "id",
        "normalized_email",
        "password_hash",
        "is_active",
        "created_at",
        "updated_at",
        "password_changed_at",
    ]
    assert isinstance(user_columns["id"]["type"], sa.Integer)
    assert user_columns["normalized_email"]["type"].length == 320
    assert user_columns["password_hash"]["type"].length == 255
    assert isinstance(user_columns["is_active"]["type"], sa.Boolean)
    for column_name in ("created_at", "updated_at", "password_changed_at"):
        assert isinstance(user_columns[column_name]["type"], sa.DateTime)
        assert user_columns[column_name]["type"].timezone is True
    assert user_columns["password_changed_at"]["nullable"] is True
    for column_name in set(user_columns) - {"password_changed_at"}:
        assert user_columns[column_name]["nullable"] is False
    assert "true" in user_columns["is_active"]["default"].lower()
    assert "now()" in user_columns["created_at"]["default"].lower()
    assert "now()" in user_columns["updated_at"]["default"].lower()

    user_unique_constraints = {
        constraint["name"]: constraint
        for constraint in inspector.get_unique_constraints("users")
    }
    assert user_unique_constraints["uq_users_normalized_email"][
        "column_names"
    ] == ["normalized_email"]
    assert {
        constraint["name"]
        for constraint in inspector.get_check_constraints("users")
    } == {
        "ck_users_normalized_email_not_blank",
        "ck_users_password_hash_not_blank",
    }

    membership_columns = {
        column["name"]: column
        for column in inspector.get_columns("company_memberships")
    }
    assert list(membership_columns) == [
        "id",
        "user_id",
        "company_id",
        "role",
        "is_active",
        "created_at",
        "updated_at",
        "deactivated_at",
    ]
    assert membership_columns["role"]["type"].length == 32
    assert membership_columns["deactivated_at"]["nullable"] is True
    for column_name in set(membership_columns) - {"deactivated_at"}:
        assert membership_columns[column_name]["nullable"] is False
    assert "true" in membership_columns["is_active"]["default"].lower()

    membership_unique_constraints = {
        constraint["name"]: constraint
        for constraint in inspector.get_unique_constraints("company_memberships")
    }
    assert membership_unique_constraints["uq_company_memberships_user_company"][
        "column_names"
    ] == ["user_id", "company_id"]
    membership_indexes = {
        index["name"]: index
        for index in inspector.get_indexes("company_memberships")
    }
    assert membership_indexes["ix_company_memberships_company_id"][
        "column_names"
    ] == ["company_id"]
    assert {
        constraint["name"]
        for constraint in inspector.get_check_constraints("company_memberships")
    } == {
        "ck_company_memberships_role_allowed",
        "ck_company_memberships_deactivated_inactive",
    }

    foreign_keys = {
        foreign_key["name"]: foreign_key
        for foreign_key in inspector.get_foreign_keys("company_memberships")
    }
    assert foreign_keys["fk_company_memberships_user"]["constrained_columns"] == [
        "user_id"
    ]
    assert foreign_keys["fk_company_memberships_user"]["referred_table"] == "users"
    assert foreign_keys["fk_company_memberships_user"]["options"]["ondelete"] == (
        "RESTRICT"
    )
    assert foreign_keys["fk_company_memberships_company"][
        "constrained_columns"
    ] == ["company_id"]
    assert foreign_keys["fk_company_memberships_company"]["referred_table"] == (
        "companies"
    )
    assert foreign_keys["fk_company_memberships_company"]["options"][
        "ondelete"
    ] == "RESTRICT"

    with migration_database.engine.connect() as connection:
        assert connection.execute(
            sa.text("SELECT id, company_id FROM api_credentials")
        ).one() == (201, 101)
    assert _current_revision(migration_database.engine) == MEMBERSHIP_REVISION


def test_postgresql_user_constraints_and_defaults(
    migration_database: MigrationDatabase,
) -> None:
    command.upgrade(migration_database.config, MEMBERSHIP_REVISION)
    normalized_email = normalize_email("  Owner@EXAMPLE.COM\t")
    with migration_database.engine.begin() as connection:
        _insert_user(connection, 1, normalized_email)
        _insert_user(connection, 2, "inactive@example.com", is_active=False)

    with migration_database.engine.connect() as connection:
        rows = connection.execute(
            sa.text(
                "SELECT id, normalized_email, password_hash, is_active, "
                "created_at, updated_at, password_changed_at "
                "FROM users ORDER BY id"
            )
        ).all()
    assert rows[0][0:4] == (1, "owner@example.com", PASSWORD_HASH, True)
    assert rows[1][0:4] == (2, "inactive@example.com", PASSWORD_HASH, False)
    for row in rows:
        assert isinstance(row.created_at, datetime)
        assert row.created_at.tzinfo is not None
        assert isinstance(row.updated_at, datetime)
        assert row.updated_at.tzinfo is not None
        assert row.password_changed_at is None

    with pytest.raises(IntegrityError, match="uq_users_normalized_email"):
        with migration_database.engine.begin() as connection:
            _insert_user(connection, 3, normalized_email)

    for blank_email in ("", "   ", "\t", "\n", " \t\n "):
        with pytest.raises(
            IntegrityError,
            match="ck_users_normalized_email_not_blank",
        ):
            with migration_database.engine.begin() as connection:
                _insert_user(connection, 4, blank_email)

    with pytest.raises(IntegrityError, match="password_hash"):
        with migration_database.engine.begin() as connection:
            connection.execute(
                sa.text(
                    "INSERT INTO users (id, normalized_email, password_hash) "
                    "VALUES (5, 'missing-hash@example.com', NULL)"
                )
            )

    for blank_hash in ("", "   ", "\t", "\n", " \t\n "):
        with pytest.raises(
            IntegrityError,
            match="ck_users_password_hash_not_blank",
        ):
            with migration_database.engine.begin() as connection:
                _insert_user(
                    connection,
                    6,
                    "blank-hash@example.com",
                    password_hash=blank_hash,
                )


def test_orm_persists_canonical_email_and_logical_duplicates_collide(
    migration_database: MigrationDatabase,
) -> None:
    command.upgrade(migration_database.config, MEMBERSHIP_REVISION)
    with Session(migration_database.engine) as session:
        user = User(
            id=1,
            normalized_email="  User@Example.COM ",
            password_hash=PASSWORD_HASH,
        )
        session.add(user)
        session.commit()
        assert user.normalized_email == "user@example.com"

    with migration_database.engine.connect() as connection:
        assert connection.scalar(
            sa.text("SELECT normalized_email FROM users WHERE id = 1")
        ) == "user@example.com"

    with Session(migration_database.engine) as session:
        duplicate = User(
            id=2,
            normalized_email="\tUSER@example.com\n",
            password_hash=PASSWORD_HASH,
        )
        session.add(duplicate)
        with pytest.raises(IntegrityError, match="uq_users_normalized_email"):
            session.flush()


def test_postgresql_membership_constraints_and_relationships(
    migration_database: MigrationDatabase,
) -> None:
    command.upgrade(migration_database.config, MEMBERSHIP_REVISION)
    with migration_database.engine.begin() as connection:
        connection.execute(
            sa.text(
                "INSERT INTO companies (id, name) VALUES "
                "(10, 'Company Ten'), (20, 'Company Twenty')"
            )
        )
        _insert_user(connection, 1, "one@example.com")
        _insert_user(connection, 2, "two@example.com")
        _insert_membership(connection, 101, 1, 10, MembershipRole.SALESPERSON.value)
        _insert_membership(
            connection,
            102,
            2,
            10,
            MembershipRole.COMPANY_ADMIN.value,
        )
        _insert_membership(
            connection,
            103,
            1,
            20,
            MembershipRole.COMPANY_ADMIN.value,
        )
        _insert_membership(
            connection,
            104,
            2,
            20,
            MembershipRole.SALESPERSON.value,
            is_active=False,
            deactivated_at="2026-09-10T10:00:00+00:00",
        )

    with Session(migration_database.engine) as session:
        company_ten = session.get(Company, 10)
        user_one = session.get(User, 1)
        assert company_ten is not None
        assert user_one is not None
        assert {(item.user_id, item.role) for item in company_ten.memberships} == {
            (1, MembershipRole.SALESPERSON.value),
            (2, MembershipRole.COMPANY_ADMIN.value),
        }
        assert {(item.company_id, item.role) for item in user_one.memberships} == {
            (10, MembershipRole.SALESPERSON.value),
            (20, MembershipRole.COMPANY_ADMIN.value),
        }
        inactive = session.get(CompanyMembership, 104)
        assert inactive is not None
        assert inactive.is_active is False
        assert inactive.deactivated_at is not None

    invalid_memberships = [
        (
            "ck_company_memberships_role_allowed",
            (105, 1, 10, "owner", True, None),
        ),
        (
            "uq_company_memberships_user_company",
            (106, 1, 10, MembershipRole.COMPANY_ADMIN.value, True, None),
        ),
        (
            "fk_company_memberships_user",
            (107, 999, 10, MembershipRole.SALESPERSON.value, True, None),
        ),
        (
            "fk_company_memberships_company",
            (108, 1, 999, MembershipRole.SALESPERSON.value, True, None),
        ),
        (
            "ck_company_memberships_deactivated_inactive",
            (
                109,
                2,
                20,
                MembershipRole.SALESPERSON.value,
                True,
                "2026-09-10T10:00:00+00:00",
            ),
        ),
    ]
    for constraint_name, membership in invalid_memberships:
        with pytest.raises(IntegrityError, match=constraint_name):
            with migration_database.engine.begin() as connection:
                _insert_membership(connection, *membership)

    with pytest.raises(IntegrityError, match="fk_company_memberships_user"):
        with migration_database.engine.begin() as connection:
            connection.execute(sa.text("DELETE FROM users WHERE id = 1"))
    with pytest.raises(IntegrityError, match="fk_company_memberships_company"):
        with migration_database.engine.begin() as connection:
            connection.execute(sa.text("DELETE FROM companies WHERE id = 10"))


def test_downgrade_removes_only_b2_objects(
    migration_database: MigrationDatabase,
) -> None:
    command.upgrade(migration_database.config, MEMBERSHIP_REVISION)
    with migration_database.engine.begin() as connection:
        connection.execute(
            sa.text("INSERT INTO companies (id, name) VALUES (1, 'Preserved Co')")
        )
        connection.execute(
            sa.text(
                "INSERT INTO api_credentials "
                "(id, key_id, company_id, secret_hash, name) VALUES "
                "(2, :key_id, 1, :secret_hash, 'Preserved key')"
            ),
            {"key_id": "P" * 22, "secret_hash": bytes(32)},
        )
        _insert_user(connection, 3, "removed@example.com")
        _insert_membership(
            connection,
            4,
            3,
            1,
            MembershipRole.SALESPERSON.value,
        )

    command.downgrade(migration_database.config, PREVIOUS_REVISION)

    inspector = sa.inspect(migration_database.engine)
    assert "users" not in inspector.get_table_names()
    assert "company_memberships" not in inspector.get_table_names()
    with migration_database.engine.connect() as connection:
        assert connection.execute(
            sa.text("SELECT id, company_id FROM api_credentials")
        ).one() == (2, 1)
        assert connection.execute(
            sa.text("SELECT id, name FROM companies")
        ).one() == (1, "Preserved Co")
    assert _current_revision(migration_database.engine) == PREVIOUS_REVISION


def test_alembic_check_reports_no_model_drift(
    migration_database: MigrationDatabase,
) -> None:
    command.upgrade(migration_database.config, MEMBERSHIP_REVISION)

    command.check(migration_database.config)


def _insert_user(
    connection: Connection,
    user_id: int,
    normalized_email: str,
    *,
    password_hash: str | None = PASSWORD_HASH,
    is_active: bool = True,
) -> None:
    connection.execute(
        sa.text(
            "INSERT INTO users "
            "(id, normalized_email, password_hash, is_active) VALUES "
            "(:id, :normalized_email, :password_hash, :is_active)"
        ),
        {
            "id": user_id,
            "normalized_email": normalized_email,
            "password_hash": password_hash,
            "is_active": is_active,
        },
    )


def _insert_membership(
    connection: Connection,
    membership_id: int,
    user_id: int,
    company_id: int,
    role: str,
    is_active: bool = True,
    deactivated_at: str | None = None,
) -> None:
    connection.execute(
        sa.text(
            "INSERT INTO company_memberships "
            "(id, user_id, company_id, role, is_active, deactivated_at) VALUES "
            "(:id, :user_id, :company_id, :role, :is_active, :deactivated_at)"
        ),
        {
            "id": membership_id,
            "user_id": user_id,
            "company_id": company_id,
            "role": role,
            "is_active": is_active,
            "deactivated_at": deactivated_at,
        },
    )


def _alembic_config() -> Config:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    return config


def _current_revision(engine: Engine) -> str | None:
    with engine.connect() as connection:
        return connection.scalar(sa.text("SELECT version_num FROM alembic_version"))
