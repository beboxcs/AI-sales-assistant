from __future__ import annotations

import inspect
import threading
from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
import sqlalchemy as sa
from argon2.exceptions import InvalidHashError, VerificationError
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.auth.human.passwords import hash_password, verify_password
from app.bootstrap import (
    BootstrapDisposition,
    CompoundBootstrapSpec,
    InventoryScopeBootstrapSpec,
    LocalTenantBootstrapConflictError,
    LocalTenantBootstrapDisabledError,
    LocalTenantBootstrapInfrastructureError,
    LocalTenantBootstrapValidationError,
    bootstrap_local_tenant,
)
from app.bootstrap import local_tenant as bootstrap
from app.database import Base, engine
from app.domain.enums import MembershipRole
from app.models import Company, CompanyMembership, Compound, InventoryScope, User


@pytest.fixture
def bootstrap_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[Engine, None, None]:
    assert engine.dialect.name == "postgresql"
    schema_name = f"local_tenant_bootstrap_{uuid4().hex}"
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
            Compound.__table__,
            InventoryScope.__table__,
        ],
    )
    monkeypatch.setattr(
        bootstrap,
        "SessionLocal",
        sessionmaker(bind=isolated_engine),
    )
    try:
        yield isolated_engine
    finally:
        isolated_engine.dispose()
        with engine.begin() as connection:
            connection.execute(sa.text(f"DROP SCHEMA {quoted_schema_name} CASCADE"))


def _request(**overrides):
    request = {
        "company_id": 7001,
        "company_name": "Example Development",
        "admin_email": " Admin@Example.COM ",
        "admin_password": "correct horse battery staple",
        "compounds": (
            CompoundBootstrapSpec(
                compound_key="compound-a",
                name="Nile Gardens",
                location="New Cairo",
            ),
        ),
        "scopes": (
            InventoryScopeBootstrapSpec(
                scope_key="scope-a",
                source_identifier="source:scope-a",
                compound_key="compound-a",
            ),
        ),
    }
    request.update(overrides)
    return request


@pytest.mark.parametrize("company_id", [True, False, "1", 1.0, 0, -1, None])
def test_company_id_is_a_strict_positive_integer(company_id):
    with pytest.raises(LocalTenantBootstrapValidationError):
        bootstrap_local_tenant(**_request(company_id=company_id))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("company_name", "  "),
        ("company_name", 1),
        ("admin_email", "  "),
        ("admin_email", "x" * 321),
        ("admin_password", "  "),
        ("admin_password", None),
    ],
)
def test_scalar_input_validation(field, value):
    with pytest.raises(LocalTenantBootstrapValidationError):
        bootstrap_local_tenant(**_request(**{field: value}))


@pytest.mark.parametrize(
    "compounds",
    [
        (),
        [],
        (CompoundBootstrapSpec(" ", "Name", "Location"),),
        (CompoundBootstrapSpec("key", " ", "Location"),),
        (CompoundBootstrapSpec("key", "Name", " "),),
        (
            CompoundBootstrapSpec("same", "One", "A"),
            CompoundBootstrapSpec(" same ", "Two", "B"),
        ),
    ],
)
def test_compound_input_validation(compounds):
    with pytest.raises(LocalTenantBootstrapValidationError):
        bootstrap_local_tenant(**_request(compounds=compounds))


@pytest.mark.parametrize(
    "scopes",
    [
        (),
        [],
        (InventoryScopeBootstrapSpec(" ", "source"),),
        (InventoryScopeBootstrapSpec("scope", " "),),
        (InventoryScopeBootstrapSpec("scope", "source", " "),),
        (
            InventoryScopeBootstrapSpec("same", "source:one"),
            InventoryScopeBootstrapSpec(" same ", "source:two"),
        ),
    ],
)
def test_scope_input_validation(scopes):
    with pytest.raises(LocalTenantBootstrapValidationError):
        bootstrap_local_tenant(**_request(scopes=scopes))


def test_duplicate_keys_fail_before_database_access(monkeypatch):
    accessed = False

    class ForbiddenSessionLocal:
        @staticmethod
        def begin():
            nonlocal accessed
            accessed = True
            raise AssertionError("database must not be accessed")

    monkeypatch.setattr(bootstrap, "SessionLocal", ForbiddenSessionLocal())
    compounds = (
        CompoundBootstrapSpec("duplicate", "One", "A"),
        CompoundBootstrapSpec(" duplicate ", "Two", "B"),
    )
    with pytest.raises(LocalTenantBootstrapValidationError):
        bootstrap_local_tenant(**_request(compounds=compounds))
    assert accessed is False


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_environment_gate_precedes_validation_hashing_locking_and_database(
    monkeypatch,
    app_env,
):
    calls = []
    monkeypatch.setattr(bootstrap.settings, "APP_ENV", app_env)
    monkeypatch.setattr(bootstrap, "hash_password", lambda _value: calls.append("hash"))
    monkeypatch.setattr(
        bootstrap,
        "_acquire_bootstrap_locks",
        lambda *_args, **_kwargs: calls.append("lock"),
    )

    class ForbiddenSessionLocal:
        @staticmethod
        def begin():
            calls.append("database")
            raise AssertionError("database must not be accessed")

    monkeypatch.setattr(bootstrap, "SessionLocal", ForbiddenSessionLocal())
    with pytest.raises(LocalTenantBootstrapDisabledError):
        bootstrap_local_tenant(**_request(company_id=False))
    assert calls == []


def test_new_bootstrap_persists_admin_and_catalog_safely(bootstrap_engine):
    raw_password = _request()["admin_password"]
    result = bootstrap_local_tenant(**_request())

    assert result.company_id == 7001
    assert result.company_disposition is BootstrapDisposition.CREATED
    assert result.admin_user_disposition is BootstrapDisposition.CREATED
    assert result.admin_membership_disposition is BootstrapDisposition.CREATED
    assert result.normalized_admin_email == "admin@example.com"
    assert [item.compound_key for item in result.compounds] == ["compound-a"]
    assert [item.scope_key for item in result.scopes] == ["scope-a"]
    assert result.compounds[0].disposition is BootstrapDisposition.CREATED
    assert result.scopes[0].disposition is BootstrapDisposition.CREATED

    with Session(bootstrap_engine) as session:
        user = session.get(User, result.admin_user_id)
        membership = session.get(CompanyMembership, result.admin_membership_id)
        compound = session.get(Compound, result.compounds[0].compound_id)
        scope = session.get(InventoryScope, result.scopes[0].scope_id)
        assert user is not None
        assert user.password_hash.startswith("$argon2id$")
        assert verify_password(raw_password, user.password_hash) is True
        assert raw_password != user.password_hash
        assert user.password_version == 1
        assert user.is_active is True
        assert membership is not None
        assert membership.role == MembershipRole.COMPANY_ADMIN.value
        assert membership.is_active is True
        assert membership.deactivated_at is None
        assert compound is not None
        assert (compound.company_id, compound.name, compound.location) == (
            7001,
            "Nile Gardens",
            "New Cairo",
        )
        assert scope is not None
        assert scope.company_id == 7001
        assert scope.compound_id == compound.id

    assert raw_password not in repr(result)


def test_strings_are_canonicalized_results_are_ordered_and_password_repr_is_safe(
    bootstrap_engine,
):
    raw_password = _request()["admin_password"]
    request = _request(
        company_name="  Example Development  ",
        compounds=(
            CompoundBootstrapSpec(" compound-b ", " West Park ", " West Cairo "),
            CompoundBootstrapSpec(" compound-a ", " Nile Gardens ", " New Cairo "),
        ),
        scopes=(
            InventoryScopeBootstrapSpec(
                " scope-b ",
                " source:scope-b ",
                " compound-b ",
            ),
            InventoryScopeBootstrapSpec(
                " scope-a ",
                " source:scope-a ",
                " compound-a ",
            ),
        ),
    )

    validated = bootstrap._validate_request(**request)
    result = bootstrap_local_tenant(**request)

    assert raw_password not in repr(validated)
    assert [item.compound_key for item in result.compounds] == [
        "compound-a",
        "compound-b",
    ]
    assert [item.scope_key for item in result.scopes] == ["scope-a", "scope-b"]
    with Session(bootstrap_engine) as session:
        company = session.get(Company, 7001)
        compounds = list(session.scalars(sa.select(Compound).order_by(Compound.id)))
        scopes = list(
            session.scalars(sa.select(InventoryScope).order_by(InventoryScope.id))
        )
        assert company.name == "Example Development"
        assert {
            (item.compound_key, item.name, item.location) for item in compounds
        } == {
            ("compound-a", "Nile Gardens", "New Cairo"),
            ("compound-b", "West Park", "West Cairo"),
        }
        assert {
            (item.scope_key, item.source_identifier) for item in scopes
        } == {
            ("scope-a", "source:scope-a"),
            ("scope-b", "source:scope-b"),
        }


def test_identical_retry_reuses_everything_and_preserves_credentials(bootstrap_engine):
    first = bootstrap_local_tenant(**_request())
    with Session(bootstrap_engine) as session:
        user = session.get(User, first.admin_user_id)
        before = (
            user.password_hash,
            user.password_version,
            user.password_changed_at,
            user.updated_at,
        )

    second = bootstrap_local_tenant(**_request())

    assert second.company_id == first.company_id
    assert second.admin_user_id == first.admin_user_id
    assert second.admin_membership_id == first.admin_membership_id
    assert second.compounds[0].compound_id == first.compounds[0].compound_id
    assert second.scopes[0].scope_id == first.scopes[0].scope_id
    assert second.company_disposition is BootstrapDisposition.REUSED
    assert second.admin_user_disposition is BootstrapDisposition.REUSED
    assert second.admin_membership_disposition is BootstrapDisposition.REUSED
    assert second.compounds[0].disposition is BootstrapDisposition.REUSED
    assert second.scopes[0].disposition is BootstrapDisposition.REUSED
    with Session(bootstrap_engine) as session:
        user = session.get(User, first.admin_user_id)
        after = (
            user.password_hash,
            user.password_version,
            user.password_changed_at,
            user.updated_at,
        )
        assert after == before
        assert session.scalar(sa.select(sa.func.count()).select_from(Company)) == 1
        assert session.scalar(sa.select(sa.func.count()).select_from(User)) == 1
        assert session.scalar(
            sa.select(sa.func.count()).select_from(CompanyMembership)
        ) == 1


def test_existing_active_user_is_reused_only_with_matching_password(bootstrap_engine):
    password = _request()["admin_password"]
    original_hash = hash_password(password)
    changed_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
    with Session(bootstrap_engine) as session:
        user = User(
            normalized_email="admin@example.com",
            password_hash=original_hash,
            password_version=4,
            password_changed_at=changed_at,
        )
        session.add(user)
        session.commit()
        user_id = user.id

    result = bootstrap_local_tenant(**_request())

    assert result.admin_user_id == user_id
    assert result.admin_user_disposition is BootstrapDisposition.REUSED
    with Session(bootstrap_engine) as session:
        user = session.get(User, user_id)
        assert (user.password_hash, user.password_version, user.password_changed_at) == (
            original_hash,
            4,
            changed_at,
        )


@pytest.mark.parametrize("state", ["wrong-password", "inactive", "malformed-hash"])
def test_existing_user_conflicts_are_non_mutating(bootstrap_engine, state):
    password = _request()["admin_password"]
    persisted_password = "different password" if state == "wrong-password" else password
    password_hash = (
        "not-an-argon2-hash"
        if state == "malformed-hash"
        else hash_password(persisted_password)
    )
    with Session(bootstrap_engine) as session:
        user = User(
            normalized_email="admin@example.com",
            password_hash=password_hash,
            is_active=state != "inactive",
        )
        session.add(user)
        session.commit()
        user_id = user.id

    with pytest.raises(LocalTenantBootstrapConflictError) as exc:
        bootstrap_local_tenant(**_request())
    assert exc.value.resource == "admin user"
    with Session(bootstrap_engine) as session:
        user = session.get(User, user_id)
        assert user.password_hash == password_hash
        assert user.is_active is (state != "inactive")
        assert session.get(Company, 7001) is None


def test_existing_user_verifier_invalid_hash_is_a_sanitized_conflict(
    bootstrap_engine,
    monkeypatch,
):
    secret = _request()["admin_password"]
    persisted_hash = "malformed persisted hash"
    with Session(bootstrap_engine) as session:
        user = User(
            normalized_email="admin@example.com",
            password_hash=persisted_hash,
        )
        session.add(user)
        session.commit()

    monkeypatch.setattr(
        bootstrap,
        "verify_password",
        lambda _password, _password_hash: (_ for _ in ()).throw(
            InvalidHashError(f"invalid {secret}: {persisted_hash}")
        ),
    )

    with pytest.raises(LocalTenantBootstrapConflictError) as exc:
        bootstrap_local_tenant(**_request())
    assert exc.value.resource == "admin user"
    assert secret not in str(exc.value)
    assert persisted_hash not in str(exc.value)
    assert exc.value.__cause__ is None
    with Session(bootstrap_engine) as session:
        assert session.get(Company, 7001) is None


def test_existing_user_verifier_verification_error_is_a_sanitized_conflict(
    bootstrap_engine,
    monkeypatch,
):
    secret = _request()["admin_password"]
    persisted_hash = "unusable persisted hash"
    with Session(bootstrap_engine) as session:
        session.add(
            User(
                normalized_email="admin@example.com",
                password_hash=persisted_hash,
            )
        )
        session.commit()

    monkeypatch.setattr(
        bootstrap,
        "verify_password",
        lambda _password, _password_hash: (_ for _ in ()).throw(
            VerificationError(f"verification {secret}: {persisted_hash}")
        ),
    )

    with pytest.raises(LocalTenantBootstrapConflictError) as exc:
        bootstrap_local_tenant(**_request())
    assert exc.value.resource == "admin user"
    assert secret not in str(exc.value)
    assert persisted_hash not in str(exc.value)
    assert exc.value.__cause__ is None
    with Session(bootstrap_engine) as session:
        assert session.get(Company, 7001) is None


def test_existing_user_unrelated_verifier_error_is_infrastructure_error(
    bootstrap_engine,
    monkeypatch,
):
    password = _request()["admin_password"]
    with Session(bootstrap_engine) as session:
        session.add(
            User(
                normalized_email="admin@example.com",
                password_hash=hash_password(password),
            )
        )
        session.commit()

    monkeypatch.setattr(
        bootstrap,
        "verify_password",
        lambda _password, _password_hash: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(LocalTenantBootstrapInfrastructureError) as exc:
        bootstrap_local_tenant(**_request())
    assert str(exc.value) == "Local tenant bootstrap unavailable"
    assert exc.value.__cause__ is None


@pytest.mark.parametrize("state", ["salesperson", "inactive", "deactivated"])
def test_membership_role_and_state_conflicts_roll_back(bootstrap_engine, state):
    password = _request()["admin_password"]
    with Session(bootstrap_engine) as session:
        company = Company(id=7001, name="Example Development")
        user = User(
            normalized_email="admin@example.com",
            password_hash=hash_password(password),
        )
        session.add_all([company, user])
        session.flush()
        membership = CompanyMembership(
            user_id=user.id,
            company_id=company.id,
            role=(
                MembershipRole.SALESPERSON.value
                if state == "salesperson"
                else MembershipRole.COMPANY_ADMIN.value
            ),
            is_active=state not in {"inactive", "deactivated"},
            deactivated_at=(
                datetime(2026, 1, 2, tzinfo=timezone.utc)
                if state == "deactivated"
                else None
            ),
        )
        session.add(membership)
        session.commit()

    with pytest.raises(LocalTenantBootstrapConflictError) as exc:
        bootstrap_local_tenant(**_request())
    assert exc.value.resource == "admin membership"
    with Session(bootstrap_engine) as session:
        assert session.scalar(sa.select(sa.func.count()).select_from(Compound)) == 0


def test_populated_company_cannot_gain_a_different_initial_admin(bootstrap_engine):
    with Session(bootstrap_engine) as session:
        company = Company(id=7001, name="Example Development")
        existing_user = User(
            normalized_email="existing@example.com",
            password_hash=hash_password("existing password"),
        )
        session.add_all([company, existing_user])
        session.flush()
        session.add(
            CompanyMembership(
                user_id=existing_user.id,
                company_id=company.id,
                role=MembershipRole.SALESPERSON.value,
            )
        )
        session.commit()

    with pytest.raises(LocalTenantBootstrapConflictError) as exc:
        bootstrap_local_tenant(**_request())
    assert exc.value.resource == "initial admin membership"
    with Session(bootstrap_engine) as session:
        assert session.scalar(sa.select(sa.func.count()).select_from(User)) == 1
        assert session.scalar(sa.select(sa.func.count()).select_from(Compound)) == 0


@pytest.mark.parametrize("conflict", ["id", "name", "split"])
def test_company_identity_conflicts(bootstrap_engine, conflict):
    with Session(bootstrap_engine) as session:
        if conflict in {"id", "split"}:
            session.add(Company(id=7001, name="Different Name"))
        if conflict in {"name", "split"}:
            session.add(Company(id=7002, name="Example Development"))
        session.commit()

    with pytest.raises(LocalTenantBootstrapConflictError) as exc:
        bootstrap_local_tenant(**_request())
    assert exc.value.resource == "company"
    with Session(bootstrap_engine) as session:
        assert session.scalar(sa.select(sa.func.count()).select_from(User)) == 0


@pytest.mark.parametrize("field", ["name", "location"])
def test_compound_mismatch_rolls_back_without_reconciliation(bootstrap_engine, field):
    first = bootstrap_local_tenant(**_request())
    changed = CompoundBootstrapSpec(
        compound_key="compound-a",
        name="Different" if field == "name" else "Nile Gardens",
        location="Different" if field == "location" else "New Cairo",
    )
    with pytest.raises(LocalTenantBootstrapConflictError) as exc:
        bootstrap_local_tenant(**_request(compounds=(changed,)))
    assert exc.value.resource == "compound"
    with Session(bootstrap_engine) as session:
        compound = session.get(Compound, first.compounds[0].compound_id)
        assert (compound.name, compound.location) == ("Nile Gardens", "New Cairo")


@pytest.mark.parametrize("conflict", ["source", "association"])
def test_scope_mismatch_rolls_back_without_reassociation(bootstrap_engine, conflict):
    compounds = (
        CompoundBootstrapSpec("compound-a", "Nile Gardens", "New Cairo"),
        CompoundBootstrapSpec("compound-b", "West Park", "West Cairo"),
    )
    first = bootstrap_local_tenant(**_request(compounds=compounds))
    changed_scope = InventoryScopeBootstrapSpec(
        scope_key="scope-a",
        source_identifier=(
            "source:different" if conflict == "source" else "source:scope-a"
        ),
        compound_key="compound-b" if conflict == "association" else "compound-a",
    )
    with pytest.raises(LocalTenantBootstrapConflictError) as exc:
        bootstrap_local_tenant(
            **_request(compounds=compounds, scopes=(changed_scope,))
        )
    assert exc.value.resource == "inventory scope"
    with Session(bootstrap_engine) as session:
        scope = session.get(InventoryScope, first.scopes[0].scope_id)
        assert scope.source_identifier == "source:scope-a"
        assert scope.compound_id == first.compounds[0].compound_id


def test_scope_cannot_associate_to_other_tenant_compound(bootstrap_engine):
    with Session(bootstrap_engine) as session:
        other_company = Company(id=8001, name="Other Company")
        session.add(other_company)
        session.flush()
        session.add(
            Compound(
                company_id=other_company.id,
                compound_key="other-only",
                name="Other Compound",
                location="Other Location",
            )
        )
        session.commit()

    scopes = (
        InventoryScopeBootstrapSpec(
            "scope-a",
            "source:scope-a",
            "other-only",
        ),
    )
    with pytest.raises(LocalTenantBootstrapConflictError) as exc:
        bootstrap_local_tenant(**_request(scopes=scopes))
    assert exc.value.resource == "scope compound"
    with Session(bootstrap_engine) as session:
        assert session.get(Company, 7001) is None


def test_identical_catalog_keys_remain_tenant_isolated(bootstrap_engine):
    first = bootstrap_local_tenant(**_request())
    second = bootstrap_local_tenant(
        **_request(
            company_id=7002,
            company_name="Second Development",
            admin_email="second@example.com",
        )
    )
    assert first.compounds[0].compound_id != second.compounds[0].compound_id
    assert first.scopes[0].scope_id != second.scopes[0].scope_id
    with Session(bootstrap_engine) as session:
        rows = list(
            session.scalars(
                sa.select(Compound).order_by(Compound.company_id)
            )
        )
        assert [(row.company_id, row.compound_key) for row in rows] == [
            (7001, "compound-a"),
            (7002, "compound-a"),
        ]


def test_scope_may_be_unassociated(bootstrap_engine):
    scopes = (InventoryScopeBootstrapSpec("scope-a", "source:scope-a"),)
    result = bootstrap_local_tenant(**_request(scopes=scopes))
    assert result.scopes[0].compound_id is None
    with Session(bootstrap_engine) as session:
        assert session.get(InventoryScope, result.scopes[0].scope_id).compound_id is None


def test_failure_after_partial_flush_rolls_back_everything(
    bootstrap_engine,
    monkeypatch,
):
    def fail_scopes(*_args, **_kwargs):
        raise RuntimeError("database details must be sanitized")

    monkeypatch.setattr(bootstrap, "_resolve_scopes", fail_scopes)
    with pytest.raises(LocalTenantBootstrapInfrastructureError) as exc:
        bootstrap_local_tenant(**_request())
    assert str(exc.value) == "Local tenant bootstrap unavailable"
    with Session(bootstrap_engine) as session:
        for model in (Company, User, CompanyMembership, Compound, InventoryScope):
            assert session.scalar(sa.select(sa.func.count()).select_from(model)) == 0


def test_unexpected_hashing_error_is_sanitized_without_secret(
    bootstrap_engine,
    monkeypatch,
):
    secret = _request()["admin_password"]
    monkeypatch.setattr(
        bootstrap,
        "hash_password",
        lambda _password: (_ for _ in ()).throw(RuntimeError(f"failed {secret}")),
    )
    with pytest.raises(LocalTenantBootstrapInfrastructureError) as exc:
        bootstrap_local_tenant(**_request())
    assert str(exc.value) == "Local tenant bootstrap unavailable"
    assert secret not in str(exc.value)
    assert exc.value.__cause__ is None
    with Session(bootstrap_engine) as session:
        assert session.get(Company, 7001) is None
        assert session.scalar(sa.select(sa.func.count()).select_from(User)) == 0


def test_unrelated_rows_are_unchanged(bootstrap_engine):
    with Session(bootstrap_engine) as session:
        other_company = Company(id=9001, name="Unrelated")
        other_user = User(
            normalized_email="unrelated@example.com",
            password_hash=hash_password("unrelated password"),
        )
        session.add_all([other_company, other_user])
        session.flush()
        other_compound = Compound(
            company_id=other_company.id,
            compound_key="compound-a",
            name="Unrelated Compound",
            location="Alexandria",
        )
        session.add(other_compound)
        session.flush()
        other_scope = InventoryScope(
            company_id=other_company.id,
            compound_id=other_compound.id,
            scope_key="scope-a",
            source_identifier="unrelated-source",
        )
        session.add(other_scope)
        session.commit()
        before = (
            other_user.password_hash,
            other_compound.name,
            other_compound.location,
            other_scope.source_identifier,
        )

    bootstrap_local_tenant(**_request())
    with Session(bootstrap_engine) as session:
        user = session.scalar(
            sa.select(User).where(User.normalized_email == "unrelated@example.com")
        )
        compound = session.scalar(
            sa.select(Compound).where(Compound.company_id == 9001)
        )
        scope = session.scalar(
            sa.select(InventoryScope).where(InventoryScope.company_id == 9001)
        )
        assert (
            user.password_hash,
            compound.name,
            compound.location,
            scope.source_identifier,
        ) == before


def test_concurrent_identical_bootstraps_return_same_identities(bootstrap_engine):
    barrier = threading.Barrier(2)
    results = []
    errors = []

    def run():
        try:
            barrier.wait(timeout=10)
            results.append(bootstrap_local_tenant(**_request()))
        except Exception as exc:  # pragma: no cover - assertion reports details
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert all(thread.is_alive() is False for thread in threads)
    assert errors == []
    assert len(results) == 2
    assert {result.company_id for result in results} == {7001}
    assert len({result.admin_user_id for result in results}) == 1
    assert len({result.admin_membership_id for result in results}) == 1
    assert len({result.compounds[0].compound_id for result in results}) == 1
    assert len({result.scopes[0].scope_id for result in results}) == 1


def test_concurrent_conflicting_bootstraps_leave_one_coherent_tenant(
    bootstrap_engine,
):
    barrier = threading.Barrier(2)
    outcomes = []

    def run(name):
        compounds = (
            CompoundBootstrapSpec("compound-a", name, "New Cairo"),
        )
        try:
            barrier.wait(timeout=10)
            outcomes.append(bootstrap_local_tenant(**_request(compounds=compounds)))
        except Exception as exc:
            outcomes.append(exc)

    threads = [
        threading.Thread(target=run, args=("Nile Gardens",)),
        threading.Thread(target=run, args=("Different Gardens",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert all(thread.is_alive() is False for thread in threads)
    successes = [item for item in outcomes if not isinstance(item, Exception)]
    conflicts = [
        item
        for item in outcomes
        if isinstance(item, LocalTenantBootstrapConflictError)
    ]
    assert len(successes) == len(conflicts) == 1
    with Session(bootstrap_engine) as session:
        assert session.scalar(sa.select(sa.func.count()).select_from(Company)) == 1
        assert session.scalar(sa.select(sa.func.count()).select_from(User)) == 1
        assert session.scalar(
            sa.select(sa.func.count()).select_from(CompanyMembership)
        ) == 1
        assert session.scalar(sa.select(sa.func.count()).select_from(Compound)) == 1
        assert session.scalar(sa.select(sa.func.count()).select_from(InventoryScope)) == 1
        compound = session.scalar(sa.select(Compound))
        assert compound.name in {"Nile Gardens", "Different Gardens"}


def test_bootstrap_module_has_no_api_fastapi_or_ai_dependency():
    source = inspect.getsource(bootstrap)
    assert "fastapi" not in source.lower()
    assert "app.api" not in source
    assert "app.ai_engine" not in source
    path = Path(inspect.getsourcefile(bootstrap))
    assert path.as_posix().endswith("app/bootstrap/local_tenant.py")
