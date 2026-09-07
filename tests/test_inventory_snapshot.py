from collections.abc import Generator
from datetime import date
from decimal import Decimal
from inspect import signature

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

import app.persistence.inventory as inventory_persistence
from app.ai_engine.schemas import InventorySnapshot
from app.database import Base, engine as database_engine
from app.domain.enums import Currency, PaymentMethod, UnitStatus
from app.models import (
    Company,
    Compound,
    InventoryScope,
    PaymentPlan,
    PaymentPlanInventoryState,
    Unit,
    UnitInventoryState,
)
from app.persistence.inventory import (
    InventoryBackendError,
    InventoryScopeNotFoundError,
    InventoryPersistenceConflictError,
    InventoryPersistenceFailedError,
    InventorySnapshotLoadError,
    load_current_inventory_snapshot,
    persist_inventory_ingestion,
)


@pytest.fixture
def inventory_session(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[Session, None, None]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(
        engine,
        tables=[
            Company.__table__,
            Compound.__table__,
            InventoryScope.__table__,
            Unit.__table__,
            PaymentPlan.__table__,
            UnitInventoryState.__table__,
            PaymentPlanInventoryState.__table__,
        ],
    )

    with Session(engine) as session:
        monkeypatch.setattr(
            inventory_persistence,
            "SessionLocal",
            lambda: _SessionContext(session),
        )
        yield session

    engine.dispose()


@pytest.fixture
def transaction_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[object, list["_TrackingSession"]], None, None]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(
        engine,
        tables=[
            Company.__table__,
            Compound.__table__,
            InventoryScope.__table__,
            Unit.__table__,
            PaymentPlan.__table__,
            UnitInventoryState.__table__,
            PaymentPlanInventoryState.__table__,
        ],
    )
    sessions: list[_TrackingSession] = []
    factory = sessionmaker(
        bind=engine,
        class_=_TrackingSession,
        autoflush=False,
    )

    def create_session(**kwargs) -> _TrackingSession:
        session = factory(**kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(inventory_persistence, "database_engine", engine)
    monkeypatch.setattr(inventory_persistence, "SessionLocal", create_session)
    yield engine, sessions
    engine.dispose()


def test_public_snapshot_loader_keeps_session_internal():
    parameters = signature(load_current_inventory_snapshot).parameters

    assert tuple(parameters) == ("company_id", "scope_key")
    assert all(
        parameter.kind.name == "KEYWORD_ONLY"
        for parameter in parameters.values()
    )


def test_snapshot_reader_owns_and_closes_session_on_success_and_failure(
    transaction_backend,
    monkeypatch: pytest.MonkeyPatch,
):
    engine, successful_sessions = transaction_backend
    with Session(engine) as setup_session:
        setup_session.add(Company(id=1, name="Snapshot Company"))
        setup_session.add(
            InventoryScope(
                id=101,
                company_id=1,
                compound_id=None,
                scope_key="scope:lifecycle",
                source_identifier="source:lifecycle",
            )
        )
        setup_session.commit()

    parameters = signature(load_current_inventory_snapshot).parameters
    snapshot = load_current_inventory_snapshot(
        company_id=1,
        scope_key="scope:lifecycle",
    )

    assert tuple(parameters) == ("company_id", "scope_key")
    assert snapshot == InventorySnapshot(scope_key="scope:lifecycle", units=[])
    assert len(successful_sessions) == 1
    assert successful_sessions[0].was_closed is True

    failed_sessions: list[_QueryFailingTrackingSession] = []
    failed_factory = sessionmaker(
        bind=engine,
        class_=_QueryFailingTrackingSession,
        autoflush=False,
    )

    def create_failing_session(**kwargs) -> _QueryFailingTrackingSession:
        session = failed_factory(**kwargs)
        failed_sessions.append(session)
        return session

    monkeypatch.setattr(
        inventory_persistence,
        "SessionLocal",
        create_failing_session,
    )

    with pytest.raises(InventorySnapshotLoadError):
        load_current_inventory_snapshot(
            company_id=1,
            scope_key="scope:lifecycle",
        )

    assert len(failed_sessions) == 1
    assert failed_sessions[0].was_closed is True


def test_public_persistence_boundary_keeps_session_internal():
    parameters = signature(persist_inventory_ingestion).parameters

    assert tuple(parameters) == ("company_id", "snapshot", "changes")
    assert all(
        parameter.kind.name == "KEYWORD_ONLY"
        for parameter in parameters.values()
    )


def test_public_inventory_errors_share_backend_base():
    assert issubclass(InventoryScopeNotFoundError, InventoryBackendError)
    assert issubclass(InventorySnapshotLoadError, InventoryBackendError)
    assert issubclass(InventoryPersistenceConflictError, InventoryBackendError)
    assert issubclass(InventoryPersistenceFailedError, InventoryBackendError)


def test_reconstructs_only_current_units_and_plans_for_exact_scope(
    inventory_session: Session,
):
    company, compound, target_scope = _add_tenant_inventory(
        inventory_session,
        company_id=1,
        compound_id=11,
        target_scope_id=101,
        other_scope_id=102,
    )
    current_unit = _add_unit(
        inventory_session,
        company_id=company.id,
        compound_id=compound.id,
        unit_id=1001,
        external_unit_id="A101",
        property_type="Duplex",
    )
    non_current_unit = _add_unit(
        inventory_session,
        company_id=company.id,
        compound_id=compound.id,
        unit_id=1002,
        external_unit_id="A102",
    )
    _add_unit_state(
        inventory_session,
        company_id=company.id,
        unit=current_unit,
        scope_id=target_scope.id,
        present=True,
    )
    _add_unit_state(
        inventory_session,
        company_id=company.id,
        unit=non_current_unit,
        scope_id=target_scope.id,
        present=False,
    )
    _add_unit_state(
        inventory_session,
        company_id=company.id,
        unit=current_unit,
        scope_id=102,
        present=True,
    )
    current_plan = _add_plan(
        inventory_session,
        company_id=company.id,
        unit=current_unit,
        plan_id=2001,
        plan_key="plan:current",
    )
    non_current_plan = _add_plan(
        inventory_session,
        company_id=company.id,
        unit=current_unit,
        plan_id=2002,
        plan_key="plan:missing",
    )
    other_scope_plan = _add_plan(
        inventory_session,
        company_id=company.id,
        unit=current_unit,
        plan_id=2003,
        plan_key="plan:other-scope",
    )
    _add_plan_state(
        inventory_session,
        company_id=company.id,
        plan=current_plan,
        scope_id=target_scope.id,
        present=True,
    )
    _add_plan_state(
        inventory_session,
        company_id=company.id,
        plan=non_current_plan,
        scope_id=target_scope.id,
        present=False,
    )
    _add_plan_state(
        inventory_session,
        company_id=company.id,
        plan=other_scope_plan,
        scope_id=102,
        present=True,
    )
    inventory_session.flush()

    snapshot = load_current_inventory_snapshot(
        company_id=company.id,
        scope_key=target_scope.scope_key,
    )

    assert isinstance(snapshot, InventorySnapshot)
    assert snapshot.scope_key == "scope:shared"
    assert len(snapshot.units) == 1
    unit = snapshot.units[0]
    assert unit.external_unit_id == "A101"
    assert unit.compound_key == "compound:shared"
    assert unit.compound_name == "Compound 1"
    assert unit.location == "New Cairo"
    assert unit.property_type == "Duplex"
    assert unit.base_price == Decimal("5500000.00")
    assert unit.base_currency == Currency.EGP
    assert unit.status == UnitStatus.AVAILABLE
    assert unit.source == "source:scope:shared"
    assert unit.last_updated == date(2026, 9, 5)
    assert [plan.plan_key for plan in unit.payment_plans] == ["plan:current"]
    assert unit.payment_plans[0].payment_method == PaymentMethod.INSTALLMENT


def test_unknown_scope_fails_closed(inventory_session: Session):
    inventory_session.add(Company(id=1, name="Company 1"))
    inventory_session.flush()

    with pytest.raises(InventoryScopeNotFoundError) as exc_info:
        load_current_inventory_snapshot(
            company_id=1,
            scope_key="scope:unknown",
        )

    assert str(exc_info.value) == "Inventory scope was not found"


def test_cross_tenant_scope_fails_closed(inventory_session: Session):
    inventory_session.add(Company(id=1, name="Company 1"))
    _add_tenant_inventory(
        inventory_session,
        company_id=2,
        compound_id=22,
        target_scope_id=201,
        other_scope_id=202,
    )
    inventory_session.flush()

    with pytest.raises(InventoryScopeNotFoundError) as exc_info:
        load_current_inventory_snapshot(
            company_id=1,
            scope_key="scope:shared",
        )

    assert str(exc_info.value) == "Inventory scope was not found"


def test_same_natural_identities_remain_isolated_between_tenants(
    inventory_session: Session,
):
    company_a, compound_a, scope_a = _add_tenant_inventory(
        inventory_session,
        company_id=1,
        compound_id=11,
        target_scope_id=101,
        other_scope_id=102,
    )
    company_b, compound_b, scope_b = _add_tenant_inventory(
        inventory_session,
        company_id=2,
        compound_id=22,
        target_scope_id=201,
        other_scope_id=202,
    )
    unit_a = _add_unit(
        inventory_session,
        company_id=company_a.id,
        compound_id=compound_a.id,
        unit_id=1001,
        external_unit_id="A101",
        property_type="Tenant A Unit",
    )
    unit_b = _add_unit(
        inventory_session,
        company_id=company_b.id,
        compound_id=compound_b.id,
        unit_id=2001,
        external_unit_id="A101",
        property_type="Tenant B Unit",
    )
    _add_unit_state(
        inventory_session,
        company_id=company_a.id,
        unit=unit_a,
        scope_id=scope_a.id,
        present=True,
    )
    _add_unit_state(
        inventory_session,
        company_id=company_b.id,
        unit=unit_b,
        scope_id=scope_b.id,
        present=True,
    )
    plan_a = _add_plan(
        inventory_session,
        company_id=company_a.id,
        unit=unit_a,
        plan_id=3001,
        plan_key="plan:shared",
        plan_name="Tenant A Plan",
    )
    plan_b = _add_plan(
        inventory_session,
        company_id=company_b.id,
        unit=unit_b,
        plan_id=4001,
        plan_key="plan:shared",
        plan_name="Tenant B Plan",
    )
    _add_plan_state(
        inventory_session,
        company_id=company_a.id,
        plan=plan_a,
        scope_id=scope_a.id,
        present=True,
    )
    _add_plan_state(
        inventory_session,
        company_id=company_b.id,
        plan=plan_b,
        scope_id=scope_b.id,
        present=True,
    )
    inventory_session.flush()

    snapshot = load_current_inventory_snapshot(
        company_id=company_a.id,
        scope_key=scope_a.scope_key,
    )

    assert [unit.property_type for unit in snapshot.units] == ["Tenant A Unit"]
    assert [
        plan.plan_name
        for unit in snapshot.units
        for plan in unit.payment_plans
    ] == ["Tenant A Plan"]


def test_snapshot_units_and_plans_have_deterministic_natural_key_order(
    inventory_session: Session,
):
    company, compound, scope = _add_tenant_inventory(
        inventory_session,
        company_id=1,
        compound_id=11,
        target_scope_id=101,
        other_scope_id=102,
    )
    unit_b = _add_unit(
        inventory_session,
        company_id=company.id,
        compound_id=compound.id,
        unit_id=1001,
        external_unit_id="B101",
    )
    unit_a = _add_unit(
        inventory_session,
        company_id=company.id,
        compound_id=compound.id,
        unit_id=1002,
        external_unit_id="A101",
    )
    for unit in (unit_b, unit_a):
        _add_unit_state(
            inventory_session,
            company_id=company.id,
            unit=unit,
            scope_id=scope.id,
            present=True,
        )
    plan_z = _add_plan(
        inventory_session,
        company_id=company.id,
        unit=unit_a,
        plan_id=2001,
        plan_key="plan:z",
    )
    plan_a = _add_plan(
        inventory_session,
        company_id=company.id,
        unit=unit_a,
        plan_id=2002,
        plan_key="plan:a",
    )
    for plan in (plan_z, plan_a):
        _add_plan_state(
            inventory_session,
            company_id=company.id,
            plan=plan,
            scope_id=scope.id,
            present=True,
        )
    inventory_session.flush()

    snapshot = load_current_inventory_snapshot(
        company_id=company.id,
        scope_key=scope.scope_key,
    )

    assert [unit.external_unit_id for unit in snapshot.units] == ["A101", "B101"]
    assert [
        plan.plan_key for plan in snapshot.units[0].payment_plans
    ] == ["plan:a", "plan:z"]


def test_query_failure_maps_to_safe_backend_exception(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        inventory_persistence,
        "SessionLocal",
        lambda: _QueryFailingSession(),
    )

    with pytest.raises(InventorySnapshotLoadError) as exc_info:
        load_current_inventory_snapshot(
            company_id=1,
            scope_key="scope:shared",
        )

    assert str(exc_info.value) == "Inventory snapshot load failed"
    assert "database details" not in str(exc_info.value)


def test_successful_persistence_commits_and_calls_low_level_once(
    transaction_backend,
    monkeypatch: pytest.MonkeyPatch,
):
    engine, sessions = transaction_backend
    calls: list[tuple[int, InventorySnapshot, list[object]]] = []
    snapshot = InventorySnapshot(scope_key="scope:test", units=[])

    def persist_once(session, *, company_id, snapshot, changes):
        calls.append((company_id, snapshot, changes))
        session.add(Company(id=1, name="Committed Company"))
        session.commit()

    monkeypatch.setattr(
        inventory_persistence,
        "persist_inventory_snapshot",
        persist_once,
    )

    persist_inventory_ingestion(
        company_id=1,
        snapshot=snapshot,
        changes=[],
    )

    with Session(engine) as verification_session:
        company = verification_session.scalar(
            select(Company).where(Company.id == 1)
        )
    assert company is not None and company.name == "Committed Company"
    assert calls == [(1, snapshot, [])]
    assert len(sessions) == 1 and sessions[0].was_closed is True


def test_inner_session_commit_cannot_publish_before_wrapper_success(
    monkeypatch: pytest.MonkeyPatch,
):
    assert database_engine.dialect.name == "postgresql"
    company_id = -2_100_000_123
    company_name = "Atomic Inner Commit Regression Company"
    with Session(database_engine) as verification_session:
        assert verification_session.get(Company, company_id) is None

    def commit_then_fail(session, *, company_id, snapshot, changes):
        session.add(Company(id=company_id, name=company_name))
        session.commit()
        raise RuntimeError("failure after inner commit")

    monkeypatch.setattr(
        inventory_persistence,
        "persist_inventory_snapshot",
        commit_then_fail,
    )

    try:
        with pytest.raises(InventoryPersistenceFailedError) as exc_info:
            persist_inventory_ingestion(
                company_id=company_id,
                snapshot=InventorySnapshot(scope_key="scope:atomic", units=[]),
                changes=[],
            )

        with Session(database_engine) as verification_session:
            published_company = verification_session.get(Company, company_id)
        assert str(exc_info.value) == "Inventory persistence failed"
        assert published_company is None
    finally:
        with Session(database_engine) as cleanup_session:
            leaked_company = cleanup_session.get(Company, company_id)
            if leaked_company is not None:
                cleanup_session.delete(leaked_company)
                cleanup_session.commit()


def test_persistence_failure_rolls_back_and_maps_to_safe_error(
    transaction_backend,
    monkeypatch: pytest.MonkeyPatch,
):
    engine, sessions = transaction_backend

    def fail_after_flush(session, *, company_id, snapshot, changes):
        session.add(Company(id=1, name="Must Roll Back"))
        session.flush()
        session.rollback()
        raise InventoryPersistenceFailedError("internal persistence details")

    monkeypatch.setattr(
        inventory_persistence,
        "persist_inventory_snapshot",
        fail_after_flush,
    )

    with pytest.raises(InventoryPersistenceFailedError) as exc_info:
        persist_inventory_ingestion(
            company_id=1,
            snapshot=InventorySnapshot(scope_key="scope:test", units=[]),
            changes=[],
        )

    with Session(engine) as verification_session:
        assert verification_session.scalar(select(Company)) is None
    assert str(exc_info.value) == "Inventory persistence failed"
    assert "internal persistence details" not in str(exc_info.value)
    assert len(sessions) == 1 and sessions[0].was_closed is True


def test_wrapped_integrity_failure_maps_to_persistence_conflict(
    transaction_backend,
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_with_integrity_error(session, *, company_id, snapshot, changes):
        try:
            raise IntegrityError(
                "insert company",
                {},
                RuntimeError("constraint details"),
            )
        except IntegrityError as exc:
            raise InventoryPersistenceFailedError(
                "internal persistence details"
            ) from exc

    monkeypatch.setattr(
        inventory_persistence,
        "persist_inventory_snapshot",
        fail_with_integrity_error,
    )

    with pytest.raises(InventoryPersistenceConflictError) as exc_info:
        persist_inventory_ingestion(
            company_id=1,
            snapshot=InventorySnapshot(scope_key="scope:test", units=[]),
            changes=[],
        )

    assert str(exc_info.value) == "Inventory persistence conflict"
    assert "constraint details" not in str(exc_info.value)
    assert not isinstance(exc_info.value, IntegrityError)


def test_direct_integrity_failure_maps_to_persistence_conflict(
    transaction_backend,
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_with_integrity_error(session, *, company_id, snapshot, changes):
        raise IntegrityError(
            "insert company",
            {},
            RuntimeError("constraint details"),
        )

    monkeypatch.setattr(
        inventory_persistence,
        "persist_inventory_snapshot",
        fail_with_integrity_error,
    )

    with pytest.raises(InventoryPersistenceConflictError) as exc_info:
        persist_inventory_ingestion(
            company_id=1,
            snapshot=InventorySnapshot(scope_key="scope:test", units=[]),
            changes=[],
        )

    assert str(exc_info.value) == "Inventory persistence conflict"
    assert "constraint details" not in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, IntegrityError)


def test_raw_database_failure_maps_to_persistence_failed_error(
    transaction_backend,
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_with_database_error(session, *, company_id, snapshot, changes):
        raise OperationalError(
            "select inventory",
            {},
            RuntimeError("database details"),
        )

    monkeypatch.setattr(
        inventory_persistence,
        "persist_inventory_snapshot",
        fail_with_database_error,
    )

    with pytest.raises(InventoryPersistenceFailedError) as exc_info:
        persist_inventory_ingestion(
            company_id=1,
            snapshot=InventorySnapshot(scope_key="scope:test", units=[]),
            changes=[],
        )

    assert str(exc_info.value) == "Inventory persistence failed"
    assert "database details" not in str(exc_info.value)
    assert not isinstance(exc_info.value, OperationalError)


class _TrackingSession(Session):
    was_closed = False

    def close(self) -> None:
        self.was_closed = True
        super().close()


class _QueryFailingTrackingSession(_TrackingSession):
    def scalar(self, statement, *args, **kwargs):
        raise RuntimeError("database details")


class _SessionContext:
    def __init__(self, session: Session):
        self.session = session

    def __enter__(self) -> Session:
        return self.session

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False


class _QueryFailingSession:
    def __enter__(self) -> "_QueryFailingSession":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def scalar(self, statement):
        raise RuntimeError("database details")


def _add_tenant_inventory(
    session: Session,
    *,
    company_id: int,
    compound_id: int,
    target_scope_id: int,
    other_scope_id: int,
) -> tuple[Company, Compound, InventoryScope]:
    company = Company(id=company_id, name=f"Company {company_id}")
    compound = Compound(
        id=compound_id,
        company_id=company_id,
        compound_key="compound:shared",
        name=f"Compound {company_id}",
        location="New Cairo",
    )
    target_scope = InventoryScope(
        id=target_scope_id,
        company_id=company_id,
        compound_id=None,
        scope_key="scope:shared",
        source_identifier="source:scope:shared",
    )
    other_scope = InventoryScope(
        id=other_scope_id,
        company_id=company_id,
        compound_id=None,
        scope_key="scope:other",
        source_identifier="source:scope:other",
    )
    session.add_all([company, compound, target_scope, other_scope])
    session.flush()
    return company, compound, target_scope


def _add_unit(
    session: Session,
    *,
    company_id: int,
    compound_id: int,
    unit_id: int,
    external_unit_id: str,
    property_type: str = "Apartment",
) -> Unit:
    unit = Unit(
        id=unit_id,
        company_id=company_id,
        compound_id=compound_id,
        external_unit_id=external_unit_id,
        property_type=property_type,
        base_price=Decimal("5500000.00"),
        base_currency=Currency.EGP.value,
        status=UnitStatus.AVAILABLE.value,
        bedrooms=3,
        bathrooms=2,
        area_sqm=Decimal("150.00"),
        delivery_date=date(2028, 6, 30),
        source_last_updated=date(2026, 9, 5),
    )
    session.add(unit)
    session.flush()
    return unit


def _add_plan(
    session: Session,
    *,
    company_id: int,
    unit: Unit,
    plan_id: int,
    plan_key: str,
    plan_name: str = "7 Years",
) -> PaymentPlan:
    plan = PaymentPlan(
        id=plan_id,
        company_id=company_id,
        unit_id=unit.id,
        plan_key=plan_key,
        source_plan_id=f"source:{plan_key}",
        plan_name=plan_name,
        payment_method=PaymentMethod.INSTALLMENT.value,
        down_payment=Decimal("500000.00"),
        installment_years=Decimal("7.00"),
        total_price=Decimal("6000000.00"),
        currency=Currency.EGP.value,
    )
    session.add(plan)
    session.flush()
    return plan


def _add_unit_state(
    session: Session,
    *,
    company_id: int,
    unit: Unit,
    scope_id: int,
    present: bool,
) -> None:
    session.add(
        UnitInventoryState(
            company_id=company_id,
            unit_id=unit.id,
            inventory_scope_id=scope_id,
            is_present_in_latest_inventory=present,
        )
    )


def _add_plan_state(
    session: Session,
    *,
    company_id: int,
    plan: PaymentPlan,
    scope_id: int,
    present: bool,
) -> None:
    session.add(
        PaymentPlanInventoryState(
            company_id=company_id,
            payment_plan_id=plan.id,
            inventory_scope_id=scope_id,
            is_present_in_latest_inventory=present,
        )
    )
