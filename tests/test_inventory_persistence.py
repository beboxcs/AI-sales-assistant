from collections.abc import Generator
from datetime import date
from decimal import Decimal
from typing import NamedTuple
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.ai_engine.change_detector import detect_inventory_changes
from app.ai_engine.schemas import InventorySnapshot, PaymentPlanSchema, UnitSchema
from app.database import Base, engine as database_engine
from app.domain.enums import Currency, PaymentMethod, UnitStatus
from app.models import (
    Company,
    Compound,
    InventoryScope,
    PaymentPlan,
    PaymentPlanInventoryState,
    Unit,
    UnitHistory,
    UnitInventoryState,
)
from app.persistence.inventory import (
    CompoundDescriptionMismatchError,
    CompoundNotFoundError,
    InventoryScopeNotFoundError,
    InventoryUnitNotFoundError,
    get_compound,
    get_inventory_scope,
    get_unit_by_identity,
    persist_inventory_snapshot,
)


class PersistenceContext(NamedTuple):
    company: Company
    inventory_scope: InventoryScope
    compound: Compound


@pytest.fixture
def inventory_session() -> Generator[Session, None, None]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(
        engine,
        tables=[Company.__table__, Compound.__table__, InventoryScope.__table__],
    )

    with Session(engine) as session:
        yield session

    engine.dispose()


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    assert database_engine.dialect.name == "postgresql"

    connection = database_engine.connect()
    outer_transaction = connection.begin()
    session = Session(
        bind=connection,
        autoflush=False,
        join_transaction_mode="create_savepoint",
    )

    try:
        yield session
    finally:
        session.close()
        if outer_transaction.is_active:
            outer_transaction.rollback()
        connection.close()


def test_get_inventory_scope_returns_scope_for_correct_company(
    inventory_session: Session,
):
    expected_scope = _add_scope(
        inventory_session,
        company_id=1,
        scope_id=101,
        scope_key="feed:primary",
    )

    actual_scope = get_inventory_scope(
        inventory_session,
        company_id=1,
        scope_key="feed:primary",
    )

    assert actual_scope is expected_scope


def test_get_inventory_scope_rejects_nonexistent_scope(
    inventory_session: Session,
):
    inventory_session.add(Company(id=1, name="Company 1"))
    inventory_session.flush()

    with pytest.raises(InventoryScopeNotFoundError):
        get_inventory_scope(
            inventory_session,
            company_id=1,
            scope_key="feed:missing",
        )


def test_get_inventory_scope_rejects_matching_scope_from_another_company(
    inventory_session: Session,
):
    inventory_session.add(Company(id=1, name="Company 1"))
    _add_scope(
        inventory_session,
        company_id=2,
        scope_id=202,
        scope_key="feed:shared",
    )

    with pytest.raises(InventoryScopeNotFoundError):
        get_inventory_scope(
            inventory_session,
            company_id=1,
            scope_key="feed:shared",
        )


def test_get_compound_resolves_by_company_and_compound_key(
    db_session: Session,
):
    tenant_a = _seed_context(db_session, compound_key="compound:shared")
    tenant_b = _seed_context(db_session, compound_key="compound:shared")

    resolved_a = get_compound(
        db_session,
        company_id=tenant_a.company.id,
        compound_key="compound:shared",
    )
    resolved_b = get_compound(
        db_session,
        company_id=tenant_b.company.id,
        compound_key="compound:shared",
    )

    assert resolved_a is tenant_a.compound
    assert resolved_b is tenant_b.compound


def test_get_compound_rejects_missing_or_cross_tenant_key(
    db_session: Session,
):
    tenant_a = _seed_context(db_session, compound_key="compound:a")
    tenant_b = _seed_context(db_session, compound_key="compound:b")

    with pytest.raises(CompoundNotFoundError):
        get_compound(
            db_session,
            company_id=tenant_a.company.id,
            compound_key=tenant_b.compound.compound_key,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"compound_name": "Conflicting Name"},
        {"location": "Conflicting Location"},
    ],
)
def test_persistence_rejects_compound_description_conflicts_atomically(
    db_session: Session,
    overrides: dict[str, str],
):
    context = _seed_context(db_session)
    db_session.commit()
    unit_data = _unit_schema(context.compound, **overrides)

    with pytest.raises(CompoundDescriptionMismatchError):
        persist_inventory_snapshot(
            db_session,
            company_id=context.company.id,
            snapshot=_snapshot(context.inventory_scope, [unit_data]),
            changes=[],
        )

    assert db_session.scalar(
        select(func.count())
        .select_from(Unit)
        .where(Unit.company_id == context.company.id)
    ) == 0


def test_persistence_inserts_unit_plan_presence_history_and_decimals(
    db_session: Session,
):
    context = _seed_context(db_session)
    plan_data = _plan_schema(
        down_payment=Decimal("550000.25"),
        installment_years=Decimal("7.50"),
        total_price=None,
    )
    unit_data = _unit_schema(
        context.compound,
        base_price=Decimal("5800000.75"),
        area_sqm=Decimal("145.25"),
        payment_plans=[plan_data],
        source="transient.csv",
    )
    snapshot = _snapshot(context.inventory_scope, [unit_data])
    changes = detect_inventory_changes(
        _snapshot(context.inventory_scope, []),
        snapshot,
    )

    persist_inventory_snapshot(
        db_session,
        company_id=context.company.id,
        snapshot=snapshot,
        changes=changes,
    )

    unit = get_unit_by_identity(
        db_session,
        company_id=context.company.id,
        compound_id=context.compound.id,
        external_unit_id=unit_data.external_unit_id,
    )
    assert unit is not None
    payment_plan = db_session.scalar(
        select(PaymentPlan).where(
            PaymentPlan.company_id == context.company.id,
            PaymentPlan.unit_id == unit.id,
            PaymentPlan.plan_key == plan_data.plan_key,
        )
    )
    unit_state = db_session.scalar(
        select(UnitInventoryState).where(
            UnitInventoryState.company_id == context.company.id,
            UnitInventoryState.unit_id == unit.id,
            UnitInventoryState.inventory_scope_id == context.inventory_scope.id,
        )
    )
    assert payment_plan is not None
    plan_state = db_session.scalar(
        select(PaymentPlanInventoryState).where(
            PaymentPlanInventoryState.company_id == context.company.id,
            PaymentPlanInventoryState.payment_plan_id == payment_plan.id,
            PaymentPlanInventoryState.inventory_scope_id
            == context.inventory_scope.id,
        )
    )
    history = db_session.scalar(
        select(UnitHistory).where(
            UnitHistory.company_id == context.company.id,
            UnitHistory.unit_id == unit.id,
            UnitHistory.inventory_scope_id == context.inventory_scope.id,
        )
    )

    assert unit.compound_id == context.compound.id
    assert unit.base_price == Decimal("5800000.75")
    assert unit.area_sqm == Decimal("145.25")
    assert unit.source_last_updated == date(2026, 9, 5)
    assert not hasattr(unit, "source")
    assert payment_plan.down_payment == Decimal("550000.25")
    assert payment_plan.installment_years == Decimal("7.50")
    assert payment_plan.total_price is None
    assert unit_state is not None
    assert unit_state.is_present_in_latest_inventory is True
    assert plan_state is not None
    assert plan_state.is_present_in_latest_inventory is True
    assert history is not None
    assert history.change_type == "added"
    assert history.changes == {}
    assert history.payment_plan_changes == []
    assert unit_state.last_seen_at == plan_state.last_seen_at == history.detected_at
    assert history.detected_at.tzinfo is not None


def test_persistence_updates_unit_and_plan_without_duplicates(
    db_session: Session,
):
    context = _seed_context(db_session)
    previous = _snapshot(
        context.inventory_scope,
        [_unit_schema(context.compound, payment_plans=[_plan_schema()])],
    )
    persist_inventory_snapshot(
        db_session,
        company_id=context.company.id,
        snapshot=previous,
        changes=detect_inventory_changes(
            _snapshot(context.inventory_scope, []),
            previous,
        ),
    )
    current = _snapshot(
        context.inventory_scope,
        [
            _unit_schema(
                context.compound,
                property_type="Duplex",
                base_price=Decimal("6100000.00"),
                status=UnitStatus.HOLD,
                payment_plans=[
                    _plan_schema(
                        plan_name="Updated Plan",
                        total_price=Decimal("6400000.00"),
                    )
                ],
            )
        ],
    )

    persist_inventory_snapshot(
        db_session,
        company_id=context.company.id,
        snapshot=current,
        changes=detect_inventory_changes(previous, current),
    )

    unit = get_unit_by_identity(
        db_session,
        company_id=context.company.id,
        compound_id=context.compound.id,
        external_unit_id="A101",
    )
    assert unit is not None
    payment_plan = db_session.scalar(
        select(PaymentPlan).where(
            PaymentPlan.company_id == context.company.id,
            PaymentPlan.unit_id == unit.id,
            PaymentPlan.plan_key == "plan:installment",
        )
    )
    assert db_session.scalar(
        select(func.count())
        .select_from(Unit)
        .where(Unit.company_id == context.company.id)
    ) == 1
    assert db_session.scalar(
        select(func.count())
        .select_from(PaymentPlan)
        .where(PaymentPlan.company_id == context.company.id)
    ) == 1
    assert unit.property_type == "Duplex"
    assert unit.base_price == Decimal("6100000.00")
    assert unit.status == "Hold"
    assert payment_plan is not None
    assert payment_plan.plan_name == "Updated Plan"
    assert payment_plan.total_price == Decimal("6400000.00")


def test_unit_updates_are_isolated_by_compound_identity(
    db_session: Session,
):
    context = _seed_context(db_session, compound_key="compound:a")
    compound_b = _add_compound(
        db_session,
        company_id=context.company.id,
        compound_key="compound:b",
        name="Compound B",
    )
    previous = _snapshot(
        context.inventory_scope,
        [
            _unit_schema(
                context.compound,
                base_price=Decimal("5000000.00"),
            ),
            _unit_schema(compound_b, base_price=Decimal("7000000.00")),
        ],
    )
    persist_inventory_snapshot(
        db_session,
        company_id=context.company.id,
        snapshot=previous,
        changes=[],
    )
    current = _snapshot(
        context.inventory_scope,
        [
            _unit_schema(
                context.compound,
                base_price=Decimal("5500000.00"),
            ),
            _unit_schema(compound_b, base_price=Decimal("7000000.00")),
        ],
    )

    persist_inventory_snapshot(
        db_session,
        company_id=context.company.id,
        snapshot=current,
        changes=detect_inventory_changes(previous, current),
    )

    unit_a = get_unit_by_identity(
        db_session,
        company_id=context.company.id,
        compound_id=context.compound.id,
        external_unit_id="A101",
    )
    unit_b = get_unit_by_identity(
        db_session,
        company_id=context.company.id,
        compound_id=compound_b.id,
        external_unit_id="A101",
    )
    assert unit_a is not None and unit_a.base_price == Decimal("5500000.00")
    assert unit_b is not None and unit_b.base_price == Decimal("7000000.00")


def test_unit_updates_are_isolated_by_company(
    db_session: Session,
):
    tenant_a = _seed_context(db_session, compound_key="compound:shared")
    tenant_b = _seed_context(db_session, compound_key="compound:shared")
    snapshot_a = _snapshot(
        tenant_a.inventory_scope,
        [_unit_schema(tenant_a.compound, base_price=Decimal("5000000.00"))],
    )
    snapshot_b = _snapshot(
        tenant_b.inventory_scope,
        [_unit_schema(tenant_b.compound, base_price=Decimal("7000000.00"))],
    )
    persist_inventory_snapshot(
        db_session,
        company_id=tenant_a.company.id,
        snapshot=snapshot_a,
        changes=[],
    )
    persist_inventory_snapshot(
        db_session,
        company_id=tenant_b.company.id,
        snapshot=snapshot_b,
        changes=[],
    )
    updated_a = _snapshot(
        tenant_a.inventory_scope,
        [_unit_schema(tenant_a.compound, base_price=Decimal("5500000.00"))],
    )

    persist_inventory_snapshot(
        db_session,
        company_id=tenant_a.company.id,
        snapshot=updated_a,
        changes=detect_inventory_changes(snapshot_a, updated_a),
    )

    unit_b = get_unit_by_identity(
        db_session,
        company_id=tenant_b.company.id,
        compound_id=tenant_b.compound.id,
        external_unit_id="A101",
    )
    assert unit_b is not None
    assert unit_b.base_price == Decimal("7000000.00")


def test_whole_unit_missing_marks_unit_and_all_plan_states_absent(
    db_session: Session,
):
    context = _seed_context(db_session)
    observed = _snapshot(
        context.inventory_scope,
        [
            _unit_schema(
                context.compound,
                payment_plans=[
                    _plan_schema(plan_key="plan:a"),
                    _plan_schema(plan_key="plan:b"),
                ],
            )
        ],
    )
    persist_inventory_snapshot(
        db_session,
        company_id=context.company.id,
        snapshot=observed,
        changes=[],
    )
    missing = _snapshot(context.inventory_scope, [])

    persist_inventory_snapshot(
        db_session,
        company_id=context.company.id,
        snapshot=missing,
        changes=detect_inventory_changes(observed, missing),
    )

    unit = get_unit_by_identity(
        db_session,
        company_id=context.company.id,
        compound_id=context.compound.id,
        external_unit_id="A101",
    )
    assert unit is not None
    unit_state = db_session.scalar(
        select(UnitInventoryState).where(
            UnitInventoryState.company_id == context.company.id,
            UnitInventoryState.unit_id == unit.id,
            UnitInventoryState.inventory_scope_id == context.inventory_scope.id,
        )
    )
    plan_states = db_session.scalars(
        select(PaymentPlanInventoryState).where(
            PaymentPlanInventoryState.company_id == context.company.id,
            PaymentPlanInventoryState.inventory_scope_id
            == context.inventory_scope.id,
        )
    ).all()
    history = db_session.scalar(
        select(UnitHistory).where(
            UnitHistory.company_id == context.company.id,
            UnitHistory.unit_id == unit.id,
            UnitHistory.inventory_scope_id == context.inventory_scope.id,
        )
    )
    assert unit_state is not None
    assert unit_state.is_present_in_latest_inventory is False
    assert len(plan_states) == 2
    assert all(state.is_present_in_latest_inventory is False for state in plan_states)
    assert unit.status == "Available"
    assert history is not None and history.change_type == "missing"
    assert history.payment_plan_changes == []


def test_missing_plan_marks_only_that_plan_absent_while_unit_stays_present(
    db_session: Session,
):
    context = _seed_context(db_session)
    previous = _snapshot(
        context.inventory_scope,
        [
            _unit_schema(
                context.compound,
                payment_plans=[
                    _plan_schema(plan_key="plan:present"),
                    _plan_schema(plan_key="plan:missing"),
                ],
            )
        ],
    )
    persist_inventory_snapshot(
        db_session,
        company_id=context.company.id,
        snapshot=previous,
        changes=[],
    )
    current = _snapshot(
        context.inventory_scope,
        [
            _unit_schema(
                context.compound,
                payment_plans=[_plan_schema(plan_key="plan:present")],
            )
        ],
    )

    persist_inventory_snapshot(
        db_session,
        company_id=context.company.id,
        snapshot=current,
        changes=detect_inventory_changes(previous, current),
    )

    unit = get_unit_by_identity(
        db_session,
        company_id=context.company.id,
        compound_id=context.compound.id,
        external_unit_id="A101",
    )
    assert unit is not None
    unit_state = db_session.scalar(
        select(UnitInventoryState).where(
            UnitInventoryState.company_id == context.company.id,
            UnitInventoryState.unit_id == unit.id,
            UnitInventoryState.inventory_scope_id == context.inventory_scope.id,
        )
    )
    states_by_key = _plan_states_by_key(
        db_session,
        company_id=context.company.id,
        unit_id=unit.id,
        inventory_scope_id=context.inventory_scope.id,
    )
    history = db_session.scalar(
        select(UnitHistory).where(
            UnitHistory.company_id == context.company.id,
            UnitHistory.unit_id == unit.id,
            UnitHistory.inventory_scope_id == context.inventory_scope.id,
        )
    )
    assert unit_state is not None
    assert unit_state.is_present_in_latest_inventory is True
    assert states_by_key["plan:present"].is_present_in_latest_inventory is True
    assert states_by_key["plan:missing"].is_present_in_latest_inventory is False
    assert history is not None
    assert history.payment_plan_changes[0]["plan_key"] == "plan:missing"
    assert history.payment_plan_changes[0]["change_type"] == "missing"


def test_unit_and_plan_reappearance_restore_presence_and_last_seen(
    db_session: Session,
):
    context = _seed_context(db_session)
    observed = _snapshot(
        context.inventory_scope,
        [_unit_schema(context.compound, payment_plans=[_plan_schema()])],
    )
    persist_inventory_snapshot(
        db_session,
        company_id=context.company.id,
        snapshot=observed,
        changes=[],
    )
    unit = get_unit_by_identity(
        db_session,
        company_id=context.company.id,
        compound_id=context.compound.id,
        external_unit_id="A101",
    )
    assert unit is not None
    first_seen = db_session.scalar(
        select(UnitInventoryState.last_seen_at).where(
            UnitInventoryState.company_id == context.company.id,
            UnitInventoryState.unit_id == unit.id,
            UnitInventoryState.inventory_scope_id == context.inventory_scope.id,
        )
    )
    missing = _snapshot(context.inventory_scope, [])
    persist_inventory_snapshot(
        db_session,
        company_id=context.company.id,
        snapshot=missing,
        changes=[],
    )

    persist_inventory_snapshot(
        db_session,
        company_id=context.company.id,
        snapshot=observed,
        changes=[],
    )

    unit_state = db_session.scalar(
        select(UnitInventoryState).where(
            UnitInventoryState.company_id == context.company.id,
            UnitInventoryState.unit_id == unit.id,
            UnitInventoryState.inventory_scope_id == context.inventory_scope.id,
        )
    )
    plan_state = db_session.scalar(
        select(PaymentPlanInventoryState).where(
            PaymentPlanInventoryState.company_id == context.company.id,
            PaymentPlanInventoryState.inventory_scope_id
            == context.inventory_scope.id,
        )
    )
    assert unit_state is not None and unit_state.is_present_in_latest_inventory is True
    assert plan_state is not None and plan_state.is_present_in_latest_inventory is True
    assert unit_state.last_seen_at == plan_state.last_seen_at
    assert first_seen is not None and unit_state.last_seen_at > first_seen


def test_history_uses_compound_key_identity_and_json_safe_decimal_values(
    db_session: Session,
):
    context = _seed_context(db_session, compound_key="compound:a")
    compound_b = _add_compound(
        db_session,
        company_id=context.company.id,
        compound_key="compound:b",
        name="Compound B",
    )
    previous = _snapshot(
        context.inventory_scope,
        [
            _unit_schema(context.compound, base_price=Decimal("5000000.10")),
            _unit_schema(compound_b, base_price=Decimal("7000000.20")),
        ],
    )
    persist_inventory_snapshot(
        db_session,
        company_id=context.company.id,
        snapshot=previous,
        changes=[],
    )
    current = _snapshot(
        context.inventory_scope,
        [
            _unit_schema(context.compound, base_price=Decimal("5500000.30")),
            _unit_schema(compound_b, base_price=Decimal("7000000.20")),
        ],
    )

    persist_inventory_snapshot(
        db_session,
        company_id=context.company.id,
        snapshot=current,
        changes=detect_inventory_changes(previous, current),
    )

    history = db_session.scalar(
        select(UnitHistory).where(
            UnitHistory.company_id == context.company.id,
            UnitHistory.inventory_scope_id == context.inventory_scope.id,
        )
    )
    assert history is not None
    assert history.unit_id == get_unit_by_identity(
        db_session,
        company_id=context.company.id,
        compound_id=context.compound.id,
        external_unit_id="A101",
    ).id
    assert history.changes["base_price"] == {
        "old": "5000000.10",
        "new": "5500000.30",
    }


def test_failure_after_unit_upsert_rolls_back_entire_operation(
    db_session: Session,
):
    context = _seed_context(db_session)
    db_session.commit()
    snapshot = _snapshot(
        context.inventory_scope,
        [_unit_schema(context.compound)],
    )
    invalid_change = detect_inventory_changes(
        _snapshot(context.inventory_scope, []),
        _snapshot(
            context.inventory_scope,
            [_unit_schema(context.compound, external_unit_id="unknown")],
        ),
    )

    with pytest.raises(InventoryUnitNotFoundError):
        persist_inventory_snapshot(
            db_session,
            company_id=context.company.id,
            snapshot=snapshot,
            changes=invalid_change,
        )

    assert db_session.scalar(
        select(func.count())
        .select_from(Unit)
        .where(Unit.company_id == context.company.id)
    ) == 0
    assert db_session.scalar(
        select(func.count())
        .select_from(UnitInventoryState)
        .where(UnitInventoryState.company_id == context.company.id)
    ) == 0


def _add_scope(
    session: Session,
    *,
    company_id: int,
    scope_id: int,
    scope_key: str,
) -> InventoryScope:
    company = Company(id=company_id, name=f"Company {company_id}")
    inventory_scope = InventoryScope(
        id=scope_id,
        company_id=company_id,
        compound_id=None,
        scope_key=scope_key,
        source_identifier=f"source:{scope_key}",
    )
    session.add_all([company, inventory_scope])
    session.flush()
    return inventory_scope


def _seed_context(
    session: Session,
    *,
    compound_key: str | None = None,
) -> PersistenceContext:
    token = uuid4().hex
    company = Company(id=_test_id(), name=f"Company {token}")
    session.add(company)
    session.flush()
    compound = _add_compound(
        session,
        company_id=company.id,
        compound_key=compound_key or f"compound:{token}",
        name=f"Compound {token}",
    )
    inventory_scope = InventoryScope(
        id=_test_id(),
        company_id=company.id,
        compound_id=None,
        scope_key=f"scope:{token}",
        source_identifier=f"source:{token}",
    )
    session.add(inventory_scope)
    session.flush()
    return PersistenceContext(company, inventory_scope, compound)


def _add_compound(
    session: Session,
    *,
    company_id: int,
    compound_key: str,
    name: str,
    location: str = "New Cairo",
) -> Compound:
    compound = Compound(
        id=_test_id(),
        company_id=company_id,
        compound_key=compound_key,
        name=name,
        location=location,
    )
    session.add(compound)
    session.flush()
    return compound


def _snapshot(
    inventory_scope: InventoryScope,
    units: list[UnitSchema],
) -> InventorySnapshot:
    return InventorySnapshot(scope_key=inventory_scope.scope_key, units=units)


def _unit_schema(compound: Compound, **overrides) -> UnitSchema:
    data = {
        "external_unit_id": "A101",
        "compound_key": compound.compound_key,
        "compound_name": compound.name,
        "location": compound.location,
        "property_type": "Apartment",
        "base_price": Decimal("5500000.00"),
        "base_currency": Currency.EGP,
        "status": UnitStatus.AVAILABLE,
        "bedrooms": 3,
        "bathrooms": 2,
        "area_sqm": Decimal("150.00"),
        "delivery_date": date(2028, 6, 30),
        "payment_plans": [],
        "source": "inventory.csv",
        "last_updated": date(2026, 9, 5),
    }
    data.update(overrides)
    return UnitSchema(**data)


def _plan_schema(**overrides) -> PaymentPlanSchema:
    data = {
        "plan_key": "plan:installment",
        "source_plan_id": "source-plan-1",
        "plan_name": "7 Years",
        "payment_method": PaymentMethod.INSTALLMENT,
        "down_payment": Decimal("500000.00"),
        "installment_years": Decimal("7.00"),
        "total_price": Decimal("6000000.00"),
        "currency": Currency.EGP,
    }
    data.update(overrides)
    return PaymentPlanSchema(**data)


def _plan_states_by_key(
    session: Session,
    *,
    company_id: int,
    unit_id: int,
    inventory_scope_id: int,
) -> dict[str, PaymentPlanInventoryState]:
    statement = (
        select(PaymentPlan, PaymentPlanInventoryState)
        .join(
            PaymentPlanInventoryState,
            PaymentPlanInventoryState.payment_plan_id == PaymentPlan.id,
        )
        .where(
            PaymentPlan.company_id == company_id,
            PaymentPlan.unit_id == unit_id,
            PaymentPlanInventoryState.company_id == company_id,
            PaymentPlanInventoryState.inventory_scope_id == inventory_scope_id,
        )
    )
    return {
        payment_plan.plan_key: state
        for payment_plan, state in session.execute(statement)
    }


def _test_id() -> int:
    return -((uuid4().int % 1_000_000_000) + 1)
