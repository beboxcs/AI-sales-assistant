from collections.abc import Generator
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import NamedTuple
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import engine
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


class TenantData(NamedTuple):
    company: Company
    compound: Compound
    inventory_scope: InventoryScope
    unit: Unit
    payment_plan: PaymentPlan


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    assert engine.dialect.name == "postgresql"

    connection = engine.connect()
    outer_transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")

    try:
        yield session
    finally:
        session.close()
        if outer_transaction.is_active:
            outer_transaction.rollback()
        connection.close()


def test_duplicate_compound_key_within_company_is_rejected(
    db_session: Session,
):
    company_id = _test_id()
    compound_key = f"compound:{uuid4().hex}"
    company = Company(id=company_id, name="Compound Key Test")
    first_compound = Compound(
        id=_test_id(),
        company_id=company_id,
        compound_key=compound_key,
        name="First Compound",
        location="New Cairo",
    )
    db_session.add_all([company, first_compound])
    db_session.flush()

    duplicate = Compound(
        id=_test_id(),
        company_id=company_id,
        compound_key=compound_key,
        name="Second Compound",
        location="Sheikh Zayed",
    )

    _assert_rejected_by(
        db_session,
        duplicate,
        "uq_compounds_company_compound_key",
    )


def test_same_compound_key_is_allowed_for_different_companies(
    db_session: Session,
):
    company_a_id = _test_id()
    company_b_id = _test_id()
    compound_key = f"compound:{uuid4().hex}"
    company_a = Company(id=company_a_id, name="Company A")
    company_b = Company(id=company_b_id, name="Company B")
    compound_a = Compound(
        id=_test_id(),
        company_id=company_a_id,
        compound_key=compound_key,
        name="Company A Compound",
        location="New Cairo",
    )
    compound_b = Compound(
        id=_test_id(),
        company_id=company_b_id,
        compound_key=compound_key,
        name="Company B Compound",
        location="New Cairo",
    )

    db_session.add_all([company_a, company_b, compound_a, compound_b])
    db_session.flush()

    assert compound_a.compound_key == compound_b.compound_key
    assert compound_a.company_id != compound_b.company_id


def test_unit_inventory_state_rejects_cross_company_scope(
    db_session: Session,
):
    tenant_a = _seed_tenant(db_session, "unit-state-a")
    tenant_b = _seed_tenant(db_session, "unit-state-b")

    state = UnitInventoryState(
        id=_test_id(),
        company_id=tenant_a.company.id,
        unit_id=tenant_a.unit.id,
        inventory_scope_id=tenant_b.inventory_scope.id,
    )

    _assert_rejected_by(
        db_session, state, "fk_unit_inventory_states_company_scope"
    )


def test_payment_plan_inventory_state_rejects_cross_company_scope(
    db_session: Session,
):
    tenant_a = _seed_tenant(db_session, "plan-state-a")
    tenant_b = _seed_tenant(db_session, "plan-state-b")

    state = PaymentPlanInventoryState(
        id=_test_id(),
        company_id=tenant_a.company.id,
        payment_plan_id=tenant_a.payment_plan.id,
        inventory_scope_id=tenant_b.inventory_scope.id,
    )

    _assert_rejected_by(
        db_session, state, "fk_payment_plan_inventory_states_company_scope"
    )


def test_unit_history_rejects_cross_company_scope(db_session: Session):
    tenant_a = _seed_tenant(db_session, "history-a")
    tenant_b = _seed_tenant(db_session, "history-b")

    history = UnitHistory(
        id=_test_id(),
        company_id=tenant_a.company.id,
        unit_id=tenant_a.unit.id,
        inventory_scope_id=tenant_b.inventory_scope.id,
        change_type="updated",
        changes={},
        payment_plan_changes=[],
        detected_at=datetime.now(timezone.utc),
    )

    _assert_rejected_by(db_session, history, "fk_unit_history_company_scope")


def test_payment_plan_rejects_company_different_from_unit(db_session: Session):
    tenant_a = _seed_tenant(db_session, "plan-company-a")
    tenant_b = _seed_tenant(db_session, "plan-company-b")

    payment_plan = PaymentPlan(
        id=_test_id(),
        company_id=tenant_b.company.id,
        unit_id=tenant_a.unit.id,
        plan_key=f"mismatched-company:{uuid4().hex}",
        payment_method="Cash",
    )

    _assert_rejected_by(db_session, payment_plan, "fk_payment_plans_company_unit")


def test_cash_payment_plan_rejects_installment_years(db_session: Session):
    tenant = _seed_tenant(db_session, "cash-invariant")

    payment_plan = PaymentPlan(
        id=_test_id(),
        company_id=tenant.company.id,
        unit_id=tenant.unit.id,
        plan_key=f"invalid-cash:{uuid4().hex}",
        payment_method="Cash",
        installment_years=Decimal("1.00"),
    )

    _assert_rejected_by(
        db_session,
        payment_plan,
        "ck_payment_plans_cash_installment_years_null",
    )


def test_duplicate_unit_business_identity_is_rejected(db_session: Session):
    tenant = _seed_tenant(db_session, "duplicate-unit")

    duplicate = Unit(
        id=_test_id(),
        company_id=tenant.company.id,
        compound_id=tenant.compound.id,
        external_unit_id=tenant.unit.external_unit_id,
        property_type="Apartment",
        status="Available",
        source_last_updated=date(2026, 9, 2),
    )

    _assert_rejected_by(
        db_session, duplicate, "uq_units_company_compound_external_unit"
    )


def test_duplicate_payment_plan_business_identity_is_rejected(
    db_session: Session,
):
    tenant = _seed_tenant(db_session, "duplicate-plan")

    duplicate = PaymentPlan(
        id=_test_id(),
        company_id=tenant.company.id,
        unit_id=tenant.unit.id,
        plan_key=tenant.payment_plan.plan_key,
        payment_method="Cash",
    )

    _assert_rejected_by(db_session, duplicate, "uq_payment_plans_unit_plan_key")


def test_duplicate_unit_inventory_state_is_rejected(db_session: Session):
    tenant = _seed_tenant(db_session, "duplicate-unit-state")
    first_state = UnitInventoryState(
        id=_test_id(),
        company_id=tenant.company.id,
        unit_id=tenant.unit.id,
        inventory_scope_id=tenant.inventory_scope.id,
    )
    db_session.add(first_state)
    db_session.flush()

    duplicate = UnitInventoryState(
        id=_test_id(),
        company_id=tenant.company.id,
        unit_id=tenant.unit.id,
        inventory_scope_id=tenant.inventory_scope.id,
    )

    _assert_rejected_by(
        db_session, duplicate, "uq_unit_inventory_states_unit_scope"
    )


def test_duplicate_payment_plan_inventory_state_is_rejected(
    db_session: Session,
):
    tenant = _seed_tenant(db_session, "duplicate-plan-state")
    first_state = PaymentPlanInventoryState(
        id=_test_id(),
        company_id=tenant.company.id,
        payment_plan_id=tenant.payment_plan.id,
        inventory_scope_id=tenant.inventory_scope.id,
    )
    db_session.add(first_state)
    db_session.flush()

    duplicate = PaymentPlanInventoryState(
        id=_test_id(),
        company_id=tenant.company.id,
        payment_plan_id=tenant.payment_plan.id,
        inventory_scope_id=tenant.inventory_scope.id,
    )

    _assert_rejected_by(
        db_session,
        duplicate,
        "uq_payment_plan_inventory_states_plan_scope",
    )


def _seed_tenant(db_session: Session, label: str) -> TenantData:
    entity_id = _test_id()
    token = uuid4().hex

    company = Company(id=entity_id, name=f"Integrity Test {label} {token}")
    db_session.add(company)
    db_session.flush()

    compound = Compound(
        id=entity_id,
        company_id=company.id,
        compound_key=f"compound:{token}",
        name=f"Compound {token}",
        location=f"Location {token}",
    )
    db_session.add(compound)
    db_session.flush()

    inventory_scope = InventoryScope(
        id=entity_id,
        company_id=company.id,
        compound_id=compound.id,
        scope_key=f"scope:{token}",
        source_identifier=f"source:{token}",
    )
    unit = Unit(
        id=entity_id,
        company_id=company.id,
        compound_id=compound.id,
        external_unit_id=f"unit:{token}",
        property_type="Apartment",
        status="Available",
        source_last_updated=date(2026, 9, 2),
    )
    db_session.add_all([inventory_scope, unit])
    db_session.flush()

    payment_plan = PaymentPlan(
        id=entity_id,
        company_id=company.id,
        unit_id=unit.id,
        plan_key=f"cash:{token}",
        payment_method="Cash",
    )
    db_session.add(payment_plan)
    db_session.flush()

    return TenantData(company, compound, inventory_scope, unit, payment_plan)


def _assert_rejected_by(
    db_session: Session,
    instance: object,
    constraint_name: str,
) -> None:
    db_session.add(instance)

    with pytest.raises(IntegrityError) as exc_info:
        db_session.flush()

    db_session.rollback()

    assert exc_info.value.orig.diag.constraint_name == constraint_name


def _test_id() -> int:
    return -((uuid4().int % 1_000_000_000) + 1)
