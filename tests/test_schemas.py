from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.ai_engine.schemas import (
    InventorySnapshot,
    PaymentPlanSchema,
    UnitSchema,
)
from app.domain.enums import Currency, PaymentMethod, UnitStatus


def test_unit_schema_uses_external_unit_id_and_excludes_old_unit_id():
    unit = _unit()

    output = unit.model_dump()

    assert output["external_unit_id"] == "A101"
    assert "unit_id" not in output


def test_unit_schema_requires_and_strips_compound_key():
    unit = _unit(compound_key="  compound-a  ")

    assert unit.compound_key == "compound-a"


def test_unit_schema_rejects_empty_compound_key():
    with pytest.raises(ValidationError, match="compound_key"):
        _unit(compound_key=" ")


def test_unit_schema_requires_compound_key():
    data = _unit().model_dump()
    data.pop("compound_key")

    with pytest.raises(ValidationError, match="compound_key"):
        UnitSchema(**data)


def test_unit_schema_allows_nullable_base_price():
    unit = _unit(base_price=None)

    assert unit.base_price is None


def test_unit_schema_defaults_to_egp_currency():
    unit = _unit()
    plan = _plan()

    assert unit.base_currency == Currency.EGP
    assert plan.currency == Currency.EGP


def test_ai_schemas_use_shared_domain_enums():
    assert PaymentPlanSchema.model_fields["payment_method"].annotation is PaymentMethod
    assert PaymentPlanSchema.model_fields["currency"].annotation is Currency
    assert UnitSchema.model_fields["base_currency"].annotation is Currency
    assert UnitSchema.model_fields["status"].annotation is UnitStatus


def test_unit_schema_allows_multiple_payment_plans():
    unit = _unit(
        payment_plans=[
            _plan(
                plan_key="terms:cash",
                payment_method=PaymentMethod.CASH,
                installment_years=None,
            ),
            _plan(plan_key="name:5_years", plan_name="5 Years"),
        ]
    )

    assert len(unit.payment_plans) == 2


def test_unit_schema_rejects_duplicate_plan_keys():
    with pytest.raises(ValidationError, match="Duplicate payment plan keys"):
        _unit(payment_plans=[_plan(plan_key="name:5_years"), _plan(plan_key="name:5_years")])


def test_payment_plan_rejects_cash_with_installment_years():
    with pytest.raises(ValidationError, match="Cash payment plans"):
        _plan(payment_method=PaymentMethod.CASH, installment_years=5)


def test_payment_plan_allows_cash_without_installment_years():
    plan = _plan(payment_method=PaymentMethod.CASH, installment_years=None)

    assert plan.payment_method == PaymentMethod.CASH
    assert plan.installment_years is None


def test_payment_plan_allows_installment_without_installment_years():
    plan = _plan(payment_method=PaymentMethod.INSTALLMENT, installment_years=None)

    assert plan.payment_method == PaymentMethod.INSTALLMENT
    assert plan.installment_years is None


def test_payment_plan_strips_surrounding_plan_key_whitespace():
    plan = _plan(plan_key="  name:5_years  ")

    assert plan.plan_key == "name:5_years"


def test_unit_schema_strips_surrounding_external_unit_id_whitespace():
    unit = _unit(external_unit_id="  A101  ")

    assert unit.external_unit_id == "A101"


def test_inventory_snapshot_strips_surrounding_scope_key_whitespace():
    snapshot = InventorySnapshot(scope_key="  developer:demo  ", units=[_unit()])

    assert snapshot.scope_key == "developer:demo"


def test_inventory_snapshot_requires_non_empty_scope_key():
    with pytest.raises(ValidationError, match="scope_key"):
        InventorySnapshot(scope_key=" ", units=[_unit()])


def test_inventory_snapshot_rejects_duplicate_external_unit_ids():
    with pytest.raises(ValidationError, match=r"Duplicate \(compound_key, external_unit_id\)"):
        InventorySnapshot(scope_key="developer:demo", units=[_unit(), _unit()])


def test_inventory_snapshot_allows_same_external_id_in_different_compounds():
    snapshot = InventorySnapshot(
        scope_key="developer:demo",
        units=[_unit(compound_key="compound-a"), _unit(compound_key="compound-b")],
    )

    assert len(snapshot.units) == 2


def test_decimal_numeric_schema_fields_are_decimal():
    unit = _unit()
    plan = _plan()

    assert isinstance(unit.base_price, Decimal)
    assert isinstance(unit.area_sqm, Decimal)
    assert isinstance(plan.down_payment, Decimal)
    assert isinstance(plan.installment_years, Decimal)
    assert isinstance(plan.total_price, Decimal)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_price", Decimal("0")),
        ("area_sqm", Decimal("0")),
        ("down_payment", Decimal("-1")),
        ("installment_years", Decimal("0")),
        ("total_price", Decimal("0")),
    ],
)
def test_decimal_numeric_bounds_remain_validated(field, value):
    data = _unit()
    if field in {"down_payment", "installment_years", "total_price"}:
        plan_data = _plan().model_dump()
        plan_data[field] = value
        with pytest.raises(ValidationError):
            PaymentPlanSchema(**plan_data)
    else:
        data = data.model_dump()
        data[field] = value
        with pytest.raises(ValidationError):
            UnitSchema(**data)


def _plan(**overrides) -> PaymentPlanSchema:
    data = {
        "plan_key": "name:5_years",
        "plan_name": "5 Years",
        "payment_method": PaymentMethod.INSTALLMENT,
        "down_payment": 550000.0,
        "installment_years": 5.0,
        "total_price": 5800000.0,
    }
    data.update(overrides)
    return PaymentPlanSchema(**data)


def _unit(**overrides) -> UnitSchema:
    data = {
        "external_unit_id": "A101",
        "compound_key": "compound-a",
        "compound_name": "Nile Gardens",
        "location": "New Cairo",
        "property_type": "Apartment",
        "base_price": 5500000.0,
        "status": UnitStatus.AVAILABLE,
        "source": "sample.csv",
        "last_updated": date(2026, 9, 1),
        "bedrooms": 3,
        "bathrooms": 2,
        "area_sqm": 150.0,
        "delivery_date": date(2028, 6, 30),
    }
    data.update(overrides)
    return UnitSchema(**data)
