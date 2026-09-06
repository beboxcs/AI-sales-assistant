import json
from datetime import date

import pytest
from pydantic import ValidationError

from app.ai_engine.change_detector import (
    InventoryChange,
    InventoryChangeType,
    PaymentPlanChangeType,
    detect_inventory_changes,
)


def test_inventory_change_strips_compound_key_whitespace():
    change = InventoryChange(
        compound_key="  compound-a  ",
        external_unit_id="A101",
        change_type=InventoryChangeType.ADDED,
    )

    assert change.compound_key == "compound-a"


def test_inventory_change_rejects_empty_compound_key():
    with pytest.raises(ValidationError, match="compound_key"):
        InventoryChange(
            compound_key=" ",
            external_unit_id="A101",
            change_type=InventoryChangeType.ADDED,
        )
from app.ai_engine.schemas import (
    InventorySnapshot,
    PaymentMethod,
    PaymentPlanSchema,
    UnitSchema,
    UnitStatus,
)


def test_detect_inventory_changes_raises_for_scope_mismatch():
    with pytest.raises(ValueError, match="scope_key"):
        detect_inventory_changes(
            _snapshot(scope_key="scope:a", units=[_unit()]),
            _snapshot(scope_key="scope:b", units=[_unit()]),
        )


def test_detect_inventory_changes_returns_empty_for_no_changes():
    assert detect_inventory_changes(_snapshot(units=[_unit()]), _snapshot(units=[_unit()])) == []


def test_detect_inventory_changes_reports_added_unit():
    changes = detect_inventory_changes(
        _snapshot(units=[_unit(external_unit_id="A101")]),
        _snapshot(units=[_unit(external_unit_id="A101"), _unit(external_unit_id="A102")]),
    )

    assert [change.external_unit_id for change in changes] == ["A102"]
    assert changes[0].change_type == InventoryChangeType.ADDED
    assert changes[0].compound_key == "compound-a"
    assert changes[0].changes == {}
    assert changes[0].payment_plan_changes == []


def test_detect_inventory_changes_reports_missing_unit_without_marking_sold():
    changes = detect_inventory_changes(
        _snapshot(units=[_unit(external_unit_id="A101"), _unit(external_unit_id="A102")]),
        _snapshot(units=[_unit(external_unit_id="A101")]),
    )

    assert [change.external_unit_id for change in changes] == ["A102"]
    assert changes[0].change_type == InventoryChangeType.MISSING
    assert changes[0].compound_key == "compound-a"
    assert "status" not in changes[0].changes


def test_detect_inventory_changes_reports_base_price_update():
    changes = detect_inventory_changes(
        _snapshot(units=[_unit(base_price=5500000.0)]),
        _snapshot(units=[_unit(base_price=5800000.0)]),
    )

    assert len(changes) == 1
    assert changes[0].change_type == InventoryChangeType.UPDATED
    assert set(changes[0].changes) == {"base_price"}
    assert changes[0].changes["base_price"].old == "5500000.0"
    assert changes[0].changes["base_price"].new == "5800000.0"
    assert isinstance(changes[0].changes["base_price"].old, str)
    assert isinstance(changes[0].changes["base_price"].new, str)


def test_detect_inventory_changes_serializes_delivery_date_changes_as_iso_strings():
    changes = detect_inventory_changes(
        _snapshot(units=[_unit(delivery_date=date(2027, 5, 1))]),
        _snapshot(units=[_unit(delivery_date=date(2028, 6, 15))]),
    )

    delivery_change = changes[0].changes["delivery_date"]
    assert delivery_change.old == "2027-05-01"
    assert delivery_change.new == "2028-06-15"


def test_detect_inventory_changes_serializes_enum_changes_as_readable_values():
    changes = detect_inventory_changes(
        _snapshot(units=[_unit(status=UnitStatus.AVAILABLE)]),
        _snapshot(units=[_unit(status=UnitStatus.SOLD)]),
    )

    status_change = changes[0].changes["status"]
    assert status_change.old == "Available"
    assert status_change.new == "Sold"


def test_detect_inventory_changes_ignores_metadata_only_updates():
    changes = detect_inventory_changes(
        _snapshot(units=[_unit(source="previous.csv", last_updated=date(2026, 8, 31))]),
        _snapshot(units=[_unit(source="current.csv", last_updated=date(2026, 9, 1))]),
    )

    assert changes == []


def test_detect_inventory_changes_reports_payment_plan_added():
    changes = detect_inventory_changes(
        _snapshot(units=[_unit(payment_plans=[_plan(plan_key="terms:cash")])]),
        _snapshot(
            units=[
                _unit(
                    payment_plans=[
                        _plan(plan_key="terms:cash"),
                        _plan(plan_key="name:5_years", plan_name="5 Years"),
                    ]
                )
            ]
        ),
    )

    plan_changes = changes[0].payment_plan_changes
    assert changes[0].change_type == InventoryChangeType.UPDATED
    assert changes[0].changes == {}
    assert len(plan_changes) == 1
    assert plan_changes[0].plan_key == "name:5_years"
    assert plan_changes[0].change_type == PaymentPlanChangeType.ADDED


def test_detect_inventory_changes_reports_payment_plan_missing():
    changes = detect_inventory_changes(
        _snapshot(
            units=[
                _unit(
                    payment_plans=[
                        _plan(plan_key="terms:cash"),
                        _plan(plan_key="name:5_years", plan_name="5 Years"),
                    ]
                )
            ]
        ),
        _snapshot(units=[_unit(payment_plans=[_plan(plan_key="terms:cash")])]),
    )

    plan_changes = changes[0].payment_plan_changes
    assert len(plan_changes) == 1
    assert plan_changes[0].plan_key == "name:5_years"
    assert plan_changes[0].change_type == PaymentPlanChangeType.MISSING


def test_detect_inventory_changes_reports_payment_plan_updated():
    changes = detect_inventory_changes(
        _snapshot(units=[_unit(payment_plans=[_plan(total_price=5800000.0)])]),
        _snapshot(units=[_unit(payment_plans=[_plan(total_price=6000000.0)])]),
    )

    plan_changes = changes[0].payment_plan_changes
    assert len(plan_changes) == 1
    assert plan_changes[0].change_type == PaymentPlanChangeType.UPDATED
    assert set(plan_changes[0].changes) == {"total_price"}
    assert plan_changes[0].changes["total_price"].old == "5800000.0"
    assert plan_changes[0].changes["total_price"].new == "6000000.0"


def test_inventory_change_model_dump_json_contains_json_compatible_values():
    changes = detect_inventory_changes(
        _snapshot(
            units=[
                _unit(
                    base_price=5500000.0,
                    delivery_date=date(2027, 5, 1),
                    payment_plans=[_plan(payment_method=PaymentMethod.INSTALLMENT)],
                )
            ]
        ),
        _snapshot(
            units=[
                _unit(
                    base_price=5600000.0,
                    delivery_date=date(2028, 6, 15),
                    payment_plans=[
                        _plan(
                            payment_method=PaymentMethod.CASH,
                            installment_years=None,
                        )
                    ],
                )
            ]
        ),
    )

    dumped_change = changes[0].model_dump(mode="json")

    json.dumps(dumped_change)
    assert dumped_change["changes"]["delivery_date"]["old"] == "2027-05-01"
    assert dumped_change["changes"]["base_price"] == {
        "old": "5500000.0",
        "new": "5600000.0",
    }
    assert dumped_change["payment_plan_changes"][0]["changes"]["payment_method"] == {
        "old": "Installment",
        "new": "Cash",
    }


def test_detect_inventory_changes_reports_multiple_payment_plan_changes():
    changes = detect_inventory_changes(
        _snapshot(
            units=[
                _unit(
                    payment_plans=[
                        _plan(plan_key="name:5_years", plan_name="5 Years", total_price=5800000.0),
                        _plan(plan_key="name:7_years", plan_name="7 Years"),
                    ]
                )
            ]
        ),
        _snapshot(
            units=[
                _unit(
                    payment_plans=[
                        _plan(plan_key="name:5_years", plan_name="5 Years", total_price=6000000.0),
                        _plan(
                            plan_key="terms:cash",
                            payment_method=PaymentMethod.CASH,
                            installment_years=None,
                        ),
                    ]
                )
            ]
        ),
    )

    plan_changes = changes[0].payment_plan_changes
    assert [plan_change.plan_key for plan_change in plan_changes] == [
        "name:5_years",
        "name:7_years",
        "terms:cash",
    ]
    assert [plan_change.change_type for plan_change in plan_changes] == [
        PaymentPlanChangeType.UPDATED,
        PaymentPlanChangeType.MISSING,
        PaymentPlanChangeType.ADDED,
    ]


def test_detect_inventory_changes_treats_changed_plan_key_as_missing_and_added():
    changes = detect_inventory_changes(
        _snapshot(units=[_unit(payment_plans=[_plan(plan_key="name:5_years")])]),
        _snapshot(units=[_unit(payment_plans=[_plan(plan_key="name:7_years", plan_name="7 Years")])]),
    )

    plan_changes = changes[0].payment_plan_changes
    assert [plan_change.plan_key for plan_change in plan_changes] == [
        "name:5_years",
        "name:7_years",
    ]
    assert [plan_change.change_type for plan_change in plan_changes] == [
        PaymentPlanChangeType.MISSING,
        PaymentPlanChangeType.ADDED,
    ]


def test_detect_inventory_changes_orders_unit_changes_deterministically():
    changes = detect_inventory_changes(
        _snapshot(units=[_unit(external_unit_id="B"), _unit(external_unit_id="A")]),
        _snapshot(units=[_unit(external_unit_id="C"), _unit(external_unit_id="A", base_price=1)]),
    )

    assert [change.external_unit_id for change in changes] == ["A", "B", "C"]


def test_detect_inventory_changes_uses_composite_unit_identity():
    previous = _snapshot(
        units=[
            _unit(compound_key="compound-a", base_price=5500000),
            _unit(compound_key="compound-b", base_price=6500000),
        ]
    )
    current = _snapshot(
        units=[
            _unit(compound_key="compound-a", base_price=5500000),
            _unit(compound_key="compound-b", base_price=6800000),
        ]
    )

    changes = detect_inventory_changes(previous, current)

    assert len(changes) == 1
    assert changes[0].compound_key == "compound-b"
    assert changes[0].external_unit_id == "A101"


def test_detect_inventory_changes_reports_composite_identity_for_missing_unit():
    previous = _snapshot(
        units=[
            _unit(compound_key="compound-a"),
            _unit(compound_key="compound-b"),
        ]
    )
    current = _snapshot(units=[_unit(compound_key="compound-a")])

    changes = detect_inventory_changes(previous, current)

    assert len(changes) == 1
    assert changes[0].change_type == InventoryChangeType.MISSING
    assert changes[0].compound_key == "compound-b"
    assert changes[0].external_unit_id == "A101"


def test_detect_inventory_changes_does_not_mutate_inputs():
    previous = _snapshot(units=[_unit(payment_plans=[_plan(total_price=5800000.0)])])
    current = _snapshot(units=[_unit(payment_plans=[_plan(total_price=6000000.0)])])
    previous_before = previous.model_dump()
    current_before = current.model_dump()

    detect_inventory_changes(previous, current)

    assert previous.model_dump() == previous_before
    assert current.model_dump() == current_before


def _snapshot(scope_key="developer:demo", units=None) -> InventorySnapshot:
    return InventorySnapshot(scope_key=scope_key, units=units or [])


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
        "payment_plans": [_plan()],
    }
    data.update(overrides)
    return UnitSchema(**data)
