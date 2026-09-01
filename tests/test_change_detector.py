from datetime import date

import pytest

from app.ai_engine.change_detector import (
    InventoryChangeType,
    detect_inventory_changes,
)
from app.ai_engine.schemas import UnitSchema, UnitStatus


def test_detect_inventory_changes_returns_empty_for_no_changes():
    previous = [_unit()]
    current = [_unit()]

    assert detect_inventory_changes(previous, current) == []


def test_detect_inventory_changes_reports_added_unit():
    previous = [_unit(unit_id="A101")]
    current = [_unit(unit_id="A101"), _unit(unit_id="A102")]

    changes = detect_inventory_changes(previous, current)

    assert len(changes) == 1
    assert changes[0].unit_id == "A102"
    assert changes[0].change_type == InventoryChangeType.ADDED
    assert changes[0].changes == {}


def test_detect_inventory_changes_reports_missing_unit_without_marking_sold():
    previous = [_unit(unit_id="A101"), _unit(unit_id="A102")]
    current = [_unit(unit_id="A101")]

    changes = detect_inventory_changes(previous, current)

    assert len(changes) == 1
    assert changes[0].unit_id == "A102"
    assert changes[0].change_type == InventoryChangeType.MISSING
    assert changes[0].change_type != InventoryChangeType.UPDATED
    assert "status" not in changes[0].changes


def test_detect_inventory_changes_reports_price_update_only():
    previous = [_unit(price=5500000.0)]
    current = [_unit(price=5800000.0)]

    changes = detect_inventory_changes(previous, current)

    assert len(changes) == 1
    assert changes[0].change_type == InventoryChangeType.UPDATED
    assert set(changes[0].changes) == {"price"}
    assert changes[0].changes["price"].old == 5500000.0
    assert changes[0].changes["price"].new == 5800000.0


def test_detect_inventory_changes_reports_status_update():
    previous = [_unit(status=UnitStatus.AVAILABLE)]
    current = [_unit(status=UnitStatus.SOLD)]

    changes = detect_inventory_changes(previous, current)

    assert len(changes) == 1
    assert changes[0].change_type == InventoryChangeType.UPDATED
    assert changes[0].changes["status"].old == "Available"
    assert changes[0].changes["status"].new == "Sold"


def test_detect_inventory_changes_groups_multiple_field_updates():
    previous = [_unit(price=5500000.0, status=UnitStatus.AVAILABLE, bedrooms=2)]
    current = [_unit(price=5800000.0, status=UnitStatus.RESERVED, bedrooms=3)]

    changes = detect_inventory_changes(previous, current)

    assert len(changes) == 1
    assert changes[0].change_type == InventoryChangeType.UPDATED
    assert set(changes[0].changes) == {"price", "status", "bedrooms"}
    assert changes[0].changes["price"].new == 5800000.0
    assert changes[0].changes["status"].new == "Reserved"
    assert changes[0].changes["bedrooms"].new == 3


def test_detect_inventory_changes_ignores_metadata_only_updates():
    previous = [
        _unit(source="previous.csv", last_updated=date(2026, 8, 31)),
    ]
    current = [
        _unit(source="current.csv", last_updated=date(2026, 9, 1)),
    ]

    assert detect_inventory_changes(previous, current) == []


def test_detect_inventory_changes_raises_for_duplicate_previous_unit_id():
    previous = [_unit(unit_id="A101"), _unit(unit_id="A101")]
    current = [_unit(unit_id="A101")]

    with pytest.raises(ValueError, match="previous snapshot"):
        detect_inventory_changes(previous, current)


def test_detect_inventory_changes_raises_for_duplicate_current_unit_id():
    previous = [_unit(unit_id="A101")]
    current = [_unit(unit_id="A101"), _unit(unit_id="A101")]

    with pytest.raises(ValueError, match="current snapshot"):
        detect_inventory_changes(previous, current)


def test_detect_inventory_changes_does_not_mutate_inputs():
    previous = [_unit(price=5500000.0)]
    current = [_unit(price=5800000.0)]
    previous_before = [unit.model_dump() for unit in previous]
    current_before = [unit.model_dump() for unit in current]

    detect_inventory_changes(previous, current)

    assert [unit.model_dump() for unit in previous] == previous_before
    assert [unit.model_dump() for unit in current] == current_before


def _unit(**overrides) -> UnitSchema:
    data = {
        "unit_id": "A101",
        "compound_name": "Nile Gardens",
        "location": "New Cairo",
        "property_type": "Apartment",
        "price": 5500000.0,
        "status": UnitStatus.AVAILABLE,
        "source": "sample.csv",
        "last_updated": date(2026, 9, 1),
        "bedrooms": 3,
        "bathrooms": 2,
        "area_sqm": 150.0,
        "down_payment": 550000.0,
        "installment_years": 7.0,
        "delivery_date": date(2028, 6, 30),
    }
    data.update(overrides)
    return UnitSchema(**data)
