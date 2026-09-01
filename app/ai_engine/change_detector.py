from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.ai_engine.schemas import UnitSchema


class InventoryChangeType(str, Enum):
    ADDED = "added"
    UPDATED = "updated"
    MISSING = "missing"


class FieldChange(BaseModel):
    old: Any
    new: Any


class InventoryChange(BaseModel):
    unit_id: str
    change_type: InventoryChangeType
    changes: dict[str, FieldChange] = Field(default_factory=dict)


TRACKED_FIELDS: tuple[str, ...] = (
    "compound_name",
    "location",
    "property_type",
    "price",
    "status",
    "bedrooms",
    "bathrooms",
    "area_sqm",
    "down_payment",
    "installment_years",
    "delivery_date",
)


def detect_inventory_changes(
    previous_units: list[UnitSchema],
    current_units: list[UnitSchema],
) -> list[InventoryChange]:
    previous_by_id = _index_units_by_id(previous_units, snapshot_name="previous")
    current_by_id = _index_units_by_id(current_units, snapshot_name="current")

    changes: list[InventoryChange] = []
    all_unit_ids = sorted(set(previous_by_id) | set(current_by_id))

    for unit_id in all_unit_ids:
        previous_unit = previous_by_id.get(unit_id)
        current_unit = current_by_id.get(unit_id)

        if previous_unit is None and current_unit is not None:
            changes.append(
                InventoryChange(
                    unit_id=unit_id,
                    change_type=InventoryChangeType.ADDED,
                )
            )
            continue

        if current_unit is None and previous_unit is not None:
            changes.append(
                InventoryChange(
                    unit_id=unit_id,
                    change_type=InventoryChangeType.MISSING,
                )
            )
            continue

        if previous_unit is None or current_unit is None:
            continue

        field_changes = _detect_field_changes(previous_unit, current_unit)
        if field_changes:
            changes.append(
                InventoryChange(
                    unit_id=unit_id,
                    change_type=InventoryChangeType.UPDATED,
                    changes=field_changes,
                )
            )

    return changes


def _index_units_by_id(
    units: list[UnitSchema],
    snapshot_name: str,
) -> dict[str, UnitSchema]:
    units_by_id: dict[str, UnitSchema] = {}
    duplicate_unit_ids: set[str] = set()

    for unit in units:
        if unit.unit_id in units_by_id:
            duplicate_unit_ids.add(unit.unit_id)
        units_by_id[unit.unit_id] = unit

    if duplicate_unit_ids:
        duplicate_list = ", ".join(sorted(duplicate_unit_ids))
        raise ValueError(
            f"Duplicate unit_id values found in {snapshot_name} snapshot: "
            f"{duplicate_list}"
        )

    return units_by_id


def _detect_field_changes(
    previous_unit: UnitSchema,
    current_unit: UnitSchema,
) -> dict[str, FieldChange]:
    changes: dict[str, FieldChange] = {}

    for field_name in TRACKED_FIELDS:
        old_value = getattr(previous_unit, field_name)
        new_value = getattr(current_unit, field_name)

        if old_value != new_value:
            changes[field_name] = FieldChange(
                old=_readable_value(old_value),
                new=_readable_value(new_value),
            )

    return changes


def _readable_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value
