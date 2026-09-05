from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.ai_engine.schemas import InventorySnapshot, PaymentPlanSchema, UnitSchema


class InventoryChangeType(str, Enum):
    ADDED = "added"
    UPDATED = "updated"
    MISSING = "missing"


class PaymentPlanChangeType(str, Enum):
    ADDED = "added"
    UPDATED = "updated"
    MISSING = "missing"


class FieldChange(BaseModel):
    old: Any
    new: Any


class PaymentPlanChange(BaseModel):
    plan_key: str
    change_type: PaymentPlanChangeType
    changes: dict[str, FieldChange] = Field(default_factory=dict)


class InventoryChange(BaseModel):
    compound_key: str
    external_unit_id: str
    change_type: InventoryChangeType
    changes: dict[str, FieldChange] = Field(default_factory=dict)
    payment_plan_changes: list[PaymentPlanChange] = Field(default_factory=list)

    @field_validator("compound_key")
    @classmethod
    def compound_key_must_not_be_empty(cls, value: str) -> str:
        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("compound_key must not be empty")
        return stripped_value


TRACKED_FIELDS: tuple[str, ...] = (
    "compound_name",
    "location",
    "property_type",
    "base_price",
    "base_currency",
    "status",
    "bedrooms",
    "bathrooms",
    "area_sqm",
    "delivery_date",
)

PAYMENT_PLAN_TRACKED_FIELDS: tuple[str, ...] = (
    "source_plan_id",
    "plan_name",
    "payment_method",
    "down_payment",
    "installment_years",
    "total_price",
    "currency",
)


def detect_inventory_changes(
    previous_snapshot: InventorySnapshot,
    current_snapshot: InventorySnapshot,
) -> list[InventoryChange]:
    if previous_snapshot.scope_key != current_snapshot.scope_key:
        raise ValueError(
            "Cannot compare inventory snapshots with different scope_key values: "
            f"{previous_snapshot.scope_key!r} != {current_snapshot.scope_key!r}"
        )

    previous_by_id = _index_units_by_identity(previous_snapshot.units, "previous")
    current_by_id = _index_units_by_identity(current_snapshot.units, "current")

    changes: list[InventoryChange] = []
    all_unit_identities = sorted(set(previous_by_id) | set(current_by_id))

    for compound_key, external_unit_id in all_unit_identities:
        unit_identity = (compound_key, external_unit_id)
        previous_unit = previous_by_id.get(unit_identity)
        current_unit = current_by_id.get(unit_identity)

        if previous_unit is None and current_unit is not None:
            changes.append(
                InventoryChange(
                    compound_key=compound_key,
                    external_unit_id=external_unit_id,
                    change_type=InventoryChangeType.ADDED,
                )
            )
            continue

        if current_unit is None and previous_unit is not None:
            changes.append(
                InventoryChange(
                    compound_key=compound_key,
                    external_unit_id=external_unit_id,
                    change_type=InventoryChangeType.MISSING,
                )
            )
            continue

        if previous_unit is None or current_unit is None:
            continue

        field_changes = _detect_field_changes(
            previous_unit,
            current_unit,
            TRACKED_FIELDS,
        )
        payment_plan_changes = _detect_payment_plan_changes(
            previous_unit.payment_plans,
            current_unit.payment_plans,
        )

        if field_changes or payment_plan_changes:
            changes.append(
                InventoryChange(
                    compound_key=compound_key,
                    external_unit_id=external_unit_id,
                    change_type=InventoryChangeType.UPDATED,
                    changes=field_changes,
                    payment_plan_changes=payment_plan_changes,
                )
            )

    return changes


def _index_units_by_identity(
    units: list[UnitSchema],
    snapshot_name: str,
) -> dict[tuple[str, str], UnitSchema]:
    units_by_identity: dict[tuple[str, str], UnitSchema] = {}
    duplicate_identities: set[tuple[str, str]] = set()

    for unit in units:
        identity = (unit.compound_key, unit.external_unit_id)
        if identity in units_by_identity:
            duplicate_identities.add(identity)
        units_by_identity[identity] = unit

    if duplicate_identities:
        duplicate_list = ", ".join(
            f"({compound_key}, {external_unit_id})"
            for compound_key, external_unit_id in sorted(duplicate_identities)
        )
        raise ValueError(
            "Duplicate (compound_key, external_unit_id) values found in "
            f"{snapshot_name} snapshot: {duplicate_list}"
        )

    return units_by_identity


def _detect_payment_plan_changes(
    previous_plans: list[PaymentPlanSchema],
    current_plans: list[PaymentPlanSchema],
) -> list[PaymentPlanChange]:
    previous_by_key = _index_payment_plans_by_key(previous_plans, "previous")
    current_by_key = _index_payment_plans_by_key(current_plans, "current")

    changes: list[PaymentPlanChange] = []
    all_plan_keys = sorted(set(previous_by_key) | set(current_by_key))

    for plan_key in all_plan_keys:
        previous_plan = previous_by_key.get(plan_key)
        current_plan = current_by_key.get(plan_key)

        if previous_plan is None and current_plan is not None:
            changes.append(
                PaymentPlanChange(
                    plan_key=plan_key,
                    change_type=PaymentPlanChangeType.ADDED,
                )
            )
            continue

        if current_plan is None and previous_plan is not None:
            changes.append(
                PaymentPlanChange(
                    plan_key=plan_key,
                    change_type=PaymentPlanChangeType.MISSING,
                )
            )
            continue

        if previous_plan is None or current_plan is None:
            continue

        field_changes = _detect_field_changes(
            previous_plan,
            current_plan,
            PAYMENT_PLAN_TRACKED_FIELDS,
        )
        if field_changes:
            changes.append(
                PaymentPlanChange(
                    plan_key=plan_key,
                    change_type=PaymentPlanChangeType.UPDATED,
                    changes=field_changes,
                )
            )

    return changes


def _index_payment_plans_by_key(
    plans: list[PaymentPlanSchema],
    snapshot_name: str,
) -> dict[str, PaymentPlanSchema]:
    plans_by_key: dict[str, PaymentPlanSchema] = {}
    duplicate_plan_keys: set[str] = set()

    for plan in plans:
        if plan.plan_key in plans_by_key:
            duplicate_plan_keys.add(plan.plan_key)
        plans_by_key[plan.plan_key] = plan

    if duplicate_plan_keys:
        duplicate_list = ", ".join(sorted(duplicate_plan_keys))
        raise ValueError(
            "Duplicate payment plan keys found in "
            f"{snapshot_name} snapshot: {duplicate_list}"
        )

    return plans_by_key


def _detect_field_changes(
    previous_model: BaseModel,
    current_model: BaseModel,
    tracked_fields: tuple[str, ...],
) -> dict[str, FieldChange]:
    changes: dict[str, FieldChange] = {}

    for field_name in tracked_fields:
        old_value = getattr(previous_model, field_name)
        new_value = getattr(current_model, field_name)

        if old_value != new_value:
            changes[field_name] = FieldChange(
                old=_readable_value(old_value),
                new=_readable_value(new_value),
            )

    return changes


def _readable_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, date | datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value
