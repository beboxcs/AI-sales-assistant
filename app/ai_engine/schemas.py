from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class UnitStatus(str, Enum):
    AVAILABLE = "Available"
    HOLD = "Hold"
    RESERVED = "Reserved"
    SOLD = "Sold"


class PaymentMethod(str, Enum):
    CASH = "Cash"
    INSTALLMENT = "Installment"


class Currency(str, Enum):
    EGP = "EGP"


class PaymentPlanSchema(BaseModel):
    plan_key: str
    source_plan_id: str | None = None
    plan_name: str | None = None
    payment_method: PaymentMethod
    down_payment: float | None = Field(default=None, ge=0)
    installment_years: float | None = Field(default=None, gt=0)
    total_price: float | None = Field(default=None, gt=0)
    currency: Currency = Currency.EGP

    @field_validator("plan_key")
    @classmethod
    def plan_key_must_not_be_empty(cls, value: str) -> str:
        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("plan_key must not be empty")
        return stripped_value

    @model_validator(mode="after")
    def cash_plan_must_not_have_installment_years(self) -> "PaymentPlanSchema":
        if (
            self.payment_method == PaymentMethod.CASH
            and self.installment_years is not None
        ):
            raise ValueError("Cash payment plans cannot have installment_years")
        return self


class UnitSchema(BaseModel):
    """Schema for a real-estate inventory unit."""

    external_unit_id: str
    compound_name: str
    location: str
    property_type: str
    base_price: float | None = Field(default=None, gt=0)
    base_currency: Currency = Currency.EGP
    status: UnitStatus

    bedrooms: int | None = Field(default=None, ge=0)
    bathrooms: int | None = Field(default=None, ge=0)
    area_sqm: float | None = Field(default=None, gt=0)
    delivery_date: date | None = None
    payment_plans: list[PaymentPlanSchema] = Field(default_factory=list)

    source: str
    last_updated: date

    @field_validator("external_unit_id")
    @classmethod
    def external_unit_id_must_not_be_empty(cls, value: str) -> str:
        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("external_unit_id must not be empty")
        return stripped_value

    @model_validator(mode="after")
    def payment_plan_keys_must_be_unique(self) -> "UnitSchema":
        plan_keys = [plan.plan_key for plan in self.payment_plans]
        duplicate_keys = sorted(
            plan_key
            for plan_key in set(plan_keys)
            if plan_keys.count(plan_key) > 1
        )
        if duplicate_keys:
            raise ValueError(
                "Duplicate payment plan keys found for unit "
                f"{self.external_unit_id}: {', '.join(duplicate_keys)}"
            )
        return self


class InventorySnapshot(BaseModel):
    scope_key: str
    units: list[UnitSchema]

    @field_validator("scope_key")
    @classmethod
    def scope_key_must_not_be_empty(cls, value: str) -> str:
        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("scope_key must not be empty")
        return stripped_value

    @model_validator(mode="after")
    def external_unit_ids_must_be_unique(self) -> "InventorySnapshot":
        unit_ids = [unit.external_unit_id for unit in self.units]
        duplicate_ids = sorted(
            unit_id
            for unit_id in set(unit_ids)
            if unit_ids.count(unit_id) > 1
        )
        if duplicate_ids:
            raise ValueError(
                "Duplicate external_unit_id values found in snapshot "
                f"{self.scope_key}: {', '.join(duplicate_ids)}"
            )
        return self
