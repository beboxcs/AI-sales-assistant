from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.domain.enums import Currency, PaymentMethod, UnitStatus


class PaymentPlanSchema(BaseModel):
    plan_key: str
    source_plan_id: str | None = None
    plan_name: str | None = None
    payment_method: PaymentMethod
    down_payment: Decimal | None = Field(default=None, ge=0)
    installment_years: Decimal | None = Field(default=None, gt=0)
    total_price: Decimal | None = Field(default=None, gt=0)
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
    compound_key: str
    compound_name: str
    location: str
    property_type: str
    base_price: Decimal | None = Field(default=None, gt=0)
    base_currency: Currency = Currency.EGP
    status: UnitStatus

    bedrooms: int | None = Field(default=None, ge=0)
    bathrooms: int | None = Field(default=None, ge=0)
    area_sqm: Decimal | None = Field(default=None, gt=0)
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

    @field_validator("compound_key")
    @classmethod
    def compound_key_must_not_be_empty(cls, value: str) -> str:
        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("compound_key must not be empty")
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
    def compound_unit_id_pairs_must_be_unique(self) -> "InventorySnapshot":
        unit_id_pairs = [
            (unit.compound_key, unit.external_unit_id)
            for unit in self.units
        ]
        duplicate_pairs = sorted(
            pair
            for pair in set(unit_id_pairs)
            if unit_id_pairs.count(pair) > 1
        )
        if duplicate_pairs:
            duplicate_values = ", ".join(
                f"({compound_key}, {external_unit_id})"
                for compound_key, external_unit_id in duplicate_pairs
            )
            raise ValueError(
                "Duplicate (compound_key, external_unit_id) values found in "
                f"snapshot {self.scope_key}: {duplicate_values}"
            )
        return self
