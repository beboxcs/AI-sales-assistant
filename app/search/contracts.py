"""Shared data contracts for backend search execution."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from app.domain.enums import Currency, PaymentMethod, UnitStatus


@dataclass(frozen=True)
class SearchExecutionContext:
    """Validated execution scope for a company-scoped search.

    ``resolved_scope_keys`` contains canonical backend identifiers already
    resolved under ``company_id``. An empty tuple represents zero executable
    scopes; it never represents all scopes.
    """

    company_id: int
    resolved_scope_keys: tuple[str, ...]


@dataclass(frozen=True)
class PaymentPlanCandidate:
    plan_key: str
    source_plan_id: str | None
    plan_name: str | None
    payment_method: PaymentMethod
    down_payment: Decimal | None
    installment_years: Decimal | None
    total_price: Decimal | None
    currency: Currency


@dataclass(frozen=True)
class UnitCandidate:
    unit_id: int
    external_unit_id: str
    compound_id: int
    compound_name: str
    location: str
    property_type: str
    bedrooms: int | None
    bathrooms: int | None
    area_sqm: Decimal | None
    base_price: Decimal | None
    base_currency: Currency
    status: UnitStatus
    delivery_date: date | None
    source_last_updated: datetime | None
    matched_scope_keys: tuple[str, ...]
    payment_plans: tuple[PaymentPlanCandidate, ...]


@dataclass(frozen=True)
class SearchExecutionResult:
    attempt_sequence: int
    candidates: tuple[UnitCandidate, ...]
