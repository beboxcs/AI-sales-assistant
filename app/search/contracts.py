"""Shared data contracts for backend search execution."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum

from app.ai_engine.search_filters import SearchFilters
from app.domain.enums import Currency, PaymentMethod, UnitStatus
from app.search.exceptions import SearchExecutionError


class PriceBasis(str, Enum):
    BASE_PRICE = "BASE_PRICE"
    PLAN_TOTAL_PRICE = "PLAN_TOTAL_PRICE"


def has_payment_plan_constraints(filters: SearchFilters) -> bool:
    """Return whether the current filters constrain payment-plan fields."""
    return any(
        value is not None
        for value in (
            filters.payment_method,
            filters.max_down_payment,
            filters.min_installment_years,
        )
    )


def resolve_price_basis(
    *,
    filters: SearchFilters,
    price_basis: PriceBasis | None,
) -> PriceBasis:
    """Resolve legacy calls and reject inconsistent execution metadata."""
    has_plan_constraints = has_payment_plan_constraints(filters)
    if price_basis is None:
        return (
            PriceBasis.PLAN_TOTAL_PRICE
            if has_plan_constraints
            else PriceBasis.BASE_PRICE
        )

    if not isinstance(price_basis, PriceBasis):
        raise SearchExecutionError("price_basis must be a PriceBasis value")
    if price_basis is PriceBasis.BASE_PRICE and has_plan_constraints:
        raise SearchExecutionError(
            "BASE_PRICE is inconsistent with active payment-plan constraints"
        )
    return price_basis


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
    source_last_updated: date | None
    matched_scope_keys: tuple[str, ...]
    payment_plans: tuple[PaymentPlanCandidate, ...]


@dataclass(frozen=True)
class SearchExecutionResult:
    attempt_sequence: int
    candidates: tuple[UnitCandidate, ...]
