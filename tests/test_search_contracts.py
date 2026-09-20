from dataclasses import FrozenInstanceError
from datetime import date
from decimal import Decimal

import pytest

from app.ai_engine.search_filters import SearchFilters
from app.domain.enums import Currency, PaymentMethod, UnitStatus
from app.search.contracts import (
    PaymentPlanCandidate,
    PriceBasis,
    SearchExecutionContext,
    SearchExecutionResult,
    UnitCandidate,
    resolve_price_basis,
)
from app.search.exceptions import (
    InvalidSearchScopeError,
    SearchExecutionError,
    SearchExecutionFailedError,
    SearchScopeResolutionFailedError,
)


def _payment_plan_candidate(
    total_price: Decimal | None = Decimal("4500000.00"),
) -> PaymentPlanCandidate:
    return PaymentPlanCandidate(
        plan_key="cash",
        source_plan_id=None,
        plan_name="Cash payment",
        payment_method=PaymentMethod.CASH,
        down_payment=None,
        installment_years=None,
        total_price=total_price,
        currency=Currency.EGP,
    )


def _unit_candidate(
    payment_plans: tuple[PaymentPlanCandidate, ...] = (),
) -> UnitCandidate:
    return UnitCandidate(
        unit_id=1,
        external_unit_id="unit-001",
        compound_id=10,
        compound_name="Example Compound",
        location="New Cairo",
        property_type="Apartment",
        bedrooms=3,
        bathrooms=2,
        area_sqm=Decimal("145.50"),
        base_price=Decimal("4500000.00"),
        base_currency=Currency.EGP,
        status=UnitStatus.AVAILABLE,
        delivery_date=date(2028, 6, 1),
        source_last_updated=date(2026, 9, 4),
        matched_scope_keys=("compound:10",),
        payment_plans=payment_plans,
    )


def test_search_contracts_can_be_instantiated_and_nested():
    context = SearchExecutionContext(
        company_id=7,
        resolved_scope_keys=("compound:10",),
    )
    payment_plan = _payment_plan_candidate()
    unit = _unit_candidate(payment_plans=(payment_plan,))
    result = SearchExecutionResult(attempt_sequence=2, candidates=(unit,))

    assert context.company_id == 7
    assert result.candidates[0].payment_plans == (payment_plan,)


@pytest.mark.parametrize(
    ("instance", "field", "value"),
    [
        (SearchExecutionContext(7, ("compound:10",)), "company_id", 8),
        (_payment_plan_candidate(), "currency", "USD"),
        (_unit_candidate(), "status", "sold"),
        (SearchExecutionResult(1, ()), "attempt_sequence", 2),
    ],
)
def test_search_contracts_are_frozen(instance, field, value):
    with pytest.raises(FrozenInstanceError):
        setattr(instance, field, value)


def test_empty_resolved_scope_keys_is_allowed():
    context = SearchExecutionContext(company_id=7, resolved_scope_keys=())

    assert context.resolved_scope_keys == ()


def test_price_basis_has_exactly_the_frozen_values():
    assert list(PriceBasis) == [
        PriceBasis.BASE_PRICE,
        PriceBasis.PLAN_TOTAL_PRICE,
    ]
    assert PriceBasis.BASE_PRICE.value == "BASE_PRICE"
    assert PriceBasis.PLAN_TOTAL_PRICE.value == "PLAN_TOTAL_PRICE"


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        (SearchFilters(), PriceBasis.BASE_PRICE),
        (
            SearchFilters(payment_method="Cash"),
            PriceBasis.PLAN_TOTAL_PRICE,
        ),
        (
            SearchFilters(max_down_payment=500_000),
            PriceBasis.PLAN_TOTAL_PRICE,
        ),
        (
            SearchFilters(min_installment_years=5),
            PriceBasis.PLAN_TOTAL_PRICE,
        ),
    ],
)
def test_omitted_price_basis_is_inferred_from_plan_constraints(
    filters: SearchFilters,
    expected: PriceBasis,
):
    assert resolve_price_basis(filters=filters, price_basis=None) is expected


def test_explicit_plan_total_price_is_valid_without_plan_constraints():
    resolved = resolve_price_basis(
        filters=SearchFilters(max_price=6_000_000),
        price_basis=PriceBasis.PLAN_TOTAL_PRICE,
    )

    assert resolved is PriceBasis.PLAN_TOTAL_PRICE


def test_explicit_base_price_rejects_active_plan_constraints():
    with pytest.raises(
        SearchExecutionError,
        match="BASE_PRICE is inconsistent",
    ):
        resolve_price_basis(
            filters=SearchFilters(payment_method="Cash"),
            price_basis=PriceBasis.BASE_PRICE,
        )


@pytest.mark.parametrize(
    "invalid_price_basis",
    ["PLAN_TOTAL_PRICE", "BASE_PRICE", 1, object()],
)
def test_invalid_explicit_price_basis_type_is_rejected(invalid_price_basis):
    with pytest.raises(
        SearchExecutionError,
        match="price_basis must be a PriceBasis value",
    ):
        resolve_price_basis(
            filters=SearchFilters(),
            price_basis=invalid_price_basis,
        )


def test_payment_plan_candidate_allows_missing_total_price():
    payment_plan = _payment_plan_candidate(total_price=None)

    assert payment_plan.total_price is None


def test_unit_candidate_uses_date_for_source_last_updated():
    unit = _unit_candidate()

    assert unit.source_last_updated == date(2026, 9, 4)
    assert UnitCandidate.__annotations__["source_last_updated"] == date | None


def test_candidates_use_shared_domain_enums_with_preserved_values():
    payment_plan = _payment_plan_candidate()
    unit = _unit_candidate(payment_plans=(payment_plan,))

    assert PaymentPlanCandidate.__annotations__["payment_method"] is PaymentMethod
    assert PaymentPlanCandidate.__annotations__["currency"] is Currency
    assert payment_plan.payment_method is PaymentMethod.CASH
    assert payment_plan.payment_method.value == "Cash"
    assert payment_plan.currency is Currency.EGP
    assert payment_plan.currency.value == "EGP"
    assert UnitCandidate.__annotations__["base_currency"] is Currency
    assert UnitCandidate.__annotations__["status"] is UnitStatus
    assert unit.base_currency is Currency.EGP
    assert unit.status is UnitStatus.AVAILABLE
    assert unit.status.value == "Available"


def test_search_exception_inheritance():
    assert issubclass(InvalidSearchScopeError, SearchExecutionError)
    assert issubclass(SearchScopeResolutionFailedError, SearchExecutionError)
    assert issubclass(SearchExecutionFailedError, SearchExecutionError)
    assert issubclass(SearchExecutionError, Exception)
