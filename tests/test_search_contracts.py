from dataclasses import FrozenInstanceError
from datetime import date
from decimal import Decimal

import pytest

from app.domain.enums import Currency, PaymentMethod, UnitStatus
from app.search.contracts import (
    PaymentPlanCandidate,
    SearchExecutionContext,
    SearchExecutionResult,
    UnitCandidate,
)
from app.search.exceptions import (
    InvalidSearchScopeError,
    SearchExecutionError,
    SearchExecutionFailedError,
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
    assert issubclass(SearchExecutionFailedError, SearchExecutionError)
    assert issubclass(SearchExecutionError, Exception)
