from dataclasses import FrozenInstanceError
from datetime import date, datetime
from decimal import Decimal

import pytest

from app.ai_engine.candidate_evaluation import (
    CandidateContractViolationError,
    evaluate_candidate,
)
from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_planner import (
    AdjustmentAction,
    FieldAdjustment,
    SearchAttempt,
)
from app.ai_engine.search_strategy import (
    ConstraintStrength,
    MatchMode,
    SearchConstraint,
    SearchField,
    SearchStrategy,
)
from app.domain.enums import Currency, PaymentMethod, UnitStatus
from app.search.contracts import PaymentPlanCandidate, PriceBasis, UnitCandidate


def _plan(
    *,
    plan_key: str = "installment-5",
    payment_method: PaymentMethod = PaymentMethod.INSTALLMENT,
    down_payment: Decimal | None = Decimal("1000000"),
    installment_years: Decimal | None = Decimal("5"),
    total_price: Decimal | None = Decimal("8200000"),
) -> PaymentPlanCandidate:
    return PaymentPlanCandidate(
        plan_key=plan_key,
        source_plan_id=None,
        plan_name=None,
        payment_method=payment_method,
        down_payment=down_payment,
        installment_years=installment_years,
        total_price=total_price,
        currency=Currency.EGP,
    )


def _candidate(
    *,
    location: str = "New Cairo",
    property_type: str = "Apartment",
    bedrooms: int | None = 3,
    bathrooms: int | None = 2,
    area_sqm: Decimal | None = Decimal("150"),
    base_price: Decimal | None = Decimal("8000000"),
    payment_plans: tuple[PaymentPlanCandidate, ...] = (),
) -> UnitCandidate:
    return UnitCandidate(
        unit_id=101,
        external_unit_id="A101",
        compound_id=1,
        compound_name="Example Compound",
        location=location,
        property_type=property_type,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        area_sqm=area_sqm,
        base_price=base_price,
        base_currency=Currency.EGP,
        status=UnitStatus.AVAILABLE,
        delivery_date=date(2028, 1, 1),
        source_last_updated=datetime(2026, 9, 5, 9, 0),
        matched_scope_keys=("compound:1",),
        payment_plans=payment_plans,
    )


def _strategy(
    filters: SearchFilters,
    *fields: SearchField,
    strength: ConstraintStrength = ConstraintStrength.SOFT,
) -> SearchStrategy:
    return SearchStrategy(
        filters=filters,
        constraints=[SearchConstraint(field=field, strength=strength) for field in fields],
    )


def _attempt(
    filters: SearchFilters,
    *,
    mode: MatchMode = MatchMode.EXACT,
    adjustments: list[FieldAdjustment] | None = None,
) -> SearchAttempt:
    return SearchAttempt(
        sequence=1 if mode is MatchMode.EXACT else 2,
        mode=mode,
        filters=filters,
        adjustments=adjustments or [],
    )


def _field(result, field: SearchField):
    return next(item for item in result.field_evaluations if item.field is field)


def test_exact_categorical_match_is_recorded():
    strategy = _strategy(
        SearchFilters(location="New Cairo", property_type="Apartment"),
        SearchField.LOCATION,
        SearchField.PROPERTY_TYPE,
        strength=ConstraintStrength.HARD,
    )

    result = evaluate_candidate(
        strategy=strategy,
        attempt=_attempt(strategy.filters),
        candidate=_candidate(),
    )

    assert result.matched_baseline_fields == (
        SearchField.LOCATION,
        SearchField.PROPERTY_TYPE,
    )
    assert all(item.satisfies_attempt for item in result.field_evaluations)


def test_exact_numeric_match_has_zero_deviation():
    strategy = _strategy(
        SearchFilters(bedrooms=3, max_price=8_000_000),
        SearchField.BEDROOMS,
        SearchField.MAX_PRICE,
    )
    result = evaluate_candidate(
        strategy=strategy,
        attempt=_attempt(strategy.filters),
        candidate=_candidate(base_price=Decimal("8000000")),
    )

    assert _field(result, SearchField.MAX_PRICE).deviation.signed_delta == Decimal("0")
    assert _field(result, SearchField.MAX_PRICE).deviation.percentage_delta == Decimal("0")


def test_near_numeric_candidate_preserves_baseline_deviation():
    strategy = _strategy(SearchFilters(max_price=8_000_000), SearchField.MAX_PRICE)
    attempted_filters = SearchFilters(max_price=8_400_000)
    attempt = _attempt(
        attempted_filters,
        mode=MatchMode.NEAR,
        adjustments=[
            FieldAdjustment(
                field=SearchField.MAX_PRICE,
                action=AdjustmentAction.WIDENED,
                original_value=8_000_000,
                adjusted_value=8_400_000,
            )
        ],
    )

    result = evaluate_candidate(
        strategy=strategy,
        attempt=attempt,
        candidate=_candidate(base_price=Decimal("8200000")),
    )
    evaluation = _field(result, SearchField.MAX_PRICE)

    assert evaluation.satisfies_attempt is True
    assert evaluation.matches_baseline is False
    assert evaluation.was_relaxed is True
    assert evaluation.deviation.signed_delta == Decimal("200000")
    assert evaluation.deviation.absolute_delta == Decimal("200000")


def test_dropped_soft_field_is_valid_and_remains_unmet_baseline_preference():
    strategy = _strategy(SearchFilters(property_type="Apartment"), SearchField.PROPERTY_TYPE)
    attempt_filters = SearchFilters(property_type=None)
    attempt = _attempt(
        attempt_filters,
        mode=MatchMode.ALTERNATIVE,
        adjustments=[
            FieldAdjustment(
                field=SearchField.PROPERTY_TYPE,
                action=AdjustmentAction.DROPPED,
                original_value="Apartment",
                adjusted_value=None,
            )
        ],
    )

    result = evaluate_candidate(
        strategy=strategy,
        attempt=attempt,
        candidate=_candidate(property_type="Villa"),
    )
    evaluation = _field(result, SearchField.PROPERTY_TYPE)

    assert evaluation.was_dropped is True
    assert evaluation.satisfies_attempt is True
    assert evaluation.matches_baseline is False
    assert result.unmet_soft_fields == (SearchField.PROPERTY_TYPE,)


def test_hard_constraint_violation_is_rejected():
    strategy = _strategy(
        SearchFilters(location="New Cairo"),
        SearchField.LOCATION,
        strength=ConstraintStrength.HARD,
    )

    with pytest.raises(CandidateContractViolationError, match="location"):
        evaluate_candidate(
            strategy=strategy,
            attempt=_attempt(strategy.filters),
            candidate=_candidate(location="Sheikh Zayed"),
        )


def test_missing_required_candidate_value_is_rejected():
    strategy = _strategy(SearchFilters(min_area_sqm=100), SearchField.MIN_AREA_SQM)

    with pytest.raises(CandidateContractViolationError, match="area"):
        evaluate_candidate(
            strategy=strategy,
            attempt=_attempt(strategy.filters),
            candidate=_candidate(area_sqm=None),
        )


def test_multiple_payment_plans_are_evaluated_individually():
    strategy = _strategy(
        SearchFilters(
            payment_method="Installment",
            max_down_payment=1_000_000,
            min_installment_years=5,
        ),
        SearchField.PAYMENT_METHOD,
        SearchField.MAX_DOWN_PAYMENT,
        SearchField.MIN_INSTALLMENT_YEARS,
    )
    plans = (
        _plan(
            plan_key="installment-5",
            down_payment=Decimal("900000"),
            installment_years=Decimal("5"),
        ),
        _plan(
            plan_key="installment-7",
            down_payment=Decimal("800000"),
            installment_years=Decimal("7"),
        ),
    )

    result = evaluate_candidate(
        strategy=strategy,
        attempt=_attempt(strategy.filters),
        candidate=_candidate(payment_plans=plans),
    )

    assert [item.plan_key for item in result.payment_plan_evaluations] == [
        "installment-5",
        "installment-7",
    ]
    assert result.payment_plan_evaluations[0].matches_payment_method is True
    assert result.payment_plan_evaluations[1].matches_payment_method is True


def test_ineligible_returned_payment_plan_is_a_contract_violation():
    strategy = _strategy(
        SearchFilters(payment_method="Installment", max_down_payment=1_000_000),
        SearchField.PAYMENT_METHOD,
        SearchField.MAX_DOWN_PAYMENT,
    )

    with pytest.raises(CandidateContractViolationError, match="cash"):
        evaluate_candidate(
            strategy=strategy,
            attempt=_attempt(strategy.filters),
            candidate=_candidate(
                payment_plans=(
                    _plan(plan_key="valid-installment", down_payment=Decimal("900000")),
                    _plan(
                        plan_key="cash",
                        payment_method=PaymentMethod.CASH,
                        down_payment=None,
                        installment_years=None,
                    ),
                )
            ),
        )


def test_payment_method_uses_shared_enum_values_directly():
    strategy = _strategy(
        SearchFilters(payment_method=PaymentMethod.INSTALLMENT),
        SearchField.PAYMENT_METHOD,
    )
    result = evaluate_candidate(
        strategy=strategy,
        attempt=_attempt(strategy.filters),
        candidate=_candidate(payment_plans=(_plan(),)),
    )

    assert _field(result, SearchField.PAYMENT_METHOD).satisfies_attempt is True
    assert result.payment_plan_evaluations[0].matches_payment_method is True


def test_payment_plan_deviations_are_recorded_for_down_payment_and_years():
    strategy = _strategy(
        SearchFilters(max_down_payment=1_000_000, min_installment_years=5),
        SearchField.MAX_DOWN_PAYMENT,
        SearchField.MIN_INSTALLMENT_YEARS,
    )
    result = evaluate_candidate(
        strategy=strategy,
        attempt=_attempt(strategy.filters),
        candidate=_candidate(
            payment_plans=(
                _plan(
                    down_payment=Decimal("800000"),
                    installment_years=Decimal("7"),
                ),
            )
        ),
    )
    plan_result = result.payment_plan_evaluations[0]

    assert plan_result.down_payment_deviation.signed_delta == Decimal("-200000")
    assert plan_result.installment_years_deviation.signed_delta == Decimal("2")


def test_nullable_plan_total_price_is_not_substituted_or_rejected_without_price_filter():
    strategy = _strategy(
        SearchFilters(payment_method="Installment"),
        SearchField.PAYMENT_METHOD,
    )
    result = evaluate_candidate(
        strategy=strategy,
        attempt=_attempt(strategy.filters),
        candidate=_candidate(payment_plans=(_plan(total_price=None),)),
    )

    assert result.payment_plan_evaluations[0].min_price_deviation is None
    assert result.payment_plan_evaluations[0].max_price_deviation is None


def test_plan_total_price_has_independent_min_and_max_deviations():
    strategy = _strategy(
        SearchFilters(min_price=7_000_000, max_price=9_000_000),
        SearchField.MIN_PRICE,
        SearchField.MAX_PRICE,
    )
    result = evaluate_candidate(
        strategy=strategy,
        attempt=_attempt(strategy.filters),
        candidate=_candidate(
            payment_plans=(_plan(total_price=Decimal("8000000")),),
        ),
    )
    plan_result = result.payment_plan_evaluations[0]

    assert plan_result.min_price_deviation.signed_delta == Decimal("1000000")
    assert plan_result.max_price_deviation.signed_delta == Decimal("-1000000")


def test_missing_plan_total_price_has_no_min_or_max_deviation():
    strategy = _strategy(
        SearchFilters(min_price=7_000_000, max_price=9_000_000),
        SearchField.MIN_PRICE,
        SearchField.MAX_PRICE,
    )

    result = evaluate_candidate(
        strategy=strategy,
        attempt=_attempt(strategy.filters),
        candidate=_candidate(payment_plans=(_plan(total_price=None),)),
    )

    plan_result = result.payment_plan_evaluations[0]
    assert plan_result.min_price_deviation is None
    assert plan_result.max_price_deviation is None


def test_dropped_plan_price_with_unknown_total_is_unmet_not_a_violation():
    strategy = _strategy(
        SearchFilters(payment_method="Installment", max_price=9_000_000),
        SearchField.PAYMENT_METHOD,
        SearchField.MAX_PRICE,
    )
    attempt_filters = SearchFilters(payment_method="Installment", max_price=None)
    attempt = _attempt(
        attempt_filters,
        mode=MatchMode.ALTERNATIVE,
        adjustments=[
            FieldAdjustment(
                field=SearchField.MAX_PRICE,
                action=AdjustmentAction.DROPPED,
                original_value=9_000_000,
                adjusted_value=None,
            )
        ],
    )

    result = evaluate_candidate(
        strategy=strategy,
        attempt=attempt,
        candidate=_candidate(payment_plans=(_plan(total_price=None),)),
    )

    assert _field(result, SearchField.MAX_PRICE).matches_baseline is False
    assert _field(result, SearchField.MAX_PRICE).satisfies_attempt is True


def test_price_without_payment_plan_filters_uses_unit_base_price():
    strategy = _strategy(SearchFilters(max_price=8_500_000), SearchField.MAX_PRICE)
    result = evaluate_candidate(
        strategy=strategy,
        attempt=_attempt(strategy.filters),
        candidate=_candidate(
            base_price=Decimal("8000000"),
            payment_plans=(_plan(total_price=None),),
        ),
    )

    assert _field(result, SearchField.MAX_PRICE).candidate_value == Decimal("8000000")
    assert _field(result, SearchField.MAX_PRICE).satisfies_attempt is True


def _evaluate_after_payment_method_drop(
    *,
    base_price: Decimal,
    plan_total: Decimal | None,
):
    strategy = _strategy(
        SearchFilters(payment_method="Cash", max_price=Decimal("6000000")),
        SearchField.PAYMENT_METHOD,
        SearchField.MAX_PRICE,
    )
    attempt = _attempt(
        SearchFilters(max_price=Decimal("6000000")),
        mode=MatchMode.ALTERNATIVE,
        adjustments=[
            FieldAdjustment(
                field=SearchField.PAYMENT_METHOD,
                action=AdjustmentAction.DROPPED,
                original_value="Cash",
                adjusted_value=None,
            )
        ],
    )
    return evaluate_candidate(
        strategy=strategy,
        attempt=attempt,
        candidate=_candidate(
            base_price=base_price,
            payment_plans=(_plan(total_price=plan_total),),
        ),
        price_basis=PriceBasis.PLAN_TOTAL_PRICE,
    )


def test_plan_total_price_rejects_over_budget_plan_after_payment_method_drop():
    with pytest.raises(CandidateContractViolationError):
        _evaluate_after_payment_method_drop(
            base_price=Decimal("5000000"),
            plan_total=Decimal("7000000"),
        )


def test_plan_total_price_accepts_qualifying_plan_after_payment_method_drop():
    result = _evaluate_after_payment_method_drop(
        base_price=Decimal("7000000"),
        plan_total=Decimal("5800000"),
    )

    assert _field(result, SearchField.MAX_PRICE).candidate_value == Decimal("5800000")
    assert _field(result, SearchField.MAX_PRICE).satisfies_attempt is True


def test_plan_total_price_never_falls_back_to_base_price_after_method_drop():
    with pytest.raises(CandidateContractViolationError):
        _evaluate_after_payment_method_drop(
            base_price=Decimal("5000000"),
            plan_total=None,
        )


def test_alternative_candidate_is_valid_after_explicit_soft_drop():
    strategy = _strategy(SearchFilters(location="New Cairo"), SearchField.LOCATION)
    attempt_filters = SearchFilters(location=None)
    attempt = _attempt(
        attempt_filters,
        mode=MatchMode.ALTERNATIVE,
        adjustments=[
            FieldAdjustment(
                field=SearchField.LOCATION,
                action=AdjustmentAction.DROPPED,
                original_value="New Cairo",
                adjusted_value=None,
            )
        ],
    )

    result = evaluate_candidate(
        strategy=strategy,
        attempt=attempt,
        candidate=_candidate(location="North Coast"),
    )

    assert result.field_evaluations[0].satisfies_attempt is True


def test_zero_baseline_has_no_percentage_deviation():
    strategy = _strategy(SearchFilters(min_price=0), SearchField.MIN_PRICE)
    result = evaluate_candidate(
        strategy=strategy,
        attempt=_attempt(strategy.filters),
        candidate=_candidate(base_price=Decimal("0")),
    )

    assert _field(result, SearchField.MIN_PRICE).deviation.percentage_delta is None


def test_evaluation_is_deterministic_and_inputs_are_not_mutated():
    strategy = _strategy(
        SearchFilters(location="New Cairo", max_price=8_000_000),
        SearchField.LOCATION,
        SearchField.MAX_PRICE,
    )
    attempt = _attempt(strategy.filters)
    candidate = _candidate()
    strategy_before = strategy.model_dump(mode="json")
    attempt_before = attempt.model_dump(mode="json")
    candidate_before = candidate

    first = evaluate_candidate(strategy=strategy, attempt=attempt, candidate=candidate)
    second = evaluate_candidate(strategy=strategy, attempt=attempt, candidate=candidate)

    assert first == second
    assert strategy.model_dump(mode="json") == strategy_before
    assert attempt.model_dump(mode="json") == attempt_before
    assert candidate == candidate_before


def test_result_contracts_are_frozen():
    strategy = _strategy(SearchFilters(location="New Cairo"), SearchField.LOCATION)
    result = evaluate_candidate(
        strategy=strategy,
        attempt=_attempt(strategy.filters),
        candidate=_candidate(),
    )

    with pytest.raises(FrozenInstanceError):
        result.unit_id = 2
