from dataclasses import FrozenInstanceError
from datetime import date, datetime
from decimal import Decimal

import pytest

from app.ai_engine.candidate_evaluation import (
    CandidateEvaluation,
    FieldEvaluation,
    evaluate_candidate,
)
from app.ai_engine.candidate_ranking import (
    CandidateRankingError,
    RankingDimension,
    ScoreComponentKind,
    rank_candidates,
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
    RankingPreference,
    RelaxationDirection,
    RelaxationRule,
    SearchConstraint,
    SearchField,
    SearchStrategy,
)
from app.domain.enums import Currency, PaymentMethod, UnitStatus
from app.search.contracts import PaymentPlanCandidate, UnitCandidate


def _plan(
    *,
    plan_key: str = "plan-a",
    payment_method: PaymentMethod = PaymentMethod.INSTALLMENT,
    down_payment: Decimal | None = Decimal("800000"),
    installment_years: Decimal | None = Decimal("7"),
    total_price: Decimal | None = Decimal("8000000"),
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
    unit_id: int = 1,
    location: str = "New Cairo",
    property_type: str = "Apartment",
    bedrooms: int | None = 3,
    bathrooms: int | None = 2,
    area_sqm: Decimal | None = Decimal("150"),
    base_price: Decimal | None = Decimal("8000000"),
    payment_plans: tuple[PaymentPlanCandidate, ...] = (),
) -> UnitCandidate:
    return UnitCandidate(
        unit_id=unit_id,
        external_unit_id=f"A{unit_id}",
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
    *,
    hard_fields: set[SearchField] | None = None,
    ranking_preferences: list[RankingPreference] | None = None,
    relaxation_rules: list[RelaxationRule] | None = None,
) -> SearchStrategy:
    hard_fields = hard_fields or set()
    constraints = [
        SearchConstraint(
            field=field,
            strength=(
                ConstraintStrength.HARD
                if field in hard_fields
                else ConstraintStrength.SOFT
            ),
        )
        for field in SearchField
        if getattr(filters, field.value) is not None
    ]
    return SearchStrategy(
        filters=filters,
        constraints=constraints,
        ranking_preferences=ranking_preferences or [],
        relaxation_rules=relaxation_rules or [],
    )


def _attempt(
    filters: SearchFilters,
    *,
    sequence: int = 1,
    mode: MatchMode = MatchMode.EXACT,
    adjustments: list[FieldAdjustment] | None = None,
) -> SearchAttempt:
    return SearchAttempt(
        sequence=sequence,
        mode=mode,
        filters=filters,
        adjustments=adjustments or [],
    )


def _evaluate(
    strategy: SearchStrategy,
    candidate: UnitCandidate,
    *,
    attempt: SearchAttempt | None = None,
) -> CandidateEvaluation:
    return evaluate_candidate(
        strategy=strategy,
        attempt=attempt or _attempt(strategy.filters),
        candidate=candidate,
    )


def _near_attempt(
    filters: SearchFilters,
    field: SearchField,
    original: object,
    adjusted: object,
) -> SearchAttempt:
    return _attempt(
        filters,
        sequence=2,
        mode=MatchMode.NEAR,
        adjustments=[
            FieldAdjustment(
                field=field,
                action=AdjustmentAction.WIDENED,
                original_value=original,
                adjusted_value=adjusted,
            )
        ],
    )


def _rank(strategy: SearchStrategy, *evaluations: CandidateEvaluation):
    return rank_candidates(strategy=strategy, evaluations=tuple(evaluations))


def test_exact_candidate_ranks_above_near_candidate():
    strategy = _strategy(
        SearchFilters(max_price=8_000_000),
        relaxation_rules=[
            RelaxationRule(
                field=SearchField.MAX_PRICE,
                max_delta=1_000_000,
                direction=RelaxationDirection.INCREASE,
            )
        ],
    )
    near_filters = SearchFilters(max_price=9_000_000)
    exact = _evaluate(strategy, _candidate(unit_id=1))
    near = _evaluate(
        strategy,
        _candidate(unit_id=2, base_price=Decimal("8500000")),
        attempt=_near_attempt(near_filters, SearchField.MAX_PRICE, 8_000_000, 9_000_000),
    )

    ranked = _rank(strategy, near, exact)

    assert [item.evaluation.unit_id for item in ranked] == [1, 2]


def test_smaller_relaxed_deviation_ranks_above_larger_deviation():
    strategy = _strategy(
        SearchFilters(max_price=8_000_000),
        relaxation_rules=[
            RelaxationRule(
                field=SearchField.MAX_PRICE,
                max_delta=1_000_000,
                direction=RelaxationDirection.INCREASE,
            )
        ],
    )
    attempt = _near_attempt(
        SearchFilters(max_price=9_000_000),
        SearchField.MAX_PRICE,
        8_000_000,
        9_000_000,
    )
    closer = _evaluate(strategy, _candidate(unit_id=1, base_price=Decimal("8200000")), attempt=attempt)
    farther = _evaluate(strategy, _candidate(unit_id=2, base_price=Decimal("8800000")), attempt=attempt)

    ranked = _rank(strategy, farther, closer)

    assert [item.evaluation.unit_id for item in ranked] == [1, 2]


@pytest.mark.parametrize(
    ("field", "dimension"),
    [
        (SearchField.BEDROOMS, RankingDimension.BEDROOMS),
        (SearchField.BATHROOMS, RankingDimension.BATHROOMS),
    ],
)
def test_relaxed_count_uses_numeric_proximity(field, dimension):
    filters = SearchFilters(**{field.value: 3})
    strategy = _strategy(
        filters,
        relaxation_rules=[
            RelaxationRule(
                field=field,
                max_delta=2,
                direction=RelaxationDirection.BOTH,
            )
        ],
    )
    attempt_a = _near_attempt(
        SearchFilters(**{field.value: 4}),
        field,
        3,
        4,
    )
    attempt_b = _near_attempt(
        SearchFilters(**{field.value: 5}),
        field,
        3,
        5,
    )
    candidate_a = _candidate(**{field.value: 4}, unit_id=1)
    candidate_b = _candidate(**{field.value: 5}, unit_id=2)

    ranked = _rank(
        strategy,
        _evaluate(strategy, candidate_b, attempt=attempt_b),
        _evaluate(strategy, candidate_a, attempt=attempt_a),
    )

    assert [item.evaluation.unit_id for item in ranked] == [1, 2]
    assert ranked[0].components[0].dimension is dimension
    assert ranked[0].components[0].score > ranked[1].components[0].score


@pytest.mark.parametrize("field", [SearchField.BEDROOMS, SearchField.BATHROOMS])
def test_exact_count_still_gets_full_score(field):
    strategy = _strategy(SearchFilters(**{field.value: 3}))
    result = _rank(
        strategy,
        _evaluate(strategy, _candidate(**{field.value: 3})),
    )[0]

    assert result.components[0].score == Decimal("1")


def test_max_price_under_original_boundary_gets_full_score():
    strategy = _strategy(SearchFilters(max_price=8_000_000))
    result = _rank(strategy, _evaluate(strategy, _candidate()))[0]

    assert result.score == Decimal("1")


def test_price_inside_baseline_range_gets_full_score():
    strategy = _strategy(SearchFilters(min_price=7_000_000, max_price=9_000_000))
    result = _rank(
        strategy,
        _evaluate(strategy, _candidate(base_price=Decimal("8000000"))),
    )[0]

    assert result.score == Decimal("1")
    assert [component.dimension for component in result.components] == [
        RankingDimension.PRICE
    ]


@pytest.mark.parametrize(
    ("base_price", "expected_score"),
    [(Decimal("9300000"), Decimal("0.7")), (Decimal("6800000"), Decimal("0.8"))],
)
def test_price_range_penalizes_only_the_violated_bound(base_price, expected_score):
    strategy = _strategy(
        SearchFilters(min_price=7_000_000, max_price=9_000_000),
        relaxation_rules=[
            RelaxationRule(
                field=SearchField.MIN_PRICE,
                max_delta=1_000_000,
                direction=RelaxationDirection.DECREASE,
            ),
            RelaxationRule(
                field=SearchField.MAX_PRICE,
                max_delta=1_000_000,
                direction=RelaxationDirection.INCREASE,
            ),
        ],
    )
    if base_price > Decimal("9000000"):
        attempt_filters = SearchFilters(min_price=7_000_000, max_price=10_000_000)
        adjustment = FieldAdjustment(
            field=SearchField.MAX_PRICE,
            action=AdjustmentAction.WIDENED,
            original_value=9_000_000,
            adjusted_value=10_000_000,
        )
    else:
        attempt_filters = SearchFilters(min_price=6_000_000, max_price=9_000_000)
        adjustment = FieldAdjustment(
            field=SearchField.MIN_PRICE,
            action=AdjustmentAction.WIDENED,
            original_value=7_000_000,
            adjusted_value=6_000_000,
        )
    attempt = _attempt(
        attempt_filters,
        sequence=2,
        mode=MatchMode.NEAR,
        adjustments=[adjustment],
    )

    result = _rank(
        strategy,
        _evaluate(strategy, _candidate(base_price=base_price), attempt=attempt),
    )[0]

    assert result.score == expected_score


def test_area_range_is_not_double_counted():
    strategy = _strategy(SearchFilters(min_area_sqm=100, max_area_sqm=200))
    result = _rank(strategy, _evaluate(strategy, _candidate(area_sqm=Decimal("150"))))[0]

    assert len(result.components) == 1
    assert result.components[0].dimension is RankingDimension.AREA


def test_soft_categorical_match_beats_mismatch():
    strategy = _strategy(SearchFilters(location="New Cairo"))
    match = _evaluate(strategy, _candidate(unit_id=1, location="New Cairo"))
    mismatch = _evaluate(
        strategy,
        _candidate(unit_id=2, location="Sheikh Zayed"),
        attempt=_attempt(
            SearchFilters(location=None),
            sequence=2,
            mode=MatchMode.ALTERNATIVE,
            adjustments=[
                FieldAdjustment(
                    field=SearchField.LOCATION,
                    action=AdjustmentAction.DROPPED,
                    original_value="New Cairo",
                    adjusted_value=None,
                )
            ],
        ),
    )
    ranked = _rank(strategy, mismatch, match)

    assert [item.evaluation.unit_id for item in ranked] == [1, 2]


def test_hard_fields_do_not_create_ranking_discrimination_components():
    strategy = _strategy(
        SearchFilters(location="New Cairo"),
        hard_fields={SearchField.LOCATION},
    )
    result = _rank(strategy, _evaluate(strategy, _candidate()))[0]

    assert result.components == ()
    assert result.score == Decimal("1")


def test_explicit_ranking_weight_changes_ordering():
    filters = SearchFilters(location="New Cairo", property_type="Apartment")
    strategy = _strategy(
        filters,
        ranking_preferences=[
            RankingPreference(field=SearchField.LOCATION, weight=3),
            RankingPreference(field=SearchField.PROPERTY_TYPE, weight=1),
        ],
    )
    location_match = _evaluate(
        strategy,
        _candidate(unit_id=1, property_type="Villa"),
        attempt=_attempt(
            SearchFilters(location="New Cairo", property_type=None),
            sequence=2,
            mode=MatchMode.ALTERNATIVE,
            adjustments=[
                FieldAdjustment(
                    field=SearchField.PROPERTY_TYPE,
                    action=AdjustmentAction.DROPPED,
                    original_value="Apartment",
                    adjusted_value=None,
                )
            ],
        ),
    )
    property_match = _evaluate(
        strategy,
        _candidate(unit_id=2, location="Sheikh Zayed"),
        attempt=_attempt(
            SearchFilters(location=None, property_type="Apartment"),
            sequence=2,
            mode=MatchMode.ALTERNATIVE,
            adjustments=[
                FieldAdjustment(
                    field=SearchField.LOCATION,
                    action=AdjustmentAction.DROPPED,
                    original_value="New Cairo",
                    adjusted_value=None,
                )
            ],
        ),
    )

    ranked = _rank(strategy, property_match, location_match)

    assert [item.evaluation.unit_id for item in ranked] == [1, 2]


def test_default_weight_is_one():
    strategy = _strategy(SearchFilters(location="New Cairo"))
    result = _rank(strategy, _evaluate(strategy, _candidate()))[0]

    assert result.components[0].weight == Decimal("1")


def test_dropped_soft_mismatch_scores_zero_for_dimension():
    strategy = _strategy(SearchFilters(location="New Cairo"))
    attempt = _attempt(
        SearchFilters(location=None),
        sequence=2,
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
    result = _rank(
        strategy,
        _evaluate(strategy, _candidate(location="Sheikh Zayed"), attempt=attempt),
    )[0]

    assert result.components[0].score == Decimal("0")


@pytest.mark.parametrize(
    ("rule", "candidate_price", "attempt_price", "expected_score"),
    [
        (
            RelaxationRule(
                field=SearchField.MAX_PRICE,
                max_delta=1_000_000,
                direction=RelaxationDirection.INCREASE,
            ),
            Decimal("8500000"),
            9_000_000,
            Decimal("0.5"),
        ),
        (
            RelaxationRule(
                field=SearchField.MAX_PRICE,
                max_percentage=10,
                direction=RelaxationDirection.INCREASE,
            ),
            Decimal("8600000"),
            8_800_000,
            Decimal("0.25"),
        ),
        (
            RelaxationRule(
                field=SearchField.MAX_PRICE,
                max_delta=1_000_000,
                max_percentage=5,
                direction=RelaxationDirection.INCREASE,
            ),
            Decimal("8200000"),
            8_400_000,
            Decimal("0.5"),
        ),
    ],
)
def test_relaxation_rule_normalizes_numeric_proximity(
    rule, candidate_price, attempt_price, expected_score
):
    strategy = _strategy(SearchFilters(max_price=8_000_000), relaxation_rules=[rule])
    result = _rank(
        strategy,
        _evaluate(
            strategy,
            _candidate(base_price=candidate_price),
            attempt=_near_attempt(
                SearchFilters(max_price=attempt_price),
                SearchField.MAX_PRICE,
                8_000_000,
                attempt_price,
            ),
        ),
    )[0]

    assert result.score == expected_score


def test_best_eligible_payment_plan_is_selected():
    strategy = _strategy(
        SearchFilters(payment_method="Installment", max_down_payment=1_000_000),
    )
    result = _rank(
        strategy,
        _evaluate(
            strategy,
            _candidate(
                payment_plans=(
                    _plan(plan_key="z-plan", down_payment=Decimal("900000")),
                    _plan(plan_key="a-plan", down_payment=Decimal("800000")),
                )
            ),
        ),
    )[0]

    assert result.best_plan_key == "a-plan"


def test_payment_plan_tie_uses_plan_key_ascending():
    strategy = _strategy(SearchFilters(payment_method="Installment"))
    result = _rank(
        strategy,
        _evaluate(
            strategy,
            _candidate(
                payment_plans=(
                    _plan(plan_key="z-plan"),
                    _plan(plan_key="a-plan"),
                )
            ),
        ),
    )[0]

    assert result.best_plan_key == "a-plan"


def test_hard_only_payment_plan_constraints_do_not_invent_best_plan():
    strategy = _strategy(
        SearchFilters(payment_method="Installment"),
        hard_fields={SearchField.PAYMENT_METHOD},
    )
    result = _rank(
        strategy,
        _evaluate(
            strategy,
            _candidate(
                payment_plans=(
                    _plan(plan_key="z-plan"),
                    _plan(plan_key="a-plan"),
                )
            ),
        ),
    )[0]

    assert result.best_plan_key is None
    assert all(
        component.kind is not ScoreComponentKind.PAYMENT_PLAN
        for component in result.components
    )


def test_better_down_payment_ranks_better_for_relaxed_plans():
    strategy = _strategy(
        SearchFilters(payment_method="Installment", max_down_payment=1_000_000),
        relaxation_rules=[
            RelaxationRule(
                field=SearchField.MAX_DOWN_PAYMENT,
                max_delta=500_000,
                direction=RelaxationDirection.INCREASE,
            )
        ],
    )
    attempt = _near_attempt(
        SearchFilters(payment_method="Installment", max_down_payment=1_500_000),
        SearchField.MAX_DOWN_PAYMENT,
        1_000_000,
        1_500_000,
    )
    better = _evaluate(
        strategy,
        _candidate(unit_id=1, payment_plans=(_plan(down_payment=Decimal("1100000")),)),
        attempt=attempt,
    )
    worse = _evaluate(
        strategy,
        _candidate(unit_id=2, payment_plans=(_plan(down_payment=Decimal("1400000")),)),
        attempt=attempt,
    )

    assert [item.evaluation.unit_id for item in _rank(strategy, worse, better)] == [1, 2]


def test_better_installment_duration_ranks_better_for_relaxed_plans():
    strategy = _strategy(
        SearchFilters(payment_method="Installment", min_installment_years=7),
        relaxation_rules=[
            RelaxationRule(
                field=SearchField.MIN_INSTALLMENT_YEARS,
                max_delta=3,
                direction=RelaxationDirection.DECREASE,
            )
        ],
    )
    attempt = _near_attempt(
        SearchFilters(payment_method="Installment", min_installment_years=4),
        SearchField.MIN_INSTALLMENT_YEARS,
        7,
        4,
    )
    better = _evaluate(
        strategy,
        _candidate(unit_id=1, payment_plans=(_plan(installment_years=Decimal("6")),)),
        attempt=attempt,
    )
    worse = _evaluate(
        strategy,
        _candidate(unit_id=2, payment_plans=(_plan(installment_years=Decimal("4")),)),
        attempt=attempt,
    )

    assert [item.evaluation.unit_id for item in _rank(strategy, worse, better)] == [1, 2]


def test_nullable_plan_total_price_never_falls_back_to_base_price():
    strategy = _strategy(SearchFilters(payment_method="Installment"))
    result = _rank(
        strategy,
        _evaluate(
            strategy,
            _candidate(
                base_price=Decimal("1000000"),
                payment_plans=(_plan(total_price=None),),
            ),
        ),
    )[0]

    assert all(component.dimension is not RankingDimension.PRICE for component in result.components)


def test_scores_are_bounded_and_decimal_based():
    strategy = _strategy(SearchFilters(max_price=8_000_000))
    result = _rank(strategy, _evaluate(strategy, _candidate()))[0]

    assert Decimal("0") <= result.score <= Decimal("1")
    assert isinstance(result.score, Decimal)
    assert all(isinstance(component.score, Decimal) for component in result.components)


def test_empty_evaluations_return_empty_tuple():
    strategy = _strategy(SearchFilters())

    assert rank_candidates(strategy=strategy, evaluations=()) == ()


def test_order_is_deterministic_regardless_of_input_evaluation_order():
    strategy = _strategy(SearchFilters(location="New Cairo"))
    first = _evaluate(strategy, _candidate(unit_id=1))
    second = _evaluate(strategy, _candidate(unit_id=2, location="New Cairo"))

    assert _rank(strategy, second, first) == _rank(strategy, first, second)


def test_equal_scores_prefer_fewer_unmet_soft_fields_then_attempt_sequence_then_unit_id():
    strategy = _strategy(SearchFilters())

    evaluations = (
        CandidateEvaluation(5, 3, (), (), (), (SearchField.LOCATION,)),
        CandidateEvaluation(2, 2, (), (), (), ()),
        CandidateEvaluation(1, 1, (), (), (), ()),
    )
    ranked = _rank(strategy, *evaluations)

    assert [item.evaluation.unit_id for item in ranked] == [1, 2, 5]


def test_result_contracts_are_frozen():
    strategy = _strategy(SearchFilters())
    result = _rank(
        strategy,
        CandidateEvaluation(1, 1, (), (), (), ()),
    )[0]

    with pytest.raises(FrozenInstanceError):
        result.rank = 2


def test_plan_components_use_payment_plan_kind_without_extra_plan_component():
    strategy = _strategy(SearchFilters(payment_method="Installment"))
    result = _rank(
        strategy,
        _evaluate(strategy, _candidate(payment_plans=(_plan(),))),
    )[0]

    assert [component.kind for component in result.components] == [
        ScoreComponentKind.PAYMENT_PLAN
    ]
    assert all(component.dimension is not RankingDimension.PRICE for component in result.components)


def test_unknown_range_facts_are_rejected_when_both_bounds_are_definitively_violated():
    strategy = _strategy(SearchFilters(min_price=7_000_000, max_price=9_000_000))
    fields = (
        FieldEvaluation(
            field=SearchField.MIN_PRICE,
            baseline_value=7_000_000,
            attempted_value=7_000_000,
            candidate_value=6_000_000,
            satisfies_attempt=False,
            matches_baseline=False,
            strength=ConstraintStrength.SOFT,
            was_relaxed=True,
            was_dropped=False,
            deviation=None,
        ),
        FieldEvaluation(
            field=SearchField.MAX_PRICE,
            baseline_value=9_000_000,
            attempted_value=9_000_000,
            candidate_value=10_000_000,
            satisfies_attempt=False,
            matches_baseline=False,
            strength=ConstraintStrength.SOFT,
            was_relaxed=True,
            was_dropped=False,
            deviation=None,
        ),
    )
    evaluation = CandidateEvaluation(1, 1, fields, (), (), ())

    with pytest.raises(CandidateRankingError):
        rank_candidates(strategy=strategy, evaluations=(evaluation,))
