from dataclasses import FrozenInstanceError
from datetime import date, datetime
from decimal import Decimal

import pytest

from app.ai_engine import search_plan_builder
from app.ai_engine.candidate_evaluation import (
    CandidateContractViolationError,
    CandidateEvaluation,
    FieldEvaluation,
    evaluate_candidate,
)
from app.ai_engine.candidate_ranking import (
    CandidateRankingError,
    RankedCandidate,
    RankingDimension,
    ScoreComponent,
    ScoreComponentKind,
    rank_candidates,
)
from app.ai_engine.recommendation_explanation import (
    ExplanationFactKind,
    RecommendationExplanationError,
    build_recommendation_explanation,
)
from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_plan_builder import build_search_plan
from app.ai_engine.search_planner import (
    AdjustmentAction,
    FieldAdjustment,
)
from app.ai_engine.search_strategy import (
    AlternativeDropRule,
    ConstraintStrength,
    MatchMode,
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


def _candidate_with_search_value(field: SearchField, value: object, **kwargs):
    if field in {SearchField.MIN_PRICE, SearchField.MAX_PRICE}:
        kwargs["base_price"] = value
    elif field in {SearchField.MIN_AREA_SQM, SearchField.MAX_AREA_SQM}:
        kwargs["area_sqm"] = value
    else:
        kwargs[field.value] = value
    return _candidate(**kwargs)


def _strategy(
    filters: SearchFilters,
    *,
    match_mode: MatchMode = MatchMode.EXACT,
    hard_fields: set[SearchField] | None = None,
    relaxation_rules: list[RelaxationRule] | None = None,
    drop_rules: list[AlternativeDropRule] | None = None,
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
        match_mode=match_mode,
        relaxation_rules=relaxation_rules or [],
        alternative_drop_rules=drop_rules or [],
    )


def _evaluate(strategy, candidate, attempt):
    return evaluate_candidate(strategy=strategy, attempt=attempt, candidate=candidate)


def _rank(strategy, evaluations):
    return rank_candidates(strategy=strategy, evaluations=tuple(evaluations))


def _near_attempt(strategy, field, adjusted):
    return next(
        attempt
        for attempt in build_search_plan(strategy).attempts
        if attempt.mode is MatchMode.NEAR
        and any(adjustment.field is field for adjustment in attempt.adjustments)
        and getattr(attempt.filters, field.value) == adjusted
    )


def _explain(strategy, ranked_candidate):
    return build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=ranked_candidate,
    )


def test_search_plan_and_evaluation_are_deterministic():
    strategy = _strategy(
        SearchFilters(max_price=8_000_000),
        match_mode=MatchMode.NEAR,
        relaxation_rules=[
            RelaxationRule(
                field=SearchField.MAX_PRICE,
                max_delta=1_000_000,
                direction=RelaxationDirection.INCREASE,
            )
        ],
    )
    plan_one = build_search_plan(strategy)
    plan_two = build_search_plan(strategy)
    candidate = _candidate(base_price=Decimal("8500000"))
    near = next(attempt for attempt in plan_one.attempts if attempt.mode is MatchMode.NEAR)

    first = _evaluate(strategy, candidate, near)
    second = _evaluate(strategy, candidate, near)

    assert plan_one == plan_two
    assert first == second


def test_full_pipeline_is_deterministic():
    strategy = _strategy(SearchFilters(location="New Cairo"))
    candidate = _candidate()

    def run_pipeline():
        plan = build_search_plan(strategy)
        evaluations = tuple(
            _evaluate(strategy, candidate, attempt) for attempt in plan.attempts
        )
        ranked = _rank(strategy, evaluations)[0]
        return plan, evaluations, ranked, _explain(strategy, ranked)

    assert run_pipeline() == run_pipeline()


def test_broad_search_is_valid_and_has_no_invented_facts():
    strategy = _strategy(SearchFilters())
    plan = build_search_plan(strategy)
    candidate = _candidate(
        base_price=None,
        area_sqm=None,
        bedrooms=None,
        bathrooms=None,
    )
    evaluation = _evaluate(strategy, candidate, plan.attempts[0])
    ranked = _rank(strategy, (evaluation,))[0]
    explanation = _explain(strategy, ranked)

    assert len(plan.attempts) == 1
    assert ranked.score == Decimal("1")
    assert ranked.components == ()
    assert explanation.strengths == ()
    assert explanation.tradeoffs == ()


def test_empty_ranking_input_returns_empty_tuple():
    assert _rank(_strategy(SearchFilters()), ()) == ()


def test_ranking_input_order_does_not_change_documented_tie_breaking():
    strategy = _strategy(SearchFilters())
    evaluations = (
        CandidateEvaluation(3, 2, (), (), (), ()),
        CandidateEvaluation(1, 2, (), (), (), ()),
        CandidateEvaluation(2, 1, (), (), (), ()),
    )

    forward = _rank(strategy, evaluations)
    reverse = _rank(strategy, tuple(reversed(evaluations)))

    assert forward == reverse
    assert [item.evaluation.unit_id for item in forward] == [2, 1, 3]


@pytest.mark.parametrize(
    ("field", "baseline", "adjusted", "candidate_value", "direction"),
    [
        (SearchField.MIN_PRICE, 7_000_000, 6_000_000, 6_000_000, RelaxationDirection.DECREASE),
        (SearchField.MAX_PRICE, 8_000_000, 9_000_000, 9_000_000, RelaxationDirection.INCREASE),
        (SearchField.MIN_AREA_SQM, 100, 80, 80, RelaxationDirection.DECREASE),
        (SearchField.MAX_AREA_SQM, 200, 220, 220, RelaxationDirection.INCREASE),
        (SearchField.BEDROOMS, 3, 5, 5, RelaxationDirection.BOTH),
        (SearchField.BATHROOMS, 2, 3, 3, RelaxationDirection.BOTH),
    ],
)
def test_exact_relaxation_frontier_has_zero_proximity_score(
    field, baseline, adjusted, candidate_value, direction
):
    filters = SearchFilters(**{field.value: baseline})
    strategy = _strategy(
        filters,
        match_mode=MatchMode.NEAR,
        relaxation_rules=[
            RelaxationRule(field=field, max_delta=abs(adjusted - baseline), direction=direction)
        ],
    )
    attempt = _near_attempt(strategy, field, adjusted)
    candidate = _candidate_with_search_value(field, candidate_value)
    ranked = _rank(strategy, (_evaluate(strategy, candidate, attempt),))[0]

    assert ranked.score == Decimal("0")


@pytest.mark.parametrize(
    ("field", "baseline", "candidate_value"),
    [
        (SearchField.MIN_PRICE, 7_000_000, 8_000_000),
        (SearchField.MAX_PRICE, 9_000_000, 8_000_000),
        (SearchField.MIN_AREA_SQM, 100, 150),
        (SearchField.MAX_AREA_SQM, 200, 150),
    ],
)
def test_favorable_side_of_one_sided_bound_scores_one(field, baseline, candidate_value):
    strategy = _strategy(SearchFilters(**{field.value: baseline}))
    ranked = _rank(
        strategy,
        (_evaluate(strategy, _candidate_with_search_value(field, candidate_value), _attempt_exact(strategy)),),
    )[0]

    assert ranked.score == Decimal("1")


def _attempt_exact(strategy):
    return build_search_plan(strategy).attempts[0]


def test_halfway_max_price_relaxation_has_decimal_half_score():
    strategy = _strategy(
        SearchFilters(max_price=8_000_000),
        match_mode=MatchMode.NEAR,
        relaxation_rules=[
            RelaxationRule(
                field=SearchField.MAX_PRICE,
                max_delta=1_000_000,
                direction=RelaxationDirection.INCREASE,
            )
        ],
    )
    attempt = _near_attempt(strategy, SearchField.MAX_PRICE, 9_000_000)
    ranked = _rank(
        strategy,
        (_evaluate(strategy, _candidate(base_price=Decimal("8500000")), attempt),),
    )[0]

    assert ranked.score == Decimal("0.5")
    assert isinstance(ranked.score, Decimal)


def test_percentage_and_combined_relaxation_caps_are_respected():
    percentage_strategy = _strategy(
        SearchFilters(max_price=8_000_000),
        match_mode=MatchMode.NEAR,
        relaxation_rules=[
            RelaxationRule(
                field=SearchField.MAX_PRICE,
                max_percentage=10,
                direction=RelaxationDirection.INCREASE,
            )
        ],
    )
    combined_strategy = _strategy(
        SearchFilters(max_price=8_000_000),
        match_mode=MatchMode.NEAR,
        relaxation_rules=[
            RelaxationRule(
                field=SearchField.MAX_PRICE,
                max_delta=1_000_000,
                max_percentage=5,
                direction=RelaxationDirection.INCREASE,
            )
        ],
    )
    percentage_attempt = _near_attempt(percentage_strategy, SearchField.MAX_PRICE, 8_800_000)
    combined_attempt = _near_attempt(combined_strategy, SearchField.MAX_PRICE, 8_400_000)

    percentage_ranked = _rank(
        percentage_strategy,
        (_evaluate(percentage_strategy, _candidate(base_price=Decimal("8800000")), percentage_attempt),),
    )[0]
    combined_ranked = _rank(
        combined_strategy,
        (_evaluate(combined_strategy, _candidate(base_price=Decimal("8400000")), combined_attempt),),
    )[0]

    assert percentage_ranked.score == Decimal("0")
    assert combined_ranked.score == Decimal("0")


def test_near_candidate_outside_admitted_attempt_is_rejected():
    strategy = _strategy(
        SearchFilters(max_price=8_000_000),
        match_mode=MatchMode.NEAR,
        relaxation_rules=[
            RelaxationRule(
                field=SearchField.MAX_PRICE,
                max_delta=1_000_000,
                direction=RelaxationDirection.INCREASE,
            )
        ],
    )
    attempt = _near_attempt(strategy, SearchField.MAX_PRICE, 9_000_000)

    with pytest.raises(CandidateContractViolationError):
        _evaluate(strategy, _candidate(base_price=Decimal("9500000")), attempt)


def test_exact_near_alternative_end_to_end_facts():
    exact_strategy = _strategy(SearchFilters(max_price=8_000_000))
    exact_ranked = _rank(
        exact_strategy,
        (_evaluate(exact_strategy, _candidate(), _attempt_exact(exact_strategy)),),
    )[0]
    exact_explanation = _explain(exact_strategy, exact_ranked)
    assert exact_ranked.score == Decimal("1")
    assert exact_explanation.strengths[0].message_key == "within_preferred_price"

    near_strategy = _strategy(
        SearchFilters(max_price=8_000_000),
        match_mode=MatchMode.NEAR,
        relaxation_rules=[
            RelaxationRule(
                field=SearchField.MAX_PRICE,
                max_delta=1_000_000,
                direction=RelaxationDirection.INCREASE,
            )
        ],
    )
    near_attempt = _near_attempt(near_strategy, SearchField.MAX_PRICE, 9_000_000)
    near_ranked = _rank(
        near_strategy,
        (_evaluate(near_strategy, _candidate(base_price=Decimal("8500000")), near_attempt),),
    )[0]
    near_explanation = _explain(near_strategy, near_ranked)
    assert Decimal("0") < near_ranked.score < Decimal("1")
    assert near_explanation.tradeoffs[0].message_key == "price_relaxed"

    alternative_strategy = _strategy(
        SearchFilters(location="New Cairo"),
        match_mode=MatchMode.ALTERNATIVE,
        drop_rules=[AlternativeDropRule(field=SearchField.LOCATION)],
    )
    alternative_attempt = build_search_plan(alternative_strategy).attempts[1]
    alternative_ranked = _rank(
        alternative_strategy,
        (
            _evaluate(
                alternative_strategy,
                _candidate(location="Sheikh Zayed"),
                alternative_attempt,
            ),
        ),
    )[0]
    alternative_explanation = _explain(alternative_strategy, alternative_ranked)
    assert alternative_explanation.tradeoffs[0].message_key == "location_dropped"


def test_hard_constraints_are_gates_not_ranking_discrimination():
    strategy = _strategy(
        SearchFilters(location="New Cairo"),
        hard_fields={SearchField.LOCATION},
    )
    ranked = _rank(
        strategy,
        (_evaluate(strategy, _candidate(), _attempt_exact(strategy)),),
    )[0]
    explanation = _explain(strategy, ranked)

    assert ranked.score == Decimal("1")
    assert ranked.components == ()
    assert explanation.strengths == ()
    assert explanation.tradeoffs == ()


def test_hard_constraint_candidate_violation_is_rejected():
    strategy = _strategy(
        SearchFilters(location="New Cairo"),
        hard_fields={SearchField.LOCATION},
    )

    with pytest.raises(CandidateContractViolationError):
        _evaluate(strategy, _candidate(location="Sheikh Zayed"), _attempt_exact(strategy))


def test_nullable_values_are_safe_when_not_required():
    strategy = _strategy(SearchFilters(location="New Cairo"))
    candidate = _candidate(
        base_price=None,
        area_sqm=None,
        bedrooms=None,
        bathrooms=None,
        payment_plans=(_plan(total_price=None),),
    )

    evaluation = _evaluate(strategy, candidate, _attempt_exact(strategy))

    assert evaluation.field_evaluations[0].candidate_value == "New Cairo"
    assert evaluation.payment_plan_evaluations[0].min_price_deviation is None
    assert evaluation.payment_plan_evaluations[0].max_price_deviation is None


@pytest.mark.parametrize(
    "filters_and_candidate",
    [
        (SearchFilters(min_price=1), _candidate(base_price=None)),
        (SearchFilters(min_area_sqm=1), _candidate(area_sqm=None)),
        (SearchFilters(bedrooms=1), _candidate(bedrooms=None)),
        (SearchFilters(bathrooms=1), _candidate(bathrooms=None)),
    ],
)
def test_required_nullable_values_fail_evaluation(filters_and_candidate):
    filters, candidate = filters_and_candidate
    strategy = _strategy(filters)

    with pytest.raises(CandidateContractViolationError):
        _evaluate(strategy, candidate, _attempt_exact(strategy))


def test_required_plan_total_does_not_fall_back_to_base_price():
    strategy = _strategy(SearchFilters(payment_method="Installment", max_price=9_000_000))
    candidate = _candidate(
        base_price=Decimal("8000000"),
        payment_plans=(_plan(total_price=None),),
    )

    with pytest.raises(CandidateContractViolationError):
        _evaluate(strategy, candidate, _attempt_exact(strategy))


def test_unit_price_and_plan_price_semantics_remain_distinct():
    unit_strategy = _strategy(SearchFilters(max_price=9_000_000))
    unit_ranked = _rank(
        unit_strategy,
        (
            _evaluate(
                unit_strategy,
                _candidate(
                    base_price=Decimal("8000000"),
                    payment_plans=(_plan(total_price=Decimal("10000000")),),
                ),
                _attempt_exact(unit_strategy),
            ),
        ),
    )[0]
    assert unit_ranked.score == Decimal("1")

    plan_strategy = _strategy(
        SearchFilters(payment_method="Installment", max_price=9_000_000)
    )
    plan_ranked = _rank(
        plan_strategy,
        (
            _evaluate(
                plan_strategy,
                _candidate(
                    base_price=Decimal("1000000"),
                    payment_plans=(_plan(total_price=Decimal("8000000")),),
                ),
                _attempt_exact(plan_strategy),
            ),
        ),
    )[0]
    assert any(component.dimension is RankingDimension.PRICE for component in plan_ranked.components)


def test_best_plan_and_equal_plan_tie_are_deterministic():
    strategy = _strategy(SearchFilters(payment_method="Installment"))
    candidate = _candidate(
        payment_plans=(
            _plan(plan_key="z-plan"),
            _plan(plan_key="a-plan"),
        )
    )
    first = _rank(strategy, (_evaluate(strategy, candidate, _attempt_exact(strategy)),))[0]
    second = _rank(strategy, (_evaluate(strategy, candidate, _attempt_exact(strategy)),))[0]

    assert first == second
    assert first.best_plan_key == "a-plan"


def test_hard_only_plan_constraints_have_no_best_plan_or_plan_facts():
    strategy = _strategy(
        SearchFilters(payment_method="Installment"),
        hard_fields={SearchField.PAYMENT_METHOD},
    )
    ranked = _rank(
        strategy,
        (
            _evaluate(
                strategy,
                _candidate(payment_plans=(_plan(plan_key="a"), _plan(plan_key="b"))),
                _attempt_exact(strategy),
            ),
        ),
    )[0]
    explanation = _explain(strategy, ranked)

    assert ranked.best_plan_key is None
    assert all(component.kind is not ScoreComponentKind.PAYMENT_PLAN for component in ranked.components)
    assert explanation.strengths == ()


def test_grouped_ranges_have_one_component_and_one_fact():
    strategy = _strategy(SearchFilters(min_price=7_000_000, max_price=9_000_000, min_area_sqm=100, max_area_sqm=200))
    ranked = _rank(
        strategy,
        (_evaluate(strategy, _candidate(), _attempt_exact(strategy)),),
    )[0]
    explanation = _explain(strategy, ranked)

    assert [component.dimension for component in ranked.components] == [
        RankingDimension.PRICE,
        RankingDimension.AREA,
    ]
    assert [fact.dimension for fact in explanation.strengths] == [
        RankingDimension.PRICE,
        RankingDimension.AREA,
    ]


def test_inconsistent_grouped_candidate_values_are_hidden():
    strategy = SearchStrategy(
        filters=SearchFilters(min_price=7_000_000, max_price=9_000_000)
    )
    fields = (
        FieldEvaluation(
            field=SearchField.MIN_PRICE,
            baseline_value=7_000_000,
            attempted_value=7_000_000,
            candidate_value=(Decimal("8000000"), Decimal("9000000")),
            satisfies_attempt=True,
            matches_baseline=True,
            strength=ConstraintStrength.SOFT,
            was_relaxed=False,
            was_dropped=False,
            deviation=None,
        ),
        FieldEvaluation(
            field=SearchField.MAX_PRICE,
            baseline_value=9_000_000,
            attempted_value=9_000_000,
            candidate_value=(Decimal("8000000"), Decimal("8500000")),
            satisfies_attempt=True,
            matches_baseline=True,
            strength=ConstraintStrength.SOFT,
            was_relaxed=False,
            was_dropped=False,
            deviation=None,
        ),
    )
    ranked = RankedCandidate(
        evaluation=CandidateEvaluation(1, 1, fields, (), (), ()),
        score=Decimal("1"),
        rank=1,
        best_plan_key=None,
        components=(
            ScoreComponent(
                dimension=RankingDimension.PRICE,
                kind=ScoreComponentKind.RANGE_PROXIMITY,
                score=Decimal("1"),
                weight=Decimal("1"),
            ),
        ),
    )

    explanation = _explain(strategy, ranked)

    assert explanation.strengths[0].candidate_value is None


def test_malformed_ranking_facts_fail_loudly():
    strategy = SearchStrategy(filters=SearchFilters())
    invalid_ranked = [
        RankedCandidate(CandidateEvaluation(1, 1, (), (), (), ()), Decimal("-0.1"), 1, None, ()),
        RankedCandidate(CandidateEvaluation(1, 1, (), (), (), ()), Decimal("1.1"), 1, None, ()),
        RankedCandidate(CandidateEvaluation(1, 1, (), (), (), ()), Decimal("NaN"), 1, None, ()),
        RankedCandidate(CandidateEvaluation(1, 1, (), (), (), ()), Decimal("1"), 0, None, ()),
    ]

    for ranked in invalid_ranked:
        with pytest.raises(RecommendationExplanationError):
            _explain(strategy, ranked)

    bad_component = ScoreComponent(
        dimension=RankingDimension.LOCATION,
        kind=ScoreComponentKind.FIELD_MATCH,
        score=Decimal("1.1"),
        weight=Decimal("0"),
    )
    with pytest.raises(RecommendationExplanationError):
        _explain(strategy, _manual_ranked_with_component(bad_component))


def _manual_ranked_with_component(component):
    return RankedCandidate(
        evaluation=CandidateEvaluation(1, 1, (), (), (), ()),
        score=Decimal("1"),
        rank=1,
        best_plan_key=None,
        components=(component,),
    )


def test_duplicate_explanation_dimensions_fail_loudly():
    strategy = SearchStrategy(filters=SearchFilters())
    components = (
        ScoreComponent(RankingDimension.LOCATION, ScoreComponentKind.FIELD_MATCH, Decimal("1"), Decimal("1")),
        ScoreComponent(RankingDimension.LOCATION, ScoreComponentKind.FIELD_MATCH, Decimal("0"), Decimal("1")),
    )
    with pytest.raises(RecommendationExplanationError):
        _explain(strategy, _manual_ranked_with_components(components))


def _manual_ranked_with_components(components):
    return RankedCandidate(
        evaluation=CandidateEvaluation(1, 1, (), (), (), ()),
        score=Decimal("1"),
        rank=1,
        best_plan_key=None,
        components=components,
    )


def test_impossible_grouped_range_facts_fail_in_ranking():
    strategy = _strategy(SearchFilters(min_price=7_000_000, max_price=9_000_000))
    fields = (
        FieldEvaluation(SearchField.MIN_PRICE, 7_000_000, 7_000_000, Decimal("6000000"), False, False, ConstraintStrength.SOFT, True, False, None),
        FieldEvaluation(SearchField.MAX_PRICE, 9_000_000, 9_000_000, Decimal("10000000"), False, False, ConstraintStrength.SOFT, True, False, None),
    )
    evaluation = CandidateEvaluation(1, 1, fields, (), (), ())

    with pytest.raises(CandidateRankingError):
        _rank(strategy, (evaluation,))


def test_immutable_pipeline_inputs_are_not_mutated():
    strategy = _strategy(SearchFilters(max_price=8_000_000))
    plan = build_search_plan(strategy)
    attempt = plan.attempts[0]
    candidate = _candidate()
    strategy_before = strategy.model_dump(mode="json")
    attempt_before = attempt.model_dump(mode="json")
    candidate_before = candidate
    evaluation = _evaluate(strategy, candidate, attempt)
    evaluation_before = evaluation
    ranked = _rank(strategy, (evaluation,))[0]
    ranked_before = ranked
    _explain(strategy, ranked)

    assert strategy.model_dump(mode="json") == strategy_before
    assert attempt.model_dump(mode="json") == attempt_before
    assert candidate == candidate_before
    assert evaluation == evaluation_before
    assert ranked == ranked_before


def test_frozen_contract_values_cannot_be_mutated():
    strategy = _strategy(SearchFilters(location="New Cairo"))
    ranked = _rank(strategy, (_evaluate(strategy, _candidate(), _attempt_exact(strategy)),))[0]
    explanation = _explain(strategy, ranked)

    with pytest.raises(FrozenInstanceError):
        explanation.rank = 2


def test_planner_caps_and_attempt_invariants_remain_respected(monkeypatch):
    monkeypatch.setattr(search_plan_builder, "MAX_NEAR_ATTEMPTS", 3)
    strategy = _strategy(
        SearchFilters(bedrooms=3, bathrooms=2, max_price=8_000_000),
        match_mode=MatchMode.NEAR,
        relaxation_rules=[
            RelaxationRule(
                field=SearchField.BEDROOMS,
                max_delta=1,
                direction=RelaxationDirection.BOTH,
                priority=1,
            ),
            RelaxationRule(
                field=SearchField.BATHROOMS,
                max_delta=1,
                direction=RelaxationDirection.BOTH,
                priority=1,
            ),
            RelaxationRule(
                field=SearchField.MAX_PRICE,
                max_delta=1_000_000,
                direction=RelaxationDirection.INCREASE,
                priority=2,
            ),
        ],
    )
    plan = build_search_plan(strategy)

    assert len(plan.attempts) <= 4
    assert plan.attempts[0].sequence == 1
    assert plan.attempts[0].mode is MatchMode.EXACT
    assert [attempt.sequence for attempt in plan.attempts] == list(
        range(1, len(plan.attempts) + 1)
    )
    states = [attempt.filters.model_dump(mode="json") for attempt in plan.attempts]
    assert len(states) == len({str(state) for state in states})


def test_alternative_cap_and_modes_remain_respected(monkeypatch):
    monkeypatch.setattr(search_plan_builder, "MAX_ALTERNATIVE_ATTEMPTS", 2)
    strategy = _strategy(
        SearchFilters(location="New Cairo", property_type="Apartment"),
        match_mode=MatchMode.ALTERNATIVE,
        drop_rules=[
            AlternativeDropRule(field=SearchField.LOCATION, priority=1),
            AlternativeDropRule(field=SearchField.PROPERTY_TYPE, priority=2),
        ],
    )
    plan = build_search_plan(strategy)

    assert len(plan.attempts) <= 3
    assert plan.attempts[0].mode is MatchMode.EXACT
    assert all(
        earlier.sequence < later.sequence
        for earlier, later in zip(plan.attempts, plan.attempts[1:])
    )
    assert all(
        attempt.mode in {MatchMode.EXACT, MatchMode.ALTERNATIVE}
        for attempt in plan.attempts
    )


def test_near_cap_alternative_seeds_only_from_admitted_numeric_states(monkeypatch):
    monkeypatch.setattr(search_plan_builder, "MAX_NEAR_ATTEMPTS", 3)
    monkeypatch.setattr(search_plan_builder, "MAX_ALTERNATIVE_ATTEMPTS", 8)
    strategy = _strategy(
        SearchFilters(location="New Cairo", bedrooms=3, bathrooms=2),
        match_mode=MatchMode.ALTERNATIVE,
        relaxation_rules=[
            RelaxationRule(
                field=SearchField.BEDROOMS,
                max_delta=1,
                direction=RelaxationDirection.BOTH,
            ),
            RelaxationRule(
                field=SearchField.BATHROOMS,
                max_delta=1,
                direction=RelaxationDirection.BOTH,
            ),
        ],
        drop_rules=[AlternativeDropRule(field=SearchField.LOCATION)],
    )

    first_plan = build_search_plan(strategy)
    second_plan = build_search_plan(strategy)
    near_attempts = [
        attempt for attempt in first_plan.attempts if attempt.mode is MatchMode.NEAR
    ]
    alternative_attempts = [
        attempt
        for attempt in first_plan.attempts
        if attempt.mode is MatchMode.ALTERNATIVE
    ]
    near_states = {
        (attempt.filters.bedrooms, attempt.filters.bathrooms)
        for attempt in near_attempts
    }
    alternative_states = {
        (attempt.filters.bedrooms, attempt.filters.bathrooms)
        for attempt in alternative_attempts
    }

    assert first_plan == second_plan
    assert first_plan.attempts[0].mode is MatchMode.EXACT
    assert len(near_attempts) <= 3
    assert alternative_states == near_states
    assert (3, 3) not in alternative_states
    assert [attempt.sequence for attempt in first_plan.attempts] == list(
        range(1, len(first_plan.attempts) + 1)
    )


@pytest.mark.parametrize(
    (
        "field",
        "baseline",
        "frontier",
        "dimension",
        "message_key",
        "plan_kwargs",
    ),
    [
        (
            SearchField.MAX_DOWN_PAYMENT,
            1_000_000,
            1_500_000,
            RankingDimension.DOWN_PAYMENT,
            "down_payment_not_matched",
            {"down_payment": Decimal("1500000")},
        ),
        (
            SearchField.MIN_INSTALLMENT_YEARS,
            7,
            4,
            RankingDimension.INSTALLMENT_YEARS,
            "installment_duration_not_matched",
            {"installment_years": Decimal("4")},
        ),
    ],
)
def test_plan_numeric_frontier_evaluates_ranks_and_explains(
    field,
    baseline,
    frontier,
    dimension,
    message_key,
    plan_kwargs,
):
    filters = SearchFilters(payment_method="Installment", **{field.value: baseline})
    direction = (
        RelaxationDirection.INCREASE
        if field is SearchField.MAX_DOWN_PAYMENT
        else RelaxationDirection.DECREASE
    )
    strategy = _strategy(
        filters,
        match_mode=MatchMode.NEAR,
        relaxation_rules=[
            RelaxationRule(field=field, max_delta=abs(frontier - baseline), direction=direction)
        ],
    )
    attempt = _near_attempt(strategy, field, frontier)
    evaluation = _evaluate(
        strategy,
        _candidate(payment_plans=(_plan(**plan_kwargs),)),
        attempt,
    )
    ranked = _rank(strategy, (evaluation,))[0]
    explanation = _explain(strategy, ranked)
    component = next(
        component
        for component in ranked.components
        if component.dimension is dimension
    )
    fact = next(fact for fact in explanation.tradeoffs if fact.dimension is dimension)

    assert evaluation.field_evaluations
    assert component.score == Decimal("0")
    assert fact.message_key == message_key
    assert fact.candidate_value is None


def test_plan_numeric_halfway_down_payment_has_decimal_proximity():
    strategy = _strategy(
        SearchFilters(payment_method="Installment", max_down_payment=1_000_000),
        match_mode=MatchMode.NEAR,
        relaxation_rules=[
            RelaxationRule(
                field=SearchField.MAX_DOWN_PAYMENT,
                max_delta=500_000,
                direction=RelaxationDirection.INCREASE,
            )
        ],
    )
    attempt = _near_attempt(strategy, SearchField.MAX_DOWN_PAYMENT, 1_500_000)
    ranked = _rank(
        strategy,
        (
            _evaluate(
                strategy,
                _candidate(
                    payment_plans=(_plan(down_payment=Decimal("1250000")),)
                ),
                attempt,
            ),
        ),
    )[0]

    component = next(
        component
        for component in ranked.components
        if component.dimension is RankingDimension.DOWN_PAYMENT
    )
    assert component.score == Decimal("0.5")


def test_invalid_weight_is_rejected_independently_of_component_score():
    component = ScoreComponent(
        dimension=RankingDimension.LOCATION,
        kind=ScoreComponentKind.FIELD_MATCH,
        score=Decimal("1"),
        weight=Decimal("0"),
    )

    with pytest.raises(RecommendationExplanationError):
        _explain(
            SearchStrategy(filters=SearchFilters()),
            _manual_ranked_with_component(component),
        )
