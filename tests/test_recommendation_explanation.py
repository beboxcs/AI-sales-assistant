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
from app.ai_engine.search_planner import (
    AdjustmentAction,
    FieldAdjustment,
    SearchAttempt,
)
from app.ai_engine.search_strategy import (
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


def _strategy(
    filters: SearchFilters,
    *,
    hard_fields: set[SearchField] | None = None,
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


def _ranked(
    strategy: SearchStrategy,
    candidate: UnitCandidate,
    *,
    attempt: SearchAttempt | None = None,
) -> RankedCandidate:
    evaluation = _evaluate(strategy, candidate, attempt=attempt)
    return rank_candidates(strategy=strategy, evaluations=(evaluation,))[0]


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


def _manual_ranked(
    *,
    components: tuple[ScoreComponent, ...] = (),
    fields: tuple[FieldEvaluation, ...] = (),
    score: Decimal = Decimal("1"),
    rank: int = 1,
    unit_id: int = 1,
    best_plan_key: str | None = None,
) -> RankedCandidate:
    evaluation = CandidateEvaluation(
        unit_id=unit_id,
        attempt_sequence=1,
        field_evaluations=fields,
        payment_plan_evaluations=(),
        matched_baseline_fields=(),
        unmet_soft_fields=(),
    )
    return RankedCandidate(
        evaluation=evaluation,
        score=score,
        rank=rank,
        best_plan_key=best_plan_key,
        components=components,
    )


def _component(
    dimension: RankingDimension,
    score: Decimal,
    *,
    kind: ScoreComponentKind = ScoreComponentKind.FIELD_MATCH,
) -> ScoreComponent:
    return ScoreComponent(
        dimension=dimension,
        kind=kind,
        score=score,
        weight=Decimal("1"),
    )


def _field(
    field: SearchField,
    baseline: object,
    candidate: object | None,
    *,
    matches: bool,
    relaxed: bool = False,
    dropped: bool = False,
) -> FieldEvaluation:
    return FieldEvaluation(
        field=field,
        baseline_value=baseline,
        attempted_value=baseline,
        candidate_value=candidate,
        satisfies_attempt=True,
        matches_baseline=matches,
        strength=ConstraintStrength.SOFT,
        was_relaxed=relaxed,
        was_dropped=dropped,
        deviation=None,
    )


def test_full_score_location_is_a_strength():
    strategy = _strategy(SearchFilters(location="New Cairo"))
    ranked = _ranked(strategy, _candidate())

    explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=ranked,
    )

    assert explanation.strengths[0].message_key == "matches_location"
    assert explanation.strengths[0].kind is ExplanationFactKind.STRENGTH
    assert explanation.strengths[0].score == Decimal("1")


def test_categorical_mismatch_is_a_tradeoff():
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
    ranked = _ranked(strategy, _candidate(location="Sheikh Zayed"), attempt=attempt)
    explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=ranked,
    )

    assert explanation.tradeoffs[0].message_key == "location_dropped"
    assert explanation.tradeoffs[0].kind is ExplanationFactKind.TRADEOFF


def test_relaxed_price_is_a_tradeoff_and_exact_price_is_a_strength():
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
    relaxed = _ranked(
        strategy,
        _candidate(base_price=Decimal("8500000")),
        attempt=_near_attempt(
            SearchFilters(max_price=9_000_000),
            SearchField.MAX_PRICE,
            8_000_000,
            9_000_000,
        ),
    )
    exact = _ranked(strategy, _candidate(base_price=Decimal("8000000")))

    relaxed_explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=relaxed,
    )
    exact_explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=exact,
    )

    assert relaxed_explanation.tradeoffs[0].message_key == "price_relaxed"
    assert exact_explanation.strengths[0].message_key == "within_preferred_price"


def test_dropped_property_type_uses_drop_specific_key():
    strategy = _strategy(SearchFilters(property_type="Apartment"))
    attempt = _attempt(
        SearchFilters(property_type=None),
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
    )
    explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=_ranked(
            strategy,
            _candidate(property_type="Villa"),
            attempt=attempt,
        ),
    )

    assert explanation.tradeoffs[0].message_key == "property_type_dropped"


@pytest.mark.parametrize(
    ("field", "dimension", "message_key"),
    [
        (SearchField.BEDROOMS, RankingDimension.BEDROOMS, "bedrooms_relaxed"),
        (SearchField.BATHROOMS, RankingDimension.BATHROOMS, "bathrooms_relaxed"),
    ],
)
def test_relaxed_count_uses_relaxed_message_key(field, dimension, message_key):
    strategy = _strategy(
        SearchFilters(**{field.value: 3}),
        relaxation_rules=[
            RelaxationRule(
                field=field,
                max_delta=2,
                direction=RelaxationDirection.BOTH,
            )
        ],
    )
    explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=_ranked(
            strategy,
            _candidate(**{field.value: 4}),
            attempt=_near_attempt(
                SearchFilters(**{field.value: 4}), field, 3, 4
            ),
        ),
    )

    assert explanation.tradeoffs[0].dimension is dimension
    assert explanation.tradeoffs[0].message_key == message_key


def test_grouped_price_emits_one_fact_with_deterministic_bounds():
    strategy = _strategy(SearchFilters(min_price=7_000_000, max_price=9_000_000))
    explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=_ranked(strategy, _candidate(base_price=Decimal("8000000"))),
    )

    facts = explanation.strengths + explanation.tradeoffs
    assert len(facts) == 1
    assert facts[0].dimension is RankingDimension.PRICE
    assert facts[0].baseline_value == (7_000_000.0, 9_000_000.0)
    assert facts[0].candidate_value == Decimal("8000000")


def test_grouped_area_emits_one_fact():
    strategy = _strategy(SearchFilters(min_area_sqm=100, max_area_sqm=200))
    explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=_ranked(strategy, _candidate(area_sqm=Decimal("150"))),
    )

    facts = explanation.strengths + explanation.tradeoffs
    assert len(facts) == 1
    assert facts[0].dimension is RankingDimension.AREA
    assert facts[0].baseline_value == (100.0, 200.0)


def test_payment_method_match_explanation():
    strategy = _strategy(SearchFilters(payment_method="Installment"))
    explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=_ranked(
            strategy,
            _candidate(payment_plans=(_plan(),)),
        ),
    )

    assert explanation.strengths[0].message_key == "matches_payment_method"


def test_best_plan_down_payment_fact_does_not_expose_aggregate_value():
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
    ranked = _ranked(
        strategy,
        _candidate(
            payment_plans=(
                _plan(plan_key="plan-a", down_payment=Decimal("1100000")),
                _plan(plan_key="plan-b", down_payment=Decimal("1400000")),
            )
        ),
        attempt=_near_attempt(
            SearchFilters(payment_method="Installment", max_down_payment=1_500_000),
            SearchField.MAX_DOWN_PAYMENT,
            1_000_000,
            1_500_000,
        ),
    )
    explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=ranked,
    )

    fact = next(
        fact
        for fact in explanation.tradeoffs
        if fact.dimension is RankingDimension.DOWN_PAYMENT
    )
    assert explanation.best_plan_key == "plan-a"
    assert fact.baseline_value == 1_000_000.0
    assert fact.candidate_value is None
    assert fact.message_key == "down_payment_relaxed"
    assert fact.score == Decimal("0.8")


def test_payment_plan_price_fact_does_not_expose_plan_total():
    strategy = _strategy(
        SearchFilters(payment_method="Installment", max_price=9_000_000)
    )
    explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=_ranked(
            strategy,
            _candidate(
                payment_plans=(
                    _plan(plan_key="plan-a", total_price=Decimal("8000000")),
                    _plan(plan_key="plan-b", total_price=Decimal("8500000")),
                )
            ),
        ),
    )

    fact = next(
        fact
        for fact in explanation.strengths
        if fact.dimension is RankingDimension.PRICE
    )
    assert fact.candidate_value is None


def test_unit_level_price_fact_preserves_concrete_candidate_value():
    strategy = _strategy(SearchFilters(max_price=9_000_000))
    explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=_ranked(
            strategy,
            _candidate(base_price=Decimal("8000000")),
        ),
    )

    fact = next(
        fact
        for fact in explanation.strengths
        if fact.dimension is RankingDimension.PRICE
    )
    assert fact.candidate_value == Decimal("8000000")


@pytest.mark.parametrize(
    ("filters", "field", "plan", "attempt_filters", "expected_key"),
    [
        (
            SearchFilters(payment_method="Installment", max_down_payment=1_000_000),
            SearchField.MAX_DOWN_PAYMENT,
            _plan(down_payment=Decimal("800000")),
            SearchFilters(payment_method="Installment", max_down_payment=1_000_000),
            "meets_down_payment_preference",
        ),
        (
            SearchFilters(payment_method="Installment", min_installment_years=7),
            SearchField.MIN_INSTALLMENT_YEARS,
            _plan(installment_years=Decimal("7")),
            SearchFilters(payment_method="Installment", min_installment_years=7),
            "meets_installment_duration",
        ),
    ],
)
def test_plan_numeric_strength_explanation(
    filters, field, plan, attempt_filters, expected_key
):
    strategy = _strategy(filters)
    explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=_ranked(
            strategy,
            _candidate(payment_plans=(plan,)),
            attempt=_attempt(attempt_filters),
        ),
    )

    assert any(fact.message_key == expected_key for fact in explanation.strengths)


def test_best_plan_key_and_none_are_preserved():
    strategy = _strategy(SearchFilters(payment_method="Installment"))
    with_plan = _ranked(strategy, _candidate(payment_plans=(_plan(),)))
    with_plan_explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=with_plan,
    )
    no_plan_explanation = build_recommendation_explanation(
        strategy=SearchStrategy(filters=SearchFilters()),
        ranked_candidate=_manual_ranked(best_plan_key=None),
    )

    assert with_plan_explanation.best_plan_key == "plan-a"
    assert no_plan_explanation.best_plan_key is None


def test_hard_only_candidate_emits_no_facts():
    strategy = _strategy(
        SearchFilters(location="New Cairo"),
        hard_fields={SearchField.LOCATION},
    )
    explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=_ranked(strategy, _candidate()),
    )

    assert explanation.strengths == ()
    assert explanation.tradeoffs == ()


def test_zero_components_emit_empty_facts():
    strategy = SearchStrategy(filters=SearchFilters())
    explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=_manual_ranked(),
    )

    assert explanation.strengths == ()
    assert explanation.tradeoffs == ()


def test_fact_order_follows_component_order():
    strategy = _strategy(SearchFilters(location="New Cairo", property_type="Apartment"))
    ranked = _ranked(strategy, _candidate())
    explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=ranked,
    )

    assert [fact.dimension for fact in explanation.strengths] == [
        RankingDimension.LOCATION,
        RankingDimension.PROPERTY_TYPE,
    ]


def test_duplicate_ranking_dimension_is_rejected():
    strategy = SearchStrategy(filters=SearchFilters())
    ranked = _manual_ranked(
        components=(
            _component(RankingDimension.LOCATION, Decimal("1")),
            _component(RankingDimension.LOCATION, Decimal("0")),
        )
    )

    with pytest.raises(RecommendationExplanationError, match="Duplicate"):
        build_recommendation_explanation(strategy=strategy, ranked_candidate=ranked)


def test_result_dataclasses_are_frozen():
    strategy = _strategy(SearchFilters(location="New Cairo"))
    explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=_ranked(strategy, _candidate()),
    )

    with pytest.raises(FrozenInstanceError):
        explanation.rank = 2
    with pytest.raises(FrozenInstanceError):
        explanation.strengths[0].message_key = "changed"


def test_inputs_are_not_mutated_and_output_is_deterministic():
    strategy = _strategy(SearchFilters(location="New Cairo"))
    ranked = _ranked(strategy, _candidate())
    strategy_before = strategy.model_dump(mode="json")
    ranked_before = ranked

    first = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=ranked,
    )
    second = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=ranked,
    )

    assert first == second
    assert strategy.model_dump(mode="json") == strategy_before
    assert ranked == ranked_before


def test_rank_and_score_are_preserved_exactly():
    strategy = SearchStrategy(filters=SearchFilters())
    ranked = _manual_ranked(score=Decimal("0.375"), rank=7, unit_id=42)
    explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=ranked,
    )

    assert explanation.unit_id == 42
    assert explanation.rank == 7
    assert explanation.score == Decimal("0.375")


@pytest.mark.parametrize(
    "score",
    [Decimal("-0.1"), Decimal("1.1"), Decimal("NaN"), Decimal("Infinity")],
)
def test_invalid_candidate_score_is_rejected(score):
    with pytest.raises(RecommendationExplanationError):
        build_recommendation_explanation(
            strategy=SearchStrategy(filters=SearchFilters()),
            ranked_candidate=_manual_ranked(score=score),
        )


def test_invalid_rank_is_rejected():
    with pytest.raises(RecommendationExplanationError):
        build_recommendation_explanation(
            strategy=SearchStrategy(filters=SearchFilters()),
            ranked_candidate=_manual_ranked(rank=0),
        )


@pytest.mark.parametrize(
    ("score", "weight"),
    [(Decimal("NaN"), Decimal("1")), (Decimal("1"), Decimal("0"))],
)
def test_invalid_component_facts_are_rejected(score, weight):
    component = ScoreComponent(
        dimension=RankingDimension.LOCATION,
        kind=ScoreComponentKind.FIELD_MATCH,
        score=score,
        weight=weight,
    )
    with pytest.raises(RecommendationExplanationError):
        build_recommendation_explanation(
            strategy=SearchStrategy(filters=SearchFilters()),
            ranked_candidate=_manual_ranked(components=(component,)),
        )


def test_grouped_dropped_range_message_is_deterministic():
    strategy = SearchStrategy(
        filters=SearchFilters(min_price=7_000_000, max_price=9_000_000)
    )
    fields = (
        _field(SearchField.MIN_PRICE, 7_000_000, 6_000_000, matches=False, dropped=True),
        _field(SearchField.MAX_PRICE, 9_000_000, 10_000_000, matches=False, dropped=True),
    )
    explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=_manual_ranked(
            components=(_component(RankingDimension.PRICE, Decimal("0")),),
            fields=fields,
        ),
    )

    assert explanation.tradeoffs[0].message_key == "price_dropped"


def test_mixed_dropped_range_uses_generic_message():
    strategy = SearchStrategy(
        filters=SearchFilters(min_price=7_000_000, max_price=9_000_000)
    )
    fields = (
        _field(SearchField.MIN_PRICE, 7_000_000, 6_000_000, matches=False, dropped=True),
        _field(SearchField.MAX_PRICE, 9_000_000, 9_500_000, matches=False),
    )
    explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=_manual_ranked(
            components=(_component(RankingDimension.PRICE, Decimal("0")),),
            fields=fields,
        ),
    )

    assert explanation.tradeoffs[0].message_key == "price_not_matched"


def test_explanation_scores_remain_decimals_and_keys_are_not_natural_language():
    strategy = _strategy(SearchFilters(location="New Cairo"))
    explanation = build_recommendation_explanation(
        strategy=strategy,
        ranked_candidate=_ranked(strategy, _candidate()),
    )

    fact = explanation.strengths[0]
    assert isinstance(fact.score, Decimal)
    assert " " not in fact.message_key
    assert fact.message_key == "matches_location"
