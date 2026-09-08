import copy
from decimal import Decimal

import pytest

from app.ai_engine import search_plan_builder
from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_plan_builder import build_search_plan
from app.ai_engine.search_strategy import NUMERIC_RELAXABLE_FIELDS
from app.ai_engine.search_planner import AdjustmentAction
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


def _strategy(filters, rules, mode=MatchMode.NEAR, drop_rules=None):
    drop_rules = drop_rules or []
    referenced_fields = {rule.field for rule in rules + drop_rules}
    return SearchStrategy(
        filters=filters,
        match_mode=mode,
        constraints=[
            SearchConstraint(field=field, strength="soft")
            for field in SearchField
            if field in referenced_fields
        ],
        relaxation_rules=rules,
        alternative_drop_rules=drop_rules,
    )


def _rule(field, direction, priority=1, max_delta=None, max_percentage=None):
    return RelaxationRule(
        field=field,
        direction=direction,
        priority=priority,
        max_delta=max_delta,
        max_percentage=max_percentage,
    )


def _drop_rule(field, priority=1):
    return AlternativeDropRule(field=field, priority=priority)


def test_exact_mode_returns_unchanged_exact_only_plan_without_mutating_filters():
    filters = SearchFilters(max_price=8_000_000, bedrooms=3)
    before = copy.deepcopy(filters.model_dump())
    strategy = _strategy(
        filters,
        [_rule(SearchField.MAX_PRICE, "increase", max_delta=500_000)],
        mode=MatchMode.EXACT,
    )

    plan = build_search_plan(strategy)

    assert len(plan.attempts) == 1
    assert plan.attempts[0].mode is MatchMode.EXACT
    assert plan.attempts[0].adjustments == []
    assert plan.attempts[0].filters.model_dump() == before
    assert strategy.filters.model_dump() == before


@pytest.mark.parametrize("mode", [MatchMode.NEAR, MatchMode.ALTERNATIVE])
def test_non_exact_without_numeric_rules_returns_exact_only(mode):
    strategy = SearchStrategy(
        filters=SearchFilters(location="New Cairo"),
        match_mode=mode,
    )

    plan = build_search_plan(strategy)

    assert len(plan.attempts) == 1
    assert plan.attempts[0].mode is MatchMode.EXACT


def test_both_branch_is_preserved_when_a_later_priority_is_added():
    plan = build_search_plan(
        _strategy(
            SearchFilters(bedrooms=3, max_price=8_000_000),
            [
                _rule(SearchField.BEDROOMS, "both", priority=1, max_delta=1),
                _rule(SearchField.MAX_PRICE, "increase", priority=2, max_percentage=5),
            ],
        )
    )

    assert [
        (attempt.filters.bedrooms, attempt.filters.max_price)
        for attempt in plan.attempts
    ] == [
        (3, 8_000_000),
        (2, 8_000_000),
        (4, 8_000_000),
        (2, 8_400_000),
        (4, 8_400_000),
    ]


def test_later_priority_operates_on_every_prior_frontier_variant():
    plan = build_search_plan(
        _strategy(
            SearchFilters(bedrooms=3, bathrooms=2, max_price=8_000_000),
            [
                _rule(SearchField.BEDROOMS, "both", priority=1, max_delta=1),
                _rule(SearchField.BATHROOMS, "both", priority=2, max_delta=1),
                _rule(SearchField.MAX_PRICE, "increase", priority=3, max_delta=400_000),
            ],
        )
    )

    states = [
        (
            attempt.filters.bedrooms,
            attempt.filters.bathrooms,
            attempt.filters.max_price,
        )
        for attempt in plan.attempts[1:]
    ]
    assert states[:2] == [(2, 2, 8_000_000), (4, 2, 8_000_000)]
    assert states[2:] == [
        (2, 1, 8_000_000),
        (2, 3, 8_000_000),
        (4, 1, 8_000_000),
        (4, 3, 8_000_000),
        (2, 1, 8_400_000),
        (2, 3, 8_400_000),
        (4, 1, 8_400_000),
        (4, 3, 8_400_000),
    ]


def test_planner_direction_metadata_covers_numeric_relaxable_fields():
    assert set(search_plan_builder.USEFUL_RELAXATION_DIRECTIONS) == set(
        NUMERIC_RELAXABLE_FIELDS
    )


def test_alternative_mode_generates_a_controlled_soft_drop():
    strategy = _strategy(
        SearchFilters(property_type="Apartment"),
        [],
        mode=MatchMode.ALTERNATIVE,
        drop_rules=[_drop_rule(SearchField.PROPERTY_TYPE)],
    )

    plan = build_search_plan(strategy)

    assert [attempt.mode for attempt in plan.attempts] == [
        MatchMode.EXACT,
        MatchMode.ALTERNATIVE,
    ]
    alternative = plan.attempts[1]
    assert alternative.filters.property_type is None
    assert alternative.adjustments[0].action is AdjustmentAction.DROPPED


@pytest.mark.parametrize("mode", [MatchMode.EXACT, MatchMode.NEAR])
def test_exact_and_near_modes_never_generate_alternatives(mode):
    strategy = _strategy(
        SearchFilters(max_price=8_000_000, payment_method="Installment"),
        [_rule(SearchField.MAX_PRICE, "increase", max_delta=400_000)],
        mode=mode,
        drop_rules=[_drop_rule(SearchField.PAYMENT_METHOD)],
    )

    plan = build_search_plan(strategy)

    assert all(attempt.mode is not MatchMode.ALTERNATIVE for attempt in plan.attempts)
    assert len(plan.attempts) == (1 if mode is MatchMode.EXACT else 2)


def test_alternative_without_drop_rules_preserves_numeric_near_behavior():
    plan = build_search_plan(
        _strategy(
            SearchFilters(max_price=8_000_000),
            [_rule(SearchField.MAX_PRICE, "increase", max_delta=400_000)],
            mode=MatchMode.ALTERNATIVE,
        )
    )

    assert [attempt.mode for attempt in plan.attempts] == [
        MatchMode.EXACT,
        MatchMode.NEAR,
    ]


def test_alternative_drop_tiers_are_cumulative_and_ordered():
    plan = build_search_plan(
        _strategy(
            SearchFilters(
                payment_method="Installment",
                property_type="Apartment",
            ),
            [],
            mode=MatchMode.ALTERNATIVE,
            drop_rules=[
                _drop_rule(SearchField.PROPERTY_TYPE, priority=2),
                _drop_rule(SearchField.PAYMENT_METHOD, priority=1),
            ],
        )
    )

    assert [
        (attempt.filters.payment_method, attempt.filters.property_type)
        for attempt in plan.attempts[1:]
    ] == [(None, "Apartment"), (None, None)]


def test_same_priority_drop_rules_apply_together_and_input_order_is_irrelevant():
    filters = SearchFilters(
        payment_method="Installment",
        property_type="Apartment",
    )
    first = build_search_plan(
        _strategy(
            filters,
            [],
            mode=MatchMode.ALTERNATIVE,
            drop_rules=[
                _drop_rule(SearchField.PROPERTY_TYPE),
                _drop_rule(SearchField.PAYMENT_METHOD),
            ],
        )
    ).model_dump(mode="json")
    second = build_search_plan(
        _strategy(
            filters,
            [],
            mode=MatchMode.ALTERNATIVE,
            drop_rules=[
                _drop_rule(SearchField.PAYMENT_METHOD),
                _drop_rule(SearchField.PROPERTY_TYPE),
            ],
        )
    ).model_dump(mode="json")

    assert first == second
    assert first["attempts"][-1]["filters"]["payment_method"] is None
    assert first["attempts"][-1]["filters"]["property_type"] is None


def test_numeric_near_widening_is_preserved_before_alternative_drop():
    plan = build_search_plan(
        _strategy(
            SearchFilters(max_price=8_000_000, payment_method="Installment"),
            [_rule(SearchField.MAX_PRICE, "increase", max_percentage=5)],
            mode=MatchMode.ALTERNATIVE,
            drop_rules=[_drop_rule(SearchField.PAYMENT_METHOD)],
        )
    )

    alternative = plan.attempts[2]
    changes = {adjustment.field: adjustment for adjustment in alternative.adjustments}
    assert alternative.filters.max_price == 8_400_000
    assert alternative.filters.payment_method is None
    assert changes[SearchField.MAX_PRICE].action is AdjustmentAction.WIDENED
    assert changes[SearchField.PAYMENT_METHOD].action is AdjustmentAction.DROPPED
    assert changes[SearchField.MAX_PRICE].original_value == 8_000_000
    assert changes[SearchField.PAYMENT_METHOD].original_value == "Installment"


def test_multiple_numeric_frontier_seeds_each_receive_alternative_drop():
    plan = build_search_plan(
        _strategy(
            SearchFilters(bedrooms=3, property_type="Apartment"),
            [_rule(SearchField.BEDROOMS, "both", max_delta=1)],
            mode=MatchMode.ALTERNATIVE,
            drop_rules=[_drop_rule(SearchField.PROPERTY_TYPE)],
        )
    )

    assert [
        (attempt.filters.bedrooms, attempt.filters.property_type)
        for attempt in plan.attempts
    ] == [(3, "Apartment"), (2, "Apartment"), (4, "Apartment"), (2, None), (4, None)]


def test_explicit_numeric_drop_is_allowed_but_not_inferred():
    explicit_plan = build_search_plan(
        _strategy(
            SearchFilters(max_price=8_000_000),
            [],
            mode=MatchMode.ALTERNATIVE,
            drop_rules=[_drop_rule(SearchField.MAX_PRICE)],
        )
    )
    implicit_plan = build_search_plan(
        _strategy(
            SearchFilters(max_price=8_000_000),
            [_rule(SearchField.MAX_PRICE, "increase", max_delta=400_000)],
            mode=MatchMode.ALTERNATIVE,
        )
    )

    assert explicit_plan.attempts[1].filters.max_price is None
    assert implicit_plan.attempts[-1].filters.max_price == 8_400_000
    assert all(attempt.mode is not MatchMode.ALTERNATIVE for attempt in implicit_plan.attempts)


def test_invalid_hard_drop_rule_fails_defensively():
    strategy = SearchStrategy.model_construct(
        filters=SearchFilters(location="New Cairo"),
        constraints=[
            SearchConstraint(field=SearchField.LOCATION, strength=ConstraintStrength.HARD)
        ],
        match_mode=MatchMode.ALTERNATIVE,
        ranking_preferences=[],
        relaxation_rules=[],
        alternative_drop_rules=[_drop_rule(SearchField.LOCATION)],
        requested_scope_keys=None,
    )

    with pytest.raises(ValueError, match="HARD"):
        build_search_plan(strategy)


def test_alternative_cap_preserves_earliest_frontier_drop_states(monkeypatch):
    monkeypatch.setattr(search_plan_builder, "MAX_ALTERNATIVE_ATTEMPTS", 4)
    strategy = _strategy(
        SearchFilters(
            bedrooms=3,
            bathrooms=2,
            location="New Cairo",
            property_type="Apartment",
            payment_method="Installment",
        ),
        [
            _rule(SearchField.BEDROOMS, "both", max_delta=1),
            _rule(SearchField.BATHROOMS, "both", max_delta=1),
        ],
        mode=MatchMode.ALTERNATIVE,
        drop_rules=[
            _drop_rule(SearchField.LOCATION, priority=1),
            _drop_rule(SearchField.PROPERTY_TYPE, priority=2),
            _drop_rule(SearchField.PAYMENT_METHOD, priority=3),
        ],
    )

    plan = build_search_plan(strategy)

    assert sum(attempt.mode is MatchMode.ALTERNATIVE for attempt in plan.attempts) == 4
    assert [
        (attempt.filters.bedrooms, attempt.filters.bathrooms, attempt.filters.location)
        for attempt in plan.attempts[-4:]
    ] == [
        (2, 2, None),
        (4, 2, None),
        (3, 1, None),
        (3, 3, None),
    ]


def test_alternative_seeds_only_from_near_states_admitted_before_cap(monkeypatch):
    monkeypatch.setattr(search_plan_builder, "MAX_NEAR_ATTEMPTS", 3)
    strategy = _strategy(
        SearchFilters(
            bedrooms=3,
            bathrooms=2,
            location="New Cairo",
        ),
        [
            _rule(SearchField.BEDROOMS, "both", max_delta=1),
            _rule(SearchField.BATHROOMS, "both", max_delta=1),
        ],
        mode=MatchMode.ALTERNATIVE,
        drop_rules=[_drop_rule(SearchField.LOCATION)],
    )

    plan = build_search_plan(strategy)

    assert [
        (attempt.filters.bedrooms, attempt.filters.bathrooms)
        for attempt in plan.attempts
        if attempt.mode is MatchMode.NEAR
    ] == [(2, 2), (4, 2), (3, 1)]
    assert [
        (attempt.filters.bedrooms, attempt.filters.bathrooms)
        for attempt in plan.attempts
        if attempt.mode is MatchMode.ALTERNATIVE
    ] == [(2, 2), (4, 2), (3, 1)]
    assert (3, 3) not in [
        (attempt.filters.bedrooms, attempt.filters.bathrooms)
        for attempt in plan.attempts
        if attempt.mode is MatchMode.ALTERNATIVE
    ]


def test_invalid_later_numeric_tier_preserves_last_admitted_frontier():
    strategy = _strategy(
        SearchFilters(
            max_price=8_000_000,
            bedrooms=0,
            location="New Cairo",
        ),
        [
            _rule(SearchField.MAX_PRICE, "increase", priority=1, max_percentage=5),
            _rule(SearchField.BEDROOMS, "decrease", priority=2, max_delta=1),
        ],
        mode=MatchMode.ALTERNATIVE,
        drop_rules=[_drop_rule(SearchField.LOCATION)],
    )

    plan = build_search_plan(strategy)

    assert plan.attempts[1].filters.max_price == 8_400_000
    assert plan.attempts[1].filters.bedrooms == 0
    alternative = plan.attempts[2]
    assert alternative.filters.max_price == 8_400_000
    assert alternative.filters.location is None
    changes = {adjustment.field: adjustment for adjustment in alternative.adjustments}
    assert changes[SearchField.MAX_PRICE].action is AdjustmentAction.WIDENED
    assert changes[SearchField.MAX_PRICE].original_value == 8_000_000
    assert changes[SearchField.LOCATION].action is AdjustmentAction.DROPPED
    assert changes[SearchField.LOCATION].original_value == "New Cairo"


def test_numeric_field_can_be_relaxed_then_explicitly_dropped():
    strategy = SearchStrategy(
        filters=SearchFilters(max_price=8_000_000),
        constraints=[SearchConstraint(field=SearchField.MAX_PRICE, strength="soft")],
        relaxation_rules=[
            _rule(SearchField.MAX_PRICE, "increase", max_percentage=5)
        ],
        alternative_drop_rules=[_drop_rule(SearchField.MAX_PRICE)],
        match_mode=MatchMode.ALTERNATIVE,
    )

    plan = build_search_plan(strategy)

    assert plan.attempts[1].filters.max_price == 8_400_000
    alternative = plan.attempts[2]
    assert alternative.filters.max_price is None
    assert alternative.adjustments[0].field is SearchField.MAX_PRICE
    assert alternative.adjustments[0].action is AdjustmentAction.DROPPED
    assert alternative.adjustments[0].original_value == 8_000_000


def test_max_price_delta_and_percentage_relaxations_use_baseline():
    delta_plan = build_search_plan(
        _strategy(
            SearchFilters(max_price=8_000_000),
            [_rule(SearchField.MAX_PRICE, "increase", max_delta=500_000)],
        )
    )
    percentage_plan = build_search_plan(
        _strategy(
            SearchFilters(max_price=8_000_000),
            [_rule(SearchField.MAX_PRICE, "increase", max_percentage=5)],
        )
    )
    both_plan = build_search_plan(
        _strategy(
            SearchFilters(max_price=8_000_000),
            [
                _rule(
                    SearchField.MAX_PRICE,
                    "increase",
                    max_delta=1_000_000,
                    max_percentage=5,
                )
            ],
        )
    )

    assert delta_plan.attempts[1].filters.max_price == 8_500_000
    assert percentage_plan.attempts[1].filters.max_price == 8_400_000
    assert both_plan.attempts[1].filters.max_price == 8_400_000
    assert both_plan.attempts[1].adjustments[0].original_value == 8_000_000
    assert isinstance(percentage_plan.attempts[1].filters.max_price, Decimal)
    assert isinstance(percentage_plan.attempts[1].adjustments[0].adjusted_value, Decimal)


def test_percentage_relaxation_uses_decimal_arithmetic():
    plan = build_search_plan(
        _strategy(
            SearchFilters(max_price=Decimal("0.10")),
            [_rule(SearchField.MAX_PRICE, "increase", max_percentage=10)],
        )
    )

    assert plan.attempts[1].filters.max_price == Decimal("0.11")
    assert plan.attempts[1].adjustments[0].original_value == Decimal("0.10")
    assert plan.attempts[1].adjustments[0].adjusted_value == Decimal("0.11")


def test_priority_tiers_are_cumulative_and_same_priority_fields_apply_together():
    plan = build_search_plan(
        _strategy(
            SearchFilters(max_price=8_000_000, bedrooms=3),
            [
                _rule(SearchField.MAX_PRICE, "increase", priority=1, max_delta=400_000),
                _rule(SearchField.BEDROOMS, "decrease", priority=2, max_delta=1),
            ],
        )
    )
    same_priority_plan = build_search_plan(
        _strategy(
            SearchFilters(max_price=8_000_000, bedrooms=3),
            [
                _rule(SearchField.BEDROOMS, "decrease", max_delta=1),
                _rule(SearchField.MAX_PRICE, "increase", max_delta=400_000),
            ],
        )
    )

    assert plan.attempts[1].filters.max_price == 8_400_000
    assert plan.attempts[1].filters.bedrooms == 3
    assert plan.attempts[2].filters.max_price == 8_400_000
    assert plan.attempts[2].filters.bedrooms == 2
    assert [
        adjustment.field for adjustment in same_priority_plan.attempts[1].adjustments
    ] == [SearchField.BEDROOMS, SearchField.MAX_PRICE]


def test_rule_input_order_does_not_change_output():
    filters = SearchFilters(max_price=8_000_000, bedrooms=3)
    rules = [
        _rule(SearchField.MAX_PRICE, "increase", priority=1, max_delta=400_000),
        _rule(SearchField.BEDROOMS, "decrease", priority=2, max_delta=1),
    ]

    first = build_search_plan(_strategy(filters, rules)).model_dump(mode="json")
    second = build_search_plan(_strategy(filters, list(reversed(rules)))).model_dump(mode="json")

    assert first == second


@pytest.mark.parametrize(
    "field, direction, original, delta, expected",
    [
        (SearchField.MAX_PRICE, "increase", 8_000_000, 400_000, 8_400_000),
        (SearchField.MIN_PRICE, "decrease", 5_000_000, 400_000, 4_600_000),
        (SearchField.MAX_AREA_SQM, "increase", 150, 10, 160),
        (SearchField.MIN_AREA_SQM, "decrease", 150, 10, 140),
        (SearchField.MAX_DOWN_PAYMENT, "increase", 1_000_000, 200_000, 1_200_000),
        (SearchField.MIN_INSTALLMENT_YEARS, "decrease", 7, 1, 6),
    ],
)
def test_one_sided_fields_use_their_useful_direction(
    field, direction, original, delta, expected
):
    values = {
        field.value: original,
    }
    strategy = _strategy(
        SearchFilters(**values),
        [_rule(field, direction, max_delta=delta)],
    )

    plan = build_search_plan(strategy)

    assert getattr(plan.attempts[1].filters, field.value) == expected


@pytest.mark.parametrize(
    "field, direction",
    [
        (SearchField.MAX_PRICE, "decrease"),
        (SearchField.MIN_PRICE, "increase"),
    ],
)
def test_incompatible_one_sided_direction_is_rejected(field, direction):
    with pytest.raises(ValueError, match="direction"):
        build_search_plan(
            _strategy(
                SearchFilters(**{field.value: 8_000_000}),
                [_rule(field, direction, max_delta=100_000)],
            )
        )


def test_both_on_one_sided_field_uses_only_useful_direction():
    plan = build_search_plan(
        _strategy(
            SearchFilters(max_price=8_000_000),
            [_rule(SearchField.MAX_PRICE, "both", max_delta=400_000)],
        )
    )

    assert len(plan.attempts) == 2
    assert plan.attempts[1].filters.max_price == 8_400_000


def test_discrete_fields_support_increase_and_decrease():
    plan = build_search_plan(
        _strategy(
            SearchFilters(bedrooms=3, bathrooms=2),
            [
                _rule(SearchField.BEDROOMS, "increase", max_delta=1),
                _rule(SearchField.BATHROOMS, "decrease", max_delta=1),
            ],
        )
    )

    assert plan.attempts[1].filters.bedrooms == 4
    assert plan.attempts[1].filters.bathrooms == 1


def test_both_count_relaxation_is_decrease_then_increase_in_field_order():
    plan = build_search_plan(
        _strategy(
            SearchFilters(bedrooms=3, bathrooms=2),
            [
                _rule(SearchField.BATHROOMS, "both", max_delta=1),
                _rule(SearchField.BEDROOMS, "both", max_delta=1),
            ],
        )
    )

    assert [
        (attempt.filters.bedrooms, attempt.filters.bathrooms)
        for attempt in plan.attempts[1:]
    ] == [(2, 2), (4, 2), (3, 1), (3, 3)]


def test_negative_discrete_result_is_skipped_without_clamping():
    plan = build_search_plan(
        _strategy(
            SearchFilters(bedrooms=0),
            [_rule(SearchField.BEDROOMS, "decrease", max_delta=1)],
        )
    )

    assert len(plan.attempts) == 1


def test_multiple_both_fields_do_not_create_cartesian_products():
    plan = build_search_plan(
        _strategy(
            SearchFilters(bedrooms=3, bathrooms=2),
            [
                _rule(SearchField.BEDROOMS, "both", max_delta=1),
                _rule(SearchField.BATHROOMS, "both", max_delta=1),
            ],
        )
    )

    assert len(plan.attempts) == 5
    assert all(
        sum(
            getattr(attempt.filters, field.value)
            != getattr(plan.attempts[0].filters, field.value)
            for field in (SearchField.BEDROOMS, SearchField.BATHROOMS)
        )
        == 1
        for attempt in plan.attempts[1:]
    )


def test_alternative_input_generates_near_attempts_only():
    strategy = _strategy(
        SearchFilters(max_price=8_000_000),
        [_rule(SearchField.MAX_PRICE, "increase", max_delta=400_000)],
        mode=MatchMode.ALTERNATIVE,
    )

    plan = build_search_plan(strategy)

    assert [attempt.mode for attempt in plan.attempts] == [
        MatchMode.EXACT,
        MatchMode.NEAR,
    ]
    assert all(
        adjustment.action is AdjustmentAction.WIDENED
        for adjustment in plan.attempts[1].adjustments
    )


def test_attempt_integrity_and_baseline_anchor_are_preserved():
    plan = build_search_plan(
        _strategy(
            SearchFilters(max_price=8_000_000, bedrooms=3),
            [
                _rule(SearchField.MAX_PRICE, "increase", max_delta=400_000),
                _rule(SearchField.BEDROOMS, "decrease", priority=2, max_delta=1),
            ],
        )
    )

    assert plan.attempts[1].adjustments[0].original_value == 8_000_000
    second_adjustments = {
        adjustment.field: adjustment for adjustment in plan.attempts[2].adjustments
    }
    assert second_adjustments[SearchField.MAX_PRICE].original_value == 8_000_000
    assert second_adjustments[SearchField.BEDROOMS].original_value == 3
    for attempt in plan.attempts[1:]:
        for adjustment in attempt.adjustments:
            assert getattr(attempt.filters, adjustment.field.value) == adjustment.adjusted_value


def test_attempt_cap_preserves_earliest_deterministic_attempts(monkeypatch):
    monkeypatch.setattr(search_plan_builder, "MAX_NEAR_ATTEMPTS", 3)
    strategy = _strategy(
        SearchFilters(
            bedrooms=3,
            bathrooms=2,
            max_price=8_000_000,
            min_price=5_000_000,
            max_area_sqm=150,
            min_area_sqm=120,
        ),
        [
            _rule(SearchField.BEDROOMS, "both", max_delta=1),
            _rule(SearchField.BATHROOMS, "both", max_delta=1),
            _rule(SearchField.MAX_PRICE, "increase", max_delta=400_000, priority=2),
            _rule(SearchField.MIN_PRICE, "decrease", max_delta=400_000, priority=3),
        ],
    )

    plan = build_search_plan(strategy)

    assert len(plan.attempts) == 4
    assert [
        (attempt.filters.bedrooms, attempt.filters.bathrooms)
        for attempt in plan.attempts[1:]
    ] == [(2, 2), (4, 2), (3, 1)]


def test_builder_is_deterministic_and_serializable():
    strategy = _strategy(
        SearchFilters(max_price=8_000_000, bedrooms=3),
        [
            _rule(SearchField.BEDROOMS, "both", max_delta=1),
            _rule(SearchField.MAX_PRICE, "increase", max_delta=400_000),
        ],
    )

    first = build_search_plan(strategy).model_dump(mode="json")
    second = build_search_plan(strategy).model_dump(mode="json")

    assert first == second
