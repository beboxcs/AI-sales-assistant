import copy

import pytest
from pydantic import ValidationError

from app.ai_engine import search_plan_builder
from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_plan_builder import build_search_plan
from app.ai_engine.search_strategy import NUMERIC_RELAXABLE_FIELDS
from app.ai_engine.search_planner import AdjustmentAction
from app.ai_engine.search_strategy import (
    MatchMode,
    RelaxationDirection,
    RelaxationRule,
    SearchConstraint,
    SearchField,
    SearchStrategy,
)


def _strategy(filters, rules, mode=MatchMode.NEAR):
    return SearchStrategy(
        filters=filters,
        match_mode=mode,
        constraints=[
            SearchConstraint(field=rule.field, strength="soft") for rule in rules
        ],
        relaxation_rules=rules,
    )


def _rule(field, direction, priority=1, max_delta=None, max_percentage=None):
    return RelaxationRule(
        field=field,
        direction=direction,
        priority=priority,
        max_delta=max_delta,
        max_percentage=max_percentage,
    )


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
