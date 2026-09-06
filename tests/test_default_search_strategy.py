from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_strategy import (
    ConstraintStrength,
    MatchMode,
    RelaxationDirection,
    SearchField,
    build_default_search_strategy,
)


def test_empty_filters_use_exact_strategy_without_policy_rules():
    strategy = build_default_search_strategy(SearchFilters())

    assert strategy.match_mode is MatchMode.EXACT
    assert strategy.constraints == []
    assert strategy.relaxation_rules == []
    assert strategy.alternative_drop_rules == []
    assert strategy.ranking_preferences == []


def test_hard_only_filters_are_classified_and_remain_exact():
    strategy = build_default_search_strategy(
        SearchFilters(
            location="New Cairo",
            property_type="Apartment",
            payment_method="Installment",
        )
    )

    assert strategy.match_mode is MatchMode.EXACT
    assert [constraint.field for constraint in strategy.constraints] == [
        SearchField.LOCATION,
        SearchField.PROPERTY_TYPE,
        SearchField.PAYMENT_METHOD,
    ]
    assert all(
        constraint.strength is ConstraintStrength.HARD
        for constraint in strategy.constraints
    )


def test_soft_fields_are_classified_without_rules_for_absent_fields():
    strategy = build_default_search_strategy(
        SearchFilters(bedrooms=3, bathrooms=2, location="New Cairo")
    )

    strengths = {
        constraint.field: constraint.strength
        for constraint in strategy.constraints
    }
    assert strengths == {
        SearchField.LOCATION: ConstraintStrength.HARD,
        SearchField.BEDROOMS: ConstraintStrength.SOFT,
        SearchField.BATHROOMS: ConstraintStrength.SOFT,
    }
    assert [rule.field for rule in strategy.relaxation_rules] == [
        SearchField.BEDROOMS,
        SearchField.BATHROOMS,
    ]


def test_bedrooms_and_bathrooms_use_both_direction_delta_rules():
    strategy = build_default_search_strategy(
        SearchFilters(bedrooms=3, bathrooms=2)
    )

    rules = {rule.field: rule for rule in strategy.relaxation_rules}
    for field in (SearchField.BEDROOMS, SearchField.BATHROOMS):
        assert rules[field].direction is RelaxationDirection.BOTH
        assert rules[field].max_delta == 1
        assert rules[field].max_percentage is None
        assert rules[field].priority == 10


def test_price_rules_use_independent_ten_percent_directions_and_priority():
    strategy = build_default_search_strategy(
        SearchFilters(min_price=5_000_000, max_price=8_000_000)
    )

    rules = {rule.field: rule for rule in strategy.relaxation_rules}
    assert rules[SearchField.MIN_PRICE].direction is RelaxationDirection.DECREASE
    assert rules[SearchField.MAX_PRICE].direction is RelaxationDirection.INCREASE
    assert rules[SearchField.MIN_PRICE].max_percentage == 10
    assert rules[SearchField.MAX_PRICE].max_percentage == 10
    assert rules[SearchField.MIN_PRICE].priority == 50
    assert rules[SearchField.MAX_PRICE].priority == 50


def test_area_rules_use_independent_ten_percent_directions_and_priority():
    strategy = build_default_search_strategy(
        SearchFilters(min_area_sqm=100, max_area_sqm=200)
    )

    rules = {rule.field: rule for rule in strategy.relaxation_rules}
    assert rules[SearchField.MIN_AREA_SQM].direction is RelaxationDirection.DECREASE
    assert rules[SearchField.MAX_AREA_SQM].direction is RelaxationDirection.INCREASE
    assert rules[SearchField.MIN_AREA_SQM].max_percentage == 10
    assert rules[SearchField.MAX_AREA_SQM].max_percentage == 10
    assert rules[SearchField.MIN_AREA_SQM].priority == 20
    assert rules[SearchField.MAX_AREA_SQM].priority == 20


def test_installment_years_and_down_payment_rules_match_policy():
    strategy = build_default_search_strategy(
        SearchFilters(
            min_installment_years=5,
            max_down_payment=1_000_000,
        )
    )

    rules = {rule.field: rule for rule in strategy.relaxation_rules}
    assert rules[SearchField.MIN_INSTALLMENT_YEARS].direction is RelaxationDirection.DECREASE
    assert rules[SearchField.MIN_INSTALLMENT_YEARS].max_delta == 1
    assert rules[SearchField.MIN_INSTALLMENT_YEARS].priority == 30
    assert rules[SearchField.MAX_DOWN_PAYMENT].direction is RelaxationDirection.INCREASE
    assert rules[SearchField.MAX_DOWN_PAYMENT].max_percentage == 10
    assert rules[SearchField.MAX_DOWN_PAYMENT].priority == 40


def test_any_populated_relaxable_field_uses_near_mode():
    strategy = build_default_search_strategy(SearchFilters(max_price=8_000_000))

    assert strategy.match_mode is MatchMode.NEAR


def test_default_policy_has_no_alternatives_or_custom_ranking():
    strategy = build_default_search_strategy(
        SearchFilters(
            location="New Cairo",
            bedrooms=3,
            max_price=8_000_000,
        )
    )

    assert strategy.alternative_drop_rules == []
    assert strategy.ranking_preferences == []


def test_default_strategy_construction_is_deterministic_and_isolated():
    filters = SearchFilters(bedrooms=3, max_price=8_000_000)

    first = build_default_search_strategy(filters)
    second = build_default_search_strategy(filters)

    assert first == second
    assert first is not second
    assert first.filters is not filters
    assert [constraint.field for constraint in first.constraints] == [
        SearchField.BEDROOMS,
        SearchField.MAX_PRICE,
    ]


def test_all_populated_soft_policy_fields_have_independent_rules():
    filters = SearchFilters(
        bedrooms=3,
        bathrooms=2,
        min_price=5_000_000,
        max_price=8_000_000,
        min_area_sqm=100,
        max_area_sqm=200,
        max_down_payment=1_000_000,
        min_installment_years=5,
    )

    strategy = build_default_search_strategy(filters)

    assert [rule.field for rule in strategy.relaxation_rules] == [
        SearchField.BEDROOMS,
        SearchField.BATHROOMS,
        SearchField.MIN_PRICE,
        SearchField.MAX_PRICE,
        SearchField.MIN_AREA_SQM,
        SearchField.MAX_AREA_SQM,
        SearchField.MAX_DOWN_PAYMENT,
        SearchField.MIN_INSTALLMENT_YEARS,
    ]
