import json
from decimal import Decimal

import pytest

from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_intent import (
    ParsedSearchIntent,
    SearchAmbiguity,
    SearchFieldDirective,
)
from app.ai_engine.search_intent_builder import build_search_strategy_from_intent
from app.ai_engine.search_strategy import (
    ConstraintStrength,
    MatchMode,
    SearchField,
    build_default_search_strategy,
)


def test_without_directives_preserves_default_strategy():
    filters = SearchFilters(location="New Cairo", bedrooms=3)

    strategy = build_search_strategy_from_intent(
        ParsedSearchIntent(filters=filters)
    )

    assert strategy == build_default_search_strategy(filters)


def test_hard_numeric_override_removes_default_relaxation_and_updates_match_mode():
    strategy = build_search_strategy_from_intent(
        ParsedSearchIntent(
            filters=SearchFilters(max_price=Decimal("5000000")),
            directives=[
                SearchFieldDirective(
                    field="max_price",
                    strength=ConstraintStrength.HARD,
                )
            ],
        )
    )

    assert _constraint_strengths(strategy)[SearchField.MAX_PRICE] is (
        ConstraintStrength.HARD
    )
    assert _relaxation_fields(strategy) == []
    assert strategy.match_mode is MatchMode.EXACT


@pytest.mark.parametrize(
    "field, filters",
    [
        (SearchField.LOCATION, SearchFilters(location="New Cairo")),
        (SearchField.PROPERTY_TYPE, SearchFilters(property_type="Apartment")),
        (SearchField.PAYMENT_METHOD, SearchFilters(payment_method="Cash")),
    ],
)
def test_soft_categorical_override_adds_alternative_drop_rule(field, filters):
    strategy = build_search_strategy_from_intent(
        ParsedSearchIntent(
            filters=filters,
            directives=[
                SearchFieldDirective(
                    field=field,
                    strength=ConstraintStrength.SOFT,
                )
            ],
        )
    )

    assert _constraint_strengths(strategy)[field] is ConstraintStrength.SOFT
    assert _drop_fields(strategy) == [field]
    assert strategy.alternative_drop_rules[0].priority == 1
    assert strategy.ranking_preferences == []
    assert _relaxation_fields(strategy) == []
    assert strategy.match_mode is MatchMode.ALTERNATIVE


def test_no_directive_categorical_preserves_default_hard_exact_behavior():
    strategy = build_search_strategy_from_intent(
        ParsedSearchIntent(filters=SearchFilters(location="New Cairo"))
    )

    assert _constraint_strengths(strategy)[SearchField.LOCATION] is (
        ConstraintStrength.HARD
    )
    assert strategy.alternative_drop_rules == []
    assert strategy.ranking_preferences == []
    assert strategy.match_mode is MatchMode.EXACT


def test_soft_numeric_override_retains_relaxation_without_alternative_drop():
    strategy = build_search_strategy_from_intent(
        ParsedSearchIntent(
            filters=SearchFilters(max_price=Decimal("5000000")),
            directives=[
                SearchFieldDirective(
                    field="max_price",
                    strength=ConstraintStrength.SOFT,
                )
            ],
        )
    )

    assert _constraint_strengths(strategy)[SearchField.MAX_PRICE] is (
        ConstraintStrength.SOFT
    )
    assert _relaxation_fields(strategy) == [SearchField.MAX_PRICE]
    assert strategy.alternative_drop_rules == []
    assert strategy.match_mode is MatchMode.NEAR


def test_preferred_numeric_field_adds_ranking_and_retains_relaxation():
    strategy = build_search_strategy_from_intent(
        ParsedSearchIntent(
            filters=SearchFilters(max_price=Decimal("5000000")),
            directives=[SearchFieldDirective(field="max_price", preferred=True)],
        )
    )

    assert _constraint_strengths(strategy)[SearchField.MAX_PRICE] is (
        ConstraintStrength.SOFT
    )
    assert _ranking_fields(strategy) == [SearchField.MAX_PRICE]
    assert strategy.ranking_preferences[0].weight == 2.0
    assert _relaxation_fields(strategy) == [SearchField.MAX_PRICE]
    assert strategy.alternative_drop_rules == []
    assert strategy.match_mode is MatchMode.NEAR


def test_preferred_categorical_field_adds_ranking_and_alternative_drop():
    strategy = build_search_strategy_from_intent(
        ParsedSearchIntent(
            filters=SearchFilters(location="New Cairo"),
            directives=[SearchFieldDirective(field="location", preferred=True)],
        )
    )

    assert _constraint_strengths(strategy)[SearchField.LOCATION] is (
        ConstraintStrength.SOFT
    )
    assert _ranking_fields(strategy) == [SearchField.LOCATION]
    assert strategy.ranking_preferences[0].weight == 2.0
    assert _drop_fields(strategy) == [SearchField.LOCATION]
    assert strategy.alternative_drop_rules[0].priority == 1
    assert strategy.match_mode is MatchMode.ALTERNATIVE


def test_explicit_allow_alternative_adds_drop_rule_and_alternative_match_mode():
    strategy = build_search_strategy_from_intent(
        ParsedSearchIntent(
            filters=SearchFilters(max_price=Decimal("5000000")),
            directives=[
                SearchFieldDirective(field="max_price", allow_alternative=True)
            ],
        )
    )

    assert _constraint_strengths(strategy)[SearchField.MAX_PRICE] is (
        ConstraintStrength.SOFT
    )
    assert _drop_fields(strategy) == [SearchField.MAX_PRICE]
    assert strategy.alternative_drop_rules[0].priority == 1
    assert strategy.match_mode is MatchMode.ALTERNATIVE


def test_soft_preferred_alternative_directive_preserves_all_semantics_once():
    strategy = build_search_strategy_from_intent(
        ParsedSearchIntent(
            filters=SearchFilters(max_price=Decimal("5000000")),
            directives=[
                SearchFieldDirective(
                    field="max_price",
                    strength=ConstraintStrength.SOFT,
                    preferred=True,
                    allow_alternative=True,
                )
            ],
        )
    )

    assert _constraint_strengths(strategy)[SearchField.MAX_PRICE] is (
        ConstraintStrength.SOFT
    )
    assert _ranking_fields(strategy) == [SearchField.MAX_PRICE]
    assert _drop_fields(strategy) == [SearchField.MAX_PRICE]
    assert _relaxation_fields(strategy) == [SearchField.MAX_PRICE]


def test_soft_preferred_categorical_has_one_ranking_and_one_drop_rule():
    strategy = build_search_strategy_from_intent(
        ParsedSearchIntent(
            filters=SearchFilters(location="New Cairo"),
            directives=[
                SearchFieldDirective(
                    field="location",
                    strength=ConstraintStrength.SOFT,
                    preferred=True,
                )
            ],
        )
    )

    assert _ranking_fields(strategy) == [SearchField.LOCATION]
    assert _drop_fields(strategy) == [SearchField.LOCATION]


def test_soft_allow_alternative_categorical_has_one_drop_rule():
    strategy = build_search_strategy_from_intent(
        ParsedSearchIntent(
            filters=SearchFilters(location="New Cairo"),
            directives=[
                SearchFieldDirective(
                    field="location",
                    strength=ConstraintStrength.SOFT,
                    allow_alternative=True,
                )
            ],
        )
    )

    assert _drop_fields(strategy) == [SearchField.LOCATION]


def test_hard_field_has_no_relaxation_ranking_or_drop_rule():
    strategy = build_search_strategy_from_intent(
        ParsedSearchIntent(
            filters=SearchFilters(bedrooms=3),
            directives=[
                SearchFieldDirective(
                    field="bedrooms",
                    strength=ConstraintStrength.HARD,
                )
            ],
        )
    )

    assert _constraint_strengths(strategy)[SearchField.BEDROOMS] is (
        ConstraintStrength.HARD
    )
    assert strategy.ranking_preferences == []
    assert strategy.alternative_drop_rules == []
    assert strategy.relaxation_rules == []


def test_multiple_directives_keep_canonical_field_order_without_duplicates():
    strategy = build_search_strategy_from_intent(
        ParsedSearchIntent(
            filters=SearchFilters(
                location="New Cairo",
                bedrooms=3,
                max_price=Decimal("5000000"),
                payment_method="Cash",
            ),
            directives=[
                SearchFieldDirective(field="payment_method", preferred=True),
                SearchFieldDirective(
                    field="max_price",
                    preferred=True,
                    allow_alternative=True,
                ),
                SearchFieldDirective(
                    field="bedrooms",
                    strength=ConstraintStrength.HARD,
                ),
                SearchFieldDirective(
                    field="location",
                    strength=ConstraintStrength.SOFT,
                ),
            ],
        )
    )

    assert _constraint_fields(strategy) == [
        SearchField.LOCATION,
        SearchField.BEDROOMS,
        SearchField.MAX_PRICE,
        SearchField.PAYMENT_METHOD,
    ]
    assert _ranking_fields(strategy) == [
        SearchField.MAX_PRICE,
        SearchField.PAYMENT_METHOD,
    ]
    assert _drop_fields(strategy) == [
        SearchField.LOCATION,
        SearchField.MAX_PRICE,
        SearchField.PAYMENT_METHOD,
    ]
    assert _relaxation_fields(strategy) == [SearchField.MAX_PRICE]


def test_ambiguities_and_unsupported_requirements_do_not_change_strategy():
    filters = SearchFilters(location="New Cairo")
    directives = [SearchFieldDirective(field="location", preferred=True)]
    plain_strategy = build_search_strategy_from_intent(
        ParsedSearchIntent(filters=filters, directives=directives)
    )

    annotated_strategy = build_search_strategy_from_intent(
        ParsedSearchIntent(
            filters=filters,
            directives=directives,
            ambiguities=[SearchAmbiguity(field="location", reason="vague area")],
            unsupported_requirements=["sea view"],
        )
    )

    assert annotated_strategy == plain_strategy


def test_builder_does_not_mutate_intent_or_source_filters():
    filters = SearchFilters(max_price=Decimal("5000000"))
    intent = ParsedSearchIntent(
        filters=filters,
        directives=[SearchFieldDirective(field="max_price", preferred=True)],
    )
    before = intent.model_dump()

    strategy = build_search_strategy_from_intent(intent)

    assert intent.model_dump() == before
    assert strategy.filters is not intent.filters
    assert filters.max_price == Decimal("5000000")


def test_decimal_values_remain_decimal_in_strategy_filters():
    strategy = build_search_strategy_from_intent(
        ParsedSearchIntent(
            filters=SearchFilters(
                max_price=Decimal("1234567890123456.78"),
                min_area_sqm=Decimal("123.45"),
            )
        )
    )

    assert strategy.filters.max_price == Decimal("1234567890123456.78")
    assert isinstance(strategy.filters.max_price, Decimal)
    assert strategy.filters.min_area_sqm == Decimal("123.45")
    assert isinstance(strategy.filters.min_area_sqm, Decimal)


def test_built_strategy_json_dump_is_json_safe():
    strategy = build_search_strategy_from_intent(
        ParsedSearchIntent(
            filters=SearchFilters(
                location="New Cairo",
                max_price=Decimal("5000000.50"),
            ),
            directives=[
                SearchFieldDirective(field="location", preferred=True),
                SearchFieldDirective(field="max_price", preferred=True),
            ],
        )
    )

    json.dumps(strategy.model_dump(mode="json"))


def _constraint_fields(strategy):
    return [constraint.field for constraint in strategy.constraints]


def _constraint_strengths(strategy):
    return {
        constraint.field: constraint.strength
        for constraint in strategy.constraints
    }


def _ranking_fields(strategy):
    return [
        ranking_preference.field
        for ranking_preference in strategy.ranking_preferences
    ]


def _drop_fields(strategy):
    return [drop_rule.field for drop_rule in strategy.alternative_drop_rules]


def _relaxation_fields(strategy):
    return [relaxation_rule.field for relaxation_rule in strategy.relaxation_rules]
