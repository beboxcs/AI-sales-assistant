import pytest
from pydantic import ValidationError

from app.ai_engine.search_filters import SearchFilters
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


def test_valid_exact_strategy_uses_defaults_and_normalizes_scopes():
    strategy = SearchStrategy(
        filters=SearchFilters(location="New Cairo", bedrooms=3),
        constraints=[
            SearchConstraint(
                field=SearchField.LOCATION,
                strength=ConstraintStrength.HARD,
            ),
            SearchConstraint(
                field=SearchField.BEDROOMS,
                strength=ConstraintStrength.SOFT,
            ),
        ],
        requested_scope_keys=["  sales_scope  "],
    )

    assert strategy.match_mode is MatchMode.EXACT
    assert strategy.requested_scope_keys == ["sales_scope"]


def test_search_fields_match_search_filters_fields():
    assert {field.value for field in SearchField} == set(SearchFilters.model_fields)


def test_empty_scope_key_list_is_rejected():
    with pytest.raises(ValidationError, match="at least one key"):
        SearchStrategy(filters=SearchFilters(), requested_scope_keys=[])


def test_populated_field_can_have_a_ranking_preference():
    strategy = SearchStrategy(
        filters=SearchFilters(location="New Cairo"),
        ranking_preferences=[RankingPreference(field="location", weight=1.0)],
    )

    assert strategy.ranking_preferences[0].field is SearchField.LOCATION


@pytest.mark.parametrize("match_mode", [MatchMode.NEAR, MatchMode.ALTERNATIVE])
def test_non_exact_match_modes_are_accepted(match_mode):
    strategy = SearchStrategy(filters=SearchFilters(), match_mode=match_mode)

    assert strategy.match_mode is match_mode


def test_scope_keys_remain_case_sensitive():
    strategy = SearchStrategy(
        filters=SearchFilters(),
        requested_scope_keys=["Scope-A", "scope-a"],
    )

    assert strategy.requested_scope_keys == ["Scope-A", "scope-a"]


def _soft_bedroom_strategy(**rule_values):
    return SearchStrategy(
        filters=SearchFilters(bedrooms=3),
        constraints=[SearchConstraint(field="bedrooms", strength="soft")],
        relaxation_rules=[RelaxationRule(field="bedrooms", **rule_values)],
    )


def _soft_field_strategy(field, filter_value, **rule_values):
    return SearchStrategy(
        filters=SearchFilters(**{field: filter_value}),
        constraints=[SearchConstraint(field=field, strength="soft")],
        relaxation_rules=[RelaxationRule(field=field, **rule_values)],
    )


def test_valid_soft_numeric_relaxation_rules():
    delta_strategy = _soft_bedroom_strategy(
        max_delta=1,
        direction=RelaxationDirection.INCREASE,
    )
    percentage_strategy = _soft_field_strategy(
        "max_price",
        6_000_000,
        max_percentage=25,
        direction=RelaxationDirection.DECREASE,
    )
    both_strategy = _soft_field_strategy(
        "max_price",
        6_000_000,
        max_delta=1,
        max_percentage=25,
        direction=RelaxationDirection.BOTH,
    )

    assert delta_strategy.relaxation_rules[0].max_delta == 1
    assert percentage_strategy.relaxation_rules[0].max_percentage == 25
    assert both_strategy.relaxation_rules[0].max_delta == 1


@pytest.mark.parametrize(
    "rule_values, message",
    [
        ({"direction": "increase"}, "requires max_delta"),
        ({"max_delta": 0, "direction": "increase"}, "max_delta"),
        ({"max_delta": -1, "direction": "increase"}, "max_delta"),
        ({"max_percentage": 0, "direction": "increase"}, "max_percentage"),
        ({"max_percentage": -1, "direction": "increase"}, "max_percentage"),
        ({"max_percentage": 101, "direction": "increase"}, "max_percentage"),
        ({"max_delta": 1, "direction": "increase", "priority": 0}, "priority"),
    ],
)
def test_relaxation_rule_values_are_validated(rule_values, message):
    with pytest.raises(ValidationError, match=message):
        _soft_bedroom_strategy(**rule_values)


def test_duplicate_relaxation_fields_are_rejected():
    with pytest.raises(ValidationError, match="relaxation rule fields must be unique"):
        SearchStrategy(
            filters=SearchFilters(bedrooms=3),
            constraints=[SearchConstraint(field="bedrooms", strength="soft")],
            relaxation_rules=[
                RelaxationRule(field="bedrooms", max_delta=1, direction="increase"),
                RelaxationRule(field="bedrooms", max_delta=2, direction="both"),
            ],
        )


def test_relaxation_requires_a_populated_soft_classified_numeric_field():
    with pytest.raises(ValidationError, match="populated filter"):
        SearchStrategy(
            filters=SearchFilters(),
            constraints=[SearchConstraint(field="bedrooms", strength="soft")],
            relaxation_rules=[
                RelaxationRule(field="bedrooms", max_delta=1, direction="both")
            ],
        )

    with pytest.raises(ValidationError, match="classified constraint"):
        SearchStrategy(
            filters=SearchFilters(bedrooms=3),
            relaxation_rules=[
                RelaxationRule(field="bedrooms", max_delta=1, direction="both")
            ],
        )

    with pytest.raises(ValidationError, match="soft constraints"):
        SearchStrategy(
            filters=SearchFilters(bedrooms=3),
            constraints=[SearchConstraint(field="bedrooms", strength="hard")],
            relaxation_rules=[
                RelaxationRule(field="bedrooms", max_delta=1, direction="both")
            ],
        )


@pytest.mark.parametrize("field", ["location", "property_type", "payment_method"])
def test_categorical_relaxation_is_rejected(field):
    filter_values = {field: "New Cairo"}
    if field == "property_type":
        filter_values[field] = "Apartment"
    elif field == "payment_method":
        filter_values[field] = "Cash"

    with pytest.raises(ValidationError, match="only support numeric fields"):
        SearchStrategy(
            filters=SearchFilters(**filter_values),
            constraints=[SearchConstraint(field=field, strength="soft")],
            relaxation_rules=[
                RelaxationRule(field=field, max_delta=1, direction="both")
            ],
        )


@pytest.mark.parametrize("direction", ["increase", "decrease", "both"])
def test_all_relaxation_directions_are_accepted(direction):
    strategy = _soft_bedroom_strategy(max_delta=1, direction=direction)

    assert strategy.relaxation_rules[0].direction.value == direction


def test_model_dump_json_includes_relaxation_rules():
    strategy = _soft_field_strategy(
        "max_price",
        6_000_000,
        max_percentage=10,
        direction=RelaxationDirection.BOTH,
        priority=2,
    )

    dumped = strategy.model_dump(mode="json")

    assert dumped["relaxation_rules"] == [
        {
            "field": "max_price",
            "max_delta": None,
            "max_percentage": 10.0,
            "direction": "both",
            "priority": 2,
        }
    ]


def test_discrete_count_deltas_require_whole_numbers():
    assert _soft_bedroom_strategy(
        max_delta=1,
        direction="both",
    ).relaxation_rules[0].max_delta == 1
    assert _soft_bedroom_strategy(
        max_delta=2.0,
        direction="both",
    ).relaxation_rules[0].max_delta == 2.0

    with pytest.raises(ValidationError, match="whole number"):
        _soft_bedroom_strategy(max_delta=0.5, direction="both")
    with pytest.raises(ValidationError, match="whole number"):
        _soft_field_strategy("bathrooms", 2, max_delta=1.5, direction="both")


@pytest.mark.parametrize("field", ["bedrooms", "bathrooms", "min_installment_years"])
def test_percentage_relaxation_is_rejected_for_discrete_or_duration_fields(field):
    filter_value = 3 if field != "min_installment_years" else 7
    with pytest.raises(ValidationError, match="percentage relaxation"):
        _soft_field_strategy(
            field,
            filter_value,
            max_percentage=10,
            direction="both",
        )


@pytest.mark.parametrize("field, value", [
    ("max_price", 6_000_000),
    ("min_price", 5_000_000),
    ("max_area_sqm", 150),
])
def test_percentage_relaxation_is_accepted_for_continuous_fields(field, value):
    strategy = _soft_field_strategy(
        field,
        value,
        max_percentage=10,
        direction="both",
    )

    assert strategy.relaxation_rules[0].max_percentage == 10


def test_installment_years_accepts_delta_relaxation():
    strategy = _soft_field_strategy(
        "min_installment_years",
        7,
        max_delta=1.5,
        direction="increase",
    )

    assert strategy.relaxation_rules[0].max_delta == 1.5


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_max_delta_must_be_finite(value):
    with pytest.raises(ValidationError):
        _soft_bedroom_strategy(max_delta=value, direction="both")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_max_percentage_must_be_finite(value):
    with pytest.raises(ValidationError):
        _soft_field_strategy(
            "max_price",
            6_000_000,
            max_percentage=value,
            direction="both",
        )


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), float("-inf")])
def test_ranking_weight_must_be_finite(weight):
    with pytest.raises(ValidationError):
        RankingPreference(field="bedrooms", weight=weight)


def test_duplicate_constraint_fields_are_rejected():
    with pytest.raises(ValidationError, match="constraint fields must be unique"):
        SearchStrategy(
            filters=SearchFilters(location="New Cairo"),
            constraints=[
                SearchConstraint(field="location", strength="hard"),
                SearchConstraint(field="location", strength="soft"),
            ],
        )


def test_constraint_must_reference_populated_filter():
    with pytest.raises(ValidationError, match="populated filter"):
        SearchStrategy(
            filters=SearchFilters(location="New Cairo"),
            constraints=[
                SearchConstraint(field="bedrooms", strength="hard"),
            ],
        )


def test_ranking_preference_requires_positive_weight_and_populated_field():
    with pytest.raises(ValidationError):
        RankingPreference(field=SearchField.BEDROOMS, weight=0)

    with pytest.raises(ValidationError, match="populated filter"):
        SearchStrategy(
            filters=SearchFilters(location="New Cairo"),
            ranking_preferences=[
                RankingPreference(field="bedrooms", weight=1.0),
            ],
        )


def test_duplicate_ranking_fields_are_rejected():
    with pytest.raises(ValidationError, match="ranking preference fields must be unique"):
        SearchStrategy(
            filters=SearchFilters(location="New Cairo"),
            ranking_preferences=[
                RankingPreference(field="location", weight=1),
                RankingPreference(field="location", weight=2),
            ],
        )


def test_scope_keys_must_be_non_empty_and_unique_after_trimming():
    with pytest.raises(ValidationError, match="empty keys"):
        SearchStrategy(filters=SearchFilters(), requested_scope_keys=["  "])

    with pytest.raises(ValidationError, match="duplicates"):
        SearchStrategy(
            filters=SearchFilters(),
            requested_scope_keys=["scope-a", " scope-a "],
        )


def test_model_dump_json_is_json_safe():
    strategy = SearchStrategy(
        filters=SearchFilters(
            location="New Cairo",
            min_price=5_000_000,
        ),
        constraints=[SearchConstraint(field="min_price", strength="hard")],
        ranking_preferences=[RankingPreference(field="min_price", weight=1.5)],
    )

    dumped = strategy.model_dump(mode="json")

    assert dumped["match_mode"] == "exact"
    assert dumped["constraints"][0] == {"field": "min_price", "strength": "hard"}
    assert dumped["ranking_preferences"][0]["weight"] == 1.5
    assert dumped["filters"]["min_price"] == 5_000_000.0
