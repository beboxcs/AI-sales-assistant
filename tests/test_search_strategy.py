import pytest
from pydantic import ValidationError

from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_strategy import (
    ConstraintStrength,
    MatchMode,
    RankingPreference,
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
