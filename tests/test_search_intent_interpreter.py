import json
from decimal import Decimal

import pytest

from app.ai_engine.search_filters import parse_search_filters
from app.ai_engine.search_intent import SearchFieldDirective
from app.ai_engine.search_intent_interpreter import interpret_search_query
from app.ai_engine.search_strategy import ConstraintStrength, SearchField


def test_plain_query_preserves_baseline_filters_without_directives():
    text = "3 bedroom apartment in New Cairo under 6M with cash"

    intent = interpret_search_query(text)

    assert intent.filters == parse_search_filters(text)
    assert intent.directives == []
    assert intent.ambiguities == []
    assert intent.unsupported_requirements == []


@pytest.mark.parametrize("text, field", [
    ("must be in New Cairo", SearchField.LOCATION),
    ("New Cairo is required", SearchField.LOCATION),
    ("only New Cairo", SearchField.LOCATION),
    ("must have an apartment", SearchField.PROPERTY_TYPE),
    ("apartment is a must-have", SearchField.PROPERTY_TYPE),
    ("villa required", SearchField.PROPERTY_TYPE),
    ("required apartment", SearchField.PROPERTY_TYPE),
    ("must-have apartment", SearchField.PROPERTY_TYPE),
    ("must have 3 bedrooms", SearchField.BEDROOMS),
    ("must be under 6 million", SearchField.MAX_PRICE),
    ("cash required", SearchField.PAYMENT_METHOD),
])
def test_explicit_required_field_emits_only_hard_strength(text, field):
    intent = interpret_search_query(text)

    assert intent.directives == [
        SearchFieldDirective(field=field, strength=ConstraintStrength.HARD)
    ]
    assert intent.ambiguities == []


@pytest.mark.parametrize("text, field", [
    ("prefer New Cairo", SearchField.LOCATION),
    ("New Cairo preferred", SearchField.LOCATION),
    ("prefer apartment", SearchField.PROPERTY_TYPE),
    ("apartment preferred", SearchField.PROPERTY_TYPE),
    ("prefer under 6 million", SearchField.MAX_PRICE),
])
def test_explicit_preference_does_not_invent_strength(text, field):
    intent = interpret_search_query(text)

    assert intent.directives == [SearchFieldDirective(field=field, preferred=True)]


@pytest.mark.parametrize("text, preferred", [
    ("New Cairo preferred but other locations are okay", True),
    ("New Cairo but other locations are okay", False),
    ("New Cairo, flexible on location", False),
])
def test_generic_location_flexibility_has_one_directive(text, preferred):
    intent = interpret_search_query(text)

    assert intent.filters.location == "New Cairo"
    assert intent.directives == [SearchFieldDirective(
        field=SearchField.LOCATION, preferred=preferred, allow_alternative=True,
    )]
    assert intent.ambiguities == []


@pytest.mark.parametrize("text, preferred", [
    ("cash; flexible on payment method", False),
    ("cash; flexible on payment methods", False),
    ("cash preferred; flexible on payment method", True),
    ("cash preferred; flexible on payment methods", True),
])
def test_payment_method_label_flexibility_has_one_directive(text, preferred):
    intent = interpret_search_query(text)

    assert intent.filters.payment_method == "Cash"
    assert intent.directives == [SearchFieldDirective(
        field=SearchField.PAYMENT_METHOD,
        preferred=preferred,
        allow_alternative=True,
    )]
    assert intent.ambiguities == []


@pytest.mark.parametrize("text", [
    "must be New Cairo but other locations are okay",
    "only New Cairo and New Cairo preferred",
])
def test_contradictory_semantics_produce_ambiguity_without_directive(text):
    intent = interpret_search_query(text)

    assert intent.filters.location == "New Cairo"
    assert intent.directives == []
    assert [ambiguity.field for ambiguity in intent.ambiguities] == [SearchField.LOCATION]


@pytest.mark.parametrize("text", [
    "target budget 6 million", "around 6 million", "budget around 6M",
])
def test_price_target_is_ambiguity_without_a_numeric_bound(text):
    intent = interpret_search_query(text)

    assert intent.filters.min_price is None
    assert intent.filters.max_price is None
    assert intent.directives == []
    assert [ambiguity.field for ambiguity in intent.ambiguities] == [SearchField.MAX_PRICE]


def test_target_suppression_retains_an_independent_explicit_price_bound():
    intent = interpret_search_query("target budget 6 million but under 7 million")

    assert intent.filters.max_price == Decimal("7000000")
    assert intent.filters.min_price is None
    assert any(ambiguity.field is SearchField.MAX_PRICE for ambiguity in intent.ambiguities)


def test_target_preference_does_not_transfer_to_an_independent_price_bound():
    intent = interpret_search_query("prefer target budget 6 million but under 7 million")

    assert intent.filters.max_price == Decimal("7000000")
    assert intent.directives == []
    assert intent.ambiguities


def test_approximate_duration_preference_does_not_become_payment_method_preference():
    intent = interpret_search_query("prefer around 8 years installments")

    assert intent.filters.payment_method == "Installment"
    assert intent.filters.min_installment_years is None
    assert intent.directives == []
    assert any(
        ambiguity.field is SearchField.MIN_INSTALLMENT_YEARS
        for ambiguity in intent.ambiguities
    )


@pytest.mark.parametrize("text, field", [
    ("around 8 years installments", SearchField.MIN_INSTALLMENT_YEARS),
    ("around 3 bedrooms", SearchField.BEDROOMS),
    ("around 2 bathrooms", SearchField.BATHROOMS),
    ("around 120 sqm", SearchField.MIN_AREA_SQM),
])
def test_approximate_non_price_numbers_do_not_become_exact_filters(text, field):
    intent = interpret_search_query(text)

    assert getattr(intent.filters, field.value) is None
    assert intent.filters.max_price is None
    assert [ambiguity.field for ambiguity in intent.ambiguities] == [field]


@pytest.mark.parametrize("text", [
    "New Cairo preferred but not flexible on location",
    "New Cairo but other locations are okay except downtown",
    "New Cairo preferred but other locations are not okay",
    "do not prefer New Cairo",
])
def test_negated_or_restricted_semantics_do_not_create_broad_flexibility(text):
    intent = interpret_search_query(text)

    assert intent.filters.location == "New Cairo"
    assert intent.directives == []
    assert [ambiguity.field for ambiguity in intent.ambiguities] == [SearchField.LOCATION]


def test_minimum_years_does_not_create_a_price_constraint():
    intent = interpret_search_query("apartment, minimum 8 years")

    assert intent.filters.min_installment_years == Decimal("8")
    assert isinstance(intent.filters.min_installment_years, Decimal)
    assert intent.filters.min_price is None
    assert intent.filters.property_type == "Apartment"
    assert intent.ambiguities == []


def test_year_suppression_preserves_an_independent_price_with_the_same_number():
    intent = interpret_search_query("minimum 8 years and at least EGP 8")

    assert intent.filters.min_installment_years == Decimal("8")
    assert intent.filters.min_price == Decimal("8")


@pytest.mark.parametrize("text", [
    "prefer New Cairo but Sheikh Zayed is okay",
    "New Cairo preferred but Sheikh Zayed is okay",
])
def test_specific_location_alternatives_recover_without_broad_fallback(text):
    with pytest.raises(ValueError):
        parse_search_filters(text)

    intent = interpret_search_query("apartment under 6M, " + text)

    assert intent.filters.location is None
    assert intent.filters.property_type == "Apartment"
    assert intent.filters.max_price == Decimal("6000000")
    assert intent.directives == []
    location_ambiguities = [
        ambiguity for ambiguity in intent.ambiguities
        if ambiguity.field is SearchField.LOCATION
    ]
    assert len(location_ambiguities) == 1
    assert "not representable" in location_ambiguities[0].reason


def test_location_aliases_for_one_city_are_not_a_conflict():
    intent = interpret_search_query("prefer New Cairo and tagamoa preferred")

    assert intent.filters.location == "New Cairo"
    assert intent.directives == [SearchFieldDirective(field="location", preferred=True)]
    assert intent.ambiguities == []


@pytest.mark.parametrize("text", ["cash and installments", "2 bedrooms and 3 bedrooms"])
def test_unresolved_parser_conflict_returns_ambiguity_without_guessing(text):
    with pytest.raises(ValueError):
        parse_search_filters(text)

    intent = interpret_search_query(text)

    assert intent.filters.model_dump(exclude_none=True) == {}
    assert intent.directives == []
    assert intent.ambiguities


@pytest.mark.parametrize("text, field", [
    ("large apartment", SearchField.MIN_AREA_SQM),
    ("cheap unit", SearchField.MAX_PRICE),
    ("low down payment", SearchField.MAX_DOWN_PAYMENT),
    ("long installments", SearchField.MIN_INSTALLMENT_YEARS),
])
def test_vague_numeric_language_records_ambiguity_without_threshold(text, field):
    intent = interpret_search_query(text)

    assert getattr(intent.filters, field.value) is None
    assert intent.filters.min_area_sqm is None
    assert intent.filters.max_area_sqm is None
    assert [ambiguity.field for ambiguity in intent.ambiguities] == [field]


@pytest.mark.parametrize("requirement", [
    "sea view", "near school", "near metro", "quiet area",
    "luxury finishing", "good investment",
])
def test_unsupported_requirement_is_recorded_without_a_filter_mapping(requirement):
    intent = interpret_search_query(requirement)

    assert intent.unsupported_requirements == [requirement]
    assert intent.filters.model_dump(exclude_none=True) == {}
    assert intent.directives == []


def test_required_quiet_area_does_not_make_an_existing_location_hard():
    intent = interpret_search_query("New Cairo; must have a quiet area")

    assert intent.filters.location == "New Cairo"
    assert intent.directives == []
    assert intent.unsupported_requirements == ["quiet area"]


def test_multiple_directives_are_canonical_and_do_not_spread_across_clauses():
    intent = interpret_search_query(
        "cash preferred; must have an apartment; prefer New Cairo; under 6 million"
    )

    assert intent.directives == [
        SearchFieldDirective(field="location", preferred=True),
        SearchFieldDirective(field="property_type", strength=ConstraintStrength.HARD),
        SearchFieldDirective(field="payment_method", preferred=True),
    ]
    assert intent.filters.max_price == Decimal("6000000")


def test_repeated_semantics_and_unsupported_requirements_do_not_duplicate_entries():
    intent = interpret_search_query(
        "prefer New Cairo and New Cairo preferred; other locations are okay; "
        "good investment and sea view and sea view"
    )

    assert intent.directives == [SearchFieldDirective(
        field="location", preferred=True, allow_alternative=True,
    )]
    assert intent.unsupported_requirements == ["sea view", "good investment"]


def test_semantics_are_case_insensitive():
    query = "New Cairo preferred but other locations are okay; sea view"

    assert interpret_search_query(query.upper()) == interpret_search_query(query)


@pytest.mark.parametrize("text", ["", " \t\n "])
def test_empty_query_is_rejected(text):
    with pytest.raises(ValueError, match="^search query must not be empty$"):
        interpret_search_query(text)


def test_repeated_results_are_equal_and_do_not_share_mutable_state():
    text = "  prefer New Cairo; under 1234567890123456.78; sea view  "
    original_text = text
    first = interpret_search_query(text)
    second = interpret_search_query(text)

    assert first == second
    assert text == original_text
    assert first.filters.max_price == Decimal("1234567890123456.78")
    assert isinstance(first.filters.max_price, Decimal)
    first.filters.location = "changed"
    first.directives.clear()
    first.unsupported_requirements.clear()
    assert interpret_search_query(text) == second
    json.dumps(second.model_dump(mode="json"))


def test_unpopulated_semantic_subject_is_ambiguous_not_an_invented_filter():
    intent = interpret_search_query("must be in Gardenia")

    assert intent.filters.location is None
    assert intent.directives == []
    assert intent.ambiguities
