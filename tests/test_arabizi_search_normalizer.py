import json
from decimal import Decimal

import pytest

from app.ai_engine.arabic_search_normalizer import normalize_arabic_search_query
from app.ai_engine.arabizi_search_normalizer import (
    interpret_arabizi_search_query,
    normalize_arabizi_search_query,
)
from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_intent import ParsedSearchIntent, SearchFieldDirective
from app.ai_engine.search_strategy import ConstraintStrength, SearchField


@pytest.mark.parametrize("text, expected", [
    ("3ayez sha2a", "want شقة"),
    ("3awez sh2a", "want شقة"),
    ("m7tag sha2a", "need شقة"),
    ("BDWR  3LA sha2a", "looking for شقة"),
    ("manate2 tanya 3ady", "مناطق تانية عادي"),
    ("mo2adam ta7t 500 alf", "مقدم لحد 500 ألف"),
    ("mo2adam 500 alf", "مقدم 500 ألف"),
    ("ta2seet 8 sneen", "8 سنين تقسيط"),
    ("ta2sit ٨٫٥ senen", "٨٫٥ سنين تقسيط"),
    ("7awaly 6 million", "حوالي 6 million"),
    ("yefadal التجمع; sea view", "يفضل التجمع; sea view"),
    ("zaied", "zayed"),
])
def test_phrase_normalization_preserves_values_and_uses_arabic_vocabulary(text, expected):
    assert normalize_arabizi_search_query(text) == expected


@pytest.mark.parametrize("text", [
    "3 bedrooms Apartment in New Cairo under 6 million cash; sea view",
    "tagamo3 tagamoa zayed",
    "  X7Unknown\tmysha2a sha2axyz _mo2adam_ 3adyish  ",
    "شقة في التجمع",
])
def test_english_arabic_and_unknown_tokens_are_preserved_exactly(text):
    assert normalize_arabizi_search_query(text) == text


def test_plain_search_extracts_real_numbers_without_semantic_directives():
    intent = interpret_arabizi_search_query(
        "3ayez sha2a 3 bedrooms fe tagamo3 ta7t 6 million cash"
    )

    assert isinstance(intent, ParsedSearchIntent)
    assert intent.filters == SearchFilters(
        property_type="Apartment", bedrooms=3, location="New Cairo",
        max_price=Decimal("6000000"), payment_method="Cash",
    )
    assert intent.directives == []
    assert intent.ambiguities == []


@pytest.mark.parametrize("word", ["lazem", "darory", "dorory"])
def test_explicit_requirement_reaches_the_interpreter_as_hard(word):
    intent = interpret_arabizi_search_query(f"{word} fe tagamo3")

    assert intent.filters == SearchFilters(location="New Cairo")
    assert intent.directives == [SearchFieldDirective(
        field="location", strength=ConstraintStrength.HARD,
    )]
    assert intent.ambiguities == []


@pytest.mark.parametrize("text", ["yefadal tagamo3", "yfdl tagamoa", "tagamo3 mofadal"])
def test_preference_does_not_invent_strength(text):
    intent = interpret_arabizi_search_query(text)

    assert intent.filters.location == "New Cairo"
    assert intent.directives == [SearchFieldDirective(field="location", preferred=True)]
    assert intent.ambiguities == []


@pytest.mark.parametrize("text, preferred", [
    ("tagamo3 bas manate2 tanya 3ady", False),
    ("tagamo3 mofadal bas manate2 tanya 3ady", True),
    ("yefadal tagamo3; tagamo3 mofadal bas manate2 tanya 3ady", True),
])
def test_generic_alternatives_and_repeated_preference_merge_into_one_directive(text, preferred):
    intent = interpret_arabizi_search_query(text)

    assert intent.filters.location == "New Cairo"
    assert intent.directives == [SearchFieldDirective(
        field="location", preferred=preferred, allow_alternative=True,
    )]
    assert intent.ambiguities == []


@pytest.mark.parametrize("text", [
    "yefadal tagamo3 bas zayed 3ady",
    "yefadal التجمع bas zaied 3ady",
])
def test_specific_alternatives_are_ambiguous_and_never_become_generic_flexibility(text):
    intent = interpret_arabizi_search_query(text)

    assert intent.filters.location is None
    assert intent.directives == []
    assert any(
        ambiguity.field is SearchField.LOCATION and "not representable" in ambiguity.reason
        for ambiguity in intent.ambiguities
    )


@pytest.mark.parametrize("text", [
    "lazem tagamo3 bas manate2 tanya 3ady",
    "tagamo3 mofadal bas manate2 tanya mesh 3ady",
    "mosh yefadal tagamo3",
])
def test_contradiction_or_negation_does_not_invent_flexibility(text):
    intent = interpret_arabizi_search_query(text)

    assert intent.filters.location == "New Cairo"
    assert intent.directives == []
    assert any(ambiguity.field is SearchField.LOCATION for ambiguity in intent.ambiguities)


@pytest.mark.parametrize("word", ["7awaly", "7awli", "ta2reban"])
def test_approximate_price_remains_ambiguity_without_a_bound(word):
    intent = interpret_arabizi_search_query(f"{word} 6 million")

    assert intent.filters == SearchFilters()
    assert intent.directives == []
    assert [ambiguity.field for ambiguity in intent.ambiguities] == [SearchField.MAX_PRICE]


def test_bare_down_payment_amount_does_not_invent_a_maximum():
    text = "mo2adam 500 alf"

    assert normalize_arabic_search_query(normalize_arabizi_search_query(text)) == "down payment 500000"
    assert interpret_arabizi_search_query(text).filters == SearchFilters()


def test_explicit_down_payment_bound_preserves_exact_decimal_amount():
    intent = interpret_arabizi_search_query("mo2adam ta7t 1234567890123.45678 alf")

    assert intent.filters == SearchFilters(max_down_payment=Decimal("1234567890123456.78"))
    assert isinstance(intent.filters.max_down_payment, Decimal)


@pytest.mark.parametrize("text, duration", [
    ("ta2seet 8 sneen", "8"),
    ("ta2sit 8.5 senen", "8.5"),
    ("TA2SIT 8 SENEN", "8"),
    ("ta2seet ٨٫٥ sneen", "8.5"),
    ("ta2seet 3la 8 sneen", "8"),
])
def test_installment_duration_preserves_the_number_without_a_false_price(text, duration):
    intent = interpret_arabizi_search_query(text)

    assert intent.filters == SearchFilters(
        payment_method="Installment", min_installment_years=Decimal(duration),
    )
    assert isinstance(intent.filters.min_installment_years, Decimal)
    assert intent.directives == []
    assert intent.ambiguities == []


def test_approximate_installment_duration_does_not_turn_into_a_minimum():
    intent = interpret_arabizi_search_query("7awaly ta2seet 8 sneen")

    assert intent.filters == SearchFilters(payment_method="Installment")
    assert [ambiguity.field for ambiguity in intent.ambiguities] == [SearchField.MIN_INSTALLMENT_YEARS]


@pytest.mark.parametrize("text", ["3ayez", "m7tag", "mo2adam", "7awaly", "Unknown7Token"])
def test_word_digits_do_not_create_numeric_filters(text):
    intent = interpret_arabizi_search_query(text)

    assert intent.filters == SearchFilters()
    assert intent.directives == []


def test_mixed_query_preserves_unsupported_english_and_is_case_insensitive():
    text = "yefadal التجمع; sha2a 3 bedrooms ta7t 6 million; sea view"
    intent = interpret_arabizi_search_query(text)

    assert intent == interpret_arabizi_search_query(text.upper())
    assert intent.filters == SearchFilters(
        location="New Cairo", property_type="Apartment", bedrooms=3,
        max_price=Decimal("6000000"),
    )
    assert intent.directives == [SearchFieldDirective(field="location", preferred=True)]
    assert intent.unsupported_requirements == ["sea view"]
    assert intent.ambiguities == []


@pytest.mark.parametrize("text", ["", " \t\n "])
def test_empty_query_keeps_existing_validation_semantics(text):
    with pytest.raises(ValueError, match="^search query must not be empty$"):
        interpret_arabizi_search_query(text)


def test_repeated_calls_are_deterministic_json_safe_and_do_not_mutate_input():
    text = "yefadal tagamo3; ta7t 1234567890123456.78; sea view"
    original = text
    normalized = normalize_arabizi_search_query(text)
    first = interpret_arabizi_search_query(text)
    second = interpret_arabizi_search_query(text)

    assert text == original
    assert normalize_arabizi_search_query(text) == normalized
    assert first == second
    assert first.filters.max_price == Decimal("1234567890123456.78")
    assert isinstance(first.filters.max_price, Decimal)
    assert json.loads(json.dumps(first.model_dump(mode="json")))["filters"]["location"] == "New Cairo"
    first.directives.clear()
    first.filters.location = "changed"
    assert interpret_arabizi_search_query(text) == second
