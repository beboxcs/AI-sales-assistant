import json
from decimal import Decimal

import pytest

from app.ai_engine.arabic_search_normalizer import interpret_arabic_search_query
from app.ai_engine.arabizi_search_normalizer import interpret_arabizi_search_query
from app.ai_engine.multilingual_search_interpreter import interpret_multilingual_search_query
from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_intent import ParsedSearchIntent
from app.ai_engine.search_intent_interpreter import interpret_search_query


@pytest.mark.parametrize("english, arabic, arabizi, mixed", [
    (
        "3 bedroom apartment in New Cairo under 6 million cash",
        "شقة ٣ غرف في التجمع أقل من ٦ مليون كاش",
        "3ayez sha2a 3 bedrooms fe tagamo3 ta7t 6 million cash",
        "عايز sha2a 3 bedrooms fe التجمع under 6 million cash",
    ),
    ("must be in New Cairo", "لازم في التجمع", "lazem fe tagamo3", "lazem في New Cairo"),
    ("prefer New Cairo", "يفضل التجمع", "yefadal tagamo3", "yefadal التجمع"),
    (
        "New Cairo preferred but other locations are okay",
        "التجمع مفضل بس مناطق تانية عادي",
        "tagamo3 mofadal bas manate2 tanya 3ady",
        "التجمع mofadal but other locations are okay",
    ),
    (
        "prefer New Cairo but Sheikh Zayed is okay",
        "يفضل التجمع بس زايد عادي",
        "yefadal tagamo3 bas zayed 3ady",
        "yefadal التجمع but Sheikh Zayed is okay",
    ),
    ("around 6 million", "حوالي ٦ مليون", "7awaly 6 million", "7awaly ٦ million"),
    ("apartment; sea view", "شقة؛ فيو بحر", "sha2a; sea view", "sha2a؛ sea view"),
])
def test_supported_languages_produce_equal_intent(english, arabic, arabizi, mixed):
    expected = interpret_search_query(english)

    assert interpret_multilingual_search_query(english) == expected
    assert interpret_multilingual_search_query(arabic) == expected
    assert interpret_multilingual_search_query(arabizi) == expected
    assert interpret_multilingual_search_query(mixed) == expected
    assert interpret_multilingual_search_query(arabic) == interpret_arabic_search_query(arabic)
    assert interpret_multilingual_search_query(arabizi) == interpret_arabizi_search_query(arabizi)


@pytest.mark.parametrize("text", [
    "3 bedrooms in New Cairo",
    "apartment under 1234567890123456.78; Unknown7Token",
    "2 bathrooms; mysha2a sha2axyz _mo2adam_ 3adyish",
    "minimum 8 years; sea view",
    "prefer tagamo3",
])
def test_english_numbers_and_unknown_tokens_preserve_existing_behavior(text):
    assert interpret_multilingual_search_query(text) == interpret_search_query(text)


@pytest.mark.parametrize("text", ["غاردينيا", "X7Unknown", "غاردينيا X7Unknown"])
def test_unknown_vocabulary_does_not_invent_filters_or_semantics(text):
    assert interpret_multilingual_search_query(text) == ParsedSearchIntent(filters=SearchFilters())


@pytest.mark.parametrize("text", [
    "lazem fe tagamo3",
    "tagamo3 mofadal bas manate2 tanya 3ady",
    "ta2seet 8 sneen",
    "7awaly 6 million",
])
def test_arabizi_routing_is_case_insensitive(text):
    assert interpret_multilingual_search_query(text.upper()) == interpret_arabizi_search_query(text)


def test_mixed_example_composes_both_normalizers():
    assert interpret_multilingual_search_query("عايز sha2a fe التجمع").filters == SearchFilters(
        property_type="Apartment", location="New Cairo",
    )


@pytest.mark.parametrize("text", [
    "under 1234567890123456.78",
    "أقل من ١٢٣٤٥٦٧٨٩٠١٢٣٤٥٦٫٧٨",
    "ta7t 1234567890123.45678 alf",
    "ta7t ١٢٣٤٥٦٧٨٩٠١٢٣٤٥٦٫٧٨",
])
def test_all_paths_preserve_exact_decimal_values(text):
    intent = interpret_multilingual_search_query(text)

    assert intent.filters.max_price == Decimal("1234567890123456.78")
    assert isinstance(intent.filters.max_price, Decimal)


@pytest.mark.parametrize("text", ["", " \t\n "])
def test_empty_queries_preserve_existing_validation(text):
    with pytest.raises(ValueError, match="^search query must not be empty$"):
        interpret_multilingual_search_query(text)


def test_repeated_results_are_equal_json_safe_and_do_not_mutate_input():
    text = "  yefadal التجمع; sha2a under 6 million; sea view  "
    original = text
    first = interpret_multilingual_search_query(text)
    second = interpret_multilingual_search_query(text)

    assert first == second
    assert text == original
    assert json.loads(json.dumps(first.model_dump(mode="json")))["filters"]["location"] == "New Cairo"
    first.filters.location = "changed"
    first.directives.clear()
    assert interpret_multilingual_search_query(text) == second
