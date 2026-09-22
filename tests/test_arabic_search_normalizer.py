import json
from decimal import Decimal

import pytest

from app.ai_engine.arabic_search_normalizer import (
    interpret_arabic_search_query,
    normalize_arabic_search_query,
)
from app.ai_engine.search_filters import LOCATION_ALIASES, SearchFilters
from app.ai_engine.search_intent import ParsedSearchIntent, SearchFieldDirective
from app.ai_engine.search_strategy import ConstraintStrength, SearchField


@pytest.mark.parametrize("text, expected", [
    ("شقة ٣ غرف ٢ حمام", "apartment 3 bedrooms 2 bathrooms"),
    ("فيلا ۳ غرف ۲ حمام", "villa 3 bedrooms 2 bathrooms"),
    ("حوالي ٦ مليون", "around 6 million"),
    ("التجمع مفضل بس مناطق تانية عادي",
     "New Cairo preferred but other locations are okay"),
    ("يفضل التجمع بس زايد عادي", "prefer New Cairo but Sheikh Zayed are okay"),
    ("أقصى مقدم ٥٠٠ ألف جنيه", "max down payment 500000 EGP"),
    ("شَـقَّة في القَاهِرَة الجَدِيدَة", "apartment in New Cairo"),
    ("Apartment في التجمع under 6M", "Apartment in New Cairo under 6M"),
    ("apartment in Gardenia under 6M", "apartment in Gardenia under 6M"),
    ("شقة في جاردينيا", "apartment in جاردينيا"),
])
def test_known_phrases_normalize_without_translating_english_or_unknown_words(text, expected):
    assert normalize_arabic_search_query(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("التجمع", "New Cairo"),
    ("القاهرة الجديدة", "New Cairo"),
    ("التجمع الخامس", "Fifth Settlement"),
    ("الشيخ زايد", "Sheikh Zayed"),
    ("زايد", "Sheikh Zayed"),
    ("٦ أكتوبر", "6th of October"),
    ("العاصمة الإدارية", "New Capital"),
    ("الساحل الشمالي", "North Coast"),
])
def test_arabic_locations_resolve_only_to_existing_canonical_locations(text, expected):
    intent = interpret_arabic_search_query(text)

    assert expected in LOCATION_ALIASES.values()
    assert intent.filters == SearchFilters(location=expected)
    assert intent.directives == []
    assert intent.ambiguities == []


@pytest.mark.parametrize("text, expected", [
    ("شقة ٣ غرف نوم ٢ حمام في التجمع أقل من ٦ مليون كاش", SearchFilters(
        property_type="Apartment", bedrooms=3, bathrooms=2, location="New Cairo",
        max_price=Decimal("6000000"), payment_method="Cash",
    )),
    ("فيلا ۴ غرف ۳ حمامات ميزانية بحد أقصى ۱۲ مليون", SearchFilters(
        property_type="Villa", bedrooms=4, bathrooms=3, max_price=Decimal("12000000"),
    )),
    ("تقسيط على ٨ سنوات؛ مقدم لحد ٥٠٠ ألف", SearchFilters(
        payment_method="Installment", min_installment_years=Decimal("8"),
        max_down_payment=Decimal("500000"),
    )),
    ("حد أدنى ٨ سنين", SearchFilters(min_installment_years=Decimal("8"))),
    ("السعر أكثر من ٥ مليون", SearchFilters(min_price=Decimal("5000000"))),
    ("Apartment في التجمع under 6M كاش", SearchFilters(
        property_type="Apartment", location="New Cairo", max_price=Decimal("6000000"),
        payment_method="Cash",
    )),
])
def test_plain_arabic_and_mixed_queries_extract_filters_without_inventing_directives(text, expected):
    intent = interpret_arabic_search_query(text)

    assert isinstance(intent, ParsedSearchIntent)
    assert intent.filters == expected
    assert intent.directives == []
    assert intent.ambiguities == []
    assert intent.unsupported_requirements == []


@pytest.mark.parametrize("text, expected", [
    ("لازم في التجمع", SearchFieldDirective(field="location", strength=ConstraintStrength.HARD)),
    ("ضروري في التجمع", SearchFieldDirective(field="location", strength=ConstraintStrength.HARD)),
    ("شقة مطلوبة", SearchFieldDirective(field="property_type", strength=ConstraintStrength.HARD)),
    ("يفضل التجمع", SearchFieldDirective(field="location", preferred=True)),
    ("التجمع مفضل", SearchFieldDirective(field="location", preferred=True)),
    ("يفضل شقة", SearchFieldDirective(field="property_type", preferred=True)),
    ("التجمع بس مناطق تانية عادي", SearchFieldDirective(field="location", allow_alternative=True)),
    ("التجمع مفضل بس مناطق تانية عادي", SearchFieldDirective(
        field="location", preferred=True, allow_alternative=True,
    )),
    ("شقة مفضلة بس انواع عقارات تانية عادي", SearchFieldDirective(
        field="property_type", preferred=True, allow_alternative=True,
    )),
    ("كاش مفضل؛ مرن في طريقة الدفع", SearchFieldDirective(
        field="payment_method", preferred=True, allow_alternative=True,
    )),
])
def test_explicit_semantics_produce_one_directive_without_extra_strength(text, expected):
    intent = interpret_arabic_search_query(text)

    assert intent.directives == [expected]
    assert intent.ambiguities == []


@pytest.mark.parametrize("text", [
    "لازم في التجمع بس مناطق تانية عادي",
    "التجمع مفضل بس مناطق تانية مش عادي",
    "مش يفضل التجمع",
])
def test_contradictory_or_negated_semantics_do_not_create_flexibility(text):
    intent = interpret_arabic_search_query(text)

    assert intent.filters.location == "New Cairo"
    assert intent.directives == []
    assert any(ambiguity.field is SearchField.LOCATION for ambiguity in intent.ambiguities)


def test_negated_arabic_location_is_suppressed():
    intent = interpret_arabic_search_query("مش التجمع")

    assert intent.filters.location is None
    assert [ambiguity.field for ambiguity in intent.ambiguities] == [
        SearchField.LOCATION
    ]


@pytest.mark.parametrize("text, field", [
    ("حوالي ٦ مليون", SearchField.MAX_PRICE),
    ("تقريباً ٦ مليون", SearchField.MAX_PRICE),
    ("ميزانية مستهدفة ٦ مليون", SearchField.MAX_PRICE),
    ("مساحة كبيرة", SearchField.MIN_AREA_SQM),
    ("سعر رخيص", SearchField.MAX_PRICE),
    ("مقدم قليل", SearchField.MAX_DOWN_PAYMENT),
    ("تقسيط طويل", SearchField.MIN_INSTALLMENT_YEARS),
])
def test_approximate_and_vague_requirements_record_ambiguity_without_thresholds(text, field):
    intent = interpret_arabic_search_query(text)

    assert getattr(intent.filters, field.value) is None
    assert intent.filters.min_price is None
    assert intent.filters.max_price is None
    assert intent.filters.min_area_sqm is None
    assert intent.filters.max_area_sqm is None
    assert intent.directives == []
    assert [ambiguity.field for ambiguity in intent.ambiguities] == [field]


def test_approximate_budget_does_not_suppress_an_independent_bound():
    intent = interpret_arabic_search_query("حوالي ٦ مليون بس أقل من ٧ مليون")

    assert intent.filters.max_price == Decimal("7000000")
    assert intent.filters.min_price is None
    assert [ambiguity.field for ambiguity in intent.ambiguities] == [SearchField.MAX_PRICE]


@pytest.mark.parametrize("text, requirement", [
    ("فيو بحر", "sea view"),
    ("إطلالة على البحر", "sea view"),
    ("منطقة هادئة", "quiet area"),
    ("قريب من المترو", "near metro"),
    ("قريب من مدرسة", "near school"),
    ("تشطيب فاخر", "luxury finishing"),
    ("استثمار كويس", "good investment"),
])
def test_unsupported_requirements_are_recorded_without_supported_filter_mappings(text, requirement):
    intent = interpret_arabic_search_query(text)

    assert intent.unsupported_requirements == [requirement]
    assert intent.filters == SearchFilters()
    assert intent.directives == []


def test_specific_location_alternative_stays_bounded_and_becomes_ambiguity():
    intent = interpret_arabic_search_query("شقة؛ يفضل التجمع بس زايد عادي")

    assert intent.filters == SearchFilters(property_type="Apartment")
    assert intent.directives == []
    assert any(
        ambiguity.field is SearchField.LOCATION and "not representable" in ambiguity.reason
        for ambiguity in intent.ambiguities
    )


@pytest.mark.parametrize("text", ["", " \t\n "])
def test_empty_queries_keep_the_interpreter_validation_contract(text):
    assert normalize_arabic_search_query(text) == ""
    with pytest.raises(ValueError, match="^search query must not be empty$"):
        interpret_arabic_search_query(text)


@pytest.mark.parametrize("text, expected", [
    ("أقل من ١٢٣٤٥٦٧٨٩٠١٢٣٤٥٦٫٧٨", Decimal("1234567890123456.78")),
    ("أقل من ۱۲۳۴۵۶۷۸۹۰۱۲۳۴۵۶٫۷۸", Decimal("1234567890123456.78")),
    ("أقل من ١٢٣٤٥٦٧٨٩٠١٢٣٫٤٥٦٧٨ ألف", Decimal("1234567890123456.78")),
    ("أقل من ١٬٢٣٤٬٥٦٧٫٨٩", Decimal("1234567.89")),
    ("أقل من 1,234.567 ألف", Decimal("1234567")),
])
def test_numeric_normalization_preserves_exact_decimal_values(text, expected):
    intent = interpret_arabic_search_query(text)

    assert intent.filters.max_price == expected
    assert isinstance(intent.filters.max_price, Decimal)


def test_normalization_is_deterministic_json_safe_and_does_not_mutate_input():
    text = "  يفضل التجمع؛ أقل من ٦ مليون؛ فيو بحر  "
    original = text
    normalized = normalize_arabic_search_query(text)
    first = interpret_arabic_search_query(text)
    second = interpret_arabic_search_query(text)

    assert text == original
    assert normalize_arabic_search_query(text) == normalized
    assert first == second
    assert json.loads(json.dumps(first.model_dump(mode="json")))["filters"]["location"] == "New Cairo"
    first.directives.clear()
    first.filters.location = "changed"
    assert interpret_arabic_search_query(text) == second
