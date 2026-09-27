from decimal import Decimal

import pytest

from app.ai_engine.search_filters import (
    SearchFilters,
    SearchQueryParseError,
    SearchQueryParseErrorKind,
    parse_search_filters,
)


def test_parse_search_filters_extracts_bedroom_apartment_location_and_max_price():
    filters = parse_search_filters("3 bedroom apartment in New Cairo under 6M")

    assert filters.bedrooms == 3
    assert filters.property_type == "Apartment"
    assert filters.location == "New Cairo"
    assert filters.max_price == Decimal("6000000")


def test_parse_search_filters_extracts_villa_zayed_and_budget():
    filters = parse_search_filters("villa in Sheikh Zayed budget 12 million")

    assert filters.property_type == "Villa"
    assert filters.location == "Sheikh Zayed"
    assert filters.max_price == Decimal("12000000")


def test_parse_search_filters_extracts_townhouse_october_cash():
    filters = parse_search_filters("2 BR townhouse in October cash")

    assert filters.bedrooms == 2
    assert filters.property_type == "Townhouse"
    assert filters.location == "6th of October"
    assert filters.payment_method == "Cash"


def test_parse_search_filters_extracts_new_capital_installment_apartment():
    filters = parse_search_filters("apartment in New Capital with installments")

    assert filters.location == "New Capital"
    assert filters.property_type == "Apartment"
    assert filters.payment_method == "Installment"


def test_parse_search_filters_extracts_max_down_payment():
    filters = parse_search_filters("max down payment 1M")

    assert filters.max_down_payment == Decimal("1000000")


def test_parse_search_filters_extracts_min_price():
    filters = parse_search_filters("at least 5M")

    assert filters.min_price == Decimal("5000000")


def test_parse_search_filters_does_not_treat_ambiguous_budget_as_max_price():
    filters = parse_search_filters("around 6M")

    assert filters.max_price is None


@pytest.mark.parametrize("text", ["3-4 bedrooms", "3+ bedrooms"])
def test_parse_search_filters_does_not_extract_ambiguous_bedrooms(text):
    filters = parse_search_filters(text)

    assert filters.bedrooms is None


def test_parse_search_filters_leaves_unknown_location_empty():
    filters = parse_search_filters("apartment in Gardenia")

    assert filters.location is None


def test_parse_search_filters_leaves_unspecified_fields_empty():
    filters = parse_search_filters("apartment in New Cairo")

    assert filters.bathrooms is None
    assert filters.min_price is None
    assert filters.max_area_sqm is None
    assert filters.payment_method is None
    assert filters.max_down_payment is None
    assert filters.min_installment_years is None


def test_parse_search_filters_raises_for_conflicting_bedroom_values():
    with pytest.raises(SearchQueryParseError, match="bedroom") as raised:
        parse_search_filters("2 bedrooms and 3 bedrooms")

    assert raised.value.kind is SearchQueryParseErrorKind.CONFLICTING_VALUES


def test_parse_search_filters_raises_for_conflicting_payment_methods():
    with pytest.raises(SearchQueryParseError, match="payment"):
        parse_search_filters("cash and installment")


@pytest.mark.parametrize(
    "money",
    [
        "6..5 million",
        "6,00,000",
        "-6 million",
        "+6 million",
        "6 billion",
        "6 bn",
        "500 thousand",
        "6 trillion",
        "6 crore",
        "6 usd",
    ],
)
def test_parse_search_filters_rejects_malformed_or_unsupported_money(money):
    with pytest.raises(SearchQueryParseError, match="money") as raised:
        parse_search_filters(f"under {money}")

    assert raised.value.kind is SearchQueryParseErrorKind.INVALID_SYNTAX


@pytest.mark.parametrize(
    ("money", "expected"),
    [
        ("0", Decimal("0")),
        ("6000000", Decimal("6000000")),
        ("6,000,000", Decimal("6000000")),
        ("6 million", Decimal("6000000")),
        ("6.5 million", Decimal("6500000")),
        ("EGP 6000000", Decimal("6000000")),
        ("1234567890123456.78", Decimal("1234567890123456.78")),
    ],
)
def test_parse_search_filters_preserves_valid_money_exactly(money, expected):
    filters = parse_search_filters(f"under {money}")

    assert filters.max_price == expected
    assert isinstance(filters.max_price, Decimal)


def test_parse_search_filters_accepts_sentence_boundary_after_money():
    filters = parse_search_filters("under 6000000. cash")

    assert filters.max_price == Decimal("6000000")
    assert filters.payment_method == "Cash"


@pytest.mark.parametrize(
    ("text", "field", "expected"),
    [
        ("under 6000000 cash", "payment_method", "Cash"),
        ("under 6000000 villa", "property_type", "Villa"),
    ],
)
def test_parse_search_filters_keeps_adjacent_supported_filter_after_money(
    text,
    field,
    expected,
):
    filters = parse_search_filters(text)

    assert filters.max_price == Decimal("6000000")
    assert getattr(filters, field) == expected


def test_parse_search_filters_scales_large_million_value_without_rounding():
    filters = parse_search_filters(
        "under 12345678901234567890123456789 million"
    )

    assert filters.max_price == Decimal("12345678901234567890123456789000000")


def test_parse_search_filters_extracts_bathrooms():
    filters = parse_search_filters("apartment with 2 baths")

    assert filters.bathrooms == 2


def test_parse_search_filters_extracts_installment_year_requirement():
    filters = parse_search_filters("installments over 8 years")

    assert filters.min_installment_years == Decimal("8")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("at least 120 sqm", "120"),
        ("minimum 120.5 m2", "120.5"),
        ("minimum area 1,200 sqm", "1200"),
        ("above 120 m²", "120"),
        ("over 120 SQM", "120"),
        (
            "at least 1234567890123456.78 sqm",
            "1234567890123456.78",
        ),
    ],
)
def test_parse_search_filters_extracts_exact_minimum_area(text, expected):
    filters = parse_search_filters(text)

    assert filters.min_area_sqm == Decimal(expected)
    assert isinstance(filters.min_area_sqm, Decimal)
    assert filters.min_price is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("under 200 sqm", "200"),
        ("below 200.25 m2", "200.25"),
        ("max area 1,200 sqm", "1200"),
        ("maximum area 200 m²", "200"),
        ("area up to 200 SQM", "200"),
    ],
)
def test_parse_search_filters_extracts_exact_maximum_area(text, expected):
    filters = parse_search_filters(text)

    assert filters.max_area_sqm == Decimal(expected)
    assert isinstance(filters.max_area_sqm, Decimal)
    assert filters.max_price is None


@pytest.mark.parametrize(
    ("text", "price_field", "price", "area_field", "area"),
    [
        (
            "under 6 million and under 200 sqm",
            "max_price",
            "6000000",
            "max_area_sqm",
            "200",
        ),
        (
            "above 4 million and at least 120 sqm",
            "min_price",
            "4000000",
            "min_area_sqm",
            "120",
        ),
    ],
)
def test_area_spans_do_not_change_independent_price_bounds(
    text,
    price_field,
    price,
    area_field,
    area,
):
    filters = parse_search_filters(text)

    assert getattr(filters, price_field) == Decimal(price)
    assert getattr(filters, area_field) == Decimal(area)


@pytest.mark.parametrize(
    "text",
    [
        "at least 120..5 sqm",
        "under 12,00 sqm",
        "above -120 sqm",
        "over +120 sqm",
        "under 0 sqm",
        "max area 200 sqft",
        "area up to 200",
        "minimum area",
    ],
)
def test_parse_search_filters_rejects_malformed_area(text):
    with pytest.raises(SearchQueryParseError, match="area|Area"):
        parse_search_filters(text)


@pytest.mark.parametrize(
    "text",
    [
        "at least 120 sqm and above 130 sqm",
        "under 200 sqm and below 190 sqm",
    ],
)
def test_parse_search_filters_rejects_conflicting_repeated_area_bounds(text):
    with pytest.raises(SearchQueryParseError, match="area"):
        parse_search_filters(text)


def test_repeated_identical_area_bounds_are_deterministic():
    text = "at least 120 sqm and minimum 120 sqm"

    assert parse_search_filters(text) == parse_search_filters(text)
    assert parse_search_filters(text).min_area_sqm == Decimal("120")


def test_non_area_numeric_fields_do_not_create_area_bounds():
    filters = parse_search_filters(
        "3 bedrooms 2 bathrooms installments over 8 years under 6 million"
    )

    assert filters.min_area_sqm is None
    assert filters.max_area_sqm is None
    assert filters.bedrooms == 3
    assert filters.bathrooms == 2
    assert filters.min_installment_years == Decimal("8")
    assert filters.max_price == Decimal("6000000")


def test_search_filters_model_validates_safe_bounds():
    with pytest.raises(ValueError):
        SearchFilters(max_price=-1)


def test_search_filters_normalize_business_numbers_to_decimal():
    filters = SearchFilters(
        min_price=0.1,
        max_price="1234567.89",
        min_area_sqm=120.25,
        max_area_sqm=180,
        max_down_payment=Decimal("750000.50"),
        min_installment_years=6.5,
    )

    assert filters.min_price == Decimal("0.1")
    assert filters.max_price == Decimal("1234567.89")
    assert filters.min_area_sqm == Decimal("120.25")
    assert filters.max_area_sqm == Decimal("180")
    assert filters.max_down_payment == Decimal("750000.50")
    assert filters.min_installment_years == Decimal("6.5")
    assert all(
        isinstance(value, Decimal)
        for value in (
            filters.min_price,
            filters.max_price,
            filters.min_area_sqm,
            filters.max_area_sqm,
            filters.max_down_payment,
            filters.min_installment_years,
        )
    )
