import pytest

from app.ai_engine.search_filters import SearchFilters, parse_search_filters


def test_parse_search_filters_extracts_bedroom_apartment_location_and_max_price():
    filters = parse_search_filters("3 bedroom apartment in New Cairo under 6M")

    assert filters.bedrooms == 3
    assert filters.property_type == "Apartment"
    assert filters.location == "New Cairo"
    assert filters.max_price == 6000000.0


def test_parse_search_filters_extracts_villa_zayed_and_budget():
    filters = parse_search_filters("villa in Sheikh Zayed budget 12 million")

    assert filters.property_type == "Villa"
    assert filters.location == "Sheikh Zayed"
    assert filters.max_price == 12000000.0


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

    assert filters.max_down_payment == 1000000.0


def test_parse_search_filters_extracts_min_price():
    filters = parse_search_filters("at least 5M")

    assert filters.min_price == 5000000.0


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
    with pytest.raises(ValueError, match="bedroom"):
        parse_search_filters("2 bedrooms and 3 bedrooms")


def test_parse_search_filters_raises_for_conflicting_payment_methods():
    with pytest.raises(ValueError, match="payment"):
        parse_search_filters("cash and installment")


def test_parse_search_filters_extracts_bathrooms():
    filters = parse_search_filters("apartment with 2 baths")

    assert filters.bathrooms == 2


def test_parse_search_filters_extracts_installment_year_requirement():
    filters = parse_search_filters("installments over 8 years")

    assert filters.min_installment_years == 8.0


def test_search_filters_model_validates_safe_bounds():
    with pytest.raises(ValueError):
        SearchFilters(max_price=-1)
