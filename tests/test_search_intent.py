import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_intent import (
    ParsedSearchIntent,
    SearchAmbiguity,
    SearchFieldDirective,
)
from app.ai_engine.search_strategy import ConstraintStrength, SearchField


def test_valid_intent_with_multiple_populated_filters_and_directives():
    intent = ParsedSearchIntent(
        filters=SearchFilters(
            location="New Cairo",
            bedrooms=3,
            max_price=Decimal("1234567.89"),
        ),
        directives=[
            SearchFieldDirective(
                field=SearchField.LOCATION,
                strength=ConstraintStrength.HARD,
            ),
            SearchFieldDirective(
                field=SearchField.BEDROOMS,
                strength=ConstraintStrength.SOFT,
            ),
        ],
    )

    assert intent.directives[0].strength is ConstraintStrength.HARD
    assert intent.directives[1].strength is ConstraintStrength.SOFT
    assert intent.filters.max_price == Decimal("1234567.89")


def test_directive_preferred_and_allow_alternative_flags():
    intent = ParsedSearchIntent(
        filters=SearchFilters(property_type="Apartment", max_down_payment=100_000),
        directives=[
            SearchFieldDirective(
                field="property_type",
                preferred=True,
            ),
            SearchFieldDirective(
                field="max_down_payment",
                allow_alternative=True,
            ),
        ],
    )

    assert intent.directives[0].preferred is True
    assert intent.directives[1].allow_alternative is True


def test_soft_directive_can_be_preferred_and_allow_alternative():
    directive = SearchFieldDirective(
        field="payment_method",
        strength=ConstraintStrength.SOFT,
        preferred=True,
        allow_alternative=True,
    )

    assert directive.strength is ConstraintStrength.SOFT
    assert directive.preferred is True
    assert directive.allow_alternative is True


def test_no_op_directive_is_rejected():
    with pytest.raises(
        ValidationError,
        match="directive must define strength, preference, or alternative behavior",
    ):
        SearchFieldDirective(field="bedrooms")


def test_hard_directive_cannot_allow_alternative():
    with pytest.raises(ValidationError, match="cannot allow alternatives"):
        SearchFieldDirective(
            field="location",
            strength=ConstraintStrength.HARD,
            allow_alternative=True,
        )


def test_hard_directive_cannot_be_preferred():
    with pytest.raises(ValidationError, match="cannot be preferred"):
        SearchFieldDirective(
            field="location",
            strength=ConstraintStrength.HARD,
            preferred=True,
        )


def test_duplicate_directive_field_is_rejected():
    with pytest.raises(ValidationError, match="directive fields must be unique"):
        ParsedSearchIntent(
            filters=SearchFilters(location="New Cairo"),
            directives=[
                SearchFieldDirective(field="location", strength="soft"),
                SearchFieldDirective(field="location", preferred=True),
            ],
        )


def test_directive_referencing_unpopulated_filter_is_rejected():
    with pytest.raises(ValidationError, match="populated filter"):
        ParsedSearchIntent(
            filters=SearchFilters(location="New Cairo"),
            directives=[SearchFieldDirective(field="bedrooms", strength="soft")],
        )


def test_ambiguity_accepts_optional_field_and_trims_reason():
    with_field = SearchAmbiguity(field="location", reason="  multiple cities  ")
    without_field = SearchAmbiguity(reason="  vague budget  ")

    assert with_field.field is SearchField.LOCATION
    assert with_field.reason == "multiple cities"
    assert without_field.field is None
    assert without_field.reason == "vague budget"


def test_blank_ambiguity_reason_is_rejected():
    with pytest.raises(ValidationError, match="must not be blank"):
        SearchAmbiguity(reason=" \t\n ")


def test_unsupported_requirements_are_trimmed():
    intent = ParsedSearchIntent(
        filters=SearchFilters(),
        unsupported_requirements=["  sea view  ", "\tserviced apartment\n"],
    )

    assert intent.unsupported_requirements == [
        "sea view",
        "serviced apartment",
    ]


def test_blank_unsupported_requirement_is_rejected():
    with pytest.raises(ValidationError, match="must not be blank"):
        ParsedSearchIntent(
            filters=SearchFilters(),
            unsupported_requirements=["parking", "   "],
        )


def test_duplicate_unsupported_requirements_after_trimming_are_rejected():
    with pytest.raises(ValidationError, match="must be unique"):
        ParsedSearchIntent(
            filters=SearchFilters(),
            unsupported_requirements=["sea view", "  sea view  "],
        )


def test_search_filters_decimal_values_are_preserved():
    min_price = Decimal("1234567890123456.78")
    min_area = Decimal("123.45")
    intent = ParsedSearchIntent(
        filters=SearchFilters(min_price=min_price, min_area_sqm=min_area),
    )

    assert intent.filters.min_price == min_price
    assert isinstance(intent.filters.min_price, Decimal)
    assert intent.filters.min_area_sqm == min_area
    assert isinstance(intent.filters.min_area_sqm, Decimal)


def test_model_dump_json_mode_is_json_safe():
    intent = ParsedSearchIntent(
        filters=SearchFilters(
            location="New Cairo",
            max_price=Decimal("5000000.50"),
        ),
        directives=[
            SearchFieldDirective(
                field="max_price",
                strength="soft",
                allow_alternative=True,
            )
        ],
        ambiguities=[SearchAmbiguity(reason="budget could mean total price")],
        unsupported_requirements=["sea view"],
    )

    dumped = intent.model_dump(mode="json")

    json.dumps(dumped)
    assert dumped["filters"]["max_price"] == "5000000.50"
    assert dumped["directives"] == [
        {
            "field": "max_price",
            "strength": "soft",
            "preferred": False,
            "allow_alternative": True,
        }
    ]
