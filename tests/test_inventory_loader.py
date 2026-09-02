from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from app.ai_engine.inventory_loader import (
    generate_plan_key,
    map_column_aliases,
    normalize_area,
    normalize_column_names,
    normalize_integer_count,
    normalize_money,
    normalize_property_type,
    process_inventory,
    validate_inventory,
)
from app.ai_engine.schemas import PaymentMethod, UnitStatus


def test_process_inventory_validates_sample_csv_with_old_safe_headers():
    file_path = Path("data/sample_inventory.csv")

    valid_units, errors = process_inventory(file_path)

    assert len(valid_units) == 4, _format_errors(errors)
    assert len(errors) == 2, _format_errors(errors)

    statuses_by_external_id = {
        unit.external_unit_id: unit.status
        for unit in valid_units
    }
    assert statuses_by_external_id["A101"] == UnitStatus.AVAILABLE
    assert statuses_by_external_id["B205"] == UnitStatus.HOLD
    assert statuses_by_external_id["D412"] == UnitStatus.SOLD

    unit_with_missing_source = next(
        unit for unit in valid_units if unit.external_unit_id == "A101"
    )
    assert unit_with_missing_source.source == file_path.name
    assert unit_with_missing_source.last_updated == date.today()
    assert unit_with_missing_source.base_price == 4200000.0
    assert unit_with_missing_source.payment_plans == []

    error_rows = {error["row_number"] for error in errors}
    assert error_rows == {6, 7}, _format_errors(errors)


def test_process_inventory_maps_company_column_aliases():
    file_path = Path("data/sample_alias_inventory.csv")

    valid_units, errors = process_inventory(file_path)

    assert errors == [], _format_errors(errors)
    assert len(valid_units) == 1

    unit = valid_units[0]
    assert unit.external_unit_id == "G701"
    assert unit.compound_name == "Cairo Gate"
    assert unit.property_type == "Apartment"
    assert unit.base_price == 6100000.0
    assert unit.status == UnitStatus.AVAILABLE
    assert unit.bedrooms == 3
    assert unit.payment_plans == []


def test_map_column_aliases_leaves_unknown_columns_unchanged():
    df = normalize_column_names(
        pd.DataFrame(
            columns=[
                "Project",
                "Total Base Price",
                "Broker Notes",
            ]
        )
    )

    mapped_df = map_column_aliases(df)

    assert list(mapped_df.columns) == [
        "compound_name",
        "base_price",
        "broker_notes",
    ]


def test_map_column_aliases_raises_on_collision():
    df = normalize_column_names(
        pd.DataFrame(
            columns=[
                "External Unit ID",
                "Unit ID",
            ]
        )
    )

    with pytest.raises(ValueError, match="external_unit_id"):
        map_column_aliases(df)


def test_map_column_aliases_does_not_map_generic_id_column():
    df = normalize_column_names(pd.DataFrame(columns=["ID", "Unit ID"]))

    mapped_df = map_column_aliases(df)

    assert list(mapped_df.columns) == ["id", "external_unit_id"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("EGP 5,500,000", 5500000.0),
        ("5.5M", 5500000.0),
        ("5.5 million", 5500000.0),
        (5500000, 5500000.0),
    ],
)
def test_normalize_money_parses_deterministic_values(value, expected):
    assert normalize_money(value) == expected


@pytest.mark.parametrize("value", ["around 5M", "starting from 5M", "contact for price"])
def test_normalize_money_leaves_ambiguous_values(value):
    assert normalize_money(value) == value


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("150 sqm", 150.0),
        ("150 m2", 150.0),
        ("150 m\u00b2", 150.0),
    ],
)
def test_normalize_area_parses_square_meter_values(value, expected):
    assert normalize_area(value) == expected


@pytest.mark.parametrize(
    ("value", "labels", "expected"),
    [
        ("3 BR", ["br", "bedroom", "bedrooms"], 3),
        ("3 bedrooms", ["br", "bedroom", "bedrooms"], 3),
        ("2 bathrooms", ["bathroom", "bathrooms"], 2),
    ],
)
def test_normalize_integer_count_parses_labeled_counts(value, labels, expected):
    assert normalize_integer_count(value, labels) == expected


@pytest.mark.parametrize("value", ["3-4 bedrooms", "3+ bedrooms", "studio"])
def test_normalize_integer_count_leaves_ambiguous_values(value):
    assert normalize_integer_count(value, ["br", "bedroom", "bedrooms"]) == value


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("apt", "Apartment"),
        ("TOWN HOUSE", "Townhouse"),
        ("twinhouse", "Twin House"),
    ],
)
def test_normalize_property_type_maps_safe_aliases(value, expected):
    assert normalize_property_type(value) == expected


def test_normalize_property_type_leaves_unknown_values():
    assert normalize_property_type("Loft") == "Loft"


def test_process_inventory_normalizes_messy_values_and_payment_plan():
    file_path = Path("data/sample_messy_inventory.csv")

    valid_units, errors = process_inventory(file_path)

    assert errors == [], _format_errors(errors)
    assert len(valid_units) == 1

    unit = valid_units[0]
    assert unit.external_unit_id == "M901"
    assert unit.compound_name == "River Walk"
    assert unit.property_type == "Twin House"
    assert unit.base_price == 5500000.0
    assert unit.area_sqm == 150.0
    assert unit.bedrooms == 3
    assert unit.bathrooms == 2
    assert unit.status == UnitStatus.AVAILABLE
    assert unit.source == file_path.name
    assert unit.last_updated == date.today()

    assert len(unit.payment_plans) == 1
    plan = unit.payment_plans[0]
    assert plan.plan_key == "name:5_years"
    assert plan.payment_method == PaymentMethod.INSTALLMENT
    assert plan.down_payment == 550000.0
    assert plan.total_price == 5800000.0


def test_validate_inventory_aggregates_multiple_payment_plan_rows():
    df = pd.DataFrame(
        [
            _row(plan_name="Cash", payment_method="Cash", total_price="EGP 7,500,000"),
            _row(plan_name="5 Years", payment_method="Installment", installment_years=5),
            _row(plan_name="7 Years", payment_method="Installment", installment_years=7),
        ]
    )

    valid_units, errors = validate_inventory(df, source_name="manual.csv")

    assert errors == [], _format_errors(errors)
    assert len(valid_units) == 1
    assert valid_units[0].external_unit_id == "A101"
    assert [plan.plan_key for plan in valid_units[0].payment_plans] == [
        "name:cash",
        "name:5_years",
        "name:7_years",
    ]


def test_validate_inventory_treats_currency_only_plan_data_as_invalid_plan():
    df = pd.DataFrame([_row(currency="EGP")])

    valid_units, errors = validate_inventory(df, source_name="manual.csv")

    assert valid_units == []
    assert len(errors) == 1
    assert "Cannot generate payment plan key" in errors[0]["message"]


def test_validate_inventory_groups_trimmed_external_unit_ids():
    df = pd.DataFrame(
        [
            _row(external_unit_id="A101", plan_name="Cash", payment_method="Cash"),
            _row(
                external_unit_id=" A101 ",
                plan_name="5 Years",
                payment_method="Installment",
                installment_years=5,
            ),
        ]
    )

    valid_units, errors = validate_inventory(df, source_name="manual.csv")

    assert errors == [], _format_errors(errors)
    assert len(valid_units) == 1
    assert valid_units[0].external_unit_id == "A101"
    assert len(valid_units[0].payment_plans) == 2


def test_validate_inventory_groups_integral_numeric_external_unit_ids():
    df = pd.DataFrame(
        [
            _row(external_unit_id=101, plan_name="Cash", payment_method="Cash"),
            _row(
                external_unit_id=101.0,
                plan_name="5 Years",
                payment_method="Installment",
                installment_years=5,
            ),
        ],
        dtype=object,
    )

    valid_units, errors = validate_inventory(df, source_name="manual.csv")

    assert errors == [], _format_errors(errors)
    assert len(valid_units) == 1
    assert valid_units[0].external_unit_id == "101"
    assert len(valid_units[0].payment_plans) == 2


def test_validate_inventory_preserves_string_external_unit_id_leading_zeros():
    df = pd.DataFrame([_row(external_unit_id="00101")])

    valid_units, errors = validate_inventory(df, source_name="manual.csv")

    assert errors == [], _format_errors(errors)
    assert len(valid_units) == 1
    assert valid_units[0].external_unit_id == "00101"


def test_validate_inventory_merges_when_later_plan_row_leaves_unit_values_blank():
    df = pd.DataFrame(
        [
            _row(plan_name="Cash", payment_method="Cash"),
            _row(
                compound_name=None,
                location=None,
                property_type=None,
                base_price=None,
                status=None,
                bedrooms=None,
                bathrooms=None,
                area_sqm=None,
                delivery_date=None,
                plan_name="5 Years",
                payment_method="Installment",
                installment_years=5,
            ),
        ]
    )

    valid_units, errors = validate_inventory(df, source_name="manual.csv")

    assert errors == [], _format_errors(errors)
    assert len(valid_units) == 1
    assert valid_units[0].compound_name == "Nile Gardens"
    assert valid_units[0].base_price == 8000000.0
    assert len(valid_units[0].payment_plans) == 2


def test_validate_inventory_merges_when_later_row_supplies_missing_unit_value():
    df = pd.DataFrame(
        [
            _row(bedrooms=None, plan_name="Cash", payment_method="Cash"),
            _row(
                bedrooms=3,
                plan_name="5 Years",
                payment_method="Installment",
                installment_years=5,
            ),
        ]
    )

    valid_units, errors = validate_inventory(df, source_name="manual.csv")

    assert errors == [], _format_errors(errors)
    assert len(valid_units) == 1
    assert valid_units[0].bedrooms == 3


def test_validate_inventory_reports_conflicting_repeated_unit_values():
    df = pd.DataFrame(
        [
            _row(plan_name="Cash", payment_method="Cash"),
            _row(plan_name="5 Years", payment_method="Installment", bedrooms=4),
        ]
    )

    valid_units, errors = validate_inventory(df, source_name="manual.csv")

    assert valid_units == []
    assert len(errors) == 1
    assert "Conflicting repeated unit-level value" in errors[0]["message"]
    assert "bedrooms" in errors[0]["message"]


def test_validate_inventory_reports_conflicting_repeated_base_price_values():
    df = pd.DataFrame(
        [
            _row(plan_name="Cash", payment_method="Cash", base_price=8000000.0),
            _row(
                plan_name="5 Years",
                payment_method="Installment",
                installment_years=5,
                base_price=8200000.0,
            ),
        ]
    )

    valid_units, errors = validate_inventory(df, source_name="manual.csv")

    assert valid_units == []
    assert len(errors) == 1
    assert "base_price" in errors[0]["message"]


def test_validate_inventory_reports_conflicting_repeated_status_values():
    df = pd.DataFrame(
        [
            _row(plan_name="Cash", payment_method="Cash", status="Available"),
            _row(
                plan_name="5 Years",
                payment_method="Installment",
                installment_years=5,
                status="Reserved",
            ),
        ]
    )

    valid_units, errors = validate_inventory(df, source_name="manual.csv")

    assert valid_units == []
    assert len(errors) == 1
    assert "status" in errors[0]["message"]


def test_validate_inventory_reports_duplicate_plan_key_within_unit():
    df = pd.DataFrame(
        [
            _row(plan_name="5 Years", payment_method="Installment", installment_years=5),
            _row(plan_name="5 Years", payment_method="Installment", installment_years=5),
        ]
    )

    valid_units, errors = validate_inventory(df, source_name="manual.csv")

    assert valid_units == []
    assert len(errors) == 1
    assert "Duplicate payment plan key" in errors[0]["message"]


def test_generate_plan_key_prefers_source_plan_id():
    plan_key = generate_plan_key(
        source_plan_id=" Plan 123 ",
        plan_name="5 Years",
        payment_method="Installment",
        installment_years=5,
    )

    assert plan_key == "source:plan_123"


def test_generate_plan_key_falls_back_to_plan_name():
    assert generate_plan_key(plan_name=" 5 Years ") == "name:5_years"


def test_generate_plan_key_falls_back_to_payment_terms():
    assert generate_plan_key(payment_method="Cash") == "terms:cash"
    assert generate_plan_key(payment_method="Installment", installment_years=7) == (
        "terms:installment:7"
    )


def test_generate_plan_key_does_not_use_mutable_financial_values():
    assert generate_plan_key(payment_method="Cash") == "terms:cash"


def test_generate_plan_key_raises_when_no_reliable_identity_exists():
    with pytest.raises(ValueError, match="Cannot generate"):
        generate_plan_key()


def _row(**overrides):
    data = {
        "external_unit_id": "A101",
        "compound_name": "Nile Gardens",
        "location": "New Cairo",
        "property_type": "Apartment",
        "base_price": 8000000.0,
        "status": "Available",
        "bedrooms": 3,
        "bathrooms": 2,
        "area_sqm": 150.0,
        "delivery_date": date(2028, 6, 30),
    }
    data.update(overrides)
    return data


def _format_errors(errors):
    return "\n".join(
        f"Row {error['row_number']}: {error['message']}"
        for error in errors
    )
