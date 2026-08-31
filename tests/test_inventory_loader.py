from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from app.ai_engine.inventory_loader import (
    map_column_aliases,
    normalize_area,
    normalize_column_names,
    normalize_integer_count,
    normalize_money,
    normalize_property_type,
    process_inventory,
)
from app.ai_engine.schemas import UnitStatus


def test_process_inventory_validates_sample_csv():
    file_path = Path("data/sample_inventory.csv")

    valid_units, errors = process_inventory(file_path)

    assert len(valid_units) == 4, _format_errors(errors)
    assert len(errors) == 2, _format_errors(errors)

    statuses_by_unit_id = {unit.unit_id: unit.status for unit in valid_units}
    assert statuses_by_unit_id["A101"] == UnitStatus.AVAILABLE
    assert statuses_by_unit_id["B205"] == UnitStatus.HOLD
    assert statuses_by_unit_id["D412"] == UnitStatus.SOLD

    unit_with_missing_source = next(unit for unit in valid_units if unit.unit_id == "A101")
    assert unit_with_missing_source.source == file_path.name
    assert unit_with_missing_source.last_updated == date.today()

    unit_with_missing_last_updated = next(unit for unit in valid_units if unit.unit_id == "D412")
    assert unit_with_missing_last_updated.last_updated == date.today()

    error_rows = {error["row_number"] for error in errors}
    assert error_rows == {6, 7}, _format_errors(errors)


def test_process_inventory_maps_company_column_aliases():
    file_path = Path("data/sample_alias_inventory.csv")

    valid_units, errors = process_inventory(file_path)

    assert errors == [], _format_errors(errors)
    assert len(valid_units) == 1

    unit = valid_units[0]
    assert unit.unit_id == "G701"
    assert unit.compound_name == "Cairo Gate"
    assert unit.property_type == "Apartment"
    assert unit.price == 6100000
    assert unit.status == UnitStatus.AVAILABLE
    assert unit.bedrooms == 3


def test_map_column_aliases_leaves_unknown_columns_unchanged():
    df = normalize_column_names(
        pd.DataFrame(
            columns=[
                "Project",
                "Total Price",
                "Broker Notes",
            ]
        )
    )

    mapped_df = map_column_aliases(df)

    assert list(mapped_df.columns) == [
        "compound_name",
        "price",
        "broker_notes",
    ]


def test_map_column_aliases_raises_on_collision():
    df = normalize_column_names(
        pd.DataFrame(
            columns=[
                "Compound Name",
                "Project",
            ]
        )
    )

    with pytest.raises(ValueError, match="compound_name"):
        map_column_aliases(df)


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
        ("150 m²", 150.0),
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


def test_process_inventory_normalizes_messy_values():
    file_path = Path("data/sample_messy_inventory.csv")

    valid_units, errors = process_inventory(file_path)

    assert errors == [], _format_errors(errors)
    assert len(valid_units) == 1

    unit = valid_units[0]
    assert unit.unit_id == "M901"
    assert unit.compound_name == "River Walk"
    assert unit.property_type == "Twin House"
    assert unit.price == 5500000.0
    assert unit.down_payment == 550000.0
    assert unit.area_sqm == 150.0
    assert unit.bedrooms == 3
    assert unit.bathrooms == 2
    assert unit.status == UnitStatus.AVAILABLE
    assert unit.source == file_path.name
    assert unit.last_updated == date.today()


def _format_errors(errors):
    return "\n".join(
        f"Row {error['row_number']}: {error['message']}"
        for error in errors
    )
