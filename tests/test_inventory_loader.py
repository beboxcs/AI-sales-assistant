from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from app.ai_engine.inventory_loader import (
    map_column_aliases,
    normalize_column_names,
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


def _format_errors(errors):
    return "\n".join(
        f"Row {error['row_number']}: {error['message']}"
        for error in errors
    )
