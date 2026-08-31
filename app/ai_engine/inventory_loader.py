from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from app.ai_engine.schemas import UnitSchema


COLUMN_ALIASES: dict[str, str] = {
    "id": "unit_id",
    "unit": "unit_id",
    "unit_code": "unit_id",
    "unit_number": "unit_id",
    "unit_no": "unit_id",
    "compound": "compound_name",
    "project": "compound_name",
    "project_name": "compound_name",
    "development": "compound_name",
    "city": "location",
    "area_location": "location",
    "district": "location",
    "zone": "location",
    "type": "property_type",
    "unit_type": "property_type",
    "property": "property_type",
    "total_price": "price",
    "selling_price": "price",
    "unit_price": "price",
    "beds": "bedrooms",
    "bedroom": "bedrooms",
    "no_of_bedrooms": "bedrooms",
    "baths": "bathrooms",
    "bathroom": "bathrooms",
    "area": "area_sqm",
    "unit_area": "area_sqm",
    "built_up_area": "area_sqm",
    "availability": "status",
    "unit_status": "status",
    "data_source": "source",
    "source_name": "source",
    "provider": "source",
    "updated_at": "last_updated",
    "last_update": "last_updated",
    "last_update_date": "last_updated",
    "update_date": "last_updated",
    "delivery": "delivery_date",
    "handover_date": "delivery_date",
    "completion_date": "delivery_date",
    "downpayment": "down_payment",
    "down_payment_amount": "down_payment",
    "installment_year": "installment_years",
    "years": "installment_years",
}


def load_inventory(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".csv":
        return pd.read_csv(path)
    if extension == ".xlsx":
        return pd.read_excel(path)

    raise ValueError(f"Unsupported inventory file type: {extension}")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = df.copy()
    normalized_df.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in normalized_df.columns
    ]
    return normalized_df


def map_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    mapped_df = df.copy()
    mapped_columns = [
        COLUMN_ALIASES.get(column, column)
        for column in mapped_df.columns
    ]

    _raise_for_column_collisions(list(mapped_df.columns), mapped_columns)

    mapped_df.columns = mapped_columns
    return mapped_df


def normalize_status(value: Any) -> Any:
    if _is_missing(value):
        return value

    normalized_value = str(value).strip()
    status_map = {
        "available": "Available",
        "hold": "Hold",
        "reserved": "Reserved",
        "sold": "Sold",
    }

    return status_map.get(normalized_value.lower(), value)


def validate_inventory(
    df: pd.DataFrame,
    source_name: str,
) -> tuple[list[UnitSchema], list[dict[str, Any]]]:
    valid_units: list[UnitSchema] = []
    errors: list[dict[str, Any]] = []

    for row_number, (_, row) in enumerate(df.iterrows(), start=2):
        row_data = _clean_missing_values(row.to_dict())

        if "status" in row_data:
            row_data["status"] = normalize_status(row_data["status"])

        if _is_missing(row_data.get("source")):
            row_data["source"] = source_name

        if _is_missing(row_data.get("last_updated")):
            row_data["last_updated"] = date.today()

        try:
            valid_units.append(UnitSchema(**row_data))
        except ValidationError as exc:
            errors.append(
                {
                    "row_number": row_number,
                    "errors": exc.errors(),
                    "message": str(exc),
                }
            )

    return valid_units, errors


def process_inventory(file_path: str | Path) -> tuple[list[UnitSchema], list[dict[str, Any]]]:
    path = Path(file_path)
    df = load_inventory(path)
    normalized_df = normalize_column_names(df)
    mapped_df = map_column_aliases(normalized_df)
    return validate_inventory(mapped_df, source_name=path.name)


def _raise_for_column_collisions(
    original_columns: list[str],
    mapped_columns: list[str],
) -> None:
    columns_by_target: dict[str, list[str]] = {}

    for original_column, mapped_column in zip(original_columns, mapped_columns):
        columns_by_target.setdefault(mapped_column, []).append(original_column)

    collisions = {
        target: originals
        for target, originals in columns_by_target.items()
        if len(originals) > 1
    }

    if not collisions:
        return

    collision_details = [
        f"{originals} -> {target}"
        for target, originals in collisions.items()
    ]
    raise ValueError(
        "Column alias collision detected: "
        + "; ".join(collision_details)
    )


def _clean_missing_values(row_data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: None if _is_missing(value) else value
        for key, value in row_data.items()
    }


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return bool(pd.isna(value))
