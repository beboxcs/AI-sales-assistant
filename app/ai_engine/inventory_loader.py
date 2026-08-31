from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from app.ai_engine.schemas import UnitSchema


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
    return validate_inventory(normalized_df, source_name=path.name)


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
