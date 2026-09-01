from datetime import date
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any

import pandas as pd
from pydantic import ValidationError

from app.ai_engine.schemas import Currency, PaymentMethod, PaymentPlanSchema, UnitSchema


COLUMN_ALIASES: dict[str, str] = {
    "unit": "external_unit_id",
    "unit_code": "external_unit_id",
    "unit_id": "external_unit_id",
    "unit_number": "external_unit_id",
    "unit_no": "external_unit_id",
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
    # Compatibility only: generic price semantics are source-dependent.
    # Production source-specific mappings should override or replace this.
    "price": "base_price",
    "total_base_price": "base_price",
    "base_unit_price": "base_price",
    "selling_price": "base_price",
    "unit_price": "base_price",
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
    "base_currency_code": "base_currency",
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
    "plan_id": "source_plan_id",
    "payment_plan_id": "source_plan_id",
    "plan": "plan_name",
    "payment_plan": "plan_name",
    "payment_plan_name": "plan_name",
    "payment_method": "payment_method",
    "payment_type": "payment_method",
    "downpayment": "down_payment",
    "down_payment_amount": "down_payment",
    "installment_year": "installment_years",
    "installment_years": "installment_years",
    "payment_years": "installment_years",
    "years": "installment_years",
    "plan_total_price": "total_price",
    "payment_plan_price": "total_price",
    "currency_code": "currency",
}

UNIT_LEVEL_FIELDS: tuple[str, ...] = (
    "external_unit_id",
    "compound_name",
    "location",
    "property_type",
    "base_price",
    "base_currency",
    "status",
    "bedrooms",
    "bathrooms",
    "area_sqm",
    "delivery_date",
    "source",
    "last_updated",
)

PAYMENT_PLAN_FIELDS: tuple[str, ...] = (
    "source_plan_id",
    "plan_name",
    "payment_method",
    "down_payment",
    "installment_years",
    "total_price",
    "currency",
)


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


def normalize_payment_method(value: Any) -> Any:
    if _is_missing(value):
        return value
    if isinstance(value, PaymentMethod):
        return value.value

    normalized_value = str(value).strip().lower()
    payment_method_map = {
        "cash": "Cash",
        "installment": "Installment",
        "installments": "Installment",
        "installment plan": "Installment",
    }

    return payment_method_map.get(normalized_value, value)


def normalize_currency(value: Any) -> Any:
    if _is_missing(value):
        return value
    if isinstance(value, Currency):
        return value.value

    normalized_value = str(value).strip().upper()
    if normalized_value == Currency.EGP.value:
        return Currency.EGP.value
    return value


def normalize_money(value: Any) -> Any:
    if _is_missing(value):
        return value
    if _is_number(value):
        return float(value)
    if not isinstance(value, str):
        return value

    normalized_value = value.strip()
    money_pattern = re.compile(
        r"^(?:egp\s*)?(\d+(?:\.\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?)"
        r"(?:\s*(m|million))?$",
        re.IGNORECASE,
    )
    match = money_pattern.fullmatch(normalized_value)
    if not match:
        return value

    amount = float(match.group(1).replace(",", ""))
    multiplier = 1_000_000 if match.group(2) else 1
    return amount * multiplier


def normalize_area(value: Any) -> Any:
    if _is_missing(value):
        return value
    if _is_number(value):
        return float(value)
    if not isinstance(value, str):
        return value

    area_pattern = re.compile(
        "^(\\d+(?:\\.\\d+)?)(?:\\s*(sqm|m2|m\u00b2))?$",
        re.IGNORECASE,
    )
    match = area_pattern.fullmatch(value.strip())
    if not match:
        return value

    return float(match.group(1))


def normalize_integer_count(value: Any, labels: list[str]) -> Any:
    if _is_missing(value):
        return value
    if _is_integer(value):
        return int(value)
    if not isinstance(value, str):
        return value

    escaped_labels = [re.escape(label) for label in labels]
    count_pattern = re.compile(
        rf"^(\d+)(?:\s*(?:{'|'.join(escaped_labels)}))?$",
        re.IGNORECASE,
    )
    match = count_pattern.fullmatch(value.strip())
    if not match:
        return value

    return int(match.group(1))


def normalize_property_type(value: Any) -> Any:
    if _is_missing(value):
        return value
    if not isinstance(value, str):
        return value

    property_type_map = {
        "apt": "Apartment",
        "apartment": "Apartment",
        "villa": "Villa",
        "townhouse": "Townhouse",
        "town house": "Townhouse",
        "twinhouse": "Twin House",
        "twin house": "Twin House",
        "duplex": "Duplex",
        "penthouse": "Penthouse",
        "chalet": "Chalet",
    }

    normalized_value = value.strip().lower()
    return property_type_map.get(normalized_value, value)


def normalize_row_values(row: dict[str, Any]) -> dict[str, Any]:
    normalized_row = row.copy()

    if "base_price" in normalized_row:
        normalized_row["base_price"] = normalize_money(normalized_row["base_price"])
    if "base_currency" in normalized_row:
        normalized_row["base_currency"] = normalize_currency(normalized_row["base_currency"])
    if "down_payment" in normalized_row:
        normalized_row["down_payment"] = normalize_money(normalized_row["down_payment"])
    if "total_price" in normalized_row:
        normalized_row["total_price"] = normalize_money(normalized_row["total_price"])
    if "currency" in normalized_row:
        normalized_row["currency"] = normalize_currency(normalized_row["currency"])
    if "payment_method" in normalized_row:
        normalized_row["payment_method"] = normalize_payment_method(
            normalized_row["payment_method"]
        )
    if "area_sqm" in normalized_row:
        normalized_row["area_sqm"] = normalize_area(normalized_row["area_sqm"])
    if "bedrooms" in normalized_row:
        normalized_row["bedrooms"] = normalize_integer_count(
            normalized_row["bedrooms"],
            labels=["br", "bedroom", "bedrooms"],
        )
    if "bathrooms" in normalized_row:
        normalized_row["bathrooms"] = normalize_integer_count(
            normalized_row["bathrooms"],
            labels=["bathroom", "bathrooms"],
        )
    if "property_type" in normalized_row:
        normalized_row["property_type"] = normalize_property_type(
            normalized_row["property_type"]
        )
    if "status" in normalized_row:
        normalized_row["status"] = normalize_status(normalized_row["status"])

    return normalized_row


def generate_plan_key(
    source_plan_id: Any = None,
    plan_name: Any = None,
    payment_method: Any = None,
    installment_years: Any = None,
) -> str:
    if not _is_missing(source_plan_id):
        return f"source:{_normalize_key_component(source_plan_id)}"
    if not _is_missing(plan_name):
        return f"name:{_normalize_key_component(plan_name)}"
    if not _is_missing(payment_method):
        normalized_payment_method = normalize_payment_method(payment_method)
        if normalized_payment_method == PaymentMethod.CASH.value:
            return "terms:cash"
        if normalized_payment_method == PaymentMethod.INSTALLMENT.value:
            if _is_missing(installment_years):
                raise ValueError(
                    "Cannot generate installment payment plan key without "
                    "installment_years"
                )
            return f"terms:installment:{_format_number_for_key(installment_years)}"

    raise ValueError("Cannot generate payment plan key from available values")


def validate_inventory(
    df: pd.DataFrame,
    source_name: str,
) -> tuple[list[UnitSchema], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    grouped_units: dict[str, dict[str, Any]] = {}

    for row_number, (_, row) in enumerate(df.iterrows(), start=2):
        row_data = _clean_missing_values(row.to_dict())
        row_data = normalize_row_values(row_data)

        if _is_missing(row_data.get("external_unit_id")):
            errors.append(
                _validation_error(
                    row_number,
                    "external_unit_id is required for inventory rows",
                )
            )
            continue

        external_unit_id = _normalize_external_unit_id(row_data["external_unit_id"])
        row_data["external_unit_id"] = external_unit_id
        unit_data = {
            field: row_data.get(field)
            for field in UNIT_LEVEL_FIELDS
            if field in row_data
        }
        group = grouped_units.setdefault(
            external_unit_id,
            {
                "unit_data": unit_data,
                "payment_plans": [],
                "plan_keys": set(),
                "row_numbers": [],
                "has_error": False,
            },
        )
        group["row_numbers"].append(row_number)

        if group["unit_data"] is not unit_data:
            conflict_message = _merge_unit_level_values(
                group["unit_data"],
                unit_data,
            )
            if conflict_message:
                group["has_error"] = True
                errors.append(_validation_error(row_number, conflict_message))
                continue

        if _row_has_payment_plan(row_data):
            try:
                payment_plan = _build_payment_plan(row_data)
            except (ValueError, ValidationError) as exc:
                group["has_error"] = True
                errors.append(_validation_error(row_number, str(exc)))
                continue

            if payment_plan.plan_key in group["plan_keys"]:
                group["has_error"] = True
                errors.append(
                    _validation_error(
                        row_number,
                        "Duplicate payment plan key found for unit "
                        f"{external_unit_id}: {payment_plan.plan_key}",
                    )
                )
                continue

            group["plan_keys"].add(payment_plan.plan_key)
            group["payment_plans"].append(payment_plan)

    valid_units: list[UnitSchema] = []
    for external_unit_id in sorted(grouped_units):
        group = grouped_units[external_unit_id]
        if group["has_error"]:
            continue

        unit_data = group["unit_data"].copy()
        if _is_missing(unit_data.get("source")):
            unit_data["source"] = source_name
        if _is_missing(unit_data.get("last_updated")):
            unit_data["last_updated"] = date.today()
        unit_data["payment_plans"] = group["payment_plans"]

        try:
            valid_units.append(UnitSchema(**unit_data))
        except ValidationError as exc:
            errors.append(
                {
                    "row_number": group["row_numbers"][0],
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


def _build_payment_plan(row_data: dict[str, Any]) -> PaymentPlanSchema:
    payment_plan_data = {
        field: row_data.get(field)
        for field in PAYMENT_PLAN_FIELDS
        if not _is_missing(row_data.get(field))
    }
    payment_plan_data["plan_key"] = generate_plan_key(
        source_plan_id=payment_plan_data.get("source_plan_id"),
        plan_name=payment_plan_data.get("plan_name"),
        payment_method=payment_plan_data.get("payment_method"),
        installment_years=payment_plan_data.get("installment_years"),
    )
    return PaymentPlanSchema(**payment_plan_data)


def _row_has_payment_plan(row_data: dict[str, Any]) -> bool:
    identity_or_value_fields = (
        "source_plan_id",
        "plan_name",
        "payment_method",
        "down_payment",
        "installment_years",
        "total_price",
        "currency",
    )
    return any(not _is_missing(row_data.get(field)) for field in identity_or_value_fields)


def _merge_unit_level_values(
    existing_unit_data: dict[str, Any],
    new_unit_data: dict[str, Any],
) -> str | None:
    for field in UNIT_LEVEL_FIELDS:
        existing_value = existing_unit_data.get(field)
        new_value = new_unit_data.get(field)

        if _is_missing(existing_value) and not _is_missing(new_value):
            existing_unit_data[field] = new_value
            continue
        if not _is_missing(existing_value) and _is_missing(new_value):
            continue

        if not _values_match(existing_value, new_value):
            return (
                "Conflicting repeated unit-level value for "
                f"{new_unit_data.get('external_unit_id')} field {field}: "
                f"{existing_value!r} != {new_value!r}"
            )
    return None


def _normalize_external_unit_id(value: Any) -> str:
    if _is_integer(value):
        return str(int(value))
    return str(value).strip()


def _values_match(left: Any, right: Any) -> bool:
    if _is_missing(left) and _is_missing(right):
        return True
    return left == right


def _validation_error(row_number: int, message: str) -> dict[str, Any]:
    return {
        "row_number": row_number,
        "errors": [],
        "message": message,
    }


def _normalize_key_component(value: Any) -> str:
    normalized_value = str(value).strip().lower()
    normalized_value = re.sub(r"\s+", "_", normalized_value)
    if not normalized_value:
        raise ValueError("Payment plan key component must not be empty")
    return normalized_value


def _format_number_for_key(value: Any) -> str:
    number_value = float(value)
    if number_value.is_integer():
        return str(int(number_value))
    return str(number_value)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return bool(pd.isna(value))


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_integer(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, Integral):
        return True
    if isinstance(value, Real):
        return value.is_integer()
    return False
