from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from app.ai_engine import inventory_ingestion
from app.ai_engine.inventory_loader import InventoryStructureError, _InventoryLoadResult
from app.ai_engine.inventory_ingestion import (
    InventoryAuthoritativeValidationError,
    InventoryPreparationError,
    InventoryProcessingResult,
    InventorySource,
    InventorySourceMalformedError,
    InventorySourceUnsupportedError,
    build_authoritative_inventory_snapshot,
)
from app.ai_engine.schemas import UnitSchema, UnitStatus


def test_csv_bytes_build_authoritative_snapshot():
    source = InventorySource(
        filename="inventory.csv",
        content=_csv_bytes([_row(unit=" A101 ", price="EGP 5.5M")]),
    )

    result = build_authoritative_inventory_snapshot(
        source=source,
        scope_key="developer:demo",
    )

    assert isinstance(result, InventoryProcessingResult)
    assert result.total_rows == 1
    assert result.accepted_units == 1
    assert result.source_filename == "inventory.csv"
    assert result.snapshot.scope_key == "developer:demo"
    unit = result.snapshot.units[0]
    assert unit.external_unit_id == "A101"
    assert unit.compound_key == "compound-a"
    assert unit.source == "inventory.csv"


def test_xlsx_bytes_build_authoritative_snapshot():
    source = InventorySource(
        filename="inventory.xlsx",
        content=_xlsx_bytes([_row(unit="B205", property="apt")]),
    )

    result = build_authoritative_inventory_snapshot(
        source=source,
        scope_key="developer:demo",
    )

    assert result.total_rows == 1
    assert result.accepted_units == 1
    assert result.snapshot.units[0].property_type == "Apartment"


def test_unsupported_extension_is_rejected():
    source = InventorySource(filename="inventory.pdf", content=b"%PDF")

    with pytest.raises(InventorySourceUnsupportedError):
        build_authoritative_inventory_snapshot(
            source=source,
            scope_key="developer:demo",
        )


@pytest.mark.parametrize(
    "source",
    [
        InventorySource(filename="inventory.csv", content=b"\xff\xfe\x00"),
        InventorySource(filename="inventory.xlsx", content=b"not an xlsx file"),
    ],
)
def test_malformed_source_is_rejected(source):
    with pytest.raises(InventorySourceMalformedError) as exc_info:
        build_authoritative_inventory_snapshot(
            source=source,
            scope_key="developer:demo",
        )

    assert exc_info.value.__cause__ is not None


def test_column_alias_collision_is_reported_as_malformed_source():
    source = InventorySource(
        filename="inventory.csv",
        content=_csv_bytes([_row(unit_id="A101")]),
    )

    with pytest.raises(InventorySourceMalformedError) as exc_info:
        build_authoritative_inventory_snapshot(
            source=source,
            scope_key="developer:demo",
        )

    assert isinstance(exc_info.value.__cause__, InventoryStructureError)


def test_empty_filename_is_rejected():
    with pytest.raises(InventoryPreparationError, match="filename"):
        InventorySource(filename=" ", content=b"unit")


def test_non_bytes_content_is_rejected():
    with pytest.raises(InventoryPreparationError, match="content"):
        InventorySource(filename="inventory.csv", content="unit")


def test_empty_scope_key_is_rejected():
    with pytest.raises(InventoryPreparationError, match="scope_key"):
        build_authoritative_inventory_snapshot(
            source=InventorySource(filename="inventory.csv", content=_csv_bytes([_row()])),
            scope_key=" ",
        )


def test_validation_error_prevents_authoritative_snapshot():
    source = InventorySource(
        filename="inventory.csv",
        content=_csv_bytes(
            [
                _row(unit="VALID"),
                _row(unit="INVALID", compound_code=""),
            ]
        ),
    )

    with pytest.raises(InventoryAuthoritativeValidationError) as exc_info:
        build_authoritative_inventory_snapshot(
            source=source,
            scope_key="developer:demo",
        )

    assert exc_info.value.errors[0].row_number == 3
    assert exc_info.value.errors[0].message == "Row failed inventory validation"
    assert isinstance(exc_info.value.errors, tuple)
    assert not hasattr(exc_info.value, "snapshot")


def test_nested_validation_errors_expose_only_stable_safe_fields():
    source = InventorySource(
        filename="inventory.csv",
        content=_csv_bytes([_row(availability="coming soon")]),
    )

    with pytest.raises(InventoryAuthoritativeValidationError) as exc_info:
        build_authoritative_inventory_snapshot(
            source=source,
            scope_key="developer:demo",
        )

    detail = exc_info.value.errors[0]
    issue = detail.errors[0]
    assert detail.row_number == 2
    assert detail.message == "Row failed schema validation"
    assert issue.type == "enum"
    assert issue.loc == ("status",)
    assert isinstance(issue.message, str)
    assert not hasattr(issue, "input")
    assert not hasattr(issue, "ctx")
    assert not hasattr(issue, "url")


def test_total_rows_counts_source_rows_not_grouped_units():
    source = InventorySource(
        filename="inventory.csv",
        content=_csv_bytes(
            [
                _row(plan="Cash", payment_method="Cash"),
                _row(plan="5 Years", payment_method="Installment", installment_years=5),
            ]
        ),
    )

    result = build_authoritative_inventory_snapshot(
        source=source,
        scope_key="developer:demo",
    )

    assert result.total_rows == 2
    assert result.accepted_units == 1
    assert [plan.plan_key for plan in result.snapshot.units[0].payment_plans] == [
        "name:cash",
        "name:5_years",
    ]


def test_unexpected_programming_error_is_not_reported_as_malformed(monkeypatch):
    def fail_processing(*_, **__):
        raise RuntimeError("bug in application code")

    monkeypatch.setattr(
        inventory_ingestion,
        "_prepare_inventory_units",
        fail_processing,
    )

    with pytest.raises(RuntimeError, match="bug in application code"):
        build_authoritative_inventory_snapshot(
            source=InventorySource(filename="inventory.csv", content=_csv_bytes([_row()])),
            scope_key="developer:demo",
        )


def test_unexpected_value_error_is_not_reported_as_malformed(monkeypatch):
    def fail_processing(*_, **__):
        raise ValueError("bug in application code")

    monkeypatch.setattr(
        inventory_ingestion,
        "_prepare_inventory_units",
        fail_processing,
    )

    with pytest.raises(ValueError, match="bug in application code"):
        build_authoritative_inventory_snapshot(
            source=InventorySource(filename="inventory.csv", content=_csv_bytes([_row()])),
            scope_key="developer:demo",
        )


def test_unexpected_os_error_is_not_reported_as_malformed_and_cleans_temp_file(monkeypatch):
    unlinked_paths = _record_unlinked_paths(monkeypatch)

    def fail_processing(*_, **__):
        raise OSError("internal filesystem failure")

    monkeypatch.setattr(
        inventory_ingestion,
        "_prepare_inventory_units",
        fail_processing,
    )

    with pytest.raises(OSError, match="internal filesystem failure"):
        build_authoritative_inventory_snapshot(
            source=InventorySource(filename="inventory.csv", content=_csv_bytes([_row()])),
            scope_key="developer:demo",
        )

    assert len(unlinked_paths) == 1
    assert not unlinked_paths[0].exists()


def test_snapshot_contract_failure_is_not_reported_as_malformed_source(monkeypatch):
    duplicate_unit = _unit_schema()

    def return_duplicate_units(*_, **__):
        return _InventoryLoadResult(
            total_rows=2,
            valid_units=[duplicate_unit, duplicate_unit],
            errors=[],
        )

    monkeypatch.setattr(
        inventory_ingestion,
        "_prepare_inventory_units",
        return_duplicate_units,
    )

    with pytest.raises(ValidationError, match="Duplicate"):
        build_authoritative_inventory_snapshot(
            source=InventorySource(filename="inventory.csv", content=_csv_bytes([_row()])),
            scope_key="developer:demo",
        )


def test_temp_file_is_cleaned_up_after_success(monkeypatch):
    unlinked_paths = _record_unlinked_paths(monkeypatch)

    build_authoritative_inventory_snapshot(
        source=InventorySource(filename="inventory.csv", content=_csv_bytes([_row()])),
        scope_key="developer:demo",
    )

    assert len(unlinked_paths) == 1
    assert not unlinked_paths[0].exists()


def test_temp_file_is_cleaned_up_after_validation_failure(monkeypatch):
    unlinked_paths = _record_unlinked_paths(monkeypatch)

    with pytest.raises(InventoryAuthoritativeValidationError):
        build_authoritative_inventory_snapshot(
            source=InventorySource(
                filename="inventory.csv",
                content=_csv_bytes([_row(compound_code="")]),
            ),
            scope_key="developer:demo",
        )

    assert len(unlinked_paths) == 1
    assert not unlinked_paths[0].exists()


def test_temp_file_is_cleaned_up_after_parser_failure(monkeypatch):
    unlinked_paths = _record_unlinked_paths(monkeypatch)

    with pytest.raises(InventorySourceMalformedError):
        build_authoritative_inventory_snapshot(
            source=InventorySource(filename="inventory.xlsx", content=b"not an xlsx"),
            scope_key="developer:demo",
        )

    assert len(unlinked_paths) == 1
    assert not unlinked_paths[0].exists()


def _record_unlinked_paths(monkeypatch) -> list[Path]:
    unlinked_paths = []
    original_unlink = Path.unlink

    def recording_unlink(self, *args, **kwargs):
        unlinked_paths.append(self)
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", recording_unlink)
    return unlinked_paths


def _csv_bytes(rows: list[dict[str, object]]) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _xlsx_bytes(rows: list[dict[str, object]]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, index=False)
    return output.getvalue()


def _row(**overrides) -> dict[str, object]:
    row = {
        "unit": "A101",
        "compound_code": "compound-a",
        "compound": "Nile Gardens",
        "city": "New Cairo",
        "property": "Apartment",
        "price": "5.5M",
        "availability": "available",
        "beds": 3,
        "baths": 2,
        "area": "150 sqm",
    }
    row.update(overrides)
    return row


def _unit_schema() -> UnitSchema:
    return UnitSchema(
        external_unit_id="A101",
        compound_key="compound-a",
        compound_name="Nile Gardens",
        location="New Cairo",
        property_type="Apartment",
        base_price=5500000.0,
        status=UnitStatus.AVAILABLE,
        source="inventory.csv",
        last_updated=date(2026, 9, 1),
    )
