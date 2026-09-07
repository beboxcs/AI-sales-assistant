from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from zipfile import BadZipFile

import pandas as pd
from openpyxl.utils.exceptions import InvalidFileException

from app.ai_engine.inventory_loader import (
    InventoryStructureError,
    _prepare_inventory_units,
)
from app.ai_engine.schemas import InventorySnapshot


class InventoryPreparationError(Exception):
    """Base error for authoritative inventory snapshot preparation."""


class InventorySourceUnsupportedError(InventoryPreparationError):
    """Raised when the source filename extension is not supported."""


class InventorySourceMalformedError(InventoryPreparationError):
    """Raised when source bytes cannot be parsed or mapped."""


class InventoryAuthoritativeValidationError(InventoryPreparationError):
    """Raised when any row fails validation for an authoritative snapshot."""

    def __init__(self, message: str, errors: list[dict[str, Any]]):
        super().__init__(message)
        self.errors = tuple(
            InventoryValidationErrorDetail.from_loader_error(error)
            for error in errors
        )


@dataclass(frozen=True)
class InventoryValidationErrorDetail:
    row_number: int | None
    message: str | None
    errors: tuple[InventoryValidationIssue, ...]

    @classmethod
    def from_loader_error(cls, error: dict[str, Any]) -> "InventoryValidationErrorDetail":
        row_number = error.get("row_number")
        issues = _safe_validation_issues(error.get("errors", []))
        return cls(
            row_number=row_number if isinstance(row_number, int) else None,
            message=_safe_outer_message(issues),
            errors=issues,
        )


@dataclass(frozen=True)
class InventoryValidationIssue:
    type: str | None
    loc: tuple[str | int, ...]
    message: str | None

    @classmethod
    def from_pydantic_error(
        cls,
        error: object,
    ) -> "InventoryValidationIssue | None":
        if not isinstance(error, dict):
            return None

        return cls(
            type=_safe_text(error.get("type")),
            loc=_safe_loc(error.get("loc")),
            message=_safe_text(error.get("msg")),
        )


@dataclass(frozen=True)
class InventorySource:
    filename: str
    content: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.filename, str):
            raise InventoryPreparationError("filename must be a string")
        if not self.filename.strip():
            raise InventoryPreparationError("filename must not be empty")
        if not isinstance(self.content, bytes):
            raise InventoryPreparationError("content must be bytes")


@dataclass(frozen=True)
class InventoryProcessingResult:
    snapshot: InventorySnapshot
    total_rows: int
    accepted_units: int
    source_filename: str


def build_authoritative_inventory_snapshot(
    *,
    source: InventorySource,
    scope_key: str,
) -> InventoryProcessingResult:
    if not isinstance(scope_key, str):
        raise InventoryPreparationError("scope_key must be a string")
    if not scope_key.strip():
        raise InventoryPreparationError("scope_key must not be empty")

    suffix = Path(source.filename).suffix.lower()
    if suffix not in {".csv", ".xlsx"}:
        raise InventorySourceUnsupportedError(
            f"Unsupported inventory file type: {suffix}"
        )

    with _temporary_inventory_file(source.content, suffix) as temp_path:
        try:
            prepared = _prepare_inventory_units(
                temp_path,
                source_name=source.filename,
            )
        except _MALFORMED_SOURCE_ERRORS as exc:
            raise InventorySourceMalformedError(
                "Inventory source could not be parsed"
            ) from exc

        if prepared.errors:
            raise InventoryAuthoritativeValidationError(
                "Inventory source contains validation errors",
                prepared.errors,
            )

        snapshot = InventorySnapshot(
            scope_key=scope_key,
            units=prepared.valid_units,
        )

        return InventoryProcessingResult(
            snapshot=snapshot,
            total_rows=prepared.total_rows,
            accepted_units=len(prepared.valid_units),
            source_filename=source.filename,
        )


_MALFORMED_SOURCE_ERRORS = (
    BadZipFile,
    InventoryStructureError,
    InvalidFileException,
    UnicodeError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
)


@contextmanager
def _temporary_inventory_file(content: bytes, suffix: str) -> Iterator[Path]:
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(content)
        yield temp_path
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _safe_validation_issues(value: object) -> tuple[InventoryValidationIssue, ...]:
    if not isinstance(value, list | tuple):
        return ()

    issues = []
    for item in value:
        issue = InventoryValidationIssue.from_pydantic_error(item)
        if issue is not None:
            issues.append(issue)
    return tuple(issues)


def _safe_outer_message(issues: tuple[InventoryValidationIssue, ...]) -> str:
    if issues:
        return "Row failed schema validation"
    return "Row failed inventory validation"


def _safe_loc(value: object) -> tuple[str | int, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, str | int))


def _safe_text(value: object) -> str | None:
    return value if isinstance(value, str) else None
