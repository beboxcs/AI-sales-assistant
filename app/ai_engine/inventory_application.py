"""Public application entry point for authoritative inventory ingestion."""

from dataclasses import dataclass

from app.ai_engine.change_detector import (
    InventoryChange,
    InventoryChangeType,
    detect_inventory_changes,
)
from app.ai_engine.inventory_ingestion import (
    InventoryAuthoritativeValidationError as PreparationValidationError,
    InventoryProcessingResult,
    InventorySource,
    InventorySourceMalformedError as PreparationMalformedError,
    InventorySourceUnsupportedError as PreparationUnsupportedError,
    build_authoritative_inventory_snapshot,
)
from app.ai_engine.schemas import InventorySnapshot
from app.persistence.inventory import (
    InventoryBackendError,
    InventoryPersistenceConflictError as BackendConflictError,
    InventoryPersistenceFailedError as BackendPersistenceError,
    InventoryScopeNotFoundError,
    InventorySnapshotLoadError,
    load_current_inventory_snapshot,
    persist_inventory_ingestion,
)


@dataclass(frozen=True)
class InventoryIngestionResult:
    scope_key: str
    source_filename: str
    total_rows: int
    accepted_units: int
    rejected_rows: int
    added_count: int
    updated_count: int
    missing_count: int
    persisted: bool


class InventoryIngestionError(Exception):
    """Base error exposed by the inventory ingestion application boundary."""


class InventorySourceUnsupportedError(InventoryIngestionError):
    """Raised when the uploaded inventory source type is unsupported."""


class InventorySourceMalformedError(InventoryIngestionError):
    """Raised when the uploaded inventory source cannot be parsed."""


class InventoryValidationFailedError(InventoryIngestionError):
    """Raised when authoritative inventory validation rejects any row."""


class InventoryScopeError(InventoryIngestionError):
    """Raised when the requested inventory scope cannot be used."""


class InventoryIngestionConflictError(InventoryIngestionError):
    """Raised when ingestion conflicts with existing backend state."""


class InventoryIngestionPersistenceError(InventoryIngestionError):
    """Raised when backend persistence cannot complete ingestion."""


def ingest_inventory(
    *,
    company_id: int,
    scope_key: str,
    source: InventorySource,
) -> InventoryIngestionResult:
    try:
        return _ingest_inventory(
            company_id=company_id,
            scope_key=scope_key,
            source=source,
        )
    except InventoryIngestionError:
        raise
    except Exception as exc:
        raise InventoryIngestionError("Inventory ingestion failed") from exc


def _ingest_inventory(
    *,
    company_id: int,
    scope_key: str,
    source: InventorySource,
) -> InventoryIngestionResult:
    _validate_scope_key(scope_key)
    prepared = _prepare_source(source=source, scope_key=scope_key)
    previous_snapshot = _load_previous_snapshot(
        company_id=company_id,
        scope_key=scope_key,
    )
    changes = detect_inventory_changes(
        previous_snapshot,
        prepared.snapshot,
    )
    _persist_ingestion(
        company_id=company_id,
        snapshot=prepared.snapshot,
        changes=changes,
    )
    return _ingestion_result(prepared, changes)


def _prepare_source(
    *,
    source: InventorySource,
    scope_key: str,
) -> InventoryProcessingResult:
    try:
        return build_authoritative_inventory_snapshot(
            source=source,
            scope_key=scope_key,
        )
    except PreparationUnsupportedError as exc:
        raise InventorySourceUnsupportedError(
            "Inventory source type is not supported"
        ) from exc
    except PreparationMalformedError as exc:
        raise InventorySourceMalformedError(
            "Inventory source could not be parsed"
        ) from exc
    except PreparationValidationError as exc:
        raise InventoryValidationFailedError(
            "Inventory source failed validation"
        ) from exc


def _validate_scope_key(scope_key: str) -> None:
    if not isinstance(scope_key, str):
        raise InventoryScopeError("Inventory scope is invalid")
    if not scope_key.strip():
        raise InventoryScopeError("Inventory scope is invalid")


def _load_previous_snapshot(*, company_id: int, scope_key: str) -> InventorySnapshot:
    try:
        return load_current_inventory_snapshot(
            company_id=company_id,
            scope_key=scope_key,
        )
    except InventoryScopeNotFoundError as exc:
        raise InventoryScopeError("Inventory scope was not found") from exc
    except InventorySnapshotLoadError as exc:
        raise InventoryIngestionPersistenceError(
            "Inventory snapshot could not be loaded"
        ) from exc


def _persist_ingestion(
    *,
    company_id: int,
    snapshot: InventorySnapshot,
    changes: list[InventoryChange],
) -> None:
    try:
        persist_inventory_ingestion(
            company_id=company_id,
            snapshot=snapshot,
            changes=changes,
        )
    except InventoryScopeNotFoundError as exc:
        raise InventoryScopeError("Inventory scope was not found") from exc
    except BackendConflictError as exc:
        raise InventoryIngestionConflictError(
            "Inventory ingestion conflicts with existing data"
        ) from exc
    except (BackendPersistenceError, InventoryBackendError) as exc:
        raise InventoryIngestionPersistenceError(
            "Inventory ingestion could not be persisted"
        ) from exc


def _ingestion_result(
    prepared: InventoryProcessingResult,
    changes: list[InventoryChange],
) -> InventoryIngestionResult:
    return InventoryIngestionResult(
        scope_key=prepared.snapshot.scope_key,
        source_filename=prepared.source_filename,
        total_rows=prepared.total_rows,
        accepted_units=prepared.accepted_units,
        rejected_rows=0,
        added_count=_count_changes(changes, InventoryChangeType.ADDED),
        updated_count=_count_changes(changes, InventoryChangeType.UPDATED),
        missing_count=_count_changes(changes, InventoryChangeType.MISSING),
        persisted=True,
    )


def _count_changes(
    changes: list[InventoryChange],
    change_type: InventoryChangeType,
) -> int:
    return sum(1 for change in changes if change.change_type == change_type)
