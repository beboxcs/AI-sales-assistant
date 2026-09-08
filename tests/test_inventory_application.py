import inspect
from datetime import date

import pytest

from app.ai_engine import inventory_application
from app.ai_engine.change_detector import InventoryChange, InventoryChangeType
from app.ai_engine.inventory_application import (
    InventoryIngestionConflictError,
    InventoryIngestionError,
    InventoryIngestionPersistenceError,
    InventoryIngestionResult,
    InventoryScopeError,
    InventorySourceMalformedError,
    InventorySourceUnsupportedError,
    InventoryValidationFailedError,
    ingest_inventory,
)
from app.ai_engine.inventory_ingestion import (
    InventoryAuthoritativeValidationError as PreparationValidationError,
)
from app.ai_engine.inventory_ingestion import (
    InventoryProcessingResult,
    InventorySource,
    InventorySourceMalformedError as PreparationMalformedError,
    InventorySourceUnsupportedError as PreparationUnsupportedError,
)
from app.ai_engine.schemas import InventorySnapshot, UnitSchema, UnitStatus
from app.persistence.inventory import (
    InventoryPersistenceConflictError as BackendConflictError,
    InventoryPersistenceFailedError as BackendPersistenceError,
    InventoryScopeNotFoundError,
    InventorySnapshotLoadError,
)


def test_ingest_inventory_persists_snapshot_and_returns_change_counts(monkeypatch):
    source = InventorySource(filename="inventory.csv", content=b"csv")
    previous_snapshot = _snapshot(units=[_unit(external_unit_id="A101")])
    current_snapshot = _snapshot(units=[_unit(external_unit_id="A102")])
    changes = [
        _change("A102", InventoryChangeType.ADDED),
        _change("A101", InventoryChangeType.UPDATED),
        _change("A103", InventoryChangeType.MISSING),
    ]
    calls = []

    monkeypatch.setattr(
        inventory_application,
        "build_authoritative_inventory_snapshot",
        lambda *, source, scope_key: InventoryProcessingResult(
            snapshot=current_snapshot,
            total_rows=3,
            accepted_units=2,
            source_filename=source.filename,
        ),
    )
    monkeypatch.setattr(
        inventory_application,
        "load_current_inventory_snapshot",
        lambda *, company_id, scope_key: previous_snapshot,
    )
    monkeypatch.setattr(
        inventory_application,
        "detect_inventory_changes",
        lambda previous_snapshot, current_snapshot: changes,
    )

    def persist(*, company_id, snapshot, changes):
        calls.append((company_id, snapshot, changes))

    monkeypatch.setattr(inventory_application, "persist_inventory_ingestion", persist)

    result = ingest_inventory(company_id=7, scope_key="developer:demo", source=source)

    assert result == InventoryIngestionResult(
        scope_key="developer:demo",
        source_filename="inventory.csv",
        total_rows=3,
        accepted_units=2,
        rejected_rows=0,
        added_count=1,
        updated_count=1,
        missing_count=1,
        persisted=True,
    )
    assert calls == [(7, current_snapshot, changes)]


@pytest.mark.parametrize(
    ("preparation_error", "application_error"),
    [
        (
            PreparationUnsupportedError("internal unsupported detail"),
            InventorySourceUnsupportedError,
        ),
        (
            PreparationMalformedError("internal malformed detail"),
            InventorySourceMalformedError,
        ),
        (
            PreparationValidationError("internal validation detail", []),
            InventoryValidationFailedError,
        ),
    ],
)
def test_preparation_failures_short_circuit_before_backend_calls(
    monkeypatch,
    preparation_error,
    application_error,
):
    calls = []

    def fail_preparation(*, source, scope_key):
        raise preparation_error

    monkeypatch.setattr(
        inventory_application,
        "build_authoritative_inventory_snapshot",
        fail_preparation,
    )
    monkeypatch.setattr(
        inventory_application,
        "load_current_inventory_snapshot",
        lambda **_: calls.append("load"),
    )
    monkeypatch.setattr(
        inventory_application,
        "detect_inventory_changes",
        lambda *_: calls.append("detect"),
    )
    monkeypatch.setattr(
        inventory_application,
        "persist_inventory_ingestion",
        lambda **_: calls.append("persist"),
    )

    with pytest.raises(application_error) as exc_info:
        ingest_inventory(
            company_id=7,
            scope_key="developer:demo",
            source=InventorySource(filename="inventory.csv", content=b"csv"),
        )

    assert exc_info.value.__cause__ is preparation_error
    assert str(exc_info.value) != str(preparation_error)
    assert calls == []


def test_unexpected_internal_error_maps_to_base_ingestion_error(monkeypatch):
    internal_error = OSError("internal filesystem detail")
    calls = []

    def fail_preparation(*, source, scope_key):
        raise internal_error

    monkeypatch.setattr(
        inventory_application,
        "build_authoritative_inventory_snapshot",
        fail_preparation,
    )
    monkeypatch.setattr(
        inventory_application,
        "load_current_inventory_snapshot",
        lambda **_: calls.append("load"),
    )
    monkeypatch.setattr(
        inventory_application,
        "persist_inventory_ingestion",
        lambda **_: calls.append("persist"),
    )

    with pytest.raises(InventoryIngestionError) as exc_info:
        ingest_inventory(
            company_id=7,
            scope_key="developer:demo",
            source=InventorySource(filename="inventory.csv", content=b"csv"),
        )

    assert exc_info.value.__cause__ is internal_error
    assert "internal filesystem detail" not in str(exc_info.value)
    assert not isinstance(exc_info.value, InventorySourceMalformedError)
    assert calls == []


@pytest.mark.parametrize("scope_key", [" ", 123])
def test_invalid_scope_key_fails_before_preparation_or_backend_calls(
    monkeypatch,
    scope_key,
):
    calls = []

    monkeypatch.setattr(
        inventory_application,
        "build_authoritative_inventory_snapshot",
        lambda **_: calls.append("prepare"),
    )
    monkeypatch.setattr(
        inventory_application,
        "load_current_inventory_snapshot",
        lambda **_: calls.append("load"),
    )
    monkeypatch.setattr(
        inventory_application,
        "persist_inventory_ingestion",
        lambda **_: calls.append("persist"),
    )

    with pytest.raises(InventoryScopeError) as exc_info:
        ingest_inventory(
            company_id=7,
            scope_key=scope_key,
            source=InventorySource(filename="inventory.csv", content=b"csv"),
        )

    assert str(exc_info.value) == "Inventory scope is invalid"
    assert calls == []


def test_unknown_scope_maps_to_inventory_scope_error_without_persisting(monkeypatch):
    backend_error = InventoryScopeNotFoundError("internal scope detail")
    calls = []
    _stub_preparation(monkeypatch)

    def fail_load(*, company_id, scope_key):
        raise backend_error

    monkeypatch.setattr(inventory_application, "load_current_inventory_snapshot", fail_load)
    monkeypatch.setattr(
        inventory_application,
        "detect_inventory_changes",
        lambda *_: calls.append("detect"),
    )
    monkeypatch.setattr(
        inventory_application,
        "persist_inventory_ingestion",
        lambda **_: calls.append("persist"),
    )

    with pytest.raises(InventoryScopeError) as exc_info:
        ingest_inventory(
            company_id=7,
            scope_key="unknown",
            source=InventorySource(filename="inventory.csv", content=b"csv"),
        )

    assert exc_info.value.__cause__ is backend_error
    assert "internal scope detail" not in str(exc_info.value)
    assert calls == []


def test_snapshot_load_failure_maps_to_persistence_error_without_later_calls(monkeypatch):
    backend_error = InventorySnapshotLoadError("internal load detail")
    calls = []
    _stub_preparation(monkeypatch)

    def fail_load(*, company_id, scope_key):
        raise backend_error

    monkeypatch.setattr(inventory_application, "load_current_inventory_snapshot", fail_load)
    monkeypatch.setattr(
        inventory_application,
        "detect_inventory_changes",
        lambda *_: calls.append("detect"),
    )
    monkeypatch.setattr(
        inventory_application,
        "persist_inventory_ingestion",
        lambda **_: calls.append("persist"),
    )

    with pytest.raises(InventoryIngestionPersistenceError) as exc_info:
        ingest_inventory(
            company_id=7,
            scope_key="developer:demo",
            source=InventorySource(filename="inventory.csv", content=b"csv"),
        )

    assert exc_info.value.__cause__ is backend_error
    assert "internal load detail" not in str(exc_info.value)
    assert calls == []


def test_detector_value_error_maps_to_generic_ingestion_error_without_persisting(monkeypatch):
    detector_error = ValueError("duplicate internal identity")
    calls = []
    _stub_preparation(monkeypatch)
    _stub_previous_snapshot(monkeypatch)

    def fail_detection(previous_snapshot, current_snapshot):
        raise detector_error

    monkeypatch.setattr(inventory_application, "detect_inventory_changes", fail_detection)
    monkeypatch.setattr(
        inventory_application,
        "persist_inventory_ingestion",
        lambda **_: calls.append("persist"),
    )

    with pytest.raises(InventoryIngestionError) as exc_info:
        ingest_inventory(
            company_id=7,
            scope_key="developer:demo",
            source=InventorySource(filename="inventory.csv", content=b"csv"),
        )

    assert exc_info.value.__cause__ is detector_error
    assert not isinstance(exc_info.value, InventoryScopeError)
    assert "duplicate internal identity" not in str(exc_info.value)
    assert calls == []


def test_persistence_conflict_maps_to_ingestion_conflict(monkeypatch):
    backend_error = BackendConflictError("internal conflict detail")
    calls = []
    _stub_preparation(monkeypatch)
    _stub_previous_snapshot(monkeypatch)
    _stub_changes(monkeypatch)

    def fail_persist(*, company_id, snapshot, changes):
        calls.append("persist")
        raise backend_error

    monkeypatch.setattr(inventory_application, "persist_inventory_ingestion", fail_persist)

    with pytest.raises(InventoryIngestionConflictError) as exc_info:
        ingest_inventory(
            company_id=7,
            scope_key="developer:demo",
            source=InventorySource(filename="inventory.csv", content=b"csv"),
        )

    assert exc_info.value.__cause__ is backend_error
    assert "internal conflict detail" not in str(exc_info.value)
    assert calls == ["persist"]


def test_persistence_failure_maps_to_ingestion_persistence_error(monkeypatch):
    backend_error = BackendPersistenceError("internal database detail")
    calls = []
    _stub_preparation(monkeypatch)
    _stub_previous_snapshot(monkeypatch)
    _stub_changes(monkeypatch)

    def fail_persist(*, company_id, snapshot, changes):
        calls.append("persist")
        raise backend_error

    monkeypatch.setattr(inventory_application, "persist_inventory_ingestion", fail_persist)

    with pytest.raises(InventoryIngestionPersistenceError) as exc_info:
        ingest_inventory(
            company_id=7,
            scope_key="developer:demo",
            source=InventorySource(filename="inventory.csv", content=b"csv"),
        )

    assert exc_info.value.__cause__ is backend_error
    assert "internal database detail" not in str(exc_info.value)
    assert calls == ["persist"]


def test_persisted_true_is_returned_only_after_persistence_succeeds(monkeypatch):
    _stub_preparation(monkeypatch)
    _stub_previous_snapshot(monkeypatch)
    _stub_changes(monkeypatch)

    def fail_persist(**_):
        raise BackendPersistenceError("internal database detail")

    monkeypatch.setattr(inventory_application, "persist_inventory_ingestion", fail_persist)

    with pytest.raises(InventoryIngestionPersistenceError):
        ingest_inventory(
            company_id=7,
            scope_key="developer:demo",
            source=InventorySource(filename="inventory.csv", content=b"csv"),
        )


def test_inventory_ingestion_application_has_no_api_or_session_imports():
    source = inspect.getsource(inventory_application)

    assert "fastapi" not in source.lower()
    assert "UploadFile" not in source
    assert "sqlalchemy" not in source.lower()
    assert "SessionLocal" not in source


def _stub_preparation(monkeypatch, snapshot=None):
    prepared_snapshot = snapshot or _snapshot(units=[_unit()])

    monkeypatch.setattr(
        inventory_application,
        "build_authoritative_inventory_snapshot",
        lambda *, source, scope_key: InventoryProcessingResult(
            snapshot=prepared_snapshot,
            total_rows=1,
            accepted_units=1,
            source_filename=source.filename,
        ),
    )


def _stub_previous_snapshot(monkeypatch, snapshot=None):
    monkeypatch.setattr(
        inventory_application,
        "load_current_inventory_snapshot",
        lambda *, company_id, scope_key: snapshot or _snapshot(),
    )


def _stub_changes(monkeypatch, changes=None):
    monkeypatch.setattr(
        inventory_application,
        "detect_inventory_changes",
        lambda previous_snapshot, current_snapshot: changes or [],
    )


def _snapshot(scope_key="developer:demo", units=None) -> InventorySnapshot:
    return InventorySnapshot(scope_key=scope_key, units=units or [])


def _change(external_unit_id: str, change_type: InventoryChangeType) -> InventoryChange:
    return InventoryChange(
        compound_key="compound-a",
        external_unit_id=external_unit_id,
        change_type=change_type,
    )


def _unit(**overrides) -> UnitSchema:
    unit_data = {
        "external_unit_id": "A101",
        "compound_key": "compound-a",
        "compound_name": "Nile Gardens",
        "location": "New Cairo",
        "property_type": "Apartment",
        "base_price": 5500000.0,
        "status": UnitStatus.AVAILABLE,
        "source": "inventory.csv",
        "last_updated": date(2026, 9, 1),
    }
    unit_data.update(overrides)
    return UnitSchema(**unit_data)
