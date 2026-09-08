from fastapi.testclient import TestClient
import pytest

from app.api.dependencies import (
    get_inventory_ingest_max_upload_bytes,
    get_inventory_ingestion_service,
    get_trusted_company_id,
)
from app.api.inventory_ingestion_seam import (
    InventoryIngestionError,
    InventoryIngestionResult,
    InventorySource,
)
from app.main import app


def test_csv_upload_is_accepted_and_passed_as_inventory_source(
    ingestion_client_factory,
):
    client, calls = ingestion_client_factory()

    response = client.post(
        "/api/v1/inventory/ingest",
        data={"scope_key": "scope:nc"},
        files={"file": ("inventory.csv", b"external_unit_id\nA101\n", "text/csv")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "accepted",
        "filename": "inventory.csv",
        "scope_key": "scope:nc",
    }
    assert len(calls) == 1
    assert calls[0]["company_id"] == 42
    assert calls[0]["scope_key"] == "scope:nc"
    assert calls[0]["source"] == InventorySource(
        filename="inventory.csv",
        content=b"external_unit_id\nA101\n",
    )
    assert not hasattr(calls[0]["source"], "file")


def test_xlsx_upload_is_accepted(ingestion_client_factory):
    client, calls = ingestion_client_factory()

    response = client.post(
        "/api/v1/inventory/ingest",
        data={"scope_key": "scope:nc"},
        files={
            "file": (
                "inventory.xlsx",
                b"fake-xlsx-bytes",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 200
    assert calls[0]["source"].filename == "inventory.xlsx"


def test_invalid_extension_is_rejected(ingestion_client_factory):
    client, calls = ingestion_client_factory()

    response = client.post(
        "/api/v1/inventory/ingest",
        data={"scope_key": "scope:nc"},
        files={"file": ("inventory.txt", b"unit", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "INVALID_INVENTORY_FILE_TYPE",
            "message": "Inventory upload must be a CSV or XLSX file",
        }
    }
    assert calls == []


def test_pdf_upload_is_rejected(ingestion_client_factory):
    client, calls = ingestion_client_factory()

    response = client.post(
        "/api/v1/inventory/ingest",
        data={"scope_key": "scope:nc"},
        files={"file": ("inventory.pdf", b"%PDF", "application/pdf")},
    )

    assert response.status_code == 400
    assert calls == []


def test_missing_scope_key_is_rejected_by_validation(ingestion_client_factory):
    client, calls = ingestion_client_factory()

    response = client.post(
        "/api/v1/inventory/ingest",
        files={"file": ("inventory.csv", b"unit", "text/csv")},
    )

    assert response.status_code == 422
    assert calls == []


def test_upload_over_size_limit_is_rejected(ingestion_client_factory):
    client, calls = ingestion_client_factory(max_upload_bytes=3)

    response = client.post(
        "/api/v1/inventory/ingest",
        data={"scope_key": "scope:nc"},
        files={"file": ("inventory.csv", b"1234", "text/csv")},
    )

    assert response.status_code == 413
    assert response.json() == {
        "error": {
            "code": "INVENTORY_UPLOAD_TOO_LARGE",
            "message": "Inventory upload exceeds the configured size limit",
        }
    }
    assert calls == []


def test_client_cannot_override_company_id(ingestion_client_factory):
    client, calls = ingestion_client_factory()

    response = client.post(
        "/api/v1/inventory/ingest",
        data={"scope_key": "scope:nc", "company_id": "999"},
        files={"file": ("inventory.csv", b"unit", "text/csv")},
    )

    assert response.status_code == 200
    assert calls[0]["company_id"] == 42


def test_application_service_is_called_exactly_once(ingestion_client_factory):
    client, calls = ingestion_client_factory()

    response = client.post(
        "/api/v1/inventory/ingest",
        data={"scope_key": "scope:nc"},
        files={"file": ("inventory.csv", b"unit", "text/csv")},
    )

    assert response.status_code == 200
    assert len(calls) == 1


def test_application_level_errors_are_mapped_to_safe_response(
    ingestion_client_factory,
):
    client, _ = ingestion_client_factory(
        error=InventoryIngestionError("postgresql://secret SELECT *")
    )

    response = client.post(
        "/api/v1/inventory/ingest",
        data={"scope_key": "scope:nc"},
        files={"file": ("inventory.csv", b"unit", "text/csv")},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "INVENTORY_INGESTION_FAILED",
            "message": "Inventory ingestion could not be completed",
        }
    }
    assert "postgresql://secret" not in response.text
    assert "SELECT *" not in response.text


def test_unexpected_application_errors_are_mapped_to_safe_response(
    ingestion_client_factory,
):
    client, _ = ingestion_client_factory(
        error=RuntimeError("sqlalchemy internal detail")
    )

    response = client.post(
        "/api/v1/inventory/ingest",
        data={"scope_key": "scope:nc"},
        files={"file": ("inventory.csv", b"unit", "text/csv")},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "INVENTORY_INGESTION_FAILED",
            "message": "Inventory ingestion could not be completed",
        }
    }
    assert "sqlalchemy internal detail" not in response.text


def test_missing_uploaded_file_is_rejected_by_validation(ingestion_client_factory):
    client, calls = ingestion_client_factory()

    response = client.post(
        "/api/v1/inventory/ingest",
        data={"scope_key": "scope:nc"},
    )

    assert response.status_code == 422
    assert calls == []


@pytest.fixture
def ingestion_client_factory():
    original_overrides = app.dependency_overrides.copy()
    clients = []

    def factory(
        *,
        max_upload_bytes: int = 1024,
        error: Exception | None = None,
    ) -> tuple[TestClient, list[dict]]:
        calls = []

        def ingest(*, company_id, scope_key, source):
            if error is not None:
                raise error
            calls.append(
                {
                    "company_id": company_id,
                    "scope_key": scope_key,
                    "source": source,
                }
            )
            return InventoryIngestionResult(
                status="accepted",
                filename=source.filename,
                scope_key=scope_key,
            )

        app.dependency_overrides[get_trusted_company_id] = lambda: 42
        app.dependency_overrides[get_inventory_ingestion_service] = lambda: ingest
        app.dependency_overrides[get_inventory_ingest_max_upload_bytes] = (
            lambda: max_upload_bytes
        )
        client = TestClient(app)
        clients.append(client)
        return client, calls

    try:
        yield factory
    finally:
        for client in clients:
            client.close()
        app.dependency_overrides = original_overrides
