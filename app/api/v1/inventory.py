from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel

from app.ai_engine.inventory_application import (
    InventoryIngestionConflictError,
    InventoryIngestionError,
    InventoryIngestionPersistenceError,
    InventoryScopeError,
    InventorySourceMalformedError,
    InventorySourceUnsupportedError,
    InventoryValidationFailedError,
)
from app.ai_engine.inventory_ingestion import InventorySource
from app.api.dependencies import (
    InventoryIngestionService,
    get_inventory_ingest_max_upload_bytes,
    get_inventory_ingestion_service,
    get_trusted_company_id,
)
from app.api.errors import APIError

router = APIRouter(prefix="/inventory", tags=["inventory"])

_ALLOWED_INVENTORY_EXTENSIONS = {".csv", ".xlsx"}


class InventoryIngestionResponse(BaseModel):
    scope_key: str
    source_filename: str
    total_rows: int
    accepted_units: int
    rejected_rows: int
    added_count: int
    updated_count: int
    missing_count: int
    persisted: bool


@router.post("/ingest", response_model=InventoryIngestionResponse)
async def ingest_inventory(
    scope_key: str = Form(...),
    file: UploadFile = File(...),
    company_id: int = Depends(get_trusted_company_id),
    ingest: InventoryIngestionService = Depends(get_inventory_ingestion_service),
    max_upload_bytes: int = Depends(get_inventory_ingest_max_upload_bytes),
) -> InventoryIngestionResponse:
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()
    if extension not in _ALLOWED_INVENTORY_EXTENSIONS:
        raise APIError(
            status_code=400,
            code="INVALID_INVENTORY_FILE_TYPE",
            message="Inventory upload must be a CSV or XLSX file",
        )

    content = await file.read()
    if len(content) > max_upload_bytes:
        raise APIError(
            status_code=413,
            code="INVENTORY_UPLOAD_TOO_LARGE",
            message="Inventory upload exceeds the configured size limit",
        )

    try:
        result = ingest(
            company_id=company_id,
            scope_key=scope_key,
            source=InventorySource(filename=filename, content=content),
        )
    except InventoryIngestionError as exc:
        raise _inventory_ingestion_api_error(exc) from exc
    except APIError:
        raise
    except Exception as exc:
        raise APIError(
            status_code=500,
            code="INVENTORY_INGESTION_FAILED",
            message="Inventory ingestion could not be completed",
        ) from exc

    return InventoryIngestionResponse(
        scope_key=result.scope_key,
        source_filename=result.source_filename,
        total_rows=result.total_rows,
        accepted_units=result.accepted_units,
        rejected_rows=result.rejected_rows,
        added_count=result.added_count,
        updated_count=result.updated_count,
        missing_count=result.missing_count,
        persisted=result.persisted,
    )


def _inventory_ingestion_api_error(exc: InventoryIngestionError) -> APIError:
    if isinstance(exc, InventorySourceUnsupportedError):
        return APIError(
            status_code=400,
            code="INVALID_INVENTORY_FILE_TYPE",
            message="Inventory upload must be a CSV or XLSX file",
        )
    if isinstance(exc, InventorySourceMalformedError):
        return APIError(
            status_code=400,
            code="INVENTORY_SOURCE_MALFORMED",
            message="Inventory upload could not be parsed",
        )
    if isinstance(exc, InventoryValidationFailedError):
        return APIError(
            status_code=422,
            code="INVENTORY_VALIDATION_FAILED",
            message="Inventory upload failed validation",
        )
    if isinstance(exc, InventoryScopeError):
        return APIError(
            status_code=404,
            code="INVENTORY_SCOPE_NOT_FOUND",
            message="Inventory scope could not be found",
        )
    if isinstance(exc, InventoryIngestionConflictError):
        return APIError(
            status_code=409,
            code="INVENTORY_INGESTION_CONFLICT",
            message="Inventory ingestion conflicts with existing data",
        )
    if isinstance(exc, InventoryIngestionPersistenceError):
        return APIError(
            status_code=500,
            code="INVENTORY_INGESTION_FAILED",
            message="Inventory ingestion could not be completed",
        )
    return APIError(
        status_code=500,
        code="INVENTORY_INGESTION_FAILED",
        message="Inventory ingestion could not be completed",
    )
