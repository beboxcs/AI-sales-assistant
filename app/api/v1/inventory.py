from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel

from app.api.dependencies import (
    get_inventory_ingest_max_upload_bytes,
    get_inventory_ingestion_service,
    get_trusted_company_id,
)
from app.api.errors import APIError
from app.api.inventory_ingestion_seam import (
    InventoryIngestionError,
    InventoryIngestionService,
    InventorySource,
)

router = APIRouter(prefix="/inventory", tags=["inventory"])

_ALLOWED_INVENTORY_EXTENSIONS = {".csv", ".xlsx"}


class InventoryIngestionResponse(BaseModel):
    status: str
    filename: str
    scope_key: str


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
        raise APIError(
            status_code=500,
            code="INVENTORY_INGESTION_FAILED",
            message="Inventory ingestion could not be completed",
        ) from exc
    except APIError:
        raise
    except Exception as exc:
        raise APIError(
            status_code=500,
            code="INVENTORY_INGESTION_FAILED",
            message="Inventory ingestion could not be completed",
        ) from exc

    return InventoryIngestionResponse(
        status=result.status,
        filename=result.filename,
        scope_key=result.scope_key,
    )
