from typing import Protocol

from app.ai_engine.inventory_application import (
    InventoryIngestionResult,
    ingest_inventory,
)
from app.ai_engine.inventory_ingestion import InventorySource
from app.config import settings
from app.database import get_db as get_db_session
from app.search.contracts import SearchExecutionContext
from app.search.scope_resolution import resolve_search_execution_context


class InventoryIngestionService(Protocol):
    def __call__(
        self,
        *,
        company_id: int,
        scope_key: str,
        source: InventorySource,
    ) -> InventoryIngestionResult: ...


class SearchContextResolver(Protocol):
    def __call__(
        self,
        *,
        company_id: int,
        requested_scope_keys: tuple[str, ...] | None,
    ) -> SearchExecutionContext: ...


def get_trusted_company_id() -> int:
    return settings.DEV_COMPANY_ID


def get_search_context_resolver() -> SearchContextResolver:
    return resolve_search_execution_context


def get_inventory_ingestion_service() -> InventoryIngestionService:
    return ingest_inventory


def get_inventory_ingest_max_upload_bytes() -> int:
    return settings.INVENTORY_INGEST_MAX_UPLOAD_BYTES

__all__ = [
    "InventoryIngestionService",
    "SearchContextResolver",
    "get_db_session",
    "get_inventory_ingest_max_upload_bytes",
    "get_inventory_ingestion_service",
    "get_search_context_resolver",
    "get_trusted_company_id",
]
