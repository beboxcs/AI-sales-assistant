from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class InventorySource:
    filename: str
    content: bytes


@dataclass(frozen=True)
class InventoryIngestionResult:
    status: str
    filename: str
    scope_key: str


class InventoryIngestionError(Exception):
    """Temporary API seam for application-level ingestion failures."""


class InventoryIngestionService(Protocol):
    def __call__(
        self,
        *,
        company_id: int,
        scope_key: str,
        source: InventorySource,
    ) -> InventoryIngestionResult: ...


def unavailable_ingest_inventory(
    *,
    company_id: int,
    scope_key: str,
    source: InventorySource,
) -> InventoryIngestionResult:
    raise NotImplementedError("Inventory ingestion service is not wired yet")
