from typing import Protocol

from app.config import settings
from app.database import get_db as get_db_session
from app.search.contracts import SearchExecutionContext
from app.search.scope_resolution import resolve_search_execution_context


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

__all__ = [
    "SearchContextResolver",
    "get_db_session",
    "get_search_context_resolver",
    "get_trusted_company_id",
]
