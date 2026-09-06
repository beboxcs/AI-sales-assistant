from typing import Protocol

from app.config import settings
from app.database import get_db as get_db_session
from app.search.contracts import SearchExecutionContext


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
    def resolver(
        *,
        company_id: int,
        requested_scope_keys: tuple[str, ...] | None,
    ) -> SearchExecutionContext:
        raise NotImplementedError(
            "Search execution context resolver is not wired yet"
        )

    return resolver

__all__ = [
    "SearchContextResolver",
    "get_db_session",
    "get_search_context_resolver",
    "get_trusted_company_id",
]
