"""Public interface for backend search execution."""

from app.ai_engine.search_filters import SearchFilters
from app.database import SessionLocal
from app.search.contracts import SearchExecutionContext, UnitCandidate
from app.search.exceptions import SearchExecutionError, SearchExecutionFailedError
from app.search.repository import search_candidates


def execute_search(
    *,
    filters: SearchFilters,
    context: SearchExecutionContext,
) -> list[UnitCandidate]:
    """Execute one resolved set of search filters within the supplied context."""
    if not context.resolved_scope_keys:
        return []

    try:
        with SessionLocal() as session:
            return search_candidates(
                session,
                filters=filters,
                context=context,
            )
    except SearchExecutionError:
        raise
    except Exception as exc:
        raise SearchExecutionFailedError("Search execution failed") from exc
