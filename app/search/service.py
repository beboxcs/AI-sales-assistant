"""Public interface for backend search execution."""

from app.ai_engine.search_filters import SearchFilters
from app.database import SessionLocal
from app.search.contracts import (
    PriceBasis,
    SearchExecutionContext,
    UnitCandidate,
    resolve_price_basis,
)
from app.search.exceptions import SearchExecutionError, SearchExecutionFailedError
from app.search.repository import search_candidates


def execute_search(
    *,
    filters: SearchFilters,
    context: SearchExecutionContext,
    price_basis: PriceBasis | None = None,
) -> list[UnitCandidate]:
    """Execute one resolved set of search filters within the supplied context."""
    resolved_price_basis = resolve_price_basis(
        filters=filters,
        price_basis=price_basis,
    )
    if not context.resolved_scope_keys:
        return []

    try:
        with SessionLocal() as session:
            return search_candidates(
                session,
                filters=filters,
                context=context,
                price_basis=resolved_price_basis,
            )
    except SearchExecutionError:
        raise
    except Exception as exc:
        raise SearchExecutionFailedError("Search execution failed") from exc
