"""Public interface for backend search execution."""

from app.ai_engine.search_filters import SearchFilters
from app.search.contracts import SearchExecutionContext, UnitCandidate


def execute_search(
    *,
    filters: SearchFilters,
    context: SearchExecutionContext,
) -> list[UnitCandidate]:
    """Execute one resolved set of search filters within the supplied context."""
    ...
