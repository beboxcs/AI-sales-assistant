"""Public application entry point for structured search."""

from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_orchestrator import (
    SearchIntelligenceResult,
    execute_search_intelligence,
)
from app.ai_engine.search_plan_builder import build_search_plan
from app.ai_engine.search_strategy import build_default_search_strategy
from app.search.contracts import SearchExecutionContext


def search_units(
    *,
    filters: SearchFilters,
    context: SearchExecutionContext,
) -> SearchIntelligenceResult:
    """Run structured search through the canonical AI search pipeline."""

    strategy = build_default_search_strategy(filters)
    plan = build_search_plan(strategy)
    return execute_search_intelligence(
        strategy=strategy,
        plan=plan,
        context=context,
    )
