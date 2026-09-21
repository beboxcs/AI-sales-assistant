"""Public application entry points for structured and natural-language search."""

from dataclasses import dataclass

from app.ai_engine.multilingual_search_interpreter import (
    interpret_multilingual_search_query,
)
from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_intent import ParsedSearchIntent
from app.ai_engine.search_intent_builder import build_search_strategy_from_intent
from app.ai_engine.search_orchestrator import (
    SearchIntelligenceResult,
    execute_search_intelligence,
)
from app.ai_engine.search_plan_builder import build_search_plan
from app.ai_engine.search_strategy import (
    SearchField,
    SearchStrategy,
    build_default_search_strategy,
)
from app.search.contracts import SearchExecutionContext


@dataclass(frozen=True)
class NaturalLanguageSearchResult:
    intent: ParsedSearchIntent
    search: SearchIntelligenceResult | None


def search_units(
    *,
    filters: SearchFilters,
    context: SearchExecutionContext,
) -> SearchIntelligenceResult:
    """Run structured search through the canonical AI search pipeline."""

    strategy = build_default_search_strategy(filters)
    return _execute_strategy(strategy=strategy, context=context)


def search_units_from_query(
    *,
    text: str,
    context: SearchExecutionContext,
) -> NaturalLanguageSearchResult:
    """Interpret and execute an eligible multilingual search query."""

    if not text.strip():
        return NaturalLanguageSearchResult(
            intent=ParsedSearchIntent(filters=SearchFilters()),
            search=None,
        )

    intent = interpret_multilingual_search_query(text)
    if intent.ambiguities or not _has_populated_supported_filters(intent.filters):
        return NaturalLanguageSearchResult(intent=intent, search=None)

    strategy = build_search_strategy_from_intent(intent)
    return NaturalLanguageSearchResult(
        intent=intent,
        search=_execute_strategy(strategy=strategy, context=context),
    )


def _has_populated_supported_filters(filters: SearchFilters) -> bool:
    return any(
        getattr(filters, field.value) is not None
        for field in SearchField
    )


def _execute_strategy(
    *,
    strategy: SearchStrategy,
    context: SearchExecutionContext,
) -> SearchIntelligenceResult:
    plan = build_search_plan(strategy)
    return execute_search_intelligence(
        strategy=strategy,
        plan=plan,
        context=context,
    )
