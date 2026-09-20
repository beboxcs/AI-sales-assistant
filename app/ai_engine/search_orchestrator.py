"""AI-side orchestration for planned search intelligence execution."""

from dataclasses import dataclass

from app.ai_engine.candidate_evaluation import (
    CandidateEvaluation,
    evaluate_candidate,
)
from app.ai_engine.candidate_ranking import RankedCandidate, rank_candidates
from app.ai_engine.recommendation_explanation import (
    RecommendationExplanation,
    build_recommendation_explanation,
)
from app.ai_engine.search_planner import SearchPlan
from app.ai_engine.search_strategy import SearchStrategy
from app.search import service as search_service
from app.search.contracts import (
    SearchExecutionContext,
    UnitCandidate,
    resolve_price_basis,
)


class SearchIntelligenceContractError(Exception):
    """Raised when backend results violate orchestration contracts."""


@dataclass(frozen=True)
class SearchIntelligenceItem:
    candidate: UnitCandidate
    ranking: RankedCandidate
    explanation: RecommendationExplanation


@dataclass(frozen=True)
class SearchIntelligenceResult:
    strategy: SearchStrategy
    plan: SearchPlan
    results: tuple[SearchIntelligenceItem, ...]


def execute_search_intelligence(
    *,
    strategy: SearchStrategy,
    plan: SearchPlan,
    context: SearchExecutionContext,
) -> SearchIntelligenceResult:
    """Execute a search plan and assemble globally ranked explanations."""

    price_basis = resolve_price_basis(
        filters=strategy.filters,
        price_basis=None,
    )
    candidates_by_unit_id: dict[int, UnitCandidate] = {}
    evaluations: list[CandidateEvaluation] = []

    for attempt in plan.attempts:
        candidates = search_service.execute_search(
            filters=attempt.filters,
            context=context,
            price_basis=price_basis,
        )
        attempt_unit_ids: set[int] = set()

        for candidate in candidates:
            if candidate.unit_id in attempt_unit_ids:
                raise SearchIntelligenceContractError(
                    "Backend returned duplicate unit_id within search attempt "
                    f"{attempt.sequence}: {candidate.unit_id}"
                )
            attempt_unit_ids.add(candidate.unit_id)

            if candidate.unit_id in candidates_by_unit_id:
                continue

            evaluation = evaluate_candidate(
                strategy=strategy,
                attempt=attempt,
                candidate=candidate,
                price_basis=price_basis,
            )
            candidates_by_unit_id[candidate.unit_id] = candidate
            evaluations.append(evaluation)

    ranked_candidates = rank_candidates(
        strategy=strategy,
        evaluations=tuple(evaluations),
    )
    results = tuple(
        SearchIntelligenceItem(
            candidate=candidates_by_unit_id[ranked.evaluation.unit_id],
            ranking=ranked,
            explanation=build_recommendation_explanation(
                strategy=strategy,
                ranked_candidate=ranked,
            ),
        )
        for ranked in ranked_candidates
    )

    return SearchIntelligenceResult(
        strategy=strategy,
        plan=plan,
        results=results,
    )
