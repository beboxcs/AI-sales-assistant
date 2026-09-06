"""Deterministic structured explanation facts for ranked candidates."""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from app.ai_engine.candidate_evaluation import FieldEvaluation
from app.ai_engine.candidate_ranking import (
    RankedCandidate,
    RankingDimension,
    ScoreComponent,
    ScoreComponentKind,
)
from app.ai_engine.search_strategy import SearchField, SearchStrategy


class RecommendationExplanationError(Exception):
    """Raised when ranked facts cannot form a valid explanation."""


class ExplanationFactKind(str, Enum):
    STRENGTH = "strength"
    TRADEOFF = "tradeoff"


@dataclass(frozen=True)
class ExplanationFact:
    dimension: RankingDimension
    kind: ExplanationFactKind
    message_key: str
    baseline_value: object | None
    candidate_value: object | None
    score: Decimal | None


@dataclass(frozen=True)
class RecommendationExplanation:
    unit_id: int
    rank: int
    score: Decimal
    best_plan_key: str | None
    strengths: tuple[ExplanationFact, ...]
    tradeoffs: tuple[ExplanationFact, ...]


_DIMENSION_FIELDS: dict[RankingDimension, tuple[SearchField, ...]] = {
    RankingDimension.LOCATION: (SearchField.LOCATION,),
    RankingDimension.PROPERTY_TYPE: (SearchField.PROPERTY_TYPE,),
    RankingDimension.BEDROOMS: (SearchField.BEDROOMS,),
    RankingDimension.BATHROOMS: (SearchField.BATHROOMS,),
    RankingDimension.PRICE: (SearchField.MIN_PRICE, SearchField.MAX_PRICE),
    RankingDimension.AREA: (SearchField.MIN_AREA_SQM, SearchField.MAX_AREA_SQM),
    RankingDimension.PAYMENT_METHOD: (SearchField.PAYMENT_METHOD,),
    RankingDimension.DOWN_PAYMENT: (SearchField.MAX_DOWN_PAYMENT,),
    RankingDimension.INSTALLMENT_YEARS: (SearchField.MIN_INSTALLMENT_YEARS,),
}

_MESSAGE_KEYS: dict[RankingDimension, tuple[str, str, str]] = {
    RankingDimension.LOCATION: (
        "matches_location",
        "location_relaxed",
        "location_not_matched",
    ),
    RankingDimension.PROPERTY_TYPE: (
        "matches_property_type",
        "property_type_relaxed",
        "property_type_not_matched",
    ),
    RankingDimension.BEDROOMS: (
        "matches_bedrooms",
        "bedrooms_relaxed",
        "bedrooms_not_matched",
    ),
    RankingDimension.BATHROOMS: (
        "matches_bathrooms",
        "bathrooms_relaxed",
        "bathrooms_not_matched",
    ),
    RankingDimension.PRICE: (
        "within_preferred_price",
        "price_relaxed",
        "price_not_matched",
    ),
    RankingDimension.AREA: (
        "within_preferred_area",
        "area_relaxed",
        "area_not_matched",
    ),
    RankingDimension.PAYMENT_METHOD: (
        "matches_payment_method",
        "payment_method_relaxed",
        "payment_method_not_matched",
    ),
    RankingDimension.DOWN_PAYMENT: (
        "meets_down_payment_preference",
        "down_payment_relaxed",
        "down_payment_not_matched",
    ),
    RankingDimension.INSTALLMENT_YEARS: (
        "meets_installment_duration",
        "installment_duration_relaxed",
        "installment_duration_not_matched",
    ),
}

_DROPPED_MESSAGE_KEYS = {
    RankingDimension.LOCATION: "location_dropped",
    RankingDimension.PROPERTY_TYPE: "property_type_dropped",
    RankingDimension.BEDROOMS: "bedrooms_dropped",
    RankingDimension.BATHROOMS: "bathrooms_dropped",
    RankingDimension.PRICE: "price_dropped",
    RankingDimension.AREA: "area_dropped",
    RankingDimension.PAYMENT_METHOD: "payment_method_dropped",
    RankingDimension.DOWN_PAYMENT: "down_payment_dropped",
    RankingDimension.INSTALLMENT_YEARS: "installment_duration_dropped",
}


def build_recommendation_explanation(
    *,
    strategy: SearchStrategy,
    ranked_candidate: RankedCandidate,
) -> RecommendationExplanation:
    """Build immutable explanation facts from existing ranking/evaluation facts."""

    _validate_ranked_candidate(ranked_candidate)
    field_evaluations = {
        field_evaluation.field: field_evaluation
        for field_evaluation in ranked_candidate.evaluation.field_evaluations
    }
    seen_dimensions: set[RankingDimension] = set()
    strengths: list[ExplanationFact] = []
    tradeoffs: list[ExplanationFact] = []

    for component in ranked_candidate.components:
        if component.dimension in seen_dimensions:
            raise RecommendationExplanationError(
                "Duplicate ranking dimension in candidate components: "
                f"{component.dimension.value}"
            )
        seen_dimensions.add(component.dimension)

        if _dimension_is_hard_only(component.dimension, strategy):
            continue

        fact = _build_fact(
            component=component,
            field_evaluations=field_evaluations,
        )
        if component.score == Decimal("1"):
            strengths.append(fact)
        else:
            tradeoffs.append(fact)

    return RecommendationExplanation(
        unit_id=ranked_candidate.evaluation.unit_id,
        rank=ranked_candidate.rank,
        score=ranked_candidate.score,
        best_plan_key=ranked_candidate.best_plan_key,
        strengths=tuple(strengths),
        tradeoffs=tuple(tradeoffs),
    )


def _validate_ranked_candidate(ranked_candidate: RankedCandidate) -> None:
    if not isinstance(ranked_candidate.rank, int) or ranked_candidate.rank < 1:
        raise RecommendationExplanationError("Ranked candidate rank must be at least 1")
    _validate_decimal(ranked_candidate.score, "candidate score")
    if not Decimal("0") <= ranked_candidate.score <= Decimal("1"):
        raise RecommendationExplanationError("Candidate score must be within [0, 1]")

    for component in ranked_candidate.components:
        _validate_decimal(component.score, "component score")
        if not Decimal("0") <= component.score <= Decimal("1"):
            raise RecommendationExplanationError(
                "Component score must be within [0, 1]"
            )
        _validate_decimal(component.weight, "component weight")
        if component.weight <= 0:
            raise RecommendationExplanationError(
                "Component weight must be greater than zero"
            )


def _validate_decimal(value: object, name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise RecommendationExplanationError(f"{name} must be a finite Decimal")


def _dimension_is_hard_only(
    dimension: RankingDimension,
    strategy: SearchStrategy,
) -> bool:
    fields = _DIMENSION_FIELDS[dimension]
    populated = [
        field
        for field in fields
        if getattr(strategy.filters, field.value) is not None
    ]
    if not populated:
        return False
    strengths = {
        constraint.field: constraint.strength for constraint in strategy.constraints
    }
    return all(strengths.get(field) is not None and strengths[field].value == "hard" for field in populated)


def _build_fact(
    *,
    component: ScoreComponent,
    field_evaluations: dict[SearchField, FieldEvaluation],
) -> ExplanationFact:
    fields = [
        field_evaluations[field]
        for field in _DIMENSION_FIELDS[component.dimension]
        if field in field_evaluations
    ]
    baseline_value, candidate_value = _fact_values(component.dimension, fields)
    if _is_plan_sourced(component):
        candidate_value = None
    message_key = _message_key(component, fields)
    kind = (
        ExplanationFactKind.STRENGTH
        if component.score == Decimal("1")
        else ExplanationFactKind.TRADEOFF
    )
    return ExplanationFact(
        dimension=component.dimension,
        kind=kind,
        message_key=message_key,
        baseline_value=baseline_value,
        candidate_value=candidate_value,
        score=component.score,
    )


def _is_plan_sourced(component: ScoreComponent) -> bool:
    if component.dimension in {
        RankingDimension.PAYMENT_METHOD,
        RankingDimension.DOWN_PAYMENT,
        RankingDimension.INSTALLMENT_YEARS,
    }:
        return True
    return (
        component.dimension is RankingDimension.PRICE
        and component.kind is ScoreComponentKind.PAYMENT_PLAN
    )


def _fact_values(
    dimension: RankingDimension,
    fields: list[FieldEvaluation],
) -> tuple[object | None, object | None]:
    if dimension in {RankingDimension.PRICE, RankingDimension.AREA}:
        by_field = {field.field: field for field in fields}
        ordered_fields = _DIMENSION_FIELDS[dimension]
        baseline_value = tuple(
            by_field[field].baseline_value if field in by_field else None
            for field in ordered_fields
        )
        candidate_values = [field.candidate_value for field in fields]
        if not candidate_values:
            return baseline_value, None
        first_value = candidate_values[0]
        candidate_value = (
            first_value
            if all(value == first_value for value in candidate_values)
            else None
        )
        return baseline_value, candidate_value

    if not fields:
        return None, None
    return fields[0].baseline_value, fields[0].candidate_value


def _message_key(
    component: ScoreComponent,
    fields: list[FieldEvaluation],
) -> str:
    if component.score == Decimal("0") and fields and all(field.was_dropped for field in fields):
        return _DROPPED_MESSAGE_KEYS[component.dimension]

    full_score, relaxed_score, zero_score = _MESSAGE_KEYS[component.dimension]
    if component.score == Decimal("1"):
        return full_score
    if component.score > Decimal("0"):
        return relaxed_score
    return zero_score
