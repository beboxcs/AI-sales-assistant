"""Deterministic ranking of H7.4 candidate-evaluation facts."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum

from app.ai_engine.candidate_evaluation import (
    CandidateEvaluation,
    FieldEvaluation,
    NumericDeviation,
    PaymentPlanEvaluation,
)
from app.ai_engine.search_strategy import (
    ConstraintStrength,
    RankingPreference,
    SearchField,
    SearchStrategy,
)


class CandidateRankingError(Exception):
    """Raised when evaluation facts cannot produce a valid ranking."""


class RankingDimension(str, Enum):
    LOCATION = "location"
    PROPERTY_TYPE = "property_type"
    BEDROOMS = "bedrooms"
    BATHROOMS = "bathrooms"
    PRICE = "price"
    AREA = "area"
    PAYMENT_METHOD = "payment_method"
    DOWN_PAYMENT = "down_payment"
    INSTALLMENT_YEARS = "installment_years"


class ScoreComponentKind(str, Enum):
    FIELD_MATCH = "field_match"
    NUMERIC_PROXIMITY = "numeric_proximity"
    RANGE_PROXIMITY = "range_proximity"
    PAYMENT_PLAN = "payment_plan"


@dataclass(frozen=True)
class ScoreComponent:
    dimension: RankingDimension
    kind: ScoreComponentKind
    score: Decimal
    weight: Decimal


@dataclass(frozen=True)
class RankedCandidate:
    evaluation: CandidateEvaluation
    score: Decimal
    rank: int
    best_plan_key: str | None
    components: tuple[ScoreComponent, ...]


_RANGE_FIELDS = {
    RankingDimension.PRICE: (SearchField.MIN_PRICE, SearchField.MAX_PRICE),
    RankingDimension.AREA: (SearchField.MIN_AREA_SQM, SearchField.MAX_AREA_SQM),
}

_PLAN_FIELDS = frozenset(
    {
        SearchField.PAYMENT_METHOD,
        SearchField.MAX_DOWN_PAYMENT,
        SearchField.MIN_INSTALLMENT_YEARS,
    }
)

_CATEGORICAL_DIMENSIONS = {
    SearchField.LOCATION: RankingDimension.LOCATION,
    SearchField.PROPERTY_TYPE: RankingDimension.PROPERTY_TYPE,
}

_NUMERIC_DIMENSIONS = {
    SearchField.BEDROOMS: RankingDimension.BEDROOMS,
    SearchField.BATHROOMS: RankingDimension.BATHROOMS,
    SearchField.MAX_DOWN_PAYMENT: RankingDimension.DOWN_PAYMENT,
    SearchField.MIN_INSTALLMENT_YEARS: RankingDimension.INSTALLMENT_YEARS,
}

_RANGE_DIMENSION_BY_FIELD = {
    field: dimension
    for dimension, fields in _RANGE_FIELDS.items()
    for field in fields
}


def rank_candidates(
    *,
    strategy: SearchStrategy,
    evaluations: tuple[CandidateEvaluation, ...],
) -> tuple[RankedCandidate, ...]:
    """Return deterministic ranked results from immutable evaluation facts."""

    if not evaluations:
        return ()

    relaxation_rules = {rule.field: rule for rule in strategy.relaxation_rules}
    ranking_preferences = {
        preference.field: preference for preference in strategy.ranking_preferences
    }
    constraint_strengths = {
        constraint.field: constraint.strength for constraint in strategy.constraints
    }
    plan_semantics_relevant = any(
        getattr(strategy.filters, field.value) is not None for field in _PLAN_FIELDS
    )

    ranked: list[RankedCandidate] = []
    for evaluation in evaluations:
        components, best_plan_key = _candidate_components(
            strategy=strategy,
            evaluation=evaluation,
            relaxation_rules=relaxation_rules,
            ranking_preferences=ranking_preferences,
            constraint_strengths=constraint_strengths,
            plan_semantics_relevant=plan_semantics_relevant,
        )
        score = _weighted_average(components)
        ranked.append(
            RankedCandidate(
                evaluation=evaluation,
                score=score,
                rank=0,
                best_plan_key=best_plan_key,
                components=tuple(components),
            )
        )

    ranked.sort(
        key=lambda result: (
            -result.score,
            len(result.evaluation.unmet_soft_fields),
            result.evaluation.attempt_sequence,
            result.evaluation.unit_id,
        )
    )
    return tuple(
        RankedCandidate(
            evaluation=result.evaluation,
            score=result.score,
            rank=index,
            best_plan_key=result.best_plan_key,
            components=result.components,
        )
        for index, result in enumerate(ranked, start=1)
    )


def _candidate_components(
    *,
    strategy: SearchStrategy,
    evaluation: CandidateEvaluation,
    relaxation_rules: dict,
    ranking_preferences: dict,
    constraint_strengths: dict,
    plan_semantics_relevant: bool,
) -> tuple[list[ScoreComponent], str | None]:
    fields = {item.field: item for item in evaluation.field_evaluations}
    components: list[ScoreComponent] = []
    processed_ranges: set[RankingDimension] = set()

    best_plan_key = None
    best_plan_components: list[ScoreComponent] = []
    if plan_semantics_relevant:
        best_plan_key, best_plan_components = _best_plan(
            strategy=strategy,
            evaluation=evaluation,
            fields=fields,
            relaxation_rules=relaxation_rules,
            ranking_preferences=ranking_preferences,
            constraint_strengths=constraint_strengths,
        )

    for field in SearchField:
        field_evaluation = fields.get(field)
        if field_evaluation is None:
            continue

        range_dimension = _RANGE_DIMENSION_BY_FIELD.get(field)
        if range_dimension is not None:
            if range_dimension in processed_ranges:
                continue
            processed_ranges.add(range_dimension)
            range_fields = _RANGE_FIELDS[range_dimension]
            active_range_fields = [
                fields[item_field]
                for item_field in range_fields
                if item_field in fields
                and _is_rankable(constraint_strengths.get(item_field))
            ]
            if plan_semantics_relevant and range_dimension is RankingDimension.PRICE:
                continue
            if active_range_fields:
                components.append(
                    _component(
                        dimension=range_dimension,
                        kind=ScoreComponentKind.RANGE_PROXIMITY,
                        score=_range_score(
                            active_range_fields,
                            relaxation_rules=relaxation_rules,
                        ),
                        weight=_range_weight(
                            range_fields=range_fields,
                            fields=fields,
                            ranking_preferences=ranking_preferences,
                            constraint_strengths=constraint_strengths,
                        ),
                    )
                )
            continue

        if field in _PLAN_FIELDS and plan_semantics_relevant:
            continue
        if not _is_rankable(constraint_strengths.get(field)):
            continue

        if field in _CATEGORICAL_DIMENSIONS:
            score = _categorical_score(field_evaluation)
            kind = ScoreComponentKind.FIELD_MATCH
            dimension = _CATEGORICAL_DIMENSIONS[field]
        elif field in {
            SearchField.BEDROOMS,
            SearchField.BATHROOMS,
            SearchField.MAX_DOWN_PAYMENT,
            SearchField.MIN_INSTALLMENT_YEARS,
        }:
            score = _numeric_field_score(
                field_evaluation,
                relaxation_rules=relaxation_rules,
            )
            kind = ScoreComponentKind.NUMERIC_PROXIMITY
            dimension = _NUMERIC_DIMENSIONS[field]
        else:
            raise CandidateRankingError(f"Unsupported ranking field: {field.value}")

        components.append(
            _component(
                dimension=dimension,
                kind=kind,
                score=score,
                weight=_field_weight(
                    field,
                    ranking_preferences=ranking_preferences,
                ),
            )
        )

    if plan_semantics_relevant:
        components.extend(best_plan_components)
    return components, best_plan_key


def _best_plan(
    *,
    strategy: SearchStrategy,
    evaluation: CandidateEvaluation,
    fields: dict,
    relaxation_rules: dict,
    ranking_preferences: dict,
    constraint_strengths: dict,
) -> tuple[str | None, list[ScoreComponent]]:
    if not evaluation.payment_plan_evaluations:
        return None, []

    scored_plans = [
        (
            plan.plan_key,
            _plan_components(
                strategy=strategy,
                plan=plan,
                fields=fields,
                relaxation_rules=relaxation_rules,
                ranking_preferences=ranking_preferences,
                constraint_strengths=constraint_strengths,
            ),
        )
        for plan in evaluation.payment_plan_evaluations
    ]
    if not any(components for _, components in scored_plans):
        return None, []
    ranked_plans = [
        (plan_key, components, _weighted_average(components))
        for plan_key, components in scored_plans
    ]
    ranked_plans.sort(key=lambda item: (-item[2], item[0]))
    best_key, best_components, _ = ranked_plans[0]
    return best_key, best_components


def _plan_components(
    *,
    strategy: SearchStrategy,
    plan: PaymentPlanEvaluation,
    fields: dict,
    relaxation_rules: dict,
    ranking_preferences: dict,
    constraint_strengths: dict,
) -> list[ScoreComponent]:
    filters = strategy.filters
    components: list[ScoreComponent] = []

    payment_method_field = fields.get(SearchField.PAYMENT_METHOD)
    if (
        filters.payment_method is not None
        and payment_method_field is not None
        and _is_rankable(constraint_strengths.get(SearchField.PAYMENT_METHOD))
    ):
        components.append(
            _component(
                dimension=RankingDimension.PAYMENT_METHOD,
                kind=ScoreComponentKind.PAYMENT_PLAN,
                score=Decimal("1") if plan.matches_payment_method else Decimal("0"),
                weight=_field_weight(
                    SearchField.PAYMENT_METHOD,
                    ranking_preferences=ranking_preferences,
                ),
            )
        )

    price_fields = [
        field
        for field in (SearchField.MIN_PRICE, SearchField.MAX_PRICE)
        if getattr(filters, field.value) is not None
        and field in fields
        and _is_rankable(constraint_strengths.get(field))
    ]
    if price_fields:
        price_score = _plan_price_score(
            plan,
            price_fields=price_fields,
            fields=fields,
            relaxation_rules=relaxation_rules,
        )
        components.append(
            _component(
                dimension=RankingDimension.PRICE,
                kind=ScoreComponentKind.PAYMENT_PLAN,
                score=price_score,
                weight=_range_weight(
                    range_fields=(SearchField.MIN_PRICE, SearchField.MAX_PRICE),
                    fields=fields,
                    ranking_preferences=ranking_preferences,
                    constraint_strengths=constraint_strengths,
                ),
            )
        )

    down_field = fields.get(SearchField.MAX_DOWN_PAYMENT)
    if (
        filters.max_down_payment is not None
        and down_field is not None
        and _is_rankable(constraint_strengths.get(SearchField.MAX_DOWN_PAYMENT))
    ):
        components.append(
            _component(
                dimension=RankingDimension.DOWN_PAYMENT,
                kind=ScoreComponentKind.PAYMENT_PLAN,
                score=_plan_numeric_score(
                    field=SearchField.MAX_DOWN_PAYMENT,
                    deviation=plan.down_payment_deviation,
                    field_evaluation=down_field,
                    relaxation_rules=relaxation_rules,
                ),
                weight=_field_weight(
                    SearchField.MAX_DOWN_PAYMENT,
                    ranking_preferences=ranking_preferences,
                ),
            )
        )

    years_field = fields.get(SearchField.MIN_INSTALLMENT_YEARS)
    if (
        filters.min_installment_years is not None
        and years_field is not None
        and _is_rankable(constraint_strengths.get(SearchField.MIN_INSTALLMENT_YEARS))
    ):
        components.append(
            _component(
                dimension=RankingDimension.INSTALLMENT_YEARS,
                kind=ScoreComponentKind.PAYMENT_PLAN,
                score=_plan_numeric_score(
                    field=SearchField.MIN_INSTALLMENT_YEARS,
                    deviation=plan.installment_years_deviation,
                    field_evaluation=years_field,
                    relaxation_rules=relaxation_rules,
                ),
                weight=_field_weight(
                    SearchField.MIN_INSTALLMENT_YEARS,
                    ranking_preferences=ranking_preferences,
                ),
            )
        )
    return components


def _plan_price_score(
    plan: PaymentPlanEvaluation,
    *,
    price_fields: list[SearchField],
    fields: dict,
    relaxation_rules: dict,
) -> Decimal:
    scores: list[Decimal] = []
    definitive_violations = 0
    for field in price_fields:
        if field is SearchField.MIN_PRICE:
            deviation = plan.min_price_deviation
        else:
            deviation = plan.max_price_deviation
        if deviation is None:
            scores.append(Decimal("0"))
            continue
        is_satisfied = (
            deviation.signed_delta >= 0
            if field is SearchField.MIN_PRICE
            else deviation.signed_delta <= 0
        )
        if is_satisfied:
            scores.append(Decimal("1"))
            continue
        definitive_violations += 1
        scores.append(
            _relaxed_score(
                field=field,
                deviation=deviation,
                field_evaluation=fields[field],
                relaxation_rules=relaxation_rules,
                distance_mode="min" if field is SearchField.MIN_PRICE else "max",
            )
        )
    if (
        len(price_fields) > 1
        and definitive_violations == len(price_fields)
        and not all(fields[field].was_dropped for field in price_fields)
    ):
        raise CandidateRankingError(
            "Plan price facts violate both populated price bounds"
        )
    return min(scores, default=Decimal("1"))


def _plan_numeric_score(
    *,
    field: SearchField,
    deviation: NumericDeviation | None,
    field_evaluation: FieldEvaluation,
    relaxation_rules: dict,
) -> Decimal:
    if deviation is None:
        return Decimal("0")
    if field is SearchField.MAX_DOWN_PAYMENT:
        if deviation.signed_delta <= 0:
            return Decimal("1")
        distance_mode = "max"
    else:
        if deviation.signed_delta >= 0:
            return Decimal("1")
        distance_mode = "min"
    return _relaxed_score(
        field=field,
        deviation=deviation,
        field_evaluation=field_evaluation,
        relaxation_rules=relaxation_rules,
        distance_mode=distance_mode,
    )


def _range_score(
    fields: list[FieldEvaluation],
    *,
    relaxation_rules: dict,
) -> Decimal:
    if all(field.matches_baseline for field in fields):
        return Decimal("1")
    violated = [field for field in fields if not field.matches_baseline]
    if len(violated) > 1 and not all(field.was_dropped for field in violated):
        raise CandidateRankingError(
            "Range evaluation facts violate both populated range bounds"
        )
    return _numeric_field_score(violated[0], relaxation_rules=relaxation_rules)


def _numeric_field_score(
    field: FieldEvaluation,
    *,
    relaxation_rules: dict,
) -> Decimal:
    if field.matches_baseline:
        return Decimal("1")
    if field.was_dropped:
        return Decimal("0")
    if not field.was_relaxed or field.deviation is None:
        return Decimal("0")

    if field.field in {
        SearchField.MAX_DOWN_PAYMENT,
        SearchField.MAX_PRICE,
        SearchField.MAX_AREA_SQM,
    }:
        distance_mode = "max"
    elif field.field in {
        SearchField.MIN_INSTALLMENT_YEARS,
        SearchField.MIN_PRICE,
        SearchField.MIN_AREA_SQM,
    }:
        distance_mode = "min"
    else:
        distance_mode = "absolute"
    return _relaxed_score(
        field=field.field,
        deviation=field.deviation,
        field_evaluation=field,
        relaxation_rules=relaxation_rules,
        distance_mode=distance_mode,
    )


def _relaxed_score(
    *,
    field: SearchField,
    deviation: NumericDeviation,
    field_evaluation: FieldEvaluation,
    relaxation_rules: dict,
    distance_mode: str,
) -> Decimal:
    if not field_evaluation.was_relaxed:
        return Decimal("0")
    rule = relaxation_rules.get(field)
    if rule is None:
        return Decimal("0")

    if distance_mode == "max":
        distance = deviation.signed_delta
    elif distance_mode == "min":
        distance = -deviation.signed_delta
    else:
        distance = deviation.absolute_delta
    if distance <= 0:
        return Decimal("1")

    allowed = _allowed_magnitude(rule, field_evaluation.baseline_value)
    if allowed is None or allowed <= 0:
        return Decimal("0")
    penalty_ratio = min(distance / allowed, Decimal("1"))
    return Decimal("1") - penalty_ratio


def _allowed_magnitude(rule, baseline_value: object) -> Decimal | None:
    baseline_decimal = _as_decimal(baseline_value, "baseline")
    candidates: list[Decimal] = []
    if rule.max_delta is not None:
        candidates.append(_as_decimal(rule.max_delta, "max_delta"))
    if rule.max_percentage is not None:
        candidates.append(
            abs(baseline_decimal)
            * _as_decimal(rule.max_percentage, "max_percentage")
            / Decimal("100")
        )
    return min(candidates) if candidates else None


def _categorical_score(field: FieldEvaluation) -> Decimal:
    return Decimal("1") if field.matches_baseline else Decimal("0")


def _field_weight(field: SearchField, *, ranking_preferences: dict) -> Decimal:
    preference = ranking_preferences.get(field)
    return _preference_weight(preference) if preference is not None else Decimal("1")


def _range_weight(
    *,
    range_fields: tuple[SearchField, SearchField],
    fields: dict,
    ranking_preferences: dict,
    constraint_strengths: dict,
) -> Decimal:
    explicit_weights = [
        _preference_weight(ranking_preferences[field])
        for field in range_fields
        if field in fields
        and _is_rankable(constraint_strengths.get(field))
        and field in ranking_preferences
    ]
    if explicit_weights:
        return max(explicit_weights)
    return Decimal("1")


def _preference_weight(preference: RankingPreference) -> Decimal:
    weight = _as_decimal(preference.weight, "ranking weight")
    if not weight.is_finite() or weight <= 0:
        raise CandidateRankingError("Ranking weights must be finite and positive")
    return weight


def _is_rankable(strength: ConstraintStrength | None) -> bool:
    return strength is not ConstraintStrength.HARD


def _component(
    *,
    dimension: RankingDimension,
    kind: ScoreComponentKind,
    score: Decimal,
    weight: Decimal,
) -> ScoreComponent:
    if not score.is_finite() or score < 0 or score > 1:
        raise CandidateRankingError(
            f"Score component for {dimension.value} is outside [0, 1]"
        )
    if not weight.is_finite() or weight <= 0:
        raise CandidateRankingError(
            f"Score component weight for {dimension.value} is invalid"
        )
    return ScoreComponent(
        dimension=dimension,
        kind=kind,
        score=score,
        weight=weight,
    )


def _weighted_average(components: list[ScoreComponent]) -> Decimal:
    if not components:
        return Decimal("1")
    total_weight = sum((component.weight for component in components), Decimal("0"))
    if not total_weight.is_finite() or total_weight <= 0:
        raise CandidateRankingError("Effective ranking weight is invalid")
    score = sum(
        (component.score * component.weight for component in components),
        Decimal("0"),
    ) / total_weight
    if not score.is_finite() or score < 0 or score > 1:
        raise CandidateRankingError("Candidate score is outside [0, 1]")
    return score


def _as_decimal(value: object, name: str) -> Decimal:
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise CandidateRankingError(f"{name} must be numeric") from exc
    if not decimal_value.is_finite():
        raise CandidateRankingError(f"{name} must be finite")
    return decimal_value
