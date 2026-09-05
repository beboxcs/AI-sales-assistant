"""Deterministic evaluation of backend candidates against search attempts."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum

from app.ai_engine.search_planner import AdjustmentAction, SearchAttempt
from app.ai_engine.search_strategy import (
    ConstraintStrength,
    SearchField,
    SearchStrategy,
)
from app.search.contracts import PaymentPlanCandidate, UnitCandidate


class CandidateEvaluationError(Exception):
    """Base error for candidate evaluation failures."""


class CandidateContractViolationError(CandidateEvaluationError):
    """Raised when a backend candidate cannot satisfy an executed attempt."""


@dataclass(frozen=True)
class NumericDeviation:
    signed_delta: Decimal
    absolute_delta: Decimal
    percentage_delta: Decimal | None


@dataclass(frozen=True)
class FieldEvaluation:
    field: SearchField
    baseline_value: object
    attempted_value: object | None
    candidate_value: object | None
    satisfies_attempt: bool
    matches_baseline: bool
    strength: ConstraintStrength | None
    was_relaxed: bool
    was_dropped: bool
    deviation: NumericDeviation | None


@dataclass(frozen=True)
class PaymentPlanEvaluation:
    plan_key: str
    matches_payment_method: bool
    down_payment_deviation: NumericDeviation | None
    installment_years_deviation: NumericDeviation | None
    min_price_deviation: NumericDeviation | None
    max_price_deviation: NumericDeviation | None


@dataclass(frozen=True)
class CandidateEvaluation:
    unit_id: int
    attempt_sequence: int
    field_evaluations: tuple[FieldEvaluation, ...]
    payment_plan_evaluations: tuple[PaymentPlanEvaluation, ...]
    matched_baseline_fields: tuple[SearchField, ...]
    unmet_soft_fields: tuple[SearchField, ...]


_PLAN_FIELDS = frozenset(
    {
        SearchField.PAYMENT_METHOD,
        SearchField.MAX_DOWN_PAYMENT,
        SearchField.MIN_INSTALLMENT_YEARS,
    }
)

_NUMERIC_FIELDS = frozenset(
    {
        SearchField.BEDROOMS,
        SearchField.BATHROOMS,
        SearchField.MIN_PRICE,
        SearchField.MAX_PRICE,
        SearchField.MIN_AREA_SQM,
        SearchField.MAX_AREA_SQM,
        SearchField.MAX_DOWN_PAYMENT,
        SearchField.MIN_INSTALLMENT_YEARS,
    }
)


def evaluate_candidate(
    *,
    strategy: SearchStrategy,
    attempt: SearchAttempt,
    candidate: UnitCandidate,
) -> CandidateEvaluation:
    """Evaluate one backend candidate without mutating any input contract."""

    constraints = {constraint.field: constraint.strength for constraint in strategy.constraints}
    adjustments = {adjustment.field: adjustment for adjustment in attempt.adjustments}
    baseline_filters = strategy.filters
    attempted_filters = attempt.filters
    plan_semantics_relevant = _plan_semantics_relevant(
        baseline_filters, attempted_filters
    )

    _validate_executed_plan_contract(
        attempted_filters=attempted_filters,
        candidate=candidate,
        plan_semantics_relevant=plan_semantics_relevant,
    )

    payment_plan_evaluations = tuple(
        _evaluate_payment_plan(
            plan,
            baseline_filters=baseline_filters,
        )
        for plan in candidate.payment_plans
    )

    field_evaluations: list[FieldEvaluation] = []
    matched_baseline_fields: list[SearchField] = []
    unmet_soft_fields: list[SearchField] = []

    for field in SearchField:
        baseline_value = getattr(baseline_filters, field.value)
        if baseline_value is None:
            continue

        attempted_value = getattr(attempted_filters, field.value)
        adjustment = adjustments.get(field)
        was_relaxed = (
            adjustment is not None
            and adjustment.action is AdjustmentAction.WIDENED
        )
        was_dropped = (
            adjustment is not None
            and adjustment.action is AdjustmentAction.DROPPED
        )
        candidate_value = _candidate_value(
            field,
            candidate,
            plan_semantics_relevant=plan_semantics_relevant,
        )

        matches_baseline = _field_satisfaction(
            field,
            candidate,
            baseline_value,
            plan_semantics_relevant=plan_semantics_relevant,
            allow_unknown=True,
        )
        if was_dropped:
            satisfies_attempt = True
        else:
            satisfies_attempt = _field_satisfaction(
                field,
                candidate,
                attempted_value,
                plan_semantics_relevant=plan_semantics_relevant,
                allow_unknown=False,
            )
            if not satisfies_attempt:
                raise CandidateContractViolationError(
                    "Candidate does not satisfy executed search field: "
                    f"{field.value}"
                )

        deviation = _field_deviation(
            field,
            baseline_value,
            candidate_value,
            plan_semantics_relevant=plan_semantics_relevant,
        )
        strength = constraints.get(field)
        evaluation = FieldEvaluation(
            field=field,
            baseline_value=baseline_value,
            attempted_value=attempted_value,
            candidate_value=candidate_value,
            satisfies_attempt=satisfies_attempt,
            matches_baseline=matches_baseline,
            strength=strength,
            was_relaxed=was_relaxed,
            was_dropped=was_dropped,
            deviation=deviation,
        )
        field_evaluations.append(evaluation)

        if matches_baseline:
            matched_baseline_fields.append(field)
        elif strength is ConstraintStrength.SOFT:
            unmet_soft_fields.append(field)

    return CandidateEvaluation(
        unit_id=candidate.unit_id,
        attempt_sequence=attempt.sequence,
        field_evaluations=tuple(field_evaluations),
        payment_plan_evaluations=payment_plan_evaluations,
        matched_baseline_fields=tuple(matched_baseline_fields),
        unmet_soft_fields=tuple(unmet_soft_fields),
    )


def _plan_semantics_relevant(baseline_filters, attempted_filters) -> bool:
    return any(
        getattr(filters, field.value) is not None
        for filters in (baseline_filters, attempted_filters)
        for field in _PLAN_FIELDS
    )


def _candidate_value(
    field: SearchField,
    candidate: UnitCandidate,
    *,
    plan_semantics_relevant: bool,
) -> object | None:
    if field is SearchField.LOCATION:
        return candidate.location
    if field is SearchField.PROPERTY_TYPE:
        return candidate.property_type
    if field is SearchField.BEDROOMS:
        return candidate.bedrooms
    if field is SearchField.BATHROOMS:
        return candidate.bathrooms
    if field in {SearchField.MIN_AREA_SQM, SearchField.MAX_AREA_SQM}:
        return candidate.area_sqm
    if field in {SearchField.MIN_PRICE, SearchField.MAX_PRICE}:
        if not plan_semantics_relevant:
            return candidate.base_price
        return _aggregate_plan_values(candidate.payment_plans, "total_price")
    if field is SearchField.PAYMENT_METHOD:
        return _aggregate_plan_values(candidate.payment_plans, "payment_method")
    if field is SearchField.MAX_DOWN_PAYMENT:
        return _aggregate_plan_values(candidate.payment_plans, "down_payment")
    if field is SearchField.MIN_INSTALLMENT_YEARS:
        return _aggregate_plan_values(candidate.payment_plans, "installment_years")
    raise CandidateEvaluationError(f"Unsupported search field: {field.value}")


def _aggregate_plan_values(
    plans: tuple[PaymentPlanCandidate, ...], field_name: str
) -> object | None:
    values = tuple(getattr(plan, field_name) for plan in plans)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


def _field_satisfaction(
    field: SearchField,
    candidate: UnitCandidate,
    threshold: object | None,
    *,
    plan_semantics_relevant: bool,
    allow_unknown: bool,
) -> bool:
    if threshold is None:
        return True

    if field in _PLAN_FIELDS:
        return _any_plan_field_satisfies(
            candidate.payment_plans,
            field,
            threshold,
            allow_unknown=allow_unknown,
        )

    if field in {SearchField.MIN_PRICE, SearchField.MAX_PRICE} and plan_semantics_relevant:
        return _any_plan_field_satisfies(
            candidate.payment_plans,
            field,
            threshold,
            allow_unknown=allow_unknown,
        )

    candidate_value = _candidate_value(
        field,
        candidate,
        plan_semantics_relevant=False,
    )
    result = _compare_value(field, candidate_value, threshold)
    if result is None:
        if allow_unknown:
            return False
        raise CandidateContractViolationError(
            "Candidate value is required to establish search field: "
            f"{field.value}"
        )
    return result


def _any_plan_field_satisfies(
    plans: tuple[PaymentPlanCandidate, ...],
    field: SearchField,
    threshold: object,
    *,
    allow_unknown: bool,
) -> bool:
    if not plans:
        if allow_unknown:
            return False
        raise CandidateContractViolationError(
            "Candidate payment plans are required to establish search field: "
            f"{field.value}"
        )

    results = []
    for plan in plans:
        value = _plan_value(plan, field)
        results.append(_compare_value(field, value, threshold))

    if any(result is True for result in results):
        return True
    if all(result is None for result in results):
        if allow_unknown:
            return False
        raise CandidateContractViolationError(
            "Candidate payment-plan values are required to establish search "
            f"field: {field.value}"
        )
    return False


def _plan_value(plan: PaymentPlanCandidate, field: SearchField) -> object | None:
    if field is SearchField.PAYMENT_METHOD:
        return plan.payment_method
    if field is SearchField.MAX_DOWN_PAYMENT:
        return plan.down_payment
    if field is SearchField.MIN_INSTALLMENT_YEARS:
        return plan.installment_years
    if field in {SearchField.MIN_PRICE, SearchField.MAX_PRICE}:
        return plan.total_price
    raise CandidateEvaluationError(f"Unsupported plan search field: {field.value}")


def _compare_value(
    field: SearchField,
    candidate_value: object | None,
    threshold: object,
) -> bool | None:
    if candidate_value is None:
        return None
    if field in {
        SearchField.LOCATION,
        SearchField.PROPERTY_TYPE,
        SearchField.BEDROOMS,
        SearchField.BATHROOMS,
        SearchField.PAYMENT_METHOD,
    }:
        return _comparable_value(candidate_value) == _comparable_value(threshold)

    candidate_decimal = _as_decimal(candidate_value, field.value)
    threshold_decimal = _as_decimal(threshold, field.value)
    if field in {
        SearchField.MIN_PRICE,
        SearchField.MIN_AREA_SQM,
        SearchField.MIN_INSTALLMENT_YEARS,
    }:
        return candidate_decimal >= threshold_decimal
    if field in {
        SearchField.MAX_PRICE,
        SearchField.MAX_AREA_SQM,
        SearchField.MAX_DOWN_PAYMENT,
    }:
        return candidate_decimal <= threshold_decimal
    raise CandidateEvaluationError(f"Unsupported search field: {field.value}")


def _field_deviation(
    field: SearchField,
    baseline_value: object,
    candidate_value: object | None,
    *,
    plan_semantics_relevant: bool,
) -> NumericDeviation | None:
    if field not in _NUMERIC_FIELDS or candidate_value is None:
        return None
    if plan_semantics_relevant and field in {
        SearchField.MIN_PRICE,
        SearchField.MAX_PRICE,
        SearchField.MAX_DOWN_PAYMENT,
        SearchField.MIN_INSTALLMENT_YEARS,
    }:
        return None
    if isinstance(candidate_value, tuple):
        return None
    return _numeric_deviation(candidate_value, baseline_value)


def _numeric_deviation(candidate_value: object, baseline_value: object) -> NumericDeviation:
    candidate_decimal = _as_decimal(candidate_value, "candidate")
    baseline_decimal = _as_decimal(baseline_value, "baseline")
    signed_delta = candidate_decimal - baseline_decimal
    absolute_delta = abs(signed_delta)
    percentage_delta = None
    if baseline_decimal != 0:
        percentage_delta = absolute_delta / abs(baseline_decimal) * Decimal("100")
    return NumericDeviation(
        signed_delta=signed_delta,
        absolute_delta=absolute_delta,
        percentage_delta=percentage_delta,
    )


def _as_decimal(value: object, field_name: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise CandidateContractViolationError(
            f"Numeric value is invalid for field: {field_name}"
        ) from exc


def _comparable_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    return value


def _evaluate_payment_plan(
    plan: PaymentPlanCandidate,
    *,
    baseline_filters,
) -> PaymentPlanEvaluation:
    baseline_payment_method = baseline_filters.payment_method
    matches_payment_method = (
        baseline_payment_method is None
        or _comparable_value(plan.payment_method)
        == _comparable_value(baseline_payment_method)
    )
    return PaymentPlanEvaluation(
        plan_key=plan.plan_key,
        matches_payment_method=matches_payment_method,
        down_payment_deviation=(
            _numeric_deviation(plan.down_payment, baseline_filters.max_down_payment)
            if baseline_filters.max_down_payment is not None
            and plan.down_payment is not None
            else None
        ),
        installment_years_deviation=(
            _numeric_deviation(
                plan.installment_years,
                baseline_filters.min_installment_years,
            )
            if baseline_filters.min_installment_years is not None
            and plan.installment_years is not None
            else None
        ),
        min_price_deviation=(
            _numeric_deviation(plan.total_price, baseline_filters.min_price)
            if baseline_filters.min_price is not None and plan.total_price is not None
            else None
        ),
        max_price_deviation=(
            _numeric_deviation(plan.total_price, baseline_filters.max_price)
            if baseline_filters.max_price is not None and plan.total_price is not None
            else None
        ),
    )


def _validate_executed_plan_contract(
    *,
    attempted_filters,
    candidate: UnitCandidate,
    plan_semantics_relevant: bool,
) -> None:
    attempt_uses_plans = any(
        getattr(attempted_filters, field.value) is not None
        for field in _PLAN_FIELDS
    ) or (
        plan_semantics_relevant
        and any(
            getattr(attempted_filters, field.value) is not None
            for field in {SearchField.MIN_PRICE, SearchField.MAX_PRICE}
        )
    )
    if not attempt_uses_plans:
        return

    if not candidate.payment_plans:
        raise CandidateContractViolationError(
            "Candidate payment plans are required by the executed search attempt"
        )

    relevant_fields = [
        field
        for field in SearchField
        if getattr(attempted_filters, field.value) is not None
        and (
            field in _PLAN_FIELDS
            or (
                field in {SearchField.MIN_PRICE, SearchField.MAX_PRICE}
                and plan_semantics_relevant
            )
        )
    ]
    for plan in candidate.payment_plans:
        result = _plan_satisfies_fields(plan, attempted_filters, relevant_fields)
        if result is False:
            raise CandidateContractViolationError(
                "Candidate payment plan contradicts the executed search attempt: "
                f"{plan.plan_key}"
            )
        if result is None:
            fields = ", ".join(field.value for field in relevant_fields)
            raise CandidateContractViolationError(
                "Candidate payment-plan values are insufficient to establish "
                f"executed fields for {plan.plan_key}: {fields}"
            )


def _plan_satisfies_fields(
    plan: PaymentPlanCandidate,
    filters,
    fields: list[SearchField],
) -> bool | None:
    unknown = False
    for field in fields:
        result = _compare_value(field, _plan_value(plan, field), getattr(filters, field.value))
        if result is None:
            unknown = True
        elif not result:
            return False
    return None if unknown else True
