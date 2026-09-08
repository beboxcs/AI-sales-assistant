from decimal import Decimal, InvalidOperation

from pydantic import ValidationError

from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_planner import (
    AdjustmentAction,
    FieldAdjustment,
    SearchAttempt,
    SearchPlan,
)
from app.ai_engine.search_strategy import (
    ConstraintStrength,
    MatchMode,
    AlternativeDropRule,
    RelaxationDirection,
    RelaxationRule,
    SearchField,
    SearchStrategy,
)


# The exact attempt is not counted. This cap keeps the MVP plan bounded.
MAX_NEAR_ATTEMPTS = 12
MAX_ALTERNATIVE_ATTEMPTS = 8


# None marks equality-like count fields; other values are their useful direction.
USEFUL_RELAXATION_DIRECTIONS: dict[
    SearchField, RelaxationDirection | None
] = {
    SearchField.BEDROOMS: None,
    SearchField.BATHROOMS: None,
    SearchField.MIN_PRICE: RelaxationDirection.DECREASE,
    SearchField.MAX_PRICE: RelaxationDirection.INCREASE,
    SearchField.MIN_AREA_SQM: RelaxationDirection.DECREASE,
    SearchField.MAX_AREA_SQM: RelaxationDirection.INCREASE,
    SearchField.MAX_DOWN_PAYMENT: RelaxationDirection.INCREASE,
    SearchField.MIN_INSTALLMENT_YEARS: RelaxationDirection.DECREASE,
}


def build_search_plan(strategy: SearchStrategy) -> SearchPlan:
    """Build an exact baseline and bounded numeric/alternative attempts."""
    baseline_filters = strategy.filters.model_copy(deep=True)
    attempts = [
        SearchAttempt(
            sequence=1,
            mode=MatchMode.EXACT,
            filters=baseline_filters,
        )
    ]

    if strategy.match_mode is MatchMode.EXACT or not strategy.relaxation_rules:
        if strategy.match_mode is MatchMode.ALTERNATIVE and strategy.alternative_drop_rules:
            _validate_alternative_drop_safety(strategy)
            attempts.extend(
                _build_alternative_attempts(
                    baseline_filters,
                    [baseline_filters],
                    strategy.alternative_drop_rules,
                    attempts,
                )
            )
        return SearchPlan(attempts=attempts)

    _validate_directions(strategy.relaxation_rules)
    near_attempts, terminal_frontier = _build_numeric_plan(
        baseline_filters, strategy.relaxation_rules
    )
    attempts.extend(near_attempts)
    if strategy.match_mode is MatchMode.ALTERNATIVE and strategy.alternative_drop_rules:
        _validate_alternative_drop_safety(strategy)
        seeds = terminal_frontier if near_attempts else [baseline_filters]
        attempts.extend(
            _build_alternative_attempts(
                baseline_filters,
                seeds,
                strategy.alternative_drop_rules,
                attempts,
            )
        )
    return SearchPlan(attempts=attempts)


def _build_numeric_plan(
    baseline_filters: SearchFilters,
    rules: list[RelaxationRule],
) -> tuple[list[SearchAttempt], list[SearchFilters]]:
    ordered_rules = sorted(
        rules,
        key=lambda rule: (rule.priority, _field_order(rule.field)),
    )
    priorities = sorted({rule.priority for rule in ordered_rules})
    frontier = [baseline_filters]
    generated: list[SearchAttempt] = []
    emitted_states = [_filter_state(baseline_filters)]
    terminal_frontier: list[SearchFilters] = []

    for priority in priorities:
        if len(generated) >= MAX_NEAR_ATTEMPTS:
            break
        transformed_frontier = []
        for incoming_filters in frontier:
            transformed_filters = _apply_one_direction_rules(
                baseline_filters,
                incoming_filters,
                ordered_rules,
                priority,
            )
            if not _contains_filter_state(transformed_frontier, transformed_filters):
                transformed_frontier.append(transformed_filters)

        both_rules = [
            rule
            for rule in ordered_rules
            if rule.priority == priority
            and USEFUL_RELAXATION_DIRECTIONS[rule.field] is None
            and rule.direction is RelaxationDirection.BOTH
        ]
        both_rules.sort(key=lambda rule: _field_order(rule.field))

        if both_rules:
            next_frontier = []
            for transformed_filters in transformed_frontier:
                for rule in both_rules:
                    for direction in (
                        RelaxationDirection.DECREASE,
                        RelaxationDirection.INCREASE,
                    ):
                        adjusted_value = _adjusted_value(
                            baseline_filters, rule, direction
                        )
                        if adjusted_value is None:
                            continue
                        candidate_filters = _with_field_value(
                            transformed_filters,
                            rule.field,
                            adjusted_value,
                        )
                        if candidate_filters is not None and not _contains_filter_state(
                            next_frontier, candidate_filters
                        ):
                            next_frontier.append(candidate_filters)
            frontier = next_frontier
        else:
            frontier = transformed_frontier

        admitted_current_frontier = []
        for candidate_filters in frontier:
            state = _filter_state(candidate_filters)
            if _state_in_list(state, emitted_states):
                admitted_current_frontier.append(candidate_filters)
                continue
            attempt = _make_near_attempt(
                baseline_filters,
                candidate_filters,
                sequence=len(generated) + 2,
            )
            if attempt is None:
                continue
            generated.append(attempt)
            emitted_states.append(state)
            admitted_current_frontier.append(candidate_filters)
            if len(generated) >= MAX_NEAR_ATTEMPTS:
                return generated, admitted_current_frontier

        if admitted_current_frontier:
            terminal_frontier = admitted_current_frontier

    return generated, terminal_frontier


def _build_alternative_attempts(
    baseline_filters: SearchFilters,
    seeds: list[SearchFilters],
    rules: list[AlternativeDropRule],
    existing_attempts: list[SearchAttempt],
) -> list[SearchAttempt]:
    ordered_rules = sorted(
        rules,
        key=lambda rule: (rule.priority, _field_order(rule.field)),
    )
    priorities = sorted({rule.priority for rule in ordered_rules})
    frontier = list(seeds)
    emitted_states = [_filter_state(attempt.filters) for attempt in existing_attempts]
    generated: list[SearchAttempt] = []
    dropped_fields: set[SearchField] = set()

    for priority in priorities:
        if len(generated) >= MAX_ALTERNATIVE_ATTEMPTS:
            break
        tier_rules = [rule for rule in ordered_rules if rule.priority == priority]
        tier_rules.sort(key=lambda rule: _field_order(rule.field))
        next_frontier = []

        for seed in frontier:
            candidate_filters = seed
            for rule in tier_rules:
                dropped_fields.add(rule.field)
                updated_filters = _with_field_value(candidate_filters, rule.field, None)
                if updated_filters is not None:
                    candidate_filters = updated_filters
            if not _contains_filter_state(next_frontier, candidate_filters):
                next_frontier.append(candidate_filters)

        frontier = next_frontier
        for candidate_filters in frontier:
            state = _filter_state(candidate_filters)
            if _state_in_list(state, emitted_states):
                continue
            attempt = _make_alternative_attempt(
                baseline_filters,
                candidate_filters,
                dropped_fields,
                sequence=len(existing_attempts) + len(generated) + 1,
            )
            if attempt is None:
                continue
            generated.append(attempt)
            emitted_states.append(state)
            if len(generated) >= MAX_ALTERNATIVE_ATTEMPTS:
                return generated

    return generated


def _validate_alternative_drop_safety(strategy: SearchStrategy) -> None:
    constraint_strengths = {
        constraint.field: constraint.strength for constraint in strategy.constraints
    }
    for rule in strategy.alternative_drop_rules:
        if constraint_strengths.get(rule.field) is ConstraintStrength.HARD:
            raise ValueError(
                "alternative drop rules must not target HARD constraints: "
                f"{rule.field.value}"
            )


def _make_alternative_attempt(
    baseline_filters: SearchFilters,
    candidate_filters: SearchFilters,
    dropped_fields: set[SearchField],
    sequence: int,
) -> SearchAttempt | None:
    adjustments = []
    for field in SearchField:
        original_value = getattr(baseline_filters, field.value)
        adjusted_value = getattr(candidate_filters, field.value)
        if original_value == adjusted_value:
            continue
        action = (
            AdjustmentAction.DROPPED
            if field in dropped_fields and adjusted_value is None
            else AdjustmentAction.WIDENED
        )
        adjustments.append(
            FieldAdjustment(
                field=field,
                action=action,
                original_value=original_value,
                adjusted_value=adjusted_value,
            )
        )

    if not adjustments:
        return None
    return SearchAttempt(
        sequence=sequence,
        mode=MatchMode.ALTERNATIVE,
        filters=candidate_filters,
        adjustments=adjustments,
    )


def _apply_one_direction_rules(
    baseline_filters: SearchFilters,
    incoming_filters: SearchFilters,
    rules: list[RelaxationRule],
    priority: int,
) -> SearchFilters:
    transformed_filters = incoming_filters
    for rule in rules:
        if rule.priority > priority:
            break
        if (
            USEFUL_RELAXATION_DIRECTIONS[rule.field] is None
            and rule.direction is RelaxationDirection.BOTH
        ):
            continue
        adjusted_value = _adjusted_value(
            baseline_filters,
            rule,
            _effective_direction(rule),
        )
        if adjusted_value is None:
            continue
        updated_filters = _with_field_value(
            transformed_filters,
            rule.field,
            adjusted_value,
        )
        if updated_filters is not None:
            transformed_filters = updated_filters
    return transformed_filters


def _make_near_attempt(
    baseline_filters: SearchFilters,
    candidate_filters: SearchFilters,
    sequence: int,
) -> SearchAttempt | None:
    adjustments = []
    for field in SearchField:
        original_value = getattr(baseline_filters, field.value)
        adjusted_value = getattr(candidate_filters, field.value)
        if original_value != adjusted_value:
            adjustments.append(
                FieldAdjustment(
                    field=field,
                    action=AdjustmentAction.WIDENED,
                    original_value=original_value,
                    adjusted_value=adjusted_value,
                )
            )

    if not adjustments:
        return None
    return SearchAttempt(
        sequence=sequence,
        mode=MatchMode.NEAR,
        filters=candidate_filters,
        adjustments=adjustments,
    )


def _with_field_value(
    filters: SearchFilters,
    field: SearchField,
    value: object,
) -> SearchFilters | None:
    data = filters.model_dump()
    data[field.value] = value
    try:
        return SearchFilters.model_validate(data)
    except ValidationError:
        return None


def _contains_filter_state(
    filters_list: list[SearchFilters], candidate: SearchFilters
) -> bool:
    candidate_state = _filter_state(candidate)
    return any(
        candidate_state == _filter_state(existing_filters)
        for existing_filters in filters_list
    )


def _state_in_list(
    candidate: tuple[object, ...], states: list[tuple[object, ...]]
) -> bool:
    return any(candidate == state for state in states)


def _adjusted_value(
    baseline_filters: SearchFilters,
    rule: RelaxationRule,
    direction: RelaxationDirection,
) -> object | None:
    original_value = getattr(baseline_filters, rule.field.value)
    magnitude = _effective_magnitude(original_value, rule)

    if direction is RelaxationDirection.INCREASE:
        adjusted_value = original_value + magnitude
    else:
        adjusted_value = original_value - magnitude

    if USEFUL_RELAXATION_DIRECTIONS[rule.field] is None:
        adjusted_value = int(adjusted_value)
        if adjusted_value < 0:
            return None
    if adjusted_value == original_value:
        return None
    return adjusted_value


def _effective_magnitude(original_value: object, rule: RelaxationRule) -> Decimal:
    candidates = []
    if rule.max_delta is not None:
        candidates.append(_to_decimal(rule.max_delta))
    if rule.max_percentage is not None:
        candidates.append(
            abs(_to_decimal(original_value))
            * _to_decimal(rule.max_percentage)
            / Decimal("100")
        )
    return min(candidates)


def _to_decimal(numeric_value: object) -> Decimal:
    if isinstance(numeric_value, Decimal):
        return numeric_value
    try:
        return Decimal(str(numeric_value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("relaxation value must be numeric") from exc


def _effective_direction(rule: RelaxationRule) -> RelaxationDirection:
    useful_direction = USEFUL_RELAXATION_DIRECTIONS[rule.field]
    if useful_direction is None:
        return rule.direction
    return useful_direction


def _validate_directions(rules: list[RelaxationRule]) -> None:
    for rule in rules:
        useful_direction = USEFUL_RELAXATION_DIRECTIONS[rule.field]
        if (
            useful_direction is not None
            and rule.direction not in {useful_direction, RelaxationDirection.BOTH}
        ):
            raise ValueError(
                f"{rule.field.value} can only be relaxed in the "
                f"{useful_direction.value} direction"
            )


def _field_order(field: SearchField) -> int:
    return list(SearchField).index(field)


def _filter_state(filters: SearchFilters) -> tuple[object, ...]:
    return tuple(getattr(filters, field.value) for field in SearchField)
