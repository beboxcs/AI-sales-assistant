from pydantic import ValidationError

from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_planner import (
    AdjustmentAction,
    FieldAdjustment,
    SearchAttempt,
    SearchPlan,
)
from app.ai_engine.search_strategy import (
    MatchMode,
    RelaxationDirection,
    RelaxationRule,
    SearchField,
    SearchStrategy,
)


# The exact attempt is not counted. This cap keeps the MVP plan bounded.
MAX_NEAR_ATTEMPTS = 12


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
    """Build an exact baseline and bounded numeric NEAR attempts."""
    baseline_filters = strategy.filters.model_copy(deep=True)
    attempts = [
        SearchAttempt(
            sequence=1,
            mode=MatchMode.EXACT,
            filters=baseline_filters,
        )
    ]

    if strategy.match_mode is MatchMode.EXACT or not strategy.relaxation_rules:
        return SearchPlan(attempts=attempts)

    _validate_directions(strategy.relaxation_rules)
    near_attempts = _build_near_attempts(
        baseline_filters, strategy.relaxation_rules
    )
    attempts.extend(near_attempts)
    return SearchPlan(attempts=attempts)


def _build_near_attempts(
    baseline_filters: SearchFilters,
    rules: list[RelaxationRule],
) -> list[SearchAttempt]:
    ordered_rules = sorted(
        rules,
        key=lambda rule: (rule.priority, _field_order(rule.field)),
    )
    priorities = sorted({rule.priority for rule in ordered_rules})
    frontier = [baseline_filters]
    generated: list[SearchAttempt] = []
    emitted_states = [_filter_state(baseline_filters)]

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

        for candidate_filters in frontier:
            state = _filter_state(candidate_filters)
            if _state_in_list(state, emitted_states):
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
            if len(generated) >= MAX_NEAR_ATTEMPTS:
                return generated

    return generated


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


def _effective_magnitude(original_value: float | int, rule: RelaxationRule) -> float:
    candidates = []
    if rule.max_delta is not None:
        candidates.append(rule.max_delta)
    if rule.max_percentage is not None:
        candidates.append(abs(original_value) * rule.max_percentage / 100)
    return min(candidates)


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
