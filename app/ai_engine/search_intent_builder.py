from app.ai_engine.search_intent import ParsedSearchIntent, SearchFieldDirective
from app.ai_engine.search_strategy import (
    AlternativeDropRule,
    ConstraintStrength,
    MatchMode,
    RankingPreference,
    RelaxationRule,
    SearchConstraint,
    SearchField,
    SearchStrategy,
    build_default_search_strategy,
)

_PREFERENCE_WEIGHT = 2.0
_ALTERNATIVE_PRIORITY = 1
_CATEGORICAL_FIELDS = frozenset(
    {
        SearchField.LOCATION,
        SearchField.PROPERTY_TYPE,
        SearchField.PAYMENT_METHOD,
    }
)


def build_search_strategy_from_intent(
    intent: ParsedSearchIntent,
) -> SearchStrategy:
    base_strategy = build_default_search_strategy(intent.filters)
    if not intent.directives:
        return base_strategy

    directives_by_field = {
        directive.field: directive for directive in intent.directives
    }
    constraint_strengths = _effective_constraint_strengths(
        base_strategy.constraints,
        directives_by_field,
    )
    alternative_drop_rules = _alternative_drop_rules(
        base_strategy.constraints,
        directives_by_field,
        constraint_strengths,
    )
    relaxation_rules = [
        rule
        for rule in base_strategy.relaxation_rules
        if constraint_strengths[rule.field] is ConstraintStrength.SOFT
    ]
    return SearchStrategy(
        filters=base_strategy.filters,
        constraints=_search_constraints(
            base_strategy.constraints,
            constraint_strengths,
        ),
        match_mode=_match_mode(relaxation_rules, alternative_drop_rules),
        ranking_preferences=_ranking_preferences(
            base_strategy.constraints,
            directives_by_field,
        ),
        relaxation_rules=relaxation_rules,
        alternative_drop_rules=alternative_drop_rules,
    )


def _search_constraints(
    constraints: list[SearchConstraint],
    constraint_strengths: dict[SearchField, ConstraintStrength],
) -> list[SearchConstraint]:
    return [
        SearchConstraint(
            field=constraint.field,
            strength=constraint_strengths[constraint.field],
        )
        for constraint in constraints
    ]


def _effective_constraint_strengths(
    constraints: list[SearchConstraint],
    directives_by_field: dict[SearchField, SearchFieldDirective],
) -> dict[SearchField, ConstraintStrength]:
    strengths = {}
    for constraint in constraints:
        directive = directives_by_field.get(constraint.field)
        if directive is None:
            strengths[constraint.field] = constraint.strength
        elif directive.strength is not None:
            strengths[constraint.field] = directive.strength
        else:
            strengths[constraint.field] = ConstraintStrength.SOFT
    return strengths


def _ranking_preferences(
    constraints: list[SearchConstraint],
    directives_by_field: dict[SearchField, SearchFieldDirective],
) -> list[RankingPreference]:
    return [
        RankingPreference(field=constraint.field, weight=_PREFERENCE_WEIGHT)
        for constraint in constraints
        if (directive := directives_by_field.get(constraint.field)) is not None
        and directive.preferred
    ]


def _alternative_drop_rules(
    constraints: list[SearchConstraint],
    directives_by_field: dict[SearchField, SearchFieldDirective],
    constraint_strengths: dict[SearchField, ConstraintStrength],
) -> list[AlternativeDropRule]:
    return [
        AlternativeDropRule(field=constraint.field, priority=_ALTERNATIVE_PRIORITY)
        for constraint in constraints
        if (directive := directives_by_field.get(constraint.field)) is not None
        if constraint_strengths[constraint.field] is ConstraintStrength.SOFT
        and _allows_alternative(constraint.field, directive)
    ]


def _allows_alternative(
    field: SearchField,
    directive: SearchFieldDirective,
) -> bool:
    if field not in _CATEGORICAL_FIELDS:
        return directive.allow_alternative
    return (
        directive.allow_alternative
        or directive.preferred
        or directive.strength is ConstraintStrength.SOFT
    )


def _match_mode(
    relaxation_rules: list[RelaxationRule],
    alternative_drop_rules: list[AlternativeDropRule],
) -> MatchMode:
    if alternative_drop_rules:
        return MatchMode.ALTERNATIVE
    if relaxation_rules:
        return MatchMode.NEAR
    return MatchMode.EXACT
