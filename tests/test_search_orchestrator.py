from datetime import date
from decimal import Decimal

import pytest

from app.ai_engine.candidate_evaluation import CandidateContractViolationError
from app.ai_engine.search_orchestrator import (
    SearchIntelligenceContractError,
    execute_search_intelligence,
)
from app.ai_engine.search_plan_builder import build_search_plan
from app.ai_engine.search_planner import SearchAttempt, SearchPlan
from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_strategy import (
    AlternativeDropRule,
    ConstraintStrength,
    MatchMode,
    RelaxationDirection,
    RelaxationRule,
    SearchConstraint,
    SearchField,
    SearchStrategy,
)
from app.domain.enums import Currency, UnitStatus
from app.search import service as search_service
from app.search.contracts import (
    PaymentPlanCandidate,
    SearchExecutionContext,
    UnitCandidate,
)


def test_exact_candidate_flow_and_final_item_assembly(monkeypatch):
    strategy = _strategy(SearchFilters(location="New Cairo"))
    plan = SearchPlan(
        attempts=[
            SearchAttempt(
                sequence=1,
                mode=MatchMode.EXACT,
                filters=strategy.filters,
            )
        ]
    )
    candidate = _candidate(unit_id=7)
    calls = []

    def execute_search(*, filters, context):
        calls.append((filters, context))
        return [candidate]

    monkeypatch.setattr(search_service, "execute_search", execute_search)
    result = execute_search_intelligence(
        strategy=strategy,
        plan=plan,
        context=_context(),
    )

    assert result.results[0].candidate is candidate
    assert result.results[0].ranking.evaluation.unit_id == 7
    assert result.results[0].explanation.unit_id == 7
    assert result.results[0].ranking.score == Decimal("1")
    assert calls == [(strategy.filters, _context())]


def test_exact_near_alternative_execute_in_plan_order(monkeypatch):
    strategy = _strategy(
        SearchFilters(location="New Cairo", max_price=8_000_000),
        match_mode=MatchMode.ALTERNATIVE,
        relaxation_rules=[
            RelaxationRule(
                field=SearchField.MAX_PRICE,
                max_delta=1_000_000,
                direction=RelaxationDirection.INCREASE,
            )
        ],
        drop_rules=[AlternativeDropRule(field=SearchField.LOCATION)],
    )
    plan = build_search_plan(strategy)
    calls = []

    def execute_search(*, filters, context):
        calls.append(filters)
        return []

    monkeypatch.setattr(search_service, "execute_search", execute_search)
    execute_search_intelligence(strategy=strategy, plan=plan, context=_context())

    assert [filters for filters in calls] == [attempt.filters for attempt in plan.attempts]
    assert [attempt.sequence for attempt in plan.attempts] == [1, 2, 3]


def test_duplicate_unit_id_within_attempt_raises_contract_error(monkeypatch):
    strategy = _strategy(SearchFilters())
    plan = _exact_plan(strategy)

    monkeypatch.setattr(
        search_service,
        "execute_search",
        lambda **_: [_candidate(unit_id=1), _candidate(unit_id=1)],
    )

    with pytest.raises(SearchIntelligenceContractError, match="duplicate unit_id"):
        execute_search_intelligence(strategy=strategy, plan=plan, context=_context())


def test_same_unit_across_attempts_keeps_first_appearance_and_attempt(monkeypatch):
    strategy = _strategy(
        SearchFilters(location="New Cairo", max_price=8_000_000),
        match_mode=MatchMode.NEAR,
        relaxation_rules=[
            RelaxationRule(
                field=SearchField.MAX_PRICE,
                max_delta=1_000_000,
                direction=RelaxationDirection.INCREASE,
            )
        ],
    )
    plan = build_search_plan(strategy)
    first = _candidate(unit_id=1, base_price=Decimal("8000000"))
    later = _candidate(unit_id=1, base_price=Decimal("8500000"))
    responses = iter(([first], [later]))
    calls = []

    def execute_search(*, filters, context):
        calls.append(filters)
        return next(responses)

    monkeypatch.setattr(search_service, "execute_search", execute_search)
    result = execute_search_intelligence(strategy=strategy, plan=plan, context=_context())

    assert len(result.results) == 1
    assert result.results[0].candidate is first
    assert result.results[0].ranking.evaluation.attempt_sequence == 1
    assert len(calls) == 2


def test_global_ranking_combines_candidates_from_multiple_attempts(monkeypatch):
    strategy = _strategy(
        SearchFilters(max_price=8_000_000),
        match_mode=MatchMode.NEAR,
        relaxation_rules=[
            RelaxationRule(
                field=SearchField.MAX_PRICE,
                max_delta=1_000_000,
                direction=RelaxationDirection.INCREASE,
            )
        ],
    )
    plan = build_search_plan(strategy)
    responses = iter(
        (
            [_candidate(unit_id=1, base_price=Decimal("7500000"))],
            [_candidate(unit_id=2, base_price=Decimal("8500000"))],
        )
    )
    monkeypatch.setattr(
        search_service,
        "execute_search",
        lambda **_: next(responses),
    )

    result = execute_search_intelligence(strategy=strategy, plan=plan, context=_context())

    assert [item.candidate.unit_id for item in result.results] == [1, 2]
    assert [item.ranking.rank for item in result.results] == [1, 2]
    assert result.results[1].explanation.tradeoffs[0].message_key == "price_relaxed"


def test_backend_exception_propagates(monkeypatch):
    strategy = _strategy(SearchFilters())
    plan = _exact_plan(strategy)
    error = RuntimeError("backend failed")

    def fail(**_):
        raise error

    monkeypatch.setattr(search_service, "execute_search", fail)

    with pytest.raises(RuntimeError) as raised:
        execute_search_intelligence(strategy=strategy, plan=plan, context=_context())
    assert raised.value is error


def test_candidate_contract_violation_propagates(monkeypatch):
    strategy = _strategy(SearchFilters(location="New Cairo"))
    plan = _exact_plan(strategy)
    monkeypatch.setattr(
        search_service,
        "execute_search",
        lambda **_: [_candidate(unit_id=1, location=None)],
    )

    with pytest.raises(CandidateContractViolationError):
        execute_search_intelligence(strategy=strategy, plan=plan, context=_context())


def test_empty_search_result_and_empty_resolved_scopes_are_valid(monkeypatch):
    strategy = _strategy(SearchFilters())
    plan = _exact_plan(strategy)
    contexts = []

    def execute_search(*, filters, context):
        contexts.append(context)
        return []

    monkeypatch.setattr(search_service, "execute_search", execute_search)
    empty_context = _context(scopes=())
    result = execute_search_intelligence(
        strategy=strategy,
        plan=plan,
        context=empty_context,
    )

    assert result.results == ()
    assert contexts == [empty_context]


def test_repeated_execution_is_deterministic(monkeypatch):
    strategy = _strategy(SearchFilters(location="New Cairo"))
    plan = _exact_plan(strategy)
    candidate = _candidate(unit_id=4)
    monkeypatch.setattr(search_service, "execute_search", lambda **_: [candidate])

    first = execute_search_intelligence(strategy=strategy, plan=plan, context=_context())
    second = execute_search_intelligence(strategy=strategy, plan=plan, context=_context())

    assert first == second


def test_final_results_contain_no_duplicate_units(monkeypatch):
    strategy = _strategy(SearchFilters())
    plan = SearchPlan(
        attempts=[
            SearchAttempt(sequence=1, mode=MatchMode.EXACT, filters=strategy.filters)
        ]
    )
    monkeypatch.setattr(
        search_service,
        "execute_search",
        lambda **_: [_candidate(unit_id=1), _candidate(unit_id=2)],
    )

    result = execute_search_intelligence(strategy=strategy, plan=plan, context=_context())

    assert len({item.candidate.unit_id for item in result.results}) == len(result.results)


def _strategy(
    filters: SearchFilters,
    *,
    match_mode: MatchMode = MatchMode.EXACT,
    relaxation_rules=None,
    drop_rules=None,
) -> SearchStrategy:
    fields = [
        field
        for field in SearchField
        if getattr(filters, field.value) is not None
    ]
    return SearchStrategy(
        filters=filters,
        constraints=[
            SearchConstraint(field=field, strength=ConstraintStrength.SOFT)
            for field in fields
        ],
        match_mode=match_mode,
        relaxation_rules=relaxation_rules or [],
        alternative_drop_rules=drop_rules or [],
    )


def _exact_plan(strategy: SearchStrategy) -> SearchPlan:
    return SearchPlan(
        attempts=[
            SearchAttempt(
                sequence=1,
                mode=MatchMode.EXACT,
                filters=strategy.filters,
            )
        ]
    )


def _context(*, scopes=("scope:one",)) -> SearchExecutionContext:
    return SearchExecutionContext(company_id=7, resolved_scope_keys=scopes)


def _candidate(
    *,
    unit_id: int,
    location: str | None = "New Cairo",
    property_type: str = "Apartment",
    bedrooms: int | None = 3,
    bathrooms: int | None = 2,
    area_sqm: Decimal | None = Decimal("150"),
    base_price: Decimal | None = Decimal("8000000"),
    payment_plans: tuple[PaymentPlanCandidate, ...] = (),
) -> UnitCandidate:
    return UnitCandidate(
        unit_id=unit_id,
        external_unit_id=f"A{unit_id}",
        compound_id=1,
        compound_name="Example Compound",
        location=location,
        property_type=property_type,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        area_sqm=area_sqm,
        base_price=base_price,
        base_currency=Currency.EGP,
        status=UnitStatus.AVAILABLE,
        delivery_date=date(2028, 1, 1),
        source_last_updated=date(2026, 9, 5),
        matched_scope_keys=("scope:one",),
        payment_plans=payment_plans,
    )
