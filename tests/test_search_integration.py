from collections.abc import Generator
from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_orchestrator import (
    SearchIntelligenceResult,
    execute_search_intelligence,
)
from app.ai_engine.search_plan_builder import build_search_plan
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
from app.database import engine
from app.domain.enums import UnitStatus
from app.models import (
    Company,
    Compound,
    InventoryScope,
    PaymentPlan,
    PaymentPlanInventoryState,
    Unit,
    UnitInventoryState,
)
from app.search import service as search_service
from app.search.contracts import SearchExecutionContext


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    assert engine.dialect.name == "postgresql"

    connection = engine.connect()
    outer_transaction = connection.begin()
    session = Session(
        bind=connection,
        autoflush=False,
        join_transaction_mode="create_savepoint",
    )
    try:
        yield session
    finally:
        session.close()
        if outer_transaction.is_active:
            outer_transaction.rollback()
        connection.close()


def test_exact_search_runs_end_to_end(db_session, monkeypatch):
    company = _add_company(db_session)
    compound = _add_compound(db_session, company, "compound:exact")
    scope = _add_scope(db_session, company, "scope:exact")
    unit = _add_unit(db_session, company, compound, "A101")
    _add_unit_state(db_session, company, unit, scope)
    strategy = _strategy(
        SearchFilters(
            location="New Cairo",
            property_type="Apartment",
            bedrooms=3,
        )
    )

    result = _run(
        db_session,
        monkeypatch,
        strategy=strategy,
        context=_context(company, scope),
    )

    assert len(result.results) == 1
    item = result.results[0]
    assert item.candidate.unit_id == unit.id
    assert item.candidate.matched_scope_keys == (scope.scope_key,)
    assert item.ranking.rank == 1
    assert item.ranking.evaluation.attempt_sequence == 1
    assert item.ranking.evaluation.unmet_soft_fields == ()
    assert item.explanation.unit_id == unit.id
    assert item.explanation.strengths


def test_near_relaxation_finds_candidate_only_after_exact_misses(
    db_session,
    monkeypatch,
):
    company = _add_company(db_session)
    compound = _add_compound(db_session, company, "compound:near")
    scope = _add_scope(db_session, company, "scope:near")
    unit = _add_unit(
        db_session,
        company,
        compound,
        "N101",
        base_price=Decimal("5000000.00"),
    )
    _add_unit_state(db_session, company, unit, scope)
    strategy = _strategy(
        SearchFilters(max_price=4_000_000),
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

    result = _run(
        db_session,
        monkeypatch,
        strategy=strategy,
        plan=plan,
        context=_context(company, scope),
    )

    assert [attempt.sequence for attempt in plan.attempts] == [1, 2]
    assert result.results[0].candidate.unit_id == unit.id
    assert result.results[0].ranking.evaluation.attempt_sequence == 2
    assert result.results[0].explanation.tradeoffs[0].message_key == "price_not_matched"
    assert len(result.results) == 1


def test_alternative_drop_finds_candidate_only_after_drop(
    db_session,
    monkeypatch,
):
    company = _add_company(db_session)
    compound = _add_compound(
        db_session,
        company,
        "compound:alternative",
        location="Sheikh Zayed",
    )
    scope = _add_scope(db_session, company, "scope:alternative")
    unit = _add_unit(db_session, company, compound, "ALT101")
    _add_unit_state(db_session, company, unit, scope)
    strategy = _strategy(
        SearchFilters(location="New Cairo", property_type="Apartment"),
        match_mode=MatchMode.ALTERNATIVE,
        drop_rules=[AlternativeDropRule(field=SearchField.LOCATION)],
    )
    plan = build_search_plan(strategy)

    result = _run(
        db_session,
        monkeypatch,
        strategy=strategy,
        plan=plan,
        context=_context(company, scope),
    )

    assert [attempt.mode for attempt in plan.attempts] == [
        MatchMode.EXACT,
        MatchMode.ALTERNATIVE,
    ]
    item = result.results[0]
    assert item.candidate.unit_id == unit.id
    assert item.ranking.evaluation.attempt_sequence == 2
    assert item.ranking.evaluation.field_evaluations[0].was_dropped is True
    assert item.explanation.tradeoffs[0].message_key == "location_dropped"


def test_first_appearance_wins_across_real_attempts(db_session, monkeypatch):
    company = _add_company(db_session)
    compound = _add_compound(db_session, company, "compound:first")
    scope = _add_scope(db_session, company, "scope:first")
    unit = _add_unit(db_session, company, compound, "FIRST")
    _add_unit_state(db_session, company, unit, scope)
    strategy = _strategy(
        SearchFilters(max_price=6_000_000),
        match_mode=MatchMode.NEAR,
        relaxation_rules=[
            RelaxationRule(
                field=SearchField.MAX_PRICE,
                max_delta=1_000_000,
                direction=RelaxationDirection.INCREASE,
            )
        ],
    )

    result = _run(
        db_session,
        monkeypatch,
        strategy=strategy,
        context=_context(company, scope),
    )

    assert [item.candidate.unit_id for item in result.results] == [unit.id]
    assert result.results[0].ranking.evaluation.attempt_sequence == 1


def test_global_ranking_combines_exact_and_near_candidates(
    db_session,
    monkeypatch,
):
    company = _add_company(db_session)
    compound = _add_compound(db_session, company, "compound:ranking")
    scope = _add_scope(db_session, company, "scope:ranking")
    exact = _add_unit(
        db_session,
        company,
        compound,
        "EXACT",
        base_price=Decimal("5000000.00"),
    )
    near = _add_unit(
        db_session,
        company,
        compound,
        "NEAR",
        base_price=Decimal("6000000.00"),
    )
    _add_unit_state(db_session, company, exact, scope)
    _add_unit_state(db_session, company, near, scope)
    strategy = _strategy(
        SearchFilters(max_price=5_000_000),
        match_mode=MatchMode.NEAR,
        relaxation_rules=[
            RelaxationRule(
                field=SearchField.MAX_PRICE,
                max_delta=1_000_000,
                direction=RelaxationDirection.INCREASE,
            )
        ],
    )

    result = _run(
        db_session,
        monkeypatch,
        strategy=strategy,
        context=_context(company, scope),
    )

    assert [item.candidate.external_unit_id for item in result.results] == [
        "EXACT",
        "NEAR",
    ]
    assert [item.ranking.rank for item in result.results] == [1, 2]
    assert [item.explanation.unit_id for item in result.results] == [
        exact.id,
        near.id,
    ]


def test_plan_aware_price_uses_total_price_not_base_price(
    db_session,
    monkeypatch,
):
    company = _add_company(db_session)
    compound = _add_compound(db_session, company, "compound:plans")
    scope = _add_scope(db_session, company, "scope:plans")
    excluded = _add_unit(
        db_session,
        company,
        compound,
        "PLAN-EXCLUDED",
        base_price=Decimal("5000000.00"),
    )
    included = _add_unit(
        db_session,
        company,
        compound,
        "PLAN-INCLUDED",
        base_price=Decimal("7000000.00"),
    )
    _add_unit_state(db_session, company, excluded, scope)
    _add_unit_state(db_session, company, included, scope)
    excluded_plan = _add_plan(
        db_session,
        company,
        excluded,
        "plan:excluded",
        total_price=Decimal("7000000.00"),
    )
    included_plan = _add_plan(
        db_session,
        company,
        included,
        "plan:included",
        total_price=Decimal("5500000.00"),
    )
    _add_plan_state(db_session, company, excluded_plan, scope)
    _add_plan_state(db_session, company, included_plan, scope)
    strategy = _strategy(
        SearchFilters(payment_method="Installment", max_price=6_000_000)
    )

    result = _run(
        db_session,
        monkeypatch,
        strategy=strategy,
        context=_context(company, scope),
    )

    assert [item.candidate.external_unit_id for item in result.results] == [
        "PLAN-INCLUDED"
    ]


def test_unit_and_payment_plan_must_be_present_in_same_scope(
    db_session,
    monkeypatch,
):
    company = _add_company(db_session)
    compound = _add_compound(db_session, company, "compound:scope")
    unit_scope = _add_scope(db_session, company, "scope:unit")
    plan_scope = _add_scope(db_session, company, "scope:plan")
    unit = _add_unit(db_session, company, compound, "SCOPE101")
    _add_unit_state(db_session, company, unit, unit_scope)
    plan = _add_plan(db_session, company, unit, "plan:scope")
    _add_plan_state(db_session, company, plan, plan_scope)
    strategy = _strategy(SearchFilters(payment_method="Installment"))

    result = _run(
        db_session,
        monkeypatch,
        strategy=strategy,
        context=_context(company, unit_scope, plan_scope),
    )

    assert result.results == ()


def test_default_available_only_behavior_flows_through_orchestrator(
    db_session,
    monkeypatch,
):
    company = _add_company(db_session)
    compound = _add_compound(db_session, company, "compound:status")
    scope = _add_scope(db_session, company, "scope:status")
    available = _add_unit(db_session, company, compound, "AVAILABLE")
    _add_unit(db_session, company, compound, "HOLD", status=UnitStatus.HOLD.value)
    _add_unit(db_session, company, compound, "SOLD", status=UnitStatus.SOLD.value)
    for unit in (available,):
        _add_unit_state(db_session, company, unit, scope)
    for external_id, status in (("HOLD", UnitStatus.HOLD.value), ("SOLD", UnitStatus.SOLD.value)):
        unit = db_session.query(Unit).filter_by(external_unit_id=external_id).one()
        _add_unit_state(db_session, company, unit, scope)
    strategy = _strategy(SearchFilters())

    result = _run(
        db_session,
        monkeypatch,
        strategy=strategy,
        context=_context(company, scope),
    )

    assert [item.candidate.external_unit_id for item in result.results] == [
        "AVAILABLE"
    ]


def test_tenant_isolation_is_preserved_through_orchestrator(
    db_session,
    monkeypatch,
):
    company_a = _add_company(db_session)
    company_b = _add_company(db_session)
    compound_a = _add_compound(db_session, company_a, "compound:shared")
    compound_b = _add_compound(db_session, company_b, "compound:shared")
    scope_a = _add_scope(db_session, company_a, "scope:shared")
    scope_b = _add_scope(db_session, company_b, "scope:shared")
    unit_a = _add_unit(db_session, company_a, compound_a, "A101")
    unit_b = _add_unit(db_session, company_b, compound_b, "A101")
    _add_unit_state(db_session, company_a, unit_a, scope_a)
    _add_unit_state(db_session, company_b, unit_b, scope_b)
    strategy = _strategy(SearchFilters())

    result = _run(
        db_session,
        monkeypatch,
        strategy=strategy,
        context=_context(company_a, scope_a),
    )

    assert [item.candidate.unit_id for item in result.results] == [unit_a.id]


def test_empty_resolved_scopes_fail_closed_through_orchestrator(
    db_session,
    monkeypatch,
):
    company = _add_company(db_session)
    compound = _add_compound(db_session, company, "compound:empty")
    scope = _add_scope(db_session, company, "scope:empty")
    unit = _add_unit(db_session, company, compound, "EMPTY")
    _add_unit_state(db_session, company, unit, scope)
    strategy = _strategy(SearchFilters())

    result = _run(
        db_session,
        monkeypatch,
        strategy=strategy,
        context=_context(company),
        scopes=(),
    )

    assert result.results == ()


def test_same_external_unit_id_in_different_compounds_remains_distinct(
    db_session,
    monkeypatch,
):
    company = _add_company(db_session)
    first_compound = _add_compound(db_session, company, "compound:first")
    second_compound = _add_compound(db_session, company, "compound:second")
    scope = _add_scope(db_session, company, "scope:identity")
    first = _add_unit(db_session, company, first_compound, "A101")
    second = _add_unit(db_session, company, second_compound, "A101")
    _add_unit_state(db_session, company, first, scope)
    _add_unit_state(db_session, company, second, scope)
    strategy = _strategy(SearchFilters())

    result = _run(
        db_session,
        monkeypatch,
        strategy=strategy,
        context=_context(company, scope),
    )

    assert {item.candidate.unit_id for item in result.results} == {first.id, second.id}
    assert len(result.results) == 2


def test_integrated_search_is_deterministic(db_session, monkeypatch):
    company = _add_company(db_session)
    compound = _add_compound(db_session, company, "compound:deterministic")
    scope = _add_scope(db_session, company, "scope:deterministic")
    unit = _add_unit(db_session, company, compound, "DET101")
    _add_unit_state(db_session, company, unit, scope)
    strategy = _strategy(SearchFilters(location="New Cairo"))
    context = _context(company, scope)

    first = _run(db_session, monkeypatch, strategy=strategy, context=context)
    second = _run(db_session, monkeypatch, strategy=strategy, context=context)

    assert isinstance(first, SearchIntelligenceResult)
    assert first == second


def _run(
    session: Session,
    monkeypatch,
    *,
    strategy: SearchStrategy,
    context: SearchExecutionContext,
    plan=None,
    scopes: tuple[str, ...] | None = None,
) -> SearchIntelligenceResult:
    monkeypatch.setattr(
        search_service,
        "SessionLocal",
        lambda: _ExistingSessionContext(session),
    )
    if scopes is not None:
        context = SearchExecutionContext(
            company_id=context.company_id,
            resolved_scope_keys=scopes,
        )
    return execute_search_intelligence(
        strategy=strategy,
        plan=plan or build_search_plan(strategy),
        context=context,
    )


class _ExistingSessionContext:
    def __init__(self, session: Session):
        self.session = session

    def __enter__(self) -> Session:
        return self.session

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False


def _strategy(
    filters: SearchFilters,
    *,
    match_mode: MatchMode = MatchMode.EXACT,
    relaxation_rules=None,
    drop_rules=None,
) -> SearchStrategy:
    populated_fields = [
        field
        for field in SearchField
        if getattr(filters, field.value) is not None
    ]
    return SearchStrategy(
        filters=filters,
        constraints=[
            SearchConstraint(
                field=field,
                strength=ConstraintStrength.SOFT,
            )
            for field in populated_fields
        ],
        match_mode=match_mode,
        relaxation_rules=relaxation_rules or [],
        alternative_drop_rules=drop_rules or [],
    )


def _context(
    company: Company,
    *scopes: InventoryScope,
) -> SearchExecutionContext:
    return SearchExecutionContext(
        company_id=company.id,
        resolved_scope_keys=tuple(scope.scope_key for scope in scopes),
    )


def _add_company(session: Session) -> Company:
    token = uuid4().hex
    company = Company(id=_test_id(), name=f"Company {token}")
    session.add(company)
    session.flush()
    return company


def _add_compound(
    session: Session,
    company: Company,
    compound_key: str,
    *,
    location: str = "New Cairo",
) -> Compound:
    compound = Compound(
        id=_test_id(),
        company_id=company.id,
        compound_key=compound_key,
        name=f"Compound {compound_key}",
        location=location,
    )
    session.add(compound)
    session.flush()
    return compound


def _add_scope(session: Session, company: Company, scope_key: str) -> InventoryScope:
    scope = InventoryScope(
        id=_test_id(),
        company_id=company.id,
        compound_id=None,
        scope_key=scope_key,
        source_identifier=f"source:{scope_key}",
    )
    session.add(scope)
    session.flush()
    return scope


def _add_unit(
    session: Session,
    company: Company,
    compound: Compound,
    external_unit_id: str,
    *,
    property_type: str = "Apartment",
    bedrooms: int | None = 3,
    bathrooms: int | None = 2,
    base_price: Decimal | None = Decimal("5000000.00"),
    area_sqm: Decimal | None = Decimal("150.00"),
    status: str = UnitStatus.AVAILABLE.value,
) -> Unit:
    unit = Unit(
        id=_test_id(),
        company_id=company.id,
        compound_id=compound.id,
        external_unit_id=external_unit_id,
        property_type=property_type,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        base_price=base_price,
        base_currency="EGP",
        status=status,
        area_sqm=area_sqm,
        delivery_date=date(2028, 6, 30),
        source_last_updated=date(2026, 9, 5),
    )
    session.add(unit)
    session.flush()
    return unit


def _add_unit_state(
    session: Session,
    company: Company,
    unit: Unit,
    scope: InventoryScope,
    *,
    present: bool = True,
) -> UnitInventoryState:
    state = UnitInventoryState(
        id=_test_id(),
        company_id=company.id,
        unit_id=unit.id,
        inventory_scope_id=scope.id,
        is_present_in_latest_inventory=present,
    )
    session.add(state)
    session.flush()
    return state


def _add_plan(
    session: Session,
    company: Company,
    unit: Unit,
    plan_key: str,
    *,
    total_price: Decimal | None = Decimal("6000000.00"),
) -> PaymentPlan:
    plan = PaymentPlan(
        id=_test_id(),
        company_id=company.id,
        unit_id=unit.id,
        plan_key=plan_key,
        source_plan_id=f"source:{plan_key}",
        plan_name=f"Plan {plan_key}",
        payment_method="Installment",
        down_payment=Decimal("500000.00"),
        installment_years=Decimal("7.00"),
        total_price=total_price,
        currency="EGP",
    )
    session.add(plan)
    session.flush()
    return plan


def _add_plan_state(
    session: Session,
    company: Company,
    plan: PaymentPlan,
    scope: InventoryScope,
    *,
    present: bool = True,
) -> PaymentPlanInventoryState:
    state = PaymentPlanInventoryState(
        id=_test_id(),
        company_id=company.id,
        payment_plan_id=plan.id,
        inventory_scope_id=scope.id,
        is_present_in_latest_inventory=present,
    )
    session.add(state)
    session.flush()
    return state


def _test_id() -> int:
    return -((uuid4().int % 1_000_000_000) + 1)
