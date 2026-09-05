from collections.abc import Generator
from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from app.ai_engine.search_filters import SearchFilters
from app.database import engine
from app.domain.enums import Currency, PaymentMethod, UnitStatus
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
from app.search.exceptions import (
    InvalidSearchScopeError,
    SearchExecutionFailedError,
)


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


def test_empty_resolved_scopes_returns_empty_without_opening_session(monkeypatch):
    def fail_if_called():
        raise AssertionError("SessionLocal must not be called for empty scopes")

    monkeypatch.setattr(search_service, "SessionLocal", fail_if_called)

    result = search_service.execute_search(
        filters=SearchFilters(),
        context=SearchExecutionContext(company_id=1, resolved_scope_keys=()),
    )

    assert result == []


def test_search_enforces_tenant_and_resolved_scope_isolation(
    db_session: Session,
    monkeypatch,
):
    company_a = _add_company(db_session)
    company_b = _add_company(db_session)
    compound_a = _add_compound(db_session, company_a, "compound:shared")
    compound_b = _add_compound(db_session, company_b, "compound:shared")
    scope_a = _add_scope(db_session, company_a, "scope:shared")
    scope_b = _add_scope(db_session, company_b, "scope:shared")
    unresolved_scope = _add_scope(db_session, company_a, "scope:unresolved")
    unit_a = _add_unit(db_session, company_a, compound_a, "A101")
    unit_b = _add_unit(db_session, company_b, compound_b, "A101")
    unresolved_unit = _add_unit(db_session, company_a, compound_a, "HIDDEN")
    _add_unit_state(db_session, company_a, unit_a, scope_a)
    _add_unit_state(db_session, company_b, unit_b, scope_b)
    _add_unit_state(db_session, company_a, unresolved_unit, unresolved_scope)

    results = _execute(
        db_session,
        monkeypatch,
        company_id=company_a.id,
        scope_keys=(scope_a.scope_key,),
    )

    assert [candidate.external_unit_id for candidate in results] == ["A101"]
    candidate = results[0]
    assert candidate.compound_id == compound_a.id
    assert candidate.compound_name == compound_a.name
    assert candidate.location == compound_a.location
    assert candidate.base_currency is Currency.EGP
    assert candidate.status is UnitStatus.AVAILABLE
    assert candidate.source_last_updated == date(2026, 9, 5)


def test_current_presence_deduplication_scope_matching_and_ordering(
    db_session: Session,
    monkeypatch,
):
    company = _add_company(db_session)
    compound = _add_compound(db_session, company, "compound:one")
    scope_a = _add_scope(db_session, company, "scope:a")
    scope_b = _add_scope(db_session, company, "scope:b")
    scope_c = _add_scope(db_session, company, "scope:c")
    first = _add_unit(db_session, company, compound, "FIRST", unit_id=101)
    multi_scope = _add_unit(db_session, company, compound, "MULTI", unit_id=202)
    absent = _add_unit(db_session, company, compound, "ABSENT", unit_id=303)
    unresolved = _add_unit(db_session, company, compound, "UNRESOLVED", unit_id=404)
    _add_unit_state(db_session, company, first, scope_a)
    _add_unit_state(db_session, company, multi_scope, scope_a)
    _add_unit_state(db_session, company, multi_scope, scope_b)
    _add_unit_state(db_session, company, absent, scope_a, present=False)
    _add_unit_state(db_session, company, unresolved, scope_c)

    results = _execute(
        db_session,
        monkeypatch,
        company_id=company.id,
        scope_keys=(scope_b.scope_key, scope_a.scope_key),
    )

    assert [candidate.unit_id for candidate in results] == [101, 202]
    assert results[0].matched_scope_keys == (scope_a.scope_key,)
    assert results[1].matched_scope_keys == (
        scope_b.scope_key,
        scope_a.scope_key,
    )


@pytest.mark.parametrize(
    ("filters", "expected_external_id"),
    [
        (SearchFilters(location="New Cairo"), "TARGET"),
        (SearchFilters(property_type="Apartment"), "TARGET"),
        (SearchFilters(bedrooms=3), "TARGET"),
        (SearchFilters(bathrooms=2), "TARGET"),
        (SearchFilters(max_price=6_000_000), "TARGET"),
        (SearchFilters(min_price=6_000_000), "OTHER"),
        (SearchFilters(max_area_sqm=200), "TARGET"),
        (SearchFilters(min_area_sqm=200), "OTHER"),
    ],
)
def test_unit_level_filters(
    db_session: Session,
    monkeypatch,
    filters: SearchFilters,
    expected_external_id: str,
):
    company = _add_company(db_session)
    target_compound = _add_compound(
        db_session,
        company,
        "compound:target",
        location="New Cairo",
    )
    other_compound = _add_compound(
        db_session,
        company,
        "compound:other",
        location="Sheikh Zayed",
    )
    scope = _add_scope(db_session, company, "scope:filters")
    target = _add_unit(
        db_session,
        company,
        target_compound,
        "TARGET",
        property_type="Apartment",
        bedrooms=3,
        bathrooms=2,
        base_price=Decimal("5000000.00"),
        area_sqm=Decimal("150.00"),
    )
    other = _add_unit(
        db_session,
        company,
        other_compound,
        "OTHER",
        property_type="Villa",
        bedrooms=4,
        bathrooms=3,
        base_price=Decimal("8000000.00"),
        area_sqm=Decimal("250.00"),
    )
    _add_unit_state(db_session, company, target, scope)
    _add_unit_state(db_session, company, other, scope)

    results = _execute(
        db_session,
        monkeypatch,
        company_id=company.id,
        scope_keys=(scope.scope_key,),
        filters=filters,
    )

    assert [candidate.external_unit_id for candidate in results] == [
        expected_external_id
    ]


@pytest.mark.parametrize(
    ("filters", "expected_plan_key"),
    [
        (SearchFilters(payment_method="Cash"), "plan:cash"),
        (SearchFilters(max_down_payment=200_000), "plan:cash"),
        (SearchFilters(min_installment_years=6), "plan:installment"),
    ],
)
def test_payment_plan_filters_return_only_matching_current_plans(
    db_session: Session,
    monkeypatch,
    filters: SearchFilters,
    expected_plan_key: str,
):
    company, scope, unit = _seed_present_unit(db_session)
    installment = _add_plan(
        db_session,
        company,
        unit,
        "plan:installment",
        payment_method="Installment",
        down_payment=Decimal("500000.00"),
        installment_years=Decimal("7.00"),
    )
    cash = _add_plan(
        db_session,
        company,
        unit,
        "plan:cash",
        payment_method="Cash",
        down_payment=Decimal("100000.00"),
        installment_years=None,
    )
    _add_plan_state(db_session, company, installment, scope)
    _add_plan_state(db_session, company, cash, scope)

    results = _execute(
        db_session,
        monkeypatch,
        company_id=company.id,
        scope_keys=(scope.scope_key,),
        filters=filters,
    )

    assert len(results) == 1
    assert [plan.plan_key for plan in results[0].payment_plans] == [
        expected_plan_key
    ]


def test_no_plan_filters_returns_all_and_only_current_same_scope_plans(
    db_session: Session,
    monkeypatch,
):
    company, scope, unit = _seed_present_unit(db_session)
    first = _add_plan(db_session, company, unit, "plan:first", plan_id=101)
    second = _add_plan(db_session, company, unit, "plan:second", plan_id=202)
    absent = _add_plan(db_session, company, unit, "plan:absent", plan_id=303)
    _add_plan_state(db_session, company, first, scope)
    _add_plan_state(db_session, company, second, scope)
    _add_plan_state(db_session, company, absent, scope, present=False)

    results = _execute(
        db_session,
        monkeypatch,
        company_id=company.id,
        scope_keys=(scope.scope_key,),
    )

    assert [plan.plan_key for plan in results[0].payment_plans] == [
        "plan:first",
        "plan:second",
    ]


def test_unit_and_plan_presence_must_overlap_in_the_same_scope(
    db_session: Session,
    monkeypatch,
):
    company = _add_company(db_session)
    compound = _add_compound(db_session, company, "compound:one")
    scope_a = _add_scope(db_session, company, "scope:a")
    scope_b = _add_scope(db_session, company, "scope:b")
    unit = _add_unit(db_session, company, compound, "A101")
    plan = _add_plan(
        db_session,
        company,
        unit,
        "plan:cash",
        payment_method="Cash",
        installment_years=None,
    )
    _add_unit_state(db_session, company, unit, scope_a)
    _add_plan_state(db_session, company, plan, scope_b)

    filtered_results = _execute(
        db_session,
        monkeypatch,
        company_id=company.id,
        scope_keys=(scope_a.scope_key, scope_b.scope_key),
        filters=SearchFilters(payment_method="Cash"),
    )
    unfiltered_results = _execute(
        db_session,
        monkeypatch,
        company_id=company.id,
        scope_keys=(scope_a.scope_key, scope_b.scope_key),
    )

    assert filtered_results == []
    assert len(unfiltered_results) == 1
    assert unfiltered_results[0].payment_plans == ()
    assert unfiltered_results[0].matched_scope_keys == (scope_a.scope_key,)


def test_plan_filtered_matched_scopes_include_only_qualifying_scopes(
    db_session: Session,
    monkeypatch,
):
    company = _add_company(db_session)
    compound = _add_compound(db_session, company, "compound:one")
    scope_a = _add_scope(db_session, company, "scope:a")
    scope_b = _add_scope(db_session, company, "scope:b")
    unit = _add_unit(db_session, company, compound, "A101")
    cash = _add_plan(
        db_session,
        company,
        unit,
        "plan:cash",
        payment_method="Cash",
        installment_years=None,
    )
    installment = _add_plan(
        db_session,
        company,
        unit,
        "plan:installment",
        payment_method="Installment",
    )
    _add_unit_state(db_session, company, unit, scope_a)
    _add_unit_state(db_session, company, unit, scope_b)
    _add_plan_state(db_session, company, cash, scope_a)
    _add_plan_state(db_session, company, installment, scope_b)

    results = _execute(
        db_session,
        monkeypatch,
        company_id=company.id,
        scope_keys=(scope_b.scope_key, scope_a.scope_key),
        filters=SearchFilters(payment_method="Cash"),
    )

    assert len(results) == 1
    assert [plan.plan_key for plan in results[0].payment_plans] == ["plan:cash"]
    assert results[0].matched_scope_keys == (scope_a.scope_key,)


def test_same_external_unit_id_in_different_compounds_remains_distinct(
    db_session: Session,
    monkeypatch,
):
    company = _add_company(db_session)
    first_compound = _add_compound(db_session, company, "compound:first")
    second_compound = _add_compound(db_session, company, "compound:second")
    scope = _add_scope(db_session, company, "scope:shared")
    first = _add_unit(db_session, company, first_compound, "A101", unit_id=101)
    second = _add_unit(db_session, company, second_compound, "A101", unit_id=202)
    _add_unit_state(db_session, company, first, scope)
    _add_unit_state(db_session, company, second, scope)

    results = _execute(
        db_session,
        monkeypatch,
        company_id=company.id,
        scope_keys=(scope.scope_key,),
    )

    assert [candidate.unit_id for candidate in results] == [101, 202]
    assert {candidate.compound_id for candidate in results} == {
        first_compound.id,
        second_compound.id,
    }


def test_unexpected_repository_failure_is_translated(monkeypatch):
    class StubSessionContext:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def fail_repository(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(search_service, "SessionLocal", StubSessionContext)
    monkeypatch.setattr(search_service, "search_candidates", fail_repository)

    with pytest.raises(SearchExecutionFailedError) as exc_info:
        search_service.execute_search(
            filters=SearchFilters(),
            context=SearchExecutionContext(
                company_id=1,
                resolved_scope_keys=("scope:a",),
            ),
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_defined_search_execution_errors_are_preserved(monkeypatch):
    class StubSessionContext:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    expected_error = InvalidSearchScopeError("invalid scope")

    def fail_repository(*args, **kwargs):
        raise expected_error

    monkeypatch.setattr(search_service, "SessionLocal", StubSessionContext)
    monkeypatch.setattr(search_service, "search_candidates", fail_repository)

    with pytest.raises(InvalidSearchScopeError) as exc_info:
        search_service.execute_search(
            filters=SearchFilters(),
            context=SearchExecutionContext(
                company_id=1,
                resolved_scope_keys=("scope:a",),
            ),
        )

    assert exc_info.value is expected_error


class _ExistingSessionContext:
    def __init__(self, session: Session):
        self.session = session

    def __enter__(self) -> Session:
        return self.session

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False


def _execute(
    session: Session,
    monkeypatch,
    *,
    company_id: int,
    scope_keys: tuple[str, ...],
    filters: SearchFilters | None = None,
):
    monkeypatch.setattr(
        search_service,
        "SessionLocal",
        lambda: _ExistingSessionContext(session),
    )
    return search_service.execute_search(
        filters=filters or SearchFilters(),
        context=SearchExecutionContext(
            company_id=company_id,
            resolved_scope_keys=scope_keys,
        ),
    )


def _seed_present_unit(session: Session) -> tuple[Company, InventoryScope, Unit]:
    company = _add_company(session)
    compound = _add_compound(session, company, "compound:one")
    scope = _add_scope(session, company, "scope:one")
    unit = _add_unit(session, company, compound, "A101")
    _add_unit_state(session, company, unit, scope)
    return company, scope, unit


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


def _add_scope(
    session: Session,
    company: Company,
    scope_key: str,
) -> InventoryScope:
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
    unit_id: int | None = None,
    property_type: str = "Apartment",
    bedrooms: int | None = 3,
    bathrooms: int | None = 2,
    base_price: Decimal | None = Decimal("5000000.00"),
    area_sqm: Decimal | None = Decimal("150.00"),
) -> Unit:
    unit = Unit(
        id=unit_id or _test_id(),
        company_id=company.id,
        compound_id=compound.id,
        external_unit_id=external_unit_id,
        property_type=property_type,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        base_price=base_price,
        base_currency="EGP",
        status="Available",
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
    plan_id: int | None = None,
    payment_method: str = "Installment",
    down_payment: Decimal | None = Decimal("500000.00"),
    installment_years: Decimal | None = Decimal("7.00"),
) -> PaymentPlan:
    plan = PaymentPlan(
        id=plan_id or _test_id(),
        company_id=company.id,
        unit_id=unit.id,
        plan_key=plan_key,
        source_plan_id=f"source:{plan_key}",
        plan_name=f"Plan {plan_key}",
        payment_method=payment_method,
        down_payment=down_payment,
        installment_years=installment_years,
        total_price=Decimal("6000000.00"),
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
