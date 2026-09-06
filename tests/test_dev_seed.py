from collections.abc import Generator
from dataclasses import replace
from decimal import Decimal
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.database import engine
from app.dev import seed
from app.models import (
    Company,
    Compound,
    InventoryScope,
    PaymentPlan,
    PaymentPlanInventoryState,
    Unit,
    UnitInventoryState,
)

EXPECTED_COUNTS = seed.SeedCounts(
    companies=1,
    scopes=2,
    compounds=2,
    units=7,
    payment_plans=6,
    unit_inventory_states=7,
    payment_plan_inventory_states=6,
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


def test_seed_creates_exact_acceptance_dataset(db_session: Session):
    counts = seed.seed_development_search_acceptance_data(db_session)
    company = _seed_company(db_session)
    scopes = _scopes_by_key(db_session, company.id)
    compounds = _compounds_by_key(db_session, company.id)
    units = _units_by_external_id(db_session, company.id)
    plans = _plans_by_key(db_session, company.id)

    assert counts == EXPECTED_COUNTS
    assert _database_counts(db_session, company.id) == EXPECTED_COUNTS
    assert company.id == seed.DEVELOPMENT_COMPANY_ID
    assert company.name == seed.DEVELOPMENT_COMPANY_NAME
    assert set(scopes) == {"scope-nc-main", "scope-zayed-main"}
    assert set(compounds) == {"palm-east", "zayed-heights"}
    assert scopes["scope-nc-main"].source_identifier == (
        "development-search-acceptance-v1:scope-nc-main"
    )
    assert scopes["scope-zayed-main"].source_identifier == (
        "development-search-acceptance-v1:scope-zayed-main"
    )
    assert scopes["scope-nc-main"].compound_id == compounds["palm-east"].id
    assert scopes["scope-zayed-main"].compound_id == compounds["zayed-heights"].id
    assert (compounds["palm-east"].name, compounds["palm-east"].location) == (
        "Palm East",
        "New Cairo",
    )
    assert (
        compounds["zayed-heights"].name,
        compounds["zayed-heights"].location,
    ) == ("Zayed Heights", "Sheikh Zayed")

    assert set(units) == {
        "A101",
        "A102",
        "A103",
        "A104",
        "A105",
        "A106",
        "Z101",
    }
    units_by_id = sorted(units.values(), key=lambda item: item.id)
    assert [unit.external_unit_id for unit in units_by_id] == [
        "A101",
        "A102",
        "A103",
        "A104",
        "A105",
        "A106",
        "Z101",
    ]
    assert all(
        units[external_id].compound_id == compounds["palm-east"].id
        for external_id in ("A101", "A102", "A103", "A104", "A105", "A106")
    )
    assert units["Z101"].compound_id == compounds["zayed-heights"].id
    assert _unit_values(units["A101"]) == (
        "Villa",
        4,
        3,
        Decimal("250.00"),
        Decimal("12000000.00"),
        "Available",
    )
    assert _unit_values(units["A102"]) == (
        "Villa",
        4,
        3,
        Decimal("260.00"),
        Decimal("12500000.00"),
        "Available",
    )
    assert _unit_values(units["A103"]) == (
        "Villa",
        3,
        3,
        Decimal("240.00"),
        Decimal("11500000.00"),
        "Available",
    )
    assert _unit_values(units["A104"]) == (
        "Apartment",
        4,
        3,
        Decimal("220.00"),
        Decimal("10000000.00"),
        "Available",
    )
    assert _unit_values(units["A105"]) == (
        "Villa",
        4,
        3,
        Decimal("255.00"),
        Decimal("10500000.00"),
        "Hold",
    )
    assert _unit_values(units["A106"]) == (
        "Villa",
        4,
        3,
        Decimal("255.00"),
        Decimal("10800000.00"),
        "Available",
    )
    assert _unit_values(units["Z101"]) == (
        "Villa",
        4,
        3,
        Decimal("250.00"),
        Decimal("11000000.00"),
        "Available",
    )
    assert all(unit.base_currency == "EGP" for unit in units.values())
    assert all(
        unit.source_last_updated == seed.SEED_SOURCE_LAST_UPDATED
        for unit in units.values()
    )

    assert set(plans) == {
        "A101-cash",
        "A101-inst-5y",
        "A102-cash",
        "A102-inst-6y",
        "A103-inst-5y",
        "A104-inst-7y",
    }
    assert _plan_values(plans["A101-cash"]) == (
        "Cash",
        Decimal("11800000.00"),
        None,
        None,
    )
    assert _plan_values(plans["A101-inst-5y"]) == (
        "Installment",
        Decimal("13000000.00"),
        Decimal("1500000.00"),
        Decimal("5.00"),
    )
    assert _plan_values(plans["A102-cash"]) == (
        "Cash",
        Decimal("12200000.00"),
        None,
        None,
    )
    assert _plan_values(plans["A102-inst-6y"]) == (
        "Installment",
        Decimal("13800000.00"),
        Decimal("2000000.00"),
        Decimal("6.00"),
    )
    assert _plan_values(plans["A103-inst-5y"]) == (
        "Installment",
        Decimal("12400000.00"),
        Decimal("1200000.00"),
        Decimal("5.00"),
    )
    assert _plan_values(plans["A104-inst-7y"]) == (
        "Installment",
        Decimal("11000000.00"),
        Decimal("900000.00"),
        Decimal("7.00"),
    )
    assert all(plan.currency == "EGP" for plan in plans.values())
    assert all(
        plan.unit.external_unit_id not in {"A105", "A106", "Z101"}
        for plan in plans.values()
    )


def test_seed_currentness_matches_unit_and_plan_scopes(db_session: Session):
    seed.seed_development_search_acceptance_data(db_session)
    company = _seed_company(db_session)
    scopes = _scopes_by_key(db_session, company.id)
    units = _units_by_external_id(db_session, company.id)
    unit_states = _unit_states_by_external_id(db_session, company.id)

    assert units["A105"].status == "Hold"
    assert unit_states["A105"].inventory_scope_id == scopes["scope-nc-main"].id
    assert unit_states["A105"].is_present_in_latest_inventory is True
    assert units["A106"].status == "Available"
    assert unit_states["A106"].inventory_scope_id == scopes["scope-nc-main"].id
    assert unit_states["A106"].is_present_in_latest_inventory is False
    assert units["A101"].status == "Available"
    assert unit_states["A101"].inventory_scope_id == scopes["scope-nc-main"].id
    assert unit_states["A101"].is_present_in_latest_inventory is True

    plan_rows = db_session.execute(
        select(PaymentPlan, PaymentPlanInventoryState, UnitInventoryState)
        .join(Unit, Unit.id == PaymentPlan.unit_id)
        .join(
            PaymentPlanInventoryState,
            PaymentPlanInventoryState.payment_plan_id == PaymentPlan.id,
        )
        .join(
            UnitInventoryState,
            (UnitInventoryState.unit_id == Unit.id)
            & (
                UnitInventoryState.inventory_scope_id
                == PaymentPlanInventoryState.inventory_scope_id
            ),
        )
        .where(PaymentPlan.company_id == company.id)
    ).all()
    assert len(plan_rows) == 6
    assert all(
        plan_state.is_present_in_latest_inventory
        for _, plan_state, _ in plan_rows
    )
    assert all(
        unit_state.is_present_in_latest_inventory
        for _, _, unit_state in plan_rows
    )


def test_noncurrent_unit_spec_cannot_create_a_current_plan_state(
    db_session: Session,
    monkeypatch,
):
    future_plan = seed._PaymentPlanSeed(
        plan_key="A106-future-plan",
        payment_method=seed.PaymentMethod.INSTALLMENT,
        total_price=Decimal("12000000"),
        down_payment=Decimal("1000000"),
        installment_years=Decimal("5"),
    )
    monkeypatch.setattr(
        seed,
        "_UNITS",
        tuple(
            replace(unit_spec, payment_plans=(future_plan,))
            if unit_spec.external_unit_id == "A106"
            else unit_spec
            for unit_spec in seed._UNITS
        ),
    )

    seed.seed_development_search_acceptance_data(db_session)
    company = _seed_company(db_session)
    unit_state = _unit_states_by_external_id(db_session, company.id)["A106"]
    plan_state = _plan_states_by_key(db_session, company.id)[
        "A106-future-plan"
    ]

    assert unit_state.is_present_in_latest_inventory is False
    assert plan_state.is_present_in_latest_inventory is False


def test_seed_rerun_is_idempotent_reconciles_owned_rows_and_preserves_unrelated_data(
    db_session: Session,
):
    first_counts = seed.seed_development_search_acceptance_data(db_session)
    company = _seed_company(db_session)
    scopes = _scopes_by_key(db_session, company.id)
    compounds = _compounds_by_key(db_session, company.id)
    units = _units_by_external_id(db_session, company.id)
    plans = _plans_by_key(db_session, company.id)
    unit_states = _unit_states_by_external_id(db_session, company.id)
    plan_states = _plan_states_by_key(db_session, company.id)

    compounds["palm-east"].name = "Wrong Name"
    scopes["scope-nc-main"].source_identifier = "wrong-source"
    units["A101"].base_price = Decimal("1")
    units["A101"].status = "Sold"
    units["A106"].status = "Hold"
    unit_states["A101"].is_present_in_latest_inventory = False
    unit_states["A106"].is_present_in_latest_inventory = True
    plans["A101-inst-5y"].total_price = Decimal("1")
    plan_states["A101-inst-5y"].is_present_in_latest_inventory = False
    unrelated = Company(id=-1_999_999_999, name="Unrelated Developer Company")
    db_session.add(unrelated)
    db_session.flush()

    second_counts = seed.seed_development_search_acceptance_data(db_session)
    refreshed_units = _units_by_external_id(db_session, company.id)
    refreshed_plans = _plans_by_key(db_session, company.id)
    refreshed_unit_states = _unit_states_by_external_id(db_session, company.id)
    refreshed_plan_states = _plan_states_by_key(db_session, company.id)

    assert first_counts == second_counts == EXPECTED_COUNTS
    assert _database_counts(db_session, company.id) == EXPECTED_COUNTS
    assert compounds["palm-east"].name == "Palm East"
    assert scopes["scope-nc-main"].source_identifier == (
        "development-search-acceptance-v1:scope-nc-main"
    )
    assert refreshed_units["A101"].base_price == Decimal("12000000")
    assert refreshed_units["A101"].status == "Available"
    assert refreshed_units["A106"].status == "Available"
    assert refreshed_unit_states["A101"].is_present_in_latest_inventory is True
    assert refreshed_unit_states["A106"].is_present_in_latest_inventory is False
    assert refreshed_plans["A101-inst-5y"].total_price == Decimal("13000000")
    assert (
        refreshed_plan_states["A101-inst-5y"].is_present_in_latest_inventory
        is True
    )
    assert db_session.get(Company, unrelated.id) is unrelated


def test_seed_isolates_overlapping_natural_keys_in_another_tenant(
    db_session: Session,
):
    other_company = Company(id=_test_id(), name=f"Other {uuid4().hex}")
    db_session.add(other_company)
    db_session.flush()
    other_compound = Compound(
        company_id=other_company.id,
        compound_key="palm-east",
        name="Unrelated Palm",
        location="Alexandria",
    )
    db_session.add(other_compound)
    db_session.flush()
    other_scope = InventoryScope(
        company_id=other_company.id,
        compound_id=other_compound.id,
        scope_key="scope-nc-main",
        source_identifier="unrelated-source",
    )
    db_session.add(other_scope)
    db_session.flush()
    other_unit = Unit(
        company_id=other_company.id,
        compound_id=other_compound.id,
        external_unit_id="A101",
        property_type="Townhouse",
        bedrooms=2,
        bathrooms=1,
        area_sqm=Decimal("99"),
        base_price=Decimal("999999"),
        base_currency="EGP",
        status="Reserved",
        delivery_date=None,
        source_last_updated=seed.SEED_SOURCE_LAST_UPDATED,
    )
    db_session.add(other_unit)
    db_session.flush()
    other_plan = PaymentPlan(
        company_id=other_company.id,
        unit_id=other_unit.id,
        plan_key="A101-cash",
        source_plan_id="unrelated-plan",
        plan_name="Unrelated Cash",
        payment_method="Cash",
        total_price=Decimal("888888"),
        down_payment=None,
        installment_years=None,
        currency="EGP",
    )
    db_session.add(other_plan)
    db_session.flush()
    other_unit_state = UnitInventoryState(
        company_id=other_company.id,
        unit_id=other_unit.id,
        inventory_scope_id=other_scope.id,
        is_present_in_latest_inventory=False,
    )
    other_plan_state = PaymentPlanInventoryState(
        company_id=other_company.id,
        payment_plan_id=other_plan.id,
        inventory_scope_id=other_scope.id,
        is_present_in_latest_inventory=False,
    )
    db_session.add_all([other_unit_state, other_plan_state])
    db_session.flush()
    unrelated_values = (
        other_compound.name,
        other_compound.location,
        other_scope.source_identifier,
        other_unit.property_type,
        other_unit.base_price,
        other_unit.status,
        other_plan.source_plan_id,
        other_plan.total_price,
        other_unit_state.is_present_in_latest_inventory,
        other_plan_state.is_present_in_latest_inventory,
    )

    first_counts = seed.seed_development_search_acceptance_data(db_session)
    second_counts = seed.seed_development_search_acceptance_data(db_session)
    seed_company = _seed_company(db_session)
    seed_units = _units_by_external_id(db_session, seed_company.id)

    assert first_counts == second_counts == EXPECTED_COUNTS
    assert _database_counts(db_session, seed_company.id) == EXPECTED_COUNTS
    assert seed_company.id == seed.DEVELOPMENT_COMPANY_ID
    assert seed_units["A101"].company_id == seed.DEVELOPMENT_COMPANY_ID
    assert seed_units["A101"].id != other_unit.id
    assert (
        other_compound.name,
        other_compound.location,
        other_scope.source_identifier,
        other_unit.property_type,
        other_unit.base_price,
        other_unit.status,
        other_plan.source_plan_id,
        other_plan.total_price,
        other_unit_state.is_present_in_latest_inventory,
        other_plan_state.is_present_in_latest_inventory,
    ) == unrelated_values


def test_seed_rejects_same_company_name_under_wrong_id_without_descendants(
    db_session: Session,
    monkeypatch,
):
    company_id = _test_id()
    company_name = f"Development Search Acceptance V1 {uuid4().hex}"
    wrong_id = _test_id()
    db_session.add(Company(id=wrong_id, name=company_name))
    db_session.flush()
    monkeypatch.setattr(seed, "DEVELOPMENT_COMPANY_ID", company_id)
    monkeypatch.setattr(seed, "DEVELOPMENT_COMPANY_NAME", company_name)

    with pytest.raises(seed.DevelopmentSeedConflictError):
        seed.seed_development_search_acceptance_data(db_session)

    _assert_no_seed_descendants(db_session, (company_id, wrong_id))


def test_seed_rejects_company_id_with_wrong_name_without_descendants(
    db_session: Session,
    monkeypatch,
):
    company_id = _test_id()
    company_name = f"Development Search Acceptance V1 {uuid4().hex}"
    db_session.add(Company(id=company_id, name="Unrelated Company"))
    db_session.flush()
    monkeypatch.setattr(seed, "DEVELOPMENT_COMPANY_ID", company_id)
    monkeypatch.setattr(seed, "DEVELOPMENT_COMPANY_NAME", company_name)

    with pytest.raises(seed.DevelopmentSeedConflictError):
        seed.seed_development_search_acceptance_data(db_session)

    _assert_no_seed_descendants(db_session, (company_id,))


def test_seed_rejects_split_company_id_and_name_collision(
    db_session: Session,
    monkeypatch,
):
    company_id = _test_id()
    company_name = f"Development Search Acceptance V1 {uuid4().hex}"
    wrong_id = _test_id()
    db_session.add_all(
        [
            Company(id=company_id, name="Unrelated Company"),
            Company(id=wrong_id, name=company_name),
        ]
    )
    db_session.flush()
    monkeypatch.setattr(seed, "DEVELOPMENT_COMPANY_ID", company_id)
    monkeypatch.setattr(seed, "DEVELOPMENT_COMPANY_NAME", company_name)

    with pytest.raises(seed.DevelopmentSeedConflictError):
        seed.seed_development_search_acceptance_data(db_session)

    _assert_no_seed_descendants(db_session, (company_id, wrong_id))


def test_run_seed_is_atomic_on_failure(monkeypatch):
    company_id = -((uuid4().int % 100_000_000) + 1_500_000_000)
    company_name = f"Development Search Acceptance V1 {uuid4().hex}"
    connection = engine.connect()
    outer_transaction = connection.begin()
    test_session_local = sessionmaker(
        bind=connection,
        autoflush=False,
        join_transaction_mode="create_savepoint",
    )
    original_upsert_unit = seed._upsert_unit

    def fail_during_unit_seed(session, company, compound, spec):
        unit = original_upsert_unit(session, company, compound, spec)
        if spec.external_unit_id == "A103":
            raise RuntimeError("simulated seed failure")
        return unit

    monkeypatch.setattr(seed, "DEVELOPMENT_COMPANY_ID", company_id)
    monkeypatch.setattr(seed, "DEVELOPMENT_COMPANY_NAME", company_name)
    monkeypatch.setattr(seed, "SessionLocal", test_session_local)
    monkeypatch.setattr(seed, "_upsert_unit", fail_during_unit_seed)

    try:
        with pytest.raises(RuntimeError, match="simulated seed failure"):
            seed.run_seed()

        with Session(bind=connection) as verification_session:
            company_count = verification_session.scalar(
                select(func.count())
                .select_from(Company)
                .where(Company.id == company_id)
            )
            assert company_count == 0
    finally:
        if outer_transaction.is_active:
            outer_transaction.rollback()
        connection.close()


def test_seed_is_explicit_and_not_coupled_to_application_startup(
    monkeypatch,
    capsys,
):
    from app.main import app

    calls = 0

    def fake_run_seed() -> seed.SeedCounts:
        nonlocal calls
        calls += 1
        return EXPECTED_COUNTS

    monkeypatch.setattr(seed, "run_seed", fake_run_seed)

    with TestClient(app) as client:
        assert client.get("/").status_code == 200
    assert calls == 0

    seed.main()

    assert calls == 1
    assert (
        "Development Search Acceptance Seed V1 applied"
        in capsys.readouterr().out
    )


def _seed_company(session: Session) -> Company:
    company = session.scalar(
        select(Company).where(Company.name == seed.DEVELOPMENT_COMPANY_NAME)
    )
    assert company is not None
    return company


def _scopes_by_key(session: Session, company_id: int) -> dict[str, InventoryScope]:
    return {
        scope.scope_key: scope
        for scope in session.scalars(
            select(InventoryScope).where(InventoryScope.company_id == company_id)
        )
    }


def _compounds_by_key(session: Session, company_id: int) -> dict[str, Compound]:
    return {
        compound.compound_key: compound
        for compound in session.scalars(
            select(Compound).where(Compound.company_id == company_id)
        )
    }


def _units_by_external_id(session: Session, company_id: int) -> dict[str, Unit]:
    return {
        unit.external_unit_id: unit
        for unit in session.scalars(
            select(Unit).where(Unit.company_id == company_id)
        )
    }


def _plans_by_key(session: Session, company_id: int) -> dict[str, PaymentPlan]:
    return {
        plan.plan_key: plan
        for plan in session.scalars(
            select(PaymentPlan).where(PaymentPlan.company_id == company_id)
        )
    }


def _unit_states_by_external_id(
    session: Session,
    company_id: int,
) -> dict[str, UnitInventoryState]:
    statement = (
        select(Unit, UnitInventoryState)
        .join(UnitInventoryState, UnitInventoryState.unit_id == Unit.id)
        .where(Unit.company_id == company_id)
    )
    return {
        unit.external_unit_id: state
        for unit, state in session.execute(statement)
    }


def _plan_states_by_key(
    session: Session,
    company_id: int,
) -> dict[str, PaymentPlanInventoryState]:
    statement = (
        select(PaymentPlan, PaymentPlanInventoryState)
        .join(
            PaymentPlanInventoryState,
            PaymentPlanInventoryState.payment_plan_id == PaymentPlan.id,
        )
        .where(PaymentPlan.company_id == company_id)
    )
    return {
        plan.plan_key: state
        for plan, state in session.execute(statement)
    }


def _database_counts(session: Session, company_id: int) -> seed.SeedCounts:
    return seed.SeedCounts(
        companies=session.scalar(
            select(func.count())
            .select_from(Company)
            .where(Company.id == company_id)
        ),
        scopes=session.scalar(
            select(func.count())
            .select_from(InventoryScope)
            .where(InventoryScope.company_id == company_id)
        ),
        compounds=session.scalar(
            select(func.count())
            .select_from(Compound)
            .where(Compound.company_id == company_id)
        ),
        units=session.scalar(
            select(func.count())
            .select_from(Unit)
            .where(Unit.company_id == company_id)
        ),
        payment_plans=session.scalar(
            select(func.count())
            .select_from(PaymentPlan)
            .where(PaymentPlan.company_id == company_id)
        ),
        unit_inventory_states=session.scalar(
            select(func.count())
            .select_from(UnitInventoryState)
            .where(UnitInventoryState.company_id == company_id)
        ),
        payment_plan_inventory_states=session.scalar(
            select(func.count())
            .select_from(PaymentPlanInventoryState)
            .where(PaymentPlanInventoryState.company_id == company_id)
        ),
    )


def _assert_no_seed_descendants(
    session: Session,
    company_ids: tuple[int, ...],
) -> None:
    for model in (
        Compound,
        InventoryScope,
        Unit,
        PaymentPlan,
        UnitInventoryState,
        PaymentPlanInventoryState,
    ):
        count = session.scalar(
            select(func.count())
            .select_from(model)
            .where(model.company_id.in_(company_ids))
        )
        assert count == 0


def _unit_values(unit: Unit):
    return (
        unit.property_type,
        unit.bedrooms,
        unit.bathrooms,
        unit.area_sqm,
        unit.base_price,
        unit.status,
    )


def _plan_values(plan: PaymentPlan):
    return (
        plan.payment_method,
        plan.total_price,
        plan.down_payment,
        plan.installment_years,
    )


def _test_id() -> int:
    return -((uuid4().int % 1_000_000_000) + 1)
