from collections.abc import Generator
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.dependencies import get_search_company_id
from app.database import engine
from app.models import (
    Company,
    Compound,
    InventoryScope,
    Unit,
    UnitInventoryState,
)
from app.main import app
from app.search import scope_resolution
from app.search import service as search_service

ACCEPTANCE_COMPANY_ID = -2_000_000_001
ACCEPTANCE_LAST_SEEN_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)
ACCEPTANCE_SOURCE_LAST_UPDATED = date(2026, 1, 1)

SEARCH_A_REQUEST = {
    "location": "New Cairo",
    "property_type": "Villa",
    "bedrooms": 4,
    "bathrooms": 3,
    "min_area_sqm": 245,
    "max_area_sqm": 255,
    "min_price": 11500000,
    "max_price": 12200000,
    "requested_scope_keys": ["scope-nc-main"],
}


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


def test_search_a_path_runs_through_public_api(db_session, monkeypatch):
    _create_search_a_acceptance_rows(db_session)
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_search_company_id] = (
        lambda: ACCEPTANCE_COMPANY_ID
    )
    monkeypatch.setattr(
        scope_resolution,
        "SessionLocal",
        lambda: _ExistingSessionContext(db_session),
    )
    monkeypatch.setattr(
        search_service,
        "SessionLocal",
        lambda: _ExistingSessionContext(db_session),
    )

    try:
        response = TestClient(app).post(
            "/api/v1/search",
            json=SEARCH_A_REQUEST,
        )
    finally:
        app.dependency_overrides = original_overrides

    assert response.status_code == 200
    results = response.json()["results"]
    external_unit_ids = [
        result["unit"]["external_unit_id"]
        for result in results
    ]
    assert "A101" in external_unit_ids
    assert "A105" not in external_unit_ids
    assert "A106" not in external_unit_ids

    top_result = results[0]
    top_unit = top_result["unit"]
    assert top_result["ranking"]["rank"] == 1
    assert top_result["ranking"]["score"] == "1"
    assert top_unit["external_unit_id"] == "A101"
    assert top_unit["compound_name"] == "Palm East"
    assert top_unit["location"] == "New Cairo"
    assert top_unit["property_type"] == "Villa"
    assert top_unit["bedrooms"] == 4
    assert top_unit["bathrooms"] == 3
    assert Decimal(top_unit["area_sqm"]) == Decimal("250")
    assert Decimal(top_unit["base_price"]) == Decimal("12000000")
    assert top_unit["status"] == "Available"
    assert "scope-nc-main" in top_unit["matched_scope_keys"]


def _create_search_a_acceptance_rows(session: Session) -> None:
    assert Company.__table__.c.id.type.python_type is int
    assert -(2**31) <= ACCEPTANCE_COMPANY_ID <= 2**31 - 1

    company = Company(
        id=ACCEPTANCE_COMPANY_ID,
        name="Search A Acceptance Company",
    )
    session.add(company)
    session.flush()

    compound = Compound(
        id=-2_100_000_001,
        company_id=ACCEPTANCE_COMPANY_ID,
        compound_key="palm-east",
        name="Palm East",
        location="New Cairo",
    )
    session.add(compound)
    session.flush()

    scope = InventoryScope(
        id=-2_100_000_101,
        company_id=ACCEPTANCE_COMPANY_ID,
        compound_id=compound.id,
        scope_key="scope-nc-main",
        source_identifier="acceptance:scope-nc-main",
    )
    session.add(scope)
    session.flush()

    a101 = _add_unit(
        session,
        unit_id=-2_100_001_001,
        compound=compound,
        external_unit_id="A101",
        status="Available",
        is_current=True,
        area_sqm=Decimal("250"),
        base_price=Decimal("12000000"),
    )
    _add_unit(
        session,
        unit_id=-2_100_001_005,
        compound=compound,
        external_unit_id="A105",
        status="Hold",
        is_current=True,
        area_sqm=Decimal("255"),
        base_price=Decimal("10500000"),
    )
    _add_unit(
        session,
        unit_id=-2_100_001_006,
        compound=compound,
        external_unit_id="A106",
        status="Available",
        is_current=False,
        area_sqm=Decimal("255"),
        base_price=Decimal("10800000"),
    )

    assert a101.external_unit_id == "A101"
    session.flush()


def _add_unit(
    session: Session,
    *,
    unit_id: int,
    compound: Compound,
    external_unit_id: str,
    status: str,
    is_current: bool,
    area_sqm: Decimal,
    base_price: Decimal,
) -> Unit:
    unit = Unit(
        id=unit_id,
        company_id=ACCEPTANCE_COMPANY_ID,
        compound_id=compound.id,
        external_unit_id=external_unit_id,
        property_type="Villa",
        bedrooms=4,
        bathrooms=3,
        area_sqm=area_sqm,
        base_price=base_price,
        base_currency="EGP",
        status=status,
        delivery_date=None,
        source_last_updated=ACCEPTANCE_SOURCE_LAST_UPDATED,
    )
    session.add(unit)
    session.flush()

    session.add(
        UnitInventoryState(
            id=unit_id - 10_000,
            company_id=ACCEPTANCE_COMPANY_ID,
            unit_id=unit.id,
            inventory_scope_id=-2_100_000_101,
            is_present_in_latest_inventory=is_current,
            last_seen_at=ACCEPTANCE_LAST_SEEN_AT,
        )
    )
    session.flush()
    return unit


class _ExistingSessionContext:
    def __init__(self, session: Session):
        self.session = session

    def __enter__(self) -> Session:
        return self.session

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False
