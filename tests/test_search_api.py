import inspect
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.ai_engine.search_filters import SearchFilters
from app.api.dependencies import (
    get_search_context_resolver,
    get_trusted_company_id,
)
from app.api.v1 import search as search_route
from app.domain.enums import Currency, PaymentMethod, UnitStatus
from app.main import app
from app.search.contracts import PaymentPlanCandidate, SearchExecutionContext, UnitCandidate
from app.search.exceptions import (
    InvalidSearchScopeError,
    SearchScopeResolutionFailedError,
)
from app.search.scope_resolution import resolve_search_execution_context


@dataclass(frozen=True)
class FakeRanking:
    rank: int
    score: Decimal
    best_plan_key: str | None
    evaluation: object = "hidden-evaluation"
    components: tuple[object, ...] = ("hidden-component",)


@dataclass(frozen=True)
class FakeFact:
    dimension: object
    kind: object
    message_key: str
    baseline_value: object | None
    candidate_value: object | None
    score: Decimal | None


@dataclass(frozen=True)
class FakeExplanation:
    strengths: tuple[FakeFact, ...]
    tradeoffs: tuple[FakeFact, ...]


@dataclass(frozen=True)
class FakeItem:
    candidate: UnitCandidate
    ranking: FakeRanking
    explanation: FakeExplanation


@dataclass(frozen=True)
class FakeSearchResult:
    strategy: object
    plan: object
    results: tuple[FakeItem, ...]


@pytest.fixture
def client(monkeypatch):
    original_overrides = app.dependency_overrides.copy()
    calls = {"resolver": [], "search": []}
    context = SearchExecutionContext(
        company_id=42,
        resolved_scope_keys=("scope:resolved",),
    )

    def resolver(
        *,
        company_id,
        requested_scope_keys,
    ):
        calls["resolver"].append((company_id, requested_scope_keys))
        return context

    def fake_search_units(*, filters, context):
        calls["search"].append((filters, context))
        return _search_result()

    app.dependency_overrides[get_trusted_company_id] = lambda: 42
    app.dependency_overrides[get_search_context_resolver] = lambda: resolver
    monkeypatch.setattr(search_route.search_application, "search_units", fake_search_units)

    try:
        yield TestClient(app), calls, context
    finally:
        app.dependency_overrides = original_overrides


def test_valid_structured_search_request(client):
    test_client, _, _ = client

    response = test_client.post(
        "/api/v1/search",
        json={
            "location": "New Cairo",
            "property_type": "Apartment",
            "bedrooms": 3,
            "bathrooms": 2,
            "min_price": 1000000,
            "max_price": 5000000,
            "min_area_sqm": 120,
            "max_area_sqm": 180,
            "payment_method": "Installment",
            "max_down_payment": 750000,
            "min_installment_years": 6,
            "requested_scope_keys": ["scope:b", "scope:a"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["unit"]["unit_id"] == 100
    assert body["results"][0]["ranking"] == {
        "rank": 1,
        "score": "0.95",
        "best_plan_key": "installment-6y",
    }


def test_request_dto_maps_explicitly_to_search_filters(client):
    test_client, calls, _ = client

    response = test_client.post(
        "/api/v1/search",
        json={
            "location": "Sheikh Zayed",
            "property_type": "Villa",
            "bedrooms": 4,
            "bathrooms": 3,
            "min_price": 2500000,
            "max_price": 9000000,
            "min_area_sqm": 200,
            "max_area_sqm": 350,
            "payment_method": "Cash",
            "max_down_payment": 1000000,
            "min_installment_years": 5,
        },
    )

    assert response.status_code == 200
    filters, _ = calls["search"][0]
    assert isinstance(filters, SearchFilters)
    assert filters == SearchFilters(
        location="Sheikh Zayed",
        property_type="Villa",
        bedrooms=4,
        bathrooms=3,
        min_price=2500000,
        max_price=9000000,
        min_area_sqm=200,
        max_area_sqm=350,
        payment_method="Cash",
        max_down_payment=1000000,
        min_installment_years=5,
    )


def test_company_id_cannot_be_supplied_in_request_body(client):
    test_client, _, _ = client

    response = test_client.post("/api/v1/search", json={"company_id": 999})

    assert response.status_code == 422


def test_requested_scope_keys_omitted_resolves_as_none(client):
    test_client, calls, _ = client

    response = test_client.post("/api/v1/search", json={})

    assert response.status_code == 200
    assert calls["resolver"][0] == (42, None)


def test_requested_scope_keys_null_resolves_as_none(client):
    test_client, calls, _ = client

    response = test_client.post(
        "/api/v1/search",
        json={"requested_scope_keys": None},
    )

    assert response.status_code == 200
    assert calls["resolver"][0] == (42, None)


def test_requested_scope_keys_empty_list_is_invalid(client):
    test_client, _, _ = client

    response = test_client.post(
        "/api/v1/search",
        json={"requested_scope_keys": []},
    )

    assert response.status_code == 422


def test_duplicate_requested_scope_keys_are_invalid(client):
    test_client, _, _ = client

    response = test_client.post(
        "/api/v1/search",
        json={"requested_scope_keys": ["scope:a", "scope:a"]},
    )

    assert response.status_code == 422


def test_requested_scope_keys_preserves_request_order(client):
    test_client, calls, _ = client

    response = test_client.post(
        "/api/v1/search",
        json={"requested_scope_keys": ["scope:c", "scope:a", "scope:b"]},
    )

    assert response.status_code == 200
    assert calls["resolver"][0] == (
        42,
        ("scope:c", "scope:a", "scope:b"),
    )


def test_invalid_search_scope_error_maps_to_generic_404(monkeypatch):
    response = _request_with_failing_resolver(
        monkeypatch,
        InvalidSearchScopeError("private tenant scope details"),
    )

    assert response.status_code == 404
    assert response.json() == _invalid_scope_response()
    assert "private tenant scope details" not in response.text


def test_unknown_and_cross_tenant_scope_errors_use_same_public_404(monkeypatch):
    unknown_response = _request_with_failing_resolver(
        monkeypatch,
        InvalidSearchScopeError("unknown scope"),
    )
    cross_tenant_response = _request_with_failing_resolver(
        monkeypatch,
        InvalidSearchScopeError("cross tenant scope"),
    )

    assert unknown_response.status_code == 404
    assert cross_tenant_response.status_code == 404
    assert unknown_response.json() == cross_tenant_response.json()
    assert unknown_response.json() == _invalid_scope_response()
    assert "unknown scope" not in unknown_response.text
    assert "cross tenant scope" not in cross_tenant_response.text


def test_scope_resolution_failure_maps_to_safe_generic_500(monkeypatch):
    response = _request_with_failing_resolver(
        monkeypatch,
        SearchScopeResolutionFailedError("database host secret"),
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "SEARCH_CONTEXT_RESOLUTION_FAILED",
            "message": "Search context could not be resolved",
        }
    }
    assert "database host secret" not in response.text


def test_unexpected_resolver_exception_maps_to_safe_generic_500(monkeypatch):
    response = _request_with_failing_resolver(
        monkeypatch,
        RuntimeError("postgresql://secret SELECT hidden_scope_table"),
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "SEARCH_CONTEXT_RESOLUTION_FAILED",
            "message": "Search context could not be resolved",
        }
    }
    assert "postgresql://secret" not in response.text
    assert "hidden_scope_table" not in response.text


def test_resolver_dependency_is_wired_to_real_backend_resolver():
    assert get_search_context_resolver() is resolve_search_execution_context


def test_api_never_constructs_search_execution_context_directly():
    source = inspect.getsource(search_route.search_units)

    assert "SearchExecutionContext(" not in source


def test_resolver_dependency_can_be_overridden(client):
    test_client, calls, context = client

    response = test_client.post("/api/v1/search", json={})

    assert response.status_code == 200
    assert calls["search"][0][1] is context


def test_trusted_company_dependency_can_be_overridden(client):
    test_client, calls, _ = client

    response = test_client.post("/api/v1/search", json={})

    assert response.status_code == 200
    assert calls["resolver"][0][0] == 42


def test_route_calls_search_units_exactly_once(client):
    test_client, calls, _ = client

    response = test_client.post("/api/v1/search", json={})

    assert response.status_code == 200
    assert len(calls["search"]) == 1


def test_strategy_and_plan_are_not_exposed(client):
    test_client, _, _ = client

    response = test_client.post("/api/v1/search", json={})

    assert response.status_code == 200
    body = response.json()
    assert "strategy" not in body
    assert "plan" not in body
    assert "hidden-strategy" not in response.text
    assert "hidden-plan" not in response.text


def test_ranked_candidate_evaluation_is_not_exposed(client):
    test_client, _, _ = client

    response = test_client.post("/api/v1/search", json={})

    assert response.status_code == 200
    assert "evaluation" not in response.json()["results"][0]["ranking"]
    assert "hidden-evaluation" not in response.text


def test_ranking_components_are_not_exposed(client):
    test_client, _, _ = client

    response = test_client.post("/api/v1/search", json={})

    assert response.status_code == 200
    assert "components" not in response.json()["results"][0]["ranking"]
    assert "hidden-component" not in response.text


def test_decimal_date_and_enum_fields_serialize_correctly(client):
    test_client, _, _ = client

    response = test_client.post("/api/v1/search", json={})

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["unit"]["area_sqm"] == "145.50"
    assert result["unit"]["base_price"] == "4500000.50"
    assert result["unit"]["base_currency"] == "EGP"
    assert result["unit"]["status"] == "Available"
    assert result["unit"]["delivery_date"] == "2028-06-01"
    assert result["unit"]["source_last_updated"] == "2026-09-04"
    assert result["unit"]["payment_plans"][0]["payment_method"] == "Installment"
    assert result["unit"]["payment_plans"][0]["total_price"] == "4700000.25"


def test_internal_failure_produces_safe_500_without_leaking_exception_text(client, monkeypatch):
    test_client, _, _ = client

    def fail_search(**_):
        raise RuntimeError("postgresql://secret SELECT sensitive_backend_table")

    monkeypatch.setattr(search_route.search_application, "search_units", fail_search)

    response = test_client.post("/api/v1/search", json={})

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "SEARCH_FAILED",
            "message": "Search could not be completed",
        }
    }
    assert "postgresql://secret" not in response.text
    assert "sensitive_backend_table" not in response.text


def test_request_body_does_not_accept_resolved_scope_keys(client):
    test_client, _, _ = client

    response = test_client.post(
        "/api/v1/search",
        json={"resolved_scope_keys": ["scope:one"]},
    )

    assert response.status_code == 422


def test_no_sqlalchemy_or_session_use_exists_in_api_route():
    source = inspect.getsource(search_route)

    assert "sqlalchemy" not in source.lower()
    assert "session" not in source.lower()
    assert "get_db" not in source
    assert "search.repository" not in source
    assert "search_service" not in source


def _request_with_failing_resolver(monkeypatch, error: Exception):
    original_overrides = app.dependency_overrides.copy()

    def resolver(*, company_id, requested_scope_keys):
        raise error

    app.dependency_overrides[get_trusted_company_id] = lambda: 42
    app.dependency_overrides[get_search_context_resolver] = lambda: resolver
    monkeypatch.setattr(
        search_route.search_application,
        "search_units",
        lambda **_: _search_result(),
    )

    try:
        return TestClient(app).post(
            "/api/v1/search",
            json={"requested_scope_keys": ["scope:private"]},
        )
    finally:
        app.dependency_overrides = original_overrides


def _invalid_scope_response():
    return {
        "error": {
            "code": "SEARCH_SCOPE_NOT_FOUND",
            "message": "Search scope selection was not found",
        }
    }


def _search_result() -> FakeSearchResult:
    payment_plan = PaymentPlanCandidate(
        plan_key="installment-6y",
        source_plan_id="plan-source-1",
        plan_name="Six year installment",
        payment_method=PaymentMethod.INSTALLMENT,
        down_payment=Decimal("700000.00"),
        installment_years=Decimal("6"),
        total_price=Decimal("4700000.25"),
        currency=Currency.EGP,
    )
    candidate = UnitCandidate(
        unit_id=100,
        external_unit_id="unit-100",
        compound_id=55,
        compound_name="Example Compound",
        location="New Cairo",
        property_type="Apartment",
        bedrooms=3,
        bathrooms=2,
        area_sqm=Decimal("145.50"),
        base_price=Decimal("4500000.50"),
        base_currency=Currency.EGP,
        status=UnitStatus.AVAILABLE,
        delivery_date=date(2028, 6, 1),
        source_last_updated=date(2026, 9, 4),
        matched_scope_keys=("scope:resolved",),
        payment_plans=(payment_plan,),
    )
    return FakeSearchResult(
        strategy="hidden-strategy",
        plan="hidden-plan",
        results=(
            FakeItem(
                candidate=candidate,
                ranking=FakeRanking(
                    rank=1,
                    score=Decimal("0.95"),
                    best_plan_key="installment-6y",
                ),
                explanation=FakeExplanation(
                    strengths=(
                        FakeFact(
                            dimension="location",
                            kind="strength",
                            message_key="matches_location",
                            baseline_value="New Cairo",
                            candidate_value="New Cairo",
                            score=Decimal("1"),
                        ),
                    ),
                    tradeoffs=(
                        FakeFact(
                            dimension="price",
                            kind="tradeoff",
                            message_key="price_relaxed",
                            baseline_value=(None, Decimal("4500000")),
                            candidate_value=Decimal("4700000.25"),
                            score=Decimal("0.8"),
                        ),
                    ),
                ),
            ),
        ),
    )
