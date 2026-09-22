from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.ai_engine.search_application import NaturalLanguageSearchResult
from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_intent import (
    ParsedSearchIntent,
    SearchAmbiguity,
    SearchFieldDirective,
)
from app.ai_engine.search_strategy import ConstraintStrength, SearchField
from app.api.dependencies import (
    get_search_company_id,
    get_search_context_resolver,
)
from app.api.security import get_current_user
from app.api.v1 import search as search_route
from app.auth.human.types import AuthenticatedUserContext
from app.domain.enums import Currency, MembershipRole, PaymentMethod, UnitStatus
from app.main import app
from app.search.contracts import (
    PaymentPlanCandidate,
    SearchExecutionContext,
    UnitCandidate,
)
from app.search.exceptions import (
    InvalidSearchScopeError,
    SearchScopeResolutionFailedError,
)


@dataclass(frozen=True)
class FakeRanking:
    rank: int
    score: Decimal
    best_plan_key: str | None
    evaluation: object = "internal-evaluation"
    components: tuple[object, ...] = ("internal-component",)


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
    strategy: object = "internal-strategy"
    plan: object = "internal-plan"
    results: tuple[FakeItem, ...] = ()


@pytest.fixture
def nl_client(monkeypatch):
    original_overrides = app.dependency_overrides.copy()
    original_openapi_schema = app.openapi_schema
    calls = {"resolver": [], "nl_search": [], "structured_search": []}
    context = SearchExecutionContext(
        company_id=42,
        resolved_scope_keys=("scope:resolved",),
    )

    def current_user():
        return AuthenticatedUserContext(
            user_id=7,
            membership_id=8,
            company_id=42,
            role=MembershipRole.SALESPERSON,
            session_id=9,
        )

    def resolver(*, company_id, requested_scope_keys):
        calls["resolver"].append((company_id, requested_scope_keys))
        return context

    def search_units_from_query(*, text, context):
        calls["nl_search"].append((text, context))
        return NaturalLanguageSearchResult(
            intent=_intent(filters=SearchFilters(location="New Cairo")),
            search=FakeSearchResult(results=(_search_item(),)),
        )

    def structured_search_units(*, filters, context):
        calls["structured_search"].append((filters, context))
        return FakeSearchResult()

    app.dependency_overrides[get_current_user] = current_user
    app.dependency_overrides[get_search_context_resolver] = lambda: resolver
    monkeypatch.setattr(
        search_route.search_application,
        "search_units_from_query",
        search_units_from_query,
    )
    monkeypatch.setattr(
        search_route.search_application,
        "search_units",
        structured_search_units,
    )

    try:
        yield TestClient(app), calls, context
    finally:
        app.dependency_overrides = original_overrides
        app.openapi_schema = original_openapi_schema


def test_authenticated_human_natural_language_search_executes(nl_client):
    client, calls, context = nl_client

    response = client.post(
        "/api/v1/search/natural-language",
        json={"query": "3 bedroom apartment in New Cairo under 6 million cash"},
    )

    assert response.status_code == 200
    assert calls["resolver"] == [(42, None)]
    assert calls["nl_search"] == [
        (
            "3 bedroom apartment in New Cairo under 6 million cash",
            context,
        )
    ]
    assert response.json()["executed"] is True


def test_unauthenticated_natural_language_request_is_rejected():
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides = {}
    try:
        response = TestClient(app).post(
            "/api/v1/search/natural-language",
            json={"query": "apartment in New Cairo"},
        )
    finally:
        app.dependency_overrides = original_overrides

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_SESSION"


def test_api_key_identity_cannot_use_human_only_endpoint(monkeypatch):
    original_overrides = app.dependency_overrides.copy()
    calls = {"resolver": [], "nl_search": []}
    app.dependency_overrides = {}
    app.dependency_overrides[get_search_context_resolver] = lambda: (
        lambda *, company_id, requested_scope_keys: calls["resolver"].append(
            (company_id, requested_scope_keys)
        )
    )
    monkeypatch.setattr(
        search_route.search_application,
        "search_units_from_query",
        lambda **kwargs: calls["nl_search"].append(kwargs),
    )
    try:
        response = TestClient(app).post(
            "/api/v1/search/natural-language",
            json={"query": "apartment in New Cairo"},
            headers={"X-API-Key": "api-key"},
        )
    finally:
        app.dependency_overrides = original_overrides

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_SESSION"
    assert calls == {"resolver": [], "nl_search": []}


def test_nl_text_cannot_override_trusted_company_id(nl_client):
    client, calls, _ = nl_client

    response = client.post(
        "/api/v1/search/natural-language",
        json={"query": "company 7 villa in New Cairo"},
    )

    assert response.status_code == 200
    assert calls["resolver"] == [(42, None)]
    assert calls["nl_search"][0][1].company_id == 42


def test_scope_resolver_runs_before_natural_language_application(monkeypatch):
    original_overrides = app.dependency_overrides.copy()
    events = []

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUserContext(
        user_id=1,
        membership_id=2,
        company_id=313,
        role=MembershipRole.COMPANY_ADMIN,
        session_id=3,
    )

    def resolver(*, company_id, requested_scope_keys):
        events.append(("resolver", company_id, requested_scope_keys))
        return SearchExecutionContext(
            company_id=company_id,
            resolved_scope_keys=("scope:313",),
        )

    def search_units_from_query(*, text, context):
        events.append(("ai", context.company_id, context.resolved_scope_keys))
        return NaturalLanguageSearchResult(
            intent=_intent(),
            search=FakeSearchResult(),
        )

    app.dependency_overrides[get_search_context_resolver] = lambda: resolver
    monkeypatch.setattr(
        search_route.search_application,
        "search_units_from_query",
        search_units_from_query,
    )

    try:
        response = TestClient(app).post(
            "/api/v1/search/natural-language",
            json={"query": "apartment"},
        )
    finally:
        app.dependency_overrides = original_overrides

    assert response.status_code == 200
    assert events == [
        ("resolver", 313, None),
        ("ai", 313, ("scope:313",)),
    ]


@pytest.mark.parametrize("field", ["company_id", "price_basis"])
def test_natural_language_request_rejects_authority_and_execution_metadata(
    nl_client,
    field,
):
    client, calls, _ = nl_client

    response = client.post(
        "/api/v1/search/natural-language",
        json={"query": "apartment", field: "hidden"},
    )

    assert response.status_code == 422
    assert calls["resolver"] == []
    assert calls["nl_search"] == []


def test_get_search_company_id_is_not_used_by_natural_language_endpoint(
    nl_client,
):
    client, _, _ = nl_client
    app.dependency_overrides[get_search_company_id] = (
        lambda: pytest.fail("NL endpoint must not use get_search_company_id")
    )

    response = client.post(
        "/api/v1/search/natural-language",
        json={"query": "apartment"},
    )

    assert response.status_code == 200


@pytest.mark.parametrize(
    ("query", "intent_factory"),
    [
        ("", lambda: _intent()),
        ("   ", lambda: _intent()),
        (
            "cheap apartment",
            lambda: _intent(
                ambiguities=[
                    SearchAmbiguity(
                        field=SearchField.MAX_PRICE,
                        reason="budget is ambiguous",
                    )
                ],
            ),
        ),
        (
            "apartment under 6 bananas",
            lambda: _intent(
                ambiguities=[
                    SearchAmbiguity(
                        field=SearchField.MAX_PRICE,
                        reason="Invalid or unsupported money value",
                    )
                ],
            ),
        ),
        (
            "sea view",
            lambda: _intent(unsupported_requirements=["sea view"]),
        ),
    ],
)
def test_non_executed_natural_language_results_preserve_null_results(
    nl_client,
    monkeypatch,
    query,
    intent_factory,
):
    client, _, _ = nl_client
    monkeypatch.setattr(
        search_route.search_application,
        "search_units_from_query",
        lambda *, text, context: NaturalLanguageSearchResult(
            intent=intent_factory(),
            search=None,
        ),
    )

    response = client.post("/api/v1/search/natural-language", json={"query": query})

    assert response.status_code == 200
    body = response.json()
    assert body["executed"] is False
    assert body["results"] is None


def test_supported_query_with_unsupported_requirement_executes_with_diagnostics(
    nl_client,
    monkeypatch,
):
    client, _, _ = nl_client
    monkeypatch.setattr(
        search_route.search_application,
        "search_units_from_query",
        lambda *, text, context: NaturalLanguageSearchResult(
            intent=_intent(
                filters=SearchFilters(location="New Cairo"),
                unsupported_requirements=["sea view"],
            ),
            search=FakeSearchResult(results=()),
        ),
    )

    response = client.post(
        "/api/v1/search/natural-language",
        json={"query": "New Cairo with sea view"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["executed"] is True
    assert body["results"] == []
    assert body["intent"]["unsupported_requirements"] == ["sea view"]


def test_successful_match_projects_existing_search_result_shape(nl_client):
    client, _, _ = nl_client

    response = client.post(
        "/api/v1/search/natural-language",
        json={"query": "apartment in New Cairo cash"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["executed"] is True
    assert body["intent"]["filters"]["location"] == "New Cairo"
    assert body["results"][0]["unit"]["matched_scope_keys"] == ["scope:resolved"]
    assert body["results"][0]["unit"]["payment_plans"][0]["payment_method"] == "Cash"
    assert body["results"][0]["ranking"] == {
        "rank": 1,
        "score": "0.95",
        "best_plan_key": "cash",
    }
    assert body["results"][0]["explanation"]["strengths"][0]["dimension"] == "location"
    assert "strategy" not in body
    assert "plan" not in body
    assert "company_id" not in str(body)
    assert "PriceBasis" not in str(body)


@pytest.mark.parametrize(
    "query",
    [
        "3 bedroom apartment in New Cairo under 6 million cash",
        "شقة في التجمع تحت ٦ مليون كاش",
        "sha2a fi tagamo3 under 6 million cash",
    ],
)
def test_multilingual_queries_reach_hazem_entry_point(nl_client, query):
    client, calls, _ = nl_client

    response = client.post("/api/v1/search/natural-language", json={"query": query})

    assert response.status_code == 200
    assert calls["nl_search"][-1][0] == query


def test_query_length_boundary(nl_client):
    client, calls, _ = nl_client

    accepted = client.post(
        "/api/v1/search/natural-language",
        json={"query": "a" * 500},
    )
    rejected = client.post(
        "/api/v1/search/natural-language",
        json={"query": "a" * 501},
    )

    assert accepted.status_code == 200
    assert rejected.status_code == 422
    assert calls["nl_search"] == [("a" * 500, calls["nl_search"][0][1])]


def test_query_must_be_strict_string(nl_client):
    client, calls, _ = nl_client

    response = client.post(
        "/api/v1/search/natural-language",
        json={"query": 123},
    )

    assert response.status_code == 422
    assert calls["resolver"] == []
    assert calls["nl_search"] == []


def test_intent_diagnostics_are_projected_explicitly(nl_client, monkeypatch):
    client, _, _ = nl_client
    monkeypatch.setattr(
        search_route.search_application,
        "search_units_from_query",
        lambda *, text, context: NaturalLanguageSearchResult(
            intent=_intent(
                filters=SearchFilters(
                    location="New Cairo",
                    property_type="Apartment",
                    bedrooms=3,
                    bathrooms=2,
                    min_price=Decimal("1000000"),
                    max_price=Decimal("6000000"),
                    min_area_sqm=Decimal("120"),
                    max_area_sqm=Decimal("180"),
                    payment_method="Cash",
                    max_down_payment=Decimal("500000"),
                    min_installment_years=Decimal("5"),
                ),
                directives=[
                    SearchFieldDirective(
                        field=SearchField.LOCATION,
                        strength=ConstraintStrength.HARD,
                    ),
                    SearchFieldDirective(
                        field=SearchField.MAX_PRICE,
                        preferred=True,
                        allow_alternative=True,
                    ),
                ],
                ambiguities=[
                    SearchAmbiguity(
                        field=SearchField.PROPERTY_TYPE,
                        reason="specific alternative is not representable",
                    )
                ],
                unsupported_requirements=["sea view"],
            ),
            search=None,
        ),
    )

    response = client.post(
        "/api/v1/search/natural-language",
        json={"query": "apartment in New Cairo with sea view"},
    )

    assert response.status_code == 200
    intent = response.json()["intent"]
    assert intent["filters"] == {
        "location": "New Cairo",
        "property_type": "Apartment",
        "bedrooms": 3,
        "bathrooms": 2,
        "min_price": "1000000",
        "max_price": "6000000",
        "min_area_sqm": "120",
        "max_area_sqm": "180",
        "payment_method": "Cash",
        "max_down_payment": "500000",
        "min_installment_years": "5",
    }
    assert intent["directives"] == [
        {
            "field": "location",
            "strength": "hard",
            "preferred": False,
            "allow_alternative": False,
        },
        {
            "field": "max_price",
            "strength": None,
            "preferred": True,
            "allow_alternative": True,
        },
    ]
    assert intent["ambiguities"] == [
        {
            "field": "property_type",
            "reason": "specific alternative is not representable",
        }
    ]
    assert intent["unsupported_requirements"] == ["sea view"]


@pytest.mark.parametrize(
    ("exception", "status_code", "code"),
    [
        (
            InvalidSearchScopeError("internal scope detail"),
            404,
            "SEARCH_SCOPE_NOT_FOUND",
        ),
        (
            SearchScopeResolutionFailedError("internal scope detail"),
            500,
            "SEARCH_CONTEXT_RESOLUTION_FAILED",
        ),
        (
            RuntimeError("internal resolver detail"),
            500,
            "SEARCH_CONTEXT_RESOLUTION_FAILED",
        ),
    ],
)
def test_scope_resolution_errors_follow_existing_mapping(
    nl_client,
    exception,
    status_code,
    code,
):
    client, calls, _ = nl_client

    def resolver(*, company_id, requested_scope_keys):
        raise exception

    app.dependency_overrides[get_search_context_resolver] = lambda: resolver

    response = client.post(
        "/api/v1/search/natural-language",
        json={"query": "apartment"},
    )

    assert response.status_code == status_code
    assert response.json()["error"]["code"] == code
    assert "internal" not in response.text
    assert calls["nl_search"] == []


def test_unexpected_ai_exception_maps_to_search_failed(nl_client, monkeypatch):
    client, _, _ = nl_client

    def failing_search(*, text, context):
        raise RuntimeError("internal AI detail")

    monkeypatch.setattr(
        search_route.search_application,
        "search_units_from_query",
        failing_search,
    )

    response = client.post(
        "/api/v1/search/natural-language",
        json={"query": "apartment"},
    )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "SEARCH_FAILED"
    assert "internal AI detail" not in response.text
    assert "executed" not in response.text


def test_openapi_documents_only_public_natural_language_request_contract(
    nl_client,
):
    client, _, _ = nl_client

    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/api/v1/search/natural-language"]["post"]
    request_ref = operation["requestBody"]["content"]["application/json"]["schema"][
        "$ref"
    ]
    component_name = request_ref.rsplit("/", 1)[-1]
    request_schema = schema["components"]["schemas"][component_name]

    assert operation["security"] == [{"SessionCookieAuth": []}]
    assert set(request_schema["properties"]) == {"query"}
    assert request_schema["properties"]["query"]["maxLength"] == 500
    assert "company_id" not in str(operation)
    assert "price_basis" not in str(operation)


def test_existing_structured_search_route_remains_unchanged(nl_client):
    client, calls, _ = nl_client
    app.dependency_overrides[get_search_company_id] = lambda: 42

    response = client.post(
        "/api/v1/search",
        json={"requested_scope_keys": ["scope:requested"]},
    )

    assert response.status_code == 200
    assert response.json() == {"results": []}
    assert calls["resolver"] == [(42, ("scope:requested",))]
    assert calls["structured_search"]
    assert calls["nl_search"] == []


def _intent(
    *,
    filters: SearchFilters | None = None,
    directives: list[SearchFieldDirective] | None = None,
    ambiguities: list[SearchAmbiguity] | None = None,
    unsupported_requirements: list[str] | None = None,
) -> ParsedSearchIntent:
    return ParsedSearchIntent(
        filters=filters or SearchFilters(),
        directives=directives or [],
        ambiguities=ambiguities or [],
        unsupported_requirements=unsupported_requirements or [],
    )


def _search_item() -> FakeItem:
    return FakeItem(
        candidate=UnitCandidate(
            unit_id=100,
            external_unit_id="unit-100",
            compound_id=200,
            compound_name="Compound",
            location="New Cairo",
            property_type="Apartment",
            bedrooms=3,
            bathrooms=2,
            area_sqm=Decimal("150"),
            base_price=Decimal("5000000"),
            base_currency=Currency.EGP,
            status=UnitStatus.AVAILABLE,
            delivery_date=date(2027, 1, 1),
            source_last_updated=date(2026, 1, 1),
            matched_scope_keys=("scope:resolved",),
            payment_plans=(
                PaymentPlanCandidate(
                    plan_key="cash",
                    source_plan_id="plan-1",
                    plan_name="Cash",
                    payment_method=PaymentMethod.CASH,
                    down_payment=None,
                    installment_years=None,
                    total_price=Decimal("5000000"),
                    currency=Currency.EGP,
                ),
            ),
        ),
        ranking=FakeRanking(
            rank=1,
            score=Decimal("0.95"),
            best_plan_key="cash",
        ),
        explanation=FakeExplanation(
            strengths=(
                FakeFact(
                    dimension=SearchField.LOCATION,
                    kind="match",
                    message_key="location_match",
                    baseline_value="New Cairo",
                    candidate_value="New Cairo",
                    score=Decimal("1"),
                ),
            ),
            tradeoffs=(),
        ),
    )
