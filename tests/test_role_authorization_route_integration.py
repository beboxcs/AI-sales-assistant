from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from app.ai_engine.inventory_application import InventoryIngestionResult
from app.api import security as api_security
from app.api.dependencies import (
    get_inventory_ingest_max_upload_bytes,
    get_inventory_ingestion_service,
    get_search_context_resolver,
)
from app.api.errors import APIError
from app.api.v1 import search as search_route
from app.auth.contracts import TrustedTenantContext
from app.auth.human.exceptions import InvalidSessionError
from app.auth.human.types import AuthenticatedUserContext
from app.domain.enums import MembershipRole
from app.main import app
from app.search.contracts import SearchExecutionContext


RAW_API_KEY_A = "rea_v1_" + ("A" * 22) + "." + ("a" * 43)
SESSION_TOKEN = "opaque-session-token"


@dataclass(frozen=True)
class EmptySearchResult:
    results: tuple = ()


@dataclass
class FakeApiKeyTenantResolver:
    company_id: int = 100
    error: Exception | None = None
    calls: list[str] | None = None

    def __post_init__(self):
        self.calls = []

    def __call__(self, *, raw_api_key: str) -> TrustedTenantContext:
        self.calls.append(raw_api_key)
        if self.error is not None:
            raise self.error
        return TrustedTenantContext(company_id=self.company_id, credential_id=8)


@dataclass
class FakeHumanSessionResolver:
    context: AuthenticatedUserContext
    error: Exception | None = None
    calls: list[str] | None = None

    def __post_init__(self):
        self.calls = []

    def __call__(self, *, token: str) -> AuthenticatedUserContext:
        self.calls.append(token)
        if self.error is not None:
            raise self.error
        return self.context


@pytest.fixture
def auth_route_client(monkeypatch):
    original_overrides = app.dependency_overrides.copy()
    original_openapi_schema = app.openapi_schema
    clients = []

    def factory(
        *,
        api_resolver: FakeApiKeyTenantResolver | None = None,
        session_resolver: FakeHumanSessionResolver | None = None,
    ):
        selected_api_resolver = api_resolver or FakeApiKeyTenantResolver()
        selected_session_resolver = session_resolver or _session_resolver()
        calls = {"resolver": [], "search": [], "ingest": []}

        def search_context_resolver(*, company_id, requested_scope_keys):
            calls["resolver"].append((company_id, requested_scope_keys))
            return SearchExecutionContext(
                company_id=company_id,
                resolved_scope_keys=("scope:resolved",),
            )

        def search_units(*, filters, context):
            calls["search"].append((filters, context))
            return EmptySearchResult()

        def ingest(*, company_id, scope_key, source):
            calls["ingest"].append(
                {
                    "company_id": company_id,
                    "scope_key": scope_key,
                    "source": source,
                }
            )
            return InventoryIngestionResult(
                scope_key=scope_key,
                source_filename=source.filename,
                total_rows=1,
                accepted_units=1,
                rejected_rows=0,
                added_count=1,
                updated_count=0,
                missing_count=0,
                persisted=True,
            )

        app.dependency_overrides[
            api_security.get_api_key_tenant_resolver
        ] = lambda: selected_api_resolver
        app.dependency_overrides[
            api_security.get_human_session_resolver
        ] = lambda: selected_session_resolver
        app.dependency_overrides[get_search_context_resolver] = (
            lambda: search_context_resolver
        )
        app.dependency_overrides[get_inventory_ingestion_service] = lambda: ingest
        app.dependency_overrides[get_inventory_ingest_max_upload_bytes] = lambda: 1024
        monkeypatch.setattr(
            search_route.search_application,
            "search_units",
            search_units,
        )
        app.openapi_schema = None

        client = TestClient(app)
        clients.append(client)
        return client, selected_api_resolver, selected_session_resolver, calls

    try:
        yield factory
    finally:
        for client in clients:
            client.close()
        app.dependency_overrides = original_overrides
        app.openapi_schema = original_openapi_schema


def test_search_allows_api_key_company_id(auth_route_client):
    client, api_resolver, session_resolver, calls = auth_route_client(
        api_resolver=FakeApiKeyTenantResolver(company_id=111)
    )

    response = client.post(
        "/api/v1/search",
        json={},
        headers={"X-API-Key": RAW_API_KEY_A},
    )

    assert response.status_code == 200
    assert calls["resolver"] == [(111, None)]
    assert calls["search"][0][1].company_id == 111
    assert api_resolver.calls == [RAW_API_KEY_A]
    assert session_resolver.calls == []


@pytest.mark.parametrize(
    "role",
    [MembershipRole.SALESPERSON, MembershipRole.COMPANY_ADMIN],
)
def test_search_allows_human_roles(auth_route_client, role):
    client, api_resolver, session_resolver, calls = auth_route_client(
        session_resolver=_session_resolver(company_id=222, role=role)
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post("/api/v1/search", json={})

    assert response.status_code == 200
    assert calls["resolver"] == [(222, None)]
    assert calls["search"][0][1].company_id == 222
    assert api_resolver.calls == []
    assert session_resolver.calls == [SESSION_TOKEN]


@pytest.mark.parametrize(
    ("headers", "set_cookie", "expected"),
    [
        ({"X-API-Key": RAW_API_KEY_A}, True, "AMBIGUOUS_CREDENTIALS"),
        ({}, False, "AUTHENTICATION_REQUIRED"),
    ],
)
def test_search_auth_failures_prevent_search_app(
    auth_route_client,
    headers,
    set_cookie,
    expected,
):
    client, api_resolver, session_resolver, calls = auth_route_client()
    if set_cookie:
        client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post("/api/v1/search", json={}, headers=headers)

    assert response.status_code in {400, 401}
    assert response.json()["error"]["code"] == expected
    assert calls["resolver"] == []
    assert calls["search"] == []
    if expected == "AMBIGUOUS_CREDENTIALS":
        assert api_resolver.calls == []
        assert session_resolver.calls == []


@pytest.mark.parametrize(
    ("api_error", "session_error", "headers", "set_cookie", "expected"),
    [
        (
            APIError(401, "INVALID_API_CREDENTIAL", "Invalid or unavailable API credential"),
            None,
            {"X-API-Key": RAW_API_KEY_A},
            False,
            "INVALID_API_CREDENTIAL",
        ),
        (
            None,
            InvalidSessionError("internal session detail"),
            {},
            True,
            "INVALID_SESSION",
        ),
    ],
)
def test_search_invalid_auth_prevents_search_app(
    auth_route_client,
    api_error,
    session_error,
    headers,
    set_cookie,
    expected,
):
    client, api_resolver, session_resolver, calls = auth_route_client(
        api_resolver=FakeApiKeyTenantResolver(error=api_error),
        session_resolver=_session_resolver(error=session_error),
    )
    if set_cookie:
        client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post("/api/v1/search", json={}, headers=headers)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == expected
    assert calls["resolver"] == []
    assert calls["search"] == []


def test_search_client_company_id_cannot_override_trusted_identity(
    auth_route_client,
):
    client, _, _, calls = auth_route_client(
        api_resolver=FakeApiKeyTenantResolver(company_id=313)
    )

    response = client.post(
        "/api/v1/search",
        json={"requested_scope_keys": ["scope:requested"]},
        params={"company_id": "999"},
        headers={"X-API-Key": RAW_API_KEY_A},
    )

    assert response.status_code == 200
    assert calls["resolver"] == [(313, ("scope:requested",))]


def test_inventory_allows_api_key(auth_route_client):
    client, api_resolver, session_resolver, calls = auth_route_client(
        api_resolver=FakeApiKeyTenantResolver(company_id=444)
    )

    response = _post_inventory(client, headers={"X-API-Key": RAW_API_KEY_A})

    assert response.status_code == 200
    assert calls["ingest"][0]["company_id"] == 444
    assert api_resolver.calls == [RAW_API_KEY_A]
    assert session_resolver.calls == []


def test_inventory_allows_company_admin_human_session(auth_route_client):
    client, api_resolver, session_resolver, calls = auth_route_client(
        session_resolver=_session_resolver(
            company_id=555,
            role=MembershipRole.COMPANY_ADMIN,
        )
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = _post_inventory(client)

    assert response.status_code == 200
    assert calls["ingest"][0]["company_id"] == 555
    assert api_resolver.calls == []
    assert session_resolver.calls == [SESSION_TOKEN]


def test_inventory_denies_salesperson_without_ingestion(auth_route_client):
    client, api_resolver, session_resolver, calls = auth_route_client(
        session_resolver=_session_resolver(
            role=MembershipRole.SALESPERSON,
            user_id=7,
            membership_id=8,
            session_id=9,
        )
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = _post_inventory(client)

    assert response.status_code == 403
    assert response.json() == {
        "error": {
            "code": "FORBIDDEN",
            "message": "You do not have permission to perform this action",
        }
    }
    assert "7" not in response.text
    assert "8" not in response.text
    assert "9" not in response.text
    assert calls["ingest"] == []
    assert api_resolver.calls == []
    assert session_resolver.calls == [SESSION_TOKEN]


@pytest.mark.parametrize(
    ("headers", "set_cookie", "expected"),
    [
        ({"X-API-Key": RAW_API_KEY_A}, True, "AMBIGUOUS_CREDENTIALS"),
        ({}, False, "AUTHENTICATION_REQUIRED"),
    ],
)
def test_inventory_auth_failures_prevent_ingestion(
    auth_route_client,
    headers,
    set_cookie,
    expected,
):
    client, api_resolver, session_resolver, calls = auth_route_client()
    if set_cookie:
        client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = _post_inventory(client, headers=headers)

    assert response.status_code in {400, 401}
    assert response.json()["error"]["code"] == expected
    assert calls["ingest"] == []
    if expected == "AMBIGUOUS_CREDENTIALS":
        assert api_resolver.calls == []
        assert session_resolver.calls == []


def test_inventory_invalid_auth_prevents_ingestion(auth_route_client):
    client, api_resolver, session_resolver, calls = auth_route_client(
        session_resolver=_session_resolver(
            error=InvalidSessionError("internal session detail")
        )
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = _post_inventory(client)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_SESSION"
    assert calls["ingest"] == []
    assert api_resolver.calls == []
    assert session_resolver.calls == [SESSION_TOKEN]


def test_api_key_is_not_subject_to_human_role_checks(auth_route_client):
    client, _, _, calls = auth_route_client(
        api_resolver=FakeApiKeyTenantResolver(company_id=909),
        session_resolver=_session_resolver(role=MembershipRole.SALESPERSON),
    )

    response = _post_inventory(client, headers={"X-API-Key": RAW_API_KEY_A})

    assert response.status_code == 200
    assert calls["ingest"][0]["company_id"] == 909


def test_cross_tenant_identity_does_not_leak_between_requests(auth_route_client):
    client, api_resolver, session_resolver, calls = auth_route_client(
        api_resolver=FakeApiKeyTenantResolver(company_id=111),
        session_resolver=_session_resolver(company_id=222),
    )

    api_response = client.post(
        "/api/v1/search",
        json={},
        headers={"X-API-Key": RAW_API_KEY_A},
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)
    session_response = client.post("/api/v1/search", json={})

    assert api_response.status_code == 200
    assert session_response.status_code == 200
    assert [call[0] for call in calls["resolver"]] == [111, 222]
    assert api_resolver.calls == [RAW_API_KEY_A]
    assert session_resolver.calls == [SESSION_TOKEN]


def test_search_and_inventory_openapi_use_or_security(auth_route_client):
    client, _, _, _ = auth_route_client()

    schema = client.get("/openapi.json").json()
    search_security = schema["paths"]["/api/v1/search"]["post"]["security"]
    inventory_security = schema["paths"]["/api/v1/inventory/ingest"]["post"][
        "security"
    ]

    assert search_security == [{"ApiKeyAuth": []}, {"SessionCookieAuth": []}]
    assert inventory_security == [{"ApiKeyAuth": []}, {"SessionCookieAuth": []}]
    assert {"ApiKeyAuth": [], "SessionCookieAuth": []} not in search_security
    assert {"ApiKeyAuth": [], "SessionCookieAuth": []} not in inventory_security
    assert "security" not in schema["paths"]["/api/v1/auth/login"]["post"]
    assert "security" not in schema["paths"]["/api/v1/health"]["get"]
    assert "security" not in schema["paths"]["/api/v1/ready"]["get"]


def _post_inventory(client: TestClient, *, headers: dict[str, str] | None = None):
    return client.post(
        "/api/v1/inventory/ingest",
        data={"scope_key": "scope:nc"},
        files={"file": ("inventory.csv", b"unit", "text/csv")},
        headers=headers or {},
    )


def _session_resolver(
    *,
    company_id: int = 202,
    role: MembershipRole = MembershipRole.COMPANY_ADMIN,
    user_id: int = 101,
    membership_id: int = 303,
    session_id: int = 404,
    error: Exception | None = None,
) -> FakeHumanSessionResolver:
    return FakeHumanSessionResolver(
        context=AuthenticatedUserContext(
            user_id=user_id,
            membership_id=membership_id,
            company_id=company_id,
            role=role,
            session_id=session_id,
        ),
        error=error,
    )
