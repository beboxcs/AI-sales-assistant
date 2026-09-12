from dataclasses import dataclass
from decimal import Decimal

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.ai_engine.inventory_application import InventoryIngestionResult
from app.api.errors import register_exception_handlers
from app.api import security as api_security
from app.api.dependencies import (
    get_inventory_ingest_max_upload_bytes,
    get_inventory_ingestion_service,
    get_search_company_id,
    get_search_context_resolver,
    get_trusted_company_id,
)
from app.api.v1 import search as search_route
from app.auth.contracts import TrustedTenantContext
from app.auth.exceptions import (
    AuthBackendError,
    ExpiredApiCredentialError,
    InvalidApiCredentialError,
    RevokedApiCredentialError,
)
from app.main import app
from app.search.contracts import SearchExecutionContext

RAW_KEY = "rea_v1_" + ("K" * 22) + "." + ("s" * 43)
SECOND_RAW_KEY = "rea_v1_" + ("L" * 22) + "." + ("t" * 43)
PEPPER = "api-auth-test-pepper"


@dataclass(frozen=True)
class EmptySearchResult:
    results: tuple = ()


@pytest.fixture
def auth_client(monkeypatch):
    original_overrides = app.dependency_overrides.copy()
    original_openapi_schema = app.openapi_schema
    clients = []

    monkeypatch.setattr(api_security.settings, "APP_ENV", "production")
    monkeypatch.setattr(api_security.settings, "API_KEY_PEPPER", PEPPER)
    monkeypatch.setattr(api_security.settings, "DEV_COMPANY_ID", None)
    app.openapi_schema = None

    def factory() -> TestClient:
        client = TestClient(app)
        clients.append(client)
        return client

    try:
        yield factory
    finally:
        for client in clients:
            client.close()
        app.dependency_overrides = original_overrides
        app.openapi_schema = original_openapi_schema


def test_missing_route_credentials_return_authentication_required(auth_client):
    response = auth_client().post("/api/v1/search", json={})

    assert response.status_code == 401
    assert response.json() == _authentication_required_response()


def test_api_key_only_does_not_trigger_credential_ambiguity():
    response = _ambiguity_client().get(
        "/credential-check",
        headers={"X-API-Key": RAW_KEY},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_session_cookie_only_does_not_trigger_credential_ambiguity():
    client = _ambiguity_client()
    client.cookies.set("rea_session", "session-token-secret")

    response = client.get("/credential-check")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_absent_credentials_do_not_trigger_credential_ambiguity():
    response = _ambiguity_client().get("/credential-check")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_key_and_session_cookie_are_rejected_as_ambiguous_credentials():
    session_token = "session-token-secret"
    client = _ambiguity_client()
    client.cookies.set("rea_session", session_token)

    response = client.get(
        "/credential-check",
        headers={"X-API-Key": RAW_KEY},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "AMBIGUOUS_CREDENTIALS",
            "message": "Use exactly one authentication credential type",
        }
    }
    assert RAW_KEY not in response.text
    assert session_token not in response.text


def test_session_cookie_openapi_scheme_foundation_is_preserved():
    schema = _ambiguity_client().get("/openapi.json").json()

    assert schema["components"]["securitySchemes"]["SessionCookieAuth"] == {
        "type": "apiKey",
        "in": "cookie",
        "name": "rea_session",
    }


def test_malformed_api_key_returns_generic_401(auth_client):
    response = auth_client().post(
        "/api/v1/search",
        json={},
        headers={"X-API-Key": "malformed"},
    )

    assert response.status_code == 401
    assert response.json() == _invalid_credential_response()


def test_invalid_supplied_key_in_local_mode_does_not_use_dev_fallback(
    auth_client,
    monkeypatch,
):
    monkeypatch.setattr(api_security.settings, "APP_ENV", "local")
    monkeypatch.setattr(api_security.settings, "DEV_COMPANY_ID", 77)
    monkeypatch.setattr(
        api_security,
        "authenticate_api_key",
        lambda raw_key, *, pepper: (_ for _ in ()).throw(
            InvalidApiCredentialError("invalid local key")
        ),
    )

    response = auth_client().post(
        "/api/v1/search",
        json={},
        headers={"X-API-Key": RAW_KEY},
    )

    assert response.status_code == 401
    assert response.json() == _invalid_credential_response()
    assert "invalid local key" not in response.text


@pytest.mark.parametrize(
    "error",
    [
        InvalidApiCredentialError("unknown key internal detail"),
        InvalidApiCredentialError("wrong secret internal detail"),
        RevokedApiCredentialError("revoked internal detail"),
        ExpiredApiCredentialError("expired internal detail"),
        InvalidApiCredentialError("inactive internal detail"),
    ],
)
def test_unavailable_credentials_return_same_generic_401(
    auth_client,
    monkeypatch,
    error,
):
    monkeypatch.setattr(
        api_security,
        "authenticate_api_key",
        lambda raw_key, *, pepper: (_ for _ in ()).throw(error),
    )

    response = auth_client().post(
        "/api/v1/search",
        json={},
        headers={"X-API-Key": RAW_KEY},
    )

    assert response.status_code == 401
    assert response.json() == _invalid_credential_response()
    assert str(error) not in response.text
    assert RAW_KEY not in response.text


def test_auth_backend_error_returns_safe_500_without_secret_leak(
    auth_client,
    monkeypatch,
):
    monkeypatch.setattr(
        api_security,
        "authenticate_api_key",
        lambda raw_key, *, pepper: (_ for _ in ()).throw(
            AuthBackendError(f"postgresql internal {pepper}")
        ),
    )

    response = auth_client().post(
        "/api/v1/search",
        json={},
        headers={"X-API-Key": RAW_KEY},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "AUTHENTICATION_UNAVAILABLE",
            "message": "Authentication could not be completed",
        }
    }
    assert PEPPER not in response.text
    assert "postgresql internal" not in response.text


def test_unexpected_auth_error_returns_safe_500_without_secret_leak(
    auth_client,
    monkeypatch,
):
    monkeypatch.setattr(
        api_security,
        "authenticate_api_key",
        lambda raw_key, *, pepper: (_ for _ in ()).throw(
            RuntimeError(f"raw={raw_key} pepper={pepper}")
        ),
    )

    response = auth_client().post(
        "/api/v1/search",
        json={},
        headers={"X-API-Key": RAW_KEY},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "AUTHENTICATION_UNAVAILABLE",
            "message": "Authentication could not be completed",
        }
    }
    assert RAW_KEY not in response.text
    assert PEPPER not in response.text


def test_valid_key_passes_company_id_into_search_scope_resolution(
    auth_client,
    monkeypatch,
):
    calls = {"auth": [], "resolver": [], "search": []}

    def authenticate(raw_key, *, pepper):
        calls["auth"].append((raw_key, pepper))
        return TrustedTenantContext(company_id=702, credential_id=9)

    def resolver(*, company_id, requested_scope_keys):
        calls["resolver"].append((company_id, requested_scope_keys))
        return SearchExecutionContext(
            company_id=company_id,
            resolved_scope_keys=("scope:resolved",),
        )

    def search_units(*, filters, context):
        calls["search"].append((filters, context))
        return EmptySearchResult()

    monkeypatch.setattr(api_security, "authenticate_api_key", authenticate)
    app.dependency_overrides[get_search_context_resolver] = lambda: resolver
    monkeypatch.setattr(search_route.search_application, "search_units", search_units)

    response = auth_client().post(
        "/api/v1/search",
        json={"requested_scope_keys": ["scope:requested"]},
        headers={"X-API-Key": RAW_KEY},
    )

    assert response.status_code == 200
    assert calls["auth"] == [(RAW_KEY, PEPPER)]
    assert calls["resolver"] == [(702, ("scope:requested",))]
    assert calls["search"][0][1].company_id == 702


def test_auth_failure_does_not_call_search_application(
    auth_client,
    monkeypatch,
):
    search_calls = []
    monkeypatch.setattr(
        search_route.search_application,
        "search_units",
        lambda **kwargs: search_calls.append(kwargs),
    )

    response = auth_client().post("/api/v1/search", json={})

    assert response.status_code == 401
    assert search_calls == []


def test_two_companies_resolve_independently_without_tenant_leakage(
    auth_client,
    monkeypatch,
):
    credentials = {
        RAW_KEY: TrustedTenantContext(company_id=100, credential_id=1),
        SECOND_RAW_KEY: TrustedTenantContext(company_id=200, credential_id=2),
    }
    resolver_company_ids = []

    monkeypatch.setattr(
        api_security,
        "authenticate_api_key",
        lambda raw_key, *, pepper: credentials[raw_key],
    )
    app.dependency_overrides[get_search_context_resolver] = lambda: (
        lambda *, company_id, requested_scope_keys: resolver_company_ids.append(
            company_id
        )
        or SearchExecutionContext(
            company_id=company_id,
            resolved_scope_keys=("scope:resolved",),
        )
    )
    monkeypatch.setattr(
        search_route.search_application,
        "search_units",
        lambda *, filters, context: EmptySearchResult(),
    )

    client = auth_client()
    first_response = client.post(
        "/api/v1/search",
        json={},
        headers={"X-API-Key": RAW_KEY},
    )
    second_response = client.post(
        "/api/v1/search",
        json={},
        headers={"X-API-Key": SECOND_RAW_KEY},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert resolver_company_ids == [100, 200]


def test_combined_inventory_route_does_not_use_local_dev_company_fallback(
    auth_client,
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(api_security.settings, "APP_ENV", "local")
    monkeypatch.setattr(api_security.settings, "DEV_COMPANY_ID", 77)
    monkeypatch.setattr(api_security.settings, "API_KEY_PEPPER", None)
    monkeypatch.setattr(
        api_security,
        "authenticate_api_key",
        lambda raw_key, *, pepper: pytest.fail("auth backend should not run"),
    )

    def ingest(*, company_id, scope_key, source):
        calls.append((company_id, scope_key, source.filename))
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

    app.dependency_overrides[get_inventory_ingestion_service] = lambda: ingest
    app.dependency_overrides[get_inventory_ingest_max_upload_bytes] = lambda: 1024

    response = auth_client().post(
        "/api/v1/inventory/ingest",
        data={"scope_key": "scope:nc"},
        files={"file": ("inventory.csv", b"unit", "text/csv")},
    )

    assert response.status_code == 401
    assert response.json() == _authentication_required_response()
    assert calls == []


def test_local_fallback_returns_company_id_without_trusted_tenant_context(
    auth_client,
    monkeypatch,
):
    monkeypatch.setattr(api_security.settings, "APP_ENV", "local")
    monkeypatch.setattr(api_security.settings, "DEV_COMPANY_ID", 77)
    monkeypatch.setattr(
        api_security,
        "resolve_trusted_tenant",
        lambda raw_api_key: pytest.fail("local fallback must not create context"),
    )

    assert api_security.get_trusted_company_id(raw_api_key=None) == 77


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_deployed_environments_never_fall_back_to_dev_company_id(
    auth_client,
    monkeypatch,
    app_env,
):
    calls = []
    monkeypatch.setattr(api_security.settings, "APP_ENV", app_env)
    monkeypatch.setattr(api_security.settings, "DEV_COMPANY_ID", 77)
    app.dependency_overrides[get_inventory_ingestion_service] = (
        lambda: lambda **kwargs: calls.append(kwargs)
    )

    response = auth_client().post(
        "/api/v1/inventory/ingest",
        data={"scope_key": "scope:nc"},
        files={"file": ("inventory.csv", b"unit", "text/csv")},
    )

    assert response.status_code == 401
    assert response.json() == _authentication_required_response()
    assert calls == []


def test_auth_failure_does_not_call_inventory_application(
    auth_client,
):
    calls = []
    app.dependency_overrides[get_inventory_ingestion_service] = (
        lambda: lambda **kwargs: calls.append(kwargs)
    )

    response = auth_client().post(
        "/api/v1/inventory/ingest",
        data={"scope_key": "scope:nc"},
        files={"file": ("inventory.csv", b"unit", "text/csv")},
    )

    assert response.status_code == 401
    assert response.json() == _authentication_required_response()
    assert calls == []


def test_valid_key_passes_company_id_into_inventory_ingestion_once(
    auth_client,
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        api_security,
        "authenticate_api_key",
        lambda raw_key, *, pepper: TrustedTenantContext(
            company_id=321,
            credential_id=8,
        ),
    )

    def ingest(*, company_id, scope_key, source):
        calls.append((company_id, scope_key, source.filename))
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

    app.dependency_overrides[get_inventory_ingestion_service] = lambda: ingest
    app.dependency_overrides[get_inventory_ingest_max_upload_bytes] = lambda: 1024

    response = auth_client().post(
        "/api/v1/inventory/ingest",
        data={"scope_key": "scope:nc", "company_id": "999"},
        files={"file": ("inventory.csv", b"unit", "text/csv")},
        headers={"X-API-Key": RAW_KEY},
    )

    assert response.status_code == 200
    assert calls == [(321, "scope:nc", "inventory.csv")]


def test_search_company_dependency_override_remains_possible(
    auth_client,
    monkeypatch,
):
    resolver_calls = []
    app.dependency_overrides[get_search_company_id] = lambda: 909
    app.dependency_overrides[get_search_context_resolver] = lambda: (
        lambda *, company_id, requested_scope_keys: resolver_calls.append(
            company_id
        )
        or SearchExecutionContext(
            company_id=company_id,
            resolved_scope_keys=("scope:resolved",),
        )
    )
    monkeypatch.setattr(
        search_route.search_application,
        "search_units",
        lambda *, filters, context: EmptySearchResult(),
    )

    response = auth_client().post("/api/v1/search", json={})

    assert response.status_code == 200
    assert resolver_calls == [909]


def test_search_decimal_parser_still_preserves_numeric_precision_with_auth(
    auth_client,
    monkeypatch,
):
    captured_filters = []
    monkeypatch.setattr(
        api_security,
        "authenticate_api_key",
        lambda raw_key, *, pepper: TrustedTenantContext(
            company_id=88,
            credential_id=3,
        ),
    )
    app.dependency_overrides[get_search_context_resolver] = lambda: (
        lambda *, company_id, requested_scope_keys: SearchExecutionContext(
            company_id=company_id,
            resolved_scope_keys=("scope:resolved",),
        )
    )

    def search_units(*, filters, context):
        captured_filters.append(filters)
        return EmptySearchResult()

    monkeypatch.setattr(search_route.search_application, "search_units", search_units)

    response = auth_client().post(
        "/api/v1/search",
        content='{"min_price": 1234567890123456.78}',
        headers={
            "content-type": "application/json",
            "X-API-Key": RAW_KEY,
        },
    )

    assert response.status_code == 200
    assert captured_filters[0].min_price == Decimal("1234567890123456.78")


def test_malformed_json_behavior_remains_unchanged_with_auth(
    auth_client,
    monkeypatch,
):
    search_calls = []
    monkeypatch.setattr(
        api_security,
        "authenticate_api_key",
        lambda raw_key, *, pepper: TrustedTenantContext(
            company_id=88,
            credential_id=3,
        ),
    )
    monkeypatch.setattr(
        search_route.search_application,
        "search_units",
        lambda **kwargs: search_calls.append(kwargs),
    )

    response = auth_client().post(
        "/api/v1/search",
        content='{"min_price": 100',
        headers={
            "content-type": "application/json",
            "X-API-Key": RAW_KEY,
        },
    )

    assert response.status_code == 422
    assert search_calls == []


def test_openapi_documents_api_key_security_and_search_body(auth_client):
    schema = auth_client().get("/openapi.json").json()

    security_scheme = schema["components"]["securitySchemes"]["ApiKeyAuth"]
    assert security_scheme == {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
    }
    assert schema["paths"]["/api/v1/search"]["post"]["security"] == [
        {"ApiKeyAuth": []},
        {"SessionCookieAuth": []},
    ]
    assert schema["paths"]["/api/v1/inventory/ingest"]["post"]["security"] == [
        {"ApiKeyAuth": []},
        {"SessionCookieAuth": []},
    ]
    assert "requestBody" in schema["paths"]["/api/v1/search"]["post"]
    assert "parameters" not in schema["paths"]["/api/v1/search"]["post"]


def _invalid_credential_response():
    return {
        "error": {
            "code": "INVALID_API_CREDENTIAL",
            "message": "Invalid or unavailable API credential",
        }
    }


def _authentication_required_response():
    return {
        "error": {
            "code": "AUTHENTICATION_REQUIRED",
            "message": "Authentication is required",
        }
    }


def _ambiguity_client() -> TestClient:
    test_app = FastAPI()
    register_exception_handlers(test_app)

    @test_app.get(
        "/credential-check",
        dependencies=[Depends(api_security.require_unambiguous_credentials)],
    )
    def credential_check():
        return {"status": "ok"}

    return TestClient(test_app)
