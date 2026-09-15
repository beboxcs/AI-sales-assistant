from dataclasses import dataclass

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api import security as api_security
from app.api.errors import register_exception_handlers
from app.auth.human.exceptions import (
    AuthInfrastructureError,
    InactiveMembershipError,
    InactiveUserError,
    InvalidCsrfTokenError,
    InvalidSessionError,
    SessionExpiredError,
    SessionRevokedError,
)
from app.main import app


TRUSTED_ORIGIN = "https://app.example.com"
SECOND_TRUSTED_ORIGIN = "http://localhost:3000"
SESSION_TOKEN = "opaque-session-token"
CSRF_TOKEN = "opaque-csrf-token"
RAW_API_KEY = "rea_v1_" + ("K" * 22) + "." + ("s" * 43)
AUTHENTICATION_REQUIRED_RESPONSE = {
    "error": {
        "code": "AUTHENTICATION_REQUIRED",
        "message": "Authentication is required",
    }
}
AMBIGUOUS_CREDENTIALS_RESPONSE = {
    "error": {
        "code": "AMBIGUOUS_CREDENTIALS",
        "message": "Use exactly one authentication credential type",
    }
}


@dataclass
class FakeCsrfService:
    issued_token: str = CSRF_TOKEN
    issue_error: Exception | None = None
    verify_error: Exception | None = None
    issue_calls: list[str] | None = None
    verify_calls: list[tuple[str, str]] | None = None

    def __post_init__(self):
        self.issue_calls = []
        self.verify_calls = []

    def issue_csrf_token(self, *, session_token: str) -> str:
        self.issue_calls.append(session_token)
        if self.issue_error is not None:
            raise self.issue_error
        return self.issued_token

    def verify_csrf_token(self, *, session_token: str, csrf_token: str) -> None:
        self.verify_calls.append((session_token, csrf_token))
        if self.verify_error is not None:
            raise self.verify_error


@pytest.fixture
def csrf_client(monkeypatch):
    original_overrides = app.dependency_overrides.copy()
    original_openapi_schema = app.openapi_schema
    clients = []
    services = []

    monkeypatch.setattr(api_security.settings, "APP_ENV", "production")
    monkeypatch.setattr(
        api_security.settings,
        "TRUSTED_BROWSER_ORIGINS",
        (TRUSTED_ORIGIN, SECOND_TRUSTED_ORIGIN),
    )
    app.openapi_schema = None

    def factory(service: FakeCsrfService | None = None):
        selected_service = service or FakeCsrfService()
        services.append(selected_service)
        app.dependency_overrides[
            api_security.get_human_csrf_service
        ] = lambda: selected_service
        client = TestClient(app)
        clients.append(client)
        return client, selected_service

    try:
        yield factory
    finally:
        for client in clients:
            client.close()
        app.dependency_overrides = original_overrides
        app.openapi_schema = original_openapi_schema


def test_csrf_issuance_with_valid_session_and_trusted_origin_succeeds(csrf_client):
    client, service = csrf_client()
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/api/v1/auth/csrf",
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 200
    assert response.json() == {"csrf_token": CSRF_TOKEN}
    assert service.issue_calls == [SESSION_TOKEN]
    assert "set-cookie" not in response.headers


def test_csrf_issuance_response_does_not_expose_internal_fields(csrf_client):
    client, service = csrf_client(FakeCsrfService(issued_token=CSRF_TOKEN))
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/api/v1/auth/csrf",
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 200
    assert set(response.json()) == {"csrf_token"}
    response_text = response.text
    assert SESSION_TOKEN not in response_text
    assert "digest" not in response_text
    assert "session_id" not in response_text
    assert "user_id" not in response_text
    assert "company_id" not in response_text
    assert service.issue_calls == [SESSION_TOKEN]


@pytest.mark.parametrize(
    ("headers", "set_session_cookie", "expected_status", "expected_body"),
    [
        ({}, False, 401, AUTHENTICATION_REQUIRED_RESPONSE),
        ({"X-API-Key": RAW_API_KEY}, False, 401, AUTHENTICATION_REQUIRED_RESPONSE),
        (
            {"X-API-Key": RAW_API_KEY},
            True,
            400,
            AMBIGUOUS_CREDENTIALS_RESPONSE,
        ),
    ],
)
def test_csrf_issuance_rejects_missing_api_key_or_ambiguous_credentials(
    csrf_client,
    headers,
    set_session_cookie,
    expected_status,
    expected_body,
):
    client, service = csrf_client()
    if set_session_cookie:
        client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/api/v1/auth/csrf",
        headers={"Origin": TRUSTED_ORIGIN, **headers},
    )

    assert response.status_code == expected_status
    assert response.json() == expected_body
    assert service.issue_calls == []
    assert RAW_API_KEY not in response.text
    assert SESSION_TOKEN not in response.text


@pytest.mark.parametrize("origin", [None, "https://attacker.example"])
def test_csrf_issuance_ambiguous_credentials_are_rejected_before_origin(
    csrf_client,
    origin,
):
    client, service = csrf_client()
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)
    headers = {"X-API-Key": RAW_API_KEY}
    if origin is not None:
        headers["Origin"] = origin

    response = client.post("/api/v1/auth/csrf", headers=headers)

    assert response.status_code == 400
    assert response.json() == AMBIGUOUS_CREDENTIALS_RESPONSE
    assert service.issue_calls == []
    assert RAW_API_KEY not in response.text
    assert SESSION_TOKEN not in response.text


@pytest.mark.parametrize(
    "error",
    [
        InvalidSessionError("unknown token"),
        SessionExpiredError("absolute expired"),
        SessionRevokedError("revoked session id 404"),
        InactiveUserError("inactive user"),
        InactiveMembershipError("inactive membership"),
    ],
)
def test_csrf_issuance_invalid_sessions_return_generic_401(csrf_client, error):
    client, service = csrf_client(FakeCsrfService(issue_error=error))
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/api/v1/auth/csrf",
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 401
    assert response.json() == _invalid_session_response()
    assert str(error) not in response.text
    assert SESSION_TOKEN not in response.text
    assert service.issue_calls == [SESSION_TOKEN]


@pytest.mark.parametrize(
    "error",
    [
        AuthInfrastructureError("database down"),
        RuntimeError("unexpected backend detail"),
    ],
)
def test_csrf_issuance_backend_failures_return_safe_500(csrf_client, error):
    client, service = csrf_client(FakeCsrfService(issue_error=error))
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/api/v1/auth/csrf",
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 500
    assert response.json() == _authentication_unavailable_response()
    assert str(error) not in response.text
    assert SESSION_TOKEN not in response.text
    assert service.issue_calls == [SESSION_TOKEN]


@pytest.mark.parametrize("origin", [None, "https://attacker.example"])
def test_csrf_issuance_requires_trusted_origin_before_backend_call(
    csrf_client,
    origin,
):
    client, service = csrf_client()
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)
    headers = {} if origin is None else {"Origin": origin}

    response = client.post("/api/v1/auth/csrf", headers=headers)

    assert response.status_code == 403
    assert response.json() == _untrusted_origin_response()
    assert service.issue_calls == []
    assert SESSION_TOKEN not in response.text


def test_csrf_issuance_accepts_configured_trusted_origin(csrf_client):
    client, service = csrf_client()
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/api/v1/auth/csrf",
        headers={"Origin": SECOND_TRUSTED_ORIGIN},
    )

    assert response.status_code == 200
    assert response.json() == {"csrf_token": CSRF_TOKEN}
    assert service.issue_calls == [SESSION_TOKEN]


def test_csrf_issuance_openapi_documents_session_cookie_only(csrf_client):
    client, _ = csrf_client()

    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/api/v1/auth/csrf"]["post"]
    response_ref = operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ]["$ref"]
    response_schema = _schema_for_ref(schema, response_ref)

    assert operation["security"] == [{"SessionCookieAuth": []}]
    assert operation["security"] != [{"ApiKeyAuth": []}, {"SessionCookieAuth": []}]
    assert "ApiKeyAuth" not in str(operation["security"])
    assert set(response_schema["properties"]) == {"csrf_token"}


@pytest.fixture
def csrf_verification_client(monkeypatch):
    test_app = FastAPI()
    register_exception_handlers(test_app)
    downstream_calls = []

    @test_app.post(
        "/protected",
        dependencies=[Depends(api_security.require_valid_human_session_csrf)],
    )
    def protected():
        downstream_calls.append("called")
        return {"status": "ok"}

    monkeypatch.setattr(
        api_security.settings,
        "TRUSTED_BROWSER_ORIGINS",
        (TRUSTED_ORIGIN,),
    )

    def factory(service: FakeCsrfService | None = None):
        selected_service = service or FakeCsrfService()
        test_app.dependency_overrides[
            api_security.get_human_csrf_service
        ] = lambda: selected_service
        test_app.openapi_schema = None
        return TestClient(test_app), selected_service, downstream_calls

    yield factory

    test_app.dependency_overrides = {}


def test_valid_csrf_verification_allows_downstream_once(csrf_verification_client):
    client, service, downstream_calls = csrf_verification_client()
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/protected",
        headers={
            "Origin": TRUSTED_ORIGIN,
            "X-CSRF-Token": CSRF_TOKEN,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert service.verify_calls == [(SESSION_TOKEN, CSRF_TOKEN)]
    assert downstream_calls == ["called"]
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize(
    ("headers", "error"),
    [
        ({"X-CSRF-Token": "wrong-csrf-token"}, InvalidCsrfTokenError("bad token")),
        ({}, None),
    ],
)
def test_csrf_verification_failures_are_generic_403(
    csrf_verification_client,
    headers,
    error,
):
    service = FakeCsrfService(verify_error=error)
    client, service, downstream_calls = csrf_verification_client(service)
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/protected",
        headers={"Origin": TRUSTED_ORIGIN, **headers},
    )

    assert response.status_code == 403
    assert response.json() == {
        "error": {
            "code": "INVALID_CSRF_TOKEN",
            "message": "Request verification failed",
        }
    }
    if error is None:
        assert service.verify_calls == []
    else:
        assert service.verify_calls == [(SESSION_TOKEN, "wrong-csrf-token")]
        assert str(error) not in response.text
    assert CSRF_TOKEN not in response.text
    assert SESSION_TOKEN not in response.text
    assert downstream_calls == []


def test_csrf_verification_invalid_session_returns_generic_401(
    csrf_verification_client,
):
    error = InvalidSessionError("session lookup detail")
    client, service, downstream_calls = csrf_verification_client(
        FakeCsrfService(verify_error=error)
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/protected",
        headers={"Origin": TRUSTED_ORIGIN, "X-CSRF-Token": CSRF_TOKEN},
    )

    assert response.status_code == 401
    assert response.json() == _invalid_session_response()
    assert str(error) not in response.text
    assert SESSION_TOKEN not in response.text
    assert CSRF_TOKEN not in response.text
    assert service.verify_calls == [(SESSION_TOKEN, CSRF_TOKEN)]
    assert downstream_calls == []


@pytest.mark.parametrize(
    "error",
    [
        AuthInfrastructureError("csrf store down"),
        RuntimeError("unexpected csrf backend detail"),
    ],
)
def test_csrf_verification_backend_failures_return_safe_500(
    csrf_verification_client,
    error,
):
    client, service, downstream_calls = csrf_verification_client(
        FakeCsrfService(verify_error=error)
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/protected",
        headers={"Origin": TRUSTED_ORIGIN, "X-CSRF-Token": CSRF_TOKEN},
    )

    assert response.status_code == 500
    assert response.json() == _authentication_unavailable_response()
    assert str(error) not in response.text
    assert SESSION_TOKEN not in response.text
    assert CSRF_TOKEN not in response.text
    assert service.verify_calls == [(SESSION_TOKEN, CSRF_TOKEN)]
    assert downstream_calls == []


def test_csrf_verification_origin_failure_precedes_backend(
    csrf_verification_client,
):
    client, service, downstream_calls = csrf_verification_client()
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/protected",
        headers={
            "Origin": "https://attacker.example",
            "X-CSRF-Token": CSRF_TOKEN,
        },
    )

    assert response.status_code == 403
    assert response.json() == _untrusted_origin_response()
    assert service.verify_calls == []
    assert SESSION_TOKEN not in response.text
    assert CSRF_TOKEN not in response.text
    assert downstream_calls == []


@pytest.mark.parametrize("origin", [None, "https://attacker.example"])
def test_csrf_verification_ambiguous_credentials_are_rejected_before_origin(
    csrf_verification_client,
    origin,
):
    client, service, downstream_calls = csrf_verification_client()
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)
    headers = {
        "X-API-Key": RAW_API_KEY,
        "X-CSRF-Token": CSRF_TOKEN,
    }
    if origin is not None:
        headers["Origin"] = origin

    response = client.post("/protected", headers=headers)

    assert response.status_code == 400
    assert response.json() == AMBIGUOUS_CREDENTIALS_RESPONSE
    assert service.verify_calls == []
    assert downstream_calls == []
    assert RAW_API_KEY not in response.text
    assert SESSION_TOKEN not in response.text
    assert CSRF_TOKEN not in response.text


def _invalid_session_response():
    return {
        "error": {
            "code": "INVALID_SESSION",
            "message": "Invalid or unavailable session",
        }
    }


def _authentication_unavailable_response():
    return {
        "error": {
            "code": "AUTHENTICATION_UNAVAILABLE",
            "message": "Authentication could not be completed",
        }
    }


def _untrusted_origin_response():
    return {
        "error": {
            "code": "UNTRUSTED_ORIGIN",
            "message": "Request origin is not trusted",
        }
    }


def _schema_for_ref(schema, ref):
    return schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
