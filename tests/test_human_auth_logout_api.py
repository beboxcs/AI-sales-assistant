from dataclasses import dataclass
from http.cookies import SimpleCookie

import pytest
from fastapi.testclient import TestClient

from app.api import security as api_security
from app.api.v1.auth import (
    get_human_session_revocation_service,
)
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
SESSION_TOKEN = "opaque-session-token"
CSRF_TOKEN = "opaque-csrf-token"
RAW_API_KEY = "rea_v1_" + ("K" * 22) + "." + ("s" * 43)

AMBIGUOUS_CREDENTIALS_RESPONSE = {
    "error": {
        "code": "AMBIGUOUS_CREDENTIALS",
        "message": "Use exactly one authentication credential type",
    }
}
AUTHENTICATION_REQUIRED_RESPONSE = {
    "error": {
        "code": "AUTHENTICATION_REQUIRED",
        "message": "Authentication is required",
    }
}
INVALID_CSRF_RESPONSE = {
    "error": {
        "code": "INVALID_CSRF_TOKEN",
        "message": "Request verification failed",
    }
}


@dataclass
class FakeCsrfService:
    verify_error: Exception | None = None
    verify_calls: list[tuple[str, str]] | None = None

    def __post_init__(self):
        self.verify_calls = []

    def issue_csrf_token(self, *, session_token: str) -> str:
        raise AssertionError("logout must not issue CSRF tokens")

    def verify_csrf_token(self, *, session_token: str, csrf_token: str) -> None:
        self.verify_calls.append((session_token, csrf_token))
        if self.verify_error is not None:
            raise self.verify_error


@dataclass
class FakeRevocationService:
    error: Exception | None = None
    calls: list[str] | None = None

    def __post_init__(self):
        self.calls = []

    def revoke_session(self, *, token: str) -> None:
        self.calls.append(token)
        if self.error is not None:
            raise self.error


@pytest.fixture
def logout_client(monkeypatch):
    original_overrides = app.dependency_overrides.copy()
    original_openapi_schema = app.openapi_schema
    clients = []

    monkeypatch.setattr(api_security.settings, "APP_ENV", "production")
    monkeypatch.setattr(
        api_security.settings,
        "TRUSTED_BROWSER_ORIGINS",
        (TRUSTED_ORIGIN,),
    )
    app.openapi_schema = None

    def factory(
        *,
        csrf_service: FakeCsrfService | None = None,
        revocation_service: FakeRevocationService | None = None,
    ):
        selected_csrf = csrf_service or FakeCsrfService()
        selected_revocation = revocation_service or FakeRevocationService()
        app.dependency_overrides[
            api_security.get_human_csrf_service
        ] = lambda: selected_csrf
        app.dependency_overrides[
            get_human_session_revocation_service
        ] = lambda: selected_revocation
        client = TestClient(app)
        clients.append(client)
        return client, selected_csrf, selected_revocation

    try:
        yield factory
    finally:
        for client in clients:
            client.close()
        app.dependency_overrides = original_overrides
        app.openapi_schema = original_openapi_schema


def test_logout_with_valid_session_trusted_origin_and_valid_csrf_succeeds(
    logout_client,
):
    client, csrf_service, revocation_service = logout_client()
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/api/v1/auth/logout",
        headers={"Origin": TRUSTED_ORIGIN, "X-CSRF-Token": CSRF_TOKEN},
    )

    assert response.status_code == 204
    assert response.content == b""
    assert csrf_service.verify_calls == [(SESSION_TOKEN, CSRF_TOKEN)]
    assert revocation_service.calls == [SESSION_TOKEN]
    _assert_session_cookie_cleared(response)


def test_logout_without_session_is_idempotent_and_does_not_require_csrf(
    logout_client,
):
    client, csrf_service, revocation_service = logout_client()

    response = client.post("/api/v1/auth/logout")

    assert response.status_code == 204
    assert response.content == b""
    assert csrf_service.verify_calls == []
    assert revocation_service.calls == []
    _assert_session_cookie_cleared(response)


def test_logout_with_api_key_only_requires_human_authentication(logout_client):
    client, csrf_service, revocation_service = logout_client()

    response = client.post(
        "/api/v1/auth/logout",
        headers={"X-API-Key": RAW_API_KEY},
    )

    assert response.status_code == 401
    assert response.json() == AUTHENTICATION_REQUIRED_RESPONSE
    assert csrf_service.verify_calls == []
    assert revocation_service.calls == []
    assert RAW_API_KEY not in response.text
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": TRUSTED_ORIGIN, "X-CSRF-Token": CSRF_TOKEN},
        {"X-CSRF-Token": CSRF_TOKEN},
        {"Origin": "https://attacker.example", "X-CSRF-Token": CSRF_TOKEN},
    ],
)
def test_logout_with_api_key_and_session_is_ambiguous_before_origin_or_csrf(
    logout_client,
    headers,
):
    client, csrf_service, revocation_service = logout_client()
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/api/v1/auth/logout",
        headers={"X-API-Key": RAW_API_KEY, **headers},
    )

    assert response.status_code == 400
    assert response.json() == AMBIGUOUS_CREDENTIALS_RESPONSE
    assert csrf_service.verify_calls == []
    assert revocation_service.calls == []
    assert RAW_API_KEY not in response.text
    assert SESSION_TOKEN not in response.text
    assert CSRF_TOKEN not in response.text
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize("origin", [None, "https://attacker.example"])
def test_logout_with_presented_session_requires_trusted_origin_before_csrf(
    logout_client,
    origin,
):
    client, csrf_service, revocation_service = logout_client()
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)
    headers = {"X-CSRF-Token": CSRF_TOKEN}
    if origin is not None:
        headers["Origin"] = origin

    response = client.post("/api/v1/auth/logout", headers=headers)

    assert response.status_code == 403
    assert response.json() == _untrusted_origin_response()
    assert csrf_service.verify_calls == []
    assert revocation_service.calls == []
    assert SESSION_TOKEN not in response.text
    assert CSRF_TOKEN not in response.text
    assert "set-cookie" not in response.headers


def test_logout_with_presented_session_requires_csrf_before_revoke(logout_client):
    client, csrf_service, revocation_service = logout_client()
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/api/v1/auth/logout",
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 403
    assert response.json() == INVALID_CSRF_RESPONSE
    assert csrf_service.verify_calls == []
    assert revocation_service.calls == []
    assert SESSION_TOKEN not in response.text
    assert "set-cookie" not in response.headers


def test_logout_with_invalid_csrf_does_not_revoke_or_clear_cookie(logout_client):
    error = InvalidCsrfTokenError("wrong csrf detail")
    client, csrf_service, revocation_service = logout_client(
        csrf_service=FakeCsrfService(verify_error=error)
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/api/v1/auth/logout",
        headers={"Origin": TRUSTED_ORIGIN, "X-CSRF-Token": CSRF_TOKEN},
    )

    assert response.status_code == 403
    assert response.json() == INVALID_CSRF_RESPONSE
    assert csrf_service.verify_calls == [(SESSION_TOKEN, CSRF_TOKEN)]
    assert revocation_service.calls == []
    assert str(error) not in response.text
    assert SESSION_TOKEN not in response.text
    assert CSRF_TOKEN not in response.text
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize(
    "error",
    [
        InvalidSessionError("unknown session detail"),
        SessionExpiredError("expired detail"),
        SessionRevokedError("revoked detail"),
        InactiveUserError("inactive user detail"),
        InactiveMembershipError("inactive membership detail"),
    ],
)
def test_logout_treats_invalid_session_from_csrf_verification_as_success(
    logout_client,
    error,
):
    client, csrf_service, revocation_service = logout_client(
        csrf_service=FakeCsrfService(verify_error=error)
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/api/v1/auth/logout",
        headers={"Origin": TRUSTED_ORIGIN, "X-CSRF-Token": CSRF_TOKEN},
    )

    assert response.status_code == 204
    assert response.content == b""
    assert csrf_service.verify_calls == [(SESSION_TOKEN, CSRF_TOKEN)]
    assert revocation_service.calls == []
    assert str(error) not in response.text
    _assert_session_cookie_cleared(response)


@pytest.mark.parametrize(
    "error",
    [
        InvalidSessionError("unknown session detail"),
        SessionExpiredError("expired detail"),
        SessionRevokedError("revoked detail"),
        InactiveUserError("inactive user detail"),
        InactiveMembershipError("inactive membership detail"),
    ],
)
def test_logout_treats_invalid_session_from_revoke_as_success(
    logout_client,
    error,
):
    client, csrf_service, revocation_service = logout_client(
        revocation_service=FakeRevocationService(error=error)
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/api/v1/auth/logout",
        headers={"Origin": TRUSTED_ORIGIN, "X-CSRF-Token": CSRF_TOKEN},
    )

    assert response.status_code == 204
    assert response.content == b""
    assert csrf_service.verify_calls == [(SESSION_TOKEN, CSRF_TOKEN)]
    assert revocation_service.calls == [SESSION_TOKEN]
    assert str(error) not in response.text
    _assert_session_cookie_cleared(response)


@pytest.mark.parametrize(
    "error",
    [
        AuthInfrastructureError("database unavailable"),
        RuntimeError("unexpected revoke backend detail"),
    ],
)
def test_logout_revoke_backend_failures_are_safe_and_do_not_clear_cookie(
    logout_client,
    error,
):
    client, csrf_service, revocation_service = logout_client(
        revocation_service=FakeRevocationService(error=error)
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/api/v1/auth/logout",
        headers={"Origin": TRUSTED_ORIGIN, "X-CSRF-Token": CSRF_TOKEN},
    )

    assert response.status_code == 500
    assert response.json() == _authentication_unavailable_response()
    assert csrf_service.verify_calls == [(SESSION_TOKEN, CSRF_TOKEN)]
    assert revocation_service.calls == [SESSION_TOKEN]
    assert str(error) not in response.text
    assert SESSION_TOKEN not in response.text
    assert CSRF_TOKEN not in response.text
    assert "set-cookie" not in response.headers


def test_logout_openapi_documents_session_cookie_only_and_empty_204(
    logout_client,
):
    client, _, _ = logout_client()

    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/api/v1/auth/logout"]["post"]

    assert operation["security"] == [{"SessionCookieAuth": []}]
    assert "ApiKeyAuth" not in str(operation["security"])
    assert operation["responses"]["204"]["description"] == "Successful Response"
    assert "content" not in operation["responses"]["204"]


def _assert_session_cookie_cleared(response):
    parsed = SimpleCookie()
    parsed.load(response.headers["set-cookie"])
    cookie = parsed[api_security.SESSION_COOKIE_NAME]

    assert cookie.value == ""
    assert cookie["max-age"] == "0"
    assert cookie["path"] == "/"
    assert cookie["domain"] == ""
    assert cookie["httponly"] is True
    assert cookie["samesite"] == "Lax"
    assert cookie["secure"] is True


def _untrusted_origin_response():
    return {
        "error": {
            "code": "UNTRUSTED_ORIGIN",
            "message": "Request origin is not trusted",
        }
    }


def _authentication_unavailable_response():
    return {
        "error": {
            "code": "AUTHENTICATION_UNAVAILABLE",
            "message": "Authentication could not be completed",
        }
    }
