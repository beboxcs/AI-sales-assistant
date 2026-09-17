from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.api import security as api_security
from app.api.password_reset_delivery import (
    PasswordResetDeliveryError,
    get_password_reset_delivery_service,
)
from app.api.v1.auth import get_password_reset_service
from app.auth.human.exceptions import (
    InvalidPasswordResetPasswordError,
    InvalidPasswordResetTokenError,
    PasswordResetInfrastructureError,
)
from app.auth.human.types import CreatedPasswordReset, PasswordResetResult
from app.main import app


TRUSTED_ORIGIN = "https://app.example.com"
SESSION_TOKEN = "opaque-session-token"
RAW_API_KEY = "rea_v1_" + ("K" * 22) + "." + ("s" * 43)
RAW_EMAIL = " Reset@Example.COM "
NORMALIZED_EMAIL = "reset@example.com"
RESET_TOKEN = "raw-reset-token-secret"
NEW_PASSWORD = "new password"
EXPIRES_AT = datetime(2026, 9, 18, 12, 30, tzinfo=timezone.utc)
PUBLIC_REQUEST_RESPONSE = {
    "message": "If the account is eligible, password reset instructions will be sent."
}


@dataclass
class FakePasswordResetService:
    request_result: CreatedPasswordReset | None = None
    request_error: Exception | None = None
    reset_result: PasswordResetResult | None = None
    reset_error: Exception | None = None
    request_calls: list[str] | None = None
    reset_calls: list[tuple[str, str]] | None = None

    def __post_init__(self):
        self.request_calls = []
        self.reset_calls = []

    def request_password_reset(self, *, email: str) -> CreatedPasswordReset | None:
        self.request_calls.append(email)
        if self.request_error is not None:
            raise self.request_error
        return self.request_result

    def reset_password(self, *, token: str, new_password: str) -> PasswordResetResult:
        self.reset_calls.append((token, new_password))
        if self.reset_error is not None:
            raise self.reset_error
        return self.reset_result or PasswordResetResult(user_id=101)


@dataclass
class FakePasswordResetDeliveryService:
    error: Exception | None = None
    calls: list[dict] | None = None

    def __post_init__(self):
        self.calls = []

    def deliver_password_reset(self, **kwargs) -> None:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error


@pytest.fixture
def password_reset_client(monkeypatch):
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
        reset_service: FakePasswordResetService | None = None,
        delivery_service: FakePasswordResetDeliveryService | None = None,
    ):
        selected_reset = reset_service or FakePasswordResetService(
            request_result=_created_password_reset()
        )
        selected_delivery = delivery_service or FakePasswordResetDeliveryService()
        app.dependency_overrides[get_password_reset_service] = lambda: selected_reset
        app.dependency_overrides[
            get_password_reset_delivery_service
        ] = lambda: selected_delivery
        client = TestClient(app)
        clients.append(client)
        return client, selected_reset, selected_delivery

    try:
        yield factory
    finally:
        for client in clients:
            client.close()
        app.dependency_overrides = original_overrides
        app.openapi_schema = original_openapi_schema


def test_password_reset_request_for_eligible_account_delivers_token_once(
    password_reset_client,
):
    client, reset_service, delivery_service = password_reset_client()

    response = client.post(
        "/api/v1/auth/password-reset/request",
        json={"email": RAW_EMAIL},
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 202
    assert response.json() == PUBLIC_REQUEST_RESPONSE
    assert reset_service.request_calls == [RAW_EMAIL]
    assert delivery_service.calls == [
        {
            "recipient_email": NORMALIZED_EMAIL,
            "token": RESET_TOKEN,
            "expires_at": EXPIRES_AT,
        }
    ]
    assert RESET_TOKEN not in response.text
    assert NORMALIZED_EMAIL not in response.text
    assert str(EXPIRES_AT.year) not in response.text
    assert "user_id" not in response.text
    assert "membership" not in response.text
    assert "set-cookie" not in response.headers


def test_password_reset_request_unknown_account_matches_eligible_public_response(
    password_reset_client,
):
    eligible_client, _, _ = password_reset_client()
    eligible_response = eligible_client.post(
        "/api/v1/auth/password-reset/request",
        json={"email": RAW_EMAIL},
        headers={"Origin": TRUSTED_ORIGIN},
    )

    unknown_service = FakePasswordResetService(request_result=None)
    unknown_client, reset_service, delivery_service = password_reset_client(
        reset_service=unknown_service
    )
    unknown_response = unknown_client.post(
        "/api/v1/auth/password-reset/request",
        json={"email": RAW_EMAIL},
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert unknown_response.status_code == 202
    assert unknown_response.json() == PUBLIC_REQUEST_RESPONSE
    assert unknown_response.text == eligible_response.text
    assert reset_service.request_calls == [RAW_EMAIL]
    assert delivery_service.calls == []
    assert "set-cookie" not in unknown_response.headers


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"email": RAW_EMAIL, "user_id": 101},
        {"email": RAW_EMAIL, "company_id": 303},
        {"email": RAW_EMAIL, "password": "secret"},
    ],
)
def test_password_reset_request_rejects_malformed_or_extra_body(
    password_reset_client,
    body,
):
    client, reset_service, delivery_service = password_reset_client()

    response = client.post(
        "/api/v1/auth/password-reset/request",
        json=body,
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 422
    assert reset_service.request_calls == []
    assert delivery_service.calls == []


@pytest.mark.parametrize("origin", [None, "https://attacker.example"])
def test_password_reset_request_requires_trusted_origin_before_backend(
    password_reset_client,
    origin,
):
    client, reset_service, delivery_service = password_reset_client()
    headers = {} if origin is None else {"Origin": origin}

    response = client.post(
        "/api/v1/auth/password-reset/request",
        json={"email": RAW_EMAIL},
        headers=headers,
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "UNTRUSTED_ORIGIN"
    assert reset_service.request_calls == []
    assert delivery_service.calls == []
    assert RAW_EMAIL not in response.text


@pytest.mark.parametrize("origin", [None, "https://attacker.example"])
def test_password_reset_request_ambiguous_credentials_win_before_origin(
    password_reset_client,
    origin,
):
    client, reset_service, delivery_service = password_reset_client()
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)
    headers = {"X-API-Key": RAW_API_KEY}
    if origin is not None:
        headers["Origin"] = origin

    response = client.post(
        "/api/v1/auth/password-reset/request",
        json={"email": RAW_EMAIL},
        headers=headers,
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "AMBIGUOUS_CREDENTIALS"
    assert reset_service.request_calls == []
    assert delivery_service.calls == []
    assert RAW_API_KEY not in response.text
    assert SESSION_TOKEN not in response.text


@pytest.mark.parametrize(
    "error",
    [
        PasswordResetInfrastructureError("postgres private detail"),
        RuntimeError("unexpected private detail"),
    ],
)
def test_password_reset_request_backend_failures_are_sanitized(
    password_reset_client,
    error,
):
    reset_service = FakePasswordResetService(request_error=error)
    client, reset_service, delivery_service = password_reset_client(
        reset_service=reset_service
    )

    response = client.post(
        "/api/v1/auth/password-reset/request",
        json={"email": RAW_EMAIL},
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "AUTHENTICATION_UNAVAILABLE"
    assert reset_service.request_calls == [RAW_EMAIL]
    assert delivery_service.calls == []
    assert str(error) not in response.text


def test_password_reset_request_delivery_failure_is_sanitized(
    password_reset_client,
):
    error = PasswordResetDeliveryError("smtp private detail raw-reset-token-secret")
    client, reset_service, delivery_service = password_reset_client(
        delivery_service=FakePasswordResetDeliveryService(error=error)
    )

    response = client.post(
        "/api/v1/auth/password-reset/request",
        json={"email": RAW_EMAIL},
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "PASSWORD_RESET_DELIVERY_UNAVAILABLE",
            "message": "Password reset delivery is unavailable",
        }
    }
    assert reset_service.request_calls == [RAW_EMAIL]
    assert delivery_service.calls != []
    assert str(error) not in response.text
    assert RESET_TOKEN not in response.text


def test_password_reset_confirm_success_calls_backend_once_without_cookie_mutation(
    password_reset_client,
):
    client, reset_service, _ = password_reset_client()

    response = client.post(
        "/api/v1/auth/password-reset/confirm",
        json={"token": RESET_TOKEN, "new_password": NEW_PASSWORD},
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 204
    assert response.content == b""
    assert reset_service.reset_calls == [(RESET_TOKEN, NEW_PASSWORD)]
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize(
    ("error", "status_code", "code", "message"),
    [
        (
            InvalidPasswordResetTokenError("expired private detail"),
            400,
            "INVALID_PASSWORD_RESET_TOKEN",
            "Password reset request is invalid or unavailable",
        ),
        (
            InvalidPasswordResetPasswordError("argon policy private detail"),
            422,
            "INVALID_PASSWORD_RESET_PASSWORD",
            "Password reset password is invalid",
        ),
        (
            PasswordResetInfrastructureError("postgres private detail"),
            500,
            "AUTHENTICATION_UNAVAILABLE",
            "Authentication could not be completed",
        ),
        (
            RuntimeError("unexpected private detail"),
            500,
            "AUTHENTICATION_UNAVAILABLE",
            "Authentication could not be completed",
        ),
    ],
)
def test_password_reset_confirm_errors_are_sanitized(
    password_reset_client,
    error,
    status_code,
    code,
    message,
):
    reset_service = FakePasswordResetService(reset_error=error)
    client, reset_service, _ = password_reset_client(reset_service=reset_service)

    response = client.post(
        "/api/v1/auth/password-reset/confirm",
        json={"token": RESET_TOKEN, "new_password": NEW_PASSWORD},
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == status_code
    assert response.json() == {"error": {"code": code, "message": message}}
    assert reset_service.reset_calls == [(RESET_TOKEN, NEW_PASSWORD)]
    assert str(error) not in response.text
    assert RESET_TOKEN not in response.text
    assert NEW_PASSWORD not in response.text
    assert "password_version" not in response.text


@pytest.mark.parametrize("origin", [None, "https://attacker.example"])
def test_password_reset_confirm_requires_trusted_origin_before_backend(
    password_reset_client,
    origin,
):
    client, reset_service, _ = password_reset_client()
    headers = {} if origin is None else {"Origin": origin}

    response = client.post(
        "/api/v1/auth/password-reset/confirm",
        json={"token": RESET_TOKEN, "new_password": NEW_PASSWORD},
        headers=headers,
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "UNTRUSTED_ORIGIN"
    assert reset_service.reset_calls == []
    assert RESET_TOKEN not in response.text
    assert NEW_PASSWORD not in response.text


@pytest.mark.parametrize("origin", [None, "https://attacker.example"])
def test_password_reset_confirm_ambiguous_credentials_win_before_origin(
    password_reset_client,
    origin,
):
    client, reset_service, _ = password_reset_client()
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)
    headers = {"X-API-Key": RAW_API_KEY}
    if origin is not None:
        headers["Origin"] = origin

    response = client.post(
        "/api/v1/auth/password-reset/confirm",
        json={"token": RESET_TOKEN, "new_password": NEW_PASSWORD},
        headers=headers,
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "AMBIGUOUS_CREDENTIALS"
    assert reset_service.reset_calls == []
    assert RAW_API_KEY not in response.text
    assert SESSION_TOKEN not in response.text


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"token": RESET_TOKEN, "new_password": NEW_PASSWORD, "email": RAW_EMAIL},
        {"token": RESET_TOKEN, "new_password": NEW_PASSWORD, "user_id": 101},
        {
            "token": RESET_TOKEN,
            "new_password": NEW_PASSWORD,
            "password_version": 3,
        },
    ],
)
def test_password_reset_confirm_rejects_malformed_or_extra_body(
    password_reset_client,
    body,
):
    client, reset_service, _ = password_reset_client()

    response = client.post(
        "/api/v1/auth/password-reset/confirm",
        json=body,
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 422
    assert reset_service.reset_calls == []


def test_password_reset_openapi_security_and_schemas(password_reset_client):
    client, *_ = password_reset_client()

    schema = client.get("/openapi.json").json()
    request_operation = schema["paths"]["/api/v1/auth/password-reset/request"][
        "post"
    ]
    confirm_operation = schema["paths"]["/api/v1/auth/password-reset/confirm"][
        "post"
    ]
    request_ref = request_operation["requestBody"]["content"]["application/json"][
        "schema"
    ]["$ref"]
    request_response_ref = request_operation["responses"]["202"]["content"][
        "application/json"
    ]["schema"]["$ref"]
    confirm_ref = confirm_operation["requestBody"]["content"]["application/json"][
        "schema"
    ]["$ref"]

    assert "security" not in request_operation
    assert "security" not in confirm_operation
    assert set(_schema_for_ref(schema, request_ref)["properties"]) == {"email"}
    assert set(_schema_for_ref(schema, confirm_ref)["properties"]) == {
        "token",
        "new_password",
    }
    assert "token" not in str(_schema_for_ref(schema, request_response_ref))
    assert "user_id" not in str(_schema_for_ref(schema, request_response_ref))
    assert "password_version" not in str(_schema_for_ref(schema, request_response_ref))


def _created_password_reset() -> CreatedPasswordReset:
    return CreatedPasswordReset(
        token=RESET_TOKEN,
        normalized_email=NORMALIZED_EMAIL,
        expires_at=EXPIRES_AT,
    )


def _schema_for_ref(schema, ref):
    return schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
