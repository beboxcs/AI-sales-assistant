from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie

import pytest
from fastapi.testclient import TestClient

from app.api import security as api_security
from app.api.v1.auth import get_human_auth_service
from app.auth.human.exceptions import (
    AuthInfrastructureError,
    InvalidCredentialsError,
)
from app.auth.human.types import (
    AuthenticatedPrincipal,
    AuthenticatedUserContext,
    CreatedSession,
)
from app.domain.enums import MembershipRole
from app.main import app


TRUSTED_ORIGIN = "https://app.example.com"
RAW_API_KEY = "rea_v1_" + ("K" * 22) + "." + ("s" * 43)


@dataclass
class FakeHumanAuthService:
    principal: AuthenticatedPrincipal
    created_session: CreatedSession
    auth_error: Exception | None = None
    create_error: Exception | None = None
    credential_calls: list[tuple[str, str]] | None = None
    session_calls: list[AuthenticatedPrincipal] | None = None

    def __post_init__(self):
        self.credential_calls = []
        self.session_calls = []

    def authenticate_credentials(
        self,
        *,
        email: str,
        password: str,
    ) -> AuthenticatedPrincipal:
        self.credential_calls.append((email, password))
        if self.auth_error is not None:
            raise self.auth_error
        return self.principal

    def create_session(
        self,
        *,
        principal: AuthenticatedPrincipal,
    ) -> CreatedSession:
        self.session_calls.append(principal)
        if self.create_error is not None:
            raise self.create_error
        return self.created_session


@pytest.fixture
def login_client(monkeypatch):
    original_overrides = app.dependency_overrides.copy()
    original_openapi_schema = app.openapi_schema
    clients = []
    services = []

    monkeypatch.setattr(api_security.settings, "APP_ENV", "production")
    monkeypatch.setattr(
        api_security.settings,
        "TRUSTED_BROWSER_ORIGINS",
        (TRUSTED_ORIGIN,),
    )
    app.openapi_schema = None

    def factory(service: FakeHumanAuthService | None = None) -> TestClient:
        selected_service = service or _human_auth_service()
        services.append(selected_service)
        app.dependency_overrides[get_human_auth_service] = lambda: selected_service
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


def test_login_with_trusted_origin_and_valid_credentials_succeeds(login_client):
    service = _human_auth_service()
    client = login_client(service)

    response = client.post(
        "/api/v1/auth/login",
        json={"email": "user@example.com", "password": "correct-password"},
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": 101,
        "company_id": 303,
        "role": "company_admin",
    }
    assert service.credential_calls == [
        ("user@example.com", "correct-password")
    ]
    assert service.session_calls == [service.principal]

    cookie = _session_cookie(response)
    assert cookie.value == service.created_session.token
    assert cookie["httponly"] is True
    assert cookie["samesite"] == "Lax"
    assert cookie["path"] == "/"
    assert cookie["domain"] == ""
    assert cookie["secure"] is True
    assert 0 < int(cookie["max-age"]) <= api_security.SESSION_COOKIE_MAX_AGE_SECONDS
    assert service.created_session.token not in response.text
    assert str(service.created_session.context.session_id) not in response.text
    assert str(service.created_session.context.membership_id) not in response.text


def test_login_response_uses_created_session_context(login_client):
    context = AuthenticatedUserContext(
        user_id=404,
        membership_id=505,
        company_id=606,
        role=MembershipRole.SALESPERSON,
        session_id=707,
    )
    service = _human_auth_service(
        created_session=CreatedSession(
            token="different-opaque-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            context=context,
        )
    )
    client = login_client(service)

    response = client.post(
        "/api/v1/auth/login",
        json={"email": "user@example.com", "password": "correct-password"},
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": 404,
        "company_id": 606,
        "role": "salesperson",
    }


def test_invalid_login_credentials_return_generic_401(login_client):
    service = _human_auth_service(
        auth_error=InvalidCredentialsError("email does not exist")
    )
    client = login_client(service)

    response = client.post(
        "/api/v1/auth/login",
        json={"email": "missing@example.com", "password": "wrong-password"},
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "code": "INVALID_CREDENTIALS",
            "message": "Invalid email or password",
        }
    }
    assert "email does not exist" not in response.text
    assert "wrong-password" not in response.text
    assert service.credential_calls == [("missing@example.com", "wrong-password")]
    assert service.session_calls == []


@pytest.mark.parametrize(
    "error",
    [
        AuthInfrastructureError("database down"),
        RuntimeError("unexpected human auth detail"),
    ],
)
def test_login_human_auth_failures_return_safe_500(login_client, error):
    service = _human_auth_service(auth_error=error)
    client = login_client(service)

    response = client.post(
        "/api/v1/auth/login",
        json={"email": "user@example.com", "password": "correct-password"},
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "AUTHENTICATION_UNAVAILABLE",
            "message": "Authentication could not be completed",
        }
    }
    assert str(error) not in response.text
    assert "correct-password" not in response.text
    assert service.credential_calls == [("user@example.com", "correct-password")]
    assert service.session_calls == []


def test_login_create_session_failure_returns_safe_500(login_client):
    service = _human_auth_service(
        create_error=AuthInfrastructureError("session store down")
    )
    client = login_client(service)

    response = client.post(
        "/api/v1/auth/login",
        json={"email": "user@example.com", "password": "correct-password"},
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "AUTHENTICATION_UNAVAILABLE",
            "message": "Authentication could not be completed",
        }
    }
    assert "session store down" not in response.text
    assert service.credential_calls == [("user@example.com", "correct-password")]
    assert service.session_calls == [service.principal]


@pytest.mark.parametrize("origin", ["https://attacker.example", None])
def test_login_untrusted_or_missing_origin_prevents_auth_service_calls(
    login_client,
    origin,
):
    service = _human_auth_service()
    client = login_client(service)
    headers = {} if origin is None else {"Origin": origin}

    response = client.post(
        "/api/v1/auth/login",
        json={"email": "user@example.com", "password": "correct-password"},
        headers=headers,
    )

    assert response.status_code == 403
    assert response.json() == {
        "error": {
            "code": "UNTRUSTED_ORIGIN",
            "message": "Request origin is not trusted",
        }
    }
    assert service.credential_calls == []
    assert service.session_calls == []


@pytest.mark.parametrize(
    ("headers", "set_session_cookie"),
    [
        ({"X-API-Key": RAW_API_KEY}, False),
        ({}, True),
        ({}, False),
    ],
)
def test_login_non_ambiguous_credential_presentations_are_allowed(
    login_client,
    headers,
    set_session_cookie,
):
    service = _human_auth_service()
    client = login_client(service)
    request_headers = {"Origin": TRUSTED_ORIGIN, **headers}
    if set_session_cookie:
        client.cookies.set("rea_session", "existing-session-token")

    response = client.post(
        "/api/v1/auth/login",
        json={"email": "user@example.com", "password": "correct-password"},
        headers=request_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": 101,
        "company_id": 303,
        "role": "company_admin",
    }
    assert service.credential_calls == [("user@example.com", "correct-password")]
    assert service.session_calls == [service.principal]


def test_login_rejects_both_api_key_and_session_cookie_as_ambiguous(
    login_client,
):
    service = _human_auth_service()
    client = login_client(service)
    client.cookies.set("rea_session", "existing-session-token")

    response = client.post(
        "/api/v1/auth/login",
        json={"email": "user@example.com", "password": "correct-password"},
        headers={
            "Origin": TRUSTED_ORIGIN,
            "X-API-Key": RAW_API_KEY,
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "AMBIGUOUS_CREDENTIALS",
            "message": "Use exactly one authentication credential type",
        }
    }
    assert RAW_API_KEY not in response.text
    assert "existing-session-token" not in response.text
    assert service.credential_calls == []
    assert service.session_calls == []


def test_login_openapi_documents_safe_request_and_response(login_client):
    client = login_client()

    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/api/v1/auth/login"]["post"]
    request_ref = operation["requestBody"]["content"]["application/json"][
        "schema"
    ]["$ref"]
    response_ref = operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ]["$ref"]
    request_schema = _schema_for_ref(schema, request_ref)
    response_schema = _schema_for_ref(schema, response_ref)

    assert "security" not in operation
    assert set(request_schema["properties"]) == {"email", "password"}
    assert set(response_schema["properties"]) == {
        "user_id",
        "company_id",
        "role",
    }
    response_text = str(response_schema)
    assert "token" not in response_text
    assert "session_id" not in response_text
    assert "membership_id" not in response_text
    assert "password" not in response_text


def _human_auth_service(
    *,
    created_session: CreatedSession | None = None,
    auth_error: Exception | None = None,
    create_error: Exception | None = None,
) -> FakeHumanAuthService:
    principal = AuthenticatedPrincipal(
        user_id=1,
        membership_id=2,
        company_id=3,
        role=MembershipRole.SALESPERSON,
    )
    context = AuthenticatedUserContext(
        user_id=101,
        membership_id=202,
        company_id=303,
        role=MembershipRole.COMPANY_ADMIN,
        session_id=404,
    )
    return FakeHumanAuthService(
        principal=principal,
        created_session=created_session
        or CreatedSession(
            token="opaque-created-session-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            context=context,
        ),
        auth_error=auth_error,
        create_error=create_error,
    )


def _session_cookie(response):
    parsed = SimpleCookie()
    parsed.load(response.headers["set-cookie"])
    return parsed[api_security.SESSION_COOKIE_NAME]


def _schema_for_ref(schema, ref):
    return schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
