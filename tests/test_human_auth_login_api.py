from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie

import pytest
from fastapi.testclient import TestClient

from app.api import security as api_security
from app.api.v1 import auth as auth_route
from app.api.v1.auth import get_human_auth_service, get_login_rate_limit_service
from app.auth.human.exceptions import (
    AuthInfrastructureError,
    InvalidCredentialsError,
    LoginRateLimitExceededError,
    LoginRateLimitInfrastructureError,
    StaleCredentialsError,
)
from app.auth.human.types import (
    AuthenticatedPrincipal,
    AuthenticatedUserContext,
    CreatedSession,
    LoginRateLimitPermit,
)
from app.domain.enums import MembershipRole
from app.main import app


TRUSTED_ORIGIN = "https://app.example.com"
RAW_API_KEY = "rea_v1_" + ("K" * 22) + "." + ("s" * 43)
CLIENT_IP = "testclient"
PERMIT = LoginRateLimitPermit(attempt_id="opaque-test-permit")
REQUEST_COMPANY_ID = 909


@dataclass
class FakeHumanAuthService:
    principal: AuthenticatedPrincipal
    created_session: CreatedSession
    auth_error: Exception | None = None
    create_error: Exception | None = None
    credential_calls: list[tuple[str, str, int]] | None = None
    session_calls: list[AuthenticatedPrincipal] | None = None
    events: list[str] | None = None

    def __post_init__(self):
        self.credential_calls = []
        self.session_calls = []
        if self.events is None:
            self.events = []

    def authenticate_credentials(
        self,
        *,
        email: str,
        password: str,
        company_id: int,
    ) -> AuthenticatedPrincipal:
        self.events.append("authenticate")
        self.credential_calls.append((email, password, company_id))
        if self.auth_error is not None:
            raise self.auth_error
        return self.principal

    def create_session(
        self,
        *,
        principal: AuthenticatedPrincipal,
    ) -> CreatedSession:
        self.events.append("create_session")
        self.session_calls.append(principal)
        if self.create_error is not None:
            raise self.create_error
        return self.created_session


@dataclass
class FakeLoginRateLimitService:
    permit: LoginRateLimitPermit = PERMIT
    check_error: Exception | None = None
    failure_error: Exception | None = None
    success_error: Exception | None = None
    aborted_error: Exception | None = None
    check_calls: list[tuple[str, str]] | None = None
    failure_calls: list[LoginRateLimitPermit] | None = None
    success_calls: list[LoginRateLimitPermit] | None = None
    aborted_calls: list[LoginRateLimitPermit] | None = None
    events: list[str] | None = None

    def __post_init__(self):
        self.check_calls = []
        self.failure_calls = []
        self.success_calls = []
        self.aborted_calls = []
        if self.events is None:
            self.events = []

    def check_login_rate_limit(
        self,
        *,
        client_ip: str,
        email: str,
    ) -> LoginRateLimitPermit:
        self.events.append("check")
        self.check_calls.append((client_ip, email))
        if self.check_error is not None:
            raise self.check_error
        return self.permit

    def record_login_failure(self, *, permit: LoginRateLimitPermit) -> None:
        self.events.append("failure")
        self.failure_calls.append(permit)
        if self.failure_error is not None:
            raise self.failure_error

    def record_login_success(self, *, permit: LoginRateLimitPermit) -> None:
        self.events.append("success")
        self.success_calls.append(permit)
        if self.success_error is not None:
            raise self.success_error

    def record_login_aborted(self, *, permit: LoginRateLimitPermit) -> None:
        self.events.append("aborted")
        self.aborted_calls.append(permit)
        if self.aborted_error is not None:
            raise self.aborted_error


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

    def factory(
        service: FakeHumanAuthService | None = None,
        rate_limit_service: FakeLoginRateLimitService | None = None,
    ) -> TestClient:
        selected_service = service or _human_auth_service()
        selected_rate_limit_service = (
            rate_limit_service or FakeLoginRateLimitService()
        )
        services.append(selected_service)
        app.dependency_overrides[get_human_auth_service] = lambda: selected_service
        app.dependency_overrides[get_login_rate_limit_service] = (
            lambda: selected_rate_limit_service
        )
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
        json=_login_payload(),
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": 101,
        "company_id": 303,
        "role": "company_admin",
    }
    assert service.credential_calls == [
        ("user@example.com", "correct-password", REQUEST_COMPANY_ID)
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


def test_password_version_is_backend_only_and_absent_from_transport(login_client, caplog):
    service = _human_auth_service()
    service.principal = replace(service.principal, password_version=987654321)
    client = login_client(service)
    response = client.post(
        "/api/v1/auth/login",
        json=_login_payload(),
        headers={"Origin": TRUSTED_ORIGIN},
    )
    assert response.status_code == 200
    assert service.session_calls == [service.principal]
    assert set(response.json()) == {"user_id", "company_id", "role"}
    assert "password_version" not in response.text
    assert "987654321" not in response.text
    assert "password_version" not in str(response.headers)
    assert "987654321" not in str(response.headers)
    assert "password_version" not in str(client.get("/openapi.json").json())
    assert "987654321" not in caplog.text
    assert "correct-password" not in caplog.text


def test_login_does_not_take_password_version_from_client(login_client):
    service = _human_auth_service()
    client = login_client(service)
    response = client.post(
        "/api/v1/auth/login",
        json=_login_payload(password_version=999),
        headers={"Origin": TRUSTED_ORIGIN},
    )
    assert response.status_code == 422
    assert service.credential_calls == []
    assert service.session_calls == []
    assert "password_version" not in str(client.get("/openapi.json").json())


def test_stale_principal_login_error_is_sanitized_and_sets_no_cookie(login_client):
    service = _human_auth_service(
        create_error=StaleCredentialsError("Human authentication unavailable"),
    )
    client = login_client(service)
    response = client.post(
        "/api/v1/auth/login",
        json=_login_payload(),
        headers={"Origin": TRUSTED_ORIGIN},
    )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "AUTHENTICATION_UNAVAILABLE"
    assert "set-cookie" not in response.headers
    assert "password_version" not in response.text


def test_login_response_uses_created_session_context(login_client):
    request_company_id = 999
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
        json=_login_payload(company_id=request_company_id),
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": 404,
        "company_id": 606,
        "role": "salesperson",
    }
    assert response.json()["company_id"] != request_company_id
    assert service.credential_calls == [
        ("user@example.com", "correct-password", request_company_id)
    ]


def test_invalid_login_credentials_return_generic_401(login_client):
    service = _human_auth_service(
        auth_error=InvalidCredentialsError("email does not exist")
    )
    client = login_client(service)

    response = client.post(
        "/api/v1/auth/login",
        json=_login_payload(email="missing@example.com", password="wrong-password"),
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
    assert service.credential_calls == [("missing@example.com", "wrong-password", REQUEST_COMPANY_ID)]
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
        json=_login_payload(),
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
    assert service.credential_calls == [("user@example.com", "correct-password", REQUEST_COMPANY_ID)]
    assert service.session_calls == []


def test_login_create_session_failure_returns_safe_500(login_client):
    service = _human_auth_service(
        create_error=AuthInfrastructureError("session store down")
    )
    client = login_client(service)

    response = client.post(
        "/api/v1/auth/login",
        json=_login_payload(),
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
    assert service.credential_calls == [("user@example.com", "correct-password", REQUEST_COMPANY_ID)]
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
        json=_login_payload(),
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
        json=_login_payload(),
        headers=request_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": 101,
        "company_id": 303,
        "role": "company_admin",
    }
    assert service.credential_calls == [("user@example.com", "correct-password", REQUEST_COMPANY_ID)]
    assert service.session_calls == [service.principal]


def test_login_rejects_both_api_key_and_session_cookie_as_ambiguous(
    login_client,
):
    service = _human_auth_service()
    client = login_client(service)
    client.cookies.set("rea_session", "existing-session-token")

    response = client.post(
        "/api/v1/auth/login",
        json=_login_payload(),
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
    assert set(request_schema["properties"]) == {
        "email",
        "password",
        "company_id",
    }
    assert set(request_schema["required"]) == {
        "email",
        "password",
        "company_id",
    }
    assert request_schema["properties"]["company_id"]["type"] == "integer"
    assert request_schema["properties"]["company_id"]["exclusiveMinimum"] == 0
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


def test_login_rate_limit_records_failure_before_invalid_credentials_response(
    login_client,
):
    events = []
    service = _human_auth_service(
        auth_error=InvalidCredentialsError("unknown email detail"),
        events=events,
    )
    rate_limit_service = FakeLoginRateLimitService(events=events)
    client = login_client(service, rate_limit_service)

    response = client.post(
        "/api/v1/auth/login",
        json=_login_payload(email="missing@example.com", password="wrong-password"),
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 401
    assert response.json() == _invalid_credentials_response()
    assert rate_limit_service.check_calls == [(CLIENT_IP, "missing@example.com")]
    assert service.credential_calls == [("missing@example.com", "wrong-password", REQUEST_COMPANY_ID)]
    assert rate_limit_service.failure_calls == [PERMIT]
    assert rate_limit_service.success_calls == []
    assert rate_limit_service.aborted_calls == []
    assert events == ["check", "authenticate", "failure"]
    assert "missing@example.com" not in response.text
    assert "wrong-password" not in response.text
    assert "unknown email detail" not in response.text


def test_login_rate_limit_records_success_before_session_creation(login_client):
    events = []
    service = _human_auth_service(events=events)
    rate_limit_service = FakeLoginRateLimitService(events=events)
    client = login_client(service, rate_limit_service)

    response = client.post(
        "/api/v1/auth/login",
        json=_login_payload(),
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": 101,
        "company_id": 303,
        "role": "company_admin",
    }
    assert rate_limit_service.check_calls == [(CLIENT_IP, "user@example.com")]
    assert service.credential_calls == [("user@example.com", "correct-password", REQUEST_COMPANY_ID)]
    assert rate_limit_service.success_calls == [PERMIT]
    assert rate_limit_service.failure_calls == []
    assert rate_limit_service.aborted_calls == []
    assert service.session_calls == [service.principal]
    assert events == ["check", "authenticate", "success", "create_session"]
    assert _session_cookie(response).value == service.created_session.token


def test_login_session_creation_failure_keeps_success_outcome(login_client):
    events = []
    error = AuthInfrastructureError("session store down")
    service = _human_auth_service(create_error=error, events=events)
    rate_limit_service = FakeLoginRateLimitService(events=events)
    client = login_client(service, rate_limit_service)

    response = client.post(
        "/api/v1/auth/login",
        json=_login_payload(),
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 500
    assert response.json() == _authentication_unavailable_response()
    assert rate_limit_service.check_calls == [(CLIENT_IP, "user@example.com")]
    assert service.credential_calls == [("user@example.com", "correct-password", REQUEST_COMPANY_ID)]
    assert rate_limit_service.success_calls == [PERMIT]
    assert rate_limit_service.failure_calls == []
    assert rate_limit_service.aborted_calls == []
    assert service.session_calls == [service.principal]
    assert events == ["check", "authenticate", "success", "create_session"]
    assert str(error) not in response.text
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize(
    "error",
    [
        AuthInfrastructureError("credential database down"),
        RuntimeError("unexpected credential backend detail"),
    ],
)
def test_login_authentication_backend_failures_record_aborted(
    login_client,
    error,
):
    events = []
    service = _human_auth_service(auth_error=error, events=events)
    rate_limit_service = FakeLoginRateLimitService(events=events)
    client = login_client(service, rate_limit_service)

    response = client.post(
        "/api/v1/auth/login",
        json=_login_payload(),
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 500
    assert response.json() == _authentication_unavailable_response()
    assert rate_limit_service.check_calls == [(CLIENT_IP, "user@example.com")]
    assert service.credential_calls == [("user@example.com", "correct-password", REQUEST_COMPANY_ID)]
    assert rate_limit_service.aborted_calls == [PERMIT]
    assert rate_limit_service.failure_calls == []
    assert rate_limit_service.success_calls == []
    assert service.session_calls == []
    assert events == ["check", "authenticate", "aborted"]
    assert str(error) not in response.text


def test_login_rate_limit_exceeded_returns_429_without_authentication(login_client):
    error = LoginRateLimitExceededError(retry_after_seconds=37)
    service = _human_auth_service()
    rate_limit_service = FakeLoginRateLimitService(check_error=error)
    client = login_client(service, rate_limit_service)

    response = client.post(
        "/api/v1/auth/login",
        json=_login_payload(email="limited@example.com", password="any-password"),
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "37"
    assert response.json() == {
        "error": {
            "code": "LOGIN_RATE_LIMITED",
            "message": "Too many login attempts. Please try again later.",
        }
    }
    assert rate_limit_service.check_calls == [(CLIENT_IP, "limited@example.com")]
    assert service.credential_calls == []
    assert rate_limit_service.failure_calls == []
    assert rate_limit_service.success_calls == []
    assert rate_limit_service.aborted_calls == []
    assert "limited@example.com" not in response.text
    assert "37" not in response.text


@pytest.mark.parametrize(
    "error",
    [
        LoginRateLimitInfrastructureError("redis private detail"),
        RuntimeError("unexpected limiter private detail"),
    ],
)
def test_login_rate_limit_precheck_failures_are_safe_500(login_client, error):
    service = _human_auth_service()
    rate_limit_service = FakeLoginRateLimitService(check_error=error)
    client = login_client(service, rate_limit_service)

    response = client.post(
        "/api/v1/auth/login",
        json=_login_payload(),
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 500
    assert response.json() == _authentication_unavailable_response()
    assert rate_limit_service.check_calls == [(CLIENT_IP, "user@example.com")]
    assert service.credential_calls == []
    assert rate_limit_service.failure_calls == []
    assert rate_limit_service.success_calls == []
    assert rate_limit_service.aborted_calls == []
    assert str(error) not in response.text


def test_login_failure_recording_failure_returns_safe_500(login_client):
    error = LoginRateLimitInfrastructureError("failure accounting down")
    service = _human_auth_service(
        auth_error=InvalidCredentialsError("wrong password detail")
    )
    rate_limit_service = FakeLoginRateLimitService(failure_error=error)
    client = login_client(service, rate_limit_service)

    response = client.post(
        "/api/v1/auth/login",
        json=_login_payload(password="wrong-password"),
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 500
    assert response.json() == _authentication_unavailable_response()
    assert rate_limit_service.failure_calls == [PERMIT]
    assert rate_limit_service.success_calls == []
    assert rate_limit_service.aborted_calls == []
    assert service.session_calls == []
    assert "wrong password detail" not in response.text
    assert str(error) not in response.text


def test_login_success_recording_failure_prevents_session_creation(login_client):
    error = LoginRateLimitInfrastructureError("success accounting down")
    service = _human_auth_service()
    rate_limit_service = FakeLoginRateLimitService(success_error=error)
    client = login_client(service, rate_limit_service)

    response = client.post(
        "/api/v1/auth/login",
        json=_login_payload(),
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 500
    assert response.json() == _authentication_unavailable_response()
    assert rate_limit_service.success_calls == [PERMIT]
    assert rate_limit_service.failure_calls == []
    assert rate_limit_service.aborted_calls == []
    assert service.session_calls == []
    assert str(error) not in response.text
    assert "set-cookie" not in response.headers


def test_login_aborted_recording_failure_returns_safe_500(login_client):
    auth_error = AuthInfrastructureError("credential store down")
    record_error = LoginRateLimitInfrastructureError("aborted accounting down")
    service = _human_auth_service(auth_error=auth_error)
    rate_limit_service = FakeLoginRateLimitService(aborted_error=record_error)
    client = login_client(service, rate_limit_service)

    response = client.post(
        "/api/v1/auth/login",
        json=_login_payload(),
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 500
    assert response.json() == _authentication_unavailable_response()
    assert rate_limit_service.aborted_calls == [PERMIT]
    assert rate_limit_service.failure_calls == []
    assert rate_limit_service.success_calls == []
    assert service.session_calls == []
    assert str(auth_error) not in response.text
    assert str(record_error) not in response.text


@pytest.mark.parametrize("origin", ["https://attacker.example", None])
def test_login_origin_rejection_happens_before_rate_limit(
    login_client,
    origin,
):
    service = _human_auth_service()
    rate_limit_service = FakeLoginRateLimitService()
    client = login_client(service, rate_limit_service)
    headers = {} if origin is None else {"Origin": origin}

    response = client.post(
        "/api/v1/auth/login",
        json=_login_payload(),
        headers=headers,
    )

    assert response.status_code == 403
    assert response.json() == {
        "error": {
            "code": "UNTRUSTED_ORIGIN",
            "message": "Request origin is not trusted",
        }
    }
    assert rate_limit_service.check_calls == []
    assert service.credential_calls == []
    assert service.session_calls == []


def test_login_ignores_forwarding_headers_for_rate_limit_identity(login_client):
    service = _human_auth_service()
    rate_limit_service = FakeLoginRateLimitService()
    client = login_client(service, rate_limit_service)

    response = client.post(
        "/api/v1/auth/login",
        json=_login_payload(),
        headers={
            "Origin": TRUSTED_ORIGIN,
            "X-Forwarded-For": "203.0.113.222",
            "Forwarded": "for=203.0.113.223",
            "X-Real-IP": "203.0.113.224",
        },
    )

    assert response.status_code == 200
    assert rate_limit_service.check_calls == [(CLIENT_IP, "user@example.com")]
    assert rate_limit_service.check_calls[0][0] != "203.0.113.222"


def test_login_missing_request_client_fails_closed_before_rate_limit(
    monkeypatch,
    login_client,
):
    service = _human_auth_service()
    rate_limit_service = FakeLoginRateLimitService()
    monkeypatch.setattr(
        auth_route,
        "_direct_client_ip",
        lambda request: (_ for _ in ()).throw(
            auth_route._human_auth_unavailable_error()
        ),
    )
    client = login_client(service, rate_limit_service)

    response = client.post(
        "/api/v1/auth/login",
        json=_login_payload(),
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 500
    assert response.json() == _authentication_unavailable_response()
    assert rate_limit_service.check_calls == []
    assert service.credential_calls == []


@pytest.mark.parametrize(
    "error",
    [
        InvalidCredentialsError("unknown email private detail"),
        InvalidCredentialsError("wrong password private detail"),
    ],
)
def test_login_unknown_email_and_wrong_password_remain_identical(
    login_client,
    error,
):
    service = _human_auth_service(auth_error=error)
    rate_limit_service = FakeLoginRateLimitService()
    client = login_client(service, rate_limit_service)

    response = client.post(
        "/api/v1/auth/login",
        json=_login_payload(password="wrong-password"),
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 401
    assert response.json() == _invalid_credentials_response()
    assert rate_limit_service.failure_calls == [PERMIT]
    assert str(error) not in response.text
    assert "wrong-password" not in response.text


def test_login_body_validation_happens_before_rate_limit(login_client):
    service = _human_auth_service()
    rate_limit_service = FakeLoginRateLimitService()
    client = login_client(service, rate_limit_service)

    response = client.post(
        "/api/v1/auth/login",
        json={"email": "user@example.com"},
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 422
    assert rate_limit_service.check_calls == []
    assert service.credential_calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {"email": "user@example.com", "password": "correct-password"},
        {
            "email": "user@example.com",
            "password": "correct-password",
            "company_id": 0,
        },
        {
            "email": "user@example.com",
            "password": "correct-password",
            "company_id": -1,
        },
        {
            "email": "user@example.com",
            "password": "correct-password",
            "company_id": "303",
        },
        {
            "email": "user@example.com",
            "password": "correct-password",
            "company_id": 303.0,
        },
        {
            "email": "user@example.com",
            "password": "correct-password",
            "company_id": True,
        },
    ],
)
def test_invalid_company_id_fails_validation_before_rate_limit(
    login_client,
    payload,
):
    service = _human_auth_service()
    rate_limit_service = FakeLoginRateLimitService()
    client = login_client(service, rate_limit_service)

    response = client.post(
        "/api/v1/auth/login",
        json=payload,
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 422
    assert rate_limit_service.check_calls == []
    assert service.credential_calls == []
    assert service.session_calls == []


@pytest.mark.parametrize(
    "error",
    [
        InvalidCredentialsError("wrong password private detail"),
        InvalidCredentialsError("unauthorized company private detail"),
        InvalidCredentialsError("nonexistent company private detail"),
        InvalidCredentialsError("inactive membership private detail"),
    ],
)
def test_login_membership_failures_remain_generic_401(login_client, error):
    service = _human_auth_service(auth_error=error)
    rate_limit_service = FakeLoginRateLimitService()
    client = login_client(service, rate_limit_service)

    response = client.post(
        "/api/v1/auth/login",
        json=_login_payload(password="wrong-password"),
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 401
    assert response.json() == _invalid_credentials_response()
    assert rate_limit_service.failure_calls == [PERMIT]
    assert str(error) not in response.text
    assert "company" not in response.text.lower()
    assert "membership" not in response.text.lower()


def test_login_rate_limit_key_excludes_company_id(login_client):
    service = _human_auth_service(
        auth_error=InvalidCredentialsError("invalid credentials")
    )
    rate_limit_service = FakeLoginRateLimitService()
    client = login_client(service, rate_limit_service)

    for company_id in (101, 202):
        response = client.post(
            "/api/v1/auth/login",
            json=_login_payload(
                email="same@example.com",
                password="wrong-password",
                company_id=company_id,
            ),
            headers={"Origin": TRUSTED_ORIGIN},
        )
        assert response.status_code == 401

    assert rate_limit_service.check_calls == [
        (CLIENT_IP, "same@example.com"),
        (CLIENT_IP, "same@example.com"),
    ]
    assert service.credential_calls == [
        ("same@example.com", "wrong-password", 101),
        ("same@example.com", "wrong-password", 202),
    ]


def test_public_human_auth_service_forwards_company_id_to_backend(monkeypatch):
    calls = []
    principal = AuthenticatedPrincipal(
        user_id=10,
        membership_id=20,
        company_id=30,
        role=MembershipRole.SALESPERSON,
        password_version=1,
    )

    def authenticate(*, email, password, company_id):
        calls.append((email, password, company_id))
        return principal

    monkeypatch.setattr(auth_route, "authenticate_credentials", authenticate)

    result = auth_route.PublicHumanAuthService().authenticate_credentials(
        email="person@example.com",
        password="private-password",
        company_id=777,
    )

    assert result is principal
    assert calls == [("person@example.com", "private-password", 777)]


def _human_auth_service(
    *,
    created_session: CreatedSession | None = None,
    auth_error: Exception | None = None,
    create_error: Exception | None = None,
    events: list[str] | None = None,
) -> FakeHumanAuthService:
    principal = AuthenticatedPrincipal(
        user_id=1,
        membership_id=2,
        company_id=3,
        role=MembershipRole.SALESPERSON,
        password_version=1,
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
        events=events,
    )


def _invalid_credentials_response():
    return {
        "error": {
            "code": "INVALID_CREDENTIALS",
            "message": "Invalid email or password",
        }
    }


def _authentication_unavailable_response():
    return {
        "error": {
            "code": "AUTHENTICATION_UNAVAILABLE",
            "message": "Authentication could not be completed",
        }
    }


def _login_payload(
    *,
    email: str = "user@example.com",
    password: str = "correct-password",
    company_id=REQUEST_COMPANY_ID,
    **extra,
):
    payload = {
        "email": email,
        "password": password,
        "company_id": company_id,
    }
    payload.update(extra)
    return payload


def _session_cookie(response):
    parsed = SimpleCookie()
    parsed.load(response.headers["set-cookie"])
    return parsed[api_security.SESSION_COOKIE_NAME]


def _schema_for_ref(schema, ref):
    return schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
