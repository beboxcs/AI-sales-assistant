from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.api import security as api_security
from app.api.invitation_delivery import (
    InvitationDeliveryError,
    get_invitation_delivery_service,
)
from app.api.v1.auth import get_invitation_acceptance_service
from app.api.v1.company import get_invitation_creation_service
from app.auth.human.exceptions import (
    AuthInfrastructureError,
    InvitationAuthorizationError,
    InvitationConflictError,
    InvitationInfrastructureError,
    InvalidCsrfTokenError,
    InvalidInvitationError,
    InvalidInvitationPasswordError,
    InvalidSessionError,
)
from app.auth.human.types import (
    AcceptedInvitation,
    AuthenticatedUserContext,
    CreatedInvitation,
)
from app.domain.enums import MembershipRole
from app.main import app


TRUSTED_ORIGIN = "https://app.example.com"
SESSION_TOKEN = "opaque-session-token"
CSRF_TOKEN = "opaque-csrf-token"
INVITATION_TOKEN = "raw-invitation-token-secret"
RAW_API_KEY = "rea_v1_" + ("K" * 22) + "." + ("s" * 43)
EXPIRES_AT = datetime(2026, 9, 18, 12, 30, tzinfo=timezone.utc)


@dataclass
class FakeSessionResolver:
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


@dataclass
class FakeCsrfService:
    error: Exception | None = None
    verify_calls: list[tuple[str, str]] | None = None

    def __post_init__(self):
        self.verify_calls = []

    def issue_csrf_token(self, *, session_token: str) -> str:
        raise AssertionError("invitation creation must not issue CSRF tokens")

    def verify_csrf_token(self, *, session_token: str, csrf_token: str) -> None:
        self.verify_calls.append((session_token, csrf_token))
        if self.error is not None:
            raise self.error


@dataclass
class FakeInvitationCreationService:
    result: CreatedInvitation
    error: Exception | None = None
    calls: list[dict] | None = None

    def __post_init__(self):
        self.calls = []

    def create_invitation(self, **kwargs) -> CreatedInvitation:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


@dataclass
class FakeDeliveryService:
    error: Exception | None = None
    calls: list[dict] | None = None

    def __post_init__(self):
        self.calls = []

    def deliver_invitation(self, **kwargs) -> None:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error


@dataclass
class FakeInvitationAcceptanceService:
    result: AcceptedInvitation
    error: Exception | None = None
    calls: list[tuple[str, str]] | None = None

    def __post_init__(self):
        self.calls = []

    def accept_invitation(self, *, token: str, password: str) -> AcceptedInvitation:
        self.calls.append((token, password))
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def invitation_client(monkeypatch):
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
        session_resolver: FakeSessionResolver | None = None,
        csrf_service: FakeCsrfService | None = None,
        creation_service: FakeInvitationCreationService | None = None,
        delivery_service: FakeDeliveryService | None = None,
        acceptance_service: FakeInvitationAcceptanceService | None = None,
    ):
        selected_session = session_resolver or FakeSessionResolver(_user_context())
        selected_csrf = csrf_service or FakeCsrfService()
        selected_creation = creation_service or FakeInvitationCreationService(
            _created_invitation()
        )
        selected_delivery = delivery_service or FakeDeliveryService()
        selected_acceptance = acceptance_service or FakeInvitationAcceptanceService(
            _accepted_invitation()
        )
        app.dependency_overrides[
            api_security.get_human_session_resolver
        ] = lambda: selected_session
        app.dependency_overrides[
            api_security.get_human_csrf_service
        ] = lambda: selected_csrf
        app.dependency_overrides[
            get_invitation_creation_service
        ] = lambda: selected_creation
        app.dependency_overrides[
            get_invitation_delivery_service
        ] = lambda: selected_delivery
        app.dependency_overrides[
            get_invitation_acceptance_service
        ] = lambda: selected_acceptance
        client = TestClient(app)
        clients.append(client)
        return (
            client,
            selected_session,
            selected_csrf,
            selected_creation,
            selected_delivery,
            selected_acceptance,
        )

    try:
        yield factory
    finally:
        for client in clients:
            client.close()
        app.dependency_overrides = original_overrides
        app.openapi_schema = original_openapi_schema


def test_company_admin_creates_invitation_and_delivers_without_token_response(
    invitation_client,
):
    client, session_resolver, csrf_service, creation_service, delivery_service, _ = (
        invitation_client()
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/api/v1/company/invitations",
        json={"email": "Sales@Example.COM", "role": "salesperson"},
        headers={"Origin": TRUSTED_ORIGIN, "X-CSRF-Token": CSRF_TOKEN},
    )

    assert response.status_code == 201
    assert response.json() == {
        "invitation_id": 123,
        "email": "sales@example.com",
        "role": "salesperson",
        "expires_at": "2026-09-18T12:30:00Z",
    }
    assert session_resolver.calls == [SESSION_TOKEN]
    assert csrf_service.verify_calls == [(SESSION_TOKEN, CSRF_TOKEN)]
    assert creation_service.calls == [
        {
            "company_id": 303,
            "created_by_user_id": 101,
            "email": "Sales@Example.COM",
            "role": MembershipRole.SALESPERSON,
        }
    ]
    assert delivery_service.calls == [
        {
            "recipient_email": "sales@example.com",
            "token": INVITATION_TOKEN,
            "expires_at": EXPIRES_AT,
            "role": MembershipRole.SALESPERSON,
        }
    ]
    assert INVITATION_TOKEN not in response.text
    assert "token" not in response.text
    assert "created_by_user_id" not in response.text


@pytest.mark.parametrize(
    ("headers", "set_cookie", "expected_status", "expected_code"),
    [
        ({}, False, 401, "AUTHENTICATION_REQUIRED"),
        ({"X-API-Key": RAW_API_KEY}, False, 401, "AUTHENTICATION_REQUIRED"),
        ({"X-API-Key": RAW_API_KEY}, True, 400, "AMBIGUOUS_CREDENTIALS"),
    ],
)
def test_invitation_creation_rejects_missing_api_key_or_ambiguous_credentials(
    invitation_client,
    headers,
    set_cookie,
    expected_status,
    expected_code,
):
    client, session_resolver, csrf_service, creation_service, delivery_service, _ = (
        invitation_client()
    )
    if set_cookie:
        client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/api/v1/company/invitations",
        json={"email": "sales@example.com", "role": "salesperson"},
        headers={"Origin": TRUSTED_ORIGIN, "X-CSRF-Token": CSRF_TOKEN, **headers},
    )

    assert response.status_code == expected_status
    assert response.json()["error"]["code"] == expected_code
    assert csrf_service.verify_calls == []
    assert creation_service.calls == []
    assert delivery_service.calls == []
    if expected_code == "AMBIGUOUS_CREDENTIALS":
        assert session_resolver.calls == []
        assert RAW_API_KEY not in response.text


@pytest.mark.parametrize("origin", [None, "https://attacker.example"])
def test_invitation_creation_ambiguous_credentials_win_before_origin(
    invitation_client,
    origin,
):
    client, session_resolver, csrf_service, creation_service, delivery_service, _ = (
        invitation_client()
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)
    headers = {"X-API-Key": RAW_API_KEY, "X-CSRF-Token": CSRF_TOKEN}
    if origin is not None:
        headers["Origin"] = origin

    response = client.post(
        "/api/v1/company/invitations",
        json={"email": "sales@example.com", "role": "salesperson"},
        headers=headers,
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "AMBIGUOUS_CREDENTIALS"
    assert csrf_service.verify_calls == []
    assert session_resolver.calls == []
    assert creation_service.calls == []
    assert delivery_service.calls == []
    assert RAW_API_KEY not in response.text
    assert SESSION_TOKEN not in response.text


@pytest.mark.parametrize("origin", [None, "https://attacker.example"])
def test_invitation_creation_origin_failure_prevents_csrf_and_backend(
    invitation_client,
    origin,
):
    client, session_resolver, csrf_service, creation_service, delivery_service, _ = (
        invitation_client()
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)
    headers = {"X-CSRF-Token": CSRF_TOKEN}
    if origin is not None:
        headers["Origin"] = origin

    response = client.post(
        "/api/v1/company/invitations",
        json={"email": "sales@example.com", "role": "salesperson"},
        headers=headers,
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "UNTRUSTED_ORIGIN"
    assert csrf_service.verify_calls == []
    assert session_resolver.calls == []
    assert creation_service.calls == []
    assert delivery_service.calls == []


@pytest.mark.parametrize("headers", [{"Origin": TRUSTED_ORIGIN}, {"Origin": TRUSTED_ORIGIN, "X-CSRF-Token": CSRF_TOKEN}])
def test_invitation_creation_missing_or_invalid_csrf_prevents_session_resolution(
    invitation_client,
    headers,
):
    error = InvalidCsrfTokenError("wrong csrf detail")
    csrf_service = FakeCsrfService(error=error)
    client, session_resolver, csrf_service, creation_service, delivery_service, _ = (
        invitation_client(csrf_service=csrf_service)
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/api/v1/company/invitations",
        json={"email": "sales@example.com", "role": "salesperson"},
        headers=headers,
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "INVALID_CSRF_TOKEN"
    if "X-CSRF-Token" in headers:
        assert csrf_service.verify_calls == [(SESSION_TOKEN, CSRF_TOKEN)]
        assert str(error) not in response.text
    else:
        assert csrf_service.verify_calls == []
    assert session_resolver.calls == []
    assert creation_service.calls == []
    assert delivery_service.calls == []


def test_invitation_creation_invalid_session_after_browser_verification(
    invitation_client,
):
    error = InvalidSessionError("expired private detail")
    client, session_resolver, csrf_service, creation_service, delivery_service, _ = (
        invitation_client(session_resolver=FakeSessionResolver(_user_context(), error=error))
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/api/v1/company/invitations",
        json={"email": "sales@example.com", "role": "salesperson"},
        headers={"Origin": TRUSTED_ORIGIN, "X-CSRF-Token": CSRF_TOKEN},
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_SESSION"
    assert csrf_service.verify_calls == [(SESSION_TOKEN, CSRF_TOKEN)]
    assert session_resolver.calls == [SESSION_TOKEN]
    assert creation_service.calls == []
    assert delivery_service.calls == []
    assert str(error) not in response.text


def test_invitation_creation_denies_salesperson_after_browser_verification(
    invitation_client,
):
    client, session_resolver, csrf_service, creation_service, delivery_service, _ = (
        invitation_client(
            session_resolver=FakeSessionResolver(
                _user_context(role=MembershipRole.SALESPERSON)
            )
        )
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/api/v1/company/invitations",
        json={"email": "sales@example.com", "role": "salesperson"},
        headers={"Origin": TRUSTED_ORIGIN, "X-CSRF-Token": CSRF_TOKEN},
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "FORBIDDEN"
    assert csrf_service.verify_calls == [(SESSION_TOKEN, CSRF_TOKEN)]
    assert session_resolver.calls == [SESSION_TOKEN]
    assert creation_service.calls == []
    assert delivery_service.calls == []


@pytest.mark.parametrize(
    ("error", "status_code", "code"),
    [
        (InvitationAuthorizationError("creator private detail"), 403, "FORBIDDEN"),
        (InvitationConflictError("membership exists detail"), 409, "INVITATION_CONFLICT"),
        (
            InvitationInfrastructureError("postgres detail"),
            500,
            "AUTHENTICATION_UNAVAILABLE",
        ),
        (RuntimeError("unexpected private detail"), 500, "AUTHENTICATION_UNAVAILABLE"),
    ],
)
def test_invitation_creation_backend_errors_are_sanitized(
    invitation_client,
    error,
    status_code,
    code,
):
    creation_service = FakeInvitationCreationService(
        _created_invitation(),
        error=error,
    )
    client, _, _, creation_service, delivery_service, _ = invitation_client(
        creation_service=creation_service
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/api/v1/company/invitations",
        json={"email": "sales@example.com", "role": "salesperson"},
        headers={"Origin": TRUSTED_ORIGIN, "X-CSRF-Token": CSRF_TOKEN},
    )

    assert response.status_code == status_code
    assert response.json()["error"]["code"] == code
    assert creation_service.calls != []
    assert delivery_service.calls == []
    assert str(error) not in response.text
    assert INVITATION_TOKEN not in response.text


def test_invitation_delivery_failure_is_safe_after_creation(invitation_client):
    error = InvitationDeliveryError("smtp://mailpit private token detail")
    client, _, _, creation_service, delivery_service, _ = invitation_client(
        delivery_service=FakeDeliveryService(error=error)
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.post(
        "/api/v1/company/invitations",
        json={"email": "sales@example.com", "role": "salesperson"},
        headers={"Origin": TRUSTED_ORIGIN, "X-CSRF-Token": CSRF_TOKEN},
    )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "INVITATION_DELIVERY_UNAVAILABLE",
            "message": "Invitation could not be delivered",
        }
    }
    assert creation_service.calls != []
    assert delivery_service.calls != []
    assert str(error) not in response.text
    assert INVITATION_TOKEN not in response.text


def test_invitation_acceptance_succeeds_without_cookie(invitation_client):
    client, _, _, _, _, acceptance_service = invitation_client()

    response = client.post(
        "/api/v1/auth/invitations/accept",
        json={"token": INVITATION_TOKEN, "password": "new password"},
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": 501,
        "company_id": 303,
        "membership_id": 909,
        "role": "salesperson",
    }
    assert acceptance_service.calls == [(INVITATION_TOKEN, "new password")]
    assert INVITATION_TOKEN not in response.text
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize("origin", [None, "https://attacker.example"])
def test_invitation_acceptance_requires_trusted_origin_before_backend(
    invitation_client,
    origin,
):
    client, _, _, _, _, acceptance_service = invitation_client()
    headers = {} if origin is None else {"Origin": origin}

    response = client.post(
        "/api/v1/auth/invitations/accept",
        json={"token": INVITATION_TOKEN, "password": "new password"},
        headers=headers,
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "UNTRUSTED_ORIGIN"
    assert acceptance_service.calls == []
    assert INVITATION_TOKEN not in response.text


@pytest.mark.parametrize("origin", [TRUSTED_ORIGIN, None, "https://attacker.example"])
def test_invitation_acceptance_rejects_ambiguous_credentials_before_origin(
    invitation_client,
    origin,
):
    client, _, _, _, _, acceptance_service = invitation_client()
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)
    headers = {"X-API-Key": RAW_API_KEY}
    if origin is not None:
        headers["Origin"] = origin

    response = client.post(
        "/api/v1/auth/invitations/accept",
        json={"token": INVITATION_TOKEN, "password": "new password"},
        headers=headers,
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "AMBIGUOUS_CREDENTIALS"
    assert acceptance_service.calls == []
    assert RAW_API_KEY not in response.text
    assert INVITATION_TOKEN not in response.text


@pytest.mark.parametrize(
    ("error", "status_code", "code"),
    [
        (InvalidInvitationError("expired private state"), 400, "INVALID_INVITATION"),
        (
            InvalidInvitationPasswordError("argon private detail"),
            422,
            "INVALID_INVITATION_PASSWORD",
        ),
        (InvitationConflictError("existing user detail"), 409, "INVITATION_CONFLICT"),
        (
            InvitationInfrastructureError("postgres private detail"),
            500,
            "AUTHENTICATION_UNAVAILABLE",
        ),
        (RuntimeError("unexpected private detail"), 500, "AUTHENTICATION_UNAVAILABLE"),
    ],
)
def test_invitation_acceptance_errors_are_sanitized(
    invitation_client,
    error,
    status_code,
    code,
):
    client, _, _, _, _, acceptance_service = invitation_client(
        acceptance_service=FakeInvitationAcceptanceService(
            _accepted_invitation(),
            error=error,
        )
    )

    response = client.post(
        "/api/v1/auth/invitations/accept",
        json={"token": INVITATION_TOKEN, "password": "new password"},
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == status_code
    assert response.json()["error"]["code"] == code
    assert acceptance_service.calls == [(INVITATION_TOKEN, "new password")]
    assert str(error) not in response.text
    assert INVITATION_TOKEN not in response.text
    assert "new password" not in response.text


def test_invitation_openapi_security_and_schemas(invitation_client):
    client, *_ = invitation_client()

    schema = client.get("/openapi.json").json()
    create_operation = schema["paths"]["/api/v1/company/invitations"]["post"]
    accept_operation = schema["paths"]["/api/v1/auth/invitations/accept"]["post"]
    create_response_ref = create_operation["responses"]["201"]["content"][
        "application/json"
    ]["schema"]["$ref"]
    accept_request_ref = accept_operation["requestBody"]["content"][
        "application/json"
    ]["schema"]["$ref"]
    accept_response_ref = accept_operation["responses"]["200"]["content"][
        "application/json"
    ]["schema"]["$ref"]

    assert create_operation["security"] == [{"SessionCookieAuth": []}]
    assert "ApiKeyAuth" not in str(create_operation["security"])
    assert "security" not in accept_operation
    assert "token" not in str(_schema_for_ref(schema, create_response_ref))
    assert set(_schema_for_ref(schema, accept_request_ref)["properties"]) == {
        "token",
        "password",
    }
    assert "password" not in str(_schema_for_ref(schema, accept_response_ref))
    assert "token" not in str(_schema_for_ref(schema, accept_response_ref))


def _user_context(
    *,
    role: MembershipRole = MembershipRole.COMPANY_ADMIN,
) -> AuthenticatedUserContext:
    return AuthenticatedUserContext(
        user_id=101,
        membership_id=202,
        company_id=303,
        role=role,
        session_id=404,
    )


def _created_invitation() -> CreatedInvitation:
    return CreatedInvitation(
        invitation_id=123,
        company_id=303,
        normalized_email="sales@example.com",
        role=MembershipRole.SALESPERSON,
        token=INVITATION_TOKEN,
        expires_at=EXPIRES_AT,
    )


def _accepted_invitation() -> AcceptedInvitation:
    return AcceptedInvitation(
        user_id=501,
        company_id=303,
        membership_id=909,
        role=MembershipRole.SALESPERSON,
    )


def _schema_for_ref(schema, ref):
    return schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
