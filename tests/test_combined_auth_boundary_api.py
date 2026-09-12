from dataclasses import dataclass

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api import security as api_security
from app.api.errors import APIError, register_exception_handlers
from app.auth.contracts import TrustedTenantContext
from app.auth.human.exceptions import (
    AuthInfrastructureError,
    InvalidSessionError,
)
from app.auth.human.types import AuthenticatedUserContext
from app.domain.enums import MembershipRole


RAW_API_KEY_A = "rea_v1_" + ("A" * 22) + "." + ("a" * 43)
RAW_API_KEY_B = "rea_v1_" + ("B" * 22) + "." + ("b" * 43)
SESSION_TOKEN = "opaque-session-token"
SECOND_SESSION_TOKEN = "second-opaque-session-token"


@dataclass
class FakeApiKeyTenantResolver:
    company_id: int = 101
    error: Exception | None = None
    calls: list[str] | None = None

    def __post_init__(self):
        self.calls = []

    def __call__(self, *, raw_api_key: str) -> TrustedTenantContext:
        self.calls.append(raw_api_key)
        if self.error is not None:
            raise self.error
        return TrustedTenantContext(company_id=self.company_id, credential_id=7)


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
def combined_client(monkeypatch):
    test_app = FastAPI()
    register_exception_handlers(test_app)
    downstream_calls = []

    @test_app.get("/combined")
    def combined(
        identity: api_security.TrustedRequestIdentity = Depends(
            api_security.get_trusted_request_identity
        ),
    ):
        downstream_calls.append(identity)
        return {
            "company_id": identity.company_id,
            "auth_method": identity.auth_method,
            "human_user_id": (
                identity.human_user.user_id
                if identity.human_user is not None
                else None
            ),
        }

    @test_app.get("/company")
    def company(
        company_id: int = Depends(api_security.get_trusted_request_company_id),
    ):
        return {"company_id": company_id}

    @test_app.post("/login")
    def login_probe():
        return {"status": "public"}

    @test_app.get("/health")
    def health_probe():
        return {"status": "ok"}

    @test_app.get("/ready")
    def readiness_probe():
        return {"status": "ready"}

    def factory(
        *,
        api_resolver: FakeApiKeyTenantResolver | None = None,
        session_resolver: FakeHumanSessionResolver | None = None,
    ):
        selected_api_resolver = api_resolver or FakeApiKeyTenantResolver()
        selected_session_resolver = session_resolver or _session_resolver()
        test_app.dependency_overrides[
            api_security.get_api_key_tenant_resolver
        ] = lambda: selected_api_resolver
        test_app.dependency_overrides[
            api_security.get_human_session_resolver
        ] = lambda: selected_session_resolver
        test_app.openapi_schema = None
        return (
            TestClient(test_app),
            selected_api_resolver,
            selected_session_resolver,
            downstream_calls,
        )

    monkeypatch.setattr(api_security.settings, "APP_ENV", "local")
    monkeypatch.setattr(api_security.settings, "DEV_COMPANY_ID", 77)

    yield factory

    test_app.dependency_overrides = {}


def test_api_key_only_authenticates_and_preserves_company_id(combined_client):
    client, api_resolver, session_resolver, downstream_calls = combined_client(
        api_resolver=FakeApiKeyTenantResolver(company_id=909),
    )

    response = client.get("/combined", headers={"X-API-Key": RAW_API_KEY_A})

    assert response.status_code == 200
    assert response.json() == {
        "company_id": 909,
        "auth_method": "api_key",
        "human_user_id": None,
    }
    assert api_resolver.calls == [RAW_API_KEY_A]
    assert session_resolver.calls == []
    assert downstream_calls[0].company_id == 909
    assert downstream_calls[0].human_user is None


def test_human_session_only_resolves_and_preserves_company_id(combined_client):
    session_resolver = _session_resolver(company_id=808, user_id=707)
    client, api_resolver, session_resolver, downstream_calls = combined_client(
        session_resolver=session_resolver,
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.get("/combined")

    assert response.status_code == 200
    assert response.json() == {
        "company_id": 808,
        "auth_method": "human_session",
        "human_user_id": 707,
    }
    assert api_resolver.calls == []
    assert session_resolver.calls == [SESSION_TOKEN]
    assert downstream_calls[0].company_id == 808
    assert downstream_calls[0].human_user is session_resolver.context


def test_both_credentials_are_ambiguous_before_backend_calls(combined_client):
    client, api_resolver, session_resolver, downstream_calls = combined_client()
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.get("/combined", headers={"X-API-Key": RAW_API_KEY_A})

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "AMBIGUOUS_CREDENTIALS",
            "message": "Use exactly one authentication credential type",
        }
    }
    assert RAW_API_KEY_A not in response.text
    assert SESSION_TOKEN not in response.text
    assert api_resolver.calls == []
    assert session_resolver.calls == []
    assert downstream_calls == []


def test_neither_credential_requires_authentication_without_dev_fallback(
    combined_client,
):
    client, api_resolver, session_resolver, downstream_calls = combined_client()

    response = client.get("/combined")

    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "code": "AUTHENTICATION_REQUIRED",
            "message": "Authentication is required",
        }
    }
    assert api_resolver.calls == []
    assert session_resolver.calls == []
    assert downstream_calls == []


def test_invalid_api_key_uses_existing_generic_b1_error(combined_client):
    client, api_resolver, session_resolver, downstream_calls = combined_client(
        api_resolver=FakeApiKeyTenantResolver(
            error=APIError(
                status_code=401,
                code="INVALID_API_CREDENTIAL",
                message="Invalid or unavailable API credential",
            )
        )
    )

    response = client.get("/combined", headers={"X-API-Key": RAW_API_KEY_A})

    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "code": "INVALID_API_CREDENTIAL",
            "message": "Invalid or unavailable API credential",
        }
    }
    assert RAW_API_KEY_A not in response.text
    assert api_resolver.calls == [RAW_API_KEY_A]
    assert session_resolver.calls == []
    assert downstream_calls == []


def test_invalid_session_uses_generic_human_session_error(combined_client):
    client, api_resolver, session_resolver, downstream_calls = combined_client(
        session_resolver=_session_resolver(
            error=InvalidSessionError("token lookup detail")
        )
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.get("/combined")

    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "code": "INVALID_SESSION",
            "message": "Invalid or unavailable session",
        }
    }
    assert "token lookup detail" not in response.text
    assert SESSION_TOKEN not in response.text
    assert api_resolver.calls == []
    assert session_resolver.calls == [SESSION_TOKEN]
    assert downstream_calls == []


@pytest.mark.parametrize(
    ("api_error", "session_error", "headers", "set_cookie"),
    [
        (RuntimeError("api backend exploded"), None, {"X-API-Key": RAW_API_KEY_A}, False),
        (None, AuthInfrastructureError("session db down"), {}, True),
        (None, RuntimeError("session backend exploded"), {}, True),
    ],
)
def test_infrastructure_and_unexpected_failures_are_sanitized(
    combined_client,
    api_error,
    session_error,
    headers,
    set_cookie,
):
    client, api_resolver, session_resolver, downstream_calls = combined_client(
        api_resolver=FakeApiKeyTenantResolver(error=api_error),
        session_resolver=_session_resolver(error=session_error),
    )
    if set_cookie:
        client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.get("/combined", headers=headers)

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "AUTHENTICATION_UNAVAILABLE",
            "message": "Authentication could not be completed",
        }
    }
    assert "exploded" not in response.text
    assert "session db down" not in response.text
    assert SESSION_TOKEN not in response.text
    assert RAW_API_KEY_A not in response.text
    assert downstream_calls == []


def test_client_supplied_company_id_cannot_influence_trusted_identity(
    combined_client,
):
    client, api_resolver, session_resolver, downstream_calls = combined_client(
        api_resolver=FakeApiKeyTenantResolver(company_id=313),
    )

    response = client.get(
        "/company",
        headers={"X-API-Key": RAW_API_KEY_A},
        params={"company_id": "999"},
    )

    assert response.status_code == 200
    assert response.json() == {"company_id": 313}
    assert api_resolver.calls == [RAW_API_KEY_A]
    assert session_resolver.calls == []


def test_cross_request_identity_does_not_leak_between_auth_methods(
    combined_client,
):
    session_resolver = _session_resolver(company_id=222, user_id=333)
    client, api_resolver, session_resolver, downstream_calls = combined_client(
        api_resolver=FakeApiKeyTenantResolver(company_id=111),
        session_resolver=session_resolver,
    )

    api_response = client.get(
        "/combined",
        headers={"X-API-Key": RAW_API_KEY_A},
    )
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SECOND_SESSION_TOKEN)
    session_response = client.get("/combined")

    assert api_response.status_code == 200
    assert api_response.json()["company_id"] == 111
    assert api_response.json()["auth_method"] == "api_key"
    assert session_response.status_code == 200
    assert session_response.json()["company_id"] == 222
    assert session_response.json()["auth_method"] == "human_session"
    assert api_resolver.calls == [RAW_API_KEY_A]
    assert session_resolver.calls == [SECOND_SESSION_TOKEN]
    assert [identity.company_id for identity in downstream_calls] == [111, 222]


def test_combined_boundary_openapi_uses_or_security(combined_client):
    client, api_resolver, session_resolver, downstream_calls = combined_client()

    schema = client.get("/openapi.json").json()
    security = schema["paths"]["/combined"]["get"]["security"]

    assert security == [{"ApiKeyAuth": []}, {"SessionCookieAuth": []}]
    assert {"ApiKeyAuth": [], "SessionCookieAuth": []} not in security
    assert "security" not in schema["paths"]["/login"]["post"]
    assert "security" not in schema["paths"]["/health"]["get"]
    assert "security" not in schema["paths"]["/ready"]["get"]
    assert api_resolver.calls == []
    assert session_resolver.calls == []
    assert downstream_calls == []


def _session_resolver(
    *,
    company_id: int = 202,
    user_id: int = 101,
    error: Exception | None = None,
) -> FakeHumanSessionResolver:
    return FakeHumanSessionResolver(
        context=AuthenticatedUserContext(
            user_id=user_id,
            membership_id=303,
            company_id=company_id,
            role=MembershipRole.COMPANY_ADMIN,
            session_id=404,
        ),
        error=error,
    )
