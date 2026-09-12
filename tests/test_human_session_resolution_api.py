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
    InvalidSessionError,
    SessionExpiredError,
    SessionRevokedError,
)
from app.auth.human.types import AuthenticatedUserContext
from app.domain.enums import MembershipRole


SESSION_TOKEN = "opaque-session-token"
RAW_API_KEY = "rea_v1_" + ("K" * 22) + "." + ("s" * 43)


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


@pytest.fixture
def session_client(monkeypatch):
    test_app = FastAPI()
    register_exception_handlers(test_app)
    downstream_calls = []

    @test_app.get("/current-user")
    def current_user(
        user: AuthenticatedUserContext = Depends(api_security.get_current_user),
    ):
        downstream_calls.append(user)
        return {
            "user_id": user.user_id,
            "membership_id": user.membership_id,
            "company_id": user.company_id,
            "role": user.role,
            "session_id": user.session_id,
        }

    def factory(resolver: FakeSessionResolver | None = None):
        selected_resolver = resolver or _session_resolver()
        test_app.dependency_overrides[
            api_security.get_human_session_resolver
        ] = lambda: selected_resolver
        test_app.openapi_schema = None
        client = TestClient(test_app)
        return client, selected_resolver, downstream_calls

    monkeypatch.setattr(api_security.settings, "APP_ENV", "local")
    monkeypatch.setattr(api_security.settings, "DEV_COMPANY_ID", 77)

    yield factory

    test_app.dependency_overrides = {}


def test_valid_session_cookie_resolves_current_user_once(session_client):
    client, resolver, downstream_calls = session_client()
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.get("/current-user")

    assert response.status_code == 200
    assert response.json() == {
        "user_id": 101,
        "membership_id": 202,
        "company_id": 303,
        "role": "company_admin",
        "session_id": 404,
    }
    assert resolver.calls == [SESSION_TOKEN]
    assert downstream_calls == [resolver.context]


def test_missing_session_cookie_returns_generic_401_without_dev_fallback(
    session_client,
):
    client, resolver, downstream_calls = session_client()

    response = client.get("/current-user")

    assert response.status_code == 401
    assert response.json() == _invalid_session_response()
    assert resolver.calls == []
    assert downstream_calls == []


def test_api_key_only_returns_invalid_session_for_human_session_dependency(
    session_client,
):
    client, resolver, downstream_calls = session_client()

    response = client.get(
        "/current-user",
        headers={"X-API-Key": RAW_API_KEY},
    )

    assert response.status_code == 401
    assert response.json() == _invalid_session_response()
    assert RAW_API_KEY not in response.text
    assert resolver.calls == []
    assert downstream_calls == []


def test_api_key_and_session_cookie_are_rejected_before_resolution(
    session_client,
):
    client, resolver, downstream_calls = session_client()
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.get(
        "/current-user",
        headers={"X-API-Key": RAW_API_KEY},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "AMBIGUOUS_CREDENTIALS",
            "message": "Use exactly one authentication credential type",
        }
    }
    assert RAW_API_KEY not in response.text
    assert SESSION_TOKEN not in response.text
    assert resolver.calls == []
    assert downstream_calls == []


@pytest.mark.parametrize(
    "error",
    [
        InvalidSessionError("token digest not found"),
        SessionExpiredError("session expired at internal timestamp"),
        SessionRevokedError("revoked by session id 123"),
        InactiveUserError("user 101 is inactive"),
        InactiveMembershipError("membership 202 disabled"),
    ],
)
def test_invalid_session_states_return_identical_public_401(
    session_client,
    error,
):
    resolver = _session_resolver(error=error)
    client, resolver, downstream_calls = session_client(resolver)
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.get("/current-user")

    assert response.status_code == 401
    assert response.json() == _invalid_session_response()
    assert str(error) not in response.text
    assert SESSION_TOKEN not in response.text
    assert resolver.calls == [SESSION_TOKEN]
    assert downstream_calls == []


@pytest.mark.parametrize(
    "error",
    [
        AuthInfrastructureError("database unavailable for token lookup"),
        RuntimeError("unexpected token resolver detail"),
    ],
)
def test_session_resolution_backend_failures_return_safe_500(
    session_client,
    error,
):
    resolver = _session_resolver(error=error)
    client, resolver, downstream_calls = session_client(resolver)
    client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = client.get("/current-user")

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "AUTHENTICATION_UNAVAILABLE",
            "message": "Authentication could not be completed",
        }
    }
    assert str(error) not in response.text
    assert SESSION_TOKEN not in response.text
    assert resolver.calls == [SESSION_TOKEN]
    assert downstream_calls == []


def test_current_user_openapi_uses_session_cookie_security(session_client):
    client, resolver, downstream_calls = session_client()

    schema = client.get("/openapi.json").json()

    assert schema["components"]["securitySchemes"]["SessionCookieAuth"] == {
        "type": "apiKey",
        "in": "cookie",
        "name": "rea_session",
    }
    assert schema["paths"]["/current-user"]["get"]["security"] == [
        {"SessionCookieAuth": []}
    ]
    assert resolver.calls == []
    assert downstream_calls == []


def _session_resolver(
    *,
    error: Exception | None = None,
) -> FakeSessionResolver:
    return FakeSessionResolver(
        context=AuthenticatedUserContext(
            user_id=101,
            membership_id=202,
            company_id=303,
            role=MembershipRole.COMPANY_ADMIN,
            session_id=404,
        ),
        error=error,
    )


def _invalid_session_response():
    return {
        "error": {
            "code": "INVALID_SESSION",
            "message": "Invalid or unavailable session",
        }
    }
