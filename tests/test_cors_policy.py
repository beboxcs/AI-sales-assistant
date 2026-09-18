import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import main as app_main
from app.api import security as api_security
from app.config import Settings


TRUSTED_ORIGIN = "https://app.example.com"
UNTRUSTED_ORIGIN = "https://attacker.example"
SESSION_TOKEN = "opaque-session-token"


@pytest.fixture
def cors_client(monkeypatch):
    monkeypatch.setattr(
        app_main.settings,
        "TRUSTED_BROWSER_ORIGINS",
        (TRUSTED_ORIGIN,),
    )
    app = app_main.create_app()
    with TestClient(app) as client:
        yield client


def test_allowed_origin_preflight_allows_expected_browser_headers(cors_client):
    response = _preflight(
        cors_client,
        origin=TRUSTED_ORIGIN,
        requested_headers="Content-Type, X-CSRF-Token",
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == TRUSTED_ORIGIN
    assert response.headers["access-control-allow-credentials"] == "true"
    assert _header_values(response, "access-control-allow-methods") == {
        "get",
        "post",
    }
    allowed_headers = _header_values(
        response,
        "access-control-allow-headers",
    )
    assert "content-type" in allowed_headers
    assert "x-csrf-token" in allowed_headers
    assert "x-api-key" not in allowed_headers


def test_disallowed_origin_preflight_does_not_grant_origin(cors_client):
    response = _preflight(
        cors_client,
        origin=UNTRUSTED_ORIGIN,
        requested_headers="Content-Type, X-CSRF-Token",
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_x_api_key_is_not_allowed_for_browser_preflight(cors_client):
    response = _preflight(
        cors_client,
        origin=TRUSTED_ORIGIN,
        requested_headers="X-API-Key",
    )

    assert response.status_code == 400
    allowed_headers = _header_values(
        response,
        "access-control-allow-headers",
    )
    assert "x-api-key" not in allowed_headers


def test_actual_trusted_origin_request_receives_cors_headers(cors_client):
    response = cors_client.get(
        "/api/v1/health",
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == TRUSTED_ORIGIN
    assert response.headers["access-control-allow-credentials"] == "true"


def test_actual_untrusted_origin_request_does_not_receive_cors_origin_grant(
    cors_client,
):
    response = cors_client.get(
        "/api/v1/health",
        headers={"Origin": UNTRUSTED_ORIGIN},
    )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_cors_does_not_replace_trusted_origin_enforcement(cors_client):
    response = cors_client.post(
        "/api/v1/auth/password-reset/request",
        json={"email": "user@example.com"},
        headers={"Origin": UNTRUSTED_ORIGIN},
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "UNTRUSTED_ORIGIN"
    assert "access-control-allow-origin" not in response.headers


def test_cors_does_not_replace_csrf_enforcement(cors_client):
    cors_client.cookies.set(api_security.SESSION_COOKIE_NAME, SESSION_TOKEN)

    response = cors_client.post(
        "/api/v1/auth/logout",
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "INVALID_CSRF_TOKEN"
    assert response.headers["access-control-allow-origin"] == TRUSTED_ORIGIN


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_deployed_settings_reject_empty_trusted_browser_origins(app_env):
    with pytest.raises(ValidationError, match="TRUSTED_BROWSER_ORIGINS"):
        _deployed_settings(app_env=app_env, TRUSTED_BROWSER_ORIGINS=())


@pytest.mark.parametrize("app_env", ["staging", "production"])
@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:3000",
        "https://localhost",
        "https://app.localhost",
    ],
)
def test_deployed_settings_reject_localhost_trusted_browser_origins(
    app_env,
    origin,
):
    with pytest.raises(ValidationError, match="TRUSTED_BROWSER_ORIGINS"):
        _deployed_settings(
            app_env=app_env,
            TRUSTED_BROWSER_ORIGINS=(origin,),
        )


@pytest.mark.parametrize("app_env", ["staging", "production"])
@pytest.mark.parametrize(
    "origin",
    [
        "http://127.0.0.1:3000",
        "https://127.4.5.6",
        "http://[::1]:3000",
    ],
)
def test_deployed_settings_reject_loopback_trusted_browser_origins(
    app_env,
    origin,
):
    with pytest.raises(ValidationError, match="TRUSTED_BROWSER_ORIGINS"):
        _deployed_settings(
            app_env=app_env,
            TRUSTED_BROWSER_ORIGINS=(origin,),
        )


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_deployed_settings_accept_non_local_https_trusted_origin(app_env):
    settings = _deployed_settings(
        app_env=app_env,
        TRUSTED_BROWSER_ORIGINS=("https://app.example.com:443",),
    )

    assert settings.TRUSTED_BROWSER_ORIGINS == ("https://app.example.com",)


def test_existing_trusted_browser_origin_normalization_remains_valid():
    settings = Settings(
        _env_file=None,
        APP_ENV="local",
        DATABASE_URL="postgresql://example",
        TRUSTED_BROWSER_ORIGINS=(
            " HTTPS://App.Example.com/ ",
            "http://localhost:3000",
            "https://app.example.com:8443",
        ),
    )

    assert settings.TRUSTED_BROWSER_ORIGINS == (
        "https://app.example.com",
        "http://localhost:3000",
        "https://app.example.com:8443",
    )


def _preflight(
    client: TestClient,
    *,
    origin: str,
    requested_headers: str,
):
    return client.options(
        "/api/v1/auth/logout",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": requested_headers,
        },
    )


def _header_values(response, header_name: str) -> set[str]:
    value = response.headers.get(header_name)
    if value is None:
        return set()
    return {item.strip().lower() for item in value.split(",")}


def _deployed_settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "APP_ENV": "production",
        "DATABASE_URL": "postgresql://example",
        "API_KEY_PEPPER": "pepper",
        "DEV_COMPANY_ID": None,
        "LOGIN_RATE_LIMIT_BACKEND": "redis",
        "REDIS_URL": "redis://redis:6379/0",
        "TRUSTED_BROWSER_ORIGINS": (TRUSTED_ORIGIN,),
        "INVITATION_DELIVERY_BACKEND": "smtp",
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": 587,
        "SMTP_FROM_EMAIL": "security@example.com",
        "INVITATION_ACCEPT_BASE_URL": "https://app.example.com/invitations/accept",
        "PASSWORD_RESET_URL": "https://app.example.com/reset-password",
    }
    values.update(overrides)
    app_env = values.pop("app_env", None)
    if app_env is not None:
        values["APP_ENV"] = app_env
    return Settings(**values)
