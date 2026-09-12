import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api import security as api_security
from app.api.errors import register_exception_handlers
from app.config import Settings


TRUSTED_ORIGIN = "https://app.example.com:8443"


@pytest.fixture
def origin_client(monkeypatch):
    monkeypatch.setattr(
        api_security.settings,
        "TRUSTED_BROWSER_ORIGINS",
        (TRUSTED_ORIGIN,),
    )

    test_app = FastAPI()
    register_exception_handlers(test_app)

    @test_app.post(
        "/browser-auth-foundation",
        dependencies=[Depends(api_security.require_trusted_origin)],
    )
    def browser_auth_foundation():
        return {"status": "ok"}

    with TestClient(test_app) as client:
        yield client


def test_exact_trusted_origin_is_accepted(origin_client):
    response = origin_client.post(
        "/browser-auth-foundation",
        headers={"Origin": TRUSTED_ORIGIN},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_different_origin_scheme_is_rejected(origin_client):
    response = origin_client.post(
        "/browser-auth-foundation",
        headers={"Origin": "http://app.example.com:8443"},
    )

    assert response.status_code == 403
    assert response.json() == _untrusted_origin_response()


def test_different_origin_host_is_rejected(origin_client):
    response = origin_client.post(
        "/browser-auth-foundation",
        headers={"Origin": "https://admin.example.com:8443"},
    )

    assert response.status_code == 403
    assert response.json() == _untrusted_origin_response()


def test_different_origin_port_is_rejected(origin_client):
    response = origin_client.post(
        "/browser-auth-foundation",
        headers={"Origin": "https://app.example.com:443"},
    )

    assert response.status_code == 403
    assert response.json() == _untrusted_origin_response()


def test_explicit_default_port_matches_configured_origin_without_port(
    origin_client,
    monkeypatch,
):
    monkeypatch.setattr(
        api_security.settings,
        "TRUSTED_BROWSER_ORIGINS",
        ("https://app.example.com",),
    )

    response = origin_client.post(
        "/browser-auth-foundation",
        headers={"Origin": "https://app.example.com:443"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_origin_without_default_port_matches_configured_explicit_default_port(
    origin_client,
    monkeypatch,
):
    configured_settings = Settings(
        _env_file=None,
        APP_ENV="local",
        DATABASE_URL="postgresql://example",
        TRUSTED_BROWSER_ORIGINS=("https://app.example.com:443",),
    )
    monkeypatch.setattr(
        api_security.settings,
        "TRUSTED_BROWSER_ORIGINS",
        configured_settings.TRUSTED_BROWSER_ORIGINS,
    )

    response = origin_client.post(
        "/browser-auth-foundation",
        headers={"Origin": "https://app.example.com"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_non_default_ports_remain_significant(origin_client):
    response = origin_client.post(
        "/browser-auth-foundation",
        headers={"Origin": "https://app.example.com:9443"},
    )

    assert response.status_code == 403
    assert response.json() == _untrusted_origin_response()


@pytest.mark.parametrize(
    "origin",
    [
        "not-an-origin",
        "null",
        "https://app.example.com:bad-port",
        "https://app.example.com/path",
        "https://user:pass@app.example.com",
    ],
)
def test_malformed_origin_is_rejected(origin_client, origin):
    response = origin_client.post(
        "/browser-auth-foundation",
        headers={"Origin": origin},
    )

    assert response.status_code == 403
    assert response.json() == _untrusted_origin_response()


def test_missing_origin_is_rejected(origin_client):
    response = origin_client.post("/browser-auth-foundation")

    assert response.status_code == 403
    assert response.json() == _untrusted_origin_response()


def test_multiple_configured_trusted_origins_are_accepted(
    origin_client,
    monkeypatch,
):
    second_origin = "http://localhost:3000"
    monkeypatch.setattr(
        api_security.settings,
        "TRUSTED_BROWSER_ORIGINS",
        (TRUSTED_ORIGIN, second_origin),
    )

    response = origin_client.post(
        "/browser-auth-foundation",
        headers={"Origin": second_origin},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_untrusted_origin_error_does_not_leak_header_values(origin_client):
    origin = "https://attacker.example"
    api_key = "rea_v1_secret-value"

    response = origin_client.post(
        "/browser-auth-foundation",
        headers={
            "Origin": origin,
            "X-API-Key": api_key,
        },
    )

    assert response.status_code == 403
    assert response.json() == _untrusted_origin_response()
    assert origin not in response.text
    assert api_key not in response.text


def _untrusted_origin_response():
    return {
        "error": {
            "code": "UNTRUSTED_ORIGIN",
            "message": "Request origin is not trusted",
        }
    }
