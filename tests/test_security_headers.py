import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.api.v1.readiness import get_database_readiness_check
from app.security_headers import STRICT_TRANSPORT_SECURITY


BASE_SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
}


@pytest.mark.parametrize("app_env", ["local", "test"])
def test_local_and_test_security_headers_exclude_hsts(app_env, monkeypatch):
    monkeypatch.setattr(app_main.settings, "APP_ENV", app_env)
    app = app_main.create_app()

    with TestClient(app) as client:
        response = client.get("/")

    assert_base_security_headers(response)
    assert "strict-transport-security" not in response.headers


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_deployed_security_headers_include_hsts(app_env, monkeypatch):
    monkeypatch.setattr(app_main.settings, "APP_ENV", app_env)
    app = app_main.create_app()

    with TestClient(app) as client:
        response = client.get("/")

    assert_base_security_headers(response)
    assert (
        response.headers["strict-transport-security"]
        == STRICT_TRANSPORT_SECURITY
    )


@pytest.mark.parametrize("app_env", ["local", "test", "staging", "production"])
def test_normal_api_endpoint_receives_security_headers(app_env, monkeypatch):
    monkeypatch.setattr(app_main.settings, "APP_ENV", app_env)
    app = app_main.create_app()

    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert_base_security_headers(response)


def test_application_error_response_receives_security_headers(monkeypatch):
    def failing_check():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(app_main.settings, "APP_ENV", "production")
    app = app_main.create_app()
    app.dependency_overrides[get_database_readiness_check] = (
        lambda: failing_check
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/ready")

    assert response.status_code == 503
    assert_base_security_headers(response)
    assert (
        response.headers["strict-transport-security"]
        == STRICT_TRANSPORT_SECURITY
    )


def assert_base_security_headers(response) -> None:
    for header_name, header_value in BASE_SECURITY_HEADERS.items():
        assert response.headers[header_name] == header_value
