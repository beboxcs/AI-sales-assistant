import pytest
from fastapi.testclient import TestClient

from app import main as app_main


@pytest.mark.parametrize("app_env", ["local", "test"])
def test_api_documentation_is_exposed_only_in_local_and_test(
    app_env,
    monkeypatch,
):
    monkeypatch.setattr(app_main.settings, "APP_ENV", app_env)

    app = app_main.create_app()

    assert app.docs_url == "/docs"
    assert app.redoc_url == "/redoc"
    assert app.openapi_url == "/openapi.json"
    assert app.debug is False


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_api_documentation_is_disabled_in_deployed_environments(
    app_env,
    monkeypatch,
):
    monkeypatch.setattr(app_main.settings, "APP_ENV", app_env)

    app = app_main.create_app()

    assert app.docs_url is None
    assert app.redoc_url is None
    assert app.openapi_url is None
    assert app.debug is False


@pytest.mark.parametrize("app_env", ["staging", "production"])
@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_deployed_environments_do_not_expose_documentation_routes(
    app_env,
    path,
    monkeypatch,
):
    monkeypatch.setattr(app_main.settings, "APP_ENV", app_env)
    app = app_main.create_app()

    with TestClient(app) as client:
        response = client.get(path)

    assert response.status_code == 404


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_disabling_documentation_keeps_normal_api_routes_registered(
    app_env,
    monkeypatch,
):
    monkeypatch.setattr(app_main.settings, "APP_ENV", app_env)
    app = app_main.create_app()

    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
