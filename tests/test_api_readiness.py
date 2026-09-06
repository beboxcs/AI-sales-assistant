import pytest
from fastapi.testclient import TestClient

from app.api.v1.readiness import get_database_readiness_check
from app.main import app


@pytest.fixture
def client():
    original_overrides = app.dependency_overrides.copy()
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides = original_overrides


def test_readiness_succeeds(client):
    app.dependency_overrides[get_database_readiness_check] = lambda: lambda: None

    response = client.get("/api/v1/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "services": {
            "api": "ok",
            "database": "ok",
        },
    }


def test_readiness_database_check_fails(client, caplog):
    def failing_check():
        raise RuntimeError("postgresql://user:password@db.internal/app")

    app.dependency_overrides[get_database_readiness_check] = lambda: failing_check

    response = client.get("/api/v1/ready")

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "status": "not_ready",
            "services": {
                "api": "ok",
                "database": "failed",
            },
        }
    }
    assert "Database readiness check failed" in caplog.text
    assert "postgresql://user:password@db.internal/app" not in response.text
