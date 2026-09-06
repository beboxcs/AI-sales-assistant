from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from app.api.errors import APIError, register_exception_handlers


def create_test_client() -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)

    router = APIRouter()

    @router.get("/test-error")
    def test_error():
        raise APIError(
            status_code=409,
            code="CONFLICT",
            message="Request conflicts with current state",
        )

    @router.get("/sensitive-error")
    def sensitive_error():
        raise APIError(
            status_code=500,
            code="INTERNAL_ERROR",
            message="Safe public message",
        ) from RuntimeError("postgresql://user:password@db.internal/app SELECT *")

    app.include_router(router)
    return TestClient(app)


def test_api_error_uses_standard_response_shape():
    client = create_test_client()

    response = client.get("/test-error")

    assert response.status_code == 409
    assert response.json() == {
        "error": {
            "code": "CONFLICT",
            "message": "Request conflicts with current state",
        }
    }


def test_api_error_response_exposes_only_safe_fields():
    client = create_test_client()

    response = client.get("/sensitive-error")

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "Safe public message",
        }
    }
    assert "postgresql://user:password@db.internal/app" not in response.text
    assert "SELECT *" not in response.text
