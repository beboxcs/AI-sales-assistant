from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_db_session


def test_api_database_dependency_can_be_overridden():
    app = FastAPI()

    @app.get("/uses-db-dependency")
    def uses_db_dependency(db=Depends(get_db_session)):
        return {"dependency": db["name"]}

    app.dependency_overrides[get_db_session] = lambda: {"name": "fake-db-session"}

    response = TestClient(app).get("/uses-db-dependency")

    assert response.status_code == 200
    assert response.json() == {"dependency": "fake-db-session"}
