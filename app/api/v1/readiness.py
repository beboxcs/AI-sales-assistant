import logging
from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text

from app.database import SessionLocal

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Readiness"])

READY_RESPONSE = {
    "status": "ready",
    "services": {
        "api": "ok",
        "database": "ok",
    },
}

NOT_READY_DETAIL = {
    "status": "not_ready",
    "services": {
        "api": "ok",
        "database": "failed",
    },
}


def check_database_ready() -> None:
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
    finally:
        db.close()


def get_database_readiness_check() -> Callable[[], None]:
    return check_database_ready


@router.get("/ready")
def readiness_check(
    check_database: Callable[[], None] = Depends(get_database_readiness_check),
):
    try:
        check_database()
    except Exception as exc:
        logger.exception("Database readiness check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=NOT_READY_DETAIL,
        ) from exc

    return READY_RESPONSE
