from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.health import router as health_router
from app.api.v1.inventory import router as inventory_router
from app.api.v1.readiness import router as readiness_router
from app.api.v1.search import router as search_router

router = APIRouter()
router.include_router(auth_router)
router.include_router(health_router)
router.include_router(inventory_router)
router.include_router(readiness_router)
router.include_router(search_router)
