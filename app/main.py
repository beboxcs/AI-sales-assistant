from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.errors import register_exception_handlers
from app.api.v1.v1_router import router as v1_router
from app.config import settings
from app.security_headers import SecurityHeadersMiddleware

CORS_ALLOW_METHODS = ["GET", "POST"]
CORS_ALLOW_HEADERS = ["Content-Type", "X-CSRF-Token"]


def create_app() -> FastAPI:
    expose_api_docs = settings.APP_ENV in {"local", "test"}
    app = FastAPI(
        title="Real Estate AI API",
        version="0.1.0",
        debug=False,
        docs_url="/docs" if expose_api_docs else None,
        redoc_url="/redoc" if expose_api_docs else None,
        openapi_url="/openapi.json" if expose_api_docs else None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.TRUSTED_BROWSER_ORIGINS),
        allow_credentials=True,
        allow_methods=CORS_ALLOW_METHODS,
        allow_headers=CORS_ALLOW_HEADERS,
    )
    app.add_middleware(
        SecurityHeadersMiddleware,
        include_hsts=settings.APP_ENV in {"staging", "production"},
    )
    register_exception_handlers(app)
    app.include_router(v1_router, prefix="/api/v1")

    @app.get("/")
    def read_root():
        return {"message": "API is up and running"}

    return app


app = create_app()
