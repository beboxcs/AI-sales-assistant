from datetime import datetime, timedelta, timezone
from typing import Protocol

from fastapi import Depends, Request, Response, Security
from fastapi.security import APIKeyCookie, APIKeyHeader

from app.api.errors import APIError
from app.auth.contracts import TrustedTenantContext
from app.auth.exceptions import AuthBackendError, InvalidApiCredentialError
from app.auth.human.exceptions import (
    AuthInfrastructureError,
    InvalidSessionError,
)
from app.auth.human.service import resolve_session
from app.auth.human.types import AuthenticatedUserContext
from app.auth.service import authenticate_api_key
from app.config import normalize_browser_origin, settings

SESSION_COOKIE_NAME = "rea_session"
SESSION_COOKIE_PATH = "/"
SESSION_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_MAX_AGE_SECONDS = 8 * 60 * 60

api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="ApiKeyAuth",
    auto_error=False,
)
session_cookie = APIKeyCookie(
    name=SESSION_COOKIE_NAME,
    scheme_name="SessionCookieAuth",
    auto_error=False,
)


class HumanSessionResolver(Protocol):
    def __call__(self, *, token: str) -> AuthenticatedUserContext: ...


class PublicHumanSessionResolver:
    def __call__(self, *, token: str) -> AuthenticatedUserContext:
        return resolve_session(token=token)


def get_human_session_resolver() -> HumanSessionResolver:
    return PublicHumanSessionResolver()


def require_unambiguous_credentials(
    raw_api_key: str | None = Security(api_key_header),
    session_token: str | None = Security(session_cookie),
) -> None:
    if raw_api_key is not None and session_token is not None:
        raise APIError(
            status_code=400,
            code="AMBIGUOUS_CREDENTIALS",
            message="Use exactly one authentication credential type",
        )


def get_current_user(
    request: Request,
    session_token: str | None = Security(session_cookie),
    resolve_human_session: HumanSessionResolver = Depends(
        get_human_session_resolver
    ),
) -> AuthenticatedUserContext:
    require_unambiguous_credentials(
        raw_api_key=request.headers.get("x-api-key"),
        session_token=session_token,
    )
    if session_token is None:
        raise _invalid_session_error()

    try:
        return resolve_human_session(token=session_token)
    except InvalidSessionError as exc:
        raise _invalid_session_error() from exc
    except AuthInfrastructureError as exc:
        raise _auth_backend_error() from exc
    except Exception as exc:
        raise _auth_backend_error() from exc


def get_trusted_company_id(
    raw_api_key: str | None = Security(api_key_header),
) -> int:
    if raw_api_key is None:
        return _local_development_company_id()
    return resolve_trusted_tenant(raw_api_key).company_id


def resolve_trusted_tenant(
    raw_api_key: str,
) -> TrustedTenantContext:
    pepper = settings.API_KEY_PEPPER
    if pepper is None or not pepper.strip():
        raise _auth_backend_error()

    try:
        return authenticate_api_key(raw_api_key, pepper=pepper)
    except InvalidApiCredentialError as exc:
        raise _invalid_credential_error() from exc
    except AuthBackendError as exc:
        raise _auth_backend_error() from exc
    except Exception as exc:
        raise _auth_backend_error() from exc


def require_trusted_origin(request: Request) -> None:
    """Reject missing, malformed, or unconfigured browser Origin headers."""
    origin = request.headers.get("origin")
    if not is_trusted_browser_origin(origin):
        raise _untrusted_origin_error()


def is_trusted_browser_origin(origin: str | None) -> bool:
    if origin is None:
        return False

    try:
        normalized_origin = normalize_browser_origin(origin)
    except ValueError:
        return False

    return normalized_origin in set(settings.TRUSTED_BROWSER_ORIGINS)


def set_session_cookie(
    response: Response,
    session_token: str,
    *,
    expires_at: datetime | None = None,
    max_age: int | timedelta | None = None,
) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        max_age=_resolve_session_cookie_max_age(
            expires_at=expires_at,
            max_age=max_age,
        ),
        httponly=True,
        secure=is_session_cookie_secure(),
        samesite=SESSION_COOKIE_SAMESITE,
        path=SESSION_COOKIE_PATH,
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path=SESSION_COOKIE_PATH,
        samesite=SESSION_COOKIE_SAMESITE,
        secure=is_session_cookie_secure(),
        httponly=True,
    )


def is_session_cookie_secure() -> bool:
    return settings.APP_ENV in {"staging", "production"}


def _resolve_session_cookie_max_age(
    *,
    expires_at: datetime | None,
    max_age: int | timedelta | None,
) -> int:
    if (expires_at is None) == (max_age is None):
        raise ValueError("Provide exactly one of expires_at or max_age")

    if expires_at is not None:
        if expires_at.tzinfo is None or expires_at.utcoffset() is None:
            raise ValueError("expires_at must be timezone-aware")
        requested_max_age = int(
            (expires_at - datetime.now(timezone.utc)).total_seconds()
        )
    elif isinstance(max_age, timedelta):
        requested_max_age = int(max_age.total_seconds())
    else:
        requested_max_age = int(max_age)

    return max(0, min(requested_max_age, SESSION_COOKIE_MAX_AGE_SECONDS))


def _local_development_company_id() -> int:
    if settings.APP_ENV == "local" and settings.DEV_COMPANY_ID is not None:
        return settings.DEV_COMPANY_ID
    raise _invalid_credential_error()


def _invalid_credential_error() -> APIError:
    return APIError(
        status_code=401,
        code="INVALID_API_CREDENTIAL",
        message="Invalid or unavailable API credential",
    )


def _invalid_session_error() -> APIError:
    return APIError(
        status_code=401,
        code="INVALID_SESSION",
        message="Invalid or unavailable session",
    )


def _auth_backend_error() -> APIError:
    return APIError(
        status_code=500,
        code="AUTHENTICATION_UNAVAILABLE",
        message="Authentication could not be completed",
    )


def _untrusted_origin_error() -> APIError:
    return APIError(
        status_code=403,
        code="UNTRUSTED_ORIGIN",
        message="Request origin is not trusted",
    )
