from fastapi import Security
from fastapi.security import APIKeyCookie, APIKeyHeader

from app.api.errors import APIError
from app.auth.contracts import TrustedTenantContext
from app.auth.exceptions import AuthBackendError, InvalidApiCredentialError
from app.auth.service import authenticate_api_key
from app.config import settings

api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="ApiKeyAuth",
    auto_error=False,
)
session_cookie = APIKeyCookie(
    name="rea_session",
    scheme_name="SessionCookieAuth",
    auto_error=False,
)


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


def _auth_backend_error() -> APIError:
    return APIError(
        status_code=500,
        code="AUTHENTICATION_UNAVAILABLE",
        message="Authentication could not be completed",
    )
