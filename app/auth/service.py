import hmac
from datetime import datetime, timezone

from app.auth.api_keys import hash_api_key_secret, parse_api_key
from app.auth.contracts import TrustedTenantContext
from app.auth.exceptions import (
    AuthenticationError,
    AuthBackendError,
    ExpiredApiCredentialError,
    InvalidApiCredentialError,
    RevokedApiCredentialError,
)
from app.auth.repository import find_api_credential_by_key_id
from app.database import SessionLocal

_INVALID_CREDENTIAL_MESSAGE = "Invalid API credential"
_BACKEND_ERROR_MESSAGE = "Authentication backend unavailable"
_DUMMY_SECRET_HASH = bytes(32)


def authenticate_api_key(
    raw_key: str,
    *,
    pepper: str,
) -> TrustedTenantContext:
    if not isinstance(pepper, str) or not pepper:
        raise AuthBackendError(_BACKEND_ERROR_MESSAGE)

    key_id, secret = parse_api_key(raw_key)
    supplied_hash = hash_api_key_secret(secret=secret, pepper=pepper)

    try:
        with SessionLocal() as session:
            credential = find_api_credential_by_key_id(
                session,
                key_id=key_id,
            )
            stored_hash = (
                credential.secret_hash
                if credential is not None
                else _DUMMY_SECRET_HASH
            )
            secret_matches = hmac.compare_digest(supplied_hash, stored_hash)

            if credential is None or not secret_matches:
                raise InvalidApiCredentialError(_INVALID_CREDENTIAL_MESSAGE)
            if credential.revoked_at is not None:
                raise RevokedApiCredentialError(_INVALID_CREDENTIAL_MESSAGE)
            if not credential.is_active:
                raise InvalidApiCredentialError(_INVALID_CREDENTIAL_MESSAGE)
            if (
                credential.expires_at is not None
                and credential.expires_at <= _utc_now()
            ):
                raise ExpiredApiCredentialError(_INVALID_CREDENTIAL_MESSAGE)

            return TrustedTenantContext(
                company_id=credential.company_id,
                credential_id=credential.id,
            )
    except AuthenticationError:
        raise
    except Exception as exc:
        raise AuthBackendError(_BACKEND_ERROR_MESSAGE) from exc


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
