"""Frozen human-auth service signatures.

Authentication establishes trusted identity. Authorization remains a separate
layer and must decide what an authenticated principal may do.
"""

from datetime import datetime, timedelta, timezone

from app.auth.human.exceptions import AuthInfrastructureError
from app.auth.human.session_repository import persist_user_session
from app.auth.human.sessions import digest_session_token, generate_session_token
from app.auth.human.types import (
    AuthenticatedPrincipal,
    AuthenticatedUserContext,
    CreatedSession,
)
from app.database import SessionLocal

SESSION_ABSOLUTE_LIFETIME = timedelta(hours=8)
_INFRASTRUCTURE_ERROR_MESSAGE = "Human authentication unavailable"


def authenticate_credentials(
    *,
    email: str,
    password: str,
) -> AuthenticatedPrincipal:
    raise NotImplementedError


def create_session(
    *,
    principal: AuthenticatedPrincipal,
) -> CreatedSession:
    try:
        expires_at = _utc_now() + SESSION_ABSOLUTE_LIFETIME
        token = generate_session_token()
        token_digest = digest_session_token(token=token)

        with SessionLocal.begin() as session:
            persisted_session = persist_user_session(
                session,
                user_id=principal.user_id,
                membership_id=principal.membership_id,
                company_id=principal.company_id,
                token_digest=token_digest,
                expires_at=expires_at,
            )
            session_id = persisted_session.id

        context = AuthenticatedUserContext(
            user_id=principal.user_id,
            membership_id=principal.membership_id,
            company_id=principal.company_id,
            role=principal.role,
            session_id=session_id,
        )
        return CreatedSession(
            token=token,
            expires_at=expires_at,
            context=context,
        )
    except Exception:
        raise AuthInfrastructureError(_INFRASTRUCTURE_ERROR_MESSAGE) from None


def resolve_session(
    *,
    token: str,
) -> AuthenticatedUserContext:
    raise NotImplementedError


def revoke_session(
    *,
    token: str,
) -> None:
    raise NotImplementedError


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
