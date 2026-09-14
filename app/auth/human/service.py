"""Frozen human-auth service signatures.

Authentication establishes trusted identity. Authorization remains a separate
layer and must decide what an authenticated principal may do.
"""

from datetime import datetime, timedelta, timezone

from app.auth.human.exceptions import (
    AuthInfrastructureError,
    InactiveMembershipError,
    InactiveUserError,
    InvalidCredentialsError,
    InvalidSessionError,
    SessionExpiredError,
    SessionRevokedError,
)
from app.auth.human.passwords import _DUMMY_PASSWORD_HASH, verify_password
from app.auth.human.repository import (
    find_active_memberships_for_user,
    find_user_by_normalized_email,
)
from app.auth.human.session_repository import (
    find_membership_by_id,
    find_user_by_id,
    find_user_session_by_token_digest_for_update,
    persist_user_session,
    refresh_user_session_activity,
    revoke_user_session,
)
from app.auth.human.sessions import digest_session_token, generate_session_token
from app.auth.human.types import (
    AuthenticatedPrincipal,
    AuthenticatedUserContext,
    CreatedSession,
)
from app.database import SessionLocal
from app.domain.enums import MembershipRole
from app.domain.users import normalize_email

SESSION_ABSOLUTE_LIFETIME = timedelta(hours=8)
SESSION_IDLE_TIMEOUT = timedelta(hours=2)
_INVALID_CREDENTIALS_MESSAGE = "Invalid credentials"
_INVALID_SESSION_MESSAGE = "Invalid session"
_INFRASTRUCTURE_ERROR_MESSAGE = "Human authentication unavailable"


def authenticate_credentials(
    *,
    email: str,
    password: str,
) -> AuthenticatedPrincipal:
    try:
        try:
            normalized_email = normalize_email(email)
        except (AttributeError, TypeError, ValueError):
            verify_password(password, _DUMMY_PASSWORD_HASH)
            raise InvalidCredentialsError(
                _INVALID_CREDENTIALS_MESSAGE
            ) from None

        with SessionLocal() as session:
            user = find_user_by_normalized_email(
                session,
                normalized_email=normalized_email,
            )
            password_hash = (
                user.password_hash if user is not None else _DUMMY_PASSWORD_HASH
            )
            password_matches = verify_password(password, password_hash)

            if user is None or user.is_active is not True or not password_matches:
                raise InvalidCredentialsError(_INVALID_CREDENTIALS_MESSAGE)

            active_memberships = [
                membership
                for membership in find_active_memberships_for_user(
                    session,
                    user_id=user.id,
                )
                if membership.is_active is True
            ]
            if len(active_memberships) != 1:
                raise InvalidCredentialsError(_INVALID_CREDENTIALS_MESSAGE)

            membership = active_memberships[0]
            try:
                role = MembershipRole(membership.role)
            except (TypeError, ValueError):
                raise AuthInfrastructureError(
                    _INFRASTRUCTURE_ERROR_MESSAGE
                ) from None

            return AuthenticatedPrincipal(
                user_id=user.id,
                membership_id=membership.id,
                company_id=membership.company_id,
                role=role,
            )
    except InvalidCredentialsError:
        raise
    except Exception:
        raise AuthInfrastructureError(_INFRASTRUCTURE_ERROR_MESSAGE) from None


def create_session(
    *,
    principal: AuthenticatedPrincipal,
) -> CreatedSession:
    try:
        created_at = _utc_now()
        expires_at = created_at + SESSION_ABSOLUTE_LIFETIME
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
                last_activity_at=created_at,
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
    if not isinstance(token, str) or not token.strip():
        raise InvalidSessionError(_INVALID_SESSION_MESSAGE)

    try:
        token_digest = digest_session_token(token=token)
    except (AttributeError, TypeError, UnicodeEncodeError):
        raise InvalidSessionError(_INVALID_SESSION_MESSAGE) from None

    try:
        with SessionLocal.begin() as session:
            user_session = find_user_session_by_token_digest_for_update(
                session,
                token_digest=token_digest,
            )
            if user_session is None:
                raise InvalidSessionError(_INVALID_SESSION_MESSAGE)

            now = _utc_now()
            if user_session.revoked_at is not None:
                raise SessionRevokedError(_INVALID_SESSION_MESSAGE)
            if now >= user_session.expires_at:
                raise SessionExpiredError(_INVALID_SESSION_MESSAGE)
            if now >= user_session.last_activity_at + SESSION_IDLE_TIMEOUT:
                raise SessionExpiredError(_INVALID_SESSION_MESSAGE)

            user = find_user_by_id(session, user_id=user_session.user_id)
            if user is None:
                raise AuthInfrastructureError(_INFRASTRUCTURE_ERROR_MESSAGE)
            if user.is_active is not True:
                raise InactiveUserError(_INVALID_SESSION_MESSAGE)

            membership = find_membership_by_id(
                session,
                membership_id=user_session.membership_id,
            )
            if membership is None:
                raise AuthInfrastructureError(_INFRASTRUCTURE_ERROR_MESSAGE)
            if membership.is_active is not True:
                raise InactiveMembershipError(_INVALID_SESSION_MESSAGE)

            if (
                user_session.user_id != user.id
                or user_session.membership_id != membership.id
                or membership.user_id != user.id
                or user_session.company_id != membership.company_id
            ):
                raise AuthInfrastructureError(_INFRASTRUCTURE_ERROR_MESSAGE)

            try:
                role = MembershipRole(membership.role)
            except (TypeError, ValueError):
                raise AuthInfrastructureError(
                    _INFRASTRUCTURE_ERROR_MESSAGE
                ) from None

            refresh_user_session_activity(
                session,
                user_session=user_session,
                last_activity_at=now,
            )
            context = AuthenticatedUserContext(
                user_id=user.id,
                membership_id=membership.id,
                company_id=membership.company_id,
                role=role,
                session_id=user_session.id,
            )
    except InvalidSessionError:
        raise
    except Exception:
        raise AuthInfrastructureError(_INFRASTRUCTURE_ERROR_MESSAGE) from None

    return context


def revoke_session(
    *,
    token: str,
) -> None:
    if not isinstance(token, str) or not token.strip():
        raise InvalidSessionError(_INVALID_SESSION_MESSAGE)

    try:
        token_digest = digest_session_token(token=token)
    except (AttributeError, TypeError, UnicodeEncodeError):
        raise InvalidSessionError(_INVALID_SESSION_MESSAGE) from None
    except Exception:
        raise AuthInfrastructureError(_INFRASTRUCTURE_ERROR_MESSAGE) from None

    try:
        with SessionLocal.begin() as session:
            user_session = find_user_session_by_token_digest_for_update(
                session,
                token_digest=token_digest,
            )
            if user_session is None:
                raise InvalidSessionError(_INVALID_SESSION_MESSAGE)

            if user_session.revoked_at is None:
                revoke_user_session(
                    session,
                    user_session=user_session,
                    revoked_at=_utc_now(),
                )
    except InvalidSessionError:
        raise
    except Exception:
        raise AuthInfrastructureError(_INFRASTRUCTURE_ERROR_MESSAGE) from None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
