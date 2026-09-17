"""Backend-only password reset and credential-generation security core."""

import hashlib
import re
import secrets
from datetime import datetime, timedelta, timezone

from app.auth.human.exceptions import (
    InvalidPasswordResetPasswordError,
    InvalidPasswordResetTokenError,
    PasswordResetError,
    PasswordResetInfrastructureError,
)
from app.auth.human.password_reset_repository import (
    find_reset_token_by_digest,
    find_user_by_normalized_email_for_update,
    persist_reset_token,
    revoke_open_reset_tokens,
)
from app.auth.human.passwords import hash_password
from app.auth.human.session_repository import (
    find_user_by_id_for_update,
    revoke_all_user_sessions,
)
from app.auth.human.types import CreatedPasswordReset, PasswordResetResult
from app.database import SessionLocal
from app.domain.users import normalize_email

PASSWORD_RESET_ABSOLUTE_LIFETIME = timedelta(minutes=30)
PASSWORD_RESET_TOKEN_ENTROPY_BYTES = 32
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_INVALID_TOKEN_MESSAGE = "Invalid password reset token"
_INVALID_PASSWORD_MESSAGE = "Invalid password reset password"
_UNAVAILABLE_MESSAGE = "Password reset unavailable"


def request_password_reset(*, email: str) -> CreatedPasswordReset | None:
    try:
        try:
            normalized_email = normalize_email(email)
        except (AttributeError, TypeError, ValueError):
            return None
        with SessionLocal.begin() as session:
            user = find_user_by_normalized_email_for_update(
                session, normalized_email=normalized_email,
            )
            if user is None or user.is_active is not True:
                return None
            now = _utc_now()
            revoke_open_reset_tokens(session, user_id=user.id, revoked_at=now)
            token = secrets.token_urlsafe(PASSWORD_RESET_TOKEN_ENTROPY_BYTES)
            expires_at = now + PASSWORD_RESET_ABSOLUTE_LIFETIME
            persist_reset_token(
                session, user_id=user.id, token_digest=_digest_token(token),
                created_at=now, expires_at=expires_at,
            )
            result = CreatedPasswordReset(token, user.normalized_email, expires_at)
        return result
    except Exception:
        raise PasswordResetInfrastructureError(_UNAVAILABLE_MESSAGE) from None


def reset_password(*, token: str, new_password: str) -> PasswordResetResult:
    try:
        if not isinstance(token, str) or _TOKEN_PATTERN.fullmatch(token) is None:
            raise InvalidPasswordResetTokenError(_INVALID_TOKEN_MESSAGE)
        token_digest = _digest_token(token)
        with SessionLocal.begin() as session:
            # Routing only: never lock a reset-token row before its User row.
            candidate = find_reset_token_by_digest(session, token_digest=token_digest)
            if candidate is None:
                raise InvalidPasswordResetTokenError(_INVALID_TOKEN_MESSAGE)
            candidate_id, user_id = candidate.id, candidate.user_id
            user = find_user_by_id_for_update(session, user_id=user_id)
            row = find_reset_token_by_digest(
                session, token_digest=token_digest, for_update=True,
            )
            if (
                user is None or row is None or row.id != candidate_id
                or row.user_id != user.id or user.id != user_id
                or type(user.password_version) is not int or user.password_version <= 0
            ):
                raise PasswordResetInfrastructureError(_UNAVAILABLE_MESSAGE)
            now = _utc_now()
            if (
                row.consumed_at is not None or row.revoked_at is not None
                or now >= row.expires_at or user.is_active is not True
            ):
                raise InvalidPasswordResetTokenError(_INVALID_TOKEN_MESSAGE)
            if row.expires_at <= row.created_at:
                raise PasswordResetInfrastructureError(_UNAVAILABLE_MESSAGE)
            try:
                password_hash = hash_password(new_password)
            except ValueError:
                if not isinstance(new_password, str) or not new_password.strip():
                    raise InvalidPasswordResetPasswordError(_INVALID_PASSWORD_MESSAGE) from None
                # A backend ValueError for a policy-valid password is not a policy rejection.
                raise
            user.password_hash = password_hash
            user.password_version += 1
            user.password_changed_at = now
            row.consumed_at = now
            session.flush()
            revoke_all_user_sessions(session, user_id=user.id, revoked_at=now)
            result = PasswordResetResult(user_id=user.id)
        return result
    except PasswordResetError:
        raise
    except Exception:
        raise PasswordResetInfrastructureError(_UNAVAILABLE_MESSAGE) from None


def _digest_token(token: str) -> bytes:
    return hashlib.sha256(token.encode("ascii")).digest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
