"""Company user invitation lifecycle and security boundary."""

import hashlib
import re
import secrets
from datetime import datetime, timedelta, timezone

from app.auth.human.exceptions import (
    InvitationAuthorizationError,
    InvitationConflictError,
    InvitationError,
    InvitationInfrastructureError,
    InvalidInvitationError,
    InvalidInvitationPasswordError,
)
from app.auth.human.invitation_repository import (
    acquire_invitation_identity_lock,
    find_invitation_by_token_digest,
    find_invitation_by_token_digest_for_update,
    find_invitations_by_identity_for_update,
    find_membership_by_user_and_company_for_update,
    find_user_by_id_for_update,
    find_user_by_normalized_email_for_update,
    persist_invitation,
    persist_membership,
    persist_user,
)
from app.auth.human.passwords import hash_password
from app.auth.human.types import AcceptedInvitation, CreatedInvitation
from app.database import SessionLocal
from app.domain.enums import MembershipRole
from app.domain.users import normalize_email

INVITATION_ABSOLUTE_LIFETIME = timedelta(hours=24)
INVITATION_TOKEN_ENTROPY_BYTES = 32

_INVALID_INVITATION_MESSAGE = "Invalid invitation"
_AUTHORIZATION_MESSAGE = "Invitation operation not permitted"
_CONFLICT_MESSAGE = "Invitation conflicts with existing identity state"
_INVALID_PASSWORD_MESSAGE = "Invitation password is invalid"
_INFRASTRUCTURE_MESSAGE = "Invitation operation unavailable"
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}\Z")


def create_invitation(
    *,
    company_id: int,
    created_by_user_id: int,
    email: str,
    role: MembershipRole,
) -> CreatedInvitation:
    try:
        normalized_email = _normalize_invited_email(email=email)
        invited_role = _validate_requested_role(role=role)
        _validate_positive_identifier(value=company_id)
        _validate_positive_identifier(value=created_by_user_id)

        lock_key = _derive_invitation_identity_lock_key(
            company_id=company_id,
            normalized_email=normalized_email,
        )

        with SessionLocal.begin() as session:
            acquire_invitation_identity_lock(session, lock_key=lock_key)
            created_at = _utc_now()
            expires_at = created_at + INVITATION_ABSOLUTE_LIFETIME
            _authorize_creator(
                session,
                company_id=company_id,
                created_by_user_id=created_by_user_id,
            )
            _reject_active_membership_conflict(
                session,
                company_id=company_id,
                normalized_email=normalized_email,
            )

            for existing in find_invitations_by_identity_for_update(
                session,
                company_id=company_id,
                normalized_email=normalized_email,
            ):
                if (
                    existing.consumed_at is None
                    and existing.revoked_at is None
                    and created_at < existing.expires_at
                ):
                    existing.revoked_at = created_at

            token = _generate_invitation_token()
            token_digest = _digest_invitation_token(token=token)
            invitation = persist_invitation(
                session,
                company_id=company_id,
                normalized_email=normalized_email,
                role=invited_role.value,
                token_digest=token_digest,
                created_by_user_id=created_by_user_id,
                created_at=created_at,
                expires_at=expires_at,
            )
            invitation_id = invitation.id

        return CreatedInvitation(
            invitation_id=invitation_id,
            company_id=company_id,
            normalized_email=normalized_email,
            role=invited_role,
            token=token,
            expires_at=expires_at,
        )
    except InvitationError:
        raise
    except Exception:
        raise InvitationInfrastructureError(_INFRASTRUCTURE_MESSAGE) from None


def accept_invitation(
    *,
    token: str,
    password: str,
) -> AcceptedInvitation:
    token_digest = _derive_valid_invitation_token_digest(token=token)

    try:
        with SessionLocal.begin() as session:
            candidate = find_invitation_by_token_digest(
                session,
                token_digest=token_digest,
            )
            if candidate is None:
                raise InvalidInvitationError(_INVALID_INVITATION_MESSAGE)
            company_id, normalized_email = (
                _validate_persisted_invitation_identity(
                    invitation=candidate,
                )
            )
            candidate_id = candidate.id
            lock_key = _derive_invitation_identity_lock_key(
                company_id=company_id,
                normalized_email=normalized_email,
            )
            acquire_invitation_identity_lock(session, lock_key=lock_key)

            invitation = find_invitation_by_token_digest_for_update(
                session,
                token_digest=token_digest,
            )
            if invitation is None:
                raise InvalidInvitationError(_INVALID_INVITATION_MESSAGE)
            if (
                invitation.id != candidate_id
                or invitation.company_id != company_id
                or invitation.normalized_email != normalized_email
            ):
                raise InvitationInfrastructureError(_INFRASTRUCTURE_MESSAGE)

            now = _utc_now()
            role = _validate_persisted_invitation(
                invitation=invitation,
                now=now,
            )
            user = find_user_by_normalized_email_for_update(
                session,
                normalized_email=invitation.normalized_email,
            )
            if user is not None and user.is_active is not True:
                raise InvitationConflictError(_CONFLICT_MESSAGE)

            if user is None:
                try:
                    password_hash = hash_password(password)
                except ValueError:
                    raise InvalidInvitationPasswordError(
                        _INVALID_PASSWORD_MESSAGE
                    ) from None
                user = persist_user(
                    session,
                    normalized_email=invitation.normalized_email,
                    password_hash=password_hash,
                    password_changed_at=now,
                )

            existing_membership = find_membership_by_user_and_company_for_update(
                session,
                user_id=user.id,
                company_id=invitation.company_id,
            )
            if existing_membership is not None:
                raise InvitationConflictError(_CONFLICT_MESSAGE)

            membership = persist_membership(
                session,
                user_id=user.id,
                company_id=invitation.company_id,
                role=role.value,
            )
            invitation.consumed_at = now
            session.flush()

            accepted = AcceptedInvitation(
                user_id=user.id,
                company_id=invitation.company_id,
                membership_id=membership.id,
                role=role,
            )
    except InvitationError:
        raise
    except Exception:
        raise InvitationInfrastructureError(_INFRASTRUCTURE_MESSAGE) from None

    return accepted


def _authorize_creator(
    session,
    *,
    company_id: int,
    created_by_user_id: int,
) -> None:
    creator = find_user_by_id_for_update(
        session,
        user_id=created_by_user_id,
    )
    if creator is None or creator.is_active is not True:
        raise InvitationAuthorizationError(_AUTHORIZATION_MESSAGE)

    membership = find_membership_by_user_and_company_for_update(
        session,
        user_id=creator.id,
        company_id=company_id,
    )
    if (
        membership is None
        or membership.is_active is not True
        or membership.role != MembershipRole.COMPANY_ADMIN.value
    ):
        raise InvitationAuthorizationError(_AUTHORIZATION_MESSAGE)


def _reject_active_membership_conflict(
    session,
    *,
    company_id: int,
    normalized_email: str,
) -> None:
    invited_user = find_user_by_normalized_email_for_update(
        session,
        normalized_email=normalized_email,
    )
    if invited_user is None:
        return
    membership = find_membership_by_user_and_company_for_update(
        session,
        user_id=invited_user.id,
        company_id=company_id,
    )
    if membership is not None and membership.is_active is True:
        raise InvitationConflictError(_CONFLICT_MESSAGE)


def _validate_persisted_invitation(*, invitation, now: datetime) -> MembershipRole:
    if (
        invitation.consumed_at is not None
        or invitation.revoked_at is not None
        or now >= invitation.expires_at
    ):
        raise InvalidInvitationError(_INVALID_INVITATION_MESSAGE)
    _validate_persisted_invitation_identity(invitation=invitation)
    try:
        role = MembershipRole(invitation.role)
    except (AttributeError, TypeError, ValueError):
        raise InvitationInfrastructureError(_INFRASTRUCTURE_MESSAGE) from None
    if not isinstance(invitation.expires_at, datetime):
        raise InvitationInfrastructureError(_INFRASTRUCTURE_MESSAGE)
    if invitation.expires_at.tzinfo is None:
        raise InvitationInfrastructureError(_INFRASTRUCTURE_MESSAGE)
    return role


def _validate_persisted_invitation_identity(
    *,
    invitation,
) -> tuple[int, str]:
    try:
        normalized_email = normalize_email(invitation.normalized_email)
    except (AttributeError, TypeError, ValueError):
        raise InvitationInfrastructureError(_INFRASTRUCTURE_MESSAGE) from None
    if normalized_email != invitation.normalized_email:
        raise InvitationInfrastructureError(_INFRASTRUCTURE_MESSAGE)
    if (
        not isinstance(invitation.company_id, int)
        or isinstance(invitation.company_id, bool)
        or invitation.company_id < 1
    ):
        raise InvitationInfrastructureError(_INFRASTRUCTURE_MESSAGE)
    return invitation.company_id, normalized_email


def _normalize_invited_email(*, email: str) -> str:
    try:
        normalized_email = normalize_email(email)
    except (AttributeError, TypeError, ValueError):
        raise InvitationConflictError(_CONFLICT_MESSAGE) from None
    if len(normalized_email) > 320:
        raise InvitationConflictError(_CONFLICT_MESSAGE)
    return normalized_email


def _validate_requested_role(*, role: MembershipRole) -> MembershipRole:
    if not isinstance(role, MembershipRole):
        raise InvitationConflictError(_CONFLICT_MESSAGE)
    return role


def _validate_positive_identifier(*, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise InvitationConflictError(_CONFLICT_MESSAGE)


def _generate_invitation_token() -> str:
    return secrets.token_urlsafe(INVITATION_TOKEN_ENTROPY_BYTES)


def _digest_invitation_token(*, token: str) -> bytes:
    return hashlib.sha256(token.encode("ascii")).digest()


def _derive_valid_invitation_token_digest(*, token: str) -> bytes:
    if not isinstance(token, str) or _TOKEN_PATTERN.fullmatch(token) is None:
        raise InvalidInvitationError(_INVALID_INVITATION_MESSAGE)
    try:
        return _digest_invitation_token(token=token)
    except (AttributeError, TypeError, UnicodeEncodeError):
        raise InvalidInvitationError(_INVALID_INVITATION_MESSAGE) from None
    except Exception:
        raise InvitationInfrastructureError(_INFRASTRUCTURE_MESSAGE) from None


def _derive_invitation_identity_lock_key(
    *,
    company_id: int,
    normalized_email: str,
) -> int:
    identity_digest = hashlib.sha256(
        f"{company_id}\0{normalized_email}".encode("utf-8")
    ).digest()
    return int.from_bytes(identity_digest[:8], "big", signed=True)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
