"""Persistence operations owned by server-side human sessions."""

from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.models import CompanyMembership, User, UserSession


def persist_user_session(
    session: Session,
    *,
    user_id: int,
    membership_id: int,
    company_id: int,
    token_digest: bytes,
    expires_at: datetime,
    last_activity_at: datetime,
) -> UserSession:
    user_session = UserSession(
        user_id=user_id,
        membership_id=membership_id,
        company_id=company_id,
        token_digest=token_digest,
        expires_at=expires_at,
        last_activity_at=last_activity_at,
    )
    session.add(user_session)
    session.flush()
    return user_session


def find_user_session_by_token_digest_for_update(
    session: Session,
    *,
    token_digest: bytes,
) -> UserSession | None:
    statement = (
        select(UserSession)
        .where(UserSession.token_digest == token_digest)
        .with_for_update()
    )
    return session.execute(statement).scalar_one_or_none()


def find_user_by_id(
    session: Session,
    *,
    user_id: int,
) -> User | None:
    return session.get(User, user_id)


def find_user_by_id_for_update(session: Session, *, user_id: int) -> User | None:
    return session.scalar(
        select(User).where(User.id == user_id).with_for_update()
        .execution_options(populate_existing=True)
    )


def find_membership_by_id_for_update(
    session: Session, *, membership_id: int,
) -> CompanyMembership | None:
    return session.scalar(
        select(CompanyMembership).where(CompanyMembership.id == membership_id)
        .with_for_update().execution_options(populate_existing=True)
    )


def revoke_all_user_sessions(
    session: Session, *, user_id: int, revoked_at: datetime,
) -> None:
    """Revoke across all companies inside the caller's transaction."""
    session.execute(
        update(UserSession)
        .where(UserSession.user_id == user_id, UserSession.revoked_at.is_(None))
        .values(revoked_at=revoked_at)
    )


def find_membership_by_id(
    session: Session,
    *,
    membership_id: int,
) -> CompanyMembership | None:
    return session.get(CompanyMembership, membership_id)


def refresh_user_session_activity(
    session: Session,
    *,
    user_session: UserSession,
    last_activity_at: datetime,
) -> None:
    user_session.last_activity_at = last_activity_at
    session.flush()


def revoke_user_session(
    session: Session,
    *,
    user_session: UserSession,
    revoked_at: datetime,
) -> None:
    user_session.revoked_at = revoked_at
    session.flush()


def replace_user_session_csrf_digest(
    session: Session,
    *,
    user_session: UserSession,
    csrf_token_digest: bytes,
) -> None:
    user_session.csrf_token_digest = csrf_token_digest
    session.flush()
