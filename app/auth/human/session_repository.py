"""Persistence operations owned by server-side human sessions."""

from datetime import datetime

from sqlalchemy.orm import Session

from app.models import UserSession


def persist_user_session(
    session: Session,
    *,
    user_id: int,
    membership_id: int,
    company_id: int,
    token_digest: bytes,
    expires_at: datetime,
) -> UserSession:
    user_session = UserSession(
        user_id=user_id,
        membership_id=membership_id,
        company_id=company_id,
        token_digest=token_digest,
        expires_at=expires_at,
    )
    session.add(user_session)
    session.flush()
    return user_session
