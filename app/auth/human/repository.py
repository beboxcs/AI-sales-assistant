"""Credential persistence operations owned by the human-auth core."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import CompanyMembership, User


def find_user_by_normalized_email(
    session: Session,
    *,
    normalized_email: str,
) -> User | None:
    statement = select(User).where(User.normalized_email == normalized_email)
    return session.execute(statement).scalar_one_or_none()


def find_active_memberships_for_user(
    session: Session,
    *,
    user_id: int,
) -> list[CompanyMembership]:
    statement = (
        select(CompanyMembership)
        .where(
            CompanyMembership.user_id == user_id,
            CompanyMembership.is_active.is_(True),
        )
        .order_by(CompanyMembership.id)
    )
    return list(session.execute(statement).scalars())
