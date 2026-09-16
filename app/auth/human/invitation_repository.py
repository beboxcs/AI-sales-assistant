"""Persistence operations owned by company invitation security core."""

from datetime import datetime

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.models import CompanyInvitation, CompanyMembership, User


def acquire_invitation_identity_lock(
    session: Session,
    *,
    lock_key: int,
) -> None:
    session.execute(
        text("SELECT pg_advisory_xact_lock(:lock_key)"),
        {"lock_key": lock_key},
    )


def find_user_by_id_for_update(
    session: Session,
    *,
    user_id: int,
) -> User | None:
    statement = select(User).where(User.id == user_id).with_for_update()
    return session.execute(statement).scalar_one_or_none()


def find_user_by_normalized_email_for_update(
    session: Session,
    *,
    normalized_email: str,
) -> User | None:
    statement = (
        select(User)
        .where(User.normalized_email == normalized_email)
        .with_for_update()
    )
    return session.execute(statement).scalar_one_or_none()


def find_membership_by_user_and_company_for_update(
    session: Session,
    *,
    user_id: int,
    company_id: int,
) -> CompanyMembership | None:
    statement = (
        select(CompanyMembership)
        .where(
            CompanyMembership.user_id == user_id,
            CompanyMembership.company_id == company_id,
        )
        .with_for_update()
    )
    return session.execute(statement).scalar_one_or_none()


def find_invitations_by_identity_for_update(
    session: Session,
    *,
    company_id: int,
    normalized_email: str,
) -> list[CompanyInvitation]:
    statement = (
        select(CompanyInvitation)
        .where(
            CompanyInvitation.company_id == company_id,
            CompanyInvitation.normalized_email == normalized_email,
        )
        .order_by(CompanyInvitation.id)
        .with_for_update()
    )
    return list(session.execute(statement).scalars())


def find_invitation_by_token_digest_for_update(
    session: Session,
    *,
    token_digest: bytes,
) -> CompanyInvitation | None:
    statement = (
        select(CompanyInvitation)
        .where(CompanyInvitation.token_digest == token_digest)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    return session.execute(statement).scalar_one_or_none()


def find_invitation_by_token_digest(
    session: Session,
    *,
    token_digest: bytes,
) -> CompanyInvitation | None:
    statement = select(CompanyInvitation).where(
        CompanyInvitation.token_digest == token_digest
    )
    return session.execute(statement).scalar_one_or_none()


def persist_invitation(
    session: Session,
    *,
    company_id: int,
    normalized_email: str,
    role: str,
    token_digest: bytes,
    created_by_user_id: int,
    created_at: datetime,
    expires_at: datetime,
) -> CompanyInvitation:
    invitation = CompanyInvitation(
        company_id=company_id,
        normalized_email=normalized_email,
        role=role,
        token_digest=token_digest,
        created_by_user_id=created_by_user_id,
        created_at=created_at,
        expires_at=expires_at,
    )
    session.add(invitation)
    session.flush()
    return invitation


def persist_user(
    session: Session,
    *,
    normalized_email: str,
    password_hash: str,
    password_changed_at: datetime,
) -> User:
    user = User(
        normalized_email=normalized_email,
        password_hash=password_hash,
        password_changed_at=password_changed_at,
    )
    session.add(user)
    session.flush()
    return user


def persist_membership(
    session: Session,
    *,
    user_id: int,
    company_id: int,
    role: str,
) -> CompanyMembership:
    membership = CompanyMembership(
        user_id=user_id,
        company_id=company_id,
        role=role,
    )
    session.add(membership)
    session.flush()
    return membership
