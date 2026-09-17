"""Password-reset persistence; transaction ownership remains in the service."""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import PasswordResetToken, User


def find_user_by_normalized_email_for_update(
    session: Session, *, normalized_email: str,
) -> User | None:
    return session.scalar(
        select(User).where(User.normalized_email == normalized_email)
        .with_for_update().execution_options(populate_existing=True)
    )


def find_reset_token_by_digest(
    session: Session, *, token_digest: bytes, for_update: bool = False,
) -> PasswordResetToken | None:
    statement = select(PasswordResetToken).where(
        PasswordResetToken.token_digest == token_digest,
    )
    if for_update:
        statement = statement.with_for_update().execution_options(populate_existing=True)
    return session.scalar(statement)


def revoke_open_reset_tokens(
    session: Session, *, user_id: int, revoked_at: datetime,
) -> None:
    rows = session.scalars(
        select(PasswordResetToken).where(
            PasswordResetToken.user_id == user_id,
            PasswordResetToken.consumed_at.is_(None),
            PasswordResetToken.revoked_at.is_(None),
        ).order_by(PasswordResetToken.id).with_for_update()
        .execution_options(populate_existing=True)
    )
    for row in rows:
        row.revoked_at = revoked_at
    # Release the partial unique index slot before adding the replacement.
    session.flush()


def persist_reset_token(
    session: Session, *, user_id: int, token_digest: bytes,
    created_at: datetime, expires_at: datetime,
) -> PasswordResetToken:
    row = PasswordResetToken(
        user_id=user_id, token_digest=token_digest,
        created_at=created_at, expires_at=expires_at,
    )
    session.add(row)
    session.flush()
    return row
