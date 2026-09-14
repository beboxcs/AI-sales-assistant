"""add user session last activity

Revision ID: f4a7d9c2e1b6
Revises: c8e1f4a6b2d9
Create Date: 2026-09-14

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f4a7d9c2e1b6"
down_revision: str | Sequence[str] | None = "c8e1f4a6b2d9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "user_sessions",
        sa.Column(
            "last_activity_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.execute(
        "UPDATE user_sessions SET last_activity_at = created_at "
        "WHERE last_activity_at IS NULL"
    )
    op.alter_column(
        "user_sessions",
        "last_activity_at",
        existing_type=sa.DateTime(timezone=True),
        nullable=False,
    )


def downgrade() -> None:
    op.drop_column("user_sessions", "last_activity_at")
