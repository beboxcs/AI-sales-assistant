"""add user session csrf token digest

Revision ID: 9d7c2a5e4f1b
Revises: f4a7d9c2e1b6
Create Date: 2026-09-14

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "9d7c2a5e4f1b"
down_revision: str | Sequence[str] | None = "f4a7d9c2e1b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "user_sessions",
        sa.Column(
            "csrf_token_digest",
            sa.LargeBinary(length=32),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("user_sessions", "csrf_token_digest")
