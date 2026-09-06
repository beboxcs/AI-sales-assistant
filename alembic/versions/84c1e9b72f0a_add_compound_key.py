"""add compound key

Revision ID: 84c1e9b72f0a
Revises: 3ca60b9db9e3
Create Date: 2026-09-05

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "84c1e9b72f0a"
down_revision: str | Sequence[str] | None = "3ca60b9db9e3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "compounds",
        sa.Column("compound_key", sa.String(), nullable=False),
    )
    op.create_unique_constraint(
        "uq_compounds_company_compound_key",
        "compounds",
        ["company_id", "compound_key"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_compounds_company_compound_key",
        "compounds",
        type_="unique",
    )
    op.drop_column("compounds", "compound_key")
