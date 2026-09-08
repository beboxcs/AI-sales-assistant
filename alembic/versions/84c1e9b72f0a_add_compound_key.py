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

RECONCILIATION_ERROR = (
    "Cannot add compounds.compound_key because legacy compound rows exist. "
    "Authoritative compound_key reconciliation is required before upgrading."
)
RECONCILIATION_ERROR_SQL = RECONCILIATION_ERROR.replace("'", "''")


def upgrade() -> None:
    op.execute(sa.text("LOCK TABLE compounds IN ACCESS EXCLUSIVE MODE"))
    op.execute(
        sa.text(
            f"""
            DO $compound_key_guard$
            BEGIN
                IF EXISTS (SELECT 1 FROM compounds) THEN
                    RAISE EXCEPTION USING MESSAGE = '{RECONCILIATION_ERROR_SQL}';
                END IF;
            END
            $compound_key_guard$
            """
        )
    )
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
