"""add user sessions

Revision ID: c8e1f4a6b2d9
Revises: b7c9d2e4f6a8
Create Date: 2026-09-12

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c8e1f4a6b2d9"
down_revision: str | Sequence[str] | None = "b7c9d2e4f6a8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_company_memberships_id_user_company",
        "company_memberships",
        ["id", "user_id", "company_id"],
    )
    op.create_table(
        "user_sessions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("membership_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("token_digest", sa.LargeBinary(length=32), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "expires_at > created_at",
            name="ck_user_sessions_expiration_after_creation",
        ),
        sa.CheckConstraint(
            "octet_length(token_digest) = 32",
            name="ck_user_sessions_token_digest_sha256_length",
        ),
        sa.ForeignKeyConstraint(
            ["membership_id", "user_id", "company_id"],
            [
                "company_memberships.id",
                "company_memberships.user_id",
                "company_memberships.company_id",
            ],
            name="fk_user_sessions_membership_identity",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_user_sessions"),
        sa.UniqueConstraint(
            "token_digest",
            name="uq_user_sessions_token_digest",
        ),
    )
    op.create_index(
        "ix_user_sessions_company_id",
        "user_sessions",
        ["company_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_user_sessions_company_id", table_name="user_sessions")
    op.drop_table("user_sessions")
    op.drop_constraint(
        "uq_company_memberships_id_user_company",
        "company_memberships",
        type_="unique",
    )
