"""add users and company memberships

Revision ID: b7c9d2e4f6a8
Revises: d4e8f2a7c1b3
Create Date: 2026-09-10

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b7c9d2e4f6a8"
down_revision: str | Sequence[str] | None = "d4e8f2a7c1b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("normalized_email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column(
            "is_active",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
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
        sa.Column(
            "password_changed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.CheckConstraint(
            "normalized_email ~ '[^[:space:]]'",
            name="ck_users_normalized_email_not_blank",
        ),
        sa.CheckConstraint(
            "password_hash ~ '[^[:space:]]'",
            name="ck_users_password_hash_not_blank",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint(
            "normalized_email",
            name="uq_users_normalized_email",
        ),
    )

    op.create_table(
        "company_memberships",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column(
            "is_active",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
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
        sa.Column("deactivated_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "role IN ('salesperson', 'company_admin')",
            name="ck_company_memberships_role_allowed",
        ),
        sa.CheckConstraint(
            "deactivated_at IS NULL OR is_active = false",
            name="ck_company_memberships_deactivated_inactive",
        ),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["companies.id"],
            name="fk_company_memberships_company",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_company_memberships_user",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_company_memberships"),
        sa.UniqueConstraint(
            "user_id",
            "company_id",
            name="uq_company_memberships_user_company",
        ),
    )
    op.create_index(
        "ix_company_memberships_company_id",
        "company_memberships",
        ["company_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_company_memberships_company_id",
        table_name="company_memberships",
    )
    op.drop_table("company_memberships")
    op.drop_table("users")
