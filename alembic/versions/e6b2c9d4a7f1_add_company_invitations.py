"""add company invitations

Revision ID: e6b2c9d4a7f1
Revises: 9d7c2a5e4f1b
Create Date: 2026-09-16

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e6b2c9d4a7f1"
down_revision: str | Sequence[str] | None = "9d7c2a5e4f1b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "company_invitations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("normalized_email", sa.String(length=320), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("token_digest", sa.LargeBinary(length=32), nullable=False),
        sa.Column("created_by_user_id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "normalized_email ~ '[^[:space:]]'",
            name="ck_company_invitations_email_not_blank",
        ),
        sa.CheckConstraint(
            "role IN ('salesperson', 'company_admin')",
            name="ck_company_invitations_role_allowed",
        ),
        sa.CheckConstraint(
            "octet_length(token_digest) = 32",
            name="ck_company_invitations_token_digest_sha256_length",
        ),
        sa.CheckConstraint(
            "expires_at > created_at",
            name="ck_company_invitations_expiration_after_creation",
        ),
        sa.CheckConstraint(
            "consumed_at IS NULL OR revoked_at IS NULL",
            name="ck_company_invitations_single_terminal_state",
        ),
        sa.CheckConstraint(
            "consumed_at IS NULL OR consumed_at >= created_at",
            name="ck_company_invitations_consumed_after_creation",
        ),
        sa.CheckConstraint(
            "revoked_at IS NULL OR revoked_at >= created_at",
            name="ck_company_invitations_revoked_after_creation",
        ),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["companies.id"],
            name="fk_company_invitations_company",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"],
            ["users.id"],
            name="fk_company_invitations_creator",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_company_invitations"),
        sa.UniqueConstraint(
            "token_digest",
            name="uq_company_invitations_token_digest",
        ),
    )
    op.create_index(
        "ix_company_invitations_company_email",
        "company_invitations",
        ["company_id", "normalized_email"],
        unique=False,
    )
    op.create_index(
        "ix_company_invitations_created_by_user_id",
        "company_invitations",
        ["created_by_user_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_company_invitations_created_by_user_id",
        table_name="company_invitations",
    )
    op.drop_index(
        "ix_company_invitations_company_email",
        table_name="company_invitations",
    )
    op.drop_table("company_invitations")
