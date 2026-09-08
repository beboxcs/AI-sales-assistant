"""add api credentials

Revision ID: d4e8f2a7c1b3
Revises: 84c1e9b72f0a
Create Date: 2026-09-08

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d4e8f2a7c1b3"
down_revision: str | Sequence[str] | None = "84c1e9b72f0a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "api_credentials",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("key_id", sa.String(length=64), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("secret_hash", sa.LargeBinary(length=32), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
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
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "key_id ~ '[^[:space:]]'",
            name="ck_api_credentials_key_id_not_blank",
        ),
        sa.CheckConstraint(
            "name ~ '[^[:space:]]'",
            name="ck_api_credentials_name_not_blank",
        ),
        sa.CheckConstraint(
            "revoked_at IS NULL OR is_active = false",
            name="ck_api_credentials_revoked_inactive",
        ),
        sa.CheckConstraint(
            "octet_length(secret_hash) = 32",
            name="ck_api_credentials_secret_hash_sha256_length",
        ),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["companies.id"],
            name="fk_api_credentials_company",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_api_credentials"),
        sa.UniqueConstraint("key_id", name="uq_api_credentials_key_id"),
    )
    op.create_index(
        "ix_api_credentials_company_id",
        "api_credentials",
        ["company_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_api_credentials_company_id",
        table_name="api_credentials",
    )
    op.drop_table("api_credentials")
