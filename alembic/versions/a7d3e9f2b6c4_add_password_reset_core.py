"""add password reset tokens and credential generations

Revision ID: a7d3e9f2b6c4
Revises: e6b2c9d4a7f1
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a7d3e9f2b6c4"
down_revision: str | Sequence[str] | None = "e6b2c9d4a7f1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("users", sa.Column("password_version", sa.Integer(), nullable=True))
    op.execute("UPDATE users SET password_version = 1")
    op.alter_column("users", "password_version", nullable=False, server_default=sa.text("1"))
    op.create_check_constraint("ck_users_password_version_positive", "users", "password_version > 0")
    op.create_table(
        "password_reset_tokens",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("token_digest", sa.LargeBinary(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_password_reset_tokens"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_password_reset_tokens_user", ondelete="RESTRICT"),
        sa.UniqueConstraint("token_digest", name="uq_password_reset_tokens_token_digest"),
        sa.CheckConstraint("octet_length(token_digest) = 32", name="ck_password_reset_tokens_digest_sha256_length"),
        sa.CheckConstraint("expires_at > created_at", name="ck_password_reset_tokens_expiration_after_creation"),
        sa.CheckConstraint("consumed_at IS NULL OR revoked_at IS NULL", name="ck_password_reset_tokens_single_terminal_state"),
        sa.CheckConstraint("consumed_at IS NULL OR consumed_at >= created_at", name="ck_password_reset_tokens_consumed_after_creation"),
        sa.CheckConstraint("revoked_at IS NULL OR revoked_at >= created_at", name="ck_password_reset_tokens_revoked_after_creation"),
    )
    op.create_index("ix_password_reset_tokens_user_id", "password_reset_tokens", ["user_id"])
    op.create_index(
        "uq_password_reset_tokens_open_user", "password_reset_tokens", ["user_id"],
        unique=True, postgresql_where=sa.text("consumed_at IS NULL AND revoked_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_password_reset_tokens_open_user", table_name="password_reset_tokens")
    op.drop_index("ix_password_reset_tokens_user_id", table_name="password_reset_tokens")
    op.drop_table("password_reset_tokens")
    op.drop_constraint("ck_users_password_version_positive", "users", type_="check")
    op.drop_column("users", "password_version")
