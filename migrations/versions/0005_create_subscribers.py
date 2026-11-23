import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "subscribers",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("email", sa.String(length=254), nullable=False),
        sa.Column("interests", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="pending", nullable=False),
        sa.Column("token", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("confirmation_sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("unsubscribed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_digest_on", sa.Date(), nullable=True),
        sa.CheckConstraint("status IN ('pending', 'active', 'unsubscribed')", name=op.f("ck_subscribers_status_known")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_subscribers")),
        sa.UniqueConstraint("email", name=op.f("uq_subscribers_email")),
        sa.UniqueConstraint("token", name=op.f("uq_subscribers_token")),
    )
    op.create_index(op.f("ix_subscribers_status"), "subscribers", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_subscribers_status"), table_name="subscribers")
    op.drop_table("subscribers")
