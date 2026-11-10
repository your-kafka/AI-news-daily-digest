import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_kind", sa.String(length=16), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("word_count", sa.Integer(), nullable=False),
        sa.Column("authors", postgresql.ARRAY(sa.Text()), server_default=sa.text("'{}'"), nullable=False),
        sa.Column(
            "metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "content_kind IN ('transcript', 'article', 'summary', 'abstract')",
            name=op.f("ck_documents_content_kind_known"),
        ),
        sa.CheckConstraint(
            "source IN ('youtube', 'anthropic', 'openai', 'deepmind', 'huggingface')",
            name=op.f("ck_documents_source_known"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint("source", "source_id", name=op.f("uq_documents_source_source_id")),
    )
    op.create_index(op.f("ix_documents_published_at"), "documents", ["published_at"], unique=False)
    op.create_index(op.f("ix_documents_source_published_at"), "documents", ["source", "published_at"], unique=False)
    op.create_table(
        "ingest_runs",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="running", nullable=False),
        sa.Column("window_hours", sa.Float(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("documents_found", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("documents_inserted", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("documents_updated", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "source IN ('youtube', 'anthropic', 'openai', 'deepmind', 'huggingface')",
            name=op.f("ck_ingest_runs_source_known"),
        ),
        sa.CheckConstraint("status IN ('running', 'succeeded', 'failed')", name=op.f("ck_ingest_runs_status_known")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ingest_runs")),
    )
    op.create_index(op.f("ix_ingest_runs_source_started_at"), "ingest_runs", ["source", "started_at"], unique=False)
    op.create_table(
        "transcripts",
        sa.Column("document_id", sa.BigInteger(), nullable=False),
        sa.Column("language_code", sa.String(length=16), nullable=False),
        sa.Column("is_generated", sa.Boolean(), nullable=False),
        sa.Column("segments", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], name=op.f("fk_transcripts_document_id_documents"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("document_id", name=op.f("pk_transcripts")),
    )


def downgrade() -> None:
    op.drop_table("transcripts")
    op.drop_index(op.f("ix_ingest_runs_source_started_at"), table_name="ingest_runs")
    op.drop_table("ingest_runs")
    op.drop_index(op.f("ix_documents_source_published_at"), table_name="documents")
    op.drop_index(op.f("ix_documents_published_at"), table_name="documents")
    op.drop_table("documents")
