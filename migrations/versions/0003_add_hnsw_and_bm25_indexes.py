import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

BM25_CONTENT = "(content::pdb.simple('stemmer=english', 'stopwords_language=english'))"


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
    op.create_index(
        op.f("ix_chunk_embeddings_embedding"),
        "chunk_embeddings",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index(
        "ix_chunk_embeddings_bm25",
        "chunk_embeddings",
        ["id", sa.text(BM25_CONTENT)],
        unique=False,
        postgresql_using="bm25",
        postgresql_with={"key_field": "'id'"},
    )


def downgrade() -> None:
    op.drop_index("ix_chunk_embeddings_bm25", table_name="chunk_embeddings")
    op.drop_index(op.f("ix_chunk_embeddings_embedding"), table_name="chunk_embeddings")
