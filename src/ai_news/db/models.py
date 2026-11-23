from collections.abc import Iterable
from datetime import date, datetime
from enum import StrEnum
from typing import Any, get_args

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ai_news.db.base import Base
from ai_news.scrapers.models import ContentKind, SourceName

SOURCE_NAMES = get_args(SourceName)
CONTENT_KINDS = get_args(ContentKind)
EMBEDDING_DIMENSIONS = 384
BM25_INDEX_NAME = "ix_chunk_embeddings_bm25"
BM25_CONTENT = "(content::pdb.simple('stemmer=english', 'stopwords_language=english'))"


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def one_of(column: str, values: Iterable[str]) -> str:
    return f"{column} IN ({', '.join(repr(str(value)) for value in values)})"


class DocumentRecord(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("source", "source_id"),
        CheckConstraint(one_of("source", SOURCE_NAMES), name="source_known"),
        CheckConstraint(one_of("content_kind", CONTENT_KINDS), name="content_kind_known"),
        Index(None, "published_at"),
        Index(None, "source", "published_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    source: Mapped[str] = mapped_column(String(32))
    source_id: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    content_kind: Mapped[str] = mapped_column(String(16))
    content_hash: Mapped[str] = mapped_column(String(64))
    word_count: Mapped[int] = mapped_column(Integer)
    authors: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'"))
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, server_default=text("'{}'::jsonb"))
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    transcript: Mapped["TranscriptRecord | None"] = relationship(back_populates="document", passive_deletes=True)


class TranscriptRecord(Base):
    __tablename__ = "transcripts"

    document_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    language_code: Mapped[str] = mapped_column(String(16))
    is_generated: Mapped[bool]
    segments: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)

    document: Mapped[DocumentRecord] = relationship(back_populates="transcript")


class IngestRun(Base):
    __tablename__ = "ingest_runs"
    __table_args__ = (
        CheckConstraint(one_of("source", SOURCE_NAMES), name="source_known"),
        CheckConstraint(one_of("status", RunStatus), name="status_known"),
        Index(None, "source", "started_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    source: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), server_default=RunStatus.RUNNING.value)
    window_hours: Mapped[float] = mapped_column(Float)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    documents_found: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    documents_inserted: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    documents_updated: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    error: Mapped[str | None] = mapped_column(Text)


class ChunkEmbeddingRecord(Base):
    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index"),
        CheckConstraint(one_of("source", SOURCE_NAMES), name="source_known"),
        Index(None, "source", "published_at"),
        Index(None, "embedding", postgresql_using="hnsw", postgresql_ops={"embedding": "vector_cosine_ops"}),
        Index(
            BM25_INDEX_NAME, "id", text(BM25_CONTENT), postgresql_using="bm25", postgresql_with={"key_field": "'id'"}
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(32))
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(Text)
    start_seconds: Mapped[float | None] = mapped_column(Float)
    end_seconds: Mapped[float | None] = mapped_column(Float)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, server_default=text("'{}'::jsonb"))
    embedding_model: Mapped[str] = mapped_column(String(128))
    content_hash: Mapped[str] = mapped_column(String(64))
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SubscriberStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    UNSUBSCRIBED = "unsubscribed"


class Subscriber(Base):
    """Someone who gets a personal daily digest by email, after confirming the address (double opt-in)."""

    __tablename__ = "subscribers"
    __table_args__ = (
        UniqueConstraint("email"),
        UniqueConstraint("token"),
        CheckConstraint(one_of("status", SubscriberStatus), name="status_known"),
        Index(None, "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    email: Mapped[str] = mapped_column(String(254))
    interests: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), server_default=SubscriberStatus.PENDING.value)
    token: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    confirmation_sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    unsubscribed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_digest_on: Mapped[date | None] = mapped_column(Date)


class DigestRecord(Base):
    """One curated digest per profile per day, with the token accounting behind it."""

    __tablename__ = "digests"
    __table_args__ = (UniqueConstraint("profile", "digest_date"), Index(None, "digest_date"))

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    profile: Mapped[str] = mapped_column(String(64))
    digest_date: Mapped[date] = mapped_column(Date)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    model: Mapped[str] = mapped_column(String(128))
    queries: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'"))
    items: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
    candidate_ids: Mapped[list[int]] = mapped_column(ARRAY(BigInteger), server_default=text("'{}'"))
    documents_in_window: Mapped[int] = mapped_column(Integer)
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    naive_input_tokens: Mapped[int] = mapped_column(Integer)
    duration_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
