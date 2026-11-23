from collections.abc import Iterator, Sequence
from datetime import datetime

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from sqlalchemy import func, or_, select
from sqlalchemy.orm import selectinload

from ai_news.db.models import ChunkEmbeddingRecord, DocumentRecord
from ai_news.db.session import get_session_factory

SEGMENTS_KEY = "segments"


class PostgresDocumentLoader(BaseLoader):
    def __init__(
        self,
        since: datetime | None = None,
        sources: Sequence[str] | None = None,
        needs_indexing: bool = False,
    ) -> None:
        self.since = since
        self.sources = sources
        self.needs_indexing = needs_indexing

    def lazy_load(self) -> Iterator[Document]:
        statement = select(DocumentRecord).options(selectinload(DocumentRecord.transcript)).order_by(DocumentRecord.id)
        if self.since is not None:
            statement = statement.where(DocumentRecord.published_at >= self.since)
        if self.sources:
            statement = statement.where(DocumentRecord.source.in_(self.sources))
        if self.needs_indexing:
            statement = only_unindexed(statement)
        with get_session_factory()() as session:
            for record in session.scalars(statement):
                yield to_langchain_document(record)


def to_langchain_document(record: DocumentRecord) -> Document:
    metadata = {
        "document_id": record.id,
        "source": record.source,
        "source_id": record.source_id,
        "url": record.url,
        "title": record.title,
        "content_kind": record.content_kind,
        "authors": record.authors,
        "published_at": record.published_at.isoformat(),
    }
    if record.transcript is not None:
        metadata[SEGMENTS_KEY] = record.transcript.segments
    return Document(id=str(record.id), page_content=record.content, metadata=metadata)


def only_unindexed(statement):
    """Keeps documents with no chunks yet, or changed since their chunks were last written."""
    indexed = (
        select(ChunkEmbeddingRecord.document_id, func.max(ChunkEmbeddingRecord.updated_at).label("indexed_at"))
        .group_by(ChunkEmbeddingRecord.document_id)
        .subquery()
    )
    return statement.outerjoin(indexed, indexed.c.document_id == DocumentRecord.id).where(
        or_(indexed.c.indexed_at.is_(None), DocumentRecord.updated_at > indexed.c.indexed_at)
    )
