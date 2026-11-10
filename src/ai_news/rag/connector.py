from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import batched
from typing import Any

from langchain_core.documents import Document
from sqlalchemy import Delete, Insert, delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from ai_news.db.models import ChunkEmbeddingRecord, DocumentRecord
from ai_news.db.session import get_session_factory

BATCH_SIZE = 500
PROMOTED_KEYS = frozenset(
    {"document_id", "chunk_index", "source", "published_at", "token_count", "section", "start_seconds", "end_seconds"}
)
UPDATABLE_COLUMNS = (
    "document_id",
    "chunk_index",
    "source",
    "published_at",
    "content",
    "token_count",
    "section",
    "start_seconds",
    "end_seconds",
    "metadata",
    "embedding_model",
    "content_hash",
    "embedding",
)


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    chunk: Document
    embedding: list[float]
    content_hash: str
    model: str


@dataclass(frozen=True, slots=True)
class StoreResult:
    saved: int
    removed: int


class EmbeddingStore:
    def __init__(self, session_factory: sessionmaker[Session] | None = None) -> None:
        self._session_factory = session_factory or get_session_factory()

    def existing_documents(self, document_ids: Iterable[int]) -> set[int]:
        with self._session_factory() as session:
            return {
                document_id
                for batch in batched(document_ids, BATCH_SIZE)
                for document_id in session.scalars(select(DocumentRecord.id).where(DocumentRecord.id.in_(batch)))
            }

    def stored_hashes(self, chunk_ids: Iterable[str]) -> dict[str, str]:
        statement = select(ChunkEmbeddingRecord.id, ChunkEmbeddingRecord.content_hash)
        with self._session_factory() as session:
            return {
                chunk_id: content_hash
                for batch in batched(chunk_ids, BATCH_SIZE)
                for chunk_id, content_hash in session.execute(statement.where(ChunkEmbeddingRecord.id.in_(batch)))
            }

    def save(self, embedded: Sequence[EmbeddedChunk], chunk_ids_by_document: Mapping[int, set[str]]) -> StoreResult:
        removed = 0
        with self._session_factory.begin() as session:
            for batch in batched(embedded, BATCH_SIZE):
                session.execute(upsert_statement(batch))
            for document_ids in batched(chunk_ids_by_document, BATCH_SIZE):
                result = session.execute(stale_chunks_statement(document_ids, chunk_ids_by_document))
                removed += result.rowcount
        return StoreResult(saved=len(embedded), removed=removed)

    def count(self) -> int:
        with self._session_factory() as session:
            return session.scalar(select(func.count()).select_from(ChunkEmbeddingRecord)) or 0


def upsert_statement(batch: Sequence[EmbeddedChunk]) -> Insert:
    statement = insert(ChunkEmbeddingRecord).values([row_values(item) for item in batch])
    return statement.on_conflict_do_update(
        index_elements=[ChunkEmbeddingRecord.id],
        set_={**{column: statement.excluded[column] for column in UPDATABLE_COLUMNS}, "updated_at": func.now()},
    )


def stale_chunks_statement(document_ids: Sequence[int], chunk_ids_by_document: Mapping[int, set[str]]) -> Delete:
    keep = [chunk_id for document_id in document_ids for chunk_id in chunk_ids_by_document[document_id]]
    return delete(ChunkEmbeddingRecord).where(
        ChunkEmbeddingRecord.document_id.in_(document_ids),
        ChunkEmbeddingRecord.id.not_in(keep),
    )


def row_values(item: EmbeddedChunk) -> dict[str, Any]:
    metadata = item.chunk.metadata
    return {
        "id": item.chunk.id,
        "document_id": metadata["document_id"],
        "chunk_index": metadata["chunk_index"],
        "source": metadata["source"],
        "published_at": datetime.fromisoformat(metadata["published_at"]),
        "content": item.chunk.page_content,
        "token_count": metadata["token_count"],
        "section": metadata.get("section"),
        "start_seconds": metadata.get("start_seconds"),
        "end_seconds": metadata.get("end_seconds"),
        "metadata": {key: value for key, value in metadata.items() if key not in PROMOTED_KEYS},
        "embedding_model": item.model,
        "content_hash": item.content_hash,
        "embedding": item.embedding,
    }
