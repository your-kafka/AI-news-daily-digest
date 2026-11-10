import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import batched
from typing import Any, Self

from sqlalchemy import and_, func, literal_column, not_, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ai_news.db.models import DocumentRecord, IngestRun, RunStatus, TranscriptRecord
from ai_news.scrapers.models import Document, Transcript

BATCH_SIZE = 500
UPDATABLE_COLUMNS = (
    "url",
    "title",
    "content",
    "content_kind",
    "content_hash",
    "word_count",
    "authors",
    "metadata",
    "published_at",
)


@dataclass(frozen=True, slots=True)
class SaveResult:
    inserted: int = 0
    updated: int = 0

    def __add__(self, other: Self) -> Self:
        return type(self)(self.inserted + other.inserted, self.updated + other.updated)


def save_documents(session: Session, documents: Iterable[Document]) -> SaveResult:
    unique_documents = {(document.source, document.source_id): document for document in documents}
    result = SaveResult()
    for batch in batched(unique_documents.values(), BATCH_SIZE):
        result += upsert_documents(session, batch)
    return result


def upsert_documents(session: Session, documents: Sequence[Document]) -> SaveResult:
    statement = insert(DocumentRecord).values([document_values(document) for document in documents])
    incoming = statement.excluded
    content_changed = DocumentRecord.content_hash != incoming.content_hash
    would_lose_transcript = and_(DocumentRecord.content_kind == "transcript", incoming.content_kind != "transcript")
    statement = statement.on_conflict_do_update(
        index_elements=[DocumentRecord.source, DocumentRecord.source_id],
        set_={**{column: incoming[column] for column in UPDATABLE_COLUMNS}, "updated_at": func.now()},
        where=and_(content_changed, not_(would_lose_transcript)),
    ).returning(
        DocumentRecord.id,
        DocumentRecord.source,
        DocumentRecord.source_id,
        literal_column("xmax = 0").label("inserted"),
    )
    rows = session.execute(statement).all()
    by_key = {(document.source, document.source_id): document for document in documents}
    save_transcripts(session, {row.id: by_key[(row.source, row.source_id)] for row in rows})
    inserted = sum(1 for row in rows if row.inserted)
    return SaveResult(inserted=inserted, updated=len(rows) - inserted)


def save_transcripts(session: Session, documents_by_id: dict[int, Document]) -> None:
    values = [
        transcript_values(document_id, document.transcript)
        for document_id, document in documents_by_id.items()
        if document.transcript is not None
    ]
    if not values:
        return
    statement = insert(TranscriptRecord).values(values)
    statement = statement.on_conflict_do_update(
        index_elements=[TranscriptRecord.document_id],
        set_={column: statement.excluded[column] for column in ("language_code", "is_generated", "segments")},
    )
    session.execute(statement)


def document_values(document: Document) -> dict[str, Any]:
    return {
        "source": document.source,
        "source_id": document.source_id,
        "url": document.url,
        "title": document.title,
        "content": document.content,
        "content_kind": document.content_kind,
        "content_hash": content_hash(document),
        "word_count": document.word_count,
        "authors": document.authors,
        "metadata": document.model_dump(mode="json")["metadata"],
        "published_at": document.published_at,
    }


def transcript_values(document_id: int, transcript: Transcript) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "language_code": transcript.language_code,
        "is_generated": transcript.is_generated,
        "segments": [segment.model_dump() for segment in transcript.segments],
    }


def content_hash(document: Document) -> str:
    return hashlib.sha256(f"{document.title}\n{document.content}".encode()).hexdigest()


def start_run(session: Session, source: str, window_hours: float) -> int:
    run = IngestRun(source=source, window_hours=window_hours)
    session.add(run)
    session.flush()
    return run.id


def finish_run(session: Session, run_id: int, found: int, result: SaveResult, error: str | None) -> None:
    session.execute(
        update(IngestRun)
        .where(IngestRun.id == run_id)
        .values(
            status=RunStatus.FAILED if error else RunStatus.SUCCEEDED,
            finished_at=func.now(),
            documents_found=found,
            documents_inserted=result.inserted,
            documents_updated=result.updated,
            error=error,
        )
    )
