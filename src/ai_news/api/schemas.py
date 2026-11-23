from datetime import date, datetime
from typing import Any, Literal

from langchain_core.documents import Document
from pydantic import BaseModel, EmailStr, Field

from ai_news.db.models import DigestRecord, DocumentRecord
from ai_news.scrapers.models import ContentKind, SourceName

SNIPPET_CHARACTERS = 300
SearchMode = Literal["quality", "fast"]
PREVIEW_CHARACTERS = 1000


class SearchHit(BaseModel):
    rank: int
    document_id: int
    doc_key: str = Field(description="Stable id across database rebuilds: source:source_id")
    source: SourceName
    title: str
    url: str = Field(description="Link to the original; for videos, to the matching moment")
    published_at: datetime
    section: str | None
    snippet: str
    relevance: float | None = Field(None, description="Cross-encoder score squashed to 0-1 (quality mode)")
    rerank_score: float | None = Field(None, description="Raw cross-encoder logit (quality mode)")
    similarity: float | None = Field(None, description="Cosine similarity of the best chunk (fast mode)")


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    count: int
    took_ms: float
    stages_ms: dict[str, float]
    results: list[SearchHit]


class DocumentDetail(BaseModel):
    id: int
    doc_key: str
    source: SourceName
    title: str
    url: str
    published_at: datetime
    content_kind: ContentKind
    authors: list[str]
    word_count: int
    preview: str
    metadata: dict[str, Any]


class Health(BaseModel):
    status: str


def snippet(text: str, limit: int = SNIPPET_CHARACTERS) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rsplit(" ", 1)[0] + "…"


def to_hit(rank: int, post: Document) -> SearchHit:
    metadata = post.metadata
    return SearchHit(
        rank=rank,
        document_id=metadata["document_id"],
        doc_key=f"{metadata['source']}:{metadata['source_id']}",
        source=metadata["source"],
        title=metadata["title"],
        url=metadata.get("timestamp_url") or metadata["url"],
        published_at=metadata["published_at"],
        section=metadata.get("section"),
        snippet=snippet(post.page_content),
        relevance=metadata.get("relevance"),
        rerank_score=metadata.get("rerank_score"),
        similarity=metadata.get("similarity") if "rerank_score" not in metadata else None,
    )


def to_detail(record: DocumentRecord) -> DocumentDetail:
    return DocumentDetail(
        id=record.id,
        doc_key=f"{record.source}:{record.source_id}",
        source=record.source,
        title=record.title,
        url=record.url,
        published_at=record.published_at,
        content_kind=record.content_kind,
        authors=record.authors,
        word_count=record.word_count,
        preview=snippet(record.content, PREVIEW_CHARACTERS),
        metadata=record.metadata_,
    )


class DigestDocument(BaseModel):
    id: int
    source: SourceName
    title: str
    url: str


class DigestEntry(BaseModel):
    headline: str
    why_it_matters: str
    documents: list[DigestDocument]


class TokenUsage(BaseModel):
    input_tokens: int = Field(description="Input tokens of every LLM call in the run (query expansion + curation)")
    output_tokens: int
    naive_input_tokens: int = Field(
        description="Tokens one curation prompt would need with every document from the window in full"
    )
    saved: float = Field(description="1 - input_tokens / naive_input_tokens")


class DigestResponse(BaseModel):
    profile: str
    title: str
    digest_date: date
    window_start: datetime
    window_end: datetime
    model: str
    queries: list[str]
    documents_in_window: int
    candidates: int
    items: list[DigestEntry]
    tokens: TokenUsage
    created_at: datetime


class ProfileSummary(BaseModel):
    name: str
    title: str
    interests: str


class QueuedTask(BaseModel):
    task_id: str
    status: str = "queued"


def to_digest(record: DigestRecord, title: str) -> DigestResponse:
    saved = 1 - record.input_tokens / record.naive_input_tokens if record.naive_input_tokens else 0.0
    return DigestResponse(
        profile=record.profile,
        title=title,
        digest_date=record.digest_date,
        window_start=record.window_start,
        window_end=record.window_end,
        model=record.model,
        queries=record.queries,
        documents_in_window=record.documents_in_window,
        candidates=len(record.candidate_ids),
        items=record.items,
        tokens=TokenUsage(
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            naive_input_tokens=record.naive_input_tokens,
            saved=round(saved, 4),
        ),
        created_at=record.created_at,
    )


class SubscriptionRequest(BaseModel):
    email: EmailStr
    interests: str = Field(min_length=3, max_length=500, description="What you want the daily digest to cover")


class SubscriptionResponse(BaseModel):
    status: Literal["check_your_inbox"] = "check_your_inbox"
    message: str = "If this address can receive a digest, a confirmation email is on its way. Click its link to start."
