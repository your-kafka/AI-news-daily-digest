from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from itertools import zip_longest
from typing import Any

from langchain_core.documents import Document
from sqlalchemy import func, select

from ai_news.db.models import ChunkEmbeddingRecord, DocumentRecord
from ai_news.db.session import get_session_factory
from ai_news.evaluation.dataset import (
    POOL_FILE,
    QUERIES_FILE,
    SNAPSHOT_FILE,
    EvalQuery,
    PooledDocument,
    document_key,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ai_news.log_config import configure_logging
from ai_news.rag.pipeline import SearchPipeline
from ai_news.rag.ranking import pool, reciprocal_rank_fusion
from ai_news.rag.retriever import Bm25Retriever, PgVectorRetriever

SYSTEMS = ("bm25", "dense", "hybrid", "hybrid_rerank")
POOL_DEPTH = 15


def first_occurrence(chunks: Sequence[Document]) -> list[Document]:
    seen: set[int] = set()
    posts = []
    for chunk in chunks:
        if chunk.metadata["document_id"] not in seen:
            seen.add(chunk.metadata["document_id"])
            posts.append(chunk)
    return posts


def system_rankings(pipeline: SearchPipeline, query: str) -> dict[str, list[Document]]:
    settings = pipeline.settings
    bm25 = Bm25Retriever(top_k=settings.retrieval_candidates).invoke(query)
    dense = PgVectorRetriever(embeddings=pipeline.embeddings, top_k=settings.retrieval_candidates).invoke(query)
    hybrid = pool(reciprocal_rank_fusion([bm25, dense], settings.rrf_k), settings.pooling_method, settings.pool_size)
    return {
        "bm25": first_occurrence(bm25),
        "dense": first_occurrence(dense),
        "hybrid": hybrid,
        "hybrid_rerank": pipeline.reranker.rerank([query], hybrid, settings.rerank_keep),
    }


def round_robin(query_id: str, rankings: Mapping[str, Sequence[Document]], depth: int) -> list[PooledDocument]:
    ranks = {
        name: {document_key(post.metadata): rank for rank, post in enumerate(posts, 1)}
        for name, posts in rankings.items()
    }
    pooled: dict[str, PooledDocument] = {}
    for row in zip_longest(*(rankings[name] for name in SYSTEMS)):
        for post in row:
            key = document_key(post.metadata) if post else None
            if key is None or key in pooled:
                continue
            if len(pooled) == depth:
                return list(pooled.values())
            pooled[key] = to_pooled(
                query_id, key, post, {name: ranks[name][key] for name in SYSTEMS if key in ranks[name]}
            )
    return list(pooled.values())


def to_pooled(query_id: str, key: str, post: Document, found_by: dict[str, int]) -> PooledDocument:
    metadata = post.metadata
    return PooledDocument(
        query_id=query_id,
        doc_key=key,
        title=metadata["title"],
        source=metadata["source"],
        url=metadata.get("timestamp_url") or metadata["url"],
        published_at=metadata["published_at"],
        passage=" ".join(post.page_content.split()),
        found_by=found_by,
    )


def corpus_snapshot(pipeline: SearchPipeline) -> dict[str, Any]:
    chunk_counts = (
        select(ChunkEmbeddingRecord.document_id, func.count().label("chunks"))
        .group_by(ChunkEmbeddingRecord.document_id)
        .subquery()
    )
    statement = (
        select(DocumentRecord.source, DocumentRecord.source_id, DocumentRecord.content_hash, chunk_counts.c.chunks)
        .join(chunk_counts, chunk_counts.c.document_id == DocumentRecord.id, isouter=True)
        .order_by(DocumentRecord.source, DocumentRecord.source_id)
    )
    with get_session_factory()() as session:
        rows = session.execute(statement).all()
    settings = pipeline.settings
    return {
        "version": "v1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "embedding_model": settings.embedding_model,
        "chunk_size_tokens": settings.chunk_size_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "documents": len(rows),
        "chunks": sum(row.chunks or 0 for row in rows),
        "documents_without_chunks": sum(1 for row in rows if not row.chunks),
        "documents_per_source": dict(Counter(row.source for row in rows)),
        "content_hashes": {f"{row.source}:{row.source_id}": row.content_hash for row in rows},
    }


def main() -> None:
    configure_logging()
    if POOL_FILE.exists():
        raise SystemExit(f"{POOL_FILE.name} already exists. Delete it to rebuild the pool; that invalidates judgments.")
    queries = read_jsonl(QUERIES_FILE, EvalQuery)
    pipeline = SearchPipeline.from_settings()
    snapshot = corpus_snapshot(pipeline)
    if snapshot["documents_without_chunks"]:
        raise SystemExit(
            f"{snapshot['documents_without_chunks']} documents have no chunks. Run the chunker and embeddings."
        )
    pooled = [
        doc for query in queries for doc in round_robin(query.id, system_rankings(pipeline, query.query), POOL_DEPTH)
    ]
    write_json(SNAPSHOT_FILE, snapshot)
    write_jsonl(POOL_FILE, pooled)
    per_query = Counter(doc.query_id for doc in pooled)
    print(
        f"Snapshot: {snapshot['documents']} documents, {snapshot['chunks']} chunks {snapshot['documents_per_source']}"
    )
    fewest = min(per_query[query.id] for query in queries)
    print(f"Pooled {len(pooled)} documents for {len(queries)} queries (min {fewest} per query)")
    for name in SYSTEMS:
        print(f"  {name:<14} retrieved {sum(1 for doc in pooled if name in doc.found_by)} of the pooled documents")


if __name__ == "__main__":
    main()
