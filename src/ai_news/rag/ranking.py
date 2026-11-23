from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from langchain_core.documents import Document

from ai_news.config import PoolingMethod
from ai_news.rag.retriever import EMBEDDING_KEY

POOLING_FUNCTIONS: dict[PoolingMethod, Callable[[list[float]], float]] = {
    "max": lambda scores: scores[0],
    "top3": lambda scores: sum(scores[:3]),
    "combmnz": lambda scores: sum(scores) * len(scores),
}


def with_metadata(document: Document, **metadata: Any) -> Document:
    return Document(id=document.id, page_content=document.page_content, metadata={**document.metadata, **metadata})


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Document]], k: int = 60) -> list[Document]:
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, Document] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.id] += 1 / (k + rank)
            chunks[chunk.id] = with_metadata(chunks[chunk.id], **chunk.metadata) if chunk.id in chunks else chunk
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
    return [with_metadata(chunks[chunk_id], rrf_score=round(scores[chunk_id], 6)) for chunk_id in ranked]


def first_per_document(chunks: Sequence[Document], limit: int | None = None) -> list[Document]:
    """Keeps each document's highest-ranked chunk, in ranking order."""
    seen: set[int] = set()
    posts = []
    for chunk in chunks:
        if chunk.metadata["document_id"] not in seen:
            seen.add(chunk.metadata["document_id"])
            posts.append(chunk)
            if len(posts) == limit:
                break
    return posts


def pool(chunks: Sequence[Document], method: PoolingMethod = "max", limit: int = 30) -> list[Document]:
    by_post: dict[int, list[Document]] = defaultdict(list)
    for chunk in chunks:
        by_post[chunk.metadata["document_id"]].append(chunk)
    pooled = []
    for post_chunks in by_post.values():
        post_chunks.sort(key=lambda chunk: chunk.metadata["rrf_score"], reverse=True)
        score = POOLING_FUNCTIONS[method]([chunk.metadata["rrf_score"] for chunk in post_chunks])
        pooled.append(with_metadata(post_chunks[0], score=round(score, 6), matched_chunks=len(post_chunks)))
    pooled.sort(key=lambda post: post.metadata["score"], reverse=True)
    return pooled[:limit]


def maximal_marginal_relevance(
    posts: Sequence[Document],
    lambda_: float = 0.7,
    limit: int = 10,
    relevance_key: str = "relevance",
) -> list[Document]:
    if not posts:
        return []
    relevance = np.array([post.metadata[relevance_key] for post in posts], dtype=np.float32)
    vectors = np.array([post.metadata[EMBEDDING_KEY] for post in posts], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity = vectors @ vectors.T
    selected: list[int] = []
    remaining = list(range(len(posts)))
    while remaining and len(selected) < limit:
        redundancy = similarity[remaining][:, selected].max(axis=1) if selected else np.zeros(len(remaining))
        gains = lambda_ * relevance[remaining] - (1 - lambda_) * redundancy
        selected.append(remaining.pop(int(np.argmax(gains))))
    return [posts[index] for index in selected]
