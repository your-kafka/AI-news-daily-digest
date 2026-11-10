import math
from collections.abc import Sequence
from pathlib import Path

from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_core.documents import Document

from ai_news.rag.embeddings import embedding_text
from ai_news.rag.ranking import with_metadata

BATCH_SIZE = 32


class CrossEncoderReranker:
    def __init__(self, model_name: str, cache_dir: Path | None = None, batch_size: int = BATCH_SIZE) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = TextCrossEncoder(model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None)

    def rerank(self, queries: Sequence[str], posts: Sequence[Document], limit: int = 20) -> list[Document]:
        if not posts or not queries:
            return []
        passages = [embedding_text(post) for post in posts]
        per_query = [list(self.model.rerank(query, passages, batch_size=self.batch_size)) for query in queries]
        scores = [max(post_scores) for post_scores in zip(*per_query, strict=True)]
        ranked = sorted(zip(posts, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [
            with_metadata(post, rerank_score=round(float(score), 4), relevance=round(sigmoid(score), 4))
            for post, score in ranked[:limit]
        ]


def sigmoid(score: float) -> float:
    return 1 / (1 + math.exp(-score))
