import logging
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Self

from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel
from langchain_groq import ChatGroq
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError, ProgrammingError

from ai_news.config import GroqSettings, RagSettings, groq_settings, rag_settings
from ai_news.log_config import configure_logging
from ai_news.rag.curation import CurationResult, Curator
from ai_news.rag.embeddings import FastEmbedEmbeddings, export_hf_token
from ai_news.rag.ranking import maximal_marginal_relevance, pool, reciprocal_rank_fusion
from ai_news.rag.reranker import CrossEncoderReranker
from ai_news.rag.retriever import EMBEDDING_KEY, Bm25Retriever, PgVectorRetriever, build_expander

log = logging.getLogger(__name__)


class SearchPipeline:
    def __init__(self, settings: RagSettings, embeddings: FastEmbedEmbeddings, reranker: CrossEncoderReranker) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.reranker = reranker

    @classmethod
    def from_settings(cls, settings: RagSettings | None = None) -> Self:
        settings = settings or rag_settings()
        export_hf_token()
        embeddings = FastEmbedEmbeddings(settings.embedding_model, cache_dir=settings.model_cache_dir)
        reranker = CrossEncoderReranker(settings.rerank_model, cache_dir=settings.model_cache_dir)
        return cls(settings, embeddings, reranker)

    def search(
        self,
        queries: Sequence[str],
        limit: int | None = None,
        since: datetime | None = None,
        sources: list[str] | None = None,
    ) -> list[Document]:
        settings = self.settings
        candidates = settings.retrieval_candidates
        retrievers = RunnableParallel(
            bm25=Bm25Retriever(top_k=candidates, since=since, sources=sources),
            dense=PgVectorRetriever(embeddings=self.embeddings, top_k=candidates, since=since, sources=sources),
        )
        results = timed("1 hybrid candidates", lambda: retrievers.batch(list(queries)))
        rankings = [ranking for result in results for ranking in result.values()]
        for name in ("bm25", "dense"):
            log.info("  %s: %d chunks", name, sum(len(result[name]) for result in results))
        chunks = timed("2 rrf", lambda: reciprocal_rank_fusion(rankings, settings.rrf_k))
        posts = timed("3 pooling", lambda: pool(chunks, settings.pooling_method, settings.pool_size))
        posts = timed("4 cross-encoder", lambda: self.reranker.rerank(queries, posts, settings.rerank_keep))
        final_count = limit or settings.retrieval_top_k
        posts = timed("5 mmr", lambda: maximal_marginal_relevance(posts, settings.mmr_lambda, final_count))
        return [without_embedding(post) for post in posts]


def timed[T](stage: str, run: Callable[[], T]) -> T:
    started = time.perf_counter()
    result = run()
    count = f", {len(result)} results" if isinstance(result, list) else ""
    log.info("stage %-19s %6.0f ms%s", stage, (time.perf_counter() - started) * 1000, count)
    return result


def without_embedding(post: Document) -> Document:
    metadata = {key: value for key, value in post.metadata.items() if key != EMBEDDING_KEY}
    return Document(id=post.id, page_content=post.page_content, metadata=metadata)


def optional_groq_settings() -> GroqSettings | None:
    try:
        return groq_settings()
    except ValidationError:
        return None


def chat_model(groq: GroqSettings, temperature: float) -> ChatGroq:
    return ChatGroq(model=groq.groq_model, api_key=groq.groq_api_key, temperature=temperature)


def print_posts(posts: Sequence[Document]) -> None:
    if not posts:
        print("\nNo posts found. Run the ingest, chunker and embeddings scripts first.")
    for rank, post in enumerate(posts, start=1):
        metadata = post.metadata
        snippet = " ".join(post.page_content.split())[:200]
        print(f"\n{rank}. [{metadata['source']}] {metadata['title']}")
        print(f"   {metadata.get('timestamp_url') or metadata['url']}")
        scores = f"rerank {metadata['rerank_score']:.2f} | pooled {metadata['score']:.4f}"
        print(f"   {metadata['published_at'][:10]} | {scores}")
        print(f"   {snippet}...")


def print_digest(result: CurationResult, posts: Sequence[Document]) -> None:
    by_document = {post.metadata["document_id"]: post for post in posts}
    if not result.digest.items:
        print("\nThe LLM found nothing worth including.")
    for rank, item in enumerate(result.digest.items, start=1):
        print(f"\n{rank}. {item.headline}")
        print(f"   {item.why_it_matters}")
        for document_id in item.document_ids:
            metadata = by_document[document_id].metadata
            link = metadata.get("timestamp_url") or metadata["url"]
            print(f"   - [{metadata['source']}] {metadata['title']}: {link}")
    print(f"\n{result.model}: {result.input_tokens} input tokens, {result.output_tokens} output tokens")


def run(interests: str, pipeline: SearchPipeline, groq: GroqSettings | None) -> None:
    settings = pipeline.settings
    if groq is None:
        log.warning("GROQ_API_KEY or GROQ_MODEL is not set: searching with your text as-is, without LLM curation")
        print_posts(pipeline.search([interests]))
        return
    queries = build_expander(chat_model(groq, temperature=0.2), settings.retrieval_query_count).invoke(interests)
    queries = queries or [interests]
    log.info("Searching with %d queries:\n  %s", len(queries), "\n  ".join(queries))
    posts = pipeline.search(queries, limit=settings.curation_candidates)
    curator = Curator(chat_model(groq, temperature=0), groq.groq_model)
    result = timed("6 llm curation", lambda: curator.curate(interests, posts))
    print_digest(result, posts)


def main() -> None:
    configure_logging()
    groq = optional_groq_settings()
    pipeline = SearchPipeline.from_settings()
    interests = input("Your interests (for example: AI agents, open-source LLMs, AI safety): ").strip()
    if not interests:
        raise SystemExit("Enter at least one interest.")
    try:
        run(interests, pipeline, groq)
    except OperationalError as error:
        raise SystemExit(f"Cannot reach PostgreSQL. Start it with `docker compose up -d`.\n{error.orig}") from error
    except ProgrammingError as error:
        raise SystemExit(f"Database schema is out of date. Run `alembic upgrade head`.\n{error.orig}") from error


if __name__ == "__main__":
    main()
