from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from alembic import command
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import func, select, text

from ai_news.db.models import EMBEDDING_DIMENSIONS, ChunkEmbeddingRecord, DocumentRecord, IngestRun, TranscriptRecord
from ai_news.db.repository import SaveResult, finish_run, save_documents, start_run
from ai_news.db.session import get_session_factory
from ai_news.evaluation.pooling import corpus_snapshot, system_rankings
from ai_news.pipelines.ingest import ingest_source
from ai_news.rag.connector import EmbeddedChunk, EmbeddingStore
from ai_news.rag.loader import SEGMENTS_KEY, PostgresDocumentLoader
from ai_news.rag.pipeline import SearchPipeline
from ai_news.rag.ranking import with_metadata
from ai_news.rag.retriever import Bm25Retriever, PgVectorRetriever
from ai_news.scrapers.fetcher import FetchError
from ai_news.scrapers.models import Document as ScrapedDocument
from ai_news.scrapers.models import Transcript, TranscriptSegment
from tests.integration.conftest import alembic_config

pytestmark = pytest.mark.integration

NOW = datetime(2026, 11, 10, tzinfo=UTC)
TOPICS = ("robot", "chip", "agent")


def scraped(source_id: str, content: str = "text", source: str = "openai", **fields) -> ScrapedDocument:
    defaults = {
        "url": f"https://example.com/{source_id}",
        "title": f"Title {source_id}",
        "published_at": NOW,
        "content_kind": "summary",
    }
    return ScrapedDocument(source=source, source_id=source_id, content=content, **{**defaults, **fields})


def save(*documents: ScrapedDocument):
    with get_session_factory().begin() as session:
        return save_documents(session, documents)


def topic_vector(topic: str) -> list[float]:
    vector = np.zeros(EMBEDDING_DIMENSIONS)
    vector[TOPICS.index(topic)] = 1.0
    return vector.tolist()


class TopicEmbeddings(Embeddings):
    """Embeds text onto one axis per topic word, so dense search is predictable."""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return next((topic_vector(topic) for topic in TOPICS if topic in text.lower()), topic_vector("agent"))


def store_chunks(rows: list[tuple[str, str, str, datetime]]) -> dict[str, int]:
    """Saves one document and one chunk per (source_id, source, content, published_at); returns their ids."""
    save(
        *(scraped(source_id, content, source, published_at=published) for source_id, source, content, published in rows)
    )
    with get_session_factory()() as session:
        ids = dict(session.execute(select(DocumentRecord.source_id, DocumentRecord.id)).all())
    embedded = [
        EmbeddedChunk(
            Document(
                id=f"{ids[source_id]}-0",
                page_content=content,
                metadata={
                    "document_id": ids[source_id],
                    "chunk_index": 0,
                    "source": source,
                    "published_at": published.isoformat(),
                    "token_count": len(content.split()),
                    "title": f"Title {source_id}",
                    "source_id": source_id,
                    "url": f"https://example.com/{source_id}",
                },
            ),
            TopicEmbeddings().embed_query(content),
            f"hash-{source_id}",
            "topic-model",
        )
        for source_id, source, content, published in rows
    ]
    EmbeddingStore().save(embedded, {ids[source_id]: {f"{ids[source_id]}-0"} for source_id, *_ in rows})
    return ids


CORPUS = [
    ("robots", "deepmind", "Robot hands learn dexterous manipulation of everyday objects", NOW),
    ("chips", "openai", "Custom AI chip designs cut inference cost for large models", NOW - timedelta(days=3)),
    ("agents", "anthropic", "Agents that remember: long-term memory for coding agent workflows", NOW),
    ("old", "openai", "An older robot story about warehouse automation", NOW - timedelta(days=60)),
]


class TestMigrations:
    def test_models_match_the_migrations(self, db):
        command.check(alembic_config())

    def test_extensions_and_indexes_exist(self, db):
        with db.connect() as connection:
            extensions = set(connection.execute(text("SELECT extname FROM pg_extension")).scalars())
            rows = connection.execute(
                text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'chunk_embeddings'")
            ).all()
        indexes = dict(rows)
        assert {"vector", "pg_search"} <= extensions
        assert "USING hnsw" in indexes["ix_chunk_embeddings_embedding"]
        assert "USING bm25" in indexes["ix_chunk_embeddings_bm25"]


class TestRepository:
    def test_insert_then_unchanged_then_update(self, db):
        assert save(scraped("a"), scraped("b")) == SaveResult(2, 0)
        assert save(scraped("a"), scraped("b")) == SaveResult(0, 0)
        assert save(scraped("a", "new text"), scraped("b")) == SaveResult(0, 1)
        with get_session_factory()() as session:
            record = session.scalar(select(DocumentRecord).where(DocumentRecord.source_id == "a"))
        assert (record.content, record.word_count) == ("new text", 2)

    def test_duplicates_in_one_batch_are_saved_once(self, db):
        assert save(scraped("a", "first"), scraped("a", "second")) == SaveResult(1, 0)

    def test_a_summary_never_replaces_a_stored_transcript(self, db):
        transcript = Transcript(
            language_code="en", is_generated=True, segments=[TranscriptSegment(start=0, duration=2, text="hello")]
        )
        video = scraped("v", "hello", source="youtube", content_kind="transcript", transcript=transcript)
        assert save(video) == SaveResult(1, 0)
        assert save(scraped("v", "just the description", source="youtube")) == SaveResult(0, 0)
        with get_session_factory()() as session:
            record = session.scalar(select(DocumentRecord))
            segments = session.scalar(select(TranscriptRecord.segments))
        assert record.content_kind == "transcript"
        assert segments == [{"start": 0.0, "duration": 2.0, "text": "hello"}]

    def test_ingest_runs_are_recorded(self, db):
        with get_session_factory().begin() as session:
            run_id = start_run(session, "openai", 24)
            finish_run(session, run_id, 3, save_documents(session, [scraped("a")]), error=None)
        with get_session_factory()() as session:
            run = session.get(IngestRun, run_id)
        assert (run.status, run.documents_found, run.documents_inserted) == ("succeeded", 3, 1)
        assert run.finished_at is not None


class TestIngest:
    def test_ingest_source_saves_what_it_found_before_failing(self, db):
        def scrape(hours):
            yield scraped("a")
            raise FetchError("HTTP 503 from feed")

        report = ingest_source("openai", scrape, 24)
        assert (report.found, report.saved.inserted, report.error) == (1, 1, "HTTP 503 from feed")
        with get_session_factory()() as session:
            run = session.scalar(select(IngestRun))
        assert (run.status, run.error) == ("failed", "HTTP 503 from feed")


class TestLoader:
    def test_loads_documents_with_transcripts_and_filters(self, db):
        transcript = Transcript(
            language_code="en", is_generated=False, segments=[TranscriptSegment(start=1, duration=2, text="hi")]
        )
        save(
            scraped("v", "hi", source="youtube", content_kind="transcript", transcript=transcript),
            scraped("old", published_at=NOW - timedelta(days=30)),
        )
        documents = PostgresDocumentLoader().load()
        assert [document.metadata["source_id"] for document in documents] == ["v", "old"]
        assert documents[0].metadata[SEGMENTS_KEY] == [{"start": 1.0, "duration": 2.0, "text": "hi"}]
        assert SEGMENTS_KEY not in documents[1].metadata
        assert len(PostgresDocumentLoader(since=NOW - timedelta(days=1)).load()) == 1
        assert len(PostgresDocumentLoader(sources=["openai"]).load()) == 1


class TestEmbeddingStore:
    def test_save_upserts_and_removes_stale_chunks(self, db):
        ids = store_chunks(CORPUS[:2])
        store = EmbeddingStore()
        assert store.count() == 2
        robots = ids["robots"]
        assert store.existing_documents([robots, 999]) == {robots}
        assert store.stored_hashes([f"{robots}-0", "missing"]) == {f"{robots}-0": "hash-robots"}

        document = Document(
            id=f"{robots}-1",
            page_content="second robot chunk",
            metadata={
                "document_id": robots,
                "chunk_index": 1,
                "source": "deepmind",
                "published_at": NOW.isoformat(),
                "token_count": 3,
                "section": "Results",
            },
        )
        result = store.save([EmbeddedChunk(document, topic_vector("robot"), "h", "m")], {robots: {f"{robots}-1"}})
        assert (result.saved, result.removed) == (1, 1)
        with get_session_factory()() as session:
            chunk_ids = set(session.scalars(select(ChunkEmbeddingRecord.id)))
            stored = session.get(ChunkEmbeddingRecord, f"{robots}-1")
        assert chunk_ids == {f"{robots}-1", f"{ids['chips']}-0"}
        assert (stored.section, stored.metadata_) == ("Results", {})

    def test_deleting_a_document_cascades_to_its_chunks(self, db):
        ids = store_chunks(CORPUS[:1])
        with get_session_factory().begin() as session:
            session.execute(DocumentRecord.__table__.delete().where(DocumentRecord.id == ids["robots"]))
        assert EmbeddingStore().count() == 0


class TestRetrievers:
    def test_bm25_matches_stemmed_keywords_and_ranks_by_score(self, db):
        store_chunks(CORPUS)
        results = Bm25Retriever(top_k=10).invoke("robots")
        assert [doc.metadata["source_id"] for doc in results][:2] in (["robots", "old"], ["old", "robots"])
        assert all(doc.metadata["bm25_score"] > 0 for doc in results)
        assert "embedding" in results[0].metadata

    def test_bm25_accepts_arbitrary_user_text(self, db):
        store_chunks(CORPUS)
        assert Bm25Retriever().invoke('C++ & "agents" (memory) OR -x:y') != []
        assert Bm25Retriever().invoke("zzzz qqqq") == []

    def test_bm25_filters(self, db):
        store_chunks(CORPUS)
        recent = Bm25Retriever(since=NOW - timedelta(days=7)).invoke("robot")
        assert [doc.metadata["source_id"] for doc in recent] == ["robots"]
        assert Bm25Retriever(sources=["anthropic"]).invoke("robot") == []

    def test_dense_returns_nearest_chunks_first(self, db):
        store_chunks(CORPUS)
        results = PgVectorRetriever(embeddings=TopicEmbeddings(), top_k=2).invoke("custom chip hardware")
        assert results[0].metadata["source_id"] == "chips"
        assert results[0].metadata["similarity"] == pytest.approx(1.0)
        assert len(results) == 2

    def test_dense_filters(self, db):
        store_chunks(CORPUS)
        results = PgVectorRetriever(
            embeddings=TopicEmbeddings(), since=NOW - timedelta(days=7), sources=["deepmind", "openai"]
        ).invoke("robot")
        assert [doc.metadata["source_id"] for doc in results][0] == "robots"
        assert "old" not in {doc.metadata["source_id"] for doc in results}
        assert "agents" not in {doc.metadata["source_id"] for doc in results}


class KeywordReranker:
    """Scores posts by query-word overlap, standing in for the cross-encoder."""

    def rerank(self, queries, posts, limit=20):
        def overlap(post):
            return max(len(set(query.lower().split()) & set(post.page_content.lower().split())) for query in queries)

        ranked = sorted(posts, key=overlap, reverse=True)[:limit]
        return [
            with_metadata(post, rerank_score=float(overlap(post)), relevance=0.5 + overlap(post) / 10)
            for post in ranked
        ]


@pytest.fixture
def search_pipeline(monkeypatch):
    from ai_news.config import RagSettings

    settings = RagSettings(_env_file=None, retrieval_candidates=20, pool_size=10, rerank_keep=10, retrieval_top_k=3)
    return SearchPipeline(settings, TopicEmbeddings(), KeywordReranker())


class TestSearchPipeline:
    def test_runs_stages_one_to_five(self, db, search_pipeline):
        store_chunks(CORPUS)
        posts = search_pipeline.search(["robot hands manipulation"])
        assert posts[0].metadata["source_id"] == "robots"
        assert len(posts) == 3
        assert all("embedding" not in post.metadata for post in posts)
        assert {"rrf_score", "score", "rerank_score", "relevance"} <= set(posts[0].metadata)

    def test_filters_and_limit(self, db, search_pipeline):
        store_chunks(CORPUS)
        posts = search_pipeline.search(["robot"], limit=1, since=NOW - timedelta(days=7), sources=["deepmind"])
        assert [post.metadata["source_id"] for post in posts] == ["robots"]

    def test_fast_search_is_the_evaluated_dense_ranking(self, db, search_pipeline):
        store_chunks(CORPUS)
        fast = search_pipeline.search_fast(["robot"], limit=3)
        dense = system_rankings(search_pipeline, "robot")["dense"][:3]
        assert [post.id for post in fast] == [post.id for post in dense]
        assert all("embedding" not in post.metadata for post in fast)
        recent = search_pipeline.search_fast(["robot"], since=NOW - timedelta(days=7), sources=["deepmind"])
        assert [post.metadata["source_id"] for post in recent] == ["robots"]

    def test_empty_database(self, db, search_pipeline):
        assert search_pipeline.search(["anything"]) == []

    def test_evaluation_rankings_and_snapshot(self, db, search_pipeline):
        store_chunks(CORPUS)
        rankings = system_rankings(search_pipeline, "robot")
        assert set(rankings) == {"bm25", "dense", "hybrid", "hybrid_rerank"}
        assert rankings["dense"][0].metadata["source_id"] in {"robots", "old"}
        snapshot = corpus_snapshot(search_pipeline)
        assert (snapshot["documents"], snapshot["chunks"], snapshot["documents_without_chunks"]) == (4, 4, 0)
        assert snapshot["content_hashes"].keys() == {
            "anthropic:agents",
            "deepmind:robots",
            "openai:chips",
            "openai:old",
        }


def test_downgrade_and_upgrade_round_trip(db):
    config = alembic_config()
    command.downgrade(config, "base")
    with db.connect() as connection:
        tables = set(connection.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")).scalars())
    assert not tables & {"documents", "transcripts", "ingest_runs", "chunk_embeddings"}
    command.upgrade(config, "head")
    with get_session_factory()() as session:
        assert session.scalar(select(func.count()).select_from(DocumentRecord)) == 0


class TestApi:
    @pytest.fixture
    def client(self, db, search_pipeline):
        from fastapi.testclient import TestClient

        from ai_news.api.app import create_app

        with TestClient(create_app(lambda: search_pipeline)) as client:
            yield client

    def test_readyz(self, client):
        assert client.get("/readyz").json() == {"status": "ready"}

    def test_search_and_document_round_trip(self, client):
        store_chunks(CORPUS)
        hits = client.get("/search", params={"q": "robot hands", "k": 2}).json()["results"]
        assert hits[0]["doc_key"] == "deepmind:robots"
        document = client.get(f"/documents/{hits[0]['document_id']}").json()
        assert (document["doc_key"], document["title"], document["content_kind"]) == (
            "deepmind:robots",
            "Title robots",
            "summary",
        )

    def test_unknown_document_is_404(self, client):
        assert client.get("/documents/123456").status_code == 404
