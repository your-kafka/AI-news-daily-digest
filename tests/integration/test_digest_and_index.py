from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from ai_news.api.app import create_app
from ai_news.db.models import ChunkEmbeddingRecord, DigestRecord
from ai_news.db.session import get_session_factory
from ai_news.pipelines import digest
from ai_news.pipelines.index import index_new_documents
from ai_news.rag import chunker as chunker_module
from ai_news.rag import embeddings as embeddings_module
from ai_news.rag.loader import PostgresDocumentLoader
from tests.integration.test_database import NOW, save, scraped

pytestmark = pytest.mark.integration


def digest_values(profile: str = "papers", day: date = date(2026, 11, 23), **overrides):
    values = {
        "profile": profile,
        "digest_date": day,
        "window_start": NOW - timedelta(hours=24),
        "window_end": NOW,
        "model": "fake-model",
        "queries": ["q1", "q2"],
        "items": [
            {
                "headline": "A paper",
                "why_it_matters": "It matters",
                "documents": [{"id": 1, "source": "huggingface", "title": "T", "url": "https://hf.co/p"}],
            }
        ],
        "candidate_ids": [1, 2, 3],
        "documents_in_window": 40,
        "input_tokens": 7000,
        "output_tokens": 400,
        "naive_input_tokens": 175000,
        "duration_ms": 9000,
    }
    return {**values, **overrides}


def save_digest(**values):
    with get_session_factory().begin() as session:
        return digest.save_digest(session, digest_values(**values))


class TestDigestStorage:
    def test_rebuilding_a_day_replaces_that_days_digest(self, db):
        first = save_digest()
        second = save_digest(input_tokens=6500, queries=["new"])
        with get_session_factory()() as session:
            rows = session.scalars(select(DigestRecord)).all()
        assert first == second
        assert [(row.input_tokens, row.queries) for row in rows] == [(6500, ["new"])]

    def test_one_digest_per_profile_per_day(self, db):
        save_digest()
        save_digest(day=date(2026, 11, 24))
        save_digest(profile="frontier-models")
        with get_session_factory()() as session:
            assert session.scalar(select(func.count()).select_from(DigestRecord)) == 3


class TestDigestApi:
    @pytest.fixture
    def client(self, db):
        with TestClient(create_app(lambda: SimpleNamespace(settings=None))) as client:
            yield client

    def test_latest_and_by_date(self, client):
        save_digest(day=date(2026, 11, 22), input_tokens=8000)
        save_digest(day=date(2026, 11, 23))
        latest = client.get("/digest/latest", params={"profile": "papers"}).json()
        assert latest["digest_date"] == "2026-11-23"
        assert latest["title"] == "Papers worth reading"
        assert latest["candidates"] == 3
        assert latest["tokens"] == {
            "input_tokens": 7000,
            "output_tokens": 400,
            "naive_input_tokens": 175000,
            "saved": 0.96,
        }
        assert latest["items"][0]["documents"][0]["url"] == "https://hf.co/p"
        older = client.get("/digest/2026-11-22", params={"profile": "papers"}).json()
        assert older["tokens"]["input_tokens"] == 8000

    def test_missing_digest_is_404(self, client):
        assert client.get("/digest/latest", params={"profile": "papers"}).status_code == 404
        assert client.get("/digest/2026-01-01", params={"profile": "papers"}).status_code == 404

    def test_readyz_checks_database_and_redis(self, client):
        assert client.get("/readyz").json() == {"status": "ready"}


def test_documents_between(db):
    save(scraped("in", published_at=NOW - timedelta(hours=2)), scraped("out", published_at=NOW - timedelta(days=3)))
    found = digest.documents_between(NOW - timedelta(hours=24), NOW)
    assert [document.source_id for document in found] == ["in"]


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=text.split())


class FakeEmbeddings:
    batch_size = 16

    def __init__(self, model_name, *args, **kwargs):
        self.model_name = model_name

    def embed_documents(self, texts):
        return [[0.1] * 384 for _ in texts]


class TestIndexing:
    @pytest.fixture(autouse=True)
    def fake_models(self, monkeypatch):
        monkeypatch.setattr(chunker_module.Tokenizer, "from_pretrained", lambda *args, **kwargs: WordTokenizer())
        monkeypatch.setattr(embeddings_module, "FastEmbedEmbeddings", FakeEmbeddings)

    def chunk_count(self) -> int:
        with get_session_factory()() as session:
            return session.scalar(select(func.count()).select_from(ChunkEmbeddingRecord))

    def test_indexes_only_new_or_changed_documents(self, db):
        save(scraped("a", "first document"), scraped("b", "second document"))
        assert len(PostgresDocumentLoader(needs_indexing=True).load()) == 2

        report = index_new_documents()
        assert (report.documents, report.chunks, report.embedding.stored.saved) == (2, 2, 2)
        assert self.chunk_count() == 2
        assert PostgresDocumentLoader(needs_indexing=True).load() == []
        assert index_new_documents().documents == 0

        save(scraped("a", "first document, rewritten"))
        [changed] = PostgresDocumentLoader(needs_indexing=True).load()
        assert changed.metadata["source_id"] == "a"
        assert index_new_documents().embedding.stored.saved == 1
        assert self.chunk_count() == 2


def test_run_digest_saves_with_the_schedule_date(db, monkeypatch):
    class FakeBuilder:
        def build(self, profile, now, window_hours, digest_date):
            return digest_values(profile=profile.name, day=digest_date)

    values = digest.run_digest("papers", FakeBuilder(), now=datetime(2026, 11, 22, 20, 0, tzinfo=UTC))
    assert values["digest_date"] == date(2026, 11, 23)  # 01:30 the next morning in Asia/Kolkata
    assert values["id"] > 0
