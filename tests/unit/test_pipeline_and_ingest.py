import logging
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from ai_news.db.repository import SaveResult
from ai_news.pipelines import ingest
from ai_news.rag import pipeline
from ai_news.rag.curation import CurationResult, Digest, DigestItem
from ai_news.scrapers.fetcher import FetchError
from tests.conftest import make_post


def ranked_post(source_id: str, document_id: int) -> Document:
    return make_post(source_id, document_id, rerank_score=2.5, score=0.0312, embedding=[0.1])


def test_timed_logs_stage_duration(caplog):
    with caplog.at_level(logging.INFO, logger=pipeline.__name__):
        assert pipeline.timed("2 rrf", lambda: [1, 2]) == [1, 2]
    assert "stage 2 rrf" in caplog.text
    assert "2 results" in caplog.text


def test_without_embedding_drops_only_the_vector():
    post = pipeline.without_embedding(ranked_post("a", 1))
    assert "embedding" not in post.metadata
    assert post.metadata["rerank_score"] == 2.5


def test_print_posts(capsys):
    pipeline.print_posts([ranked_post("a", 1)])
    output = capsys.readouterr().out
    assert "1. [openai] Title a" in output
    assert "rerank 2.50 | pooled 0.0312" in output
    pipeline.print_posts([])
    assert "No posts found" in capsys.readouterr().out


def test_print_digest(capsys):
    digest = Digest(items=[DigestItem(headline="Big news", why_it_matters="It matters", document_ids=[1])])
    pipeline.print_digest(CurationResult(digest, "m", 7000, 400), [ranked_post("a", 1)])
    output = capsys.readouterr().out
    assert "1. Big news" in output
    assert "- [openai] Title a: https://example.com/a" in output
    assert "m: 7000 input tokens, 400 output tokens" in output
    pipeline.print_digest(CurationResult(Digest(items=[]), "m", 0, 0), [])
    assert "found nothing" in capsys.readouterr().out


class FakePipeline:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(retrieval_query_count=3, curation_candidates=15)
        self.calls: list[tuple[list[str], int | None]] = []

    def search(self, queries, limit=None):
        self.calls.append((list(queries), limit))
        return [ranked_post("a", 1)]


def test_run_without_groq_searches_the_raw_interests(capsys):
    fake = FakePipeline()
    pipeline.run("open-source LLMs", fake, groq=None)
    assert fake.calls == [(["open-source LLMs"], None)]
    assert "Title a" in capsys.readouterr().out


def test_optional_groq_settings_is_none_when_unset(monkeypatch):
    monkeypatch.setattr(pipeline, "groq_settings", lambda: pipeline.GroqSettings(_env_file=None))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert pipeline.optional_groq_settings() is None


class TestIngest:
    def documents(self):
        return [SimpleNamespace(title="one"), SimpleNamespace(title="two")]

    def test_collect_returns_everything_on_success(self):
        documents, error = ingest.collect("openai", lambda hours: self.documents(), 24)
        assert len(documents) == 2
        assert error is None

    @pytest.mark.parametrize(
        ("exception", "message"),
        [
            (FetchError("HTTP 503"), "HTTP 503"),
            (RuntimeError("YOUTUBE_API_KEY is not set"), "YOUTUBE_API_KEY is not set"),
            (KeyError("title"), "KeyError: 'title'"),
        ],
    )
    def test_collect_keeps_documents_found_before_a_failure(self, exception, message):
        def scrape(hours):
            yield from self.documents()
            raise exception

        documents, error = ingest.collect("openai", scrape, 24)
        assert len(documents) == 2
        assert error == message

    def test_source_report_counts_unchanged(self):
        report = ingest.SourceReport("openai", 10, SaveResult(inserted=3, updated=2), None)
        assert report.unchanged == 5

    def test_print_report(self, capsys):
        reports = [
            ingest.SourceReport("openai", 3, SaveResult(1, 1), None),
            ingest.SourceReport("youtube", 0, SaveResult(), "blocked"),
        ]
        ingest.print_report(reports, 24)
        lines = capsys.readouterr().out.splitlines()
        assert lines[1].strip() == "Ingested the last 24 hours into PostgreSQL:"
        assert lines[3].split() == ["openai", "3", "1", "1", "1", "ok"]
        assert lines[4].split()[-1] == "failed"


def test_save_result_adds():
    assert SaveResult(1, 2) + SaveResult(3, 4) == SaveResult(4, 6)
