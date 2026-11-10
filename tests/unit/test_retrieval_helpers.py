from datetime import UTC, datetime

import pytest
from langchain_core.language_models import FakeListChatModel
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from ai_news.config import RagSettings
from ai_news.db.models import ChunkEmbeddingRecord
from ai_news.rag.reranker import CrossEncoderReranker
from ai_news.rag.retriever import build_expander, filtered, parse_queries
from tests.conftest import make_post


class TestParseQueries:
    def test_strips_list_markers_blank_lines_and_duplicates(self):
        text = "1. agents with memory\n- agents with memory\n\n* GUI agents\n2) open-weight models\n• robotics"
        assert parse_queries(text) == ["agents with memory", "GUI agents", "open-weight models", "robotics"]

    def test_removes_reasoning_blocks(self):
        assert parse_queries("<think>\nlet me think\n</think>\nquery one\nquery two") == ["query one", "query two"]


def test_build_expander_caps_query_count():
    llm = FakeListChatModel(responses=["one\ntwo\nthree\nfour"])
    assert build_expander(llm, query_count=2).invoke("agents") == ["one", "two"]


def compiled(statement) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


def test_filtered_adds_only_the_filters_given():
    base = select(ChunkEmbeddingRecord.id)
    assert "WHERE" not in compiled(filtered(base, None, None))
    assert "WHERE" not in compiled(filtered(base, None, []))
    both = compiled(filtered(base, datetime(2026, 11, 1, tzinfo=UTC), ["openai"]))
    assert "published_at >=" in both
    assert "source IN" in both


class FakeCrossEncoder:
    """Scores a passage by how many query words it contains."""

    def rerank(self, query, passages, batch_size):
        words = set(query.lower().split())
        return [float(sum(word in passage.lower() for word in words)) for passage in passages]


@pytest.fixture
def reranker():
    instance = CrossEncoderReranker.__new__(CrossEncoderReranker)
    instance.model_name = "fake"
    instance.batch_size = 32
    instance.model = FakeCrossEncoder()
    return instance


class TestReranker:
    POSTS = [make_post("robots", title="Robot hands"), make_post("chips", title="Custom AI chips")]

    def test_orders_by_score_and_adds_relevance(self, reranker):
        result = reranker.rerank(["custom chips"], self.POSTS)
        assert [post.metadata["source_id"] for post in result] == ["chips", "robots"]
        assert result[0].metadata["rerank_score"] == 2.0
        assert 0.5 < result[0].metadata["relevance"] < 1

    def test_takes_the_best_score_over_several_queries(self, reranker):
        result = reranker.rerank(["chips", "robot hands"], self.POSTS)
        assert [(post.metadata["source_id"], post.metadata["rerank_score"]) for post in result] == [
            ("robots", 2.0),
            ("chips", 1.0),
        ]

    def test_limit_and_empty_inputs(self, reranker):
        assert len(reranker.rerank(["chips"], self.POSTS, limit=1)) == 1
        assert reranker.rerank([], self.POSTS) == []
        assert reranker.rerank(["chips"], []) == []


class TestRagSettings:
    def test_defaults(self):
        settings = RagSettings(_env_file=None)
        assert (settings.rrf_k, settings.pooling_method, settings.mmr_lambda) == (60, "max", 0.7)

    def test_overlap_must_be_smaller_than_chunk(self):
        with pytest.raises(ValueError, match="CHUNK_OVERLAP_TOKENS"):
            RagSettings(_env_file=None, chunk_size_tokens=100, chunk_overlap_tokens=100)

    def test_rejects_unknown_pooling_method_and_bad_lambda(self):
        with pytest.raises(ValueError):
            RagSettings(_env_file=None, pooling_method="mean")
        with pytest.raises(ValueError):
            RagSettings(_env_file=None, mmr_lambda=1.5)

    def test_reads_environment_and_ignores_empty_values(self, monkeypatch):
        monkeypatch.setenv("RRF_K", "10")
        monkeypatch.setenv("POOL_SIZE", "")
        settings = RagSettings(_env_file=None)
        assert settings.rrf_k == 10
        assert settings.pool_size == 30
