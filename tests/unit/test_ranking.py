import numpy as np
import pytest
from langchain_core.documents import Document

from ai_news.rag.ranking import maximal_marginal_relevance, pool, reciprocal_rank_fusion, with_metadata
from ai_news.rag.reranker import sigmoid
from tests.conftest import make_chunk


def test_with_metadata_copies_and_overrides():
    original = make_chunk("c1", 1, score=1)
    updated = with_metadata(original, score=2, extra="x")
    assert updated.metadata == {"document_id": 1, "score": 2, "extra": "x"}
    assert original.metadata == {"document_id": 1, "score": 1}
    assert updated.id == "c1"


class TestReciprocalRankFusion:
    def test_chunk_found_by_several_rankings_wins(self):
        bm25 = [make_chunk("c1", 1, bm25_score=9), make_chunk("c2", 2, bm25_score=5)]
        dense = [make_chunk("c3", 3, similarity=0.9), make_chunk("c2", 2, similarity=0.8)]
        fused = reciprocal_rank_fusion([bm25, dense, [make_chunk("c2", 2)]], k=60)
        assert [chunk.id for chunk in fused] == ["c2", "c1", "c3"]
        assert fused[0].metadata["rrf_score"] == pytest.approx(2 / 62 + 1 / 61, abs=1e-6)

    def test_metadata_from_every_ranking_is_merged(self):
        fused = reciprocal_rank_fusion([[make_chunk("c", 1, bm25_score=5)], [make_chunk("c", 1, similarity=0.8)]])
        assert fused[0].metadata["bm25_score"] == 5
        assert fused[0].metadata["similarity"] == 0.8

    def test_smaller_k_rewards_top_ranks_more(self):
        ranking = [make_chunk("a", 1), make_chunk("b", 2)]
        sharp = reciprocal_rank_fusion([ranking], k=1)
        flat = reciprocal_rank_fusion([ranking], k=1000)
        sharp_ratio = sharp[0].metadata["rrf_score"] / sharp[1].metadata["rrf_score"]
        flat_ratio = flat[0].metadata["rrf_score"] / flat[1].metadata["rrf_score"]
        assert sharp_ratio > flat_ratio

    def test_empty(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []


class TestPool:
    # Document 1 has three mediocre chunks; document 2 has one strong chunk.
    CHUNKS = [
        make_chunk("a", 2, rrf_score=0.05),
        make_chunk("b", 1, rrf_score=0.03),
        make_chunk("c", 1, rrf_score=0.02),
        make_chunk("d", 1, rrf_score=0.02),
    ]

    @pytest.mark.parametrize(("method", "winner"), [("max", 2), ("top3", 1), ("combmnz", 1)])
    def test_methods_rank_posts_differently(self, method, winner):
        posts = pool(self.CHUNKS, method)
        assert posts[0].metadata["document_id"] == winner
        assert len(posts) == 2

    def test_keeps_best_chunk_and_counts_matches(self):
        post = next(post for post in pool(self.CHUNKS, "top3") if post.metadata["document_id"] == 1)
        assert post.id == "b"
        assert post.metadata["matched_chunks"] == 3
        assert post.metadata["score"] == pytest.approx(0.07)

    def test_combmnz_multiplies_by_match_count(self):
        post = next(post for post in pool(self.CHUNKS, "combmnz") if post.metadata["document_id"] == 1)
        assert post.metadata["score"] == pytest.approx(0.07 * 3)

    def test_limit(self):
        assert len(pool(self.CHUNKS, "max", limit=1)) == 1
        assert pool([], "max") == []


def mmr_post(post_id: str, logit: float, vector: list[float]) -> Document:
    return make_chunk(post_id, 0, relevance=sigmoid(logit), embedding=np.array(vector))


class TestMaximalMarginalRelevance:
    # A and A2 are near-duplicates of the same story; B is a different story, slightly less relevant.
    POSTS = [
        mmr_post("A", 5.0, [1, 0, 0]),
        mmr_post("A2", 4.9, [0.99, 0.1, 0]),
        mmr_post("B", 4.0, [0, 1, 0]),
    ]

    def test_lambda_one_is_pure_relevance(self):
        assert [post.id for post in maximal_marginal_relevance(self.POSTS, 1.0, 3)] == ["A", "A2", "B"]

    def test_pushes_near_duplicate_down(self):
        assert [post.id for post in maximal_marginal_relevance(self.POSTS, 0.7, 3)] == ["A", "B", "A2"]

    def test_limit_and_empty(self):
        assert [post.id for post in maximal_marginal_relevance(self.POSTS, 0.7, 2)] == ["A", "B"]
        assert maximal_marginal_relevance([], 0.7, 2) == []

    def test_unnormalised_vectors_are_normalised(self):
        posts = [mmr_post("A", 5.0, [10, 0]), mmr_post("A2", 4.9, [5, 0.5]), mmr_post("B", 4.0, [0, 3])]
        assert [post.id for post in maximal_marginal_relevance(posts, 0.7, 3)] == ["A", "B", "A2"]


def test_sigmoid():
    assert sigmoid(0) == 0.5
    assert sigmoid(10) > 0.9999
    assert sigmoid(-10) < 0.0001
