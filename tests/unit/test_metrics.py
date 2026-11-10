import pytest

from ai_news.evaluation.metrics import (
    dcg,
    gain,
    judged_at_k,
    ndcg_at_k,
    randomization_test,
    recall_at_k,
    reciprocal_rank,
)

QRELS = {"a": 3, "b": 0, "c": 2, "d": 1}


def test_gain_is_exponential():
    assert [gain(grade) for grade in range(4)] == [0, 1, 3, 7]


def test_dcg_discounts_by_log_rank():
    assert dcg([3, 0, 2]) == pytest.approx(7 + 0 + 3 / 2)


class TestNdcg:
    def test_worked_example(self):
        assert ndcg_at_k(["a", "b", "c"], {"a": 3, "b": 0, "c": 2}) == pytest.approx(8.5 / (7 + 3 / 1.5849625), 1e-6)

    def test_perfect_ranking_scores_one(self):
        assert ndcg_at_k(["a", "c", "d", "b"], QRELS) == pytest.approx(1.0)

    def test_unjudged_documents_count_as_zero(self):
        assert ndcg_at_k(["x", "a", "c", "d"], QRELS) < ndcg_at_k(["a", "c", "d", "x"], QRELS)

    def test_ideal_uses_all_judgments_not_just_retrieved(self):
        assert ndcg_at_k(["a"], QRELS) < 1.0

    def test_only_top_k_counts(self):
        assert ndcg_at_k(["b", "a"], QRELS, k=1) == 0.0

    def test_no_relevant_documents(self):
        assert ndcg_at_k(["a"], {"a": 0}) == 0.0
        assert ndcg_at_k([], QRELS) == 0.0


def test_judged_at_k():
    assert judged_at_k(["a", "x", "b", "y"], QRELS, k=4) == 0.5
    assert judged_at_k(["a"], QRELS, k=10) == 0.1
    assert judged_at_k([], QRELS) == 0.0


def test_recall_counts_grade_two_and_up():
    assert recall_at_k(["a", "d"], QRELS) == 0.5
    assert recall_at_k(["a", "c"], QRELS) == 1.0
    assert recall_at_k(["a"], {"a": 1}) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["b", "d", "c"], QRELS) == pytest.approx(1 / 3)
    assert reciprocal_rank(["b", "d"], QRELS) == 0.0
    assert reciprocal_rank(["b", "c"], QRELS, k=1) == 0.0


class TestRandomizationTest:
    def test_identical_systems_are_not_significant(self):
        scores = [0.1, 0.5, 0.9, 0.3]
        assert randomization_test(scores, scores, trials=500) == 1.0

    def test_consistent_large_gain_is_significant(self):
        better = [0.9] * 20
        worse = [0.1] * 20
        assert randomization_test(better, worse, trials=2000) < 0.01

    def test_is_symmetric_and_deterministic(self):
        a = [0.2, 0.8, 0.5, 0.6, 0.1]
        b = [0.3, 0.6, 0.5, 0.4, 0.2]
        assert randomization_test(a, b) == randomization_test(b, a)
        assert randomization_test(a, b, seed=1) == randomization_test(a, b, seed=1)

    def test_lengths_must_match(self):
        with pytest.raises(ValueError):
            randomization_test([0.1], [0.1, 0.2])
