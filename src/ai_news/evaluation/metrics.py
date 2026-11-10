import math
import random
from collections.abc import Mapping, Sequence

RELEVANT_GRADE = 2


def gain(grade: int) -> float:
    return 2**grade - 1


def dcg(grades: Sequence[int]) -> float:
    return sum(gain(grade) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))


def ndcg_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float:
    ideal = dcg(sorted(qrels.values(), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg([qrels.get(key, 0) for key in ranking[:k]]) / ideal


def judged_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float:
    top = ranking[:k]
    return sum(1 for key in top if key in qrels) / k if top else 0.0


def recall_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float:
    relevant = {key for key, grade in qrels.items() if grade >= RELEVANT_GRADE}
    if not relevant:
        return 0.0
    return len(relevant & set(ranking[:k])) / len(relevant)


def reciprocal_rank(ranking: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float:
    for rank, key in enumerate(ranking[:k], start=1):
        if qrels.get(key, 0) >= RELEVANT_GRADE:
            return 1 / rank
    return 0.0


def randomization_test(a: Sequence[float], b: Sequence[float], trials: int = 10_000, seed: int = 0) -> float:
    """Two-sided paired randomization test: the p-value that mean(a) - mean(b) arose by chance."""
    differences = [x - y for x, y in zip(a, b, strict=True)]
    observed = abs(sum(differences))
    rng = random.Random(seed)
    extreme = sum(
        1 for _ in range(trials) if abs(sum(d if rng.random() < 0.5 else -d for d in differences)) >= observed - 1e-12
    )
    return (extreme + 1) / (trials + 1)
