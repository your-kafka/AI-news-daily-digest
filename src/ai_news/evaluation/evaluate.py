import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from statistics import mean

from langchain_core.documents import Document

from ai_news.evaluation.dataset import (
    EVAL_DIR,
    QUERIES_FILE,
    EvalQuery,
    Judgment,
    document_key,
    read_jsonl,
    read_judgments,
    write_json,
)
from ai_news.evaluation.metrics import judged_at_k, ndcg_at_k, randomization_test, recall_at_k, reciprocal_rank
from ai_news.evaluation.pooling import system_rankings
from ai_news.log_config import configure_logging
from ai_news.rag.pipeline import SearchPipeline

RESULTS_DIR = EVAL_DIR / "results"
K = 10
BASELINE = "dense"
SYSTEMS = ("bm25", "dense", "hybrid", "hybrid_rerank", "full")
METRICS: dict[str, Callable[[Sequence[str], Mapping[str, int], int], float]] = {
    "ndcg@10": ndcg_at_k,
    "recall@10": recall_at_k,
    "mrr@10": reciprocal_rank,
    "judged@10": judged_at_k,
}


def build_qrels(judgments: Sequence[Judgment]) -> tuple[dict[str, dict[str, int]], str]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    human = 0
    for judgment in judgments:
        grade = judgment.human_grade if judgment.human_grade is not None else judgment.llm_grade
        if grade is not None:
            qrels[judgment.query_id][judgment.doc_key] = grade
            human += judgment.human_grade is not None
    graded = sum(len(docs) for docs in qrels.values())
    source = "human" if human == graded else "llm" if human == 0 else f"mixed ({human}/{graded} human)"
    return dict(qrels), source


def rank_keys(posts: Sequence[Document]) -> list[str]:
    return [document_key(post.metadata) for post in posts]


def run_systems(pipeline: SearchPipeline, query: str) -> dict[str, list[str]]:
    rankings = {name: rank_keys(posts) for name, posts in system_rankings(pipeline, query).items()}
    rankings["full"] = rank_keys(pipeline.search([query], limit=K))
    return rankings


def score(rankings: Mapping[str, Sequence[str]], qrels: Mapping[str, int]) -> dict[str, dict[str, float]]:
    return {
        system: {metric: round(function(ranking, qrels, K), 4) for metric, function in METRICS.items()}
        for system, ranking in rankings.items()
    }


def averages(per_query: Sequence[dict], system: str, metric: str) -> float:
    return mean(row["scores"][system][metric] for row in per_query) if per_query else 0.0


def print_table(title: str, per_query: Sequence[dict]) -> None:
    print(f"\n{title} ({len(per_query)} queries)")
    print(f"  {'system':<15}" + "".join(f"{metric:>11}" for metric in METRICS))
    for system in SYSTEMS:
        cells = "".join(f"{averages(per_query, system, metric):>11.3f}" for metric in METRICS)
        print(f"  {system:<15}{cells}")


def main() -> None:
    configure_logging()
    logging.getLogger("ai_news.rag.pipeline").setLevel(logging.WARNING)
    queries = read_jsonl(QUERIES_FILE, EvalQuery)
    qrels, grade_source = build_qrels(read_judgments())
    missing = [query.id for query in queries if query.id not in qrels]
    if missing:
        raise SystemExit(f"No judgments for {', '.join(missing)}. Run labeling.py first.")
    pipeline = SearchPipeline.from_settings()
    per_query = []
    for query in queries:
        rankings = run_systems(pipeline, query.query)
        per_query.append(
            {"id": query.id, "type": query.type, "query": query.query, "scores": score(rankings, qrels[query.id])}
        )

    print(f"Grades: {grade_source}. Unjudged documents count as not relevant; judged@10 shows how many were judged.")
    print_table("All queries", per_query)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in per_query:
        by_type[row["type"]].append(row)
    for query_type, rows in by_type.items():
        print_table(f"type = {query_type}", rows)

    baseline = [row["scores"][BASELINE]["ndcg@10"] for row in per_query]
    significance = {}
    print(f"\nnDCG@10 against {BASELINE} (paired randomization test, 10,000 trials)")
    for system in SYSTEMS:
        if system == BASELINE:
            continue
        values = [row["scores"][system]["ndcg@10"] for row in per_query]
        p_value = randomization_test(values, baseline)
        significance[system] = {"delta": round(mean(values) - mean(baseline), 4), "p_value": round(p_value, 4)}
        print(f"  {system:<15} {mean(values) - mean(baseline):+.3f}   p = {p_value:.4f}")

    settings = pipeline.settings
    created = datetime.now(UTC)
    path = RESULTS_DIR / f"{created:%Y-%m-%dT%H%M%S}.json"
    write_json(
        path,
        {
            "created_at": created.isoformat(timespec="seconds"),
            "grades": grade_source,
            "settings": settings.model_dump(mode="json", exclude={"hf_token"}),
            "summary": {
                system: {metric: round(averages(per_query, system, metric), 4) for metric in METRICS}
                for system in SYSTEMS
            },
            "significance_vs_" + BASELINE: significance,
            "per_query": per_query,
        },
    )
    print(f"\nSaved {path.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
