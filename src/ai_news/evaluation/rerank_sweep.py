"""Trades cross-encoder latency against nDCG@10: how many candidates to rerank, and how much of each passage.

Stage-1 candidates are fetched once per query; each setting then reruns stages 3-5 on them, so the
numbers differ only by the setting under test.

    uv run python src/ai_news/evaluation/rerank_sweep.py
"""

import time
from itertools import product
from statistics import mean

from ai_news.evaluation.dataset import EVAL_DIR, QUERIES_FILE, EvalQuery, read_jsonl, read_judgments, write_json
from ai_news.evaluation.evaluate import build_qrels, rank_keys
from ai_news.evaluation.latency import percentile
from ai_news.evaluation.metrics import judged_at_k, ndcg_at_k
from ai_news.log_config import configure_logging
from ai_news.rag.pipeline import SearchPipeline
from ai_news.rag.ranking import maximal_marginal_relevance, pool, reciprocal_rank_fusion
from ai_news.rag.retriever import Bm25Retriever, PgVectorRetriever

POOL_SIZES = (30, 20, 15)
PASSAGE_CHARACTERS = (None, 1200, 800, 600, 400)
K = 10
REPEATS = 2  # timing runs per query and setting; the faster one is kept to damp scheduler noise


def main() -> None:
    configure_logging()
    queries = read_jsonl(QUERIES_FILE, EvalQuery)
    qrels, grade_source = build_qrels(read_judgments())
    pipeline = SearchPipeline.from_settings()
    settings = pipeline.settings
    reranker = pipeline.reranker

    candidates = {}
    for query in queries:
        bm25 = Bm25Retriever(top_k=settings.retrieval_candidates).invoke(query.query)
        dense = PgVectorRetriever(embeddings=pipeline.embeddings, top_k=settings.retrieval_candidates).invoke(
            query.query
        )
        candidates[query.id] = reciprocal_rank_fusion([bm25, dense], settings.rrf_k)
    reranker.rerank([queries[0].query], pool(candidates[queries[0].id], settings.pooling_method, 5))  # warm-up

    rows = []
    for pool_size, characters in product(POOL_SIZES, PASSAGE_CHARACTERS):
        reranker.max_characters = characters
        rerank_ms, rerank_ndcg, full_ndcg, judged = [], [], [], []
        for query in queries:
            posts = pool(candidates[query.id], settings.pooling_method, pool_size)
            timings = []
            for _ in range(REPEATS):
                started = time.perf_counter()
                reranked = reranker.rerank([query.query], posts, settings.rerank_keep)
                timings.append((time.perf_counter() - started) * 1000)
            final = maximal_marginal_relevance(reranked, settings.mmr_lambda, K)
            rerank_ms.append(min(timings))
            rerank_ndcg.append(ndcg_at_k(rank_keys(reranked), qrels[query.id], K))
            full_ndcg.append(ndcg_at_k(rank_keys(final), qrels[query.id], K))
            judged.append(judged_at_k(rank_keys(final), qrels[query.id], K))
        row = {
            "pool_size": pool_size,
            "passage_characters": characters,
            "ndcg@10_rerank": round(mean(rerank_ndcg), 4),
            "ndcg@10_full": round(mean(full_ndcg), 4),
            "judged@10_full": round(mean(judged), 3),
            "cross_encoder_ms_p50": round(percentile(rerank_ms, 50), 1),
            "cross_encoder_ms_p95": round(percentile(rerank_ms, 95), 1),
        }
        rows.append(row)
        print(
            f"pool {pool_size:>2}  chars {characters or 'all':>4}  nDCG rerank {row['ndcg@10_rerank']:.3f}  "
            f"full {row['ndcg@10_full']:.3f}  judged {row['judged@10_full']:.2f}  "
            f"cross-encoder p50 {row['cross_encoder_ms_p50']:>6.0f} ms  p95 {row['cross_encoder_ms_p95']:>6.0f} ms",
            flush=True,
        )
    path = EVAL_DIR / "results" / "rerank_sweep.json"
    write_json(path, {"grades": grade_source, "model": settings.rerank_model, "rows": rows})
    print(f"\nSaved {path.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
