"""Replays the 40 evaluation queries against a running API and reports latency percentiles.

uv run uvicorn ai_news.api.app:app            # in one terminal
uv run python src/ai_news/evaluation/latency.py --rounds 5 --concurrency 1
"""

import argparse
import platform
import statistics
import time
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import httpx

from ai_news.evaluation.dataset import EVAL_DIR, QUERIES_FILE, EvalQuery, read_jsonl, write_json

RESULTS_DIR = EVAL_DIR / "latency"


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, so p95 is a latency a real request actually had."""
    ordered = sorted(values)
    rank = max(1, round(q / 100 * len(ordered) + 0.5 - 1e-9))
    return ordered[min(rank, len(ordered)) - 1]


def summarize(values: Sequence[float]) -> dict[str, float]:
    return {
        "p50": round(percentile(values, 50), 1),
        "p95": round(percentile(values, 95), 1),
        "p99": round(percentile(values, 99), 1),
        "mean": round(statistics.fmean(values), 1),
        "max": round(max(values), 1),
    }


def timed_search(client: httpx.Client, query: str, k: int, mode: str = "quality") -> tuple[float, dict[str, float]]:
    started = time.perf_counter()
    response = client.get("/search", params={"q": query, "k": k, "mode": mode})
    elapsed = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return elapsed, response.json()["stages_ms"]


def run(base_url: str, queries: Sequence[str], rounds: int, concurrency: int, k: int, mode: str = "quality") -> dict:
    with httpx.Client(base_url=base_url, timeout=60) as client:
        for query in queries[:5]:
            timed_search(client, query, k, mode)  # warm-up: model sessions, connection pool, Postgres caches
        workload = [query for _ in range(rounds) for query in queries]
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(lambda query: timed_search(client, query, k, mode), workload))
        wall_seconds = time.perf_counter() - started

    stages: dict[str, list[float]] = defaultdict(list)
    for _, stage_ms in results:
        for stage, duration in stage_ms.items():
            stages[stage].append(duration)
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "requests": len(results),
        "mode": mode,
        "concurrency": concurrency,
        "k": k,
        "throughput_rps": round(len(results) / wall_seconds, 2),
        "end_to_end_ms": summarize([elapsed for elapsed, _ in results]),
        "stages_ms": {stage: summarize(values) for stage, values in sorted(stages.items())},
    }


def print_report(report: dict) -> None:
    print(
        f"\n{report['requests']} requests, mode {report['mode']}, concurrency {report['concurrency']}, "
        f"{report['throughput_rps']} req/s on {report['machine']}"
    )
    print(f"  {'':<22}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>8}  (ms)")
    rows = {"end to end (client)": report["end_to_end_ms"], **report["stages_ms"]}
    for name, stats in rows.items():
        print(f"  {name:<22}{stats['p50']:>8.1f}{stats['p95']:>8.1f}{stats['p99']:>8.1f}{stats['max']:>8.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--rounds", type=int, default=5, help="times to replay the 40 queries")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--mode", choices=("quality", "fast"), default="quality")
    args = parser.parse_args()
    queries = [query.query for query in read_jsonl(QUERIES_FILE, EvalQuery)]
    report = run(args.url, queries, args.rounds, args.concurrency, args.k, args.mode)
    print_report(report)
    path = RESULTS_DIR / f"{report['created_at'].replace(':', '')}-{args.mode}-c{args.concurrency}.json"
    write_json(path, report)
    print(f"\nSaved {path.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
