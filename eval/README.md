# Evaluation set v1

The test collection for measuring retrieval quality (nDCG@10, Recall, MRR). It follows the plan in the main README's [Evaluation](../README.md#evaluation) section.

| File | What it holds | Made by |
|---|---|---|
| `queries.jsonl` | 40 test queries, each with an `id`, the `query` text and a `type`: `exact` (names of models, papers, products), `paraphrase` (a concept in other words), `time` (scoped to a period) or `cross_source` (a story covered by several sources) | Written by hand, before looking at any system output |
| `snapshot.json` | The frozen corpus: settings, counts, and the `content_hash` of every document, keyed `source:source_id` | `pooling.py` |
| `pool.jsonl` | Up to 15 documents per query, collected round-robin from four systems: BM25 only, dense only, hybrid (BM25 + dense + RRF + pooling) and hybrid + cross-encoder. `found_by` records each system's rank for the document | `pooling.py` |
| `judgments.csv` | One row per (query, document): the LLM's grade and reason, and an empty `human_grade` column for review | `labeling.py` |
| `results/` | One JSON file per evaluation run: settings, averages per system, significance, and every query's scores | `evaluate.py` |

Documents are identified by `source:source_id` (for example `huggingface:papers/2510.12345`), never by database id, so the set survives a rebuilt database.

## Grades

| Grade | Meaning |
|---|---|
| 3 | Highly relevant: mainly about what the query asks for |
| 2 | Relevant: covers the query substantially, but as one of several topics or only partly |
| 1 | Marginal: mentions the topic in passing, or only loosely related |
| 0 | Not relevant |

The full rubric the LLM follows is `RUBRIC` in `src/ai_news/evaluation/labeling.py`.

## Building it

```bash
uv run python src/ai_news/evaluation/pooling.py    # needs queries.jsonl; writes snapshot.json and pool.jsonl
uv run python src/ai_news/evaluation/labeling.py   # needs GROQ_API_KEY and GROQ_MODEL; writes judgments.csv
uv run python src/ai_news/evaluation/evaluate.py   # runs five systems on the 40 queries; writes results/<time>.json
```

`evaluate.py` scores with `human_grade` where it is filled in and falls back to `llm_grade` otherwise, and prints which kind of grades it used. Documents that were never judged count as not relevant, so check `judged@10` before trusting a difference.

`pooling.py` refuses to overwrite an existing `pool.jsonl`, because a new pool invalidates the judgments. `labeling.py` saves after every query. If Groq's rate limits stop it, run it again and it continues where it stopped. It never overwrites a `human_grade`.

## Reviewing

Open `judgments.csv` in a spreadsheet, read each row's query, title and passage (follow the `url` when the passage isn't enough), and fill in `human_grade` with 0–3. Agreeing with the LLM is fine; copy its grade. Review every row. Metrics are computed from human grades, and LLM–human agreement (Cohen's κ) is reported alongside them.

## Freezing

The snapshot is only frozen if the database stops changing. Before running ingestion again, export the database (for example `docker compose exec db pg_dump -U lodestar -Fc lodestar > eval/snapshot_v1.dump`), or compare against `snapshot.json` and report the drift.
