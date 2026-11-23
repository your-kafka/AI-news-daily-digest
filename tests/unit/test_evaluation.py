import json

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from ai_news.evaluation import dataset, evaluate
from ai_news.evaluation.dataset import (
    EvalQuery,
    Judgment,
    PooledDocument,
    document_key,
    read_jsonl,
    read_judgments,
    write_json,
    write_jsonl,
    write_judgments,
)
from ai_news.evaluation.labeling import (
    DocumentGrade,
    Grades,
    format_documents,
    grade_query,
    in_query_order,
    keep_human_grades,
)
from ai_news.evaluation.pooling import round_robin, to_pooled
from ai_news.rag.ranking import first_per_document
from tests.conftest import make_post


def judgment(doc_key: str = "openai:a", query_id: str = "q01", **fields) -> Judgment:
    defaults = {"query": "q", "title": "T", "url": "u", "passage": "p"}
    return Judgment(query_id=query_id, doc_key=doc_key, **{**defaults, **fields})


def pooled(doc_key: str, query_id: str = "q01") -> PooledDocument:
    return PooledDocument(
        query_id=query_id,
        doc_key=doc_key,
        title=f"Title {doc_key}",
        source="openai",
        url="u",
        published_at="2026-11-01T00:00:00+00:00",
        passage="word " * 400,
        found_by={},
    )


class TestDataset:
    def test_document_key(self):
        assert document_key({"source": "huggingface", "source_id": "papers/2510.1"}) == "huggingface:papers/2510.1"

    def test_judgments_round_trip_through_csv(self, tmp_path):
        rows = [
            judgment(query='q, with "quotes"', passage="line1\nline2", llm_grade=2, llm_reason="r"),
            judgment("openai:b", human_grade=0),
        ]
        path = tmp_path / "judgments.csv"
        write_judgments(rows, path)
        assert read_judgments(path) == rows

    def test_empty_grade_cells_read_as_none_and_zero_stays_zero(self, tmp_path):
        path = tmp_path / "judgments.csv"
        write_judgments([judgment(llm_grade=0)], path)
        [row] = read_judgments(path)
        assert row.llm_grade == 0
        assert row.human_grade is None

    @pytest.mark.parametrize("grade", [-1, 4])
    def test_grades_outside_zero_to_three_are_rejected(self, grade):
        with pytest.raises(ValidationError):
            judgment(human_grade=grade)

    def test_missing_judgments_file_reads_as_empty(self, tmp_path):
        assert read_judgments(tmp_path / "missing.csv") == []

    def test_jsonl_round_trip_skips_blank_lines(self, tmp_path):
        path = tmp_path / "sub" / "queries.jsonl"
        queries = [EvalQuery(id="q01", query="a", type="exact"), EvalQuery(id="q02", query="b", type="time")]
        write_jsonl(path, queries)
        path.write_text(path.read_text() + "\n\n")
        assert read_jsonl(path, EvalQuery) == queries

    def test_write_json_keeps_unicode(self, tmp_path):
        path = tmp_path / "x" / "snapshot.json"
        write_json(path, {"title": "Le Chonk — Mistral"})
        assert "—" in path.read_text()
        assert json.loads(path.read_text()) == {"title": "Le Chonk — Mistral"}

    def test_query_files_are_valid(self):
        queries = read_jsonl(dataset.QUERIES_FILE, EvalQuery)
        assert len(queries) == 40
        assert len({query.id for query in queries}) == 40


class TestPooling:
    def test_first_per_document_keeps_each_documents_best_chunk(self):
        chunks = [make_post("a", 1), make_post("a", 1), make_post("b", 2)]
        assert [chunk.metadata["source_id"] for chunk in first_per_document(chunks)] == ["a", "b"]
        assert len(first_per_document(chunks, limit=1)) == 1

    RANKINGS = {
        "bm25": [make_post("a"), make_post("b"), make_post("c")],
        "dense": [make_post("b"), make_post("d")],
        "hybrid": [make_post("e"), make_post("a")],
        "hybrid_rerank": [make_post("f")],
    }

    def test_round_robin_takes_turns_and_records_ranks(self):
        result = round_robin("q01", self.RANKINGS, depth=15)
        assert [doc.doc_key for doc in result] == [f"openai:{key}" for key in "abefdc"]
        assert result[0].found_by == {"bm25": 1, "hybrid": 2}
        assert result[1].found_by == {"bm25": 2, "dense": 1}

    def test_round_robin_stops_at_depth(self):
        assert [doc.doc_key for doc in round_robin("q01", self.RANKINGS, depth=3)] == [
            "openai:a",
            "openai:b",
            "openai:e",
        ]

    def test_to_pooled_prefers_timestamp_url_and_collapses_whitespace(self):
        post = make_post("v", source="youtube", timestamp_url="https://youtu.be/v&t=30s")
        post.page_content = "one\n\n two"
        doc = to_pooled("q01", "youtube:v", post, {"dense": 1})
        assert doc.url == "https://youtu.be/v&t=30s"
        assert doc.passage == "one two"


class TestLabeling:
    def test_format_documents_numbers_and_truncates(self):
        text = format_documents([pooled("openai:a"), pooled("openai:b")])
        assert text.startswith("[1] openai · 2026-11-01 · Title openai:a\n")
        assert "\n\n[2] openai · 2026-11-01 · Title openai:b\n" in text
        assert len(text.split("\n\n")[0]) < 1600

    def test_grade_query_maps_grades_by_number(self):
        grades = Grades(
            grades=[
                DocumentGrade(document=2, reason="on topic", grade=3),
                DocumentGrade(document=9, reason="no such document", grade=1),
            ]
        )
        query = EvalQuery(id="q01", query="q", type="exact")
        result = grade_query(RunnableLambda(lambda _: grades), query, [pooled("openai:a"), pooled("openai:b")], "m")
        assert [(row.doc_key, row.llm_grade, row.llm_reason) for row in result] == [
            ("openai:a", None, ""),
            ("openai:b", 3, "m: on topic"),
        ]
        assert len(result[0].passage) == 600

    def test_relabelling_never_overwrites_human_grades(self):
        old = [judgment(llm_grade=1, human_grade=3)]
        new = [judgment(llm_grade=2), judgment("openai:new", llm_grade=0)]
        kept = keep_human_grades(new, old)
        assert [(row.llm_grade, row.human_grade) for row in kept] == [(2, 3), (0, None)]

    def test_in_query_order(self):
        queries = [EvalQuery(id=f"q0{i}", query="q", type="exact") for i in (1, 2, 3)]
        judgments = {"q02": [judgment(query_id="q02")], "q01": [judgment(query_id="q01")]}
        assert [row.query_id for row in in_query_order(judgments, queries)] == ["q01", "q02"]


class TestEvaluate:
    def test_build_qrels_prefers_human_grades(self):
        qrels, source = evaluate.build_qrels(
            [judgment(llm_grade=1, human_grade=3), judgment("openai:b", llm_grade=2), judgment("openai:c")]
        )
        assert qrels == {"q01": {"openai:a": 3, "openai:b": 2}}
        assert source == "mixed (1/2 human)"

    @pytest.mark.parametrize(
        ("rows", "expected"),
        [
            ([judgment(llm_grade=1)], "llm"),
            ([judgment(llm_grade=1, human_grade=2)], "human"),
        ],
    )
    def test_build_qrels_names_the_grade_source(self, rows, expected):
        assert evaluate.build_qrels(rows)[1] == expected

    def test_score_computes_every_metric_for_every_system(self):
        scores = evaluate.score({"good": ["a", "b"], "bad": ["x"]}, {"a": 3, "b": 2})
        assert scores["good"] == {"ndcg@10": 1.0, "recall@10": 1.0, "mrr@10": 1.0, "judged@10": 0.2}
        assert scores["bad"]["ndcg@10"] == 0.0

    def test_rank_keys(self):
        assert evaluate.rank_keys([make_post("a"), make_post("b", source="anthropic")]) == [
            "openai:a",
            "anthropic:b",
        ]

    def test_averages_and_table(self, capsys):
        rows = [
            {"scores": {system: {metric: 1.0 for metric in evaluate.METRICS} for system in evaluate.SYSTEMS}},
            {"scores": {system: {metric: 0.0 for metric in evaluate.METRICS} for system in evaluate.SYSTEMS}},
        ]
        assert evaluate.averages(rows, "bm25", "ndcg@10") == 0.5
        assert evaluate.averages([], "bm25", "ndcg@10") == 0.0
        evaluate.print_table("All", rows)
        assert "All (2 queries)" in capsys.readouterr().out
