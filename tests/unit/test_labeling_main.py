from types import SimpleNamespace

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr

from ai_news.evaluation import dataset, labeling
from ai_news.evaluation.dataset import EvalQuery, Judgment, PooledDocument, read_judgments, write_jsonl, write_judgments
from ai_news.evaluation.labeling import DocumentGrade, Grades


@pytest.fixture
def eval_dir(tmp_path, monkeypatch):
    queries = [EvalQuery(id=f"q0{i}", query=f"query {i}", type="exact") for i in (1, 2, 3)]
    pool = [
        PooledDocument(
            query_id=query.id,
            doc_key=f"openai:{query.id}-{n}",
            title="T",
            source="openai",
            url="u",
            published_at="2026-11-01",
            passage="p",
            found_by={},
        )
        for query in queries
        for n in (1, 2)
    ]
    write_jsonl(tmp_path / "queries.jsonl", queries)
    write_jsonl(tmp_path / "pool.jsonl", pool)
    judgments = tmp_path / "judgments.csv"
    monkeypatch.setattr(labeling, "QUERIES_FILE", tmp_path / "queries.jsonl")
    monkeypatch.setattr(labeling, "POOL_FILE", tmp_path / "pool.jsonl")
    monkeypatch.setattr(labeling, "read_judgments", lambda: dataset.read_judgments(judgments))
    monkeypatch.setattr(labeling, "write_judgments", lambda rows: dataset.write_judgments(rows, judgments))
    monkeypatch.setattr(
        labeling, "groq_settings", lambda: SimpleNamespace(groq_model="m", groq_api_key=SecretStr("key"))
    )
    return judgments


def install_grader(monkeypatch, behaviour):
    calls: list[str] = []

    def grade(inputs):
        calls.append(inputs["query"])
        return behaviour(inputs["query"])

    monkeypatch.setattr(labeling, "build_grader", lambda model, api_key: RunnableLambda(grade))
    return calls


def all_grades(query):
    return Grades(grades=[DocumentGrade(document=n, reason="r", grade=n) for n in (1, 2)])


def test_grades_every_query_and_skips_unusable_answers(eval_dir, monkeypatch, capsys):
    def behaviour(query):
        if query == "query 2":
            raise ValueError("model returned broken JSON")
        return all_grades(query)

    install_grader(monkeypatch, behaviour)
    labeling.main()
    rows = read_judgments(eval_dir)
    assert [(row.query_id, row.llm_grade) for row in rows] == [("q01", 1), ("q01", 2), ("q03", 1), ("q03", 2)]
    assert "4/4 judgments have an LLM grade" in capsys.readouterr().out


def test_resumes_without_regrading_and_keeps_human_grades(eval_dir, monkeypatch):
    done = [
        Judgment(
            query_id="q01",
            query="query 1",
            doc_key=f"openai:q01-{n}",
            title="T",
            url="u",
            passage="p",
            llm_grade=0,
            human_grade=3,
        )
        for n in (1, 2)
    ]
    partial = Judgment(
        query_id="q02", query="query 2", doc_key="openai:q02-1", title="T", url="u", passage="p", human_grade=2
    )
    write_judgments([*done, partial], eval_dir)
    calls = install_grader(monkeypatch, all_grades)
    labeling.main()
    assert calls == ["query 2", "query 3"]
    rows = {row.doc_key: row for row in read_judgments(eval_dir)}
    assert (rows["openai:q01-1"].llm_grade, rows["openai:q01-1"].human_grade) == (0, 3)
    assert (rows["openai:q02-1"].llm_grade, rows["openai:q02-1"].human_grade) == (1, 2)
    assert len(rows) == 6


def test_stops_on_rate_limits_and_saves_progress(eval_dir, monkeypatch):
    def behaviour(query):
        if query == "query 2":
            raise RuntimeError("429 tokens per day exceeded")
        return all_grades(query)

    install_grader(monkeypatch, behaviour)
    with pytest.raises(SystemExit, match=r"(?s)Stopped at q02.*run this again"):
        labeling.main()
    assert {row.query_id for row in read_judgments(eval_dir)} == {"q01"}
