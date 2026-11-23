import httpx
import pytest

from ai_news.evaluation import latency
from ai_news.rag.reranker import CrossEncoderReranker
from tests.conftest import make_post


@pytest.mark.parametrize(("q", "expected"), [(50, 50), (95, 95), (99, 99), (100, 100), (1, 1)])
def test_percentile_is_nearest_rank(q, expected):
    assert latency.percentile(list(range(100, 0, -1)), q) == expected


def test_percentile_of_small_samples_returns_a_real_value():
    assert latency.percentile([10.0, 20.0, 30.0], 95) == 30.0
    assert latency.percentile([10.0, 20.0, 30.0], 50) == 20.0


def test_summarize():
    assert latency.summarize([1.0, 2.0, 3.0, 4.0]) == {"p50": 2.0, "p95": 4.0, "p99": 4.0, "mean": 2.5, "max": 4.0}


def test_run_replays_queries_and_collects_stage_timings(monkeypatch, capsys):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params["q"])
        return httpx.Response(200, json={"stages_ms": {"4 cross-encoder": 100.0, "1 hybrid candidates": 20.0}})

    real_client = httpx.Client
    monkeypatch.setattr(
        latency.httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)
    )
    report = latency.run("http://api", ["a", "b"], rounds=3, concurrency=2, k=10)
    assert report["requests"] == 6
    assert seen.count("a") == 4  # one warm-up + three rounds
    assert report["stages_ms"]["4 cross-encoder"]["p95"] == 100.0
    latency.print_report(report)
    assert "6 requests, mode quality, concurrency 2" in capsys.readouterr().out


class RecordingModel:
    def __init__(self) -> None:
        self.passages: list[str] = []

    def rerank(self, query, passages, batch_size):
        self.passages = list(passages)
        return [1.0] * len(passages)


def test_reranker_truncates_passages():
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.batch_size, reranker.max_characters, reranker.model = 32, 10, RecordingModel()
    reranker.rerank(["q"], [make_post("a")])
    assert reranker.model.passages == ["Title a\n\nt"]
