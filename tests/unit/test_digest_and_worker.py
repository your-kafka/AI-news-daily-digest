from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from ai_news.config import RagSettings
from ai_news.pipelines import digest
from ai_news.rag.curation import CurationResult, Digest, DigestItem
from ai_news.rag.ranking import with_metadata
from tests.conftest import make_post

NOW = datetime(2026, 11, 23, 1, 30, tzinfo=UTC)
MODEL = {"model_name": "fake-model"}  # the usage callback only counts responses that name their model


class TestProfiles:
    def test_the_shipped_profiles_load(self):
        from ai_news.config import PROJECT_ROOT

        profiles = digest.load_profiles(PROJECT_ROOT / "config" / "profiles.yaml")
        assert set(profiles) == {"frontier-models", "agents-and-tooling", "safety-and-policy", "papers"}

    def test_duplicate_names_are_rejected(self, tmp_path):
        path = tmp_path / "profiles.yaml"
        profile = "  - {name: papers, title: P, interests: new papers}\n"
        path.write_text("profiles:\n" + profile * 2)
        with pytest.raises(ValueError, match="duplicate"):
            digest.load_profiles(path)

    def test_names_must_be_url_safe(self, tmp_path):
        path = tmp_path / "profiles.yaml"
        path.write_text("profiles:\n  - {name: My Papers, title: P, interests: new papers}\n")
        with pytest.raises(ValidationError):
            digest.load_profiles(path)


def window_document(document_id: int, content: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=document_id, source="openai", published_at=NOW, title=f"Title {document_id}", content=content
    )


def test_naive_prompt_contains_every_document_in_full():
    prompt = digest.naive_prompt("agents", [window_document(1, "alpha " * 50), window_document(2, "beta")])
    assert "Reader's interests: agents" in prompt
    assert "[1] openai · 2026-11-23 · Title 1\n" + "alpha " * 49 + "alpha" in prompt
    assert "[2] openai · 2026-11-23 · Title 2\nbeta" in prompt


def test_o200k_counter_counts_tokens():
    count = digest.o200k_tokens()
    assert 0 < count("hello world") < count("hello world " * 10)
    assert count("<|endoftext|>") > 1  # special tokens are counted as text, never rejected


def test_total_tokens_sums_every_model():
    usage = {"a": {"input_tokens": 100, "output_tokens": 10}, "b": {"input_tokens": 50, "output_tokens": 5}}
    assert digest.total_tokens(usage) == (150, 15)
    assert digest.total_tokens({}) == (0, 0)


def test_digest_items_resolve_documents():
    posts = [make_post("a", 1), make_post("v", 2, source="youtube", timestamp_url="https://yt/v&t=9s")]
    result = CurationResult(
        Digest(items=[DigestItem(headline="H", why_it_matters="W", document_ids=[2, 1])]), "m", 0, 0
    )
    [item] = digest.digest_items(result, posts)
    assert [document["url"] for document in item["documents"]] == ["https://yt/v&t=9s", "https://example.com/a"]


def test_local_today_uses_the_schedule_time_zone():
    assert digest.local_today(datetime(2026, 11, 22, 20, 0, tzinfo=UTC), "Asia/Kolkata") == date(2026, 11, 23)
    assert digest.local_today(datetime(2026, 11, 22, 20, 0, tzinfo=UTC), "UTC") == date(2026, 11, 22)


class StructuredFakeModel(GenericFakeChatModel):
    """A fake chat model that supports with_structured_output and reports token usage like a real one."""

    def with_structured_output(self, schema, include_raw=False, **kwargs):
        def parse(message):
            return {"raw": message, "parsed": schema.model_validate_json(message.content), "parsing_error": None}

        return self | RunnableLambda(parse)


class FakePipeline:
    def __init__(self) -> None:
        self.settings = RagSettings(_env_file=None, retrieval_query_count=2, curation_candidates=5)
        self.calls = []

    def search(self, queries, limit=None, since=None):
        self.calls.append((queries, limit, since))
        return [with_metadata(make_post("a", 1), relevance=0.9), with_metadata(make_post("b", 2), relevance=0.8)]


def test_builder_records_every_llm_calls_tokens_and_the_naive_cost(monkeypatch):
    window = [window_document(i, "word " * 300) for i in range(1, 21)]
    monkeypatch.setattr(digest, "documents_between", lambda start, end: window)
    expansion = GenericFakeChatModel(
        messages=iter(
            [AIMessage("agent memory\ncoding agents", usage_metadata=usage(300, 20), response_metadata=MODEL)]
        )
    )
    chosen = Digest(items=[DigestItem(headline="Agents remember", why_it_matters="It matters", document_ids=[2])])
    curation = StructuredFakeModel(
        messages=iter([AIMessage(chosen.model_dump_json(), usage_metadata=usage(4000, 150), response_metadata=MODEL)])
    )
    pipeline = FakePipeline()
    builder = digest.DigestBuilder(
        pipeline, expansion, curation, "fake-model", count_tokens=lambda text: len(text.split())
    )

    profile = digest.Profile(name="agents", title="Agents", interests="AI agents with memory")
    values = builder.build(profile, NOW, window_hours=24, digest_date=date(2026, 11, 23))

    assert pipeline.calls == [(["agent memory", "coding agents"], 5, datetime(2026, 11, 22, 1, 30, tzinfo=UTC))]
    assert (values["input_tokens"], values["output_tokens"]) == (4300, 170)
    assert values["naive_input_tokens"] > 20 * 300
    assert values["documents_in_window"] == 20
    assert values["candidate_ids"] == [1, 2]
    assert values["items"][0]["documents"] == [
        {"id": 2, "source": "openai", "title": "Title b", "url": "https://example.com/b"}
    ]


def usage(input_tokens: int, output_tokens: int) -> dict[str, int]:
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}


class TestCelery:
    @pytest.fixture
    def eager(self):
        from ai_news.worker.celery_app import app

        app.conf.task_always_eager = True
        app.conf.task_eager_propagates = True
        yield app
        app.conf.task_always_eager = False
        app.conf.task_eager_propagates = False

    def test_routes_and_schedule(self):
        from ai_news.config import WorkerSettings
        from ai_news.worker.celery_app import app, beat_schedule

        router = app.amqp.router
        assert router.route({}, "ingest.source")["queue"].name == "ingest"
        assert router.route({}, "index.new_documents")["queue"].name == "index"
        assert router.route({}, "digest.build")["queue"].name == "index"
        schedule = beat_schedule(WorkerSettings(_env_file=None, ingest_every_hours=6, digest_hour=7))
        assert str(schedule["refresh"]["schedule"]) == "<crontab: 0 */6 * * * (m/h/dM/MY/d)>"
        assert str(schedule["daily-digests"]["schedule"]) == "<crontab: 0 7 * * * (m/h/dM/MY/d)>"
        assert app.conf.task_acks_late and app.conf.worker_prefetch_multiplier == 1

    def test_refresh_scrapes_every_source_then_indexes(self, eager, monkeypatch):
        from ai_news.db.repository import SaveResult
        from ai_news.pipelines.index import IndexReport
        from ai_news.pipelines.ingest import SourceReport
        from ai_news.scrapers.registry import SCRAPERS
        from ai_news.worker import tasks

        scraped, indexed = [], []

        def fake_ingest(source, scrape, hours):
            scraped.append((source, hours))
            return SourceReport(source, 3, SaveResult(2, 1), None)

        monkeypatch.setattr(tasks, "ingest_source", fake_ingest)
        monkeypatch.setattr(tasks, "index_new_documents", lambda: indexed.append(True) or IndexReport(2, 5, None))
        tasks.refresh.delay(hours=8)
        assert sorted(scraped) == sorted((source, 8) for source in SCRAPERS)
        assert indexed == [True]

    def test_build_all_digests_fans_out_per_profile(self, eager, monkeypatch):
        from ai_news.worker import tasks

        built = []

        def fake_run(profile, builder):
            built.append(profile)
            return {"id": 1, "digest_date": date(2026, 11, 23), "items": [], "input_tokens": 1, "naive_input_tokens": 9}

        monkeypatch.setattr(tasks, "run_digest", fake_run)
        monkeypatch.setattr(tasks, "default_builder", lambda pipeline: None)
        monkeypatch.setattr(tasks, "search_pipeline", lambda: None)
        assert (
            set(tasks.build_all_digests.delay().get())
            == set(built)
            == {
                "frontier-models",
                "agents-and-tooling",
                "safety-and-policy",
                "papers",
            }
        )
