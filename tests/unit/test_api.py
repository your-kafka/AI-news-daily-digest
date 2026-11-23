from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError, ProgrammingError

from ai_news.api.app import create_app, server_timing
from ai_news.api.schemas import snippet
from ai_news.config import RagSettings
from ai_news.rag.ranking import with_metadata
from tests.conftest import make_post


class FakeSearcher:
    def __init__(self, error: Exception | None = None) -> None:
        self.settings = RagSettings(_env_file=None)
        self.error = error
        self.calls: list[dict] = []

    def search(self, queries, limit=None, since=None, sources=None, timings=None):
        self.calls.append({"queries": queries, "limit": limit, "since": since, "sources": sources})
        if self.error:
            raise self.error
        timings.update({"1 hybrid candidates": 80.0, "4 cross-encoder": 300.0})
        video = make_post(
            "vid", 2, source="youtube", timestamp_url="https://youtube.com/watch?v=vid&t=42s", section=None
        )
        article = make_post("post", 1, section="Results")
        article.page_content = "word " * 200
        return [with_metadata(post, relevance=0.98, rerank_score=3.9) for post in (article, video)]

    def search_fast(self, queries, limit=None, since=None, sources=None, timings=None):
        self.calls.append({"fast": True, "queries": queries, "limit": limit})
        timings["1 dense candidates"] = 40.0
        return [with_metadata(make_post("post", 1), similarity=0.83)]


@pytest.fixture
def searcher():
    return FakeSearcher()


@pytest.fixture
def client(searcher):
    with TestClient(create_app(lambda: searcher)) as test_client:
        yield test_client


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_search_returns_ranked_hits_with_timings(client, searcher):
    response = client.get("/search", params={"q": "  agent memory  "})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "agent memory"
    assert body["count"] == 2
    assert body["stages_ms"] == {"1 hybrid candidates": 80.0, "4 cross-encoder": 300.0}
    first, second = body["results"]
    assert (first["rank"], first["doc_key"], first["section"]) == (1, "openai:post", "Results")
    assert first["snippet"].endswith("…")
    assert len(first["snippet"]) <= 301
    assert second["url"] == "https://youtube.com/watch?v=vid&t=42s"
    assert 'stage1;desc="1 hybrid candidates";dur=80.0' in response.headers["Server-Timing"]
    assert searcher.calls == [{"queries": ["agent memory"], "limit": None, "since": None, "sources": None}]


def test_search_passes_filters(client, searcher):
    params = [("q", "chips"), ("k", "5"), ("source", "openai"), ("source", "youtube"), ("since", "2026-11-01")]
    assert client.get("/search", params=params).status_code == 200
    assert searcher.calls[0]["limit"] == 5
    assert searcher.calls[0]["sources"] == ["openai", "youtube"]
    assert searcher.calls[0]["since"] == datetime(2026, 11, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": ""},
        {"q": "   "},
        {"q": "x" * 501},
        {"q": "x", "k": 0},
        {"q": "x", "k": 51},
        {"q": "x", "source": "reddit"},
        {"q": "x", "since": "yesterday"},
    ],
)
def test_search_rejects_bad_input(client, params):
    assert client.get("/search", params=params).status_code == 422


@pytest.mark.parametrize(
    ("error", "detail"),
    [
        (OperationalError("SELECT", {}, Exception("connection refused")), "Database unavailable"),
        (ProgrammingError("SELECT", {}, Exception("no such table")), "Database schema is out of date"),
    ],
)
def test_database_errors_become_503(error, detail):
    with TestClient(create_app(lambda: FakeSearcher(error))) as client:
        response = client.get("/search", params={"q": "x"})
    assert (response.status_code, response.json()) == (503, {"detail": detail})


def test_openapi_documents_every_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {
        "/healthz",
        "/readyz",
        "/search",
        "/documents/{document_id}",
        "/profiles",
        "/digest/latest",
        "/digest/{day}",
        "/admin/ingest",
        "/subscriptions",
    }


def test_server_timing_format():
    assert server_timing({"4 cross-encoder": 12.34}, 20.0) == 'stage4;desc="4 cross-encoder";dur=12.3, total;dur=20.0'


def test_snippet_collapses_whitespace_and_cuts_at_a_word():
    assert snippet("a  b\n\nc") == "a b c"
    assert snippet("alpha beta gamma", limit=12) == "alpha beta…"


def test_fast_mode_uses_dense_retrieval_only(client, searcher):
    body = client.get("/search", params={"q": "chips", "mode": "fast", "k": 3}).json()
    assert body["mode"] == "fast"
    assert body["stages_ms"] == {"1 dense candidates": 40.0}
    hit = body["results"][0]
    assert (hit["similarity"], hit["relevance"], hit["rerank_score"]) == (0.83, None, None)
    assert searcher.calls == [{"fast": True, "queries": ["chips"], "limit": 3}]


def test_quality_is_the_default_mode(client):
    body = client.get("/search", params={"q": "chips"}).json()
    assert body["mode"] == "quality"
    assert body["results"][0]["similarity"] is None


def test_unknown_mode_is_rejected(client):
    assert client.get("/search", params={"q": "chips", "mode": "turbo"}).status_code == 422


class TestAdminAndProfiles:
    @pytest.fixture
    def app_and_client(self, searcher, monkeypatch):
        from types import SimpleNamespace

        from pydantic import SecretStr

        from ai_news.api import app as app_module

        settings = SimpleNamespace(admin_token=SecretStr("s3cret"), profiles_file=None, redis_url="redis://x")
        monkeypatch.setattr(app_module, "worker_settings", lambda: settings)
        application = create_app(lambda: searcher)
        application.dependency_overrides[app_module.get_enqueuer] = lambda: lambda: "task-123"
        application.dependency_overrides[app_module.get_profiles] = lambda: {
            "papers": app_module.Profile(name="papers", title="Papers", interests="new ML papers")
        }
        with TestClient(application) as test_client:
            yield settings, test_client

    def test_admin_ingest_queues_a_refresh(self, app_and_client):
        _, client = app_and_client
        response = client.post("/admin/ingest", headers={"Authorization": "Bearer s3cret"})
        assert (response.status_code, response.json()) == (202, {"task_id": "task-123", "status": "queued"})

    @pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "s3cret"}])
    def test_admin_ingest_rejects_bad_tokens(self, app_and_client, headers):
        _, client = app_and_client
        response = client.post("/admin/ingest", headers=headers)
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_admin_is_disabled_without_a_token(self, app_and_client):
        settings, client = app_and_client
        settings.admin_token = None
        assert client.post("/admin/ingest", headers={"Authorization": "Bearer s3cret"}).status_code == 403

    def test_profiles(self, app_and_client):
        _, client = app_and_client
        assert client.get("/profiles").json() == [{"name": "papers", "title": "Papers", "interests": "new ML papers"}]

    def test_unknown_profile_is_404(self, app_and_client):
        from ai_news.api import app as app_module

        _, client = app_and_client
        client.app.dependency_overrides[app_module.get_session] = lambda: None
        response = client.get("/digest/latest", params={"profile": "nope"})
        assert response.status_code == 404
        assert "see /profiles" in response.json()["detail"]


def test_readyz_is_503_when_redis_is_down(searcher):
    import redis

    from ai_news.api import app as app_module

    class FakeSession:
        def execute(self, statement):
            return None

    def redis_down():
        raise redis.exceptions.ConnectionError("refused")

    application = create_app(lambda: searcher)
    application.dependency_overrides[app_module.get_session] = FakeSession
    application.dependency_overrides[app_module.ping_redis] = redis_down
    with TestClient(application) as client:
        response = client.get("/readyz")
    assert (response.status_code, response.json()) == (503, {"detail": "Task queue unavailable"})
