import logging
from functools import cache
from typing import Any

from celery import chord, group
from celery.signals import worker_process_init
from groq import APIConnectionError, RateLimitError
from sqlalchemy.exc import OperationalError

from ai_news.config import worker_settings
from ai_news.db.session import get_engine
from ai_news.log_config import configure_logging
from ai_news.pipelines.digest import default_builder, load_profiles, run_digest
from ai_news.pipelines.index import index_new_documents
from ai_news.pipelines.ingest import ingest_source
from ai_news.rag.pipeline import SearchPipeline
from ai_news.scrapers.registry import SCRAPERS
from ai_news.worker.celery_app import app

log = logging.getLogger(__name__)
RETRY = {"autoretry_for": (OperationalError,), "retry_backoff": 30, "retry_jitter": True, "max_retries": 3}
LLM_RETRY = {**RETRY, "autoretry_for": (OperationalError, RateLimitError, APIConnectionError), "retry_backoff": 60}


@worker_process_init.connect
def reset_after_fork(**kwargs: Any) -> None:
    """Forked worker processes must not reuse the parent's database connections."""
    configure_logging()
    if get_engine.cache_info().currsize:
        get_engine().dispose(close=False)


@cache
def search_pipeline() -> SearchPipeline:
    """Loads the embedding and cross-encoder models once per worker process, on first use."""
    return SearchPipeline.from_settings()


@app.task(name="ingest.source", **RETRY)
def ingest_source_task(source: str, hours: float) -> dict[str, Any]:
    report = ingest_source(source, SCRAPERS[source], hours)
    return {
        "source": source,
        "found": report.found,
        "inserted": report.saved.inserted,
        "updated": report.saved.updated,
        "error": report.error,
    }


@app.task(name="index.new_documents", **RETRY)
def index_task(ingest_results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    report = index_new_documents()
    embedded = report.embedding.stored.saved if report.embedding else 0
    return {"documents": report.documents, "chunks": report.chunks, "embedded": embedded}


@app.task(name="ingest.refresh")
def refresh(hours: float | None = None) -> str:
    """Scrapes every source in parallel, then indexes whatever is new once they have all finished."""
    hours = hours or worker_settings().ingest_window_hours
    workflow = chord(group(ingest_source_task.s(source, hours) for source in SCRAPERS), index_task.s())
    return workflow.apply_async().id


@app.task(name="digest.build", **LLM_RETRY)
def build_digest_task(profile: str) -> dict[str, Any]:
    values = run_digest(profile, default_builder(search_pipeline()))
    return {
        "id": values["id"],
        "profile": profile,
        "digest_date": values["digest_date"].isoformat(),
        "items": len(values["items"]),
        "input_tokens": values["input_tokens"],
        "naive_input_tokens": values["naive_input_tokens"],
    }


@app.task(name="digest.build_all")
def build_all_digests() -> list[str]:
    profiles = load_profiles(worker_settings().profiles_file)
    group(build_digest_task.s(name) for name in profiles).apply_async()
    return list(profiles)
