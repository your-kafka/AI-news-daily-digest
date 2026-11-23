import logging
import secrets
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Protocol

import redis
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from langchain_core.documents import Document
from pydantic import ValidationError
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session

from ai_news.api.schemas import (
    DigestResponse,
    DocumentDetail,
    Health,
    ProfileSummary,
    QueuedTask,
    SearchMode,
    SearchResponse,
    SubscriptionRequest,
    SubscriptionResponse,
    to_detail,
    to_digest,
    to_hit,
)
from ai_news.config import EmailSettings, RagSettings, email_settings, worker_settings
from ai_news.db.models import DigestRecord, DocumentRecord
from ai_news.db.session import get_engine, get_session_factory
from ai_news.log_config import configure_logging
from ai_news.pipelines.digest import Profile, load_profiles
from ai_news.rag.pipeline import SearchPipeline
from ai_news.scrapers.models import SourceName
from ai_news.subscriptions import Outcome, confirm, subscribe, unsubscribe

MAX_QUERY_CHARACTERS = 500
MAX_RESULTS = 50
DASHBOARD = Path(__file__).parent / "static" / "index.html"

log = logging.getLogger(__name__)


class Searcher(Protocol):
    settings: RagSettings

    def search(
        self,
        queries: list[str],
        limit: int | None = None,
        since: datetime | None = None,
        sources: list[str] | None = None,
        timings: dict[str, float] | None = None,
    ) -> list[Document]: ...

    def search_fast(
        self,
        queries: list[str],
        limit: int | None = None,
        since: datetime | None = None,
        sources: list[str] | None = None,
        timings: dict[str, float] | None = None,
    ) -> list[Document]: ...


def get_searcher(request: Request) -> Searcher:
    return request.app.state.searcher


def get_session() -> Iterator[Session]:
    with get_session_factory()() as session:
        yield session


def get_profiles() -> dict[str, Profile]:
    return load_profiles(worker_settings().profiles_file)


def ping_redis() -> None:
    redis.Redis.from_url(worker_settings().redis_url, socket_timeout=1, socket_connect_timeout=1).ping()


def queue_refresh() -> str:
    from ai_news.worker.tasks import refresh

    return refresh.delay().id


def get_enqueuer() -> Callable[[], str]:
    return queue_refresh


class Jobs:
    """Background jobs the API hands to Celery; tests replace it."""

    def send_confirmation(self, subscriber_id: int) -> None:
        from ai_news.worker.tasks import send_confirmation_task

        send_confirmation_task.delay(subscriber_id)

    def send_first_digest(self, subscriber_id: int) -> None:
        from ai_news.worker.tasks import subscriber_digest_task

        subscriber_digest_task.delay(subscriber_id)


def get_jobs() -> Jobs:
    return Jobs()


def get_email_settings() -> EmailSettings:
    try:
        return email_settings()
    except ValidationError as error:
        raise HTTPException(
            status_code=503, detail="Email isn't set up on this server: set SMTP_USERNAME, SMTP_PASSWORD, EMAIL_FROM"
        ) from error


def message_page(title: str, message: str, status_code: int = 200) -> HTMLResponse:
    from html import escape

    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{escape(title)} · Lodestar</title><style>"
        ":root{color-scheme:light dark;--bg:#f6f7f9;--card:#fff;--ink:#1d2433;--muted:#6b7280;--accent:#3b5bdb}"
        "@media (prefers-color-scheme:dark){:root{--bg:#0f1218;--card:#171b24;--ink:#e6e8ee;"
        "--muted:#9aa3b2;--accent:#7b93ff}}"
        "body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--ink);"
        "font-family:-apple-system,Segoe UI,Roboto,sans-serif;padding:16px}"
        ".card{max-width:440px;background:var(--card);border-radius:14px;padding:32px;text-align:center}"
        "a{color:var(--accent)}</style></head><body><div class='card'>"
        f"<h1>{escape(title)}</h1><p>{escape(message)}</p><p><a href='/'>Back to Lodestar</a></p></div></body></html>"
    )
    return HTMLResponse(page, status_code=status_code)


def require_admin(authorization: Annotated[str | None, Header()] = None) -> None:
    token = worker_settings().admin_token
    if token is None:
        raise HTTPException(status_code=403, detail="Admin endpoints are disabled: set ADMIN_TOKEN to enable them")
    expected = f"Bearer {token.get_secret_value()}"
    if authorization is None or not secrets.compare_digest(authorization.encode(), expected.encode()):
        raise HTTPException(
            status_code=401, detail="Missing or wrong admin token", headers={"WWW-Authenticate": "Bearer"}
        )


def server_timing(stages_ms: dict[str, float], total_ms: float) -> str:
    """Formats stage durations for the Server-Timing header, which browser dev tools and load tools can read."""
    entries = [f'stage{stage.split()[0]};desc="{stage}";dur={duration:.1f}' for stage, duration in stages_ms.items()]
    return ", ".join([*entries, f"total;dur={total_ms:.1f}"])


def as_utc(moment: datetime | None) -> datetime | None:
    if moment is None or moment.tzinfo is not None:
        return moment
    return moment.replace(tzinfo=UTC)


def create_app(searcher_factory: Callable[[], Searcher] = SearchPipeline.from_settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        app.state.searcher = searcher_factory()  # loads the embedding and cross-encoder models once
        log.info("Search models loaded")
        yield
        if get_engine.cache_info().currsize:
            get_engine().dispose()

    app = FastAPI(
        title="Lodestar",
        summary="Multi-stage retrieval over recent AI news",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(OperationalError)
    async def database_unavailable(request: Request, error: OperationalError) -> JSONResponse:
        log.error("Database unavailable: %s", error.orig)
        return JSONResponse(status_code=503, content={"detail": "Database unavailable"})

    @app.exception_handler(ProgrammingError)
    async def schema_out_of_date(request: Request, error: ProgrammingError) -> JSONResponse:
        log.error("Database schema problem: %s", error.orig)
        return JSONResponse(status_code=503, content={"detail": "Database schema is out of date"})

    @app.exception_handler(redis.exceptions.ConnectionError)
    async def queue_unavailable(request: Request, error: redis.exceptions.ConnectionError) -> JSONResponse:
        log.error("Redis unavailable: %s", error)
        return JSONResponse(status_code=503, content={"detail": "Task queue unavailable"})

    @app.get("/healthz", tags=["health"])
    def healthz() -> Health:
        """Liveness: the process is up."""
        return Health(status="ok")

    @app.get("/readyz", tags=["health"], responses={503: {"description": "Database or Redis unreachable"}})
    def readyz(session: Annotated[Session, Depends(get_session)], _: Annotated[None, Depends(ping_redis)]) -> Health:
        """Readiness: the database answers, the schema is migrated and Redis is reachable."""
        session.execute(text("SELECT 1 FROM digests LIMIT 1"))
        return Health(status="ready")

    @app.get("/search", tags=["search"])
    def search(
        response: Response,
        searcher: Annotated[Searcher, Depends(get_searcher)],
        q: Annotated[str, Query(min_length=1, max_length=MAX_QUERY_CHARACTERS, description="What to search for")],
        k: Annotated[int | None, Query(ge=1, le=MAX_RESULTS, description="Number of results")] = None,
        source: Annotated[list[SourceName] | None, Query(description="Only these sources (repeatable)")] = None,
        since: Annotated[datetime | None, Query(description="Only posts published at or after this time")] = None,
        mode: Annotated[
            SearchMode,
            Query(description="quality: stages 1-5 with the cross-encoder; fast: dense retrieval only"),
        ] = "quality",
    ) -> SearchResponse:
        """Searches recent AI news.

        `quality` runs BM25 + dense candidates, RRF, pooling, the cross-encoder and MMR (nDCG@10 0.81 on the
        eval set, about 2 s on a laptop CPU). `fast` runs dense retrieval only (nDCG@10 0.75, well under 400 ms).
        """
        query = q.strip()
        if not query:
            raise HTTPException(status_code=422, detail="q must contain non-whitespace characters")
        started = time.perf_counter()
        stages_ms: dict[str, float] = {}
        run = searcher.search if mode == "quality" else searcher.search_fast
        posts = run([query], limit=k, since=as_utc(since), sources=source, timings=stages_ms)
        took_ms = (time.perf_counter() - started) * 1000
        response.headers["Server-Timing"] = server_timing(stages_ms, took_ms)
        return SearchResponse(
            query=query,
            mode=mode,
            count=len(posts),
            took_ms=round(took_ms, 1),
            stages_ms={stage: round(duration, 1) for stage, duration in stages_ms.items()},
            results=[to_hit(rank, post) for rank, post in enumerate(posts, start=1)],
        )

    @app.get("/documents/{document_id}", tags=["documents"], responses={404: {"description": "No such document"}})
    def document(document_id: int, session: Annotated[Session, Depends(get_session)]) -> DocumentDetail:
        """One stored document: metadata, a preview of its content and a link to the original."""
        record = session.scalar(select(DocumentRecord).where(DocumentRecord.id == document_id))
        if record is None:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        return to_detail(record)

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(DASHBOARD, media_type="text/html")

    @app.post("/subscriptions", tags=["subscriptions"], status_code=202, responses={409: {"description": "Full"}})
    def create_subscription(
        request: SubscriptionRequest,
        session: Annotated[Session, Depends(get_session)],
        jobs: Annotated[Jobs, Depends(get_jobs)],
        settings: Annotated[EmailSettings, Depends(get_email_settings)],
    ) -> SubscriptionResponse:
        """Asks for a daily digest by email. Nothing is sent but a confirmation email until its link is clicked.

        The answer is the same whether or not the address is already subscribed, so the form can't be used to
        find out who reads Lodestar.
        """
        result = subscribe(session, request.email, request.interests, datetime.now(UTC), settings.max_subscribers)
        if result.outcome == Outcome.FULL:
            raise HTTPException(status_code=409, detail="Lodestar isn't taking new subscribers right now")
        session.commit()
        if result.outcome == Outcome.SEND_CONFIRMATION:
            jobs.send_confirmation(result.subscriber_id)
        return SubscriptionResponse()

    @app.get("/subscriptions/confirm", include_in_schema=False)
    def confirm_subscription(
        token: str, session: Annotated[Session, Depends(get_session)], jobs: Annotated[Jobs, Depends(get_jobs)]
    ) -> HTMLResponse:
        subscriber, activated = confirm(session, token, datetime.now(UTC))
        if subscriber is None:
            return message_page("Link not valid", "This confirmation link is wrong or has been replaced.", 404)
        session.commit()
        if activated:
            jobs.send_first_digest(subscriber.id)
            return message_page("You're subscribed", "Your first digest is being written and will arrive shortly.")
        return message_page("Already confirmed", f"This subscription is {subscriber.status}.")

    @app.get("/unsubscribe", include_in_schema=False)
    def unsubscribe_page(token: str, session: Annotated[Session, Depends(get_session)]) -> HTMLResponse:
        subscriber = unsubscribe(session, token, datetime.now(UTC))
        if subscriber is None:
            return message_page("Link not valid", "This unsubscribe link is wrong or has been replaced.", 404)
        session.commit()
        return message_page("Unsubscribed", "You won't get any more digests. You can subscribe again any time.")

    @app.get("/profiles", tags=["digest"])
    def profiles(known: Annotated[dict[str, Profile], Depends(get_profiles)]) -> list[ProfileSummary]:
        """The reader profiles a digest is built for every morning."""
        return [ProfileSummary(**profile.model_dump()) for profile in known.values()]

    def find_digest(session: Session, profile: Profile, day: date | None) -> DigestResponse:
        statement = select(DigestRecord).where(DigestRecord.profile == profile.name)
        statement = statement.where(DigestRecord.digest_date == day) if day else statement
        record = session.scalar(statement.order_by(DigestRecord.digest_date.desc()).limit(1))
        if record is None:
            when = f" for {day}" if day else ""
            raise HTTPException(status_code=404, detail=f"No digest for {profile.name}{when} yet")
        return to_digest(record, profile.title)

    def known_profile(name: str, known: dict[str, Profile]) -> Profile:
        if name not in known:
            raise HTTPException(status_code=404, detail=f"Unknown profile {name!r}; see /profiles")
        return known[name]

    @app.get("/digest/latest", tags=["digest"], responses={404: {"description": "Unknown profile or no digest yet"}})
    def latest_digest(
        session: Annotated[Session, Depends(get_session)],
        known: Annotated[dict[str, Profile], Depends(get_profiles)],
        profile: Annotated[str, Query(description="Profile name, see /profiles")],
    ) -> DigestResponse:
        """The most recent curated digest for a profile, with its token accounting."""
        return find_digest(session, known_profile(profile, known), None)

    @app.get(
        "/digest/{day}", tags=["digest"], responses={404: {"description": "Unknown profile or no digest that day"}}
    )
    def digest_on(
        day: date,
        session: Annotated[Session, Depends(get_session)],
        known: Annotated[dict[str, Profile], Depends(get_profiles)],
        profile: Annotated[str, Query(description="Profile name, see /profiles")],
    ) -> DigestResponse:
        """The digest a profile got on one day (YYYY-MM-DD, in the schedule's time zone)."""
        return find_digest(session, known_profile(profile, known), day)

    @app.post("/admin/ingest", tags=["admin"], status_code=202, dependencies=[Depends(require_admin)])
    def trigger_ingest(enqueue: Annotated[Callable[[], str], Depends(get_enqueuer)]) -> QueuedTask:
        """Queues a scrape of every source followed by indexing. Needs `Authorization: Bearer <ADMIN_TOKEN>`."""
        return QueuedTask(task_id=enqueue())

    return app


app = create_app()
