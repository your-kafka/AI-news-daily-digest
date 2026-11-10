import logging
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.exc import OperationalError, ProgrammingError

from ai_news.db.repository import SaveResult, finish_run, save_documents, start_run
from ai_news.db.session import get_session_factory
from ai_news.log_config import configure_logging
from ai_news.scrapers.fetcher import FetchError
from ai_news.scrapers.models import Document, SourceName
from ai_news.scrapers.registry import SCRAPERS, Scraper

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SourceReport:
    source: SourceName
    found: int
    saved: SaveResult
    error: str | None

    @property
    def unchanged(self) -> int:
        return self.found - self.saved.inserted - self.saved.updated


def ingest(hours: float = 24) -> list[SourceReport]:
    return [ingest_source(source, scrape, hours) for source, scrape in SCRAPERS.items()]


def ingest_source(source: SourceName, scrape: Scraper, hours: float) -> SourceReport:
    session_factory = get_session_factory()
    with session_factory.begin() as session:
        run_id = start_run(session, source, hours)
    log.info("Scraping %s", source)
    documents, error = collect(source, scrape, hours)
    with session_factory.begin() as session:
        saved = save_documents(session, documents)
        finish_run(session, run_id, len(documents), saved, error)
    log.info("%s: %d found, %d new, %d updated", source, len(documents), saved.inserted, saved.updated)
    return SourceReport(source, len(documents), saved, error)


def collect(source: SourceName, scrape: Scraper, hours: float) -> tuple[list[Document], str | None]:
    documents: list[Document] = []
    try:
        documents.extend(scrape(hours))
    except (FetchError, RuntimeError) as error:
        log.error("%s stopped early: %s", source, error)
        return documents, str(error)
    except Exception as error:
        log.exception("%s crashed, continuing with the other sources", source)
        return documents, f"{type(error).__name__}: {error}"
    return documents, None


def print_report(reports: Sequence[SourceReport], hours: float) -> None:
    print(f"\nIngested the last {hours:g} hours into PostgreSQL:")
    print(f"  {'source':<12} {'found':>6} {'new':>6} {'updated':>8} {'unchanged':>10}  status")
    for report in reports:
        status = "ok" if report.error is None else "failed"
        print(
            f"  {report.source:<12} {report.found:>6} {report.saved.inserted:>6} "
            f"{report.saved.updated:>8} {report.unchanged:>10}  {status}"
        )


def main(hours: float = 24) -> None:
    configure_logging()
    try:
        reports = ingest(hours)
    except OperationalError as error:
        raise SystemExit(f"Cannot reach PostgreSQL. Start it with `docker compose up -d`.\n{error.orig}") from error
    except ProgrammingError as error:
        raise SystemExit(f"Database schema is missing. Run `alembic upgrade head`.\n{error.orig}") from error
    print_report(reports, hours)


if __name__ == "__main__":
    main()
