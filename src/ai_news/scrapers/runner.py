from collections.abc import Callable, Iterable

from ai_news.log_config import configure_logging
from ai_news.scrapers.models import Document


def run_scraper(scrape: Callable[[float], Iterable[Document]], hours: float = 24) -> None:
    configure_logging()
    count = 0
    for document in scrape(hours):
        print(document.summary(), flush=True)
        count += 1
    print(f"\n{count} documents published in the last {hours:g} hours")
