import logging
from collections.abc import Iterator
from urllib.parse import urlsplit

from ai_news.scrapers.fetcher import Fetcher, FetchError
from ai_news.scrapers.models import Document
from ai_news.scrapers.parsing import ArticlePage, hours_ago, recent_feed_entries
from ai_news.scrapers.runner import run_scraper

FEED_URL = "https://deepmind.google/blog/rss.xml"

log = logging.getLogger(__name__)


def scrape(hours: float = 24) -> Iterator[Document]:
    since = hours_ago(hours)
    with Fetcher() as fetcher:
        feed = fetcher.get(FEED_URL).content
        for entry, published_at in recent_feed_entries(feed, since):
            try:
                page = ArticlePage(fetcher.get(entry.link).text)
            except FetchError as error:
                log.warning("Skipping post: %s", error)
                continue
            yield Document(
                source="deepmind",
                source_id=urlsplit(entry.link).path.strip("/"),
                url=entry.link,
                title=entry.title,
                published_at=published_at,
                content=page.markdown(),
                content_kind="article",
                authors=page.authors(),
            )


if __name__ == "__main__":
    run_scraper(scrape)
