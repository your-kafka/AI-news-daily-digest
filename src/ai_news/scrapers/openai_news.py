from collections.abc import Iterator
from urllib.parse import urlsplit

from ai_news.scrapers.fetcher import Fetcher
from ai_news.scrapers.models import Document
from ai_news.scrapers.parsing import hours_ago, recent_feed_entries
from ai_news.scrapers.runner import run_scraper

FEED_URL = "https://openai.com/news/rss.xml"


def scrape(hours: float = 24) -> Iterator[Document]:
    since = hours_ago(hours)
    with Fetcher() as fetcher:
        feed = fetcher.get(FEED_URL).content
    for entry, published_at in recent_feed_entries(feed, since):
        yield Document(
            source="openai",
            source_id=urlsplit(entry.link).path.strip("/"),
            url=entry.link,
            title=entry.title,
            published_at=published_at,
            content=entry.get("summary", ""),
            content_kind="summary",
            metadata={"categories": [tag.term for tag in entry.get("tags", [])]},
        )


if __name__ == "__main__":
    run_scraper(scrape)
