import logging
from collections.abc import Iterator
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from ai_news.scrapers.fetcher import Fetcher, FetchError
from ai_news.scrapers.models import Document
from ai_news.scrapers.parsing import ArticlePage, days_since, hours_ago, parse_datetime, recent_feed_entries
from ai_news.scrapers.runner import run_scraper

BLOG_FEED_URL = "https://huggingface.co/blog/feed.xml"
DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"
DAILY_PAPERS_LIMIT = 100

log = logging.getLogger(__name__)


def scrape(hours: float = 24) -> Iterator[Document]:
    since = hours_ago(hours)
    with Fetcher() as fetcher:
        yield from blog_posts(fetcher, since)
        yield from daily_papers(fetcher, since)


def blog_posts(fetcher: Fetcher, since: datetime) -> Iterator[Document]:
    feed = fetcher.get(BLOG_FEED_URL).content
    for entry, published_at in recent_feed_entries(feed, since):
        try:
            page = ArticlePage(fetcher.get(entry.link).text)
        except FetchError as error:
            log.warning("Skipping post: %s", error)
            continue
        yield Document(
            source="huggingface",
            source_id=urlsplit(entry.link).path.strip("/"),
            url=entry.link,
            title=entry.title,
            published_at=published_at,
            content=page.markdown(),
            content_kind="article",
            authors=page.authors(),
            metadata={"section": "blog"},
        )


def daily_papers(fetcher: Fetcher, since: datetime) -> Iterator[Document]:
    for day in reversed(days_since(since)):
        params = {"date": day.isoformat(), "limit": DAILY_PAPERS_LIMIT}
        for item in fetcher.get(DAILY_PAPERS_URL, params=params).json():
            yield paper_document(item["paper"])


def paper_document(paper: dict[str, Any]) -> Document:
    arxiv_id = paper["id"]
    return Document(
        source="huggingface",
        source_id=f"papers/{arxiv_id}",
        url=f"https://huggingface.co/papers/{arxiv_id}",
        title=" ".join(paper["title"].split()),
        published_at=parse_datetime(paper.get("submittedOnDailyAt") or paper.get("publishedAt")),
        content=" ".join(paper.get("summary", "").split()),
        content_kind="abstract",
        authors=[author["name"] for author in paper.get("authors", []) if author.get("name")],
        metadata={
            "section": "papers",
            "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
            "upvotes": paper.get("upvotes", 0),
            "github_repo": paper.get("githubRepo"),
        },
    )


if __name__ == "__main__":
    run_scraper(scrape)
