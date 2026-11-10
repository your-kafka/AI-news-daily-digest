import logging
from collections.abc import Iterator
from datetime import datetime
from urllib.parse import urljoin, urlsplit

import lxml.html
from defusedxml import ElementTree

from ai_news.scrapers.fetcher import Fetcher, FetchError
from ai_news.scrapers.models import Document
from ai_news.scrapers.parsing import ArticlePage, hours_ago, parse_datetime
from ai_news.scrapers.runner import run_scraper

SITE_URL = "https://www.anthropic.com"
SITE_HOST = "www.anthropic.com"
SECTIONS = ("news", "research", "engineering")
SITEMAP_NAMESPACES = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
ARTICLE_BODY_XPATH = "//div[contains(@class, '__body')]"

log = logging.getLogger(__name__)


def scrape(hours: float = 24) -> Iterator[Document]:
    since = hours_ago(hours)
    with Fetcher() as fetcher:
        for url in candidate_urls(fetcher, since):
            try:
                page = ArticlePage(fetcher.get(url).text)
            except FetchError as error:
                log.warning("Skipping article: %s", error)
                continue
            published_at = page.published_at()
            if published_at and published_at >= since:
                yield to_document(url, page, published_at)


def candidate_urls(fetcher: Fetcher, since: datetime) -> list[str]:
    sitemap = sitemap_articles(fetcher)
    recently_modified = {url for url, modified_at in sitemap.items() if modified_at is None or modified_at >= since}
    not_in_sitemap_yet = listed_articles(fetcher) - set(sitemap)
    return sorted(recently_modified | not_in_sitemap_yet)


def sitemap_articles(fetcher: Fetcher) -> dict[str, datetime | None]:
    root = ElementTree.fromstring(fetcher.get(f"{SITE_URL}/sitemap.xml").content)
    articles = {}
    for entry in root.iterfind("sm:url", SITEMAP_NAMESPACES):
        url = (entry.findtext("sm:loc", namespaces=SITEMAP_NAMESPACES) or "").strip()
        if section_of(url):
            articles[url] = parse_datetime(entry.findtext("sm:lastmod", namespaces=SITEMAP_NAMESPACES))
    return articles


def listed_articles(fetcher: Fetcher) -> set[str]:
    urls = set()
    for section in SECTIONS:
        tree = lxml.html.fromstring(fetcher.get(f"{SITE_URL}/{section}").text)
        for href in tree.xpath("//a/@href"):
            url = urljoin(SITE_URL, href).split("#")[0].split("?")[0]
            if section_of(url):
                urls.add(url)
    return urls


def section_of(url: str) -> str | None:
    parts = urlsplit(url)
    path_segments = parts.path.strip("/").split("/")
    if parts.netloc == SITE_HOST and len(path_segments) == 2 and path_segments[0] in SECTIONS:
        return path_segments[0]
    return None


def to_document(url: str, page: ArticlePage, published_at: datetime) -> Document:
    return Document(
        source="anthropic",
        source_id=urlsplit(url).path.strip("/"),
        url=url,
        title=page.title() or url,
        published_at=published_at,
        content=page.markdown(ARTICLE_BODY_XPATH),
        content_kind="article",
        authors=page.authors(),
        metadata={"section": section_of(url), "description": page.meta("description")},
    )


if __name__ == "__main__":
    run_scraper(scrape)
