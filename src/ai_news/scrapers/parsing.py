import json
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Any

import feedparser
import lxml.html
import trafilatura


def hours_ago(hours: float) -> datetime:
    return datetime.now(UTC) - timedelta(hours=hours)


def days_since(moment: datetime) -> list[date]:
    first_day = moment.astimezone(UTC).date()
    today = datetime.now(UTC).date()
    return [first_day + timedelta(days=offset) for offset in range((today - first_day).days + 1)]


def parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def recent_feed_entries(feed: bytes, since: datetime) -> Iterator[tuple[feedparser.FeedParserDict, datetime]]:
    for entry in feedparser.parse(feed).entries:
        published = entry.get("published_parsed") or entry.get("updated_parsed")
        if published is None:
            continue
        published_at = datetime(*published[:6], tzinfo=UTC)
        if published_at >= since:
            yield entry, published_at


class ArticlePage:
    def __init__(self, html: str) -> None:
        self.html = html
        self.tree = lxml.html.fromstring(html)
        self.linked_data = list(linked_data_objects(self.tree))

    def meta(self, name: str) -> str | None:
        values = self.tree.xpath("//meta[@property=$name or @name=$name]/@content", name=name)
        return values[0].strip() if values else None

    def published_at(self) -> datetime | None:
        candidates = [item.get("datePublished") for item in self.linked_data]
        candidates += [self.meta("article:published_time"), self.meta("published_time")]
        for value in candidates:
            if parsed := parse_datetime(value):
                return parsed
        return None

    def title(self) -> str | None:
        for item in self.linked_data:
            headline = item.get("headline")
            if isinstance(headline, str) and headline.strip():
                return " ".join(headline.split())
        heading = " ".join(self.tree.xpath("string(//h1)").split())
        return heading or self.meta("og:title")

    def authors(self) -> list[str]:
        for item in self.linked_data:
            author = item.get("author")
            people = author if isinstance(author, list) else [author]
            names = [
                person["name"].strip()
                for person in people
                if isinstance(person, dict) and isinstance(person.get("name"), str)
            ]
            if names:
                return names
        listed = self.meta("article:author") or self.meta("author") or ""
        return [name.strip() for name in listed.split(",") if name.strip()]

    def markdown(self, container_xpath: str | None = None) -> str:
        containers = self.tree.xpath(container_xpath) if container_xpath else []
        html = lxml.html.tostring(containers[0], encoding="unicode") if containers else self.html
        text = trafilatura.extract(html, output_format="markdown", include_tables=True, include_comments=False)
        return text or ""


def linked_data_objects(tree: lxml.html.HtmlElement) -> Iterator[dict[str, Any]]:
    for script in tree.xpath("//script[@type='application/ld+json']/text()"):
        try:
            data = json.loads(script)
        except ValueError:
            continue
        if isinstance(data, dict):
            data = data.get("@graph", [data])
        if isinstance(data, list):
            yield from (item for item in data if isinstance(item, dict))
