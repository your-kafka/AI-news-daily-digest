from datetime import UTC, datetime, timedelta

import pytest

from ai_news.scrapers.models import Document
from ai_news.scrapers.parsing import ArticlePage, days_since, hours_ago, parse_datetime, recent_feed_entries


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-11-05T09:30:00Z", datetime(2026, 11, 5, 9, 30, tzinfo=UTC)),
        (" 2026-11-05T09:30:00+05:30 ", datetime(2026, 11, 5, 4, 0, tzinfo=UTC)),
        ("2026-11-05", datetime(2026, 11, 5, tzinfo=UTC)),
        ("not a date", None),
        ("", None),
        (None, None),
        (12345, None),
    ],
)
def test_parse_datetime(value, expected):
    assert parse_datetime(value) == expected


def test_parse_datetime_assumes_utc_for_naive_values():
    assert parse_datetime("2026-11-05T09:30:00").tzinfo == UTC


def test_hours_ago():
    assert abs(datetime.now(UTC) - timedelta(hours=5) - hours_ago(5)) < timedelta(seconds=5)


def test_days_since_includes_first_day_and_today():
    days = days_since(datetime.now(UTC) - timedelta(days=2))
    assert len(days) == 3
    assert days[-1] == datetime.now(UTC).date()


def test_recent_feed_entries_filters_by_date_and_skips_undated(fixtures):
    entries = list(recent_feed_entries((fixtures / "feed.xml").read_bytes(), datetime(2026, 1, 1, tzinfo=UTC)))
    assert [entry.title for entry, _ in entries] == ["New model released"]
    assert entries[0][1] == datetime(2026, 11, 9, 12, tzinfo=UTC)


class TestArticlePage:
    @pytest.fixture
    def page(self, fixtures):
        return ArticlePage((fixtures / "article.html").read_text())

    @pytest.fixture
    def plain(self, fixtures):
        return ArticlePage((fixtures / "plain_article.html").read_text())

    def test_reads_linked_data_and_skips_invalid_json(self, page):
        assert page.title() == "Introducing Claude Tools"
        assert page.published_at() == datetime(2026, 11, 5, 9, 30, tzinfo=UTC)
        assert page.authors() == ["Ada Lovelace", "Alan Turing"]

    def test_meta(self, page):
        assert page.meta("description") == "A short description."
        assert page.meta("missing") is None

    def test_falls_back_to_meta_tags(self, plain):
        assert plain.published_at() == datetime(2026, 11, 2, 8, tzinfo=UTC)
        assert plain.authors() == ["Grace Hopper", "Katherine Johnson"]
        assert plain.title() == "Only an OG title"

    def test_markdown_from_container_drops_page_chrome(self, page):
        text = page.markdown("//div[contains(@class, '__body')]")
        assert "typed interface" in text
        assert "Partners who tested" in text
        assert "Copyright" not in text
        assert "Heading title" not in text

    def test_markdown_with_missing_container_uses_whole_page(self, page):
        assert "typed interface" in page.markdown("//div[@id='missing']")

    def test_no_dates_or_authors(self):
        page = ArticlePage("<html><body><h1>Just a heading</h1></body></html>")
        assert page.published_at() is None
        assert page.authors() == []
        assert page.title() == "Just a heading"


def test_document_word_count_and_summary():
    document = Document(
        source="openai",
        source_id="index/x",
        url="https://openai.com/index/x",
        title="Title",
        published_at=datetime(2026, 11, 5, 9, 30, tzinfo=UTC),
        content="one two three",
        content_kind="summary",
    )
    assert document.word_count == 3
    assert document.summary().startswith("2026-11-05 09:30 UTC  openai")
    assert "https://openai.com/index/x" in document.summary()
