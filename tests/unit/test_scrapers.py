from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from ai_news.scrapers import anthropic_news, deepmind_blog, huggingface, openai_news, youtube
from ai_news.scrapers.fetcher import FetchError
from ai_news.scrapers.registry import SCRAPERS
from ai_news.scrapers.runner import run_scraper
from ai_news.scrapers.youtube_channels import CHANNELS, Channel

SINCE = datetime(2026, 11, 1, tzinfo=UTC)


class FakeResponse:
    def __init__(self, body: str | bytes | list | dict) -> None:
        self.body = body

    @property
    def content(self) -> bytes:
        return self.body if isinstance(self.body, bytes) else self.text.encode()

    @property
    def text(self) -> str:
        return self.body.decode() if isinstance(self.body, bytes) else self.body

    def json(self):
        return self.body


class FakeFetcher:
    """Serves canned responses by URL; a missing URL raises FetchError like a 404 would."""

    def __init__(self, pages: dict[str, str | bytes | list | dict | FetchError]) -> None:
        self.pages = pages
        self.requests: list[tuple[str, dict | None]] = []

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def get(self, url, params=None, headers=None):
        self.requests.append((url, params))
        page = self.pages.get(url, FetchError(f"HTTP 404 from {url}"))
        if isinstance(page, FetchError):
            raise page
        return FakeResponse(page)


@pytest.fixture
def since(monkeypatch):
    for module in (anthropic_news, deepmind_blog, huggingface, openai_news, youtube):
        monkeypatch.setattr(module, "hours_ago", lambda hours: SINCE)
    return SINCE


def install(monkeypatch, module, pages) -> FakeFetcher:
    fetcher = FakeFetcher(pages)
    monkeypatch.setattr(module, "Fetcher", fetcher)
    return fetcher


def test_registry_has_every_source():
    assert set(SCRAPERS) == {"youtube", "anthropic", "openai", "deepmind", "huggingface"}


def test_run_scraper_prints_summaries(capsys, fixtures, monkeypatch, since):
    install(monkeypatch, openai_news, {openai_news.FEED_URL: (fixtures / "feed.xml").read_bytes()})
    run_scraper(openai_news.scrape, hours=24)
    output = capsys.readouterr().out
    assert "New model released" in output
    assert "1 documents published in the last 24 hours" in output


class TestOpenAI:
    def test_builds_documents_from_the_feed(self, fixtures, monkeypatch, since):
        install(monkeypatch, openai_news, {openai_news.FEED_URL: (fixtures / "feed.xml").read_bytes()})
        [document] = openai_news.scrape()
        assert document.source_id == "index/new-model"
        assert document.content == "A new model is out."
        assert document.content_kind == "summary"
        assert document.metadata == {"categories": ["Product", "Research"]}


class TestDeepMind:
    def test_fetches_each_post_and_skips_failures(self, fixtures, monkeypatch, since):
        feed = (fixtures / "feed.xml").read_text().replace("new-model", "good").replace("2025", "2026")
        pages = {
            deepmind_blog.FEED_URL: feed,
            "https://example.com/index/good/": (fixtures / "article.html").read_text(),
        }
        install(monkeypatch, deepmind_blog, pages)
        [document] = deepmind_blog.scrape()
        assert document.source_id == "index/good"
        assert document.authors == ["Ada Lovelace", "Alan Turing"]
        assert "typed interface" in document.content


class TestAnthropic:
    @pytest.mark.parametrize(
        ("url", "section"),
        [
            ("https://www.anthropic.com/news/claude", "news"),
            ("https://www.anthropic.com/research/paper", "research"),
            ("https://www.anthropic.com/engineering/post", "engineering"),
            ("https://www.anthropic.com/news", None),
            ("https://www.anthropic.com/news/a/b", None),
            ("https://www.anthropic.com/careers/x", None),
            ("https://example.com/news/claude", None),
        ],
    )
    def test_section_of(self, url, section):
        assert anthropic_news.section_of(url) == section

    def test_sitemap_keeps_only_articles(self, fixtures):
        fetcher = FakeFetcher({f"{anthropic_news.SITE_URL}/sitemap.xml": (fixtures / "sitemap.xml").read_bytes()})
        assert anthropic_news.sitemap_articles(fetcher) == {
            "https://www.anthropic.com/news/fresh-post": datetime(2026, 11, 9, tzinfo=UTC),
            "https://www.anthropic.com/research/old-paper": datetime(2025, 1, 1, tzinfo=UTC),
            "https://www.anthropic.com/engineering/no-lastmod": None,
        }

    def anthropic_site(self, fixtures) -> dict:
        listing = '<a href="/news/brand-new?ref=x#top">New</a><a href="/news/fresh-post">Fresh</a><a href="/jobs">J</a>'
        old_article = (fixtures / "plain_article.html").read_text().replace("2026-11-02", "2025-06-01")
        return {
            f"{anthropic_news.SITE_URL}/sitemap.xml": (fixtures / "sitemap.xml").read_bytes(),
            f"{anthropic_news.SITE_URL}/news": listing,
            f"{anthropic_news.SITE_URL}/research": "<html></html>",
            f"{anthropic_news.SITE_URL}/engineering": "<html></html>",
            "https://www.anthropic.com/news/fresh-post": (fixtures / "article.html").read_text(),
            "https://www.anthropic.com/news/brand-new": (fixtures / "plain_article.html").read_text(),
            "https://www.anthropic.com/engineering/no-lastmod": old_article,
        }

    def test_candidates_are_recent_sitemap_entries_plus_unlisted_articles(self, fixtures):
        fetcher = FakeFetcher(self.anthropic_site(fixtures))
        assert anthropic_news.candidate_urls(fetcher, SINCE) == [
            "https://www.anthropic.com/engineering/no-lastmod",
            "https://www.anthropic.com/news/brand-new",
            "https://www.anthropic.com/news/fresh-post",
        ]

    def test_scrape_keeps_recent_articles_only(self, fixtures, monkeypatch, since):
        install(monkeypatch, anthropic_news, self.anthropic_site(fixtures))
        documents = {document.source_id: document for document in anthropic_news.scrape()}
        assert set(documents) == {"news/brand-new", "news/fresh-post"}
        fresh = documents["news/fresh-post"]
        assert fresh.title == "Introducing Claude Tools"
        assert fresh.metadata == {"section": "news", "description": "A short description."}
        assert "typed interface" in fresh.content

    def test_scrape_skips_articles_that_fail_to_load(self, fixtures, monkeypatch, since):
        site = self.anthropic_site(fixtures)
        site["https://www.anthropic.com/news/fresh-post"] = FetchError("HTTP 500")
        install(monkeypatch, anthropic_news, site)
        assert [document.source_id for document in anthropic_news.scrape()] == ["news/brand-new"]


PAPER = {
    "id": "2611.01234",
    "title": "  Scaling\n  Agents ",
    "summary": "We   study agents.",
    "submittedOnDailyAt": "2026-11-09T10:00:00.000Z",
    "authors": [{"name": "A. Researcher"}, {"name": ""}, {}],
    "upvotes": 42,
    "githubRepo": "https://github.com/x/y",
}


class TestHuggingFace:
    def test_paper_document(self):
        document = huggingface.paper_document(PAPER)
        assert document.source_id == "papers/2611.01234"
        assert document.title == "Scaling Agents"
        assert document.content == "We study agents."
        assert document.authors == ["A. Researcher"]
        assert document.published_at == datetime(2026, 11, 9, 10, tzinfo=UTC)
        assert document.metadata["upvotes"] == 42
        assert document.metadata["arxiv_url"] == "https://arxiv.org/abs/2611.01234"

    def test_scrape_reads_blog_and_one_papers_request_per_day(self, fixtures, monkeypatch):
        recent = (datetime.now(UTC) - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        monkeypatch.setattr(huggingface, "hours_ago", lambda hours: recent)
        feed = (fixtures / "feed.xml").read_text().replace("Mon, 09 Nov 2026", recent.strftime("%a, %d %b %Y"))
        pages = {
            huggingface.BLOG_FEED_URL: feed,
            "https://example.com/index/new-model/": (fixtures / "article.html").read_text(),
            huggingface.DAILY_PAPERS_URL: [{"paper": PAPER}],
        }
        fetcher = install(monkeypatch, huggingface, pages)
        documents = list(huggingface.scrape())
        assert [document.metadata["section"] for document in documents] == ["blog", "papers", "papers"]
        days = [params["date"] for url, params in fetcher.requests if url == huggingface.DAILY_PAPERS_URL]
        assert days == [datetime.now(UTC).date().isoformat(), recent.date().isoformat()]


AI_CHANNEL = Channel("AI Explained", "UC1")
GENERAL_CHANNEL = Channel("Lex Fridman", "UC2", ai_focused=False)


def video(**overrides) -> youtube.Video:
    fields = {
        "video_id": "abc",
        "title": "A video",
        "description": "",
        "channel": AI_CHANNEL,
        "published_at": datetime(2026, 11, 9, tzinfo=UTC),
        "duration": timedelta(minutes=20),
        "live_status": "none",
        "views": 10,
    }
    return youtube.Video(**{**fields, **overrides})


class TestYouTube:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("PT1H2M3S", timedelta(hours=1, minutes=2, seconds=3)),
            ("PT45S", timedelta(seconds=45)),
            ("P1DT1M", timedelta(days=1, minutes=1)),
            ("P0D", timedelta()),
            ("garbage", timedelta()),
        ],
    )
    def test_parse_duration(self, value, expected):
        assert youtube.parse_duration(value) == expected

    @pytest.mark.parametrize(
        ("overrides", "reason"),
        [
            ({}, None),
            ({"live_status": "upcoming"}, "upcoming stream"),
            ({"duration": timedelta(minutes=2)}, "short video"),
            ({"channel": GENERAL_CHANNEL, "title": "Cooking with physics"}, "not about AI"),
            ({"channel": GENERAL_CHANNEL, "title": "Sam Altman on AGI"}, None),
            ({"channel": GENERAL_CHANNEL, "description": "We discuss large language models and LLMs"}, None),
            ({"channel": GENERAL_CHANNEL, "title": "Training a rail network"}, "not about AI"),
        ],
    )
    def test_skip_reason(self, overrides, reason):
        assert youtube.skip_reason(video(**overrides)) == reason

    def test_to_video(self):
        item = {
            "id": "abc",
            "snippet": {"title": "T", "publishedAt": "2026-11-09T10:00:00Z", "liveBroadcastContent": "none"},
            "contentDetails": {"duration": "PT10M"},
            "statistics": {"viewCount": "1234"},
        }
        result = youtube.to_video(item, AI_CHANNEL)
        assert (result.duration, result.views, result.description) == (timedelta(minutes=10), 1234, "")
        assert result.url == "https://www.youtube.com/watch?v=abc"

    def fetched(self, *texts: str):
        snippets = [SimpleNamespace(start=float(i * 5), duration=5.0, text=text) for i, text in enumerate(texts)]
        return SimpleNamespace(language_code="en", is_generated=True, snippets=snippets)

    def test_document_from_transcript_drops_blank_segments(self):
        document = youtube.to_document(video(), self.fetched("hello\n world", "  ", "again"))
        assert document.content_kind == "transcript"
        assert document.content == "hello world again"
        assert [segment.start for segment in document.transcript.segments] == [0.0, 10.0]
        assert document.metadata["duration_seconds"] == 1200

    def test_document_without_transcript_uses_description(self):
        for fetched in (None, self.fetched(" ")):
            document = youtube.to_document(video(description="About the video"), fetched)
            assert (document.content_kind, document.content, document.transcript) == (
                "summary",
                "About the video",
                None,
            )

    def test_find_recent_videos_and_scrape(self, monkeypatch, since):
        channel = CHANNELS[0]
        api = youtube.API_URL
        playlist_item = {"contentDetails": {"videoId": "new", "videoPublishedAt": "2026-11-09T00:00:00Z"}}
        old_item = {"contentDetails": {"videoId": "old", "videoPublishedAt": "2026-10-01T00:00:00Z"}}
        details = {
            "id": "new",
            "snippet": {"title": "New AI video", "publishedAt": "2026-11-09T00:00:00Z"},
            "contentDetails": {"duration": "PT12M"},
        }
        responses = {
            f"{api}/channels": {
                "items": [{"id": channel.channel_id, "contentDetails": {"relatedPlaylists": {"uploads": "UU1"}}}]
            },
            f"{api}/playlistItems": {"items": [playlist_item, old_item]},
            f"{api}/videos": {"items": [details]},
        }
        install(monkeypatch, youtube, responses)
        monkeypatch.setattr(youtube, "scraper_settings", lambda: SimpleNamespace(youtube_api_key=SecretStr("key")))
        monkeypatch.setattr(youtube.TranscriptFetcher, "fetch", lambda self, video_id: self_fetched(video_id))

        def self_fetched(video_id):
            return self.fetched(f"transcript of {video_id}")

        [document] = youtube.scrape()
        assert document.source_id == "new"
        assert document.content == "transcript of new"
        assert document.authors == [channel.name]

    def test_scrape_needs_an_api_key(self, monkeypatch):
        monkeypatch.setattr(youtube, "scraper_settings", lambda: SimpleNamespace(youtube_api_key=None))
        with pytest.raises(RuntimeError, match="YOUTUBE_API_KEY"):
            list(youtube.scrape())


class TestTranscriptFetcher:
    class FakeApi:
        def __init__(self, error: Exception | None) -> None:
            self.error = error
            self.calls = 0

        def fetch(self, video_id, languages):
            self.calls += 1
            if self.error:
                raise self.error
            return f"transcript {video_id}"

    def make(self, monkeypatch, error=None):
        monkeypatch.setattr(youtube.time, "sleep", lambda seconds: None)
        fetcher = youtube.TranscriptFetcher()
        fetcher._api = self.FakeApi(error)
        return fetcher

    def test_returns_the_transcript(self, monkeypatch):
        assert self.make(monkeypatch).fetch("abc") == "transcript abc"

    def test_missing_transcript_returns_none_and_keeps_going(self, monkeypatch):
        fetcher = self.make(monkeypatch, youtube.CouldNotRetrieveTranscript("abc"))
        assert fetcher.fetch("abc") is None
        assert fetcher.fetch("def") is None
        assert fetcher._api.calls == 2

    def test_stops_asking_once_blocked(self, monkeypatch):
        fetcher = self.make(monkeypatch, youtube.RequestBlocked("abc"))
        assert fetcher.fetch("abc") is None
        assert fetcher.fetch("def") is None
        assert fetcher._api.calls == 1
