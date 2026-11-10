import logging
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import batched
from typing import Any

from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    FetchedTranscript,
    RequestBlocked,
    YouTubeTranscriptApi,
)

from ai_news.config import scraper_settings
from ai_news.scrapers.fetcher import Fetcher
from ai_news.scrapers.models import Document, Transcript, TranscriptSegment
from ai_news.scrapers.parsing import hours_ago, parse_datetime
from ai_news.scrapers.runner import run_scraper
from ai_news.scrapers.youtube_channels import CHANNELS, Channel

API_URL = "https://www.googleapis.com/youtube/v3"
PAGE_SIZE = 50
TRANSCRIPT_LANGUAGES = ("en", "en-US", "en-GB")
TRANSCRIPT_DELAY_SECONDS = 2.0
SHORTS_MAX_LENGTH = timedelta(minutes=3)
TOPIC_CHECK_CHARACTERS = 500
DURATION_PATTERN = re.compile(
    r"P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?"
)
AI_TOPICS = re.compile(
    r"\b(?:ai|agi|llms?|gpt[\w.-]*|chatgpt|claude|gemini|llama|deepseek|qwen|grok|mistral"
    r"|openai|anthropic|deepmind|artificial intelligence|machine learning|deep learning"
    r"|neural net\w*|transformers?|diffusion models?|reinforcement learning|agentic"
    r"|chatbots?|superintelligence|alphafold)\b",
    re.IGNORECASE,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Video:
    video_id: str
    title: str
    description: str
    channel: Channel
    published_at: datetime
    duration: timedelta
    live_status: str
    views: int

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


class YouTubeDataApi:
    def __init__(self, fetcher: Fetcher, api_key: str) -> None:
        self._fetcher = fetcher
        self._headers = {"X-Goog-Api-Key": api_key}

    def uploads_playlists(self, channel_ids: list[str]) -> dict[str, str]:
        return {
            item["id"]: item["contentDetails"]["relatedPlaylists"]["uploads"]
            for batch in batched(channel_ids, PAGE_SIZE)
            for item in self._list("channels", part="contentDetails", id=",".join(batch))
        }

    def video_ids_since(self, playlist_id: str, since: datetime) -> list[str]:
        video_ids = []
        for item in self._list("playlistItems", part="contentDetails", playlistId=playlist_id, maxResults=PAGE_SIZE):
            published_at = parse_datetime(item["contentDetails"].get("videoPublishedAt"))
            if published_at and published_at >= since:
                video_ids.append(item["contentDetails"]["videoId"])
        return video_ids

    def video_details(self, video_ids: list[str]) -> list[dict[str, Any]]:
        return [
            item
            for batch in batched(video_ids, PAGE_SIZE)
            for item in self._list("videos", part="snippet,contentDetails,statistics", id=",".join(batch))
        ]

    def _list(self, resource: str, **params: str | int) -> list[dict[str, Any]]:
        response = self._fetcher.get(f"{API_URL}/{resource}", params=params, headers=self._headers)
        return response.json().get("items", [])


class TranscriptFetcher:
    def __init__(self) -> None:
        self._api = YouTubeTranscriptApi()
        self._blocked = False

    def fetch(self, video_id: str) -> FetchedTranscript | None:
        if self._blocked:
            return None
        time.sleep(TRANSCRIPT_DELAY_SECONDS)
        try:
            return self._api.fetch(video_id, languages=TRANSCRIPT_LANGUAGES)
        except RequestBlocked:
            log.error("YouTube is blocking transcript requests from this network, skipping remaining transcripts")
            self._blocked = True
        except CouldNotRetrieveTranscript as error:
            log.warning("No transcript for %s (%s)", video_id, type(error).__name__)
        return None


def scrape(hours: float = 24) -> Iterator[Document]:
    api_key = scraper_settings().youtube_api_key
    if api_key is None:
        raise RuntimeError("YOUTUBE_API_KEY is not set. Add a YouTube Data API v3 key to your .env file.")
    since = hours_ago(hours)
    with Fetcher(min_interval=0.1) as fetcher:
        videos = find_recent_videos(YouTubeDataApi(fetcher, api_key.get_secret_value()), since)
    transcripts = TranscriptFetcher()
    for video in videos:
        if reason := skip_reason(video):
            log.info("Skipping %s: %s", reason, video.title)
            continue
        yield to_document(video, transcripts.fetch(video.video_id))


def find_recent_videos(api: YouTubeDataApi, since: datetime) -> list[Video]:
    channels = {channel.channel_id: channel for channel in CHANNELS}
    playlists = api.uploads_playlists(list(channels))
    for channel_id in channels.keys() - playlists.keys():
        log.warning("Channel not found: %s (%s)", channels[channel_id].name, channel_id)
    channel_of_video = {
        video_id: channels[channel_id]
        for channel_id, playlist_id in playlists.items()
        for video_id in api.video_ids_since(playlist_id, since)
    }
    videos = [to_video(item, channel_of_video[item["id"]]) for item in api.video_details(list(channel_of_video))]
    return sorted(videos, key=lambda video: video.published_at, reverse=True)


def to_video(item: dict[str, Any], channel: Channel) -> Video:
    snippet = item["snippet"]
    return Video(
        video_id=item["id"],
        title=snippet["title"],
        description=snippet.get("description", ""),
        channel=channel,
        published_at=datetime.fromisoformat(snippet["publishedAt"]),
        duration=parse_duration(item["contentDetails"].get("duration", "")),
        live_status=snippet.get("liveBroadcastContent", "none"),
        views=int(item.get("statistics", {}).get("viewCount", 0)),
    )


def parse_duration(value: str) -> timedelta:
    match = DURATION_PATTERN.fullmatch(value)
    if match is None:
        return timedelta()
    return timedelta(**{unit: int(amount) for unit, amount in match.groupdict(default="0").items()})


def skip_reason(video: Video) -> str | None:
    if video.live_status != "none":
        return f"{video.live_status} stream"
    if video.duration <= SHORTS_MAX_LENGTH:
        return "short video"
    topic_text = f"{video.title}\n{video.description[:TOPIC_CHECK_CHARACTERS]}"
    if not video.channel.ai_focused and not AI_TOPICS.search(topic_text):
        return "not about AI"
    return None


def to_document(video: Video, fetched: FetchedTranscript | None) -> Document:
    transcript = to_transcript(fetched) if fetched else None
    return Document(
        source="youtube",
        source_id=video.video_id,
        url=video.url,
        title=video.title,
        published_at=video.published_at,
        content=" ".join(segment.text for segment in transcript.segments) if transcript else video.description,
        content_kind="transcript" if transcript else "summary",
        authors=[video.channel.name],
        transcript=transcript,
        metadata={
            "channel": video.channel.name,
            "channel_id": video.channel.channel_id,
            "duration_seconds": int(video.duration.total_seconds()),
            "views": video.views,
            "description": video.description,
        },
    )


def to_transcript(fetched: FetchedTranscript) -> Transcript | None:
    segments = [
        TranscriptSegment(start=snippet.start, duration=snippet.duration, text=" ".join(snippet.text.split()))
        for snippet in fetched.snippets
        if snippet.text.strip()
    ]
    if not segments:
        return None
    return Transcript(language_code=fetched.language_code, is_generated=fetched.is_generated, segments=segments)


if __name__ == "__main__":
    run_scraper(scrape)
