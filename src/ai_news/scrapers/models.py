from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, Field

SourceName = Literal["youtube", "anthropic", "openai", "deepmind", "huggingface"]
ContentKind = Literal["transcript", "article", "summary", "abstract"]


class TranscriptSegment(BaseModel):
    start: float
    duration: float
    text: str


class Transcript(BaseModel):
    language_code: str
    is_generated: bool
    segments: list[TranscriptSegment]


class Document(BaseModel):
    source: SourceName
    source_id: str
    url: str
    title: str
    published_at: AwareDatetime
    content: str
    content_kind: ContentKind
    authors: list[str] = Field(default_factory=list)
    transcript: Transcript | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    fetched_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    def summary(self) -> str:
        published = self.published_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
        return f"{published}  {self.source:<11}  {self.word_count:>6,} words  {self.title}\n{'':22}{self.url}"
