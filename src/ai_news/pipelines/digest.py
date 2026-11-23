"""Builds one curated digest per profile: expand interests into queries, search the last day, let an LLM curate.

Every run records the real token usage of all its LLM calls next to the naive cost: the tokens a single
curation prompt would need if it contained every document from the same window in full.

    uv run python src/ai_news/pipelines/digest.py frontier-models
"""

import argparse
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import tiktoken
import yaml
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ai_news.config import worker_settings
from ai_news.db.models import DigestRecord, DocumentRecord
from ai_news.db.session import get_session_factory
from ai_news.log_config import configure_logging
from ai_news.rag.curation import CURATION_PROMPT, MAX_DIGEST_ITEMS, CurationResult, Curator
from ai_news.rag.pipeline import SearchPipeline, chat_model
from ai_news.rag.retriever import build_expander

TokenCounter = Callable[[str], int]

log = logging.getLogger(__name__)


class Profile(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9-]+$", max_length=64)
    title: str
    interests: str = Field(min_length=3)


class ProfileFile(BaseModel):
    profiles: list[Profile]


def load_profiles(path: Path) -> dict[str, Profile]:
    profiles = ProfileFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))).profiles
    by_name = {profile.name: profile for profile in profiles}
    if len(by_name) != len(profiles):
        raise ValueError(f"{path} has duplicate profile names")
    return by_name


@cache
def o200k_tokens() -> TokenCounter:
    """gpt-oss and recent OpenAI models use the o200k vocabulary; its chat-format special tokens are not counted."""
    encoding = tiktoken.get_encoding("o200k_base")
    return lambda text: len(encoding.encode(text, disallowed_special=()))


def naive_prompt(interests: str, documents: Sequence[DocumentRecord]) -> str:
    candidates = "\n\n".join(
        f"[{document.id}] {document.source} · {document.published_at:%Y-%m-%d} · {document.title}\n{document.content}"
        for document in documents
    )
    return CURATION_PROMPT.format(max_items=MAX_DIGEST_ITEMS, interests=interests, candidates=candidates)


def total_tokens(usage: dict[str, dict[str, Any]]) -> tuple[int, int]:
    return (
        sum(model_usage.get("input_tokens", 0) for model_usage in usage.values()),
        sum(model_usage.get("output_tokens", 0) for model_usage in usage.values()),
    )


def digest_items(result: CurationResult, posts: Sequence[Document]) -> list[dict[str, Any]]:
    by_id = {post.metadata["document_id"]: post.metadata for post in posts}
    return [
        {
            "headline": item.headline,
            "why_it_matters": item.why_it_matters,
            "documents": [
                {
                    "id": document_id,
                    "source": by_id[document_id]["source"],
                    "title": by_id[document_id]["title"],
                    "url": by_id[document_id].get("timestamp_url") or by_id[document_id]["url"],
                }
                for document_id in item.document_ids
            ],
        }
        for item in result.digest.items
    ]


@dataclass(frozen=True, slots=True)
class DigestBuilder:
    pipeline: SearchPipeline
    expansion_llm: BaseChatModel
    curation_llm: BaseChatModel
    model_name: str
    count_tokens: TokenCounter

    def build(self, profile: Profile, now: datetime, window_hours: float, digest_date: date) -> dict[str, Any]:
        since = now - timedelta(hours=window_hours)
        started = time.perf_counter()
        settings = self.pipeline.settings
        with get_usage_metadata_callback() as usage:
            expander = build_expander(self.expansion_llm, settings.retrieval_query_count)
            queries = expander.invoke(profile.interests) or [profile.interests]
            posts = self.pipeline.search(queries, limit=settings.curation_candidates, since=since)
            result = Curator(self.curation_llm, self.model_name).curate(profile.interests, posts)
        input_tokens, output_tokens = total_tokens(usage.usage_metadata)
        if input_tokens == 0:
            log.warning(
                "No token usage was reported for %s; the token accounting for this digest is missing", profile.name
            )
        window_documents = documents_between(since, now)
        return {
            "profile": profile.name,
            "digest_date": digest_date,
            "window_start": since,
            "window_end": now,
            "model": self.model_name,
            "queries": list(queries),
            "items": digest_items(result, posts),
            "candidate_ids": [post.metadata["document_id"] for post in posts],
            "documents_in_window": len(window_documents),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "naive_input_tokens": self.count_tokens(naive_prompt(profile.interests, window_documents)),
            "duration_ms": round((time.perf_counter() - started) * 1000),
        }


def documents_between(start: datetime, end: datetime) -> list[DocumentRecord]:
    statement = (
        select(DocumentRecord)
        .where(DocumentRecord.published_at >= start, DocumentRecord.published_at <= end)
        .order_by(DocumentRecord.published_at)
    )
    with get_session_factory()() as session:
        return list(session.scalars(statement))


def save_digest(session: Session, values: dict[str, Any]) -> int:
    """Inserts the digest, or replaces the one already built for this profile and day."""
    statement = insert(DigestRecord).values(values)
    updatable = {column: statement.excluded[column] for column in values if column not in ("profile", "digest_date")}
    statement = statement.on_conflict_do_update(
        index_elements=[DigestRecord.profile, DigestRecord.digest_date], set_=updatable
    ).returning(DigestRecord.id)
    return session.execute(statement).scalar_one()


def local_today(now: datetime, timezone: str) -> date:
    return now.astimezone(ZoneInfo(timezone)).date()


def default_builder(pipeline: SearchPipeline) -> DigestBuilder:
    from ai_news.config import groq_settings

    groq = groq_settings()
    return DigestBuilder(
        pipeline=pipeline,
        expansion_llm=chat_model(groq, temperature=0.2),
        curation_llm=chat_model(groq, temperature=0),
        model_name=groq.groq_model,
        count_tokens=o200k_tokens(),
    )


def run_digest(profile_name: str, builder: DigestBuilder, now: datetime | None = None) -> dict[str, Any]:
    settings = worker_settings()
    profile = load_profiles(settings.profiles_file)[profile_name]
    now = now or datetime.now(UTC)
    values = builder.build(profile, now, settings.digest_window_hours, local_today(now, settings.schedule_timezone))
    with get_session_factory().begin() as session:
        values["id"] = save_digest(session, values)
    log.info(
        "Digest %s for %s: %d items from %d candidates (%d documents in window); %d input tokens vs %d naive; %d ms",
        values["digest_date"],
        profile_name,
        len(values["items"]),
        len(values["candidate_ids"]),
        values["documents_in_window"],
        values["input_tokens"],
        values["naive_input_tokens"],
        values["duration_ms"],
    )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profile", help="a profile name from config/profiles.yaml")
    args = parser.parse_args()
    configure_logging()
    values = run_digest(args.profile, default_builder(SearchPipeline.from_settings()))
    for rank, item in enumerate(values["items"], start=1):
        print(f"\n{rank}. {item['headline']}\n   {item['why_it_matters']}")
        for document in item["documents"]:
            print(f"   - [{document['source']}] {document['title']}: {document['url']}")
    saved = 1 - values["input_tokens"] / values["naive_input_tokens"] if values["naive_input_tokens"] else 0
    print(
        f"\n{values['input_tokens']:,} input tokens (all LLM calls) vs {values['naive_input_tokens']:,} naive "
        f"({values['documents_in_window']} documents in the window): {saved:.0%} fewer"
    )


if __name__ == "__main__":
    main()
