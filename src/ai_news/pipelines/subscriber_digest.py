"""Sends confirmation emails and personal daily digests to subscribers."""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from ai_news.config import email_settings, worker_settings
from ai_news.db.models import DigestRecord, Subscriber, SubscriberStatus
from ai_news.db.session import get_session_factory
from ai_news.notify.mailer import Mailer
from ai_news.notify.templates import confirmation_email, digest_email
from ai_news.pipelines.digest import DigestBuilder, Profile, local_today, save_digest
from ai_news.subscriptions import confirm_url, unsubscribe_url

log = logging.getLogger(__name__)


def profile_name(subscriber_id: int) -> str:
    return f"subscriber-{subscriber_id}"


def send_confirmation(subscriber_id: int, mailer: Mailer) -> str:
    with get_session_factory()() as session:
        subscriber = session.get(Subscriber, subscriber_id)
        if subscriber is None or subscriber.status != SubscriberStatus.PENDING:
            return "skipped"
        email = confirmation_email(
            subscriber.email,
            subscriber.interests,
            confirm_url(email_settings().public_base_url, subscriber.token),
        )
    mailer.send(email)
    return "sent"


def send_subscriber_digest(
    subscriber_id: int,
    builder: Callable[[], DigestBuilder],
    mailer: Mailer,
    now: datetime | None = None,
) -> str:
    """Builds (or reuses) today's digest for one subscriber and emails it, at most once per day."""
    now = now or datetime.now(UTC)
    settings = worker_settings()
    today = local_today(now, settings.schedule_timezone)
    name = profile_name(subscriber_id)
    with get_session_factory()() as session:
        subscriber = session.get(Subscriber, subscriber_id)
        if subscriber is None or subscriber.status != SubscriberStatus.ACTIVE:
            return "skipped: not active"
        if subscriber.last_digest_on == today:
            return "skipped: already sent today"
        email_address, interests, token = subscriber.email, subscriber.interests, subscriber.token
        saved = session.scalar(
            select(DigestRecord).where(DigestRecord.profile == name, DigestRecord.digest_date == today)
        )
        items: list[dict[str, Any]] | None = saved.items if saved else None

    if items is None:  # a retry after a failed send reuses the saved digest instead of paying for it again
        profile = Profile(name=name, title="Personal digest", interests=interests)
        values = builder().build(profile, now, settings.digest_window_hours, today)
        with get_session_factory().begin() as session:
            save_digest(session, values)
        items = values["items"]

    base_url = email_settings().public_base_url
    mailer.send(digest_email(email_address, interests, today, items, unsubscribe_url(base_url, token)))
    with get_session_factory().begin() as session:
        session.get(Subscriber, subscriber_id).last_digest_on = today
    log.info("Sent digest %s with %d items to subscriber %d", today, len(items), subscriber_id)
    return f"sent {len(items)} items"
