"""Email subscriptions with double opt-in: nothing but one confirmation email is sent until the link is clicked."""

import secrets
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ai_news.db.models import Subscriber, SubscriberStatus

CONFIRMATION_COOLDOWN = timedelta(minutes=10)


class Outcome(StrEnum):
    SEND_CONFIRMATION = "send_confirmation"
    NOTHING_TO_DO = "nothing_to_do"  # already active, or a confirmation was sent moments ago
    FULL = "full"


@dataclass(frozen=True, slots=True)
class SubscribeResult:
    outcome: Outcome
    subscriber_id: int | None = None


def normalize_email(email: str) -> str:
    return email.strip().lower()


def new_token() -> str:
    return secrets.token_urlsafe(32)


def subscribe(session: Session, email: str, interests: str, now: datetime, max_subscribers: int) -> SubscribeResult:
    email = normalize_email(email)
    interests = " ".join(interests.split())
    subscriber = session.scalar(select(Subscriber).where(Subscriber.email == email))
    if subscriber is not None and subscriber.status == SubscriberStatus.ACTIVE:
        return SubscribeResult(Outcome.NOTHING_TO_DO, subscriber.id)
    if subscriber is None:
        if count_subscribed(session) >= max_subscribers:
            return SubscribeResult(Outcome.FULL)
        subscriber = Subscriber(email=email, interests=interests, token=new_token())
        session.add(subscriber)
    else:
        recently_sent = subscriber.status == SubscriberStatus.PENDING and (
            subscriber.confirmation_sent_at is not None
            and now - subscriber.confirmation_sent_at < CONFIRMATION_COOLDOWN
        )
        if recently_sent:
            return SubscribeResult(Outcome.NOTHING_TO_DO, subscriber.id)
        subscriber.interests = interests
        subscriber.status = SubscriberStatus.PENDING
        subscriber.token = new_token()
        subscriber.unsubscribed_at = None
    subscriber.confirmation_sent_at = now
    session.flush()
    return SubscribeResult(Outcome.SEND_CONFIRMATION, subscriber.id)


def count_subscribed(session: Session) -> int:
    statuses = (SubscriberStatus.PENDING, SubscriberStatus.ACTIVE)
    return session.scalar(select(func.count()).select_from(Subscriber).where(Subscriber.status.in_(statuses))) or 0


def by_token(session: Session, token: str) -> Subscriber | None:
    if not token or len(token) > 64:
        return None
    return session.scalar(select(Subscriber).where(Subscriber.token == token))


def confirm(session: Session, token: str, now: datetime) -> tuple[Subscriber | None, bool]:
    """Activates a pending subscriber. Returns the subscriber and whether this call activated them."""
    subscriber = by_token(session, token)
    if subscriber is None or subscriber.status != SubscriberStatus.PENDING:
        return subscriber, False
    subscriber.status = SubscriberStatus.ACTIVE
    subscriber.confirmed_at = now
    return subscriber, True


def unsubscribe(session: Session, token: str, now: datetime) -> Subscriber | None:
    subscriber = by_token(session, token)
    if subscriber is not None and subscriber.status != SubscriberStatus.UNSUBSCRIBED:
        subscriber.status = SubscriberStatus.UNSUBSCRIBED
        subscriber.unsubscribed_at = now
    return subscriber


def due_subscribers(session: Session, today: date) -> list[int]:
    statement = select(Subscriber.id).where(
        Subscriber.status == SubscriberStatus.ACTIVE,
        or_(Subscriber.last_digest_on.is_(None), Subscriber.last_digest_on < today),
    )
    return list(session.scalars(statement.order_by(Subscriber.id)))


def confirm_url(base_url: str, token: str) -> str:
    return f"{base_url.rstrip('/')}/subscriptions/confirm?token={token}"


def unsubscribe_url(base_url: str, token: str) -> str:
    return f"{base_url.rstrip('/')}/unsubscribe?token={token}"
