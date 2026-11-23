from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import select

from ai_news.api import app as app_module
from ai_news.api.app import create_app
from ai_news.config import EmailSettings
from ai_news.db.models import DigestRecord, Subscriber
from ai_news.db.session import get_session_factory
from ai_news.pipelines import subscriber_digest
from ai_news.subscriptions import CONFIRMATION_COOLDOWN, Outcome, subscribe
from tests.integration.test_digest_and_index import digest_values

pytestmark = pytest.mark.integration

SETTINGS = EmailSettings(
    _env_file=None,
    smtp_username="sender@example.com",
    smtp_password=SecretStr("x"),
    email_from="sender@example.com",
    public_base_url="https://lodestar.test",
    max_subscribers=3,
)
NOW = datetime(2026, 11, 23, 2, 0, tzinfo=UTC)


class FakeJobs:
    def __init__(self) -> None:
        self.confirmations: list[int] = []
        self.first_digests: list[int] = []

    def send_confirmation(self, subscriber_id):
        self.confirmations.append(subscriber_id)

    def send_first_digest(self, subscriber_id):
        self.first_digests.append(subscriber_id)


@pytest.fixture
def jobs():
    return FakeJobs()


@pytest.fixture
def client(db, jobs):
    application = create_app(lambda: SimpleNamespace(settings=None))
    application.dependency_overrides[app_module.get_jobs] = lambda: jobs
    application.dependency_overrides[app_module.get_email_settings] = lambda: SETTINGS
    with TestClient(application) as test_client:
        yield test_client


def subscriber(email: str = "reader@example.com") -> Subscriber:
    with get_session_factory()() as session:
        return session.scalar(select(Subscriber).where(Subscriber.email == email))


class TestSubscribeFlow:
    def test_subscribe_confirm_unsubscribe(self, client, jobs):
        response = client.post("/subscriptions", json={"email": " Reader@Example.com ", "interests": "coding  agents"})
        assert response.status_code == 202
        assert response.json()["status"] == "check_your_inbox"
        pending = subscriber()
        assert (pending.status, pending.interests) == ("pending", "coding agents")
        assert jobs.confirmations == [pending.id]

        page = client.get("/subscriptions/confirm", params={"token": pending.token})
        assert "Your first digest is being written" in page.text
        assert subscriber().status == "active"
        assert jobs.first_digests == [pending.id]
        assert "Already confirmed" in client.get("/subscriptions/confirm", params={"token": pending.token}).text
        assert jobs.first_digests == [pending.id]

        assert "Unsubscribed" in client.get("/unsubscribe", params={"token": pending.token}).text
        assert subscriber().status == "unsubscribed"

    def test_answers_the_same_for_existing_subscribers(self, client, jobs):
        body = {"email": "reader@example.com", "interests": "agents"}
        first = client.post("/subscriptions", json=body)
        again = client.post("/subscriptions", json=body)
        assert first.json() == again.json()
        assert len(jobs.confirmations) == 1  # inside the cooldown: no second email

    def test_bad_links_and_input(self, client):
        assert client.get("/subscriptions/confirm", params={"token": "nope"}).status_code == 404
        assert client.get("/unsubscribe", params={"token": "x" * 100}).status_code == 404
        assert client.post("/subscriptions", json={"email": "not-an-email", "interests": "agents"}).status_code == 422
        assert client.post("/subscriptions", json={"email": "a@example.com", "interests": "x"}).status_code == 422

    def test_stops_taking_subscribers_when_full(self, client):
        for n in range(3):
            assert (
                client.post("/subscriptions", json={"email": f"r{n}@example.com", "interests": "ai news"}).status_code
                == 202
            )
        assert (
            client.post("/subscriptions", json={"email": "r9@example.com", "interests": "ai news"}).status_code == 409
        )

    def test_dashboard_is_served(self, client):
        page = client.get("/")
        assert page.status_code == 200
        assert "Get your daily digest" in page.text


class TestSubscribeRules:
    def test_resubscribing_after_unsubscribe_needs_a_new_confirmation(self, db):
        with get_session_factory().begin() as session:
            first = subscribe(session, "a@example.com", "agents", NOW, 10)
            row = session.get(Subscriber, first.subscriber_id)
            old_token = row.token
            row.status = "unsubscribed"
        with get_session_factory().begin() as session:
            again = subscribe(session, "a@example.com", "robots", NOW + timedelta(days=1), 10)
            row = session.get(Subscriber, again.subscriber_id)
            assert (again.outcome, row.status, row.interests) == (Outcome.SEND_CONFIRMATION, "pending", "robots")
            assert row.token != old_token

    def test_pending_can_ask_again_after_the_cooldown(self, db):
        with get_session_factory().begin() as session:
            subscribe(session, "a@example.com", "agents", NOW, 10)
            later = subscribe(session, "a@example.com", "agents", NOW + CONFIRMATION_COOLDOWN, 10)
        assert later.outcome == Outcome.SEND_CONFIRMATION


class FakeMailer:
    def __init__(self, fail: bool = False) -> None:
        self.sent = []
        self.fail = fail

    def send(self, email):
        if self.fail:
            raise OSError("smtp down")
        self.sent.append(email)


class CountingBuilder:
    def __init__(self) -> None:
        self.builds = 0

    def build(self, profile, now, window_hours, digest_date):
        self.builds += 1
        return digest_values(profile=profile.name, day=digest_date)


def active_subscriber() -> int:
    with get_session_factory().begin() as session:
        result = subscribe(session, "reader@example.com", "agents", NOW, 10)
        session.get(Subscriber, result.subscriber_id).status = "active"
        return result.subscriber_id


class TestSubscriberDigest:
    @pytest.fixture(autouse=True)
    def settings(self, monkeypatch):
        monkeypatch.setattr(subscriber_digest, "email_settings", lambda: SETTINGS)

    def test_sends_once_per_day(self, db):
        subscriber_id = active_subscriber()
        builder, mailer = CountingBuilder(), FakeMailer()
        result = subscriber_digest.send_subscriber_digest(subscriber_id, lambda: builder, mailer, NOW)
        assert result == "sent 1 items"
        [email] = mailer.sent
        assert email.to == "reader@example.com"
        assert email.unsubscribe_url.startswith("https://lodestar.test/unsubscribe?token=")
        assert subscriber().last_digest_on == date(2026, 11, 23)
        again = subscriber_digest.send_subscriber_digest(subscriber_id, lambda: builder, mailer, NOW)
        assert again == "skipped: already sent today"
        assert (builder.builds, len(mailer.sent)) == (1, 1)

    def test_a_retry_after_a_failed_send_reuses_the_saved_digest(self, db):
        subscriber_id = active_subscriber()
        builder = CountingBuilder()
        with pytest.raises(OSError):
            subscriber_digest.send_subscriber_digest(subscriber_id, lambda: builder, FakeMailer(fail=True), NOW)
        assert subscriber().last_digest_on is None
        mailer = FakeMailer()
        subscriber_digest.send_subscriber_digest(subscriber_id, lambda: builder, mailer, NOW)
        assert (builder.builds, len(mailer.sent)) == (1, 1)
        with get_session_factory()() as session:
            profiles = session.scalars(select(DigestRecord.profile)).all()
        assert profiles == [f"subscriber-{subscriber_id}"]

    def test_skips_inactive_subscribers(self, db):
        with get_session_factory().begin() as session:
            pending_id = subscribe(session, "p@example.com", "agents", NOW, 10).subscriber_id
        assert subscriber_digest.send_subscriber_digest(pending_id, CountingBuilder, FakeMailer(), NOW).startswith(
            "skipped"
        )

    def test_confirmation_email_goes_to_pending_only(self, db):
        with get_session_factory().begin() as session:
            pending_id = subscribe(session, "p@example.com", "agents", NOW, 10).subscriber_id
        mailer = FakeMailer()
        assert subscriber_digest.send_confirmation(pending_id, mailer) == "sent"
        assert "https://lodestar.test/subscriptions/confirm?token=" in mailer.sent[0].text
        assert subscriber_digest.send_confirmation(active_subscriber(), mailer) == "skipped"
