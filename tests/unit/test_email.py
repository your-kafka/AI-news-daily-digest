from datetime import date

import pytest
from pydantic import SecretStr

from ai_news.config import EmailSettings
from ai_news.notify import mailer as mailer_module
from ai_news.notify.mailer import Email, Mailer, build_message
from ai_news.notify.templates import confirmation_email, digest_email

SETTINGS = EmailSettings(
    _env_file=None,
    smtp_username="sender@example.com",
    smtp_password=SecretStr("app-password"),
    email_from="sender@example.com",
)

ITEMS = [
    {
        "headline": "Agents get <memory>",
        "why_it_matters": "Long tasks & fewer resets",
        "documents": [{"id": 1, "source": "anthropic", "title": "Memory for agents", "url": "https://a.com/m?x=1&y=2"}],
    }
]


class TestTemplates:
    def test_confirmation_email_links_and_escapes(self):
        email = confirmation_email("a@b.com", "agents <script>", "https://site/confirm?token=abc")
        assert email.subject == "Confirm your Lodestar AI digest"
        assert "https://site/confirm?token=abc" in email.text
        assert "agents &lt;script&gt;" in email.html
        assert "<script>" not in email.html

    def test_digest_email_lists_items_and_unsubscribe(self):
        email = digest_email("a@b.com", "agents", date(2026, 11, 23), ITEMS, "https://site/unsubscribe?token=t")
        assert email.subject == "Your AI digest for 23 Nov 2026"
        assert "1. Agents get <memory>" in email.text
        assert "[anthropic] Memory for agents: https://a.com/m?x=1&y=2" in email.text
        assert "Agents get &lt;memory&gt;" in email.html
        assert 'href="https://a.com/m?x=1&amp;y=2"' in email.html
        assert email.unsubscribe_url == "https://site/unsubscribe?token=t"
        assert "Unsubscribe: https://site/unsubscribe?token=t" in email.text

    def test_empty_digest_says_so(self):
        email = digest_email("a@b.com", "agents", date(2026, 11, 23), [], "https://site/u")
        assert "Nothing matched your interests" in email.text


def test_build_message_sets_headers_and_both_parts():
    message = build_message(Email("to@example.com", "Hi", "plain", "<p>html</p>", "https://site/u"), SETTINGS)
    assert message["From"] == "Lodestar <sender@example.com>"
    assert message["List-Unsubscribe"] == "<https://site/u>"
    assert message["Message-ID"].endswith("@example.com>")
    assert [part.get_content_type() for part in message.iter_parts()] == ["text/plain", "text/html"]


class FakeSMTP:
    instances: list["FakeSMTP"] = []

    def __init__(self, host, port, timeout):
        self.host, self.port, self.calls = host, port, []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def starttls(self, context):
        self.calls.append("starttls")

    def login(self, username, password):
        self.calls.append(("login", username, password))

    def send_message(self, message):
        self.calls.append(("send", message["To"], message["Subject"]))


def test_mailer_uses_starttls_and_logs_in(monkeypatch):
    FakeSMTP.instances.clear()
    monkeypatch.setattr(mailer_module.smtplib, "SMTP", FakeSMTP)
    Mailer(SETTINGS).send(Email("to@example.com", "Hello", "body"))
    [smtp] = FakeSMTP.instances
    assert (smtp.host, smtp.port) == ("smtp.gmail.com", 587)
    assert smtp.calls == [
        "starttls",
        ("login", "sender@example.com", "app-password"),
        ("send", "to@example.com", "Hello"),
    ]


def test_email_settings_need_credentials(monkeypatch):
    for name in ("SMTP_USERNAME", "SMTP_PASSWORD", "EMAIL_FROM"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError):
        EmailSettings(_env_file=None)
