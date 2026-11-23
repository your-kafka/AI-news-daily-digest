"""Sends email through an SMTP server (Gmail by default) using the account in .env.

Check the setup by sending yourself a test message:

    uv run python src/ai_news/notify/mailer.py you@example.com
"""

import argparse
import logging
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formataddr, make_msgid

import certifi

from ai_news.config import EmailSettings, email_settings
from ai_news.log_config import configure_logging

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Email:
    to: str
    subject: str
    text: str
    html: str | None = None
    unsubscribe_url: str | None = None


def build_message(email: Email, settings: EmailSettings) -> EmailMessage:
    message = EmailMessage()
    message["From"] = formataddr((settings.email_from_name, settings.email_from))
    message["To"] = email.to
    message["Subject"] = email.subject
    message["Message-ID"] = make_msgid(domain=settings.email_from.rsplit("@", 1)[-1])
    if email.unsubscribe_url:
        # Lets Gmail and other clients show their own "Unsubscribe" button.
        message["List-Unsubscribe"] = f"<{email.unsubscribe_url}>"
    message.set_content(email.text)
    if email.html:
        message.add_alternative(email.html, subtype="html")
    return message


class Mailer:
    def __init__(self, settings: EmailSettings | None = None) -> None:
        self.settings = settings or email_settings()

    def send(self, email: Email) -> None:
        settings = self.settings
        message = build_message(email, settings)
        # certifi's CA bundle: python.org builds on macOS ship without system certificates.
        context = ssl.create_default_context(cafile=certifi.where())
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as smtp:
            smtp.starttls(context=context)
            smtp.login(settings.smtp_username, settings.smtp_password.get_secret_value())
            smtp.send_message(message)
        log.info("Sent %r to %s", email.subject, email.to)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("to", help="address to send the test message to")
    args = parser.parse_args()
    configure_logging()
    try:
        Mailer().send(
            Email(
                to=args.to,
                subject="Lodestar test email",
                text="If you can read this, Lodestar can send email from this account.",
            )
        )
    except smtplib.SMTPAuthenticationError as error:
        raise SystemExit(
            "Gmail rejected the login. SMTP_PASSWORD must be an App Password "
            "(myaccount.google.com/apppasswords), not your normal password.\n"
            f"{error.smtp_error.decode(errors='replace')}"
        ) from error
    print(f"Sent a test email to {args.to}. Check the inbox (and the spam folder).")


if __name__ == "__main__":
    main()
