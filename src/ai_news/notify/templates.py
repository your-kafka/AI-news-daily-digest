from collections.abc import Sequence
from datetime import date
from html import escape
from typing import Any

from ai_news.notify.mailer import Email

WRAPPER = """<!doctype html>
<html><body style="margin:0;padding:24px;background:#f6f7f9;font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#1d2433">
<div style="max-width:620px;margin:0 auto;background:#fff;border-radius:12px;padding:28px;border:1px solid #e4e7ec">
{body}
<p style="margin-top:32px;font-size:12px;color:#6b7280">{footer}</p>
</div></body></html>"""


def confirmation_email(to: str, interests: str, confirm_url: str) -> Email:
    text = (
        f"Someone (hopefully you) asked Lodestar for a daily AI news digest about:\n\n  {interests}\n\n"
        f"Confirm here to start receiving it:\n{confirm_url}\n\n"
        "If this wasn't you, ignore this email and nothing will be sent."
    )
    body = (
        '<h1 style="font-size:20px;margin:0 0 12px">Confirm your daily AI digest</h1>'
        "<p>Someone (hopefully you) asked for a daily digest about:</p>"
        f'<blockquote style="margin:12px 0;padding:10px 14px;background:#f1f5ff;border-radius:8px">{escape(interests)}</blockquote>'
        f'<p><a href="{escape(confirm_url)}" style="display:inline-block;background:#3b5bdb;color:#fff;'
        'padding:10px 18px;border-radius:8px;text-decoration:none;font-weight:600">Confirm subscription</a></p>'
    )
    footer = "If this wasn't you, ignore this email and nothing will be sent."
    return Email(to, "Confirm your Lodestar AI digest", text, WRAPPER.format(body=body, footer=footer))


def digest_email(
    to: str, interests: str, digest_date: date, items: Sequence[dict[str, Any]], unsubscribe_url: str
) -> Email:
    subject = f"Your AI digest for {digest_date:%-d %b %Y}"
    if not items:
        lines = ["Nothing matched your interests in the last 24 hours. We'll try again tomorrow."]
        html_items = f"<p>{lines[0]}</p>"
    else:
        lines, blocks = [], []
        for rank, item in enumerate(items, start=1):
            lines.append(f"{rank}. {item['headline']}\n   {item['why_it_matters']}")
            lines.extend(f"   - [{doc['source']}] {doc['title']}: {doc['url']}" for doc in item["documents"])
            links = "".join(
                f'<li style="margin:2px 0"><a href="{escape(doc["url"])}" style="color:#3b5bdb">{escape(doc["title"])}</a>'
                f' <span style="color:#6b7280;font-size:12px">· {escape(doc["source"])}</span></li>'
                for doc in item["documents"]
            )
            blocks.append(
                f'<div style="margin:0 0 20px"><h2 style="font-size:16px;margin:0 0 4px">{rank}. {escape(item["headline"])}</h2>'
                f'<p style="margin:0 0 6px;color:#374151">{escape(item["why_it_matters"])}</p>'
                f'<ul style="margin:0;padding-left:18px;font-size:14px">{links}</ul></div>'
            )
        html_items = "".join(blocks)
    text = (
        f"Your AI digest for {digest_date:%-d %b %Y}\nInterests: {interests}\n\n"
        + "\n\n".join(lines)
        + f"\n\nUnsubscribe: {unsubscribe_url}"
    )
    body = (
        f'<h1 style="font-size:20px;margin:0 0 4px">Your AI digest · {digest_date:%-d %b %Y}</h1>'
        f'<p style="margin:0 0 24px;color:#6b7280;font-size:14px">Interests: {escape(interests)}</p>{html_items}'
    )
    footer = f'You get this because you subscribed on Lodestar. <a href="{escape(unsubscribe_url)}">Unsubscribe</a>'
    return Email(to, subject, text, WRAPPER.format(body=body, footer=footer), unsubscribe_url)
