from collections.abc import Callable, Iterable

from ai_news.scrapers import anthropic_news, deepmind_blog, huggingface, openai_news, youtube
from ai_news.scrapers.models import Document, SourceName

Scraper = Callable[[float], Iterable[Document]]

SCRAPERS: dict[SourceName, Scraper] = {
    "youtube": youtube.scrape,
    "anthropic": anthropic_news.scrape,
    "openai": openai_news.scrape,
    "deepmind": deepmind_blog.scrape,
    "huggingface": huggingface.scrape,
}
