from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document

FIXTURES = Path(__file__).parent / "fixtures"


def make_chunk(chunk_id: str, document_id: int, text: str | None = None, **metadata: Any) -> Document:
    return Document(id=chunk_id, page_content=text or chunk_id, metadata={"document_id": document_id, **metadata})


def make_post(source_id: str, document_id: int | None = None, source: str = "openai", **metadata: Any) -> Document:
    return Document(
        id=f"{source_id}-0",
        page_content=f"text about {source_id}",
        metadata={
            "document_id": document_id if document_id is not None else abs(hash(source_id)) % 10_000,
            "source": source,
            "source_id": source_id,
            "title": f"Title {source_id}",
            "url": f"https://example.com/{source_id}",
            "published_at": "2026-11-01T00:00:00+00:00",
            **metadata,
        },
    )


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES
