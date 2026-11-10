import json
import re
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from tokenizers import Tokenizer

from ai_news.config import PROJECT_ROOT, rag_settings
from ai_news.log_config import configure_logging
from ai_news.rag.loader import SEGMENTS_KEY, PostgresDocumentLoader

CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks.json"
PLAIN_TEXT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*#*$", re.MULTILINE)
FENCED_CODE_PATTERN = re.compile(r"^(`{3,}|~{3,}).*?^\1", re.MULTILINE | re.DOTALL)


@dataclass(frozen=True, slots=True)
class Timeline:
    offsets: list[int]
    starts: list[float]
    ends: list[float]

    @classmethod
    def from_segments(cls, segments: Sequence[dict[str, Any]]) -> Self:
        offsets = []
        position = 0
        for segment in segments:
            offsets.append(position)
            position += len(segment["text"]) + 1
        starts = [segment["start"] for segment in segments]
        ends = [segment["start"] + segment["duration"] for segment in segments]
        return cls(offsets, starts, ends)

    def span(self, start: int, end: int) -> tuple[float, float]:
        first = max(bisect_right(self.offsets, start) - 1, 0)
        last = max(bisect_right(self.offsets, end) - 1, first)
        return self.starts[first], self.ends[last]


@dataclass(frozen=True, slots=True)
class Outline:
    offsets: list[int]
    headings: list[str]

    @classmethod
    def from_markdown(cls, text: str) -> Self:
        code_spans = [match.span() for match in FENCED_CODE_PATTERN.finditer(text)]
        matches = [
            match
            for match in HEADING_PATTERN.finditer(text)
            if not any(start <= match.start() < end for start, end in code_spans)
        ]
        return cls([match.start() for match in matches], [match.group(1) for match in matches])

    def section_at(self, offset: int) -> str | None:
        index = bisect_right(self.offsets, offset) - 1
        return self.headings[index] if index >= 0 else None


class DocumentChunker(BaseDocumentTransformer):
    def __init__(
        self,
        embedding_model: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        settings = rag_settings()
        self.embedding_model = embedding_model or settings.embedding_model
        self.chunk_size = settings.chunk_size_tokens if chunk_size is None else chunk_size
        self.chunk_overlap = settings.chunk_overlap_tokens if chunk_overlap is None else chunk_overlap
        token = settings.hf_token.get_secret_value() if settings.hf_token else None
        self.tokenizer = Tokenizer.from_pretrained(self.embedding_model, token=token)
        options: dict[str, Any] = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "length_function": self.count_tokens,
        }
        self.markdown_splitter = RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN, **options)
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=PLAIN_TEXT_SEPARATORS,
            keep_separator="end",
            **options,
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        return [chunk for document in documents for chunk in self.split(document)]

    def split(self, document: Document) -> list[Document]:
        metadata = {key: value for key, value in document.metadata.items() if key != SEGMENTS_KEY}
        is_article = metadata.get("content_kind") == "article"
        splitter = self.markdown_splitter if is_article else self.text_splitter
        texts = splitter.split_text(document.page_content)
        starts = locate(document.page_content, texts)
        outline = Outline.from_markdown(document.page_content) if is_article else None
        segments = document.metadata.get(SEGMENTS_KEY)
        timeline = Timeline.from_segments(segments) if segments else None
        return [
            self.to_chunk(text, start, index, len(texts), metadata, outline, timeline)
            for index, (text, start) in enumerate(zip(texts, starts, strict=True))
        ]

    def to_chunk(
        self,
        text: str,
        start: int,
        index: int,
        count: int,
        document_metadata: dict[str, Any],
        outline: Outline | None,
        timeline: Timeline | None,
    ) -> Document:
        metadata = {
            **document_metadata,
            "chunk_index": index,
            "chunk_count": count,
            "start_index": start,
            "token_count": self.count_tokens(text),
        }
        if outline is not None:
            metadata["section"] = outline.section_at(start)
        if timeline is not None:
            begin, end = timeline.span(start, start + len(text))
            metadata["start_seconds"] = round(begin, 2)
            metadata["end_seconds"] = round(end, 2)
            metadata["timestamp_url"] = f"{metadata['url']}&t={int(begin)}s"
        return Document(id=f"{metadata['document_id']}-{index}", page_content=text, metadata=metadata)


def locate(text: str, pieces: Sequence[str]) -> list[int]:
    starts = []
    cursor = 0
    for piece in pieces:
        start = text.find(piece, cursor)
        starts.append(start)
        cursor = max(start, 0) + 1
    return starts


def print_summary(documents: Sequence[Document], chunks: Sequence[Document]) -> None:
    documents_per_source: dict[str, int] = defaultdict(int)
    tokens_per_source: dict[str, list[int]] = defaultdict(list)
    for document in documents:
        documents_per_source[document.metadata["source"]] += 1
    for chunk in chunks:
        tokens_per_source[chunk.metadata["source"]].append(chunk.metadata["token_count"])
    print(f"\n{'source':<12} {'documents':>9} {'chunks':>7} {'avg tokens':>11} {'max tokens':>11}")
    for source, count in sorted(documents_per_source.items()):
        tokens = tokens_per_source[source]
        average = sum(tokens) / len(tokens) if tokens else 0
        print(f"{source:<12} {count:>9} {len(tokens):>7} {average:>11.0f} {max(tokens, default=0):>11}")
    print(f"{'total':<12} {len(documents):>9} {len(chunks):>7}")


def save_chunks(chunker: DocumentChunker, chunks: Sequence[Document], path: Path = CHUNKS_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "embedding_model": chunker.embedding_model,
        "chunk_size_tokens": chunker.chunk_size,
        "chunk_overlap_tokens": chunker.chunk_overlap,
        "chunk_count": len(chunks),
        "chunks": [{"id": chunk.id, "text": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    configure_logging()
    documents = PostgresDocumentLoader().load()
    chunker = DocumentChunker()
    chunks = chunker.transform_documents(documents)
    print(
        f"chunk size {chunker.chunk_size} tokens, overlap {chunker.chunk_overlap} tokens, "
        f"tokenizer {chunker.embedding_model}"
    )
    print_summary(documents, chunks)
    path = save_chunks(chunker, chunks)
    print(f"\nSaved {len(chunks)} chunks to {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
