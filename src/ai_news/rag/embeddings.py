import json
import logging
import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import batched
from pathlib import Path
from typing import Self

from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy.exc import OperationalError, ProgrammingError

from ai_news.config import PROJECT_ROOT, rag_settings
from ai_news.db.models import EMBEDDING_DIMENSIONS
from ai_news.log_config import configure_logging
from ai_news.rag.chunker import CHUNKS_FILE
from ai_news.rag.connector import EmbeddedChunk, EmbeddingStore, StoreResult

BATCH_SIZE = 64

log = logging.getLogger(__name__)


class FastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name: str, batch_size: int = BATCH_SIZE, cache_dir: Path | None = None) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


@dataclass(frozen=True, slots=True)
class ChunkFile:
    embedding_model: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    chunks: list[Document]

    @classmethod
    def read(cls, path: Path = CHUNKS_FILE) -> Self:
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [
            Document(id=item["id"], page_content=item["text"], metadata=item["metadata"]) for item in payload["chunks"]
        ]
        return cls(payload["embedding_model"], payload["chunk_size_tokens"], payload["chunk_overlap_tokens"], chunks)


@dataclass(frozen=True, slots=True)
class EmbeddingReport:
    in_file: int
    missing_documents: int
    up_to_date: int
    stored: StoreResult
    table_rows: int


def embedding_text(chunk: Document) -> str:
    title = chunk.metadata["title"]
    section = chunk.metadata.get("section")
    heading = f"{title} — {section}" if section and section != title else title
    return f"{heading}\n\n{chunk.page_content}"


def fingerprint(chunk: Document, model_name: str) -> str:
    return sha256(f"{model_name}\n{embedding_text(chunk)}".encode()).hexdigest()


def embed_chunks(embeddings: FastEmbedEmbeddings, chunks: Sequence[Document]) -> list[EmbeddedChunk]:
    embedded: list[EmbeddedChunk] = []
    for batch in batched(chunks, embeddings.batch_size):
        vectors = embeddings.embed_documents([embedding_text(chunk) for chunk in batch])
        if len(vectors[0]) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"{embeddings.model_name} returns {len(vectors[0])}-dimension vectors, "
                f"but chunk_embeddings.embedding holds {EMBEDDING_DIMENSIONS}"
            )
        embedded.extend(
            EmbeddedChunk(chunk, vector, fingerprint(chunk, embeddings.model_name), embeddings.model_name)
            for chunk, vector in zip(batch, vectors, strict=True)
        )
        log.info("Embedded %d/%d chunks", len(embedded), len(chunks))
    return embedded


def chunk_ids_by_document(chunks: Sequence[Document]) -> dict[int, set[str]]:
    grouped: dict[int, set[str]] = defaultdict(set)
    for chunk in chunks:
        grouped[chunk.metadata["document_id"]].add(chunk.id)
    return grouped


def export_hf_token() -> None:
    token = rag_settings().hf_token
    if token is not None:
        os.environ.setdefault("HF_TOKEN", token.get_secret_value())


def embed_chunk_file(chunk_file: ChunkFile, store: EmbeddingStore) -> EmbeddingReport:
    settings = rag_settings()
    model_name = settings.embedding_model
    known_documents = store.existing_documents({chunk.metadata["document_id"] for chunk in chunk_file.chunks})
    chunks = [chunk for chunk in chunk_file.chunks if chunk.metadata["document_id"] in known_documents]
    stored_hashes = store.stored_hashes(chunk.id for chunk in chunks)
    pending = [chunk for chunk in chunks if stored_hashes.get(chunk.id) != fingerprint(chunk, model_name)]
    embeddings = FastEmbedEmbeddings(model_name, cache_dir=settings.model_cache_dir) if pending else None
    embedded = embed_chunks(embeddings, pending) if embeddings else []
    stored = store.save(embedded, chunk_ids_by_document(chunks))
    return EmbeddingReport(
        in_file=len(chunk_file.chunks),
        missing_documents=len(chunk_file.chunks) - len(chunks),
        up_to_date=len(chunks) - len(pending),
        stored=stored,
        table_rows=store.count(),
    )


def print_report(chunk_file: ChunkFile, report: EmbeddingReport) -> None:
    path = CHUNKS_FILE.relative_to(PROJECT_ROOT)
    settings = f"size {chunk_file.chunk_size_tokens}, overlap {chunk_file.chunk_overlap_tokens}"
    print(f"\n{path}: {report.in_file} chunks ({settings}, {chunk_file.embedding_model})")
    print(f"  embedded and saved         {report.stored.saved:>6}")
    print(f"  already up to date         {report.up_to_date:>6}")
    print(f"  removed (no longer exist)  {report.stored.removed:>6}")
    print(f"  skipped (document missing) {report.missing_documents:>6}")
    print(f"\nchunk_embeddings now holds {report.table_rows} rows")


def main() -> None:
    configure_logging()
    if not CHUNKS_FILE.exists():
        raise SystemExit("data/chunks.json not found. Run `python src/ai_news/rag/chunker.py` first.")
    chunk_file = ChunkFile.read()
    model_name = rag_settings().embedding_model
    if chunk_file.embedding_model != model_name:
        raise SystemExit(
            f"data/chunks.json was chunked for {chunk_file.embedding_model}, but EMBEDDING_MODEL is {model_name}. "
            "Re-run the chunker first."
        )
    export_hf_token()
    try:
        report = embed_chunk_file(chunk_file, EmbeddingStore())
    except OperationalError as error:
        raise SystemExit(f"Cannot reach PostgreSQL. Start it with `docker compose up -d`.\n{error.orig}") from error
    except ProgrammingError as error:
        raise SystemExit(f"Database schema is out of date. Run `alembic upgrade head`.\n{error.orig}") from error
    print_report(chunk_file, report)


if __name__ == "__main__":
    main()
