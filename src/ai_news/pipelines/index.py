import logging
from dataclasses import dataclass

from ai_news.log_config import configure_logging
from ai_news.rag.chunker import DocumentChunker
from ai_news.rag.connector import EmbeddingStore
from ai_news.rag.embeddings import ChunkFile, EmbeddingReport, embed_chunk_file, export_hf_token
from ai_news.rag.loader import PostgresDocumentLoader

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexReport:
    documents: int
    chunks: int
    embedding: EmbeddingReport | None


def index_new_documents() -> IndexReport:
    """Chunks and embeds documents that are new or changed since they were last indexed."""
    documents = PostgresDocumentLoader(needs_indexing=True).load()
    if not documents:
        log.info("Nothing to index")
        return IndexReport(documents=0, chunks=0, embedding=None)
    export_hf_token()
    chunker = DocumentChunker()
    chunks = list(chunker.transform_documents(documents))
    chunk_file = ChunkFile(chunker.embedding_model, chunker.chunk_size, chunker.chunk_overlap, chunks)
    report = embed_chunk_file(chunk_file, EmbeddingStore())
    log.info(
        "Indexed %d documents: %d chunks, %d embedded, %d unchanged, %d removed",
        len(documents),
        len(chunks),
        report.stored.saved,
        report.up_to_date,
        report.stored.removed,
    )
    return IndexReport(documents=len(documents), chunks=len(chunks), embedding=report)


if __name__ == "__main__":
    configure_logging()
    index_new_documents()
