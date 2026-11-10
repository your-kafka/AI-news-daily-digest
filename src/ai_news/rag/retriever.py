import re
from datetime import datetime
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda
from sqlalchemy import Select, Text, bindparam, func, select

from ai_news.db.models import ChunkEmbeddingRecord
from ai_news.db.session import get_session_factory

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
EMBEDDING_KEY = "embedding"
MIN_EF_SEARCH = 40
LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
REASONING_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

EXPANSION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write search queries for a semantic search engine over recent AI news: blog posts, research "
            "papers and YouTube video transcripts. Given a reader's interests, write {count} different search "
            "queries that together cover every interest. Each query is one descriptive sentence that uses the "
            "concrete terms a matching post would contain, such as model names, techniques, companies and "
            "tools. Output one query per line, with no numbering and no other text.",
        ),
        ("human", "{interests}"),
    ]
)


class PgVectorRetriever(BaseRetriever):
    embeddings: Embeddings
    top_k: int = 100
    since: datetime | None = None
    sources: list[str] | None = None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        vector = self.embeddings.embed_query(QUERY_INSTRUCTION + query)
        distance = ChunkEmbeddingRecord.embedding.cosine_distance(vector).label("distance")
        statement = filtered(select(ChunkEmbeddingRecord, distance), self.since, self.sources)
        with get_session_factory()() as session:
            session.execute(select(func.set_config("hnsw.ef_search", str(max(self.top_k, MIN_EF_SEARCH)), True)))
            session.execute(select(func.set_config("hnsw.iterative_scan", "strict_order", True)))
            rows = session.execute(statement.order_by(distance).limit(self.top_k)).all()
        return [to_document(record, similarity=round(1 - distance, 4)) for record, distance in rows]


class Bm25Retriever(BaseRetriever):
    top_k: int = 100
    since: datetime | None = None
    sources: list[str] | None = None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        score = func.pdb.score(ChunkEmbeddingRecord.id).label("score")
        matches = ChunkEmbeddingRecord.content.op("|||")(bindparam("query", query, type_=Text))
        statement = filtered(select(ChunkEmbeddingRecord, score).where(matches), self.since, self.sources)
        with get_session_factory()() as session:
            rows = session.execute(statement.order_by(score.desc()).limit(self.top_k)).all()
        return [to_document(record, bm25_score=round(score, 4)) for record, score in rows]


def filtered(statement: Select, since: datetime | None, sources: list[str] | None) -> Select:
    if since is not None:
        statement = statement.where(ChunkEmbeddingRecord.published_at >= since)
    if sources:
        statement = statement.where(ChunkEmbeddingRecord.source.in_(sources))
    return statement


def to_document(record: ChunkEmbeddingRecord, **scores: Any) -> Document:
    metadata = {
        **record.metadata_,
        "document_id": record.document_id,
        "source": record.source,
        "section": record.section,
        "start_seconds": record.start_seconds,
        "published_at": record.published_at.isoformat(),
        EMBEDDING_KEY: record.embedding,
        **scores,
    }
    return Document(id=record.id, page_content=record.content, metadata=metadata)


def parse_queries(text: str) -> list[str]:
    lines = REASONING_BLOCK.sub("", text).splitlines()
    queries = (LIST_MARKER.sub("", line).strip() for line in lines)
    return list(dict.fromkeys(query for query in queries if query))


def build_expander(llm: BaseChatModel, query_count: int) -> Runnable:
    return (
        RunnableLambda(lambda interests: {"interests": interests})
        | EXPANSION_PROMPT.partial(count=query_count)
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda text: parse_queries(text)[:query_count])
    )
