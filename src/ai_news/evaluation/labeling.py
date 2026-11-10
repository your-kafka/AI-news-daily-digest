import logging
from collections import defaultdict
from collections.abc import Sequence

from groq import BadRequestError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, ValidationError

from ai_news.config import groq_settings
from ai_news.evaluation.dataset import (
    JUDGMENTS_FILE,
    POOL_FILE,
    QUERIES_FILE,
    EvalQuery,
    Judgment,
    PooledDocument,
    read_jsonl,
    read_judgments,
    write_judgments,
)
from ai_news.log_config import configure_logging

PASSAGE_CHARACTERS = 1500
REVIEW_PASSAGE_CHARACTERS = 600
MAX_RETRIES = 8

RUBRIC = """Grade how relevant each document is to the search query, as a reader who typed that query would judge it.

3 = highly relevant: the document is mainly about what the query asks for. The reader would want it at the top.
2 = relevant: the document substantially covers the query, but only as one of several topics or only partly.
1 = marginal: the query's topic is mentioned in passing, or the document is only loosely related.
0 = not relevant: the document does not address the query.

You see each document's title and one passage from it. Judge the document, not just the passage: a title that
is clearly about the query can lift the grade. Time words in a query ("this month", "recent") refer to the
documents' publish dates. Grade every document, and give a short reason for each grade."""

GRADING_PROMPT = ChatPromptTemplate.from_messages(
    [("system", RUBRIC), ("human", "Query: {query}\n\nDocuments:\n\n{documents}")]
)

log = logging.getLogger(__name__)


class DocumentGrade(BaseModel):
    document: int = Field(description="The document's number in the list")
    reason: str = Field(description="One short sentence explaining the grade")
    grade: int = Field(ge=0, le=3, description="Relevance grade from 0 to 3")


class Grades(BaseModel):
    grades: list[DocumentGrade]


def build_grader(model: str, api_key: str) -> Runnable:
    llm = ChatGroq(model=model, api_key=api_key, temperature=0, max_retries=MAX_RETRIES, reasoning_effort="low")
    json_schema = llm.with_structured_output(Grades, method="json_schema")
    tool_calling = llm.with_structured_output(Grades, method="function_calling")
    return GRADING_PROMPT | json_schema.with_fallbacks([tool_calling], exceptions_to_handle=(BadRequestError,))


def format_documents(pooled: Sequence[PooledDocument]) -> str:
    return "\n\n".join(
        f"[{number}] {doc.source} · {doc.published_at[:10]} · {doc.title}\n{doc.passage[:PASSAGE_CHARACTERS]}"
        for number, doc in enumerate(pooled, start=1)
    )


def grade_query(grader: Runnable, query: EvalQuery, pooled: Sequence[PooledDocument], model: str) -> list[Judgment]:
    result: Grades = grader.invoke({"query": query.query, "documents": format_documents(pooled)})
    by_number = {grade.document: grade for grade in result.grades}
    return [
        Judgment(
            query_id=query.id,
            query=query.query,
            doc_key=doc.doc_key,
            title=doc.title,
            url=doc.url,
            passage=doc.passage[:REVIEW_PASSAGE_CHARACTERS],
            llm_grade=by_number[number].grade if number in by_number else None,
            llm_reason=f"{model}: {by_number[number].reason}" if number in by_number else "",
        )
        for number, doc in enumerate(pooled, start=1)
    ]


def keep_human_grades(fresh: Sequence[Judgment], existing: Sequence[Judgment]) -> list[Judgment]:
    human = {(judgment.query_id, judgment.doc_key): judgment.human_grade for judgment in existing}
    return [
        judgment.model_copy(update={"human_grade": human.get((judgment.query_id, judgment.doc_key))})
        for judgment in fresh
    ]


def in_query_order(judgments: dict[str, list[Judgment]], queries: Sequence[EvalQuery]) -> list[Judgment]:
    return [judgment for query in queries for judgment in judgments.get(query.id, [])]


def main() -> None:
    configure_logging()
    settings = groq_settings()
    queries = read_jsonl(QUERIES_FILE, EvalQuery)
    pool_by_query: dict[str, list[PooledDocument]] = defaultdict(list)
    for doc in read_jsonl(POOL_FILE, PooledDocument):
        pool_by_query[doc.query_id].append(doc)
    judgments: dict[str, list[Judgment]] = defaultdict(list)
    for judgment in read_judgments():
        judgments[judgment.query_id].append(judgment)
    grader = build_grader(settings.groq_model, settings.groq_api_key.get_secret_value())
    for query in queries:
        existing = judgments.get(query.id, [])
        done = {judgment.doc_key for judgment in existing if judgment.llm_grade is not None}
        if done >= {doc.doc_key for doc in pool_by_query[query.id]}:
            continue
        try:
            fresh = grade_query(grader, query, pool_by_query[query.id], settings.groq_model)
        except (ValidationError, ValueError, BadRequestError) as error:
            log.warning("%s: the model returned unusable grades, skipping for now (%s)", query.id, error)
            continue
        except Exception as error:
            write_judgments(in_query_order(judgments, queries))
            raise SystemExit(
                f"Stopped at {query.id}: {error}\nProgress is saved; run this again to continue."
            ) from error
        judgments[query.id] = keep_human_grades(fresh, existing)
        write_judgments(in_query_order(judgments, queries))
        missing = sum(1 for judgment in fresh if judgment.llm_grade is None)
        log.info("%s graded %d documents%s", query.id, len(fresh) - missing, f", {missing} missing" if missing else "")
    all_judgments = in_query_order(judgments, queries)
    graded = sum(1 for judgment in all_judgments if judgment.llm_grade is not None)
    print(f"\n{graded}/{len(all_judgments)} judgments have an LLM grade in {JUDGMENTS_FILE.name}")


if __name__ == "__main__":
    main()
