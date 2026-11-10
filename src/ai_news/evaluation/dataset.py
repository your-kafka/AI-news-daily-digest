import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field

from ai_news.config import PROJECT_ROOT

EVAL_DIR = PROJECT_ROOT / "eval"
QUERIES_FILE = EVAL_DIR / "queries.jsonl"
SNAPSHOT_FILE = EVAL_DIR / "snapshot.json"
POOL_FILE = EVAL_DIR / "pool.jsonl"
JUDGMENTS_FILE = EVAL_DIR / "judgments.csv"

QueryType = Literal["exact", "paraphrase", "time", "cross_source"]
Grade = Annotated[
    Annotated[int, Field(ge=0, le=3)] | None, BeforeValidator(lambda value: None if value in ("", None) else value)
]


class EvalQuery(BaseModel):
    id: str
    query: str
    type: QueryType


class PooledDocument(BaseModel):
    query_id: str
    doc_key: str
    title: str
    source: str
    url: str
    published_at: str
    passage: str
    found_by: dict[str, int]


class Judgment(BaseModel):
    query_id: str
    query: str
    doc_key: str
    title: str
    url: str
    passage: str
    llm_grade: Grade = None
    llm_reason: str = ""
    human_grade: Grade = None


def document_key(metadata: Mapping[str, Any]) -> str:
    return f"{metadata['source']}:{metadata['source_id']}"


def read_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [model.model_validate_json(line) for line in lines if line.strip()]


def write_jsonl(path: Path, items: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(item.model_dump_json() + "\n" for item in items), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_judgments(path: Path = JUDGMENTS_FILE) -> list[Judgment]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return [Judgment.model_validate(row) for row in csv.DictReader(file)]


def write_judgments(judgments: Iterable[Judgment], path: Path = JUDGMENTS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(Judgment.model_fields))
        writer.writeheader()
        writer.writerows(judgment.model_dump() for judgment in judgments)
