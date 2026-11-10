from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

MAX_DIGEST_ITEMS = 10

CURATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You curate a short digest of recent AI news for one reader. You get the reader's interests and "
            "numbered candidate posts, each with its document id, source, date, title and best matching passage. "
            "Pick at most {max_items} stories that best match the interests, most important first, and skip "
            "candidates that are off-topic or minor. When several candidates cover the same story, merge them "
            "into one item that lists all their document ids. For each item write a short headline and one "
            "sentence on why it matters to this reader. Use only facts from the candidates.",
        ),
        ("human", "Reader's interests: {interests}\n\nCandidates:\n\n{candidates}"),
    ]
)


class DigestItem(BaseModel):
    headline: str = Field(description="Short headline for the story")
    why_it_matters: str = Field(description="One sentence on why this story matters to the reader")
    document_ids: list[int] = Field(description="Document ids of every candidate that covers this story")


class Digest(BaseModel):
    items: list[DigestItem] = Field(description="The chosen stories, most important first")


@dataclass(frozen=True, slots=True)
class CurationResult:
    digest: Digest
    model: str
    input_tokens: int
    output_tokens: int


class CurationError(Exception):
    pass


class Curator:
    def __init__(self, llm: BaseChatModel, model_name: str, max_items: int = MAX_DIGEST_ITEMS) -> None:
        self.model_name = model_name
        self.max_items = max_items
        self.chain = CURATION_PROMPT.partial(max_items=str(max_items)) | llm.with_structured_output(
            Digest, include_raw=True
        )

    def curate(self, interests: str, posts: Sequence[Document]) -> CurationResult:
        if not posts:
            return CurationResult(Digest(items=[]), self.model_name, 0, 0)
        output = self.chain.invoke({"interests": interests, "candidates": format_candidates(posts)})
        if output["parsed"] is None:
            raise CurationError(f"{self.model_name} did not return a valid digest: {output['parsing_error']}")
        usage = output["raw"].usage_metadata or {}
        digest = keep_known_documents(output["parsed"], {post.metadata["document_id"] for post in posts})
        return CurationResult(
            Digest(items=digest.items[: self.max_items]),
            self.model_name,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )


def format_candidates(posts: Sequence[Document]) -> str:
    return "\n\n".join(
        f"[{post.metadata['document_id']}] {post.metadata['source']} · {post.metadata['published_at'][:10]} · "
        f"{post.metadata['title']}\n{' '.join(post.page_content.split())}"
        for post in posts
    )


def keep_known_documents(digest: Digest, known_ids: set[int]) -> Digest:
    items = []
    for item in digest.items:
        document_ids = [document_id for document_id in dict.fromkeys(item.document_ids) if document_id in known_ids]
        if document_ids:
            items.append(item.model_copy(update={"document_ids": document_ids}))
    return Digest(items=items)
