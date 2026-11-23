import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from ai_news.rag.curation import CurationError, Curator, Digest, DigestItem, format_candidates, keep_known_documents


def candidate(document_id: int, title: str = "Title one") -> Document:
    return Document(
        page_content="passage\n  text",
        metadata={
            "document_id": document_id,
            "source": "youtube",
            "published_at": "2026-11-01T10:00:00+00:00",
            "title": title,
        },
    )


def item(*document_ids: int, headline: str = "h") -> DigestItem:
    return DigestItem(headline=headline, why_it_matters="w", document_ids=list(document_ids))


class StubLLM:
    """Stands in for a chat model's with_structured_output(..., include_raw=True)."""

    def __init__(self, parsed: Digest | None, usage: dict | None = None, error: str | None = None) -> None:
        self.parsed = parsed
        self.usage = usage
        self.error = error
        self.prompts: list[str] = []

    def with_structured_output(self, schema, include_raw, method=None):
        assert schema is Digest and include_raw
        self.methods = [*getattr(self, "methods", []), method]

        def respond(prompt):
            self.prompts.append(prompt.to_string())
            raw = AIMessage(content="", usage_metadata=self.usage)
            return {"parsed": self.parsed, "raw": raw, "parsing_error": self.error}

        return RunnableLambda(respond)


def test_format_candidates():
    text = format_candidates([candidate(1), candidate(2, "Title two")])
    assert text == (
        "[1] youtube · 2026-11-01 · Title one\npassage text\n\n[2] youtube · 2026-11-01 · Title two\npassage text"
    )


def test_keep_known_documents_drops_unknown_and_duplicate_ids():
    digest = Digest(items=[item(1, 1, 99), item(99), item(2, 1)])
    kept = keep_known_documents(digest, {1, 2})
    assert [entry.document_ids for entry in kept.items] == [[1], [2, 1]]


def test_curate_caps_items_and_reads_token_usage():
    usage = {"input_tokens": 6900, "output_tokens": 410, "total_tokens": 7310}
    llm = StubLLM(Digest(items=[item(1, headline=f"h{i}") for i in range(12)]), usage)
    result = Curator(llm, "stub-model").curate("agents", [candidate(1)])
    assert len(result.digest.items) == 10
    assert (result.model, result.input_tokens, result.output_tokens) == ("stub-model", 6900, 410)
    assert "at most 10 stories" in llm.prompts[0]
    assert "Reader's interests: agents" in llm.prompts[0]
    assert "[1] youtube · 2026-11-01 · Title one" in llm.prompts[0]


def test_curate_without_usage_metadata_reports_zero_tokens():
    result = Curator(StubLLM(Digest(items=[item(1)])), "m", max_items=3).curate("x", [candidate(1)])
    assert (result.input_tokens, result.output_tokens) == (0, 0)


def test_curate_with_no_candidates_skips_the_llm():
    llm = StubLLM(Digest(items=[item(1)]))
    assert Curator(llm, "m").curate("x", []).digest.items == []
    assert llm.prompts == []


def test_curate_raises_when_the_model_returns_nothing_usable():
    with pytest.raises(CurationError, match="did not return a valid digest"):
        Curator(StubLLM(None, error="bad json"), "m").curate("x", [candidate(1)])


def test_falls_back_to_tool_calling_when_groq_rejects_the_json():
    import httpx
    from groq import BadRequestError

    request = httpx.Request("POST", "https://api.groq.com")
    rejected = BadRequestError("tool_use_failed", response=httpx.Response(400, request=request), body=None)

    class FlakyLLM(StubLLM):
        def with_structured_output(self, schema, include_raw, method=None):
            runnable = super().with_structured_output(schema, include_raw, method)
            if method == "json_schema":
                return RunnableLambda(lambda prompt: (_ for _ in ()).throw(rejected))
            return runnable

    llm = FlakyLLM(Digest(items=[item(1)]))
    result = Curator(llm, "m").curate("x", [candidate(1)])
    assert [entry.document_ids for entry in result.digest.items] == [[1]]
    assert llm.methods == ["json_schema", "function_calling"]
