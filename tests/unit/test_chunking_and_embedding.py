import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from ai_news.rag import chunker as chunker_module
from ai_news.rag import embeddings as embeddings_module
from ai_news.rag.chunker import DocumentChunker, Outline, Timeline, locate, print_summary, save_chunks
from ai_news.rag.connector import StoreResult
from ai_news.rag.embeddings import (
    ChunkFile,
    chunk_ids_by_document,
    embed_chunk_file,
    embed_chunks,
    embedding_text,
    fingerprint,
)


class WordTokenizer:
    """One token per whitespace-separated word, standing in for the model's tokenizer."""

    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=text.split())


@pytest.fixture
def chunker(monkeypatch):
    monkeypatch.setattr(chunker_module.Tokenizer, "from_pretrained", lambda *args, **kwargs: WordTokenizer())
    return DocumentChunker(embedding_model="fake/model", chunk_size=20, chunk_overlap=0)


def source_document(text: str, content_kind: str = "article", **metadata) -> Document:
    return Document(
        page_content=text,
        metadata={
            "document_id": 7,
            "source": "anthropic",
            "url": "https://example.com/watch?v=x",
            "title": "Title",
            "content_kind": content_kind,
            "published_at": "2026-11-01T00:00:00+00:00",
            **metadata,
        },
    )


class TestTimeline:
    SEGMENTS = [
        {"text": "hello world", "start": 0.0, "duration": 2.0},
        {"text": "second part", "start": 2.0, "duration": 3.0},
        {"text": "third", "start": 5.0, "duration": 1.5},
    ]

    def test_offsets_account_for_joining_spaces(self):
        assert Timeline.from_segments(self.SEGMENTS).offsets == [0, 12, 24]

    def test_span_maps_characters_to_seconds(self):
        timeline = Timeline.from_segments(self.SEGMENTS)
        assert timeline.span(0, 5) == (0.0, 2.0)
        assert timeline.span(13, 26) == (2.0, 6.5)
        assert timeline.span(30, 10) == (5.0, 6.5)


class TestOutline:
    def test_tracks_the_heading_above_each_offset(self):
        text = "intro\n# First\nbody\n## Second ##\nmore"
        outline = Outline.from_markdown(text)
        assert outline.headings == ["First", "Second"]
        assert outline.section_at(0) is None
        assert outline.section_at(text.index("body")) == "First"
        assert outline.section_at(len(text) - 1) == "Second"

    def test_ignores_hash_lines_inside_code_fences(self):
        text = "# Real\n```bash\n# not a heading\n```\n~~~\n# also not\n~~~\ntext"
        assert Outline.from_markdown(text).headings == ["Real"]


def test_locate_finds_overlapping_pieces_in_order():
    assert locate("abcabc", ["abc", "abc"]) == [0, 3]
    assert locate("aaaa", ["aa", "aa", "aa"]) == [0, 1, 2]
    assert locate("abc", ["zzz"]) == [-1]


class TestDocumentChunker:
    def test_article_chunks_get_sections_and_ids(self, chunker):
        body = " ".join(f"word{i}" for i in range(15))
        text = f"# Intro\n\n{body}\n\n# Details\n\n{body}"
        chunks = chunker.split(source_document(text))
        assert [chunk.id for chunk in chunks] == ["7-0", "7-1"]
        assert [chunk.metadata["section"] for chunk in chunks] == ["Intro", "Details"]
        assert all(chunk.metadata["token_count"] <= 20 for chunk in chunks)
        assert chunks[1].metadata["chunk_count"] == 2
        assert text[chunks[1].metadata["start_index"] :].startswith(chunks[1].page_content)

    def test_transcript_chunks_get_timestamps_and_drop_segments(self, chunker):
        words = [f"w{i}" for i in range(30)]
        segments = [{"text": word, "start": float(i), "duration": 1.0} for i, word in enumerate(words)]
        document = source_document(" ".join(words), content_kind="transcript", segments=segments)
        chunks = chunker.transform_documents([document])
        assert len(chunks) == 2
        assert "segments" not in chunks[0].metadata
        assert "section" not in chunks[0].metadata
        assert chunks[0].metadata["start_seconds"] == 0.0
        assert chunks[1].metadata["start_seconds"] > 10
        assert (
            chunks[1].metadata["timestamp_url"]
            == f"https://example.com/watch?v=x&t={int(chunks[1].metadata['start_seconds'])}s"
        )

    def test_count_tokens(self, chunker):
        assert chunker.count_tokens("three small words") == 3

    def test_save_and_summary(self, chunker, tmp_path, capsys):
        documents = [source_document("one two three")]
        chunks = chunker.transform_documents(documents)
        path = save_chunks(chunker, chunks, tmp_path / "data" / "chunks.json")
        payload = json.loads(path.read_text())
        assert payload["chunk_count"] == 1
        assert payload["embedding_model"] == "fake/model"
        print_summary(documents, chunks)
        assert "anthropic" in capsys.readouterr().out


def chunk(chunk_id: str, document_id: int, section: str | None = None) -> Document:
    return Document(
        id=chunk_id, page_content="body", metadata={"document_id": document_id, "title": "Title", "section": section}
    )


def test_embedding_text_prefixes_title_and_distinct_section():
    assert embedding_text(chunk("a", 1)) == "Title\n\nbody"
    assert embedding_text(chunk("a", 1, "Results")) == "Title — Results\n\nbody"
    assert embedding_text(chunk("a", 1, "Title")) == "Title\n\nbody"


def test_fingerprint_changes_with_model_and_text():
    assert fingerprint(chunk("a", 1), "m1") != fingerprint(chunk("a", 1), "m2")
    assert fingerprint(chunk("a", 1), "m1") != fingerprint(chunk("a", 1, "S"), "m1")
    assert fingerprint(chunk("a", 1), "m1") == fingerprint(chunk("b", 2), "m1")


def test_chunk_ids_by_document():
    assert chunk_ids_by_document([chunk("1-0", 1), chunk("1-1", 1), chunk("2-0", 2)]) == {1: {"1-0", "1-1"}, 2: {"2-0"}}


class FakeEmbeddings:
    model_name = "fake/model"
    batch_size = 2

    def __init__(self, *args, dimensions: int = 384, **kwargs) -> None:
        self.dimensions = dimensions
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return [[0.1] * self.dimensions for _ in texts]


def test_embed_chunks_batches_and_checks_dimensions():
    embeddings = FakeEmbeddings()
    embedded = embed_chunks(embeddings, [chunk(f"1-{i}", 1) for i in range(3)])
    assert embeddings.calls == 2
    assert [item.chunk.id for item in embedded] == ["1-0", "1-1", "1-2"]
    assert embedded[0].content_hash == fingerprint(chunk("1-0", 1), "fake/model")
    with pytest.raises(ValueError, match="768-dimension"):
        embed_chunks(FakeEmbeddings(dimensions=768), [chunk("1-0", 1)])


def test_chunk_file_reads_what_the_chunker_saves(tmp_path):
    path = tmp_path / "chunks.json"
    path.write_text(
        json.dumps(
            {
                "embedding_model": "m",
                "chunk_size_tokens": 400,
                "chunk_overlap_tokens": 60,
                "chunks": [{"id": "1-0", "text": "body", "metadata": {"document_id": 1}}],
            }
        )
    )
    chunk_file = ChunkFile.read(path)
    assert chunk_file.embedding_model == "m"
    assert chunk_file.chunks[0].page_content == "body"


class FakeStore:
    def __init__(self, documents: set[int], hashes: dict[str, str]) -> None:
        self.documents = documents
        self.hashes = hashes
        self.saved = None

    def existing_documents(self, document_ids):
        return set(document_ids) & self.documents

    def stored_hashes(self, chunk_ids):
        return {chunk_id: self.hashes[chunk_id] for chunk_id in chunk_ids if chunk_id in self.hashes}

    def save(self, embedded, chunk_ids):
        self.saved = (embedded, chunk_ids)
        return StoreResult(saved=len(embedded), removed=0)

    def count(self):
        return 99


def test_embed_chunk_file_embeds_only_changed_chunks_of_known_documents(monkeypatch):
    monkeypatch.setattr(
        embeddings_module, "rag_settings", lambda: SimpleNamespace(embedding_model="fake/model", model_cache_dir=None)
    )
    monkeypatch.setattr(embeddings_module, "FastEmbedEmbeddings", FakeEmbeddings)
    unchanged = chunk("1-0", 1)
    chunk_file = ChunkFile("fake/model", 400, 60, [unchanged, chunk("1-1", 1, "New"), chunk("9-0", 9)])
    store = FakeStore({1}, {"1-0": fingerprint(unchanged, "fake/model")})
    report = embed_chunk_file(chunk_file, store)
    assert (report.in_file, report.missing_documents, report.up_to_date, report.table_rows) == (3, 1, 1, 99)
    embedded, chunk_ids = store.saved
    assert [item.chunk.id for item in embedded] == ["1-1"]
    assert chunk_ids == {1: {"1-0", "1-1"}}


def test_embed_chunk_file_with_nothing_to_do_loads_no_model(monkeypatch):
    monkeypatch.setattr(
        embeddings_module, "rag_settings", lambda: SimpleNamespace(embedding_model="m", model_cache_dir=None)
    )

    def no_model(*args, **kwargs):
        raise AssertionError("the model should not load")

    monkeypatch.setattr(embeddings_module, "FastEmbedEmbeddings", no_model)
    current = chunk("1-0", 1)
    report = embed_chunk_file(ChunkFile("m", 400, 60, [current]), FakeStore({1}, {"1-0": fingerprint(current, "m")}))
    assert report.stored.saved == 0
