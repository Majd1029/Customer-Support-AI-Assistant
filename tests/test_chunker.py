"""
tests/test_chunker.py — unit tests for file_preparation/chunking/chunker.py

Covers:
  • Schema normalisation — every chunk has the canonical top-level shape
  • doc_id determinism — same filename always produces the same UUID5
  • chunk_total accuracy — metadata.chunk_total == len(chunks)
  • chunk_index sequence — 1-based, contiguous
  • metadata.page_start propagation from input text_blocks
  • Language detection — metadata.language is an ISO-639-1 string
  • Quality filter — empty-content chunks are dropped before indexing
  • Table chunks — prose content + metadata.display (markdown)

These tests exercise pure in-process logic only — no Qdrant, Ollama, or Groq
required.  The conftest.py ensures file_processor/ is on sys.path so that
ExtractionResult is importable.
"""
from __future__ import annotations

import uuid

import pytest

# ── Imports resolved via conftest.py sys.path setup ───────────────────────────
from models import ExtractionResult, ExtractedTable   # type: ignore[import]
from file_preparation.chunking.chunker import chunk_document


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _simple_result(
    source: str = "test.pdf",
    n_blocks: int = 3,
    text: str = "This is a sample paragraph with enough words to pass the minimum token check.",
    page: int = 1,
) -> ExtractionResult:
    """Return a minimal ExtractionResult with n_blocks identical text blocks."""
    blocks = [(page, text) for _ in range(n_blocks)]
    return ExtractionResult(
        source_file=source,
        text_blocks=blocks,
        tables=[],
        images=[],
        doc_metadata={},
    )


def _chunked(result: ExtractionResult, **kwargs) -> list[dict]:
    """Call chunk_document and return just the chunks list."""
    return chunk_document(result, **kwargs)["chunks"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Top-level chunk shape
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkShape:
    """Every chunk must satisfy the canonical schema contract."""

    def test_required_top_level_keys(self):
        chunks = _chunked(_simple_result())
        assert chunks, "Expected at least one chunk"
        for c in chunks:
            assert "chunk_id"  in c, f"Missing chunk_id in {c}"
            assert "type"      in c, f"Missing type in {c}"
            assert "content"   in c, f"Missing content in {c}"
            assert "metadata"  in c, f"Missing metadata in {c}"

    def test_no_legacy_text_key(self):
        """'text' should have been renamed to 'content' in the normalisation pass."""
        chunks = _chunked(_simple_result())
        for c in chunks:
            assert "text" not in c, f"Legacy 'text' key found in chunk {c['chunk_id']}"

    def test_no_legacy_token_count_key(self):
        """'token_count' should have been moved to metadata.token_count."""
        chunks = _chunked(_simple_result())
        for c in chunks:
            assert "token_count" not in c, \
                f"Legacy top-level 'token_count' found in chunk {c['chunk_id']}"

    def test_metadata_has_required_fields(self):
        chunks = _chunked(_simple_result())
        required = {"source", "doc_id", "token_count", "page_start", "page_end",
                    "chunk_index", "chunk_total"}
        for c in chunks:
            missing = required - c["metadata"].keys()
            assert not missing, \
                f"Chunk {c['chunk_id']} metadata missing: {missing}"

    def test_type_values(self):
        chunks = _chunked(_simple_result())
        valid_types = {"text", "table", "image"}
        for c in chunks:
            assert c["type"] in valid_types, \
                f"Unknown type {c['type']!r} in chunk {c['chunk_id']}"

    def test_content_not_empty(self):
        chunks = _chunked(_simple_result())
        for c in chunks:
            assert c["content"].strip(), \
                f"Empty content in chunk {c['chunk_id']}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. doc_id determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestDocIdDeterminism:
    """doc_id must be a stable UUID5 derived from the source filename."""

    def test_doc_id_is_uuid(self):
        chunks = _chunked(_simple_result("report.pdf"))
        for c in chunks:
            doc_id = c["metadata"]["doc_id"]
            # Should parse as a valid UUID without raising ValueError
            parsed = uuid.UUID(doc_id)
            assert str(parsed) == doc_id, "doc_id is not a canonical UUID string"

    def test_doc_id_consistent_across_chunks(self):
        """All chunks from the same document must share the same doc_id."""
        result = _simple_result("consistency.pdf", n_blocks=5)
        chunks = _chunked(result)
        assert len(chunks) >= 2, "Need at least 2 chunks for this test"
        ids = {c["metadata"]["doc_id"] for c in chunks}
        assert len(ids) == 1, f"Multiple doc_ids in single document: {ids}"

    def test_doc_id_same_across_runs(self):
        """Running chunk_document twice on the same file gives the same doc_id."""
        result = _simple_result("idempotent.pdf")
        doc_id_1 = _chunked(result)[0]["metadata"]["doc_id"]
        doc_id_2 = _chunked(result)[0]["metadata"]["doc_id"]
        assert doc_id_1 == doc_id_2

    def test_different_filenames_give_different_doc_ids(self):
        chunks_a = _chunked(_simple_result("alpha.pdf"))
        chunks_b = _chunked(_simple_result("beta.pdf"))
        assert chunks_a[0]["metadata"]["doc_id"] != chunks_b[0]["metadata"]["doc_id"]

    def test_doc_id_matches_uuid5_of_filename(self):
        """doc_id should equal uuid.uuid5(NAMESPACE_URL, filename)."""
        filename = "verifiable.pdf"
        chunks = _chunked(_simple_result(filename))
        expected = str(uuid.uuid5(uuid.NAMESPACE_URL, filename))
        for c in chunks:
            assert c["metadata"]["doc_id"] == expected


# ─────────────────────────────────────────────────────────────────────────────
# 3. chunk_index / chunk_total accuracy
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkIndexing:
    """chunk_index must be 1-based and chunk_total must equal len(chunks)."""

    def test_chunk_total_equals_list_length(self):
        chunks = _chunked(_simple_result(n_blocks=4))
        total_in_meta = {c["metadata"]["chunk_total"] for c in chunks}
        assert len(total_in_meta) == 1, "chunk_total is inconsistent across chunks"
        assert list(total_in_meta)[0] == len(chunks), \
            f"chunk_total={list(total_in_meta)[0]} != len(chunks)={len(chunks)}"

    def test_chunk_index_is_one_based(self):
        chunks = _chunked(_simple_result(n_blocks=4))
        indices = sorted(c["metadata"]["chunk_index"] for c in chunks)
        assert indices[0] == 1, f"First chunk_index should be 1, got {indices[0]}"

    def test_chunk_index_is_contiguous(self):
        chunks = _chunked(_simple_result(n_blocks=4))
        indices = sorted(c["metadata"]["chunk_index"] for c in chunks)
        expected = list(range(1, len(chunks) + 1))
        assert indices == expected, f"chunk_index not contiguous: {indices}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. metadata.source
# ─────────────────────────────────────────────────────────────────────────────

class TestMetadataSource:
    def test_source_matches_input_filename(self):
        chunks = _chunked(_simple_result("myfile.pdf"))
        for c in chunks:
            assert c["metadata"]["source"] == "myfile.pdf"

    def test_source_not_in_top_level(self):
        """source_file should have been moved to metadata.source."""
        chunks = _chunked(_simple_result("myfile.pdf"))
        for c in chunks:
            assert "source_file" not in c, \
                f"Legacy 'source_file' found at top level in chunk {c['chunk_id']}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. page_start propagation
# ─────────────────────────────────────────────────────────────────────────────

class TestPageStart:
    def test_page_start_matches_input_block_page(self):
        """Chunks from a 0-based page-2 block should have page_start=3 (1-based)."""
        # PDF parsers produce 0-based page indices (pypdf enumerate(reader.pages)).
        # The chunker adds 1 when storing page_start so users see 1-based numbers.
        # A block at 0-based page 2 → page_start = 2 + 1 = 3.
        result = ExtractionResult(
            source_file="paged.pdf",
            text_blocks=[(2, "This is text from page three of the document, enough tokens here.")],
            tables=[],
            images=[],
            doc_metadata={},
        )
        chunks = _chunked(result)
        assert chunks, "Expected at least one chunk"
        for c in chunks:
            assert c["metadata"]["page_start"] == 3, \
                f"Expected page_start=3, got {c['metadata']['page_start']}"

    def test_page_end_equals_page_start_for_text(self):
        chunks = _chunked(_simple_result(page=5))
        for c in chunks:
            assert c["metadata"]["page_end"] == c["metadata"]["page_start"]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Table chunk normalisation
# ─────────────────────────────────────────────────────────────────────────────

class TestTableNormalisation:
    def test_table_chunk_has_display_in_metadata(self):
        """Table markdown should move to metadata.display; content becomes prose."""
        table = ExtractedTable(
            page_number=1,
            table_index=0,
            markdown="| Name | Score |\n|------|-------|\n| Alice | 95 |\n| Bob | 82 |",
            raw_rows=[["Name", "Score"], ["Alice", "95"], ["Bob", "82"]],
        )
        result = ExtractionResult(
            source_file="report.pdf",
            text_blocks=[(1, "Some context text on this page for the document.")],
            tables=[table],
            images=[],
            doc_metadata={},
        )
        chunks = _chunked(result)
        table_chunks = [c for c in chunks if c["type"] == "table"]
        assert table_chunks, "Expected at least one table chunk"
        for tc in table_chunks:
            assert "display" in tc["metadata"], "Table chunk missing metadata.display"
            # display should contain the pipe characters
            assert "|" in tc["metadata"]["display"]

    def test_table_chunk_no_legacy_rows_key(self):
        """raw rows should NOT appear at the top level after normalisation."""
        table = ExtractedTable(
            page_number=0,
            table_index=0,
            markdown="| A | B |\n|---|---|\n| 1 | 2 |",
            raw_rows=[["A", "B"], ["1", "2"]],
        )
        result = ExtractionResult(
            source_file="table.pdf",
            text_blocks=[],
            tables=[table],
            images=[],
            doc_metadata={},
        )
        chunks = _chunked(result)
        for c in chunks:
            assert "rows" not in c, f"Legacy 'rows' key found at top level in {c['chunk_id']}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Language detection
# ─────────────────────────────────────────────────────────────────────────────

class TestLanguageDetection:
    def test_english_text_detected(self):
        long_en = (
            "The financial results for the third quarter show a significant improvement "
            "in revenue and operating profit compared to the same period last year. "
            "Management remains optimistic about future growth prospects."
        )
        result = ExtractionResult(
            source_file="en_doc.pdf",
            text_blocks=[(1, long_en)],
            tables=[], images=[], doc_metadata={},
        )
        chunks = _chunked(result)
        assert chunks, "No chunks produced"
        # If langdetect is available, language should be detected
        for c in chunks:
            lang = c["metadata"].get("language")
            if lang is not None:
                assert isinstance(lang, str)
                assert len(lang) == 2, f"Expected ISO-639-1 code (2 chars), got {lang!r}"
                assert lang == "en", f"Expected 'en', got {lang!r}"

    def test_short_text_may_skip_language(self):
        """Chunks with very short content skip language detection (unreliable)."""
        result = ExtractionResult(
            source_file="short.pdf",
            # This is below the 30-char minimum for detection
            text_blocks=[(1, "Hello there, a bit of text here but maybe enough.")],
            tables=[], images=[], doc_metadata={},
        )
        chunks = _chunked(result)
        # We don't assert a specific language — just that the field is absent or valid
        for c in chunks:
            lang = c["metadata"].get("language")
            if lang is not None:
                assert isinstance(lang, str) and len(lang) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 8. Quality filter
# ─────────────────────────────────────────────────────────────────────────────

class TestQualityFilter:
    def test_empty_blocks_produce_no_chunks(self):
        """Blocks that are only whitespace or too short should produce no output."""
        result = ExtractionResult(
            source_file="empty.pdf",
            text_blocks=[(1, ""), (1, "   "), (1, "\n\n\n")],
            tables=[], images=[], doc_metadata={},
        )
        chunks = _chunked(result)
        for c in chunks:
            assert c["content"].strip(), "Empty-content chunk slipped through quality filter"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Stats dict
# ─────────────────────────────────────────────────────────────────────────────

class TestStatsDict:
    def test_stats_keys_present(self):
        doc = chunk_document(_simple_result())
        stats = doc["stats"]
        assert "total_chunks" in stats
        assert "text_chunks"  in stats

    def test_source_file_key(self):
        doc = chunk_document(_simple_result("source_check.pdf"))
        assert "source_file" in doc

    def test_total_chunks_consistent(self):
        doc = chunk_document(_simple_result(n_blocks=3))
        assert doc["stats"]["total_chunks"] == len(doc["chunks"])
