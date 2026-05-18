"""
tests/test_store.py — unit tests for store-layer pure functions.

Covers:
  • normalize_for_embedding — strips markdown syntax before BGE-M3 encoding
  • make_point_id — deterministic UUID5 from chunk_id string

No Qdrant connection required — these functions are pure Python.
"""
from __future__ import annotations

import uuid

import pytest

from file_preparation.indexing.indexer import normalize_for_embedding
from file_preparation.indexing.store   import make_point_id


# ─────────────────────────────────────────────────────────────────────────────
# 1. normalize_for_embedding
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeForEmbedding:
    """normalize_for_embedding should strip markdown noise and return plain text."""

    def test_plain_text_unchanged(self):
        text = "The revenue grew by twelve percent year over year."
        result = normalize_for_embedding(text)
        assert "revenue grew by twelve percent" in result

    def test_strips_atx_headings(self):
        text = "## Introduction\nThis is the introduction."
        result = normalize_for_embedding(text)
        assert "##" not in result
        assert "Introduction" in result

    def test_strips_multiple_heading_levels(self):
        for level in range(1, 7):
            text = "#" * level + " Section Title\nBody text."
            result = normalize_for_embedding(text)
            assert "#" not in result or result.startswith(" "), \
                f"Heading level {level} not stripped: {result!r}"

    def test_strips_bold_italic(self):
        text = "This is **bold** and *italic* text."
        result = normalize_for_embedding(text)
        assert "**" not in result
        assert "*bold*" not in result
        assert "bold" in result
        assert "italic" in result

    def test_strips_pipe_table_rows(self):
        text = "| Name | Score |\n| Alice | 95 |\n| Bob | 82 |"
        result = normalize_for_embedding(text)
        # Pipe characters should be gone (or heavily reduced)
        assert "| Alice" not in result

    def test_strips_inline_code(self):
        text = "Use the `get_client()` function to connect."
        result = normalize_for_embedding(text)
        assert "`" not in result
        assert "get_client" not in result or "function to connect" in result

    def test_strips_horizontal_rules(self):
        text = "Above the rule.\n---\nBelow the rule."
        result = normalize_for_embedding(text)
        assert "---" not in result
        assert "Above the rule" in result
        assert "Below the rule" in result

    def test_resolves_markdown_links(self):
        text = "Read the [documentation](https://docs.example.com) for details."
        result = normalize_for_embedding(text)
        assert "documentation" in result
        # URL should be stripped
        assert "https://" not in result

    def test_empty_string(self):
        result = normalize_for_embedding("")
        assert result == ""

    def test_returns_stripped_string(self):
        """Result should not have leading or trailing whitespace."""
        text = "  ## Title  \n  Some text.  "
        result = normalize_for_embedding(text)
        assert result == result.strip()

    def test_collapses_excessive_whitespace(self):
        text = "Word   with   extra   spaces."
        result = normalize_for_embedding(text)
        # No runs of more than one space (pattern collapses 2+ spaces to one)
        assert "  " not in result

    def test_collapses_blank_lines(self):
        text = "First paragraph.\n\n\n\n\nSecond paragraph."
        result = normalize_for_embedding(text)
        # Should not have more than two consecutive newlines
        assert "\n\n\n" not in result


# ─────────────────────────────────────────────────────────────────────────────
# 2. make_point_id
# ─────────────────────────────────────────────────────────────────────────────

class TestMakePointId:
    def test_returns_uuid(self):
        point_id = make_point_id("some_chunk_id_001")
        # Should be a valid UUID string
        parsed = uuid.UUID(str(point_id))
        assert parsed is not None

    def test_deterministic(self):
        """Same chunk_id always produces the same point ID."""
        a = make_point_id("chunk_abc_p1_txt0")
        b = make_point_id("chunk_abc_p1_txt0")
        assert str(a) == str(b)

    def test_different_ids_produce_different_points(self):
        a = make_point_id("chunk_1")
        b = make_point_id("chunk_2")
        assert str(a) != str(b)
