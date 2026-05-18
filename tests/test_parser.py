"""
tests/test_parser.py — unit tests for file_processor/parser.py

Covers:
  • DOCX page estimation heuristic (_DOCX_CHARS_PER_PAGE)
    - A short document should have all blocks on page 1
    - A document with enough text to exceed _DOCX_CHARS_PER_PAGE should
      have at least one block on page > 1
    - doc_metadata["estimated_pages"] reflects the heuristic result

No Ollama, Groq, or Qdrant required.
python-docx must be installed (it is a core project dependency).
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytest

# parser.py uses within-package sibling imports (from models import …)
# conftest.py adds file_processor/ to sys.path so this import works.
try:
    import docx as _docx_pkg   # python-docx
    _PYTHON_DOCX_AVAILABLE = True
except ImportError:
    _PYTHON_DOCX_AVAILABLE = False

from parser import _DOCX_CHARS_PER_PAGE   # type: ignore[import]   (via conftest sys.path)
from parser import _parse_docx             # type: ignore[import]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_docx(paragraphs: list[str]) -> Path:
    """
    Write an in-memory DOCX to a NamedTemporaryFile and return its Path.
    The file is NOT auto-deleted — caller is responsible for cleanup.
    """
    import docx
    doc = docx.Document()
    for text in paragraphs:
        doc.add_paragraph(text)

    tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
    doc.save(tmp.name)
    tmp.close()
    return Path(tmp.name)


def _cleanup(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 1. Constant sanity
# ─────────────────────────────────────────────────────────────────────────────

class TestDocxCharsPerPageConstant:
    def test_constant_is_positive_int(self):
        assert isinstance(_DOCX_CHARS_PER_PAGE, int)
        assert _DOCX_CHARS_PER_PAGE > 0

    def test_constant_in_expected_range(self):
        """Should be in the range 1 000–10 000 characters per page."""
        assert 1_000 <= _DOCX_CHARS_PER_PAGE <= 10_000


# ─────────────────────────────────────────────────────────────────────────────
# 2. Page estimation — short document
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _PYTHON_DOCX_AVAILABLE, reason="python-docx not installed")
class TestDocxPageEstimationShort:
    def test_short_doc_all_on_page_1(self):
        """A document under _DOCX_CHARS_PER_PAGE chars should stay on page 1."""
        short_para = "This is a short paragraph."  # well under 3 000 chars
        path = _make_docx([short_para] * 3)
        try:
            text_blocks, tables, images, doc_meta = _parse_docx(path)
            pages = {pg for pg, _ in text_blocks}
            assert pages <= {1}, f"Short doc has pages: {pages}"
            assert doc_meta.get("estimated_pages", 1) == 1
        finally:
            _cleanup(path)

    def test_empty_doc_returns_empty_blocks(self):
        path = _make_docx([])
        try:
            text_blocks, tables, images, doc_meta = _parse_docx(path)
            assert isinstance(text_blocks, list)
            assert isinstance(tables, list)
            assert isinstance(images, list)
        finally:
            _cleanup(path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Page estimation — long document
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _PYTHON_DOCX_AVAILABLE, reason="python-docx not installed")
class TestDocxPageEstimationLong:
    def test_long_doc_has_multiple_pages(self):
        """
        A document with total text > _DOCX_CHARS_PER_PAGE should produce at
        least one text block on page > 1.
        """
        # Each paragraph is ~200 chars; write enough to exceed the threshold
        para = "A" * 200 + " " + "B" * 200 + " sample paragraph with text."
        # Need > _DOCX_CHARS_PER_PAGE total chars
        n_paras = (_DOCX_CHARS_PER_PAGE // len(para)) + 5
        path = _make_docx([para] * n_paras)
        try:
            text_blocks, tables, images, doc_meta = _parse_docx(path)
            pages = {pg for pg, _ in text_blocks}
            assert max(pages) > 1, \
                f"Long doc should span >1 estimated page, got pages: {pages}"
        finally:
            _cleanup(path)

    def test_estimated_pages_in_metadata(self):
        """doc_metadata should include 'estimated_pages' key."""
        para = "X" * 300 + " words "
        n_paras = (_DOCX_CHARS_PER_PAGE // len(para)) + 3
        path = _make_docx([para] * n_paras)
        try:
            text_blocks, tables, images, doc_meta = _parse_docx(path)
            assert "estimated_pages" in doc_meta, \
                "doc_metadata missing 'estimated_pages' key"
            assert doc_meta["estimated_pages"] >= 1
        finally:
            _cleanup(path)

    def test_page_numbers_are_monotonically_non_decreasing(self):
        """
        Page numbers in text_blocks should never decrease as we move through
        the document — the heuristic accumulates chars linearly.
        """
        para = "W" * 150 + " words here. "
        n_paras = (_DOCX_CHARS_PER_PAGE // len(para)) + 8
        path = _make_docx([para] * n_paras)
        try:
            text_blocks, tables, images, doc_meta = _parse_docx(path)
            if not text_blocks:
                return   # no blocks → nothing to check
            pages = [pg for pg, _ in text_blocks]
            for i in range(1, len(pages)):
                assert pages[i] >= pages[i - 1], \
                    f"Page numbers decreased at index {i}: {pages[i - 1]} → {pages[i]}"
        finally:
            _cleanup(path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Parse result shape
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _PYTHON_DOCX_AVAILABLE, reason="python-docx not installed")
class TestDocxParseShape:
    def test_returns_four_tuple(self):
        path = _make_docx(["Hello world."])
        try:
            result = _parse_docx(path)
            assert len(result) == 4, "Expected 4-tuple (text_blocks, tables, images, doc_meta)"
        finally:
            _cleanup(path)

    def test_text_blocks_are_page_text_pairs(self):
        path = _make_docx(["First paragraph.", "Second paragraph."])
        try:
            text_blocks, _, _, _ = _parse_docx(path)
            for item in text_blocks:
                assert len(item) == 2, f"Expected (page, text) pair, got {item!r}"
                pg, text = item
                assert isinstance(pg, int),  f"Page should be int, got {type(pg)}"
                assert isinstance(text, str), f"Text should be str, got {type(text)}"
        finally:
            _cleanup(path)

    def test_doc_metadata_is_dict(self):
        path = _make_docx(["Any content."])
        try:
            _, _, _, doc_meta = _parse_docx(path)
            assert isinstance(doc_meta, dict)
        finally:
            _cleanup(path)
