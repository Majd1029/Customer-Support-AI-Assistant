"""
tests/test_csv_chunker.py — Unit tests for csv_query_engine/csv_chunker.py.

Tests cover:
  • describe_csv: correct row/column counts, hash stability
  • chunk_csv_file: chunk boundaries, totals, schema text
  • build_qdrant_chunks: output shape and required fields
  • ColumnSummary generation: numeric vs categorical stats
  • _slugify_columns: column name normalisation

No PostgreSQL or Qdrant connection required — all tests are pure Python.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from csv_query_engine.csv_chunker import (
    CsvChunkMetadata,
    CsvFileDescription,
    DEFAULT_CHUNK_SIZE,
    ColumnSummary,
    _build_schema_text,
    _compute_file_hash,
    _slugify_columns,
    _summarise_column,
    build_qdrant_chunks,
    chunk_csv_file,
    describe_csv,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Column slugification
# ─────────────────────────────────────────────────────────────────────────────

class TestSlugifyColumns:
    def test_lowercase_names(self):
        df = pd.DataFrame({"Product Name": [1], "Revenue ($)": [2]})
        result = _slugify_columns(df)
        assert "product_name" in result.columns
        assert "revenue_" in result.columns or "revenue" in result.columns

    def test_empty_name_gets_placeholder(self):
        df = pd.DataFrame({"": [1], "  ": [2]})
        result = _slugify_columns(df)
        for col in result.columns:
            assert len(col) > 0

    def test_original_df_not_mutated(self):
        df = pd.DataFrame({"Column A": [1, 2]})
        original_cols = list(df.columns)
        _slugify_columns(df)
        assert list(df.columns) == original_cols


# ─────────────────────────────────────────────────────────────────────────────
# 2. Column summarisation
# ─────────────────────────────────────────────────────────────────────────────

class TestSummariseColumn:
    def test_numeric_column_has_min_max_mean(self):
        s = pd.Series([1.0, 2.0, 3.0, None], name="revenue")
        cs = _summarise_column(s)
        assert cs.min_val == 1.0
        assert cs.max_val == 3.0
        assert cs.mean_val == pytest.approx(2.0)

    def test_null_count_and_pct(self):
        s = pd.Series([1, None, None, 4], name="val")
        cs = _summarise_column(s)
        assert cs.null_count == 2
        assert cs.null_pct == 50.0

    def test_categorical_column_has_top_values(self):
        s = pd.Series(["cat", "dog", "cat", "bird"], name="animal")
        cs = _summarise_column(s)
        assert cs.top_values is not None
        assert "cat" in cs.top_values

    def test_high_cardinality_column_no_top_values(self):
        import random, string
        values = ["".join(random.choices(string.ascii_lowercase, k=8)) for _ in range(30)]
        s = pd.Series(values, name="uid")
        cs = _summarise_column(s)
        # 30 unique values > _MAX_UNIQUE_DISPLAY (20) → top_values should be None
        assert cs.top_values is None

    def test_fully_null_numeric_handled_gracefully(self):
        s = pd.Series([None, None, None], dtype="float64", name="empty")
        cs = _summarise_column(s)
        assert cs.min_val is None
        assert cs.max_val is None


# ─────────────────────────────────────────────────────────────────────────────
# 3. describe_csv
# ─────────────────────────────────────────────────────────────────────────────

class TestDescribeCsv:
    def test_returns_correct_total_rows(self, sample_csv_file):
        desc = describe_csv(sample_csv_file)
        assert desc.total_rows == 5

    def test_returns_correct_columns(self, sample_csv_file):
        desc = describe_csv(sample_csv_file)
        assert set(desc.columns) == {"id", "name", "category", "revenue", "active"}

    def test_schema_text_is_non_empty(self, sample_csv_file):
        desc = describe_csv(sample_csv_file)
        assert len(desc.schema_text) > 50

    def test_schema_text_contains_filename(self, sample_csv_file):
        desc = describe_csv(sample_csv_file)
        assert sample_csv_file.name in desc.schema_text

    def test_file_hash_is_hex_string(self, sample_csv_file):
        desc = describe_csv(sample_csv_file)
        assert len(desc.file_hash) == 32
        assert all(c in "0123456789abcdef" for c in desc.file_hash)

    def test_chunk_total_correct_for_small_file(self, sample_csv_file):
        """5-row file with default chunk_size → 1 chunk."""
        desc = describe_csv(sample_csv_file, chunk_size=DEFAULT_CHUNK_SIZE)
        assert desc.chunk_total == 1

    def test_chunk_total_correct_for_exact_multiple(self, tmp_path):
        """10 rows with chunk_size=5 → 2 chunks."""
        path = tmp_path / "exact.csv"
        rows = ["a,b"] + [f"{i},{i*2}" for i in range(10)]
        path.write_text("\n".join(rows))
        desc = describe_csv(path, chunk_size=5)
        assert desc.chunk_total == 2

    def test_sample_rows_present(self, sample_csv_file):
        desc = describe_csv(sample_csv_file)
        assert len(desc.sample_rows) >= 1
        assert isinstance(desc.sample_rows[0], dict)

    def test_dtypes_dict_populated(self, sample_csv_file):
        desc = describe_csv(sample_csv_file)
        assert len(desc.dtypes) == len(desc.columns)

    def test_col_summaries_count_matches_columns(self, sample_csv_file):
        desc = describe_csv(sample_csv_file)
        assert len(desc.col_summaries) == len(desc.columns)


# ─────────────────────────────────────────────────────────────────────────────
# 4. chunk_csv_file
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkCsvFile:
    def test_small_file_produces_one_chunk(self, sample_csv_file):
        chunks = list(chunk_csv_file(sample_csv_file, chunk_size=1000))
        assert len(chunks) == 1

    def test_chunk_total_field_is_consistent(self, sample_csv_file):
        chunks = list(chunk_csv_file(sample_csv_file, chunk_size=1000))
        for c in chunks:
            assert c.chunk_total == len(chunks)

    def test_chunk_index_is_sequential(self, sample_csv_file):
        chunks = list(chunk_csv_file(sample_csv_file, chunk_size=2))
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_row_ranges_are_contiguous(self, sample_csv_file):
        chunks = list(chunk_csv_file(sample_csv_file, chunk_size=2))
        cursor = 0
        for c in chunks:
            assert c.row_start == cursor
            cursor = c.row_end

    def test_total_rows_equals_sum_of_chunks(self, sample_csv_file):
        chunks = list(chunk_csv_file(sample_csv_file, chunk_size=2))
        assert sum(c.row_count for c in chunks) == 5

    def test_large_file_produces_multiple_chunks(self, large_csv_file):
        chunks = list(chunk_csv_file(large_csv_file, chunk_size=10_000))
        assert len(chunks) == 3   # 25000 / 10000 = ceil(2.5) = 3

    def test_each_chunk_has_schema_text(self, sample_csv_file):
        for chunk in chunk_csv_file(sample_csv_file, chunk_size=3):
            assert isinstance(chunk.schema_text, str)
            assert len(chunk.schema_text) > 0

    def test_each_chunk_has_col_summaries(self, sample_csv_file):
        for chunk in chunk_csv_file(sample_csv_file, chunk_size=3):
            assert len(chunk.col_summaries) == len(chunk.columns)

    def test_file_hash_same_across_chunks(self, sample_csv_file):
        chunks = list(chunk_csv_file(sample_csv_file, chunk_size=2))
        hashes = {c.file_hash for c in chunks}
        assert len(hashes) == 1, "All chunks should have the same file hash"

    def test_file_hash_changes_when_content_changes(self, tmp_path):
        path = tmp_path / "changing.csv"
        path.write_text("a,b\n1,2\n")
        hash1 = _compute_file_hash(path)

        path.write_text("a,b\n1,2\n3,4\n")
        hash2 = _compute_file_hash(path)

        assert hash1 != hash2


# ─────────────────────────────────────────────────────────────────────────────
# 5. build_qdrant_chunks
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildQdrantChunks:
    def test_returns_list(self, sample_csv_file):
        desc = describe_csv(sample_csv_file)
        chunks = build_qdrant_chunks(desc, source_file="sample.csv")
        assert isinstance(chunks, list)

    def test_at_least_one_schema_chunk(self, sample_csv_file):
        desc = describe_csv(sample_csv_file)
        chunks = build_qdrant_chunks(desc, source_file="sample.csv")
        schema_chunks = [c for c in chunks if c.get("type") == "csv_schema"]
        assert len(schema_chunks) >= 1

    def test_chunk_has_required_fields(self, sample_csv_file):
        desc = describe_csv(sample_csv_file)
        chunks = build_qdrant_chunks(desc, source_file="sample.csv")
        for chunk in chunks:
            assert "chunk_id"  in chunk
            assert "type"      in chunk
            assert "content"   in chunk
            assert "metadata"  in chunk

    def test_owner_id_propagated(self, sample_csv_file):
        desc = describe_csv(sample_csv_file)
        chunks = build_qdrant_chunks(desc, source_file="sample.csv", owner_id="alice")
        for chunk in chunks:
            assert chunk["metadata"]["owner_id"] == "alice"

    def test_doc_id_is_deterministic(self, sample_csv_file):
        desc = describe_csv(sample_csv_file)
        chunks1 = build_qdrant_chunks(desc, source_file="sample.csv")
        chunks2 = build_qdrant_chunks(desc, source_file="sample.csv")
        ids1 = [c["metadata"]["doc_id"] for c in chunks1]
        ids2 = [c["metadata"]["doc_id"] for c in chunks2]
        assert ids1 == ids2

    def test_content_is_non_empty(self, sample_csv_file):
        desc = describe_csv(sample_csv_file)
        chunks = build_qdrant_chunks(desc, source_file="sample.csv")
        for chunk in chunks:
            assert len(chunk["content"].strip()) > 0
