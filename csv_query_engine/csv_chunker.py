"""
csv_chunker.py — CSV chunking pipeline for the RAG pipeline.

Instead of permanently loading entire CSV files into PostgreSQL, this module:
  1. Splits large CSV files into fixed-size row chunks (default 10 000 rows).
  2. Generates schema + statistical metadata for each chunk.
  3. Produces a human-readable summary text suitable for embedding in Qdrant
     so that CSV files become searchable alongside other document types.

The resulting CsvChunkMetadata objects are passed to the CSV session manager
(csv_session.py) which loads only the required chunks into a temporary
PostgreSQL table for query execution, then drops the table immediately.

Public API
----------
chunk_csv_file(path, chunk_size)   → Iterator[CsvChunkMetadata]
describe_csv(path)                 → CsvFileDescription
build_schema_text(desc)            → str   (for Qdrant embedding)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import chardet
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_CHUNK_SIZE   = 10_000   # rows per chunk
_MAX_SAMPLE_ROWS     = 5        # sample rows kept in metadata
_MAX_UNIQUE_DISPLAY  = 20       # max unique values shown for low-cardinality columns
_MAX_STR_LEN         = 120      # truncate long string values in summaries


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ColumnSummary:
    """Statistical summary for a single CSV column."""
    name:        str
    dtype:       str
    null_count:  int
    null_pct:    float
    # Numeric stats (None for non-numeric columns)
    min_val:     Optional[float] = None
    max_val:     Optional[float] = None
    mean_val:    Optional[float] = None
    # Categorical stats (None for numeric / high-cardinality columns)
    unique_count: Optional[int]  = None
    top_values:   Optional[list] = None   # up to _MAX_UNIQUE_DISPLAY values


@dataclass
class CsvChunkMetadata:
    """Metadata for a single chunk (slice of rows) of a CSV file."""
    file_path:   str
    file_hash:   str              # MD5 hex — used for cache invalidation
    chunk_index: int              # 0-based
    chunk_total: int
    row_start:   int              # inclusive, 0-based (not counting header)
    row_end:     int              # exclusive
    row_count:   int
    columns:     list[str]
    dtypes:      dict[str, str]   # column → pandas dtype string
    sample_rows: list[dict]       # first _MAX_SAMPLE_ROWS rows
    col_summaries: list[ColumnSummary] = field(default_factory=list)
    schema_text: str = ""         # plain-text description for Qdrant embedding


@dataclass
class CsvFileDescription:
    """
    Top-level description of an entire CSV file.

    Produced by ``describe_csv()`` before chunking.  The schema_text field
    is suitable for embedding as a single "schema chunk" in Qdrant so users
    can find the file by asking natural-language questions about its columns.
    """
    file_path:   str
    file_hash:   str
    total_rows:  int
    columns:     list[str]
    dtypes:      dict[str, str]
    col_summaries: list[ColumnSummary]
    sample_rows: list[dict]       # first _MAX_SAMPLE_ROWS rows
    chunk_total: int              # how many chunks the file will be split into
    chunk_size:  int
    schema_text: str = ""         # plain-text for Qdrant embedding


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_file_hash(path: str | Path) -> str:
    """Return the MD5 hex digest of a file (for cache-invalidation)."""
    md5 = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65_536), b""):
            md5.update(block)
    return md5.hexdigest()


def _detect_encoding(path: str | Path, sample_bytes: int = 65_536) -> str:
    """Sniff file encoding; fall back to utf-8."""
    with open(path, "rb") as fh:
        raw = fh.read(sample_bytes)
    detected = chardet.detect(raw).get("encoding")
    return detected or "utf-8"


def _safe_read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read CSV with auto-detected encoding and relaxed parsing."""
    encoding = _detect_encoding(path)
    return pd.read_csv(
        path,
        encoding=encoding,
        on_bad_lines="skip",
        low_memory=False,
        **kwargs,
    )


def _slugify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to lowercase alphanumeric + underscores."""
    df = df.copy()
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_") or f"col_{i}"
        for i, col in enumerate(df.columns)
    ]
    return df


def _summarise_column(series: pd.Series) -> ColumnSummary:
    """Compute a ColumnSummary for a single pandas Series."""
    name       = str(series.name)
    dtype      = str(series.dtype)
    null_count = series.isna().sum()
    null_pct   = round(100.0 * null_count / max(len(series), 1), 1)

    # Numeric stats
    min_val = max_val = mean_val = None
    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if len(non_null):
            min_val  = round(float(non_null.min()),  4)
            max_val  = round(float(non_null.max()),  4)
            mean_val = round(non_null.mean(), 4)

    # Categorical / text stats
    unique_count = top_values = None
    if not pd.api.types.is_numeric_dtype(series):
        uq = series.nunique(dropna=True)
        unique_count = uq
        if uq <= _MAX_UNIQUE_DISPLAY:
            top_values = [
                str(v)[:_MAX_STR_LEN]
                for v in series.dropna().value_counts().head(_MAX_UNIQUE_DISPLAY).index.tolist()
            ]

    return ColumnSummary(
        name=name, dtype=dtype,
        null_count=null_count, null_pct=null_pct,
        min_val=min_val, max_val=max_val, mean_val=mean_val,
        unique_count=unique_count, top_values=top_values,
    )


def _build_schema_text(
    file_path: str,
    total_rows: int,
    columns: list[str],
    dtypes: dict[str, str],
    col_summaries: list[ColumnSummary],
    sample_rows: list[dict],
    chunk_info: str = "",
) -> str:
    """
    Build a human-readable schema description suitable for Qdrant embedding.

    The text is intentionally verbose so semantic search can match a wide
    variety of natural-language questions about the dataset.
    """
    filename = Path(file_path).name
    lines: list[str] = []

    lines.append(f"[CSV File] {filename}")
    lines.append(f"Rows: {total_rows:,}  |  Columns ({len(columns)}): {', '.join(columns)}")
    if chunk_info:
        lines.append(chunk_info)
    lines.append("")

    lines.append("Column details:")
    for cs in col_summaries:
        detail = f"  • {cs.name} ({cs.dtype})"
        if cs.null_pct > 0:
            detail += f"  [{cs.null_pct:.1f}% null]"
        if cs.min_val is not None:
            detail += f"  range [{cs.min_val}, {cs.max_val}]  mean {cs.mean_val}"
        elif cs.top_values:
            vals = ", ".join(str(v)[:40] for v in cs.top_values[:8])
            detail += f"  values: {vals}"
        lines.append(detail)

    if sample_rows:
        lines.append("")
        lines.append("Sample rows:")
        for row in sample_rows[:3]:
            parts = [f"{k}={str(v)[:40]}" for k, v in row.items()]
            lines.append("  " + " | ".join(parts[:6]))

    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def describe_csv(
    path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> CsvFileDescription:
    """
    Read the full CSV file and return a CsvFileDescription with statistics.

    For very large files this reads the whole file once to compute global
    statistics; prefer ``chunk_csv_file()`` for streaming processing.

    Parameters
    ----------
    path:
        Path to the CSV file.
    chunk_size:
        Row chunk size used to compute ``chunk_total``.

    Returns
    -------
    CsvFileDescription
    """
    path       = Path(path)
    file_hash  = _compute_file_hash(path)
    df         = _safe_read_csv(path)
    df         = _slugify_columns(df)

    total_rows    = len(df)
    columns       = df.columns.tolist()
    dtypes        = {col: str(df[col].dtype) for col in columns}
    col_summaries = [_summarise_column(df[col]) for col in columns]
    sample_rows   = df.head(_MAX_SAMPLE_ROWS).to_dict(orient="records")
    chunk_total   = max(1, (total_rows + chunk_size - 1) // chunk_size)

    schema_text = _build_schema_text(
        file_path=str(path),
        total_rows=total_rows,
        columns=columns,
        dtypes=dtypes,
        col_summaries=col_summaries,
        sample_rows=sample_rows,
    )

    return CsvFileDescription(
        file_path=str(path),
        file_hash=file_hash,
        total_rows=total_rows,
        columns=columns,
        dtypes=dtypes,
        col_summaries=col_summaries,
        sample_rows=sample_rows,
        chunk_total=chunk_total,
        chunk_size=chunk_size,
        schema_text=schema_text,
    )


def chunk_csv_file(
    path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[CsvChunkMetadata]:
    """
    Yield CsvChunkMetadata objects for each row-chunk of the CSV file.

    The CSV is read in a single pass; each chunk is materialised in memory
    one at a time so memory usage scales with ``chunk_size``, not file size.

    Parameters
    ----------
    path:
        Path to the CSV file.
    chunk_size:
        Maximum number of rows per chunk.

    Yields
    ------
    CsvChunkMetadata
        One object per chunk.  The last chunk may have fewer than
        ``chunk_size`` rows.
    """
    path      = Path(path)
    file_hash = _compute_file_hash(path)
    encoding  = _detect_encoding(path)

    # First pass: count total rows to compute chunk_total
    total_rows = sum(1 for _ in open(path, encoding=encoding, errors="replace")) - 1
    total_rows = max(total_rows, 0)
    chunk_total = max(1, (total_rows + chunk_size - 1) // chunk_size)

    # Second pass: stream chunks
    reader = pd.read_csv(
        path,
        encoding=encoding,
        on_bad_lines="skip",
        low_memory=False,
        chunksize=chunk_size,
    )

    chunk_index = 0
    row_cursor  = 0
    for raw_chunk in reader:
        df    = _slugify_columns(raw_chunk)
        nrows = len(df)

        columns  = df.columns.tolist()
        dtypes   = {col: str(df[col].dtype) for col in columns}
        summaries = [_summarise_column(df[col]) for col in columns]
        samples  = df.head(_MAX_SAMPLE_ROWS).to_dict(orient="records")

        chunk_info = (
            f"Chunk {chunk_index + 1}/{chunk_total}  "
            f"rows {row_cursor + 1}–{row_cursor + nrows}"
        )
        schema_text = _build_schema_text(
            file_path=str(path),
            total_rows=nrows,
            columns=columns,
            dtypes=dtypes,
            col_summaries=summaries,
            sample_rows=samples,
            chunk_info=chunk_info,
        )

        yield CsvChunkMetadata(
            file_path=str(path),
            file_hash=file_hash,
            chunk_index=chunk_index,
            chunk_total=chunk_total,
            row_start=row_cursor,
            row_end=row_cursor + nrows,
            row_count=nrows,
            columns=columns,
            dtypes=dtypes,
            sample_rows=samples,
            col_summaries=summaries,
            schema_text=schema_text,
        )

        chunk_index += 1
        row_cursor  += nrows


def build_qdrant_chunks(
    desc: CsvFileDescription,
    source_file: str,
    owner_id: str = "",
) -> list[dict]:
    """
    Build a list of RAG pipeline chunk dicts from a CsvFileDescription.

    Each returned dict matches the canonical chunk schema so it can be
    indexed directly with ``index_chunks()`` from the embedding pipeline.

    One "schema chunk" is produced for the whole file (type="csv_schema").
    If the file was actually split into multiple row-chunks, additional
    chunk dicts are produced for each chunk's statistical summary.

    Parameters
    ----------
    desc:
        CsvFileDescription produced by ``describe_csv()``.
    source_file:
        The original filename to stamp on every chunk.
    owner_id:
        Owner identifier for RBAC filtering (empty = anonymous).

    Returns
    -------
    list[dict]
        Chunk dicts ready for the indexing pipeline.
    """
    import uuid

    doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, source_file))
    chunks: list[dict] = []

    # Main schema chunk
    chunks.append({
        "chunk_id": f"{doc_id}_{Path(source_file).stem}_schema",
        "type":     "csv_schema",
        "content":  desc.schema_text,
        "metadata": {
            "source":      source_file,
            "doc_id":      doc_id,
            "file_hash":   desc.file_hash,
            "total_rows":  desc.total_rows,
            "columns":     desc.columns,
            "chunk_size":  desc.chunk_size,
            "chunk_total": desc.chunk_total,
            "section":     "csv_schema",
            "owner_id":    owner_id,
            "token_count": len(desc.schema_text) // 4,
            "page_start":  1,
            "page_end":    1,
        },
    })

    return chunks
