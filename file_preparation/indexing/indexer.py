"""
indexer.py — Embedding and indexing pipeline.

Two entry points:
  • index_chunks(chunks, client)    — embed + upsert a pre-built chunk list
  • index_document(path, client)    — parse → chunk → embed → upsert

Hybrid vector layout per chunk:
  • Dense  — BGE-M3 1024-dim cosine vector (semantic similarity)
  • Sparse — BM25 TF weights via bm25_encoder (lexical matching);
             IDF applied server-side by Qdrant when Modifier.IDF is available

Retrieval logic (retrieve, HyDE, MMR, decompose, reranking) lives in
file_preparation/retrieval/retriever.py.

Usage:
    from file_preparation.indexing.indexer import index_document
    from file_preparation.indexing.store   import get_client, ensure_collection

    client = get_client()
    ensure_collection(client, "documents")

    stats = index_document(Path("report.pdf"), client)
    # {'source': 'report.pdf', 'indexed': 42, 'skipped': 0, ...}
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from loguru import logger

from file_preparation.embedding.embedder     import encode                              # type: ignore[import]
from file_preparation.retrieval.bm25_encoder import bm25_encode, update_corpus_avg_dl  # type: ignore[import]
from file_preparation.indexing.store         import (                                   # noqa: E402
    build_point,
    upsert,
    ensure_collection,
    delete_by_source,
    COLLECTION_NAME,
)
from file_preparation.chunking.chunker       import chunk_document                     # type: ignore[import]
from file_processor.extract                  import process_file                        # type: ignore[import]

# ── Constants ─────────────────────────────────────────────────────────────────
_EMBED_BATCH      = 8       # texts per BGE-M3 inference call
_MAX_SAFE_TOKENS  = 7500    # warn if chunk content exceeds this (BGE-M3 max: 8192)


# ── Text normalisation ────────────────────────────────────────────────────────

# Compiled patterns for markdown stripping (applied before embedding only —
# original content is preserved in the payload for display).
_MD_PATTERNS = [
    (re.compile(r'\|[^\n]+\|'),            ' '),   # pipe table rows
    (re.compile(r'^#+\s+', re.M),         ''),    # ATX headings  (# ## ###)
    (re.compile(r'\*{1,3}([^*]+)\*{1,3}'), r'\1'),# bold / italic
    (re.compile(r'`{1,3}[^`]*`{1,3}'),    ' '),   # inline / fenced code
    (re.compile(r'^---+\s*$', re.M),      ''),    # horizontal rules
    (re.compile(r'\[([^\]]+)\]\([^)]+\)'), r'\1'),# markdown links → text
    (re.compile(r'[ \t]{2,}'),            ' '),   # collapse extra whitespace
    (re.compile(r'\n{3,}'),               '\n\n'),# collapse blank lines
]


def normalize_for_embedding(text: str) -> str:
    """
    Lightly strip markdown syntax from chunk content before embedding.

    Markdown symbols (pipe tables, heading markers, bold/italic asterisks,
    code fences) are not natural language and dilute the semantic vector.
    The original content is kept intact in the Qdrant payload for display.

    Args:
        text: Raw chunk content (may contain markdown).

    Returns:
        Cleaned plain-text string, safe to pass to BGE-M3.
    """
    result = text
    for pattern, replacement in _MD_PATTERNS:
        result = pattern.sub(replacement, result)
    return result.strip()


# ── index_chunks ──────────────────────────────────────────────────────────────

def index_chunks(
    chunks:      list[dict[str, Any]],
    client:      Any,
    collection:  str         = COLLECTION_NAME,
    batch_size:  int         = _EMBED_BATCH,
    source:      str | None  = None,
    clean_first: bool        = True,
) -> dict[str, Any]:
    """
    Embed a list of canonical chunks and upsert them into Qdrant.

    Each chunk must follow the schema produced by chunk_document():
        {
            "chunk_id": str,
            "type":     str,
            "content":  str,
            "metadata": { ... }
        }

    Image chunks whose content is empty (uncaptioned) are skipped silently.

    Args:
        chunks:      Chunk list from chunk_document().
        client:      Active QdrantClient.
        collection:  Target Qdrant collection.
        batch_size:  Texts per BGE-M3 inference call.
        source:      Source filename (e.g. "report.pdf"). When provided and
                     clean_first=True, all existing points for this source are
                     deleted before upserting — prevents stale chunks from
                     accumulating when a document is re-chunked.
        clean_first: Delete existing points for `source` before indexing.
                     Only takes effect when `source` is also given.

    Returns:
        {
            "indexed":  int,         # points upserted
            "skipped":  int,         # empty-content chunks skipped
            "collection": str,
            "by_type":  dict,        # {"text": n, "table": n, "image": n}
        }
    """
    # ── Delete stale chunks for this source ───────────────────────────────────
    if source and clean_first:
        logger.info(f"  Cleaning stale chunks for source='{source}' …")
        delete_by_source(client, source, collection)

    # ── Filter out empty-content chunks ───────────────────────────────────────
    embeddable = [c for c in chunks if c.get("content", "").strip()]
    skipped    = len(chunks) - len(embeddable)

    if skipped:
        logger.debug(f"  Skipping {skipped} chunk(s) with empty content.")

    if not embeddable:
        logger.warning("  No embeddable chunks — nothing indexed.")
        return {"indexed": 0, "skipped": skipped, "collection": collection,
                "by_type": {}}

    # ── Normalize content for embedding (strip markdown syntax) ───────────────
    texts = []
    for c in embeddable:
        raw   = c["content"]
        clean = normalize_for_embedding(raw)

        # Warn if the normalised text is still very long (silent truncation risk)
        estimated_tokens = len(clean) // 4
        if estimated_tokens > _MAX_SAFE_TOKENS:
            logger.warning(
                f"  Chunk '{c.get('chunk_id', '?')}' is ~{estimated_tokens} tokens "
                f"(limit {_MAX_SAFE_TOKENS}). BGE-M3 will silently truncate it."
            )

        texts.append(clean if clean else raw)   # fallback to raw if normalization empties it

    # ── BM25 sparse vectors (computed over raw content) ───────────────────────
    # BM25 runs on the original text — not the markdown-stripped version —
    # so domain terminology and heading words are preserved for lexical search.
    raw_contents = [c["content"] for c in embeddable]
    all_sparse: list[dict[int, float]] = [bm25_encode(raw) for raw in raw_contents]
    logger.debug(f"  BM25 encoded {len(all_sparse)} chunk(s).")

    # Update the persisted corpus avg_dl so future BM25 calls use the true
    # corpus average instead of the hardcoded prior (100 tokens).
    update_corpus_avg_dl(raw_contents)

    # ── BGE-M3 dense vectors (embed in batches) ───────────────────────────────
    all_dense: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        logger.debug(
            f"  Embedding batch {i // batch_size + 1} "
            f"({len(batch_texts)} texts) …"
        )
        emb = encode(batch_texts, batch_size=batch_size)
        all_dense.extend(emb.dense)

    # ── Build PointStructs ────────────────────────────────────────────────────
    points   = []
    by_type: dict[str, int] = {}

    for chunk, dense_vec, sparse_vec in zip(embeddable, all_dense, all_sparse):
        ctype = chunk.get("type", "text")
        by_type[ctype] = by_type.get(ctype, 0) + 1

        payload = {
            "chunk_id": chunk["chunk_id"],
            "type":     ctype,
            "content":  chunk["content"],   # original content (not normalized)
            **chunk.get("metadata", {}),
        }
        points.append(build_point(chunk["chunk_id"], dense_vec, sparse_vec, payload))

    # ── Upsert ────────────────────────────────────────────────────────────────
    total = upsert(client, points, collection=collection)
    logger.info(
        f"  Indexed {total} point(s) into '{collection}' "
        f"({', '.join(f'{n} {t}' for t, n in by_type.items())})."
    )

    return {
        "indexed":    total,
        "skipped":    skipped,
        "collection": collection,
        "by_type":    by_type,
    }


# ── index_document ────────────────────────────────────────────────────────────

def index_document(
    path:            Path | str,
    client:          Any,
    collection:      str         = COLLECTION_NAME,
    *,
    caption:         bool        = True,
    caption_backend: str         = "groq",
    pdf_pass:        str | None  = None,
    batch_size:      int         = _EMBED_BATCH,
    ensure:          bool        = True,
    clean_first:     bool        = True,
) -> dict[str, Any]:
    """
    Full pipeline: parse → chunk → embed → upsert.

    Stale chunks from a previous indexing run are automatically deleted before
    upserting (controlled by clean_first).

    Args:
        path:            Path to the document file.
        client:          Active QdrantClient.
        collection:      Target collection name (created if ensure=True).
        caption:         Enable image captioning (default True — Groq).
        caption_backend: "groq" or "llava".
        pdf_pass:        Password for encrypted PDFs.
        batch_size:      Texts per embedding batch.
        ensure:          Create the collection if it doesn't exist.
        clean_first:     Delete existing points for this source before indexing.

    Returns:
        {
            "source":       str,
            "total_chunks": int,
            "indexed":      int,
            "skipped":      int,
            "collection":   str,
            "by_type":      { "text": int, "table": int, "image": int },
        }
    """
    path = Path(path)
    logger.info(f"  Indexing document: {path.name}")

    if ensure:
        ensure_collection(client, collection)

    result = process_file(
        path,
        caption=caption,
        caption_backend=caption_backend,
        pdf_pass=pdf_pass,
    )

    doc    = chunk_document(result)
    chunks = doc.get("chunks", [])
    logger.info(f"  {len(chunks)} chunk(s) produced from '{path.name}'.")

    stats = index_chunks(
        chunks,
        client,
        collection=collection,
        batch_size=batch_size,
        source=path.name,
        clean_first=clean_first,
    )

    return {
        "source":       path.name,
        "total_chunks": len(chunks),
        "indexed":      stats["indexed"],
        "skipped":      stats["skipped"],
        "collection":   collection,
        "by_type":      stats.get("by_type", {}),
    }
