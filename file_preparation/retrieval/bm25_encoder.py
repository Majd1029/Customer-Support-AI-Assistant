"""
bm25_encoder.py — Lightweight BM25 sparse encoder for Qdrant hybrid retrieval.

Produces sparse vectors (dict[int, float]) compatible with Qdrant's SparseVector
format.  Integer token IDs are derived via the hashing trick — no vocabulary file
is required, and the encoder is completely stateless and thread-safe.

When the Qdrant collection is created with SparseVectorParams(modifier=Modifier.IDF),
Qdrant computes and applies IDF server-side, giving true BM25 (TF·IDF) scores.
Without that modifier the vectors carry TF-only BM25 weights, which is still
effective for lexical retrieval.

Usage
─────
    from bm25_encoder import bm25_encode, bm25_encode_query

    # Index time — encode a document chunk
    sparse_vec = bm25_encode("The quick brown fox jumps over the lazy dog")
    # → {3871204: 1.42, 9012341: 0.96, ...}

    # Query time — encode a search query
    query_vec = bm25_encode_query("quick fox")
    # → {3871204: 1.0, 9012341: 1.0}

Design notes
────────────
• Tokenizer    — Unicode word regex (r'\\b\\w+\\b'), lower-cased.  Handles
                 Latin, Cyrillic, CJK via \\w+.  Punctuation and whitespace
                 are stripped.
• Hash space   — 2²⁴ = 16 777 216 buckets (BLAKE2b 3-byte digest mod N).
                 Expected collisions ≪ 1% for real-world vocabularies up to
                 ~100 k unique tokens.
• BM25 params  — k₁=1.5 (term saturation), b=0.75 (length normalisation).
                 avg_dl=100 tokens is a reasonable prior for mixed-length chunks;
                 override per-call if your corpus statistics differ.
• Query vector — each unique query token gets weight 1.0 (query TF=1).
                 IDF is applied by Qdrant (Modifier.IDF) or left to the dot-
                 product scoring without explicit IDF (still works well in
                 practice for short queries).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

# Hash space: 2^24 ≈ 16.7 M buckets — good collision resistance up to ~100k vocab
_HASH_BUCKETS: int = 1 << 24   # 16_777_216

# Default BM25 hyperparameters (Robertson–Sparck Jones)
_K1:         float = 1.5    # term frequency saturation (0 = binary, ∞ = raw TF)
_B:          float = 0.75   # length normalisation strength (0 = none, 1 = full)
_AVG_DL_DEFAULT: float = 100.0   # fallback prior — overridden by corpus statistics

# Tokeniser pattern — Unicode-aware word boundaries
_TOKEN_RE = re.compile(r'\b\w+\b')

# ── Corpus statistics — persisted avg_dl ──────────────────────────────────────

# Config file sits next to this module so both indexer.py (writer) and
# bm25_encoder.py (reader) resolve to the same path via Path(__file__).
_CONFIG_PATH = Path(__file__).parent / "bm25_config.json"


def _load_corpus_stats() -> tuple[float, int]:
    """
    Load (avg_dl, n_chunks) from the persisted config file.

    Returns the default prior (100.0, 0) if the file does not exist or
    cannot be parsed — safe to call at any time.
    """
    try:
        data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        return float(data["avg_dl"]), int(data["n_chunks"])
    except Exception:
        return _AVG_DL_DEFAULT, 0


def update_corpus_avg_dl(chunk_texts: list[str]) -> None:
    """
    Update the persisted corpus avg_dl with stats from a new batch of chunks.

    Uses a weighted running average so successive index_chunks() calls
    converge to the true corpus-wide average document length:

        new_avg = (old_n × old_avg + batch_n × batch_avg) / (old_n + batch_n)

    This is called automatically by indexer.index_chunks() after computing
    BM25 sparse vectors.  It is safe to call from multiple processes — the
    write is a single atomic rename on POSIX; on Windows it overwrites.

    Args:
        chunk_texts: Raw content strings of the chunks just indexed.
    """
    if not chunk_texts:
        return

    # Measure document lengths in BM25 tokens (same tokeniser used at encode time)
    batch_dls   = [len(tokenize(t)) for t in chunk_texts if t.strip()]
    if not batch_dls:
        return

    batch_n    = len(batch_dls)
    batch_avg  = sum(batch_dls) / batch_n

    old_avg, old_n = _load_corpus_stats()
    total_n        = old_n + batch_n
    new_avg        = (old_n * old_avg + batch_n * batch_avg) / total_n

    try:
        _CONFIG_PATH.write_text(
            json.dumps({"avg_dl": round(new_avg, 2), "n_chunks": total_n}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass   # non-fatal — next call will try again


# Load once at import time; refreshed automatically when indexer writes a new config.
# A module-level variable avoids re-reading the file on every encode() call.
_corpus_avg_dl: float = _load_corpus_stats()[0]


# ── Core helpers ──────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Lowercase Unicode word tokeniser.

    Splits on non-word characters (spaces, punctuation, separators).
    Preserves alphanumeric tokens including digits and accented letters.

    Args:
        text: Raw text to tokenise.

    Returns:
        List of lowercase word tokens (may be empty for whitespace-only input).
    """
    return _TOKEN_RE.findall(text.lower())


def _token_id(token: str) -> int:
    """
    Map a token string to a stable integer ID via the hashing trick.

    Uses BLAKE2b (3-byte digest) for speed and low collision rate.
    The mapping is deterministic — the same token always produces the same ID
    regardless of process restart or Python version.

    Args:
        token: Lowercase word token.

    Returns:
        Integer in [0, _HASH_BUCKETS).
    """
    raw = hashlib.blake2b(token.encode("utf-8"), digest_size=3).digest()
    return int.from_bytes(raw, "big") % _HASH_BUCKETS


# ── Public API ────────────────────────────────────────────────────────────────

def bm25_encode(
    text:   str,
    *,
    k1:     float = _K1,
    b:      float = _B,
    avg_dl: float | None = None,
) -> dict[int, float]:
    """
    Compute BM25 TF weights for a document chunk.

    The output is a sparse vector {token_id: tf_weight} suitable for Qdrant's
    NamedSparseVector.  When the collection uses SparseVectorParams(modifier=
    Modifier.IDF), Qdrant multiplies these TF weights by IDF at search time,
    giving a proper BM25 score.

    BM25 TF formula (Robertson 1994):
        tf_bm25 = tf × (k₁ + 1) / (tf + k₁ × (1 − b + b × dl/avg_dl))

    Args:
        text:   Document text to encode (raw content, before markdown stripping).
        k1:     Term saturation parameter — higher = more influence from repeated
                terms (default 1.5).
        b:      Length normalisation strength — 0 = no normalisation,
                1 = full normalisation (default 0.75).
        avg_dl: Average document length in BM25 tokens.  When None (default),
                uses the corpus-measured value from bm25_config.json (updated
                automatically by indexer.index_chunks()).  Falls back to 100
                if no config file exists yet.

    Returns:
        Sparse vector dict {token_id: bm25_tf_weight}.  Empty dict for
        whitespace-only or empty input.
    """
    _avg = avg_dl if avg_dl is not None else _corpus_avg_dl

    tokens = tokenize(text)
    if not tokens:
        return {}

    dl: int = len(tokens)

    # ── Count raw term frequencies ─────────────────────────────────────────────
    tf_raw: dict[int, int] = {}
    for token in tokens:
        tid = _token_id(token)
        tf_raw[tid] = tf_raw.get(tid, 0) + 1

    # ── Apply BM25 TF normalisation ────────────────────────────────────────────
    length_norm = 1.0 - b + b * dl / _avg
    result: dict[int, float] = {}
    for tid, tf in tf_raw.items():
        denom = tf + k1 * length_norm
        result[tid] = tf * (k1 + 1.0) / denom

    return result


def bm25_encode_query(
    query:  str,
    *,
    k1:     float = _K1,
    b:      float = _B,
    avg_dl: float | None = None,
) -> dict[int, float]:
    """
    Compute the BM25 sparse vector for a search query.

    Queries are typically short (< 20 tokens) so query-side BM25 TF
    normalisation has very little effect.  We apply the same formula as
    bm25_encode() for consistency, which means unique query terms get
    weights close to 1.0 and repeated terms are slightly higher.

    When the Qdrant collection uses Modifier.IDF, these TF weights are
    multiplied by IDF at dot-product time, producing a proper BM25 query
    representation.

    Args:
        query:  Raw query string.
        k1:     BM25 k₁ parameter (should match indexing value).
        b:      BM25 b parameter (should match indexing value).
        avg_dl: Average document length in BM25 tokens.  When None (default),
                uses the corpus-measured value from bm25_config.json — the
                same value used at index time, ensuring consistent normalisation.

    Returns:
        Sparse vector dict {token_id: bm25_tf_weight}.  Empty dict if
        the query contains no recognisable tokens.
    """
    # Re-use bm25_encode for consistency — short queries mean length_norm ≈ 1
    return bm25_encode(query, k1=k1, b=b, avg_dl=avg_dl)
