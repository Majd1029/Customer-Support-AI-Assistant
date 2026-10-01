"""
remote_embedder.py — BGE-M3 dense embeddings via the Hugging Face Inference API.

Drop-in alternative to embedder.py (same public API: encode, encode_query,
dense_dim, Embeddings) for hosts that cannot load the ~2.3 GB model locally
(e.g. a 512 MB free-tier server). Selected with EMBEDDING_BACKEND=hf_api —
see file_preparation/embedding/backend.py.

Dense vectors come from the same model (BAAI/bge-m3, CLS pooling + L2
normalisation), so they are interchangeable with the local backend's
dense vectors. BGE-M3's SPLADE-style sparse weights are not exposed by the
API; `.sparse` is filled with BM25 term weights instead (only semantic
memory uses it — document search already uses BM25 for its sparse side).

Environment:
    HF_TOKEN               Hugging Face access token with "Inference Providers"
                           permission (required)
    EMBEDDING_API_MODEL    default "BAAI/bge-m3"
    EMBEDDING_API_URL      full endpoint override (any TEI-compatible
                           /feature-extraction endpoint)
    EMBEDDING_API_BATCH    texts per request (default 16)
    EMBEDDING_API_TIMEOUT  seconds per request (default 60)
"""

from __future__ import annotations

import math
import os
import time

import httpx
from loguru import logger

from file_preparation.embedding.embedder import Embeddings

_DENSE_DIM = 1024

_MODEL     = os.getenv("EMBEDDING_API_MODEL", "BAAI/bge-m3")
_API_URL   = os.getenv(
    "EMBEDDING_API_URL",
    f"https://router.huggingface.co/hf-inference/models/{_MODEL}/pipeline/feature-extraction",
)
_BATCH     = int(os.getenv("EMBEDDING_API_BATCH", "16"))
_TIMEOUT   = float(os.getenv("EMBEDDING_API_TIMEOUT", "60"))
_RETRIES   = 4   # 429 / 5xx / cold-start 503 → exponential backoff (2, 4, 8, 16 s)

_client: httpx.Client | None = None


def _token() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or ""
    if not token:
        raise RuntimeError(
            "EMBEDDING_BACKEND=hf_api requires HF_TOKEN (a Hugging Face token with "
            "'Inference Providers' permission)."
        )
    return token


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=_TIMEOUT)
    return _client


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


def _post_batch(texts: list[str]) -> list[list[float]]:
    """Embed one batch, retrying on rate limits, cold starts and server errors."""
    payload = {"inputs": texts, "normalize": True, "truncate": True}
    headers = {"Authorization": f"Bearer {_token()}"}

    for attempt in range(_RETRIES + 1):
        try:
            resp = _get_client().post(_API_URL, json=payload, headers=headers)
        except httpx.TransportError as exc:
            if attempt == _RETRIES:
                raise RuntimeError(f"Embedding API unreachable: {exc}") from exc
            wait = 2 ** (attempt + 1)
            logger.warning(f"  [EMB-API] transport error ({exc}); retrying in {wait}s …")
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            data = resp.json()
            # A single string returns one vector; a list returns a list of vectors.
            if data and isinstance(data[0], (int, float)):
                data = [data]
            if len(data) != len(texts):
                raise RuntimeError(
                    f"Embedding API returned {len(data)} vectors for {len(texts)} inputs."
                )
            return [_l2_normalize([float(x) for x in v]) for v in data]

        if resp.status_code in (429, 500, 502, 503, 504) and attempt < _RETRIES:
            wait = 2 ** (attempt + 1)
            logger.warning(
                f"  [EMB-API] HTTP {resp.status_code}; retrying in {wait}s "
                f"(attempt {attempt + 1}/{_RETRIES}) …"
            )
            time.sleep(wait)
            continue

        raise RuntimeError(f"Embedding API error HTTP {resp.status_code}: {resp.text[:300]}")

    raise RuntimeError("Embedding API: retries exhausted")  # pragma: no cover


def _sparse_for(texts: list[str]) -> list[dict[int, float]]:
    from file_preparation.retrieval.bm25_encoder import bm25_encode
    return [bm25_encode(t) for t in texts]


# ── Public API (mirrors embedder.py) ─────────────────────────────────────────

def encode(texts: list[str], batch_size: int = 0) -> Embeddings:
    """Encode passage texts. `.dense` = 1024-dim BGE-M3 vectors, `.sparse` = BM25 weights."""
    if not texts:
        return Embeddings(dense=[], sparse=[])

    step  = batch_size if batch_size > 0 else _BATCH
    dense: list[list[float]] = []
    for i in range(0, len(texts), step):
        dense.extend(_post_batch(texts[i : i + step]))

    return Embeddings(dense=dense, sparse=_sparse_for(texts))


def encode_query(query: str, batch_size: int = 1) -> Embeddings:
    """Encode a single query string (BGE-M3 needs no query prefix)."""
    return encode([query], batch_size=batch_size)


def dense_dim() -> int:
    """Return the dense vector dimension (1024 for BGE-M3)."""
    return _DENSE_DIM
