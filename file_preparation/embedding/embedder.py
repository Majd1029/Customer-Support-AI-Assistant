"""
embedder.py — BGE-M3 embedding model singleton.

Produces dense (1024-dim) + sparse (SPLADE lexical) vectors for each text.
ColBERT multi-vector output is intentionally disabled — too expensive to store.

Usage:
    from embedder import encode, encode_query

    dense_vecs, sparse_vecs = encode(["Hello world", "Foo bar"])
    d_q, s_q = encode_query("What is the capital of France?")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

from loguru import logger

# ── Model constants ───────────────────────────────────────────────────────────
_MODEL_NAME        = "BAAI/bge-m3"
_DENSE_DIM         = 1024
_BATCH_SIZE_CPU    = 8    # safe for CPU inference
_BATCH_SIZE_GPU    = 32   # safe for GPU inference (raise to 64 with ≥16 GB VRAM)

# ── Singleton ─────────────────────────────────────────────────────────────────
_model      = None
_device     = None   # resolved on first load: "cuda", "mps", or "cpu"


class Embeddings(NamedTuple):
    """Output of encode() — parallel lists, one entry per input text."""
    dense:  list[list[float]]           # shape [N, 1024]
    sparse: list[dict[int, float]]      # SPLADE weights: token_id → weight


def _detect_device() -> str:
    """Return the best available device: 'cuda', 'mps', or 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _default_batch_size() -> int:
    """Return the recommended batch size for the detected device."""
    return _BATCH_SIZE_GPU if _detect_device() != "cpu" else _BATCH_SIZE_CPU


def _load_model():
    """Lazy-load the BGE-M3 model (downloads on first call, ~2 GB)."""
    global _model, _device
    if _model is not None:
        return _model

    try:
        from FlagEmbedding import BGEM3FlagModel   # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "FlagEmbedding is required for BGE-M3 embeddings. "
            "Install it with:  pip install FlagEmbedding"
        ) from exc

    _device = _detect_device()
    _use_fp16 = _device != "cpu"   # FP16 on CPU is emulated (slower); only enable on CUDA/MPS
    logger.info(f"  Loading BGE-M3 model ({_MODEL_NAME}) on {_device.upper()} (fp16={_use_fp16}) …")

    # Suppress the HuggingFace "use __call__ instead of encode+pad" performance
    # hint — it originates inside FlagEmbedding's tokenisation loop and is not
    # actionable from our code.
    import transformers
    transformers.logging.set_verbosity_error()

    _model = BGEM3FlagModel(
        _MODEL_NAME,
        use_fp16=_use_fp16,
    )

    # Restore normal verbosity so other HF warnings (e.g. weight mismatches)
    # are not silenced globally.
    transformers.logging.set_verbosity_warning()
    logger.info(f"  BGE-M3 model ready (device={_device}, default batch_size={_default_batch_size()}).")
    return _model


# ── Public API ────────────────────────────────────────────────────────────────

def encode(texts: list[str], batch_size: int = 0) -> Embeddings:
    """
    Encode a list of passage texts.

    Returns an Embeddings namedtuple with:
      .dense  — list of 1024-dim float lists
      .sparse — list of dicts mapping int token-id → float SPLADE weight

    Args:
        texts:      Input strings (typically chunk["content"]).
        batch_size: Number of texts per inference batch (0 = auto: 8 on CPU, 32 on GPU).
    """
    if not texts:
        return Embeddings(dense=[], sparse=[])

    model = _load_model()
    effective_batch = batch_size if batch_size > 0 else _default_batch_size()

    output = model.encode(
        texts,
        batch_size=effective_batch,
        max_length=8192,            # BGE-M3 supports up to 8192 tokens
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,  # disabled — expensive to store
    )

    dense_list:  list[list[float]]      = [v.tolist() for v in output["dense_vecs"]]
    sparse_list: list[dict[int, float]] = []

    for entry in output["lexical_weights"]:
        # entry is a dict of {token_id_str: weight} or {int: weight}
        sparse_list.append({int(k): float(v) for k, v in entry.items()})

    return Embeddings(dense=dense_list, sparse=sparse_list)


def encode_query(query: str, batch_size: int = 1) -> Embeddings:
    """
    Encode a single query string.

    BGE-M3 is symmetric — no special query prefix is needed (unlike E5 models).
    Returns a single-element Embeddings namedtuple.

    Args:
        query: The user's search query.
    """
    return encode([query], batch_size=batch_size)


def dense_dim() -> int:
    """Return the dense vector dimension (1024 for BGE-M3)."""
    return _DENSE_DIM
