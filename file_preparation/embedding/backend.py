"""
backend.py — selects the embedding implementation at import time.

    EMBEDDING_BACKEND=local   (default) → embedder.py: BGE-M3 loaded in-process
                                          via FlagEmbedding (needs ~3-4 GB RAM)
    EMBEDDING_BACKEND=hf_api            → remote_embedder.py: same model served by
                                          the Hugging Face Inference API

Both expose encode / encode_query / dense_dim / Embeddings with identical
signatures, so callers import from here and never need to know which one runs.
"""

from __future__ import annotations

import os

from loguru import logger

from file_preparation.embedding.embedder import Embeddings

BACKEND = os.getenv("EMBEDDING_BACKEND", "local").strip().lower()

if BACKEND in ("hf_api", "api", "remote"):
    from file_preparation.embedding.remote_embedder import dense_dim, encode, encode_query
    logger.info("[EMB] Using remote BGE-M3 embeddings (Hugging Face Inference API).")
else:
    from file_preparation.embedding.embedder import dense_dim, encode, encode_query

__all__ = ["BACKEND", "Embeddings", "dense_dim", "encode", "encode_query"]
