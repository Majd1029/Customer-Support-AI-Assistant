"""
file_preparation/embedding — BGE-M3 embedding model.

Public API
----------
encode(texts, batch_size)   → Embeddings(dense, sparse)
encode_query(query)         → Embeddings(dense, sparse)
dense_dim()                 → 1024
Embeddings                  → NamedTuple(dense, sparse)
"""

from .backend import encode, encode_query, dense_dim, Embeddings  # noqa: E402  (EMBEDDING_BACKEND selects local/API)

__all__ = [
    "encode",
    "encode_query",
    "dense_dim",
    "Embeddings",
]
