"""
file_preparation/retrieval — structured retrieval layer for the RAG pipeline.

Hybrid retrieval stack
──────────────────────
  Dense   — BGE-M3 1024-dim cosine vectors (semantic similarity)
  Sparse  — BM25 TF weights via bm25_encoder (lexical matching;
            IDF applied server-side by Qdrant when Modifier.IDF is available)
  Fusion  — Reciprocal Rank Fusion (RRF) → top-K candidates

Public exports
──────────────
    bm25_encode           — BM25 TF sparse vector for a document chunk
    bm25_encode_query     — BM25 TF sparse vector for a search query
    tokenize              — shared Unicode word tokeniser

    retrieve              — hybrid RRF search (raw dicts); HyDE / rerank / MMR / decompose
    SourceFilter          — typed source / language / type filter
    RetrievedChunk        — a single retrieved chunk with score, metadata, neighbors
    RetrievalResult       — full output of a retrieval call (chunks + timing + hops)
    retrieve_evidence     — single-hop retrieval with context expansion
    multihop_retrieve     — two-hop retrieval that follows entity references
    score_answer_confidence — Groq-based answer groundedness score (0.0–1.0)

    rerank                — cross-encoder second stage (jinaai/jina-reranker-v2-base-multilingual)
    reranker_available    — True if sentence-transformers is installed
"""

from .bm25_encoder import bm25_encode, bm25_encode_query, tokenize, update_corpus_avg_dl
from .retriever import (
    retrieve,
    SourceFilter,
    RetrievedChunk,
    RetrievalResult,
    retrieve_evidence,
    multihop_retrieve,
    score_answer_confidence,
)
from .reranker import rerank, reranker_available

__all__ = [
    # BM25 encoder
    "bm25_encode",
    "bm25_encode_query",
    "tokenize",
    "update_corpus_avg_dl",
    # Core retrieval
    "retrieve",
    "SourceFilter",
    "RetrievedChunk",
    "RetrievalResult",
    "retrieve_evidence",
    "multihop_retrieve",
    "score_answer_confidence",
    # Reranker
    "rerank",
    "reranker_available",
]
