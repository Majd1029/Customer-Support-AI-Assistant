"""
retriever.py — structured retrieval layer for the RAG pipeline.

Wraps the lower-level hybrid-search / neighbor-expansion functions from
file_preparation/indexing/store.py into a typed, pipeline-friendly API and adds
higher-level capabilities:

  1. retrieve()           — hybrid RRF search with HyDE / reranking / MMR /
                            query decomposition (moved here from indexer.py so that
                            ALL retrieval logic lives in a single folder).

  2. retrieve_evidence()  — top-K chunks with context-expansion neighbors,
                            returned as typed RetrievedChunk objects.

  3. multihop_retrieve()  — two-hop retrieval that follows entity references
                            across chunks via Groq-generated follow-up queries.

  4. score_answer_confidence() — Groq rates how well a generated answer is
                                 grounded in the retrieved evidence (0.0–1.0).

Public API
──────────
    from file_preparation.retrieval import (
        retrieve,
        retrieve_evidence,
        multihop_retrieve,
        score_answer_confidence,
        SourceFilter,
        RetrievedChunk,
        RetrievalResult,
    )

    # Typed source / language / type filtering
    sf = SourceFilter(sources=["q3_report.pdf"], languages=["en"])

    # Low-level hybrid search (returns raw dicts)
    hits = retrieve("What are the key findings?", client, limit=10, rerank=True)

    # High-level retrieval with context expansion (returns RetrievalResult)
    result = retrieve_evidence(
        "What were the key findings?", client, source_filter=sf, limit=5
    )
    for chunk in result.chunks:
        print(chunk.score, chunk.content[:80])
        for nb in chunk.neighbors:
            print("  neighbor:", nb.get("content", "")[:60])

    # Two-hop retrieval — follows entity references across chunks
    result2 = multihop_retrieve("What caused the revenue drop?", client)
    print(f"Hops: {result2.hops}, follow-up queries: {result2.hop2_queries}")

    # Score how well a generated answer is grounded in the evidence
    confidence = score_answer_confidence(answer_text, result.chunks)
    print(f"Confidence: {confidence:.2f}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from file_preparation.embedding.embedder     import encode_query                      # type: ignore[import]
from file_preparation.retrieval.bm25_encoder import bm25_encode_query                 # type: ignore[import]
from file_preparation.indexing.store         import (                                  # type: ignore[import]
    search,
    get_neighbors,
    COLLECTION_NAME,
    DENSE_VECTOR,
)

# Reranker is optional — imported lazily to avoid a 1 GB model download at
# import time.  Only materialised when rerank=True is passed to retrieve().
try:
    from file_preparation.retrieval.reranker import rerank as _rerank, reranker_available as _reranker_available  # type: ignore[import]
    _RERANKER_IMPORTABLE = True
except ImportError:
    _RERANKER_IMPORTABLE = False

# numpy is used by _cosine_similarity for fast dot-product in _mmr.
try:
    import numpy as _np
    _NUMPY_AVAILABLE = True
except ImportError:
    _np = None          # type: ignore[assignment]
    _NUMPY_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

# Model used for HyDE expansion, query decomposition, multi-hop, and confidence
_MODEL            = "meta-llama/llama-4-scout-17b-16e-instruct"
_HYDE_MAX_TOKENS  = 256   # keep it short — just enough to anchor the embedding
_MULTIHOP_MAX_TOKENS = 200   # raised from 120 — 120 could truncate two verbose queries
_CONFIDENCE_MAX_TOKENS = 10   # we only need a single decimal number


# ── Groq singleton ────────────────────────────────────────────────────────────

_groq_client = None


def _get_groq_client():
    """Lazy singleton — reads GROQ_MEMORY_API_KEY (or GROQ_API_KEY) from env / .env file."""
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv(
            dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env",
            override=False,
        )
        from groq import Groq  # type: ignore[import]
        # GROQ_MEMORY_API_KEY is the dedicated key for retrieval/memory calls.
        # Falls back to GROQ_API_KEY for backwards compatibility.
        api_key = os.environ.get("GROQ_MEMORY_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.debug("  [retriever] GROQ_API_KEY not set — Groq features disabled.")
            return None
        _groq_client = Groq(api_key=api_key)
        logger.debug("  [retriever] Groq client initialised (singleton).")
    except Exception as e:
        logger.warning(f"  [retriever] Groq client init failed: {e}")
        return None
    return _groq_client


# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    A single retrieved chunk with its relevance score, metadata, and any
    context-expanded neighbor chunks fetched from the same document.
    """
    chunk_id:  str
    content:   str
    score:     float
    metadata:  dict
    neighbors: list[dict] = field(default_factory=list)
    hop:       int        = 1   # 1 = first-hop result, 2 = multi-hop follow-up


@dataclass
class RetrievalResult:
    """
    The full output of a retrieval call.

    Attributes:
        query:        The original query string.
        chunks:       Ranked list of RetrievedChunk objects (hop-1 first, then hop-2).
        total:        Total number of chunks returned.
        elapsed_ms:   Wall-clock time for the entire retrieval call (ms).
        hops:         1 for standard retrieval, 2 for multi-hop.
        sub_queries:  Sub-queries generated when decompose=True.
        hop2_queries: Follow-up queries generated during multi-hop retrieval.
    """
    query:        str
    chunks:       list[RetrievedChunk]
    total:        int
    elapsed_ms:   float
    hops:         int       = 1
    sub_queries:  list[str] = field(default_factory=list)
    hop2_queries: list[str] = field(default_factory=list)


@dataclass
class SourceFilter:
    """
    Typed metadata filter for retrieval calls.

    All fields are optional — only non-None fields are applied as filters.
    A list with multiple values → any-of / OR match inside that field.
    All specified fields are ANDed together.

    Examples:
        SourceFilter(sources=["report.pdf"])
        SourceFilter(languages=["en", "ar"])
        SourceFilter(types=["text", "table"])
        SourceFilter(sources=["q3.pdf", "q4.pdf"], languages=["en"])
    """
    sources:    list[str] | None = None   # match by source filename
    languages:  list[str] | None = None   # match by ISO-639-1 language code
    types:      list[str] | None = None   # match by chunk type: text / table / image

    def to_filters(self, base: dict | None = None) -> dict | None:
        """
        Merge into the flat filters dict accepted by retrieve() / store.search().

        Args:
            base: Existing filters dict to merge into (not mutated).

        Returns:
            Merged filters dict, or None if all fields are empty.
        """
        f: dict[str, Any] = dict(base) if base else {}
        if self.sources:
            f["source"]   = self.sources
        if self.languages:
            f["language"] = self.languages
        if self.types:
            f["type"]     = self.types
        return f or None


# ── HyDE query expansion ──────────────────────────────────────────────────────

def _hyde_expand(query: str) -> str:
    """
    Hypothetical Document Embeddings (HyDE) query expansion.

    Calls Groq to generate a short hypothetical passage that would answer the
    query.  Embedding this passage rather than the raw query string dramatically
    improves recall for short or keyword-only queries — the model bridges the
    vocabulary gap between the query and the documents.

    Falls back to the original query string if Groq is unavailable or the
    call fails, so the pipeline degrades gracefully.

    Args:
        query: The user's search query.

    Returns:
        A hypothetical passage (str) or the original query on failure.
    """
    try:
        client = _get_groq_client()
        if client is None:
            logger.debug("  HyDE: Groq unavailable — using raw query.")
            return query

        system_prompt = (
            "You are a search engine assistant. "
            "Given a user query, write a short factual passage (2-4 sentences) "
            "that would directly answer the query. "
            "Write only the passage — no preamble, no explanation."
        )
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": query},
            ],
            max_tokens=_HYDE_MAX_TOKENS,
            temperature=0.1,   # very low temp = deterministic, factual hypothesis
        )
        hypothesis = response.choices[0].message.content.strip()
        logger.debug(f"  HyDE hypothesis ({len(hypothesis)} chars): {hypothesis[:120]} …")
        return hypothesis

    except Exception as e:
        logger.warning(f"  HyDE expansion failed ({e}) — falling back to raw query.")
        return query


# ── MMR diversification helpers ───────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Fast dot-product cosine similarity (vectors assumed L2-normalised).

    Uses numpy when available; falls back to a pure-Python loop otherwise.
    BGE-M3 dense vectors are L2-normalised, so dot product equals cosine
    similarity with no additional division needed.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    if _np is not None:
        return float(_np.dot(
            _np.asarray(a, dtype=_np.float32),
            _np.asarray(b, dtype=_np.float32),
        ))
    return float(sum(x * y for x, y in zip(a, b)))


def _mmr(
    results:    list[dict],
    limit:      int,
    mmr_lambda: float = 0.5,
) -> list[dict]:
    """
    Maximal Marginal Relevance diversification.

    Selects `limit` results that balance relevance (high score) and
    diversity (low similarity to already-selected results).

    The tradeoff is controlled by mmr_lambda:
        1.0  → pure relevance (same as no MMR)
        0.5  → balanced (default)
        0.0  → pure diversity

    Each result must have a "dense_vector" field (fetched with with_vectors=True).
    Results without vectors fall back to score-only selection.

    Args:
        results:    Candidates from hybrid search, sorted by score descending.
                    The caller's list and its dicts are NOT mutated.
        limit:      How many results to return.
        mmr_lambda: Relevance vs diversity tradeoff (0.0 – 1.0).

    Returns:
        `limit` results re-ordered by MMR score (new list, caller's dicts intact).
    """
    if not results or limit <= 0:
        return results[:limit]

    # ── Uniform-score guard ───────────────────────────────────────────────────
    # When all candidate scores are within 0.01 of each other (e.g. after
    # reranking on a tiny candidate set, or when all RRF ranks are equal),
    # MMR's normalised scores collapse to ~1.0 for every candidate and the
    # diversification step degenerates into arbitrary ordering.  Skip MMR and
    # return the top-k in their existing sort order instead.
    max_score = max(r["score"] for r in results)
    min_score = min(r["score"] for r in results)
    if (max_score - min_score) < 0.01:
        logger.debug(
            f"MMR: scores uniform (range={max_score - min_score:.4f}) — "
            "skipping diversification, returning top-k by score."
        )
        return results[:limit]

    # Avoid division by zero in normalisation
    if max_score == 0.0:
        max_score = 1.0

    # Work on a local list of (norm_score, original_dict) tuples so we never
    # write into the caller's dicts.  An exception at any point leaves the
    # caller's results completely unchanged.
    candidates: list[tuple[float, dict]] = [
        (r["score"] / max_score, r) for r in results
    ]

    selected_norm_scores: list[float] = []
    selected_dicts:       list[dict]  = []
    selected_vecs:        list[list[float]] = []

    while candidates and len(selected_dicts) < limit:
        if not selected_dicts:
            # First pick: highest relevance score — guaranteed to be index 0
            # since results arrive sorted by score descending.
            best_idx = 0
        else:
            best_idx = -1
            best_mmr = float("-inf")

            for i, (norm_score, cand) in enumerate(candidates):
                cand_vec = cand.get("dense_vector", [])
                if cand_vec and selected_vecs:
                    max_sim = max(
                        _cosine_similarity(cand_vec, sv)
                        for sv in selected_vecs
                        if sv
                    )
                else:
                    max_sim = 0.0

                mmr_score = mmr_lambda * norm_score - (1 - mmr_lambda) * max_sim
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i

        if best_idx == -1:
            break

        norm_score, best_dict = candidates.pop(best_idx)   # O(n) — acceptable for small MMR candidate sets (≤40)
        selected_norm_scores.append(norm_score)
        selected_dicts.append(best_dict)
        selected_vecs.append(best_dict.get("dense_vector", []))

    return selected_dicts


# ── Query decomposition ───────────────────────────────────────────────────────

def _decompose_query(query: str) -> list[str]:
    """
    Query decomposition via Groq.

    Splits a compound or multi-faceted question into a list of simpler
    sub-queries that can each be answered independently.  Running hybrid
    search on each sub-query and merging results improves recall for
    questions that combine multiple topics (e.g. "What are the revenue
    figures and headcount by region?").

    Falls back to [query] (single-element list) if Groq is unavailable
    or the question is already simple enough not to decompose.

    Args:
        query: The original user question.

    Returns:
        List of sub-query strings (at least one element — the original
        query if decomposition is skipped or fails).
    """
    try:
        client = _get_groq_client()
        if client is None:
            logger.debug("  Decompose: Groq unavailable — using original query.")
            return [query]

        system_prompt = (
            "You are a search assistant. Given a user question, break it into "
            "simple, independent sub-questions that together cover the original "
            "question. Output ONLY the sub-questions, one per line, with no "
            "numbering, bullets, or extra text. "
            "If the question is already simple, output it unchanged on a single line."
        )
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": query},
            ],
            max_tokens=200,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        sub_queries = [line.strip() for line in raw.splitlines() if line.strip()]
        if not sub_queries:
            return [query]
        logger.debug(f"  Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
        return sub_queries

    except Exception as e:
        logger.warning(f"  Query decomposition failed ({e}) — using original query.")
        return [query]


# ── Core retrieval ─────────────────────────────────────────────────────────────

def retrieve(
    query:             str,
    client:            Any,
    collection:        str          = COLLECTION_NAME,
    limit:             int          = 10,
    prefetch_k:        int          = 40,
    filters:           dict | None  = None,
    min_score:         float | None = None,
    rerank:            bool         = False,
    rerank_top_k:      int | None   = None,
    use_hyde:          bool         = False,
    mmr:               bool         = False,
    mmr_lambda:        float        = 0.5,
    decompose:         bool         = False,
    _out_sub_queries:  list | None  = None,   # optional mutable output — caller populates RetrievalResult.sub_queries
) -> list[dict[str, Any]]:
    """
    Embed a query and run hybrid (dense + sparse) search via RRF.

    Optional enhancements that can be stacked in any combination:
      • HyDE       — generate a hypothetical document via Groq, embed that
                     instead of the raw query (bridges vocabulary gap).
      • Rerank     — apply cross-encoder (jina-reranker-v2-base-multilingual) on first-stage
                     results for higher precision.
      • MMR        — Maximal Marginal Relevance diversification; re-orders
                     results to balance relevance and diversity so the top-k
                     chunks are not all near-duplicates from the same section.
      • Decompose  — split compound questions into sub-queries via Groq, run
                     each independently, then merge + deduplicate results.

    Args:
        query:         Natural language question or keyword string.
        client:        Active QdrantClient.
        collection:    Target collection name.
        limit:         Maximum results to return (after all post-processing).
        prefetch_k:    Candidate count per sub-query before RRF fusion.
        filters:       Optional equality/any-of filters:
                         {"language": "en"}            → exact match
                         {"type": ["text", "table"]}   → any-of match
        min_score:     Discard results with RRF score below this threshold
                       (applied before reranking / MMR).
        rerank:        Apply cross-encoder reranking. Requires sentence-transformers.
                       Falls back silently to first-stage ranking if unavailable.
        rerank_top_k:  How many reranked results to return. Defaults to `limit`.
        use_hyde:      Expand query via HyDE before embedding.
                       Falls back to raw query if Groq is unavailable.
        mmr:           Apply Maximal Marginal Relevance diversification.
                       Requires dense vectors (fetched automatically when True).
        mmr_lambda:    MMR relevance vs diversity tradeoff (0.0–1.0).
                       1.0 = pure relevance, 0.0 = pure diversity (default 0.5).
        decompose:     Decompose compound query into sub-queries via Groq,
                       retrieve for each, and merge results before post-processing.

    Returns:
        Score-sorted list of { "chunk_id", "score", "payload" } dicts.
        When reranked, each result also has "first_stage_score".
    """
    if not query.strip():
        logger.warning("  retrieve() called with an empty query — returning [].")
        return []

    # ── Query decomposition ───────────────────────────────────────────────────
    if decompose:
        sub_queries = _decompose_query(query)
        # Expose generated sub-queries to the caller so RetrievalResult.sub_queries
        # can be populated.  Using a mutable list avoids changing the return type.
        if _out_sub_queries is not None:
            _out_sub_queries.extend(sub_queries)
    else:
        sub_queries = [query]

    # ── Run retrieval for each sub-query, then merge ──────────────────────────
    # MMR needs vectors; request them upfront if MMR is enabled.
    need_vectors = mmr

    # Track best score per chunk_id across all sub-queries.
    # First-occurrence dedup (seen_ids set) would keep the lowest score from
    # whatever sub-query happened to run first; instead we keep the highest.
    best_by_id: dict[str, dict[str, Any]] = {}

    for sq in sub_queries:
        # HyDE expansion improves dense recall by generating a hypothetical
        # passage to embed.  BM25 must always use the original sub-query —
        # tokenizing a hallucinated passage would match words the user never
        # wrote and miss the exact terms they searched for.
        embed_text = _hyde_expand(sq) if use_hyde else sq
        if use_hyde:
            logger.debug(f"  HyDE dense text ({len(embed_text)} chars): {embed_text[:80]} …")

        logger.debug(f"  Encoding sub-query: {sq[:80]} …")
        emb        = encode_query(embed_text)  # BGE-M3 dense — uses HyDE passage when enabled
        sparse_vec = bm25_encode_query(sq)     # BM25 sparse  — always uses original query terms

        # Cast wider net when reranking so cross-encoder has enough candidates.
        first_stage_limit = max(limit, prefetch_k) if rerank else limit

        hits = search(
            client,
            dense_vec=emb.dense[0],
            sparse_vec=sparse_vec,
            collection=collection,
            limit=first_stage_limit,
            prefetch_k=prefetch_k,
            filters=filters,
            min_score=min_score,
            with_vectors=need_vectors,
        )

        for h in hits:
            cid = h.get("chunk_id", "")
            # Keep the hit with the highest score for this chunk_id across sub-queries.
            if cid not in best_by_id or h["score"] > best_by_id[cid]["score"]:
                best_by_id[cid] = h

    all_results: list[dict[str, Any]] = list(best_by_id.values())

    logger.debug(f"  First-stage retrieved {len(all_results)} unique result(s) "
                 f"across {len(sub_queries)} sub-query(ies).")

    if not all_results:
        return []

    # Re-sort merged results by score descending before post-processing.
    all_results.sort(key=lambda r: r["score"], reverse=True)

    # ── Cross-encoder reranking (second stage) ────────────────────────────────
    if rerank and all_results:
        if _RERANKER_IMPORTABLE and _reranker_available():
            # When MMR follows reranking, keep a larger pool so MMR has enough
            # candidates to diversify from.  Without this, rerank returns exactly
            # `limit` results and MMR has no room to swap in diverse chunks.
            final_k = rerank_top_k if rerank_top_k is not None else (limit * 2 if mmr else limit)
            all_results = _rerank(query, all_results, top_k=final_k)
            logger.debug(f"  Reranked → {len(all_results)} result(s) "
                         f"(pool={'enlarged for MMR' if mmr else 'standard'}).")

            # Re-apply min_score against the cross-encoder score (sigmoid [0,1]).
            # The first-stage RRF score and the cross-encoder score live on
            # completely different scales, so a single threshold can't serve both.
            # Chunks that survived the RRF threshold but are irrelevant per the
            # cross-encoder (score < min_score) are dropped here.
            if min_score is not None:
                before = len(all_results)
                all_results = [r for r in all_results if r["score"] >= min_score]
                if len(all_results) < before:
                    logger.debug(
                        f"  Post-rerank min_score={min_score}: "
                        f"dropped {before - len(all_results)} low-confidence result(s)."
                    )
        else:
            logger.warning(
                "  rerank=True requested but reranker is not available. "
                "Falling back to first-stage RRF ranking. "
                "Install sentence-transformers and download jinaai/jina-reranker-v2-base-multilingual."
            )
            all_results = all_results[:limit]
    elif not rerank:
        all_results = all_results[:limit]

    # ── MMR diversification ───────────────────────────────────────────────────
    if mmr and all_results:
        logger.debug(f"  MMR diversification (lambda={mmr_lambda}) on "
                     f"{len(all_results)} candidates …")
        all_results = _mmr(all_results, limit=limit, mmr_lambda=mmr_lambda)
        logger.debug(f"  MMR → {len(all_results)} result(s).")

    return all_results


# ── retrieve_evidence ──────────────────────────────────────────────────────────

def retrieve_evidence(
    query:          str,
    client:         Any,
    *,
    collection:     str                  = COLLECTION_NAME,
    limit:          int                  = 5,
    context_window: int                  = 0,
    rerank:         bool                 = False,
    use_hyde:       bool                 = False,
    mmr:            bool                 = False,
    mmr_lambda:     float                = 0.5,
    decompose:      bool                 = False,
    filters:        dict | None          = None,
    source_filter:  SourceFilter | None  = None,
    min_score:      float | None         = None,
    prefetch_k:     int                  = 10,
) -> RetrievalResult:
    """
    Retrieve the top-K most relevant chunks for a query.

    This is the primary retrieval entry point.  It wraps the lower-level
    retrieve() (hybrid RRF search with optional HyDE / reranking / MMR /
    decomposition) and then fetches context-expansion neighbors for each
    primary hit, returning a structured RetrievalResult.

    Args:
        query:          Natural language question or keyword string.
        client:         Active QdrantClient.
        collection:     Qdrant collection name.
        limit:          Number of primary chunks to return.
        context_window: Adjacent chunks to fetch around each primary hit
                        for context expansion (0 = disabled).
        rerank:         Apply cross-encoder reranking (jina-reranker-v2-base-multilingual).
        use_hyde:       Expand query via Hypothetical Document Embeddings.
        mmr:            Apply Maximal Marginal Relevance diversification.
        mmr_lambda:     MMR relevance/diversity tradeoff (0.0–1.0).
        decompose:      Decompose compound query into sub-queries via Groq.
        filters:        Raw filters dict — merged with source_filter if both given.
        source_filter:  Typed source / language / type filter (applied on top of
                        any raw filters).
        min_score:      Discard hits with RRF score below this threshold.
        prefetch_k:     Candidate count per sub-query before RRF fusion.

    Returns:
        RetrievalResult with ranked RetrievedChunk objects.  Each chunk
        carries its neighbors (context-expansion results) in .neighbors.
    """
    t0 = time.perf_counter()

    # Merge typed SourceFilter into the raw filters dict
    merged_filters = source_filter.to_filters(filters) if source_filter else filters

    # Collect sub-queries so they can be surfaced in RetrievalResult.sub_queries
    sub_queries_out: list[str] = []

    raw_hits = retrieve(
        query,
        client,
        collection        = collection,
        limit             = limit,
        prefetch_k        = prefetch_k,
        filters           = merged_filters,
        min_score         = min_score,
        rerank            = rerank,
        use_hyde          = use_hyde,
        mmr               = mmr,
        mmr_lambda        = mmr_lambda,
        decompose         = decompose,
        _out_sub_queries  = sub_queries_out if decompose else None,
    )

    chunks: list[RetrievedChunk] = []
    seen:   set[str]             = set()

    for hit in raw_hits:
        cid     = hit.get("chunk_id", "")
        payload = hit.get("payload", {})

        # Fetch neighbor chunks for context expansion
        neighbors: list[dict] = []
        if context_window > 0:
            source      = payload.get("source", "")
            chunk_index = payload.get("chunk_index")
            if source and chunk_index is not None:
                try:
                    neighbors = get_neighbors(
                        client,
                        source      = source,
                        chunk_index = chunk_index,
                        collection  = collection,
                        window      = context_window,
                    )
                except Exception:
                    pass  # neighbor expansion is best-effort

                # EML section guard — chunk_index is document-global, so a
                # body chunk at index 3 can bleed into the header chunk at
                # index 1 (or into an attachment chunk further down).  Restrict
                # neighbors to the same EML section as the anchor chunk so the
                # LLM is never handed a header when it asked about body content.
                anchor_section = payload.get("section", "")
                _EML_SECTIONS  = {"email_header", "email_body", "email_attachment"}
                if anchor_section in _EML_SECTIONS:
                    neighbors = [
                        nb for nb in neighbors
                        if nb.get("section") == anchor_section
                    ]

        if cid not in seen:
            seen.add(cid)
            chunks.append(RetrievedChunk(
                chunk_id  = cid,
                content   = payload.get("content", ""),
                score     = hit.get("score", 0.0),
                metadata  = {k: v for k, v in payload.items() if k != "content"},
                neighbors = neighbors,
                hop       = 1,
            ))

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        f"  [retriever] retrieve_evidence: {len(chunks)} chunk(s) in {elapsed:.0f} ms "
        f"(limit={limit}, rerank={rerank}, hyde={use_hyde}, mmr={mmr}, "
        f"decompose={decompose})"
    )

    return RetrievalResult(
        query        = query,
        chunks       = chunks,
        total        = len(chunks),
        elapsed_ms   = round(elapsed, 1),
        hops         = 1,
        sub_queries  = sub_queries_out,
    )


# ── Multi-hop retrieval ────────────────────────────────────────────────────────

def _extract_hop2_queries(query: str, top_chunks: list[RetrievedChunk]) -> list[str]:
    """
    Ask Groq to generate 1–2 follow-up search queries based on what the
    first-hop chunks revealed.

    Returns an empty list if Groq is unavailable, the passages already
    provide sufficient context, or the call fails.
    """
    client = _get_groq_client()
    if client is None:
        return []

    # Compact summary of the top-5 chunks (first 500 chars each).
    # More coverage → better follow-up queries, especially for multi-document corpora.
    passages = "\n\n".join(
        f"[{i + 1}] {c.content[:500]}"
        for i, c in enumerate(top_chunks[:5])
    )

    system_prompt = (
        "You are a search assistant helping retrieve additional evidence. "
        "Given a question and some initial retrieved passages, identify "
        "1–2 specific follow-up search queries that would fetch additional "
        "context needed to fully answer the question. "
        "If the passages already provide sufficient context, respond with exactly: NONE\n"
        "Otherwise, output one query per line — no bullets, no numbering, no extra text."
    )
    user_prompt = (
        f"Question: {query}\n\n"
        f"Initial passages:\n{passages}\n\n"
        "Follow-up search queries (or NONE):"
    )

    try:
        response = client.chat.completions.create(
            model     = _MODEL,
            messages  = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens  = _MULTIHOP_MAX_TOKENS,
            temperature = 0.0,
        )
        raw = response.choices[0].message.content.strip()
        if raw.upper().startswith("NONE"):
            logger.debug("  [retriever] Multi-hop: passages are sufficient — no hop-2.")
            return []
        queries = [line.strip() for line in raw.splitlines() if line.strip()]
        logger.debug(f"  [retriever] Hop-2 queries: {queries}")
        return queries[:2]   # hard cap at 2 follow-up queries
    except Exception as e:
        logger.warning(f"  [retriever] Hop-2 query extraction failed: {e}")
        return []


def multihop_retrieve(
    query:          str,
    client:         Any,
    *,
    collection:     str                  = COLLECTION_NAME,
    limit:          int                  = 5,
    hop2_limit:     int                  = 3,
    context_window: int                  = 0,
    rerank:         bool                 = False,
    use_hyde:       bool                 = False,
    mmr:            bool                 = False,
    mmr_lambda:     float                = 0.5,
    decompose:      bool                 = False,
    filters:        dict | None          = None,
    source_filter:  SourceFilter | None  = None,
    min_score:      float | None         = None,
    prefetch_k:     int                  = 10,
) -> RetrievalResult:
    """
    Two-hop retrieval that follows entity references across chunks.

    Hop 1 — standard retrieve_evidence() for the original query.
    Hop 2 — Groq extracts 1–2 follow-up queries from the top hop-1 chunks
             and retrieves additional evidence for each.  Results are merged
             and deduplicated; hop-1 chunks come first (scored), hop-2 chunks
             are appended with their hop=2 marker.

    Falls back gracefully to single-hop when Groq is unavailable or when
    the hop-1 passages already contain sufficient context.

    Args:
        query:          Original question.
        client:         Active QdrantClient.
        collection:     Qdrant collection name.
        limit:          Primary chunks from hop 1.
        hop2_limit:     Chunks to retrieve per hop-2 follow-up query.
        context_window: Neighbor expansion applied to both hops.
        rerank:         Apply cross-encoder reranking to hop-1 results.
        use_hyde:       HyDE query expansion on hop-1.
        mmr:            MMR diversification on hop-1 results.
        mmr_lambda:     MMR relevance/diversity tradeoff (0.0–1.0).
        decompose:      Query decomposition on hop-1.
        filters:        Raw metadata filters dict.
        source_filter:  Typed source / language / type filter.
        min_score:      Minimum RRF score threshold (applied to both hops).
        prefetch_k:     Candidate count before RRF fusion (both hops).

    Returns:
        RetrievalResult with hops=2 and hop2_queries populated (or hops=1
        if hop-2 was skipped).
    """
    t0 = time.perf_counter()

    # ── Hop 1 ──────────────────────────────────────────────────────────────────
    hop1 = retrieve_evidence(
        query,
        client,
        collection     = collection,
        limit          = limit,
        context_window = context_window,
        rerank         = rerank,
        use_hyde       = use_hyde,
        mmr            = mmr,
        mmr_lambda     = mmr_lambda,
        decompose      = decompose,
        filters        = filters,
        source_filter  = source_filter,
        min_score      = min_score,
        prefetch_k     = prefetch_k,
    )

    if not hop1.chunks:
        return hop1  # nothing retrieved — nothing to follow up on

    # ── Generate hop-2 queries ─────────────────────────────────────────────────
    hop2_queries = _extract_hop2_queries(query, hop1.chunks)

    if not hop2_queries:
        # Hop-2 not needed or Groq unavailable — return hop-1 result as-is
        return RetrievalResult(
            query        = query,
            chunks       = hop1.chunks,
            total        = hop1.total,
            elapsed_ms   = round((time.perf_counter() - t0) * 1000, 1),
            hops         = 1,
            hop2_queries = [],
        )

    # ── Hop 2 ──────────────────────────────────────────────────────────────────
    seen_ids:    set[str]             = {c.chunk_id for c in hop1.chunks}
    hop2_chunks: list[RetrievedChunk] = []

    for hq in hop2_queries:
        hop2_result = retrieve_evidence(
            hq,
            client,
            collection     = collection,
            limit          = hop2_limit,
            context_window = context_window,
            filters        = filters,
            source_filter  = source_filter,
            min_score      = min_score,
            prefetch_k     = prefetch_k,
        )
        for chunk in hop2_result.chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                chunk.hop = 2
                hop2_chunks.append(chunk)

    merged  = hop1.chunks + hop2_chunks
    elapsed = (time.perf_counter() - t0) * 1000

    logger.info(
        f"  [retriever] multihop_retrieve: hop1={len(hop1.chunks)}, "
        f"hop2={len(hop2_chunks)} new, total={len(merged)} in {elapsed:.0f} ms"
    )

    return RetrievalResult(
        query        = query,
        chunks       = merged,
        total        = len(merged),
        elapsed_ms   = round(elapsed, 1),
        hops         = 2,
        hop2_queries = hop2_queries,
    )


# ── Answer confidence scoring ─────────────────────────────────────────────────

def score_answer_confidence(
    answer: str,
    chunks: list[RetrievedChunk],
    *,
    groq_client: Any = None,
) -> float | None:
    """
    Score how well a generated answer is grounded in the retrieved evidence.

    Asks Groq to rate groundedness on a 0.0–1.0 scale:
        0.0 — fully unsupported or directly contradicted by the sources
        0.5 — partially supported (some claims grounded, some not)
        1.0 — every claim is directly supported by the provided passages

    Args:
        answer:      The generated answer text to evaluate.
        chunks:      Retrieved evidence chunks used to generate the answer.
        groq_client: Optional pre-initialised Groq client; uses the module
                     singleton if omitted.

    Returns:
        Float in [0.0, 1.0] rounded to 2 decimal places, or None on failure.
        None means the score is unavailable — callers should omit it from
        the response rather than surface a misleading zero.
    """
    if not answer.strip() or not chunks:
        return None

    client = groq_client or _get_groq_client()
    if client is None:
        return None

    # Evidence budget: 2 400 chars total spread evenly across all primary chunks
    # (up to 12).  Prioritising all primary chunks over a hard cap of 6 means
    # the scorer sees every chunk the LLM drew on, not just the top 6.
    # Per-chunk allocation shrinks proportionally so the total stays constant.
    _EVIDENCE_BUDGET  = 2_400   # total chars for evidence passages
    _MAX_ANSWER_CHARS = 1_200   # larger window — long answers were being cut mid-sentence

    max_chunks      = min(len(chunks), 12)
    chars_per_chunk = max(150, _EVIDENCE_BUDGET // max_chunks)

    passages = "\n\n".join(
        f"[{i + 1}] {c.content[:chars_per_chunk]}"
        for i, c in enumerate(chunks[:max_chunks])
    )

    # System + user split is required for chat models (llama-4-scout, etc.).
    # A "Score:" suffix in a single user message is silently ignored — the model
    # generates a full explanation instead.  The system role carries the strict
    # format constraint, the user message carries the evidence and the answer,
    # and stop=["\n"] truncates output at the first newline so we never receive
    # an essay even if the model tries to write one.
    system_msg = (
        "You are a grounding scorer. "
        "Respond with ONLY a single decimal number between 0.0 and 1.0. "
        "No explanation, no reasoning, no other text — just the number."
    )
    user_msg = (
        "Score how well the answer is supported by these passages.\n\n"
        f"Passages:\n{passages}\n\n"
        f"Answer: {answer[:_MAX_ANSWER_CHARS]}\n\n"
        "Scale: 1.0 = fully grounded | 0.5 = partially grounded | 0.0 = hallucinated\n\n"
        "Output a single decimal number only:"
    )

    try:
        response = client.chat.completions.create(
            model       = _MODEL,
            messages    = [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens  = 10,      # "0.87" is 3-4 tokens; 10 is generous
            temperature = 0.0,
            stop        = ["\n"],  # hard-stop after the number line
        )
        raw = response.choices[0].message.content.strip()
        # Extract the first token that looks like a decimal / integer
        import re as _re
        m = _re.search(r"\b([01](?:\.\d+)?|\d?\.\d+)\b", raw)
        if not m:
            raise ValueError(f"No decimal found in response: {raw!r}")
        score = float(m.group(1))
        score = max(0.0, min(1.0, score))
        logger.debug(f"  [retriever] Confidence score: {score:.2f}  (raw={raw!r})")
        return round(score, 2)
    except Exception as e:
        logger.warning(f"  [retriever] Confidence scoring failed: {e}")
        return None


# ── CLI test mode ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick CLI test for hybrid retrieval.

    Usage:
        python retriever.py "your query here"
        python retriever.py "your query" --limit 5
        python retriever.py "your query" --rerank
        python retriever.py "your query" --hyde
        python retriever.py "your query" --mmr
        python retriever.py "your query" --multihop
        python retriever.py "your query" --limit 8 --rerank --hyde
    """
    import argparse
    import textwrap

    from loguru import logger as _cli_logger
    from store import get_client, ensure_collection   # type: ignore[import]

    # ── Arg parsing ───────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Test hybrid BM25 + BGE-M3 retrieval from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query",      type=str,            help="Search query")
    parser.add_argument("--limit",    type=int, default=5, help="Number of results (default 5)")
    parser.add_argument("--window",   type=int, default=1, help="Context-expansion window (default 1)")
    parser.add_argument("--rerank",   action="store_true", help="Apply cross-encoder reranking")
    parser.add_argument("--hyde",     action="store_true", help="HyDE query expansion")
    parser.add_argument("--mmr",      action="store_true", help="MMR diversification")
    parser.add_argument("--decompose",action="store_true", help="Query decomposition")
    parser.add_argument("--multihop", action="store_true", help="Two-hop retrieval")
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME,
                        help=f"Qdrant collection (default: {COLLECTION_NAME})")
    args = parser.parse_args()

    # ── Connect ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Query      : {args.query}")
    print(f"  Limit      : {args.limit}")
    print(f"  Rerank     : {args.rerank}")
    print(f"  HyDE       : {args.hyde}")
    print(f"  MMR        : {args.mmr}")
    print(f"  Decompose  : {args.decompose}")
    print(f"  Multi-hop  : {args.multihop}")
    print(f"  Collection : {args.collection}")
    print(f"{'='*60}\n")

    try:
        _client = get_client()
        _client.get_collections()
        print("  ✓ Qdrant connected\n")
    except Exception as _e:
        print(f"  ✗ Could not connect to Qdrant: {_e}")
        print("    Make sure Qdrant is running:  docker run -p 6333:6333 qdrant/qdrant")
        raise SystemExit(1)

    # ── Retrieve ──────────────────────────────────────────────────────────────
    _kwargs: dict = dict(
        collection     = args.collection,
        limit          = args.limit,
        context_window = args.window,
        rerank         = args.rerank,
        use_hyde       = args.hyde,
        mmr            = args.mmr,
        decompose      = args.decompose,
    )

    if args.multihop:
        _result = multihop_retrieve(args.query, _client, **_kwargs)
    else:
        _result = retrieve_evidence(args.query, _client, **_kwargs)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"  Retrieved {_result.total} chunk(s) in {_result.elapsed_ms:.0f} ms  "
          f"[hops={_result.hops}]\n")

    if not _result.chunks:
        print("  (no results — is the collection indexed?)")
        raise SystemExit(0)

    _W = 72   # display width for content preview

    for i, chunk in enumerate(_result.chunks, 1):
        hop_tag  = f" [hop-{chunk.hop}]" if chunk.hop > 1 else ""
        source   = chunk.metadata.get("source", "?")
        page     = chunk.metadata.get("page_start", "?")
        section  = chunk.metadata.get("section", "")
        lang     = chunk.metadata.get("language", "")
        ctype    = chunk.metadata.get("type", "text")

        header = (
            f"  [{i}] score={chunk.score:.4f}{hop_tag}  "
            f"type={ctype}  source={source}  page={page}"
        )
        if section:
            header += f"  section={section!r}"
        if lang:
            header += f"  lang={lang}"

        print(header)

        # Content preview — wrap at _W chars, indent continuation lines
        preview = chunk.content[:400].replace("\n", " ")
        wrapped = textwrap.fill(preview, width=_W, initial_indent="      ",
                                subsequent_indent="      ")
        print(wrapped)

        # Neighbor chunks (context expansion)
        if chunk.neighbors:
            print(f"      ↳ {len(chunk.neighbors)} neighbor(s):")
            for nb in chunk.neighbors:
                nb_preview = nb.get("content", "")[:120].replace("\n", " ")
                print(f"         • {nb_preview}")

        print()
