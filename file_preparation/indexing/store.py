"""
store.py — Qdrant vector store helpers.

Manages a single Qdrant collection with two named vectors:
  • "dense"  — 1024-dim cosine (BGE-M3 dense output)
  • "sparse" — BM25 lexical weights (hashing-trick TF + Qdrant IDF modifier)

Hybrid search fuses both via Reciprocal Rank Fusion (RRF).

Usage:
    from store import get_client, ensure_collection, upsert, search

    client = get_client()
    ensure_collection(client, "documents")
    upsert(client, "documents", points)
    results = search(client, "documents", dense_vec, sparse_vec, limit=10)
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger


# ── Retry helper ──────────────────────────────────────────────────────────────

def _with_retry(fn, *args, retries: int = 3, base_delay: float = 1.0, **kwargs):
    """
    Call fn(*args, **kwargs) with exponential backoff on failure.

    Retries on any exception (connection errors, timeouts, transient Qdrant
    errors).  Raises the last exception if all retries are exhausted.

    Args:
        fn:         Callable to retry.
        retries:    Maximum number of additional attempts after the first.
        base_delay: Initial sleep in seconds; doubles on each retry (1s, 2s, 4s).
    """
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"  Qdrant call failed (attempt {attempt + 1}/{retries + 1}): {e}. "
                    f"Retrying in {delay:.1f}s …"
                )
                time.sleep(delay)
            else:
                logger.error(f"  Qdrant call failed after {retries + 1} attempts: {e}")
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Qdrant call failed: no attempts made")

# ── Qdrant imports ────────────────────────────────────────────────────────────
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm
    from qdrant_client.http.models import (
        Distance,
        FieldCondition,
        Filter,
        Fusion,
        FusionQuery,
        MatchAny,
        MatchValue,
        NamedSparseVector,
        NamedVector,
        PayloadSchemaType,
        PointStruct,
        Prefetch,
        SparseVector,
        SparseVectorParams,
        VectorParams,
        VectorsConfig,
    )
    _QDRANT_AVAILABLE = True

    # Modifier.IDF — added in qdrant-client 1.8.0.  When available, Qdrant
    # applies IDF normalisation server-side so stored BM25 TF weights become
    # proper BM25 scores at query time.  Falls back to plain TF on older clients.
    try:
        from qdrant_client.http.models import Modifier as _Modifier
        _BM25_SPARSE_PARAMS = SparseVectorParams(modifier=_Modifier.IDF)
        logger.debug("  [store] Modifier.IDF available — BM25 sparse vectors will use server-side IDF.")
    except Exception:
        _BM25_SPARSE_PARAMS = SparseVectorParams()
        logger.debug("  [store] Modifier.IDF not available (qdrant-client < 1.8) — using raw TF sparse vectors.")

except ImportError:
    _QDRANT_AVAILABLE    = False
    _BM25_SPARSE_PARAMS  = None   # placeholder; _require_qdrant() raises before it's used

# ── Constants ─────────────────────────────────────────────────────────────────
COLLECTION_NAME  = "documents"   # default collection name
DENSE_DIM        = 1024
DENSE_VECTOR     = "dense"
SPARSE_VECTOR    = "sparse"
UPSERT_BATCH     = 64            # points per upsert call

# Payload fields that get Qdrant keyword indexes for fast filtered search.
# doc_id is included so scroll/filter by document GUID is O(log n).
_INDEXED_PAYLOAD_FIELDS = ["source", "language", "type", "doc_id"]

# Payload fields that get Qdrant integer indexes (used in range filters)
_INDEXED_INTEGER_FIELDS = ["chunk_index"]


# ── Client factory ────────────────────────────────────────────────────────────

def get_client(
    host: str = "localhost",
    port: int = 6333,
    *,
    url:     str | None = None,
    api_key: str | None = None,
    timeout: int = 30,
) -> "QdrantClient":
    """
    Return a Qdrant client.

    Resolution order for connection params:
      1. Explicit url / api_key arguments
      2. QDRANT_URL / QDRANT_API_KEY environment variables (loaded from .env)
      3. localhost:6333 default

    Args:
        host:    Qdrant host (ignored if url is given or QDRANT_URL is set).
        port:    Qdrant port (ignored if url is given or QDRANT_URL is set).
        url:     Full Qdrant Cloud URL, e.g. "https://xxx.qdrant.io:6333".
        api_key: Qdrant Cloud API key (for cloud deployments).
        timeout: Request timeout in seconds.
    """
    _require_qdrant()

    # Auto-load .env so QDRANT_URL / QDRANT_API_KEY are available
    try:
        from dotenv import load_dotenv
        _env = Path(__file__).resolve().parent.parent.parent / ".env"
        load_dotenv(dotenv_path=_env, override=False)
    except ImportError:
        pass

    # Resolve connection params — explicit args win over env vars
    resolved_url     = url     or os.environ.get("QDRANT_URL")
    resolved_api_key = api_key or os.environ.get("QDRANT_API_KEY")

    if resolved_url:
        logger.info(f"  Connecting to Qdrant Cloud: {resolved_url}")
        return QdrantClient(url=resolved_url, api_key=resolved_api_key, timeout=timeout)

    logger.info(f"  Connecting to Qdrant at {host}:{port}")
    return QdrantClient(host=host, port=port, timeout=timeout)


# ── Collection management ─────────────────────────────────────────────────────

def ensure_collection(
    client:   "QdrantClient",
    name:     str  = COLLECTION_NAME,
    *,
    recreate: bool = False,
) -> None:
    """
    Create the collection if it does not exist (or recreate if requested).

    Vector layout:
      • "dense"  — VectorParams(size=1024, distance=Cosine)  [BGE-M3 dense]
      • "sparse" — SparseVectorParams(modifier=Modifier.IDF) [BM25 TF weights;
                   IDF applied server-side when Modifier.IDF is available]

    Payload indexes are created on: source, language, type.
    These allow fast filtered search without full payload scans.

    Args:
        client:   Active QdrantClient.
        name:     Collection name.
        recreate: Drop and recreate if True (destroys all indexed data).
    """
    _require_qdrant()

    existing = {c.name for c in client.get_collections().collections}

    if name in existing:
        if recreate:
            logger.warning(f"  Recreating collection '{name}' — all data will be lost.")
            client.delete_collection(name)
        else:
            logger.debug(f"  Collection '{name}' already exists — skipping creation.")
            return

    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR: VectorParams(
                size=DENSE_DIM,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR: _BM25_SPARSE_PARAMS,
        } if _BM25_SPARSE_PARAMS is not None else None,
    )
    logger.info(f"  Collection '{name}' created (dense={DENSE_DIM}d + sparse).")

    # Create payload indexes for fast filtered search.
    # Without these, Qdrant does a full collection scan for every filter.
    for field in _INDEXED_PAYLOAD_FIELDS:
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.debug(f"  Payload index created: '{field}' on '{name}'.")
        except Exception as e:
            logger.warning(f"  Could not create payload index for '{field}': {e}")

    # Integer indexes for range-filter fields (e.g. chunk_index in get_neighbors).
    for field in _INDEXED_INTEGER_FIELDS:
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=PayloadSchemaType.INTEGER,
            )
            logger.debug(f"  Integer payload index created: '{field}' on '{name}'.")
        except Exception as e:
            logger.warning(f"  Could not create integer payload index for '{field}': {e}")


# ── Point helpers ─────────────────────────────────────────────────────────────

def make_point_id(chunk_id: str) -> str:
    """
    Derive a deterministic UUID-5 point ID from a chunk_id string.

    Using UUID5 keeps IDs stable across re-indexing runs, so upserts are
    idempotent (same chunk → same point, existing point is overwritten).
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def build_point(
    chunk_id:   str,
    dense_vec:  list[float],
    sparse_vec: dict[int, float],
    payload:    dict[str, Any],
) -> "PointStruct":
    """
    Construct a Qdrant PointStruct from a single chunk's embeddings + metadata.

    Args:
        chunk_id:   Canonical chunk identifier (used as UUID5 seed).
        dense_vec:  1024-dim float list from BGE-M3 dense output.
        sparse_vec: Dict of {token_id: bm25_tf_weight} from bm25_encoder.
                    When the collection uses Modifier.IDF, Qdrant applies IDF
                    server-side at query time, giving proper BM25 TF·IDF scores.
        payload:    Arbitrary metadata stored alongside the vector.
    """
    _require_qdrant()
    indices = list(sparse_vec.keys())
    values  = list(sparse_vec.values())

    return PointStruct(
        id=make_point_id(chunk_id),
        vector={
            DENSE_VECTOR:  dense_vec,
            SPARSE_VECTOR: SparseVector(indices=indices, values=values),
        },
        payload=payload,
    )


# ── Upsert ────────────────────────────────────────────────────────────────────

def upsert(
    client:     "QdrantClient",
    points:     list["PointStruct"],
    collection: str = COLLECTION_NAME,
    batch_size: int = UPSERT_BATCH,
) -> int:
    """
    Upsert points into the collection in batches.

    Returns the total number of points upserted.
    """
    _require_qdrant()
    total = 0
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        _with_retry(
            client.upsert,
            collection_name=collection,
            points=batch,
            wait=True,
        )
        total += len(batch)
        logger.debug(f"  Upserted {total}/{len(points)} points into '{collection}'.")
    return total


# ── Delete by source ──────────────────────────────────────────────────────────

def delete_by_source(
    client:     "QdrantClient",
    source:     str,
    collection: str = COLLECTION_NAME,
) -> int:
    """
    Delete all indexed points whose payload.source matches the given filename.

    Call this before re-indexing a document to prevent stale chunks from
    accumulating when the document is re-chunked with a different chunk count.

    Args:
        client:     Active QdrantClient.
        source:     The source filename (e.g. "report.pdf").
        collection: Target collection name.

    Returns:
        Number of points deleted (Qdrant reports this as operation_id, so we
        return -1 when the exact count is unavailable).
    """
    _require_qdrant()

    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        logger.debug(f"  delete_by_source: collection '{collection}' does not exist — nothing to delete.")
        return 0

    result = client.delete(
        collection_name=collection,
        points_selector=qm.FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            )
        ),
        wait=True,
    )
    logger.info(f"  Deleted points for source='{source}' from '{collection}'.")
    return -1   # Qdrant's delete response does not include a count


# ── List indexed documents ────────────────────────────────────────────────────

def list_indexed_documents(
    client:     "QdrantClient",
    collection: str = COLLECTION_NAME,
    owner_id:   str = "",
) -> list[dict[str, Any]]:
    """
    Scroll through the collection and aggregate chunk counts by source filename.

    Uses Qdrant's cursor-based pagination so collections of any size are fully
    covered — there is no hard point cap.

    When ``owner_id`` is provided, only chunks whose ``owner_id`` payload field
    matches that value are counted.  This gives per-user document isolation when
    files are tagged with an owner at upload time.

    Returns a list of dicts sorted by chunk count descending:
        [{ "source": str, "chunks": int, "types": {"text": int, ...} }]

    Args:
        client:     Active QdrantClient.
        collection: Target collection name.
        owner_id:   When non-empty, filter to only this owner's documents.
    """
    _require_qdrant()

    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        return []

    # Build an optional owner filter for the scroll query
    scroll_filter: Filter | None = None
    if owner_id:
        scroll_filter = Filter(
            must=[FieldCondition(key="owner_id", match=MatchValue(value=owner_id))]
        )

    docs:   dict[str, dict] = {}
    offset: Any             = None
    total_scanned           = 0

    while True:
        response, next_offset = client.scroll(
            collection_name=collection,
            limit=256,           # page size — not a total cap
            offset=offset,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=False,
        )

        for point in response:
            payload = point.payload or {}
            src     = payload.get("source", "<unknown>")
            ctype   = payload.get("type", "text")

            if src not in docs:
                docs[src] = {"source": src, "chunks": 0, "types": {}}

            docs[src]["chunks"] += 1
            docs[src]["types"][ctype] = docs[src]["types"].get(ctype, 0) + 1

        total_scanned += len(response)

        # next_offset is None when Qdrant has no more pages
        if next_offset is None:
            break
        offset = next_offset

    logger.debug(f"  list_indexed_documents: scanned {total_scanned} points in '{collection}'.")
    return sorted(docs.values(), key=lambda d: d["chunks"], reverse=True)


# ── Context expansion ────────────────────────────────────────────────────────

def get_neighbors(
    client:     "QdrantClient",
    source:     str,
    chunk_index: int,
    collection: str = COLLECTION_NAME,
    window:     int = 1,
) -> list[dict[str, Any]]:
    """
    Fetch chunks adjacent to a given chunk_index within the same source document.

    Used to expand context around a retrieved chunk — the most relevant
    sentence is often at a chunk boundary, so fetching ±window neighbors
    gives the LLM more complete context without retrieving the whole document.

    Args:
        client:      Active QdrantClient.
        source:      Source filename (e.g. "report.pdf").
        chunk_index: The 1-based chunk_index of the anchor chunk.
        collection:  Target collection name.
        window:      How many chunks before and after to fetch (default 1).

    Returns:
        List of payload dicts for the neighboring chunks, sorted by chunk_index.
        The anchor chunk itself is NOT included.
    """
    _require_qdrant()

    low  = max(1, chunk_index - window)
    high = chunk_index + window

    # Build a filter: same source AND chunk_index in [low, high] AND != anchor
    from qdrant_client.http.models import Range, ValuesCount  # noqa: F401

    try:
        response, _ = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value=source)),
                    FieldCondition(
                        key="chunk_index",
                        range=qm.Range(gte=low, lte=high),
                    ),
                ],
                must_not=[
                    FieldCondition(key="chunk_index", match=MatchValue(value=chunk_index)),
                ],
            ),
            limit=window * 2,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        logger.warning(f"  get_neighbors failed for source='{source}' index={chunk_index}: {e}")
        return []

    neighbors = [dict(pt.payload or {}) for pt in response]
    neighbors.sort(key=lambda p: p.get("chunk_index", 0))
    return neighbors


# ── Document retrieval by doc_id ──────────────────────────────────────────────

def get_chunks_by_doc_id(
    client:     "QdrantClient",
    doc_id:     str,
    collection: str = COLLECTION_NAME,
    limit:      int = 1000,
) -> list[dict[str, Any]]:
    """
    Return all chunks that belong to a document identified by its doc_id GUID.

    Every chunk produced by chunk_document() carries metadata.doc_id — a UUID5
    derived from the source filename — so this call fetches every text, table,
    and image chunk from a specific document in a single Qdrant payload-filter
    scroll (O(log n) with the keyword index on doc_id).

    Args:
        client:     Active QdrantClient.
        doc_id:     UUID5 string (e.g. "3f2a1b4c-…").
        collection: Target collection name.
        limit:      Maximum chunks to return (default 1 000).

    Returns:
        List of payload dicts sorted by chunk_index (document order).
    """
    _require_qdrant()

    try:
        response, _ = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))],
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        logger.warning(f"  get_chunks_by_doc_id failed for doc_id='{doc_id}': {e}")
        return []

    chunks = [dict(pt.payload or {}) for pt in response]
    chunks.sort(key=lambda p: p.get("chunk_index", 0))
    return chunks


# ── Collection statistics ─────────────────────────────────────────────────────

def get_collection_stats(
    client:     "QdrantClient",
    collection: str = COLLECTION_NAME,
) -> dict:
    """
    Return point count, vector count, and storage info for a collection.

    Returns a plain dict so the caller can serialise it directly to JSON:
        {
            "collection":    str,
            "exists":        bool,
            "points_count":  int,
            "vectors_count": int,
            "segments_count": int,
            "status":        str,   # "green" | "yellow" | "red"
            "disk_bytes":    int | None,
        }
    """
    _require_qdrant()

    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        return {"collection": collection, "exists": False}

    info = client.get_collection(collection_name=collection)

    disk_bytes = None
    try:
        disk_bytes = getattr(info.optimizer_status, "optimizations_total", None)   # proxy when full payload unavailable
    except Exception:
        pass

    return {
        "collection":     collection,
        "exists":         True,
        "points_count":   info.points_count  or 0,
        "vectors_count":  getattr(info, "vectors_count", None) or info.indexed_vectors_count or 0,
        "segments_count": info.segments_count if hasattr(info, "segments_count") else None,
        "status":         str(info.status) if info.status else "unknown",
        "disk_bytes":     disk_bytes,
    }


# ── Hybrid search ─────────────────────────────────────────────────────────────

def search(
    client:       "QdrantClient",
    dense_vec:    list[float],
    sparse_vec:   dict[int, float],
    collection:   str          = COLLECTION_NAME,
    limit:        int          = 10,
    prefetch_k:   int          = 10,
    filters:      dict | None  = None,
    min_score:    float | None = None,
    with_vectors: bool         = False,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval: dense ANN + sparse BM25-style, fused via RRF.

    Args:
        client:     Active QdrantClient.
        dense_vec:  Query dense vector (1024 floats).
        sparse_vec: Query sparse vector ({token_id: weight}).
        collection: Target collection name.
        limit:      Number of results to return.
        prefetch_k: Candidates gathered per sub-query before fusion.
        filters:      Optional metadata filters. Each key/value pair can be:
                        - str  → exact match  e.g. {"language": "en"}
                        - list → any-of match e.g. {"type": ["text", "table"]}
        min_score:    Discard results with RRF score below this threshold.
                      Useful to filter out irrelevant results when the query
                      does not match anything in the index.
        with_vectors: If True, include the dense vector in each result dict
                      under the key "dense_vector". Used by MMR diversification.

    Returns:
        List of dicts: [{ "chunk_id", "score", "payload", ?"dense_vector" }]
        Sorted by descending RRF score.
    """
    _require_qdrant()

    # Each Prefetch sub-query must fetch at least `limit` candidates so that
    # RRF fusion always has enough material to fill the requested result set.
    # Without this, a caller with limit=50 and prefetch_k=40 would silently
    # get fewer results than requested.
    prefetch_k = max(prefetch_k, limit)

    qdrant_filter = _build_filter(filters) if filters else None

    s_indices = list(sparse_vec.keys())
    s_values  = list(sparse_vec.values())

    prefetch = [
        Prefetch(
            query=dense_vec,
            using=DENSE_VECTOR,
            limit=prefetch_k,
            filter=qdrant_filter,
        ),
        Prefetch(
            query=SparseVector(indices=s_indices, values=s_values),
            using=SPARSE_VECTOR,
            limit=prefetch_k,
            filter=qdrant_filter,
        ),
    ]

    response = _with_retry(
        client.query_points,
        collection_name=collection,
        prefetch=prefetch,
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=True,
        with_vectors=[DENSE_VECTOR] if with_vectors else False,
    )

    results = []
    for hit in response.points:
        if min_score is not None and hit.score < min_score:
            continue
        payload = dict(hit.payload or {})
        entry = {
            "chunk_id": payload.get("chunk_id", str(hit.id)),
            "score":    hit.score,
            "payload":  payload,
        }
        if with_vectors and hit.vector:
            # hit.vector is a dict when named vectors are used
            vec = hit.vector
            if isinstance(vec, dict):
                entry["dense_vector"] = vec.get(DENSE_VECTOR, [])
            else:
                entry["dense_vector"] = list(vec) if vec else []
        results.append(entry)

    return results


# ── Snapshot / backup ─────────────────────────────────────────────────────────

def create_snapshot(
    client:     "QdrantClient",
    collection: str = COLLECTION_NAME,
) -> dict:
    """
    Create a Qdrant snapshot of the collection.

    Snapshots are stored server-side inside the Qdrant container at
    `/qdrant/snapshots/<collection>/`.  Mount that path to a host volume
    to persist them across container restarts.

    Returns a dict with the snapshot name and creation time.
    """
    _require_qdrant()
    result = client.create_snapshot(collection_name=collection)
    if result is None:
        raise RuntimeError(f"Failed to create snapshot for collection '{collection}'")
    logger.info(f"  Snapshot created for '{collection}': {result.name}")
    return {
        "collection":   collection,
        "name":         result.name,
        "creation_time": result.creation_time if result.creation_time else None,
        "size":         result.size,
    }


def list_snapshots(
    client:     "QdrantClient",
    collection: str = COLLECTION_NAME,
) -> list[dict]:
    """List all available snapshots for a collection."""
    _require_qdrant()
    snapshots = client.list_snapshots(collection_name=collection)
    return [
        {
            "name":          s.name,
            "creation_time": s.creation_time if s.creation_time else None,
            "size":          s.size,
        }
        for s in snapshots
    ]


def delete_snapshot(
    client:        "QdrantClient",
    snapshot_name: str,
    collection:    str = COLLECTION_NAME,
) -> bool:
    """
    Delete a named snapshot from Qdrant.

    Returns True on success.
    """
    _require_qdrant()
    client.delete_snapshot(collection_name=collection, snapshot_name=snapshot_name)
    logger.info(f"  Snapshot deleted: '{snapshot_name}' from '{collection}'.")
    return True


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_filter(filters: dict) -> "Filter":
    """
    Convert a flat {field: value} dict into a Qdrant must-filter.

    Supports:
      - str value  → MatchValue (exact match)
      - list value → MatchAny  (any-of match)

    Examples:
        {"language": "en"}
        {"type": ["text", "table"]}
        {"language": "en", "source": "report.pdf"}
    """
    conditions = []
    for field, value in filters.items():
        if isinstance(value, list):
            conditions.append(
                FieldCondition(key=field, match=MatchAny(any=value))
            )
        else:
            conditions.append(
                FieldCondition(key=field, match=MatchValue(value=value))
            )
    return Filter(must=conditions)


def _require_qdrant() -> None:
    """Raise ImportError if qdrant-client is not installed."""
    if not _QDRANT_AVAILABLE:
        raise ImportError(
            "qdrant-client is not installed. "
            "Run: pip install qdrant-client"
        )                