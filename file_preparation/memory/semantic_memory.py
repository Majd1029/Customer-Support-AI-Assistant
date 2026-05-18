"""
file_preparation/memory/semantic_memory.py

Sprint 5 — Semantic User Memory
================================
Qdrant-backed long-term preference storage that survives session boundaries.

Unlike the structured key-value store in user_memory.py, semantic memory holds
free-text preference statements that are retrieved by *semantic similarity*
rather than exact key lookup.  This is better suited to nuanced preferences
like "I prefer concise answers with code examples" or "I work mostly on Arabic
NLP tasks".

Architecture
------------
- Qdrant collection ``user_preferences`` with two named vectors:
    dense  : 1024-dim BGE-M3 cosine (reuses existing embedder singleton)
    sparse : SPLADE dot-product
- Each point stores:
    payload: { user_id, text, created_at, preference_id }
    id     : UUID5(user_id + text) — deterministic, so storing the same
             statement twice is a no-op (Qdrant upsert).

Graceful degradation
--------------------
Every public function catches all exceptions and returns a safe default.
A missing Qdrant instance, unavailable embedder, or network hiccup never
propagates to the caller — the request proceeds without semantic memory.

Public API
----------
    remember_preference(user_id, text)              → str  (preference_id)
    recall_preferences(user_id, query, limit=5)     → list[str]
    delete_preference(user_id, preference_id)       → bool
    delete_all_preferences(user_id)                 → int
    list_preferences(user_id)                       → list[dict]
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger
from file_preparation.utils.groq_retry import call_groq_with_retry as _groq_retry

# ---------------------------------------------------------------------------
# Qdrant + embedder access
# ---------------------------------------------------------------------------

_QDRANT_AVAILABLE   = False
_EMBEDDER_AVAILABLE = False
_qdrant_client      = None
_encode_query_fn    = None
_encode_fn          = None

COLLECTION          = "user_preferences"
DENSE_DIM           = 1024

try:
    from file_preparation.indexing.store import get_client  # type: ignore[import]
    _qdrant_client    = get_client()
    _qdrant_client.get_collections()   # connectivity check
    _QDRANT_AVAILABLE = True
    logger.info("[SEM_MEM] Qdrant connected.")
except Exception as _qe:
    logger.warning(f"[SEM_MEM] Qdrant unavailable ({_qe}) — semantic memory disabled.")

try:
    from file_preparation.embedding.embedder import encode_query, encode  # type: ignore[import]
    _encode_query_fn    = encode_query
    _encode_fn          = encode
    _EMBEDDER_AVAILABLE = True
    logger.info("[SEM_MEM] BGE-M3 embedder available.")
except Exception as _ee:
    logger.warning(f"[SEM_MEM] Embedder unavailable ({_ee}) — semantic memory disabled.")


def _available() -> bool:
    return _QDRANT_AVAILABLE and _EMBEDDER_AVAILABLE and _qdrant_client is not None


# ---------------------------------------------------------------------------
# Collection bootstrap
# ---------------------------------------------------------------------------

def _ensure_collection() -> None:
    """Create the user_preferences collection if it doesn't exist yet."""
    if not _available():
        return
    try:
        from qdrant_client.models import (   # type: ignore[import]
            Distance, VectorParams,
            SparseVectorParams,
        )
        existing = [c.name for c in _qdrant_client.get_collections().collections]
        if COLLECTION in existing:
            return

        _qdrant_client.create_collection(
            collection_name     = COLLECTION,
            vectors_config      = {
                "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config = {
                "sparse": SparseVectorParams(),
            },
        )
        # Keyword index on user_id for fast per-user filtering
        from qdrant_client.models import PayloadSchemaType  # type: ignore[import]
        _qdrant_client.create_payload_index(
            collection_name = COLLECTION,
            field_name      = "user_id",
            field_schema    = PayloadSchemaType.KEYWORD,
        )
        logger.info(f"[SEM_MEM] Created Qdrant collection '{COLLECTION}'.")
    except Exception as exc:
        logger.warning(f"[SEM_MEM] _ensure_collection failed: {exc}")


# Bootstrap on import (best-effort)
_ensure_collection()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(user_id: str, text: str) -> str:
    """Deterministic UUID5 from (user_id, text) — dedup via Qdrant upsert."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, f"{user_id}::{text}"))


def _embed_text(text: str) -> tuple[list[float], dict[int, float]] | None:
    """
    Embed a single text string. Returns (dense, sparse) or None on failure.
    """
    if not _EMBEDDER_AVAILABLE or _encode_fn is None:
        return None
    try:
        result = _encode_fn([text])
        dense  = result.dense[0]
        sparse = result.sparse[0]
        return dense, sparse
    except Exception as exc:
        logger.warning(f"[SEM_MEM] embedding failed: {exc}")
        return None


def _embed_query(text: str) -> tuple[list[float], dict[int, float]] | None:
    """Embed a query string. Returns (dense, sparse) or None on failure."""
    if not _EMBEDDER_AVAILABLE or _encode_query_fn is None:
        return None
    try:
        result = _encode_query_fn(text)
        dense  = result.dense[0]
        sparse = result.sparse[0]
        return dense, sparse
    except Exception as exc:
        logger.warning(f"[SEM_MEM] query embedding failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def remember_preference(user_id: str, text: str) -> str:
    """
    Store a free-text preference statement for the user.

    The same statement stored twice produces the same UUID5 point ID —
    the second upsert is a no-op (idempotent).

    Parameters
    ----------
    user_id : The user this preference belongs to.
    text    : Free-text preference, e.g. "I prefer concise answers with code examples."

    Returns
    -------
    preference_id (UUID5 string).  Returns "" on failure.
    """
    if not _available():
        return ""
    text = text.strip()
    if not text:
        return ""

    vectors = _embed_text(text)
    if vectors is None:
        return ""

    dense, sparse = vectors
    pref_id = _make_id(user_id, text)

    try:
        from qdrant_client.models import PointStruct, SparseVector   # type: ignore[import]
        point = PointStruct(
            id      = pref_id,
            vector  = {
                "dense":  dense,
                "sparse": SparseVector(
                    indices = list(sparse.keys()),
                    values  = list(sparse.values()),
                ),
            },
            payload = {
                "user_id":       user_id,
                "text":          text,
                "preference_id": pref_id,
                "created_at":    datetime.now(timezone.utc).isoformat(),
            },
        )
        _qdrant_client.upsert(collection_name=COLLECTION, points=[point])
        logger.debug(f"[SEM_MEM] stored preference for {user_id!r}: {text[:60]!r}")
        return pref_id
    except Exception as exc:
        logger.warning(f"[SEM_MEM] remember_preference failed: {exc}")
        return ""


def recall_preferences(
    user_id: str,
    query:   str,
    limit:   int = 5,
    min_score: float = 0.3,
) -> list[str]:
    """
    Retrieve the user's preferences most semantically relevant to ``query``.

    Uses Qdrant hybrid RRF search (dense + sparse) filtered to ``user_id``.

    Parameters
    ----------
    user_id   : Only return this user's preferences.
    query     : The current user question (used as the search query).
    limit     : Maximum number of preferences to return.
    min_score : Minimum RRF score — filters out very low-relevance results.

    Returns
    -------
    List of preference text strings, most relevant first.
    Returns [] on any failure or when no preferences are stored.
    """
    if not _available():
        return []

    vectors = _embed_query(query)
    if vectors is None:
        return []

    dense, sparse = vectors

    try:
        from qdrant_client.models import (   # type: ignore[import]
            Prefetch, Filter, FieldCondition, MatchValue, FusionQuery, Fusion, SparseVector,
        )
        user_filter = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )
        results = _qdrant_client.query_points(
            collection_name = COLLECTION,
            prefetch        = [
                Prefetch(
                    query  = dense,
                    using  = "dense",
                    filter = user_filter,
                    limit  = limit * 2,
                ),
                Prefetch(
                    query  = SparseVector(
                        indices = list(sparse.keys()),
                        values  = list(sparse.values()),
                    ),
                    using  = "sparse",
                    filter = user_filter,
                    limit  = limit * 2,
                ),
            ],
            query        = FusionQuery(fusion=Fusion.RRF),
            query_filter = user_filter,
            limit        = limit,
            with_payload = True,
        ).points

        texts = []
        for r in results:
            if min_score and r.score < min_score:
                continue
            text = (r.payload or {}).get("text", "")
            if text:
                texts.append(text)
        return texts

    except Exception as exc:
        logger.warning(f"[SEM_MEM] recall_preferences failed: {exc}")
        return []


def delete_preference(user_id: str, preference_id: str) -> bool:
    """
    Delete a single preference by its ID.

    Returns True if deleted, False on failure or not found.
    """
    if not _available():
        return False
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore[import]
        _qdrant_client.delete(
            collection_name = COLLECTION,
            points_selector = [preference_id],
        )
        return True
    except Exception as exc:
        logger.warning(f"[SEM_MEM] delete_preference failed: {exc}")
        return False


def delete_all_preferences(user_id: str) -> int:
    """
    Delete ALL preferences for a user.

    Returns the number of points deleted (approximate — Qdrant does not
    guarantee an exact count on batch deletes).
    """
    if not _available():
        return 0
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore[import]
        # Count first so we can return a meaningful number
        count_result = _qdrant_client.count(
            collection_name = COLLECTION,
            count_filter    = Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            exact = True,
        )
        count = count_result.count if count_result else 0

        _qdrant_client.delete(
            collection_name = COLLECTION,
            points_selector = Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
        )
        logger.info(f"[SEM_MEM] deleted {count} preferences for {user_id!r}")
        return count
    except Exception as exc:
        logger.warning(f"[SEM_MEM] delete_all_preferences failed: {exc}")
        return 0


def list_preferences(user_id: str, limit: int = 50) -> list[dict]:
    """
    Return all stored preferences for a user as a list of dicts.
    Each dict has keys: preference_id, text, created_at.
    Ordered by created_at descending (newest first).
    """
    if not _available():
        return []
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore[import]
        records, _ = _qdrant_client.scroll(
            collection_name = COLLECTION,
            scroll_filter   = Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit        = limit,
            with_payload = True,
            with_vectors = False,
        )
        items = []
        for r in records:
            p = r.payload or {}
            items.append({
                "preference_id": p.get("preference_id", str(r.id)),
                "text":          p.get("text", ""),
                "created_at":    p.get("created_at", ""),
            })
        # Sort newest first
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return items
    except Exception as exc:
        logger.warning(f"[SEM_MEM] list_preferences failed: {exc}")
        return []


def semantic_memory_available() -> bool:
    """Returns True when both Qdrant and the embedder are reachable."""
    return _available()


# ---------------------------------------------------------------------------
# Auto-extraction — Sprint 5
# ---------------------------------------------------------------------------

_PREF_EXTRACT_MODEL   = "qwen/qwen3-32b"
_PREF_EXTRACT_TIMEOUT = 20      # seconds — runs on background path

_PREF_EXTRACT_SYSTEM = (
    "You detect explicit user preference statements from a conversation. "
    "Preference statements express how the user likes things done, e.g.:\n"
    "  - 'I prefer concise answers'\n"
    "  - 'Please always include code examples'\n"
    "  - 'I work mostly on Arabic NLP tasks'\n"
    "  - 'Respond in French'\n\n"
    "Return a JSON array of short, self-contained preference strings (each ≤ 20 words). "
    "Only include preferences the user has *explicitly stated* — never infer. "
    "Return [] if no preferences are found.\n\n"
    "Output ONLY valid JSON, nothing else."
)


def _call_groq_extract_prefs(turns_text: str, groq_client) -> list[str]:
    """
    Ask Groq to extract preference statements from recent conversation turns.
    Returns a list of preference strings, or [] on any failure.
    """
    prompt = f"{_PREF_EXTRACT_SYSTEM}\n\nConversation:\n{turns_text}"
    try:
        raw = _groq_retry(
            groq_client,
            model       = _PREF_EXTRACT_MODEL,
            prompt      = prompt,
            max_tokens  = 512,
            temperature = 0.0,
            timeout     = _PREF_EXTRACT_TIMEOUT,
            label       = "[SEM_MEM]",
            strip_think = True,
        )
        # Tolerate code-fenced JSON
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw.strip())
        data = json.loads(raw)
        if not isinstance(data, list):
            return []
        return [str(p).strip() for p in data if p and str(p).strip()]
    except Exception as exc:
        logger.debug(f"[SEM_MEM] preference extraction failed: {exc}")
        return []


def extract_and_store_preferences(
    user_id:     str,
    turns:       list,          # list[Turn] from buffer.py
    groq_client,
    max_turns:   int = 10,
) -> list[str]:
    """
    Extract explicit preference statements from recent conversation turns and
    persist them as semantic memories via remember_preference().

    Only the most recent ``max_turns`` *user* turns are examined to keep the
    Groq prompt compact.  Idempotency is guaranteed by UUID5 point IDs — the
    same statement stored twice is a no-op.

    Called as a BackgroundTask from api.py — never on the critical request path.

    Returns the list of newly stored preference strings (empty list on failure
    or when no preferences were found).
    """
    if groq_client is None or not _available():
        return []

    # Only user turns contain preference expressions
    user_turns = [t for t in turns if t.role == "user"][-max_turns:]
    if not user_turns:
        return []

    turns_text = "\n".join(f"User: {t.content}" for t in user_turns)

    preferences = _call_groq_extract_prefs(turns_text, groq_client)
    if not preferences:
        return []

    stored: list[str] = []
    for pref in preferences:
        pref_id = remember_preference(user_id, pref)
        if pref_id:
            stored.append(pref)

    if stored:
        logger.info(
            f"[SEM_MEM] stored {len(stored)} extracted preference(s) for {user_id!r}: "
            + "; ".join(repr(p[:40]) for p in stored)
        )
    return stored


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_preferences_for_prompt(preferences: list[str]) -> str:
    """
    Format a list of recalled preference strings as a compact prompt block.

    Example output:
        [USER PREFERENCES]
        - I prefer concise answers with code examples.
        - I work mostly on Arabic NLP tasks.
        [END USER PREFERENCES]
    """
    if not preferences:
        return ""
    lines = "\n".join(f"- {p}" for p in preferences)
    return f"[USER PREFERENCES]\n{lines}\n[END USER PREFERENCES]"
