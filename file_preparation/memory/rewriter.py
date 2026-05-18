"""
file_preparation/memory/rewriter.py

Query rewriter — converts underspecified follow-up questions into
fully self-contained retrieval queries using conversation history.

Three-tier routing (memory_layer_design.md §3.3):

    Tier 0 — Skip  (< 2 ms)
        Query is self-contained: long enough, no pronouns/references detected.
        Returns the raw query unchanged.

    Tier 1 — Entity injection  (2–5 ms, no LLM)
        Extracts named entities and resolves demonstrative references
        ("that", "it", "the same period") from the buffer using regex.
        Prepends resolved entities as a keyword prefix to the raw query.
        Handles the majority of real follow-up questions cheaply.

    Tier 2 — LLM rewriting  (300–600 ms, Groq call)
        Fallback for queries that Tier 1 cannot resolve:
        - Contains pronouns that entity injection can't resolve
        - Very short (< 4 words) with no resolvable entities
        The conversation history + new query are sent to Groq/qwen3-32b
        (or Qwen2.5 via Ollama) which returns a standalone rewritten query.
        Uses temperature=0.0 for deterministic output.

Usage
-----
    from file_preparation.memory.rewriter import rewrite_query
    from file_preparation.memory.buffer   import buffer_as_text, Turn

    rewritten, tier = rewrite_query(
        query   = "What about the next quarter?",
        turns   = read_buffer(session_id, user_id),
        groq_client = groq_client,   # pass None to disable Tier 2
    )
    # rewritten → "What were the Q1 2025 financial results?"
    # tier      → "llm"  ("skip" | "entity" | "llm")
"""

from __future__ import annotations

import re
import threading
import time
from typing import Optional

from loguru import logger

from .buffer import Turn, buffer_as_text


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Queries longer than this word-count threshold AND free of reference signals
# are considered self-contained — skip rewriting entirely.
_SELF_CONTAINED_MIN_WORDS = 8

# Words/patterns that signal the query needs prior context to be understood.
# Compiled once at module load.
_REFERENCE_SIGNALS = re.compile(
    r"\b("
    r"it|its|they|them|their|this|that|these|those|"
    r"the same|same|above|previous|prior|earlier|"
    r"last|next|another|other|such|he|she|him|her|"
    r"explain|elaborate|expand|continue|more|further|again"
    r")\b",
    re.IGNORECASE,
)

# Patterns that look like named entities or important terms to carry forward.
# Matches: ALLCAPS abbreviations, Title-case words (not sentence starters),
# numbers with units, quoted terms, percentages, fiscal references.
_ENTITY_PATTERNS = [
    re.compile(r"\b[A-Z]{2,}\b"),                              # Abbreviations: Q3, YoY, EBITDA
    re.compile(r"\b[A-Z][a-z]{2,}(?:\s[A-Z][a-z]{2,})*\b"),  # Proper nouns: Cloud Services
    re.compile(r"\b\$[\d,.]+[BMK]?\b"),                        # Dollar amounts: $4.2B
    re.compile(r"\b\d{4}\b"),                                  # Years: 2024
    re.compile(r"\bQ[1-4]\s?\d{4}\b", re.IGNORECASE),         # Quarters: Q3 2024
    re.compile(r"\b\d+(?:\.\d+)?%\b"),                        # Percentages: 12%
    re.compile(r'"[^"]{3,40}"'),                               # Quoted terms
]

# Common English words that match title-case or ALLCAPS patterns but carry no
# entity meaning.  Filtered out in _extract_entities().
_STOPWORDS: set[str] = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "not", "no", "nor", "so",
    "yet", "both", "either", "neither", "each", "every", "all", "any",
    "few", "more", "most", "other", "some", "such", "only", "own", "same",
    "than", "too", "very", "just", "also", "as", "if", "because", "while",
    "although", "though", "since", "when", "where", "which", "who", "whom",
    "what", "how", "why", "that", "this", "these", "those", "there", "here",
    "revenue", "income", "expenses", "costs", "profit", "loss",  # generic finance terms
    "results", "figures", "data", "information", "details", "answer",
    "question", "report", "said", "stated", "noted", "according",
}

# Model used for Tier-2 LLM rewriting via Groq
_REWRITE_MODEL = "qwen/qwen3-32b"
_MAX_TOKENS    = 256   # short — we only need a single rewritten sentence
_TEMPERATURE   = 0.0   # deterministic

# Prompt for Tier-2 LLM rewriting
_REWRITE_PROMPT = """\
You are a search query rewriter for a document retrieval system.

Given the CONVERSATION HISTORY and the USER'S LATEST QUESTION, rewrite the \
question so it is fully self-contained and optimised for semantic vector search. \
The rewritten query should not require any prior context to be understood.

Rules:
- Replace all pronouns and references ("it", "that", "the same period", etc.) \
  with the specific entities they refer to.
- Keep the rewritten query concise — one sentence, ≤ 25 words.
- Output ONLY the rewritten query. No explanation, no preamble, no punctuation \
  beyond the sentence itself.

CONVERSATION HISTORY:
{history}

USER'S LATEST QUESTION: {question}

REWRITTEN QUERY:"""

# Strip Qwen3 <think> blocks from the LLM response before using it
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Tier-2 rewrite cache
# ---------------------------------------------------------------------------
# Caches LLM rewrite results so the same (session_id, raw_query) never pays
# the 300–600 ms Groq round-trip twice.  The cache is session-scoped — a query
# that looks identical but arrives in a different session may legitimately
# resolve to a different rewrite (different prior context).
#
# TTL = 5 minutes.  Eviction runs opportunistically on every cache write.
# Thread-safe via a single lock (the cache is only updated at Tier-2, so
# contention is rare).

_REWRITE_CACHE: dict[tuple[str, str], tuple[str, str, float]] = {}
# key   → (session_id, normalised_query)
# value → (rewritten_query, tier, timestamp)

_CACHE_TTL    = 300.0   # seconds  (5 min)
_CACHE_LOCK   = threading.Lock()


def _normalise_key(query: str) -> str:
    """Lower-case, collapse whitespace — makes the cache key robust to trivial variations."""
    return " ".join(query.lower().split())


def _cache_get(session_id: str, query: str) -> tuple[str, str] | None:
    """Return (rewritten, tier) if a fresh cache entry exists, else None."""
    key = (session_id, _normalise_key(query))
    with _CACHE_LOCK:
        entry = _REWRITE_CACHE.get(key)
        if entry is None:
            return None
        rewritten, tier, ts = entry
        if time.monotonic() - ts > _CACHE_TTL:
            del _REWRITE_CACHE[key]
            return None
        return rewritten, tier


def _cache_set(session_id: str, query: str, rewritten: str, tier: str) -> None:
    """Store a rewrite result and evict stale entries."""
    key = (session_id, _normalise_key(query))
    now = time.monotonic()
    with _CACHE_LOCK:
        _REWRITE_CACHE[key] = (rewritten, tier, now)
        # Opportunistic eviction — drop entries older than TTL
        stale = [k for k, v in _REWRITE_CACHE.items() if now - v[2] > _CACHE_TTL]
        for k in stale:
            del _REWRITE_CACHE[k]


# ---------------------------------------------------------------------------
# Tier 0 — Self-contained check
# ---------------------------------------------------------------------------

def _is_self_contained(query: str) -> bool:
    """
    Return True if the query is long enough and free of reference signals,
    meaning it can be sent to retrieval unchanged.
    """
    words = re.findall(r'\b\w+\b', query)
    if len(words) < _SELF_CONTAINED_MIN_WORDS:
        return False
    if _REFERENCE_SIGNALS.search(query):
        return False
    return True


# ---------------------------------------------------------------------------
# Tier 1 — Entity injection (no LLM)
# ---------------------------------------------------------------------------

def _extract_entities(turns: list[Turn], max_turns: int = 6) -> list[str]:
    """
    Extract candidate entities from the most recent `max_turns` turns
    using regex patterns.  Returns a deduplicated list preserving order
    of first occurrence (most recent turns take priority).
    """
    entities: list[str] = []
    seen: set[str] = set()

    for turn in reversed(turns[-max_turns:]):
        for pattern in _ENTITY_PATTERNS:
            for match in pattern.findall(turn.content):
                token = match.strip()
                if token and token.lower() not in seen and token.lower() not in _STOPWORDS:
                    seen.add(token.lower())
                    entities.append(token)

    return entities


def _entity_inject(query: str, turns: list[Turn]) -> str:
    """
    Prepend resolved entities as a keyword context prefix to the query.

    Example:
        entities = ["Q3 2024", "$4.2B", "Cloud Services"]
        query    = "What about the operating margin?"
        result   = "Q3 2024, $4.2B, Cloud Services: What about the operating margin?"
    """
    entities = _extract_entities(turns)
    if not entities:
        return query
    prefix = ", ".join(entities[:6])   # cap to avoid prefix bloat
    return f"{prefix}: {query}"


def _entity_injection_helped(original: str, rewritten: str) -> bool:
    """
    Heuristic: entity injection 'helped' if at least one substantive entity
    was prepended — i.e. an entity of ≥ 4 characters that does not already
    appear verbatim (case-insensitive) in the original query.

    The old +5 char length check passed even for micro-entities like "Q3" or
    "$1" that add no retrieval value.  This stricter check requires at least
    one entity in the prefix that is genuinely new to the query.
    """
    if rewritten == original:
        return False

    # The prefix is everything before the first ": " separator.
    if ": " not in rewritten:
        return False

    prefix = rewritten.split(": ", 1)[0]
    original_lower = original.lower()

    for entity in prefix.split(", "):
        entity = entity.strip()
        if len(entity) >= 4 and entity.lower() not in original_lower:
            return True

    return False


# ---------------------------------------------------------------------------
# Tier 2 — LLM rewriting (Groq)
# ---------------------------------------------------------------------------

def _llm_rewrite(
    query:       str,
    turns:       list[Turn],
    groq_client,
    model:       str = _REWRITE_MODEL,
) -> str:
    """
    Send the conversation history + query to Groq and return a
    self-contained rewritten query.

    Falls back to the original query on any error.
    """
    history = buffer_as_text(turns[-8:])   # last 8 turns — enough context, not too many tokens
    prompt  = _REWRITE_PROMPT.format(history=history, question=query)

    try:
        t0 = time.monotonic()
        resp = groq_client.chat.completions.create(
            model       = model,
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = _MAX_TOKENS,
            temperature = _TEMPERATURE,
            timeout     = 15,   # rewriting is on the hot path — fail fast
        )
        raw = resp.choices[0].message.content.strip()

        # Strip Qwen3 thinking blocks
        raw = _THINK_RE.sub("", raw).strip()

        # Strip leading/trailing quotes the model sometimes adds
        raw = raw.strip('"').strip("'").strip()

        elapsed = (time.monotonic() - t0) * 1000
        logger.debug(f"[MEMORY] LLM rewrite done in {elapsed:.0f} ms: {raw!r:.80}")
        return raw if raw else query

    except Exception as exc:
        logger.warning(f"[MEMORY] LLM rewrite failed ({exc}) — using original query.")
        return query


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rewrite_query(
    query:       str,
    turns:       list[Turn],
    *,
    groq_client  = None,   # pass the shared Groq singleton from api.py
    model:       str = _REWRITE_MODEL,
    force_llm:   bool = False,   # skip Tier 0/1 and go straight to LLM
    session_id:  str = "",       # used for Tier-2 cache keying; empty = no caching
) -> tuple[str, str]:
    """
    Rewrite a follow-up query using conversation history.

    Parameters
    ----------
    query        : The user's latest raw question.
    turns        : The trimmed session buffer (list[Turn], chronological).
    groq_client  : Groq client singleton.  Pass None to disable Tier 2.
    model        : Groq model for Tier-2 rewriting.
    force_llm    : Skip Tier 0/1 and always use Tier-2 LLM rewriting.

    Returns
    -------
    (rewritten_query, tier_used)
    tier_used is one of: "skip" | "entity" | "llm"

    Examples
    --------
    >>> rewrite_query("What about the next quarter?", turns, groq_client=gc)
    ("What were the Q1 2025 financial results for Acme Corp?", "llm")

    >>> rewrite_query("What were the Q3 2024 revenue figures by region?", [])
    ("What were the Q3 2024 revenue figures by region?", "skip")

    >>> rewrite_query("And the operating margin?", turns)
    ("Q3 2024, $4.2B, Cloud Services: And the operating margin?", "entity")
    """
    if not turns:
        # No history → nothing to resolve → always skip
        return query, "skip"

    if not force_llm:
        # ── Tier 0: self-contained check ──────────────────────────────────
        if _is_self_contained(query):
            logger.debug(f"[MEMORY] rewrite tier=skip: {query!r:.60}")
            return query, "skip"

        # ── Tier 1: entity injection ───────────────────────────────────────
        entity_rewritten = _entity_inject(query, turns)
        if _entity_injection_helped(query, entity_rewritten):
            logger.debug(f"[MEMORY] rewrite tier=entity: {entity_rewritten!r:.80}")
            return entity_rewritten, "entity"

    # ── Tier 2: LLM rewriting ──────────────────────────────────────────────
    if groq_client is None:
        # No Groq client available — fall back to entity result or raw query
        fallback = entity_rewritten if not force_llm else _entity_inject(query, turns)
        logger.debug(f"[MEMORY] rewrite tier=entity (no groq client): {fallback!r:.80}")
        return fallback, "entity"

    # Check Tier-2 cache before paying the Groq round-trip cost.
    if session_id:
        cached = _cache_get(session_id, query)
        if cached is not None:
            logger.debug(f"[MEMORY] rewrite tier=llm (cache hit): {cached[0]!r:.80}")
            return cached

    rewritten = _llm_rewrite(query, turns, groq_client, model=model)
    logger.debug(f"[MEMORY] rewrite tier=llm: {rewritten!r:.80}")

    # Populate cache for subsequent identical follow-ups in the same session.
    if session_id:
        _cache_set(session_id, query, rewritten, "llm")

    return rewritten, "llm"
