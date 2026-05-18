"""
file_preparation/memory/summariser.py

Incremental session summariser — Sprint 3 of the memory layer.

When a session's un-absorbed token total exceeds SUMMARISE_THRESHOLD (80% of
TOKEN_BUDGET) this module folds the oldest evictable turns into a compact
rolling summary stored in buffer._SUMMARIES.  The summarised turns are then
removed from the live buffer via absorb_turns(), preventing them from appearing
redundantly in future retrieval queries or generation prompts.

The rolling summary accumulates across calls: each new summary prepends the
prior one so no context is permanently lost.

Designed to run as a FastAPI BackgroundTask — never on the hot path.

Architecture
------------
    summarise_session(session_id, user_id, groq_client)
        → read_all_turns()         : get every turn without budget trimming
        → trim_buffer()            : identify which turns WOULD survive the budget
        → turns to summarise       : those NOT in the trim result (oldest evictable)
        → Groq qwen/qwen3-32b      : generate compact summary (<= 200 words)
        → write_summary()          : persist updated rolling summary
        → absorb_turns()           : remove summarised turns from live buffer

    should_summarise(session_id)   : cheap pre-flight check (no Groq needed)

Concurrency
-----------
    _SUMMARISING keeps a set of session IDs currently being summarised.
    A second BackgroundTask for the same session exits immediately if the
    first has not yet completed — a silent no-op, not an error.

Usage in api.py
---------------
    from file_preparation.memory.summariser import summarise_session

    # In the /ask BackgroundTask, after write_turn():
    if should_summarise(req.session_id):
        background_tasks.add_task(
            summarise_session,
            req.session_id,
            req.session_id,   # user_id
            _groq_client,
        )
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from loguru import logger

from file_preparation.utils.groq_retry import call_groq_with_retry as _call_groq_with_retry
from .buffer import (
    Turn,
    TOKEN_BUDGET,
    read_all_turns,
    get_summary,
    write_summary,
    absorb_turns,
    trim_buffer,
    buffer_as_text,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SUMMARISE_MODEL    = "qwen/qwen3-32b"
_SUMMARISE_TOKENS   = 512     # generous for a 200-word summary + Qwen3 thinking
_SUMMARISE_TEMP     = 0.0     # deterministic
_SUMMARISE_TIMEOUT  = 30      # fail fast — summarisation is not user-facing

# Maximum chars of prior summary to carry forward (prevents unbounded growth)
_MAX_PRIOR_SUMMARY_CHARS = 800

_SUMMARISE_PROMPT = """\
You are a conversation summariser for a document Q&A system.

PRIOR SUMMARY (may be empty):
{prior_summary}

NEW CONVERSATION TURNS TO SUMMARISE:
{turns_text}

Write a single concise paragraph (≤ 200 words) that captures:
- The topics or documents the user asked about
- Key facts, figures, or conclusions established so far
- Any unresolved questions or follow-ups the user raised

Merge the prior summary and new turns into one unified summary.
Output ONLY the summary paragraph. No preamble, no labels, no bullet points."""


# ---------------------------------------------------------------------------
# Concurrency guard — prevents two BackgroundTasks from summarising the
# same session simultaneously (wastes a Groq call; last write wins but
# the first absorb leaves the second with nothing to absorb, causing a
# confusing empty-summary write).
# ---------------------------------------------------------------------------

_SUMMARISING:      set[str] = set()
_SUMMARISING_LOCK: threading.Lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarise_session(
    session_id:  str,
    user_id:     str,
    groq_client,
    *,
    model: str = _SUMMARISE_MODEL,
) -> bool:
    """
    Summarise the oldest evictable turns in a session and absorb them.

    Parameters
    ----------
    session_id   : Session to summarise.
    user_id      : Must match the session owner.
    groq_client  : Groq SDK client (pass None to skip — returns False).
    model        : Groq model for summarisation.

    Returns
    -------
    True if a summary was generated and turns were absorbed; False otherwise.
    """
    if groq_client is None:
        logger.debug("[MEMORY] summariser: no groq client — skipping.")
        return False

    # ── Concurrency guard — one summarisation per session at a time ───────────
    with _SUMMARISING_LOCK:
        if session_id in _SUMMARISING:
            logger.debug(
                f"[MEMORY] summariser: session {session_id[:8]}… already being "
                "summarised — skipping duplicate task."
            )
            return False
        _SUMMARISING.add(session_id)

    try:
        return _do_summarise(session_id, user_id, groq_client, model)
    finally:
        with _SUMMARISING_LOCK:
            _SUMMARISING.discard(session_id)


def _do_summarise(
    session_id:  str,
    user_id:     str,
    groq_client,
    model:       str,
) -> bool:
    """Inner implementation — called only when the concurrency guard is held."""

    # ── Identify turns to summarise ───────────────────────────────────────────
    # read_all_turns() returns the raw buffer without budget trimming.
    # We then run trim_buffer() ourselves to see which turns would survive
    # the window — the remainder (oldest turns) are the eviction candidates.
    all_turns = read_all_turns(session_id, user_id)
    if not all_turns:
        return False

    surviving_ids = {t.turn_id for t in trim_buffer(all_turns, budget=TOKEN_BUDGET)}
    to_summarise  = [t for t in all_turns if t.turn_id not in surviving_ids]

    if not to_summarise:
        logger.debug(f"[MEMORY] summariser: no evictable turns for session {session_id[:8]}…")
        return False

    # ── Retrieve prior rolling summary ────────────────────────────────────────
    prior = get_summary(session_id, user_id) or ""
    # Cap prior summary to prevent unbounded prompt growth
    if len(prior) > _MAX_PRIOR_SUMMARY_CHARS:
        prior = prior[:_MAX_PRIOR_SUMMARY_CHARS] + "…"

    turns_text = buffer_as_text(to_summarise)

    # ── Call Groq (with 429 retry) ─────────────────────────────────────────────
    prompt = _SUMMARISE_PROMPT.format(
        prior_summary = prior or "(none — this is the first summarisation for this session)",
        turns_text    = turns_text,
    )

    try:
        t0  = time.monotonic()
        raw = _call_groq_with_retry(
            groq_client,
            model        = model,
            prompt       = prompt,
            max_tokens   = _SUMMARISE_TOKENS,
            temperature  = _SUMMARISE_TEMP,
            timeout      = _SUMMARISE_TIMEOUT,
            label        = "[MEMORY] summariser",
            strip_think  = True,
        )
        raw     = raw.strip('"').strip("'").strip()
        elapsed = (time.monotonic() - t0) * 1000

        if not raw:
            logger.warning("[MEMORY] summariser: Groq returned empty summary — skipping.")
            return False

        logger.info(
            f"[MEMORY] summariser: {len(to_summarise)} turns → {len(raw)} char summary "
            f"({elapsed:.0f}ms) for session {session_id[:8]}…"
        )

    except Exception as exc:
        logger.warning(f"[MEMORY] summariser: Groq call failed ({exc}) — skipping.")
        return False

    # ── Persist summary and absorb turns ──────────────────────────────────────
    try:
        write_summary(session_id, user_id, raw)
        absorb_turns(session_id, user_id, len(to_summarise))
    except Exception as exc:
        logger.warning(f"[MEMORY] summariser: write/absorb failed ({exc}).")
        return False

    return True
