"""
file_preparation/memory/orchestrator.py

MemoryContext — the object that carries all memory data into the RAG pipeline.
load_memory_context() — assembles it with concurrent reads, never blocking the
hot path on a single failure.

This is the only file the api.py /ask handler needs to import directly.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from .buffer import Turn, read_buffer, get_summary, format_buffer_for_prompt, buffer_as_text
from .rewriter import rewrite_query
from .user_memory import get_facts, format_facts_for_prompt
from .semantic_memory import recall_preferences, format_preferences_for_prompt


# ---------------------------------------------------------------------------
# MemoryContext — carries all memory data for one request
# ---------------------------------------------------------------------------

@dataclass
class MemoryContext:
    """
    All memory data assembled for a single /ask request.

    Fields
    ------
    session_id           : The session this context belongs to.
    user_id              : The authenticated user.
    turns                : Trimmed buffer turns (chronological, token-budget trimmed).
    rewritten_query      : Query after rewriting (may equal original if self-contained).
    rewrite_tier         : Which rewrite tier was used ("skip" | "entity" | "llm").
    summary              : Rolling summary of older absorbed turns (None if not yet generated).
    user_facts           : Sprint 4 — structured key-value facts about the user (persists across sessions).
    recalled_preferences : Sprint 5 — semantically relevant free-text preferences recalled for this query.
    enabled              : False if memory_enabled=False was passed — all fields empty.
    """
    session_id:           str
    user_id:              str
    turns:                list[Turn]       = field(default_factory=list)
    rewritten_query:      str              = ""
    rewrite_tier:         str              = "skip"
    summary:              Optional[str]    = None
    user_facts:           dict             = field(default_factory=dict)
    recalled_preferences: list[str]        = field(default_factory=list)
    enabled:              bool             = True

    @property
    def has_history(self) -> bool:
        return bool(self.turns)

    def prompt_messages(self) -> list[dict]:
        """Return turns as [{"role": ..., "content": ...}] for the LLM."""
        return format_buffer_for_prompt(self.turns)

    def history_as_text(self) -> str:
        """Return buffer as flat text for injection into system/user prompts."""
        return buffer_as_text(self.turns)

    def summary_as_context(self) -> str:
        """
        Return the rolling summary formatted for injection into GenerationConfig.
        Returns an empty string when no summary exists.
        """
        return self.summary.strip() if self.summary else ""

    def user_facts_as_context(self) -> str:
        """
        Return structured user facts formatted for prompt injection (Sprint 4).
        Returns an empty string when no facts are stored.
        """
        return format_facts_for_prompt(self.user_facts)

    def recalled_preferences_as_context(self) -> str:
        """
        Return semantically recalled preferences formatted for prompt injection (Sprint 5).
        Returns an empty string when no preferences were recalled.
        """
        return format_preferences_for_prompt(self.recalled_preferences)

    def to_dict(self) -> dict:
        return {
            "session_id":            self.session_id,
            "user_id":               self.user_id,
            "turn_count":            len(self.turns),
            "rewrite_tier":          self.rewrite_tier,
            "rewritten_query":       self.rewritten_query,
            "has_summary":           self.summary is not None,
            "user_facts_count":      len(self.user_facts),
            "recalled_prefs_count":  len(self.recalled_preferences),
            "enabled":               self.enabled,
        }


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

async def load_memory_context(
    session_id:   str,
    user_id:      str,
    raw_query:    str,
    *,
    groq_client   = None,
    memory_enabled: bool = True,
) -> MemoryContext:
    """
    Build a MemoryContext for the current request.

    Reads the session buffer and rewrites the query concurrently where possible.
    Any failure in a component degrades gracefully — the request always proceeds.

    Parameters
    ----------
    session_id     : Caller-managed session ID (UUID from the client/cookie).
    user_id        : Authenticated user ID (from JWT, never from request body).
    raw_query      : The user's original question.
    groq_client    : Groq singleton for Tier-2 LLM rewriting. Pass None to limit
                     rewriting to entity injection (Tier 1).
    memory_enabled : When False, returns an empty MemoryContext immediately.
                     Use this to A/B test stateless vs. memory-augmented requests.

    Returns
    -------
    MemoryContext with populated turns and rewritten_query.
    """
    if not memory_enabled:
        return MemoryContext(
            session_id      = session_id,
            user_id         = user_id,
            turns           = [],
            rewritten_query = raw_query,
            rewrite_tier    = "skip",
            enabled         = False,
        )

    # ── Read buffer, summary, user facts, and recalled preferences concurrently ─
    # All four are I/O-bound (dict lookups or DB/Qdrant queries).
    # asyncio.gather() runs them truly in parallel — each failure degrades
    # gracefully without blocking the others.
    loop = asyncio.get_running_loop()

    async def _read_turns() -> list[Turn]:
        try:
            return await loop.run_in_executor(None, read_buffer, session_id, user_id)
        except Exception as exc:
            logger.warning(f"[MEMORY] buffer read failed ({exc}) — proceeding without history.")
            return []

    async def _read_summary() -> Optional[str]:
        try:
            return await loop.run_in_executor(None, get_summary, session_id, user_id)
        except Exception as exc:
            logger.warning(f"[MEMORY] summary read failed ({exc}) — proceeding without summary.")
            return None

    async def _read_facts() -> dict:
        # Sprint 4 — structured user facts (persists across sessions via PostgreSQL)
        try:
            return await loop.run_in_executor(None, get_facts, user_id)
        except Exception as exc:
            logger.warning(f"[MEMORY] user facts read failed ({exc}) — proceeding without facts.")
            return {}

    async def _read_preferences() -> list[str]:
        # Sprint 5 — semantic preferences (recalled relative to the current query)
        try:
            return await loop.run_in_executor(None, recall_preferences, user_id, raw_query)
        except Exception as exc:
            logger.warning(f"[MEMORY] semantic recall failed ({exc}) — proceeding without preferences.")
            return []

    turns, summary, user_facts, recalled_preferences = await asyncio.gather(
        _read_turns(),
        _read_summary(),
        _read_facts(),
        _read_preferences(),
    )

    # ── Rewrite query ──────────────────────────────────────────────────────────
    rewritten_query = raw_query
    rewrite_tier    = "skip"
    if turns:
        try:
            rewritten_query, rewrite_tier = await loop.run_in_executor(
                None,
                lambda: rewrite_query(
                    raw_query, turns,
                    groq_client=groq_client,
                    session_id=session_id,
                ),
            )
        except Exception as exc:
            logger.warning(f"[MEMORY] query rewrite failed ({exc}) — using raw query.")
            rewritten_query = raw_query
            rewrite_tier    = "skip"

    if rewrite_tier != "skip":
        logger.info(
            f"[MEMORY] [{rewrite_tier}] {raw_query!r:.50} → {rewritten_query!r:.70}"
        )

    if user_facts:
        logger.debug(f"[MEMORY] {len(user_facts)} user fact(s) loaded for {user_id!r}")
    if recalled_preferences:
        logger.debug(f"[MEMORY] {len(recalled_preferences)} preference(s) recalled for {user_id!r}")

    return MemoryContext(
        session_id           = session_id,
        user_id              = user_id,
        turns                = turns,
        rewritten_query      = rewritten_query,
        rewrite_tier         = rewrite_tier,
        summary              = summary,
        user_facts           = user_facts,
        recalled_preferences = recalled_preferences,
        enabled              = True,
    )
