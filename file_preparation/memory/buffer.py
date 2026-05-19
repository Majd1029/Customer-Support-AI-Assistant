"""
file_preparation/memory/buffer.py

Conversation buffer — short-term, per-session turn storage.

Storage backend
---------------
Two backends, selected at module import time:

  Redis  (preferred for production / multi-worker)
         Activated when REDIS_URL is set in the environment.
         All state is stored externally — any number of Uvicorn workers
         can read and write the same sessions without coordination.
         Key layout:
           session:{session_id}:turns    → Redis list (RPUSH / LRANGE)
           session:{session_id}:summary  → Redis string (SET / GET)
           session:{session_id}:owner    → Redis string — user_id ownership
         All keys carry a TTL of SESSION_TTL seconds, refreshed on every write.

  In-process dict  (default for local dev / single-worker)
         Activated when REDIS_URL is not set (or redis package is unavailable).
         Thread-safe via threading.Lock.  Sessions are lost on restart.
         Sufficient for single-worker research / CI.

Both backends expose the same public API so callers never need to branch:
    write_turn(), read_buffer(), read_all_turns(), clear_session(),
    write_summary(), get_summary(), absorb_turns(),
    format_buffer_for_prompt(), buffer_as_text(), list_sessions(),
    session_token_total(), should_summarise(), trim_buffer()

Turn lifecycle
--------------
    write_turn()       → appends a turn, then trims to token budget
    read_buffer()      → returns trimmed, chronologically ordered turns
    clear_session()    → wipes all turns for a session (e.g. on logout)
    format_buffer_for_prompt() → list[{"role": ..., "content": ...}] for the LLM

Windowing strategy
------------------
Token-budget window (recommended in memory_layer_design.md §2.3 Strategy B).
The buffer keeps as many recent turns as fit within TOKEN_BUDGET tokens,
always preserving the last MIN_KEEP_TURNS turns verbatim even if they
exceed the budget — discontinuity is worse than a slight overflow.

Token counts are computed once at write time using tiktoken (cl100k_base),
the same encoder used by the rest of the pipeline.  Falls back to
len(content) // 4 if tiktoken is unavailable.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from loguru import logger
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

# ---------------------------------------------------------------------------
# Token counting — reuse the project's cached tiktoken encoder
# ---------------------------------------------------------------------------

try:
    import tiktoken as _tiktoken
    _enc = _tiktoken.get_encoding("cl100k_base")
    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception:
    logger.warning("[MEMORY] tiktoken unavailable — using len//4 fallback for token counts.")
    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOKEN_BUDGET:       int = 1_500
SUMMARISE_THRESHOLD: int = int(TOKEN_BUDGET * 0.8)   # 1 200 tokens
MIN_KEEP_TURNS:     int = 2
SESSION_TTL:        int = 7_200    # 2 hours (in seconds)

_REDIS_URL: str | None = os.getenv("REDIS_URL")


# ---------------------------------------------------------------------------
# Turn dataclass
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    turn_id:          str
    session_id:       str
    user_id:          str
    role:             str
    content:          str
    timestamp:        float
    token_count:      int
    summary_absorbed: bool

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(**d)


# ---------------------------------------------------------------------------
# Token-budget trimming (backend-agnostic, used by both backends)
# ---------------------------------------------------------------------------

def trim_buffer(turns: list[Turn], budget: int = TOKEN_BUDGET) -> list[Turn]:
    """
    Return the most recent turns that fit within `budget` tokens.

    The last MIN_KEEP_TURNS turns are always included even if they exceed
    the budget — context discontinuity is worse than a slight overflow.
    Strategy B from memory_layer_design.md §2.3.
    """
    if not turns:
        return []

    must_keep  = turns[-MIN_KEEP_TURNS:]
    candidates = turns[:-MIN_KEEP_TURNS]

    kept: list[Turn] = []
    remaining = budget - sum(t.token_count for t in must_keep)
    for turn in reversed(candidates):
        if remaining <= 0:
            break
        kept.append(turn)
        remaining -= turn.token_count

    return list(reversed(kept)) + must_keep


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------

class _RedisBackend:
    """
    Redis-backed conversation buffer.

    Each session uses three keys:
      - session:{sid}:turns   → Redis list of JSON-encoded Turn dicts
      - session:{sid}:summary → Redis string (rolling summary text)
      - session:{sid}:owner   → Redis string (user_id — for ownership checks)

    All keys are given SESSION_TTL-second TTLs, refreshed on every write.
    """

    def __init__(self, url: str) -> None:
        import redis
        self._r = redis.from_url(url, decode_responses=True)
        self._r.ping()
        logger.info(f"[MEMORY] Redis backend active ({url})")

    # ── Key helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _k_turns(sid: str) -> str:   return f"session:{sid}:turns"
    @staticmethod
    def _k_summary(sid: str) -> str: return f"session:{sid}:summary"
    @staticmethod
    def _k_owner(sid: str) -> str:   return f"session:{sid}:owner"

    def _refresh_ttl(self, sid: str) -> None:
        """Refresh TTL on all session keys after any write."""
        for key in (self._k_turns(sid), self._k_summary(sid), self._k_owner(sid)):
            self._r.expire(key, SESSION_TTL)

    # ── Ownership ────────────────────────────────────────────────────────────

    def _get_owner(self, sid: str) -> str | None:
        res = self._r.get(self._k_owner(sid))
        return str(res) if res is not None else None

    def _set_owner(self, sid: str, user_id: str) -> None:
        self._r.set(self._k_owner(sid), user_id, ex=SESSION_TTL)

    def _check_owner(self, sid: str, user_id: str) -> bool:
        """Return True if user_id is the owner (or session is new)."""
        owner = self._get_owner(sid)
        return owner is None or owner == user_id

    # ── Turn I/O ─────────────────────────────────────────────────────────────

    def write_turn(self, session_id: str, user_id: str, role: str, content: str) -> Turn:
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role {role!r}")

        owner = self._get_owner(session_id)
        if owner and owner != user_id:
            raise PermissionError(f"Session {session_id!r} belongs to a different user.")

        turn = Turn(
            turn_id          = str(uuid.uuid4()),
            session_id       = session_id,
            user_id          = user_id,
            role             = role,
            content          = content,
            timestamp        = time.time(),
            token_count      = _count_tokens(content),
            summary_absorbed = False,
        )
        pipe = self._r.pipeline()
        pipe.rpush(self._k_turns(session_id), json.dumps(turn.to_dict()))
        pipe.set(self._k_owner(session_id), user_id)
        for key in (self._k_turns(session_id), self._k_summary(session_id),
                    self._k_owner(session_id)):
            pipe.expire(key, SESSION_TTL)
        pipe.execute()

        logger.debug(
            f"[MEMORY/Redis] write_turn session={session_id[:8]}… role={role} "
            f"tokens={turn.token_count}"
        )
        return turn

    def _all_turns(self, session_id: str) -> list[Turn]:
        raw = self._r.lrange(self._k_turns(session_id), 0, -1)
        if not isinstance(raw, list):
            raw = []
        return [Turn.from_dict(json.loads(str(r))) for r in raw]

    def read_all_turns(self, session_id: str, user_id: str) -> list[Turn]:
        if not self._check_owner(session_id, user_id):
            logger.warning(f"[MEMORY/Redis] read_all_turns: ownership denied for {user_id!r}")
            return []
        return self._all_turns(session_id)

    def read_buffer(self, session_id: str, user_id: str,
                    budget: int = TOKEN_BUDGET) -> list[Turn]:
        if not self._check_owner(session_id, user_id):
            logger.warning(f"[MEMORY/Redis] read_buffer: ownership denied for {user_id!r}")
            return []
        return trim_buffer(self._all_turns(session_id), budget=budget)

    def clear_session(self, session_id: str, user_id: str) -> int:
        if not self._check_owner(session_id, user_id):
            raise PermissionError(f"Session {session_id!r} does not belong to {user_id!r}.")
        count = self._r.llen(self._k_turns(session_id))
        if not isinstance(count, int):
            count = 0
        pipe = self._r.pipeline()
        pipe.delete(self._k_turns(session_id))
        pipe.delete(self._k_summary(session_id))
        pipe.delete(self._k_owner(session_id))
        pipe.execute()
        logger.info(f"[MEMORY/Redis] cleared session {session_id[:8]}… ({count} turns)")
        return count

    def session_token_total(self, session_id: str) -> int:
        turns = self._all_turns(session_id)
        return sum(t.token_count for t in turns if not t.summary_absorbed)

    # ── Summary ──────────────────────────────────────────────────────────────

    def write_summary(self, session_id: str, user_id: str, summary_text: str) -> None:
        if not self._check_owner(session_id, user_id):
            raise PermissionError(f"Session {session_id!r} does not belong to {user_id!r}.")
        self._r.set(self._k_summary(session_id), summary_text.strip(), ex=SESSION_TTL)
        self._refresh_ttl(session_id)
        logger.debug(
            f"[MEMORY/Redis] summary written for session {session_id[:8]}… "
            f"({len(summary_text)} chars)"
        )

    def get_summary(self, session_id: str, user_id: str) -> str | None:
        if not self._check_owner(session_id, user_id):
            return None
        res = self._r.get(self._k_summary(session_id))
        return str(res) if res is not None else None

    # ── Absorb ───────────────────────────────────────────────────────────────

    def absorb_turns(self, session_id: str, user_id: str, count: int) -> list[Turn]:
        if not self._check_owner(session_id, user_id):
            raise PermissionError(f"Session {session_id!r} does not belong to {user_id!r}.")

        all_turns = self._all_turns(session_id)
        candidates = [t for t in all_turns if not t.summary_absorbed]
        to_absorb  = candidates[:count]
        if not to_absorb:
            return []

        absorbed_ids = {t.turn_id for t in to_absorb}
        remaining    = [t for t in all_turns if t.turn_id not in absorbed_ids]

        # Rewrite the Redis list atomically
        k = self._k_turns(session_id)
        pipe = self._r.pipeline()
        pipe.delete(k)
        for t in remaining:
            pipe.rpush(k, json.dumps(t.to_dict()))
        pipe.expire(k, SESSION_TTL)
        pipe.execute()

        logger.debug(
            f"[MEMORY/Redis] absorbed {len(to_absorb)} turns from session {session_id[:8]}…"
        )
        return to_absorb

    # ── Admin ────────────────────────────────────────────────────────────────

    def list_sessions(self) -> list[dict]:
        """Scan Redis for all session turn keys and return summary dicts."""
        result = []
        for key in self._r.scan_iter("session:*:turns"):
            sid   = key.split(":")[1]
            turns = self._all_turns(sid)
            if turns:
                result.append({
                    "session_id":   sid,
                    "user_id":      turns[0].user_id,
                    "turn_count":   len(turns),
                    "total_tokens": sum(t.token_count for t in turns),
                    "oldest":       turns[0].timestamp,
                    "newest":       turns[-1].timestamp,
                })
        return result


# ---------------------------------------------------------------------------
# In-process dict backend (default)
# ---------------------------------------------------------------------------

class _DictBackend:
    """Thread-safe in-process dict — single-worker only."""

    def __init__(self) -> None:
        self._store:     dict[str, list[Turn]] = {}
        self._summaries: dict[str, str]        = {}
        self._lock = threading.Lock()
        logger.debug("[MEMORY] In-process dict backend active.")

    def _evict_expired(self) -> None:
        cutoff  = time.time() - SESSION_TTL
        expired = [
            sid for sid, turns in self._store.items()
            if turns and turns[-1].timestamp < cutoff
        ]
        for sid in expired:
            del self._store[sid]
            self._summaries.pop(sid, None)
        if expired:
            logger.debug(f"[MEMORY] evicted {len(expired)} expired session(s)")

    def write_turn(self, session_id: str, user_id: str, role: str, content: str) -> Turn:
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role {role!r} — must be 'user' or 'assistant'.")

        turn = Turn(
            turn_id          = str(uuid.uuid4()),
            session_id       = session_id,
            user_id          = user_id,
            role             = role,
            content          = content,
            timestamp        = time.time(),
            token_count      = _count_tokens(content),
            summary_absorbed = False,
        )
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = []

            existing = self._store[session_id]
            if existing and existing[0].user_id != user_id:
                raise PermissionError(f"Session {session_id!r} belongs to a different user.")

            self._store[session_id].append(turn)
            self._evict_expired()

        logger.debug(
            f"[MEMORY] write_turn session={session_id[:8]}… role={role} "
            f"tokens={turn.token_count}"
        )
        return turn

    def read_all_turns(self, session_id: str, user_id: str) -> list[Turn]:
        with self._lock:
            turns = self._store.get(session_id, [])
            if not turns:
                return []
            if turns[0].user_id != user_id:
                logger.warning(
                    f"[MEMORY] read_all_turns: user {user_id!r} denied on "
                    f"session {session_id[:8]}…"
                )
                return []
            return list(turns)

    def read_buffer(self, session_id: str, user_id: str,
                    budget: int = TOKEN_BUDGET) -> list[Turn]:
        with self._lock:
            turns = self._store.get(session_id, [])
            if not turns:
                return []
            if turns[0].user_id != user_id:
                logger.warning(
                    f"[MEMORY] read_buffer: user {user_id!r} denied on "
                    f"session {session_id[:8]}…"
                )
                return []
            return trim_buffer(list(turns), budget=budget)

    def clear_session(self, session_id: str, user_id: str) -> int:
        with self._lock:
            turns = self._store.get(session_id, [])
            if not turns:
                return 0
            if turns[0].user_id != user_id:
                raise PermissionError(
                    f"Session {session_id!r} does not belong to user {user_id!r}."
                )
            count = len(turns)
            del self._store[session_id]
            self._summaries.pop(session_id, None)
        logger.info(f"[MEMORY] cleared session {session_id[:8]}… ({count} turns)")
        return count

    def session_token_total(self, session_id: str) -> int:
        with self._lock:
            turns = self._store.get(session_id, [])
            return sum(t.token_count for t in turns if not t.summary_absorbed)

    def write_summary(self, session_id: str, user_id: str, summary_text: str) -> None:
        with self._lock:
            turns = self._store.get(session_id, [])
            if turns and turns[0].user_id != user_id:
                raise PermissionError(
                    f"Session {session_id!r} does not belong to user {user_id!r}."
                )
            self._summaries[session_id] = summary_text.strip()
        logger.debug(
            f"[MEMORY] summary written for session {session_id[:8]}… "
            f"({len(summary_text)} chars)"
        )

    def get_summary(self, session_id: str, user_id: str) -> str | None:
        with self._lock:
            turns = self._store.get(session_id, [])
            if turns and turns[0].user_id != user_id:
                logger.warning(
                    f"[MEMORY] get_summary: user {user_id!r} denied on "
                    f"session {session_id[:8]}…"
                )
                return None
            return self._summaries.get(session_id)

    def absorb_turns(self, session_id: str, user_id: str, count: int) -> list[Turn]:
        with self._lock:
            turns = self._store.get(session_id, [])
            if not turns:
                return []
            if turns[0].user_id != user_id:
                raise PermissionError(
                    f"Session {session_id!r} does not belong to user {user_id!r}."
                )
            candidates   = [t for t in turns if not t.summary_absorbed]
            to_absorb    = candidates[:count]
            absorbed_ids = {t.turn_id for t in to_absorb}
            self._store[session_id] = [t for t in turns if t.turn_id not in absorbed_ids]

        logger.debug(
            f"[MEMORY] absorbed {len(to_absorb)} turns from session {session_id[:8]}…"
        )
        return to_absorb

    def list_sessions(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "session_id":   sid,
                    "user_id":      turns[0].user_id if turns else "?",
                    "turn_count":   len(turns),
                    "total_tokens": sum(t.token_count for t in turns),
                    "oldest":       turns[0].timestamp if turns else None,
                    "newest":       turns[-1].timestamp if turns else None,
                }
                for sid, turns in self._store.items()
            ]


# ---------------------------------------------------------------------------
# Backend selection (module-level singleton)
# ---------------------------------------------------------------------------

def _build_backend() -> "_RedisBackend | _DictBackend":
    if _REDIS_URL:
        try:
            return _RedisBackend(_REDIS_URL)
        except Exception as exc:
            logger.warning(
                f"[MEMORY] Redis unavailable ({exc}) — "
                f"falling back to in-process dict (single-worker only)."
            )
    return _DictBackend()


_backend: "_RedisBackend | _DictBackend" = _build_backend()


# ---------------------------------------------------------------------------
# Public API  (thin delegates to whichever backend is active)
# ---------------------------------------------------------------------------

def write_turn(session_id: str, user_id: str, role: str, content: str) -> Turn:
    """Append one turn to the session buffer."""
    return _backend.write_turn(session_id, user_id, role, content)


def read_all_turns(session_id: str, user_id: str) -> list[Turn]:
    """Return every turn without budget trimming (for summariser / fact extractor)."""
    return _backend.read_all_turns(session_id, user_id)


def read_buffer(session_id: str, user_id: str, budget: int = TOKEN_BUDGET) -> list[Turn]:
    """Return token-budget-trimmed, chronologically ordered turns."""
    return _backend.read_buffer(session_id, user_id, budget=budget)


def clear_session(session_id: str, user_id: str) -> int:
    """Delete all turns for a session. Returns the count deleted."""
    return _backend.clear_session(session_id, user_id)


def session_token_total(session_id: str) -> int:
    """Return total token count of all un-absorbed turns."""
    return _backend.session_token_total(session_id)


def should_summarise(session_id: str) -> bool:
    """Return True when un-absorbed tokens exceed SUMMARISE_THRESHOLD (80% of budget)."""
    return session_token_total(session_id) >= SUMMARISE_THRESHOLD


def write_summary(session_id: str, user_id: str, summary_text: str) -> None:
    """Persist a rolling summary, replacing any prior one."""
    _backend.write_summary(session_id, user_id, summary_text)


def get_summary(session_id: str, user_id: str) -> str | None:
    """Return the rolling summary, or None if none exists."""
    return _backend.get_summary(session_id, user_id)


def absorb_turns(session_id: str, user_id: str, count: int) -> list[Turn]:
    """Remove the oldest `count` un-absorbed turns and return them."""
    return _backend.absorb_turns(session_id, user_id, count)


def list_sessions() -> list[dict]:
    """Return a summary of all active sessions."""
    return _backend.list_sessions()


# ---------------------------------------------------------------------------
# Prompt formatting helpers
# ---------------------------------------------------------------------------

def format_buffer_for_prompt(turns: list[Turn]) -> list[dict]:
    """Convert turns to [{"role": ..., "content": ...}] for the LLM."""
    return [{"role": t.role, "content": t.content} for t in turns]


def buffer_as_text(turns: list[Turn]) -> str:
    """Flat text representation: [user] ... / [assistant] ..."""
    return "\n".join(f"[{t.role}] {t.content}" for t in turns)


# ---------------------------------------------------------------------------
# Backend identification (for /health endpoint and tests)
# ---------------------------------------------------------------------------

def backend_name() -> str:
    """Return 'redis' or 'dict' — useful for /health and test assertions."""
    return "redis" if isinstance(_backend, _RedisBackend) else "dict"
