"""
file_preparation/memory/user_memory.py

Sprint 4 — Structured User Memory
===================================
Persistent key-value "user facts" that survive session boundaries.

Storage backends (priority order)
----------------------------------
1. PostgreSQL  — the same PG instance used by the CSV module.
   Table: ``user_facts``  (user_id TEXT, key TEXT, value TEXT, updated_at TIMESTAMPTZ)
   Created automatically on first use; idempotent.

2. In-process dict  — transparent fallback when PostgreSQL is unavailable
   (missing credentials, network down, psycopg2 not installed).
   Facts are lost on server restart in this mode, but all callers work identically.

Public API
----------
    set_fact(user_id, key, value)           → None
    get_facts(user_id)                      → dict[str, str]
    get_fact(user_id, key)                  → str | None
    delete_fact(user_id, key)               → bool
    clear_user_facts(user_id)               → int      (rows deleted)
    extract_and_store_facts(user_id, turns, groq_client) → dict[str, str]
    format_facts_for_prompt(facts)          → str

Thread safety
-------------
In-process dict operations are protected by ``_MEM_LOCK``.
PostgreSQL operations use per-call connections (psycopg2 is thread-safe at the
connection-pool level when you open a fresh connection per call).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

from loguru import logger
from file_preparation.utils.groq_retry import call_groq_with_retry as _groq_retry

# ---------------------------------------------------------------------------
# PostgreSQL backend (optional)
# ---------------------------------------------------------------------------

_PG_AVAILABLE = False
_pg_engine    = None

def _init_pg() -> bool:
    """Try to initialise a SQLAlchemy engine from env vars. Returns True on success."""
    global _PG_AVAILABLE, _pg_engine
    if _PG_AVAILABLE:
        return True
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")
    except ImportError:
        pass

    try:
        from sqlalchemy import create_engine, text as _text  # type: ignore
        host     = os.environ.get("PG_HOST")
        port     = os.environ.get("PG_PORT", "5432")
        db       = os.environ.get("PG_DB")
        user     = os.environ.get("PG_USER")
        password = os.environ.get("PG_PASSWORD")

        if not all([host, db, user, password]):
            logger.info("[USER_MEM] PostgreSQL env vars not set — using in-process fallback.")
            return False

        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
        _pg_engine = create_engine(url, pool_pre_ping=True, pool_size=3, max_overflow=5)

        # Create the user_facts table if it doesn't exist
        with _pg_engine.connect() as conn:
            conn.execute(_text("""
                CREATE TABLE IF NOT EXISTS user_facts (
                    user_id    TEXT        NOT NULL,
                    key        TEXT        NOT NULL,
                    value      TEXT        NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (user_id, key)
                )
            """))
            conn.commit()

        _PG_AVAILABLE = True
        logger.info("[USER_MEM] PostgreSQL backend ready (user_facts table ensured).")
        return True

    except Exception as exc:
        logger.warning(f"[USER_MEM] PostgreSQL unavailable ({exc}) — using in-process fallback.")
        return False


# Attempt PG init at module load — failure is silent (falls back to dict)
_init_pg()


# ---------------------------------------------------------------------------
# In-process fallback store
# ---------------------------------------------------------------------------

# Structure: { user_id: { key: value } }
_MEM_STORE: dict[str, dict[str, str]] = {}
_MEM_LOCK  = threading.Lock()


# ---------------------------------------------------------------------------
# Groq-backed fact extraction
# ---------------------------------------------------------------------------

_EXTRACT_MODEL   = "qwen/qwen3-32b"
_EXTRACT_TIMEOUT = 20    # seconds — on the background path, be generous
_THINK_RE        = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

_EXTRACT_SYSTEM = (
    "You extract factual statements the user has made about themselves from a "
    "conversation. Examples: name, language preference, job role, project name, "
    "goals, constraints.\n\n"
    "Return a JSON object whose keys are short snake_case identifiers "
    "(e.g. 'preferred_language', 'name', 'role', 'project') and whose values "
    "are concise strings. Only include facts explicitly stated by the user — "
    "never infer or hallucinate. Return {} if no facts are found.\n\n"
    "Output ONLY valid JSON, nothing else."
)


def _call_groq_extract(turns_text: str, groq_client) -> dict[str, str]:
    """
    Ask Groq to extract key-value user facts from a block of conversation text.
    Returns {} on any failure.
    """
    prompt = f"{_EXTRACT_SYSTEM}\n\nConversation:\n{turns_text}"
    try:
        raw = _groq_retry(
            groq_client,
            model       = _EXTRACT_MODEL,
            prompt      = prompt,
            max_tokens  = 512,
            temperature = 0.0,
            timeout     = _EXTRACT_TIMEOUT,
            label       = "[USER_MEM]",
            strip_think = True,
        )
        # Tolerate code-fenced JSON blocks
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw.strip())
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items() if v is not None}
    except Exception as exc:
        logger.debug(f"[USER_MEM] fact extraction failed: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Core CRUD — dispatches to PG or in-process dict
# ---------------------------------------------------------------------------

def set_fact(user_id: str, key: str, value: str) -> None:
    """
    Upsert a single user fact.

    Parameters
    ----------
    user_id : The user this fact belongs to.
    key     : Short identifier, e.g. "preferred_language", "name".
    value   : The fact value (always stored as a string).
    """
    key   = key.strip().lower()
    value = value.strip()
    if not key or not value:
        return

    if _PG_AVAILABLE and _pg_engine is not None:
        try:
            from sqlalchemy import text as _text
            with _pg_engine.connect() as conn:
                conn.execute(_text("""
                    INSERT INTO user_facts (user_id, key, value, updated_at)
                    VALUES (:uid, :key, :val, NOW())
                    ON CONFLICT (user_id, key)
                    DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                """), {"uid": user_id, "key": key, "val": value})
                conn.commit()
            logger.debug(f"[USER_MEM] PG set {user_id!r} {key!r}={value!r}")
            return
        except Exception as exc:
            logger.warning(f"[USER_MEM] PG set_fact failed ({exc}); falling back to dict.")

    # In-process fallback
    with _MEM_LOCK:
        _MEM_STORE.setdefault(user_id, {})[key] = value
    logger.debug(f"[USER_MEM] dict set {user_id!r} {key!r}={value!r}")


def get_facts(user_id: str) -> dict[str, str]:
    """Return all stored facts for a user as a dict. Empty dict if none."""
    if _PG_AVAILABLE and _pg_engine is not None:
        try:
            from sqlalchemy import text as _text
            with _pg_engine.connect() as conn:
                rows = conn.execute(
                    _text("SELECT key, value FROM user_facts WHERE user_id = :uid ORDER BY key"),
                    {"uid": user_id},
                ).fetchall()
            return {r[0]: r[1] for r in rows}
        except Exception as exc:
            logger.warning(f"[USER_MEM] PG get_facts failed ({exc}); falling back to dict.")

    with _MEM_LOCK:
        return dict(_MEM_STORE.get(user_id, {}))


def get_fact(user_id: str, key: str) -> str | None:
    """Return the value for a single key, or None if not set."""
    return get_facts(user_id).get(key.strip().lower())


def delete_fact(user_id: str, key: str) -> bool:
    """
    Delete a single user fact.

    Returns True if the fact existed and was deleted, False otherwise.
    """
    key = key.strip().lower()
    if _PG_AVAILABLE and _pg_engine is not None:
        try:
            from sqlalchemy import text as _text
            with _pg_engine.connect() as conn:
                result = conn.execute(
                    _text("DELETE FROM user_facts WHERE user_id = :uid AND key = :key"),
                    {"uid": user_id, "key": key},
                )
                conn.commit()
            return (result.rowcount or 0) > 0
        except Exception as exc:
            logger.warning(f"[USER_MEM] PG delete_fact failed ({exc}); falling back to dict.")

    with _MEM_LOCK:
        facts = _MEM_STORE.get(user_id, {})
        if key in facts:
            del facts[key]
            return True
        return False


def clear_user_facts(user_id: str) -> int:
    """
    Delete ALL facts for a user.

    Returns the number of rows deleted.
    """
    if _PG_AVAILABLE and _pg_engine is not None:
        try:
            from sqlalchemy import text as _text
            with _pg_engine.connect() as conn:
                result = conn.execute(
                    _text("DELETE FROM user_facts WHERE user_id = :uid"),
                    {"uid": user_id},
                )
                conn.commit()
            count = result.rowcount or 0
            logger.info(f"[USER_MEM] PG cleared {count} facts for {user_id!r}")
            return count
        except Exception as exc:
            logger.warning(f"[USER_MEM] PG clear_user_facts failed ({exc}); falling back to dict.")

    with _MEM_LOCK:
        facts = _MEM_STORE.pop(user_id, {})
    logger.info(f"[USER_MEM] dict cleared {len(facts)} facts for {user_id!r}")
    return len(facts)


# ---------------------------------------------------------------------------
# Auto-extraction
# ---------------------------------------------------------------------------

def extract_and_store_facts(
    user_id:     str,
    turns:       list,          # list[Turn] from buffer.py
    groq_client,
    max_turns:   int = 10,
) -> dict[str, str]:
    """
    Extract user facts from recent conversation turns and persist them.

    Only the most recent ``max_turns`` turns are examined to keep the Groq
    prompt compact.  Only *user* turns are passed to the extractor — assistant
    responses don't contain user-asserted facts.

    Returns the dict of newly extracted (and stored) facts.
    Called as a BackgroundTask from api.py — never on the critical request path.
    """
    if groq_client is None:
        return {}

    # Only consider user turns (assistant text isn't user-stated fact)
    user_turns = [t for t in turns if t.role == "user"][-max_turns:]
    if not user_turns:
        return {}

    turns_text = "\n".join(f"User: {t.content}" for t in user_turns)

    facts = _call_groq_extract(turns_text, groq_client)
    if not facts:
        return {}

    # Merge with existing facts — only overwrite when the new value is more specific
    # (simple heuristic: longer value wins for the same key)
    existing = get_facts(user_id)
    stored: dict[str, str] = {}
    for key, value in facts.items():
        if len(value) >= len(existing.get(key, "")):
            set_fact(user_id, key, value)
            stored[key] = value

    if stored:
        logger.info(f"[USER_MEM] stored {len(stored)} extracted facts for {user_id!r}: {list(stored.keys())}")
    return stored


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_facts_for_prompt(facts: dict[str, str]) -> str:
    """
    Format a facts dict as a compact block for injection into the system prompt.

    Example output:
        [USER FACTS]
        name: Majd
        preferred_language: Arabic
        role: ML Engineer
        project: Secure AI Assistant
        [END USER FACTS]
    """
    if not facts:
        return ""
    lines = "\n".join(f"{k}: {v}" for k, v in sorted(facts.items()))
    return f"[USER FACTS]\n{lines}\n[END USER FACTS]"
