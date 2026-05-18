"""
feedback/store.py — Customer Support Feedback Store
=====================================================

Persists per-interaction feedback (thumbs up/down ratings, escalation outcome,
intent classification) to help measure and improve RAG quality over time.

Storage backends (priority order):
  1. PostgreSQL — table: support_feedback
     (created automatically on first use; idempotent)
  2. In-process list — transparent fallback when PostgreSQL is unavailable.
     Entries are lost on server restart in this mode.

Table schema
------------
    CREATE TABLE support_feedback (
        id            SERIAL PRIMARY KEY,
        session_id    TEXT,
        user_id       TEXT,
        question      TEXT,
        answer        TEXT,
        intent        TEXT,
        rating        INTEGER CHECK (rating IN (-1, 1)),   -- 1=helpful, -1=not helpful
        escalated     BOOLEAN  DEFAULT FALSE,
        escalation_reason TEXT,
        eval_verdict  TEXT,
        confidence    FLOAT,
        created_at    TIMESTAMPTZ DEFAULT NOW()
    );

Public API
----------
    store_feedback(entry)            → str   (feedback_id)
    get_feedback_summary()           → dict
    list_feedback(limit, offset)     → list[dict]
    delete_feedback(feedback_id)     → bool
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from datetime import datetime, timezone

from loguru import logger


def _ts_to_iso(ts: float) -> str:
    """Convert a Unix timestamp float to an ISO-8601 UTC string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

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
        from sqlalchemy import create_engine, text as _text  # type: ignore
        host     = os.environ.get("PG_HOST")
        port     = os.environ.get("PG_PORT", "5432")
        db       = os.environ.get("PG_DB")
        user     = os.environ.get("PG_USER")
        password = os.environ.get("PG_PASSWORD")

        if not all([host, db, user, password]):
            return False

        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
        engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 5})

        with engine.connect() as conn:
            conn.execute(_text("""
                CREATE TABLE IF NOT EXISTS support_feedback (
                    id               SERIAL PRIMARY KEY,
                    feedback_id      TEXT UNIQUE NOT NULL,
                    session_id       TEXT,
                    user_id          TEXT,
                    question         TEXT,
                    answer           TEXT,
                    intent           TEXT,
                    rating           INTEGER CHECK (rating IN (-1, 0, 1)),
                    escalated        BOOLEAN DEFAULT FALSE,
                    escalation_reason TEXT,
                    eval_verdict     TEXT,
                    confidence       FLOAT,
                    elapsed_ms       FLOAT,
                    created_at       TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            conn.execute(_text("""
                CREATE INDEX IF NOT EXISTS idx_feedback_session_id
                ON support_feedback (session_id)
            """))
            conn.execute(_text("""
                CREATE INDEX IF NOT EXISTS idx_feedback_intent
                ON support_feedback (intent)
            """))
            conn.commit()

        _pg_engine    = engine
        _PG_AVAILABLE = True
        logger.info("[FEEDBACK] PostgreSQL backend initialised (support_feedback table ready)")
        return True

    except Exception as exc:
        logger.warning(f"[FEEDBACK] PostgreSQL unavailable — using in-process fallback: {exc}")
        return False


# ---------------------------------------------------------------------------
# In-process fallback store
# ---------------------------------------------------------------------------

_MEM_STORE: list[dict] = []
_MEM_LOCK  = threading.Lock()


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

@dataclass
class FeedbackEntry:
    """A single feedback record."""
    question:          str
    answer:            str

    # Optional fields — filled in by the API handler
    session_id:        str   = ""
    user_id:           str   = ""
    intent:            str   = ""
    rating:            int   = 0    # 1=helpful, -1=not helpful, 0=not rated
    escalated:         bool  = False
    escalation_reason: str   = ""
    eval_verdict:      str   = ""
    confidence:        Optional[float] = None
    elapsed_ms:        Optional[float] = None

    # Auto-assigned
    feedback_id:       str   = field(default_factory=lambda: str(uuid.uuid4()))
    created_at:        float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "feedback_id":       self.feedback_id,
            "session_id":        self.session_id,
            "user_id":           self.user_id,
            "question":          self.question,
            "answer":            self.answer,
            "intent":            self.intent,
            "rating":            self.rating,
            "escalated":         self.escalated,
            "escalation_reason": self.escalation_reason,
            "eval_verdict":      self.eval_verdict,
            "confidence":        self.confidence,
            "elapsed_ms":        self.elapsed_ms,
            # Always return an ISO-8601 string so the format is consistent
            # whether the entry was stored in PostgreSQL or the in-process dict.
            "created_at":        _ts_to_iso(self.created_at),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def store_feedback(entry: FeedbackEntry) -> str:
    """
    Persist a feedback entry.

    Returns the feedback_id (UUID string).
    """
    _init_pg()

    if _PG_AVAILABLE and _pg_engine is not None:
        try:
            from sqlalchemy import text as _text
            with _pg_engine.connect() as conn:
                conn.execute(_text("""
                    INSERT INTO support_feedback
                        (feedback_id, session_id, user_id, question, answer,
                         intent, rating, escalated, escalation_reason,
                         eval_verdict, confidence, elapsed_ms)
                    VALUES
                        (:feedback_id, :session_id, :user_id, :question, :answer,
                         :intent, :rating, :escalated, :escalation_reason,
                         :eval_verdict, :confidence, :elapsed_ms)
                    ON CONFLICT (feedback_id) DO NOTHING
                """), {
                    "feedback_id":       entry.feedback_id,
                    "session_id":        entry.session_id,
                    "user_id":           entry.user_id,
                    "question":          entry.question[:4000],   # guard against giant inputs
                    "answer":            entry.answer[:8000],
                    "intent":            entry.intent,
                    "rating":            entry.rating,
                    "escalated":         entry.escalated,
                    "escalation_reason": entry.escalation_reason,
                    "eval_verdict":      entry.eval_verdict,
                    "confidence":        entry.confidence,
                    "elapsed_ms":        entry.elapsed_ms,
                })
                conn.commit()
            logger.debug(f"[FEEDBACK] Stored {entry.feedback_id} in PostgreSQL")
            return entry.feedback_id
        except Exception as exc:
            logger.warning(f"[FEEDBACK] PG write failed: {exc} — falling back to in-process")

    # In-process fallback
    with _MEM_LOCK:
        _MEM_STORE.append(entry.to_dict())
    logger.debug(f"[FEEDBACK] Stored {entry.feedback_id} in memory (PG unavailable)")
    return entry.feedback_id


def get_feedback_summary() -> dict:
    """
    Return aggregate statistics across all stored feedback.

    Returns:
        {
          "total":           int,
          "helpful":         int,           # rating == 1
          "not_helpful":     int,           # rating == -1
          "escalated":       int,
          "by_intent":       {intent: count, ...},
          "by_verdict":      {verdict: count, ...},
          "avg_confidence":  float | None,
          "satisfaction_pct": float | None, # helpful / (helpful + not_helpful) * 100
        }
    """
    _init_pg()

    if _PG_AVAILABLE and _pg_engine is not None:
        try:
            from sqlalchemy import text as _text
            with _pg_engine.connect() as conn:
                row = conn.execute(_text("""
                    SELECT
                        COUNT(*)                                           AS total,
                        SUM(CASE WHEN rating =  1 THEN 1 ELSE 0 END)      AS helpful,
                        SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END)      AS not_helpful,
                        SUM(CASE WHEN escalated THEN 1 ELSE 0 END)        AS escalated,
                        AVG(confidence)                                    AS avg_confidence
                    FROM support_feedback
                """)).fetchone()

                by_intent = {
                    r[0]: r[1]
                    for r in conn.execute(_text("""
                        SELECT intent, COUNT(*) FROM support_feedback
                        WHERE intent IS NOT NULL AND intent != ''
                        GROUP BY intent ORDER BY 2 DESC
                    """)).fetchall()
                }

                by_verdict = {
                    r[0]: r[1]
                    for r in conn.execute(_text("""
                        SELECT eval_verdict, COUNT(*) FROM support_feedback
                        WHERE eval_verdict IS NOT NULL AND eval_verdict != ''
                        GROUP BY eval_verdict ORDER BY 2 DESC
                    """)).fetchall()
                }

            total       = int(row[0] or 0)
            helpful     = int(row[1] or 0)
            not_helpful = int(row[2] or 0)
            escalated   = int(row[3] or 0)
            avg_conf    = float(row[4]) if row[4] is not None else None

            rated = helpful + not_helpful
            satisfaction = round(helpful / rated * 100, 1) if rated > 0 else None

            return {
                "total":            total,
                "helpful":          helpful,
                "not_helpful":      not_helpful,
                "escalated":        escalated,
                "by_intent":        by_intent,
                "by_verdict":       by_verdict,
                "avg_confidence":   round(avg_conf, 3) if avg_conf else None,
                "satisfaction_pct": satisfaction,
            }
        except Exception as exc:
            logger.warning(f"[FEEDBACK] PG summary failed: {exc} — using in-process data")

    # In-process fallback
    with _MEM_LOCK:
        entries = list(_MEM_STORE)

    total       = len(entries)
    helpful     = sum(1 for e in entries if e.get("rating") == 1)
    not_helpful = sum(1 for e in entries if e.get("rating") == -1)
    escalated   = sum(1 for e in entries if e.get("escalated"))
    confidences = [e["confidence"] for e in entries if e.get("confidence") is not None]
    avg_conf    = sum(confidences) / len(confidences) if confidences else None

    by_intent: dict[str, int] = {}
    by_verdict: dict[str, int] = {}
    for e in entries:
        if e.get("intent"):
            by_intent[e["intent"]] = by_intent.get(e["intent"], 0) + 1
        if e.get("eval_verdict"):
            by_verdict[e["eval_verdict"]] = by_verdict.get(e["eval_verdict"], 0) + 1

    rated = helpful + not_helpful
    satisfaction = round(helpful / rated * 100, 1) if rated > 0 else None

    return {
        "total":            total,
        "helpful":          helpful,
        "not_helpful":      not_helpful,
        "escalated":        escalated,
        "by_intent":        by_intent,
        "by_verdict":       by_verdict,
        "avg_confidence":   round(avg_conf, 3) if avg_conf else None,
        "satisfaction_pct": satisfaction,
    }


def list_feedback(limit: int = 50, offset: int = 0) -> list[dict]:
    """Return recent feedback entries (newest first)."""
    _init_pg()

    if _PG_AVAILABLE and _pg_engine is not None:
        try:
            from sqlalchemy import text as _text
            with _pg_engine.connect() as conn:
                rows = conn.execute(_text("""
                    SELECT feedback_id, session_id, user_id, question, intent,
                           rating, escalated, escalation_reason, eval_verdict,
                           confidence, elapsed_ms, created_at
                    FROM support_feedback
                    ORDER BY created_at DESC
                    LIMIT :limit OFFSET :offset
                """), {"limit": limit, "offset": offset}).fetchall()
            return [
                {
                    "feedback_id":       r[0],
                    "session_id":        r[1],
                    "user_id":           r[2],
                    "question":          r[3],
                    "intent":            r[4],
                    "rating":            r[5],
                    "escalated":         r[6],
                    "escalation_reason": r[7],
                    "eval_verdict":      r[8],
                    "confidence":        r[9],
                    "elapsed_ms":        r[10],
                    "created_at":        str(r[11]),
                }
                for r in rows
            ]
        except Exception as exc:
            logger.warning(f"[FEEDBACK] PG list failed: {exc}")

    # In-process fallback
    with _MEM_LOCK:
        entries = list(reversed(_MEM_STORE))
    page = entries[offset: offset + limit]
    # Normalise created_at to ISO string (stored as float in _MEM_STORE)
    for e in page:
        if isinstance(e.get("created_at"), float):
            e = dict(e)
            e["created_at"] = _ts_to_iso(e["created_at"])
    return page


def delete_feedback(feedback_id: str) -> bool:
    """Delete a single feedback record by ID. Returns True if found and deleted."""
    _init_pg()

    if _PG_AVAILABLE and _pg_engine is not None:
        try:
            from sqlalchemy import text as _text
            with _pg_engine.connect() as conn:
                result = conn.execute(_text(
                    "DELETE FROM support_feedback WHERE feedback_id = :fid"
                ), {"fid": feedback_id})
                conn.commit()
            return result.rowcount > 0
        except Exception as exc:
            logger.warning(f"[FEEDBACK] PG delete failed: {exc}")

    with _MEM_LOCK:
        before = len(_MEM_STORE)
        _MEM_STORE[:] = [e for e in _MEM_STORE if e.get("feedback_id") != feedback_id]
        return len(_MEM_STORE) < before
