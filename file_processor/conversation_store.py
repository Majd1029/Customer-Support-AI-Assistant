"""
conversation_store.py — PostgreSQL-backed conversation and message persistence.

Tables created automatically:
  conversations(session_id PK, user_id, label, share_token, created_at, last_active)
  conversation_messages(id BIGSERIAL PK, session_id, user_id, role, content, metadata TEXT, created_at)

Public API:
  create_or_touch_session(session_id, user_id, label) → bool
  update_session_label(session_id, user_id, label)    → bool
  list_sessions(user_id)                              → list[{id, label, createdAt, lastActive, shareToken}]
  save_messages(session_id, user_id, messages)        → bool
  load_messages(session_id, user_id)                  → list[{role, content, metadata}]
  delete_session(session_id, user_id)                 → bool
  share_conversation(session_id, user_id)             → str  (share token, idempotent)
  get_shared_conversation(share_token)                → dict | None
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)

from loguru import logger

_PG_PARAMS = {
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     int(os.getenv("PG_PORT", "5432")),
    "dbname":   os.getenv("PG_DB", "csvstore"),
    "user":     os.getenv("PG_USER", "csvuser"),
    "password": os.getenv("PG_PASSWORD", ""),
}


def _get_conn():
    import psycopg2
    return psycopg2.connect(**_PG_PARAMS)


def _ensure_tables() -> bool:
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        session_id   TEXT PRIMARY KEY,
                        user_id      TEXT NOT NULL,
                        label        TEXT NOT NULL DEFAULT 'New chat',
                        share_token  TEXT UNIQUE,
                        created_at   TIMESTAMPTZ DEFAULT now(),
                        last_active  TIMESTAMPTZ DEFAULT now()
                    )
                """)
                # Migrations: add columns to existing tables
                cur.execute("""
                    ALTER TABLE conversations
                    ADD COLUMN IF NOT EXISTS share_token TEXT UNIQUE
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_conversations_user_id
                    ON conversations(user_id)
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                        id          BIGSERIAL PRIMARY KEY,
                        session_id  TEXT NOT NULL
                            REFERENCES conversations(session_id) ON DELETE CASCADE,
                        user_id     TEXT NOT NULL,
                        role        TEXT NOT NULL,
                        content     TEXT NOT NULL,
                        metadata    TEXT,
                        created_at  TIMESTAMPTZ DEFAULT now()
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_conv_messages_session
                    ON conversation_messages(session_id, id)
                """)
                # Add FK constraint on existing tables (idempotent via DO NOTHING)
                cur.execute("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.table_constraints
                            WHERE constraint_name = 'fk_conv_messages_session'
                        ) THEN
                            ALTER TABLE conversation_messages
                            ADD CONSTRAINT fk_conv_messages_session
                            FOREIGN KEY (session_id)
                            REFERENCES conversations(session_id) ON DELETE CASCADE;
                        END IF;
                    END $$;
                """)
            conn.commit()
        logger.info("[CONV] conversation tables ready (with FK constraints)")
        return True
    except Exception as e:
        logger.warning(f"[CONV] PostgreSQL unavailable — using in-process fallback: {e}")
        return False


_PG_OK = _ensure_tables()

# In-process fallback
_SESSIONS:  dict[str, dict] = {}   # session_id → row
_MESSAGES:  dict[str, list] = {}   # session_id → [row, ...]


def create_or_touch_session(session_id: str, user_id: str, label: str = "New chat") -> bool:
    if _PG_OK:
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO conversations (session_id, user_id, label)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (session_id) DO UPDATE SET last_active = now()
                    """, (session_id, user_id, label))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"[CONV] create_or_touch_session: {e}")
            return False
    else:
        if session_id not in _SESSIONS:
            _SESSIONS[session_id] = {
                "session_id": session_id, "user_id": user_id, "label": label,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_active": datetime.now(timezone.utc).isoformat(),
            }
        return True


def update_session_label(session_id: str, user_id: str, label: str) -> bool:
    if _PG_OK:
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE conversations SET label=%s, last_active=now() "
                        "WHERE session_id=%s AND user_id=%s",
                        (label, session_id, user_id),
                    )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"[CONV] update_session_label: {e}")
            return False
    else:
        s = _SESSIONS.get(session_id)
        if s and s["user_id"] == user_id:
            s["label"] = label
        return True


def list_sessions(user_id: str) -> list[dict]:
    """Return user's sessions ordered newest-first."""
    if _PG_OK:
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT session_id, label,
                               EXTRACT(EPOCH FROM created_at)*1000,
                               EXTRACT(EPOCH FROM last_active)*1000,
                               share_token
                        FROM conversations
                        WHERE user_id=%s
                        ORDER BY last_active DESC
                        LIMIT 200
                    """, (user_id,))
                    rows = cur.fetchall()
            return [
                {
                    "id": r[0], "label": r[1],
                    "createdAt": int(r[2]), "lastActive": int(r[3]),
                    "shareToken": r[4],
                }
                for r in rows
            ]
        except Exception as e:
            logger.error(f"[CONV] list_sessions: {e}")
            return []
    else:
        return sorted(
            [
                {
                    "id": s["session_id"], "label": s["label"],
                    "createdAt": 0, "lastActive": 0,
                    "shareToken": s.get("share_token"),
                }
                for s in _SESSIONS.values()
                if s["user_id"] == user_id
            ],
            reverse=True,
        )


def save_messages(session_id: str, user_id: str, messages: list[dict]) -> bool:
    """Replace all messages for a session (full overwrite). Validates ownership."""
    if _PG_OK:
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    # Ensure session exists and belongs to this user.
                    # INSERT…ON CONFLICT so the very first save also creates the session.
                    cur.execute("""
                        INSERT INTO conversations (session_id, user_id, label, last_active)
                        VALUES (%s, %s, 'Chat', now())
                        ON CONFLICT (session_id) DO UPDATE
                            SET last_active = now()
                            WHERE conversations.user_id = EXCLUDED.user_id
                    """, (session_id, user_id))
                    # If the session belongs to a *different* user the ON CONFLICT DO UPDATE
                    # WHERE clause silently fails — rowcount=0 means ownership check failed.
                    cur.execute(
                        "SELECT 1 FROM conversations WHERE session_id=%s AND user_id=%s",
                        (session_id, user_id),
                    )
                    if not cur.fetchone():
                        logger.warning(
                            f"[CONV] save_messages: session {session_id!r} "
                            f"does not belong to user {user_id!r} — write blocked"
                        )
                        return False
                    # Full overwrite of messages (safe because ownership confirmed above)
                    cur.execute(
                        "DELETE FROM conversation_messages WHERE session_id=%s",
                        (session_id,),
                    )
                    for msg in messages:
                        meta = msg.get("metadata")
                        cur.execute(
                            """INSERT INTO conversation_messages
                               (session_id, user_id, role, content, metadata)
                               VALUES (%s,%s,%s,%s,%s)""",
                            (session_id, user_id, msg["role"], msg["content"],
                             json.dumps(meta) if meta is not None else None),
                        )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"[CONV] save_messages: {e}")
            return False
    else:
        # Fallback: verify ownership in-process
        s = _SESSIONS.get(session_id)
        if s and s["user_id"] != user_id:
            return False
        _MESSAGES[session_id] = list(messages)
        return True


def load_messages(session_id: str, user_id: str) -> list[dict]:
    """Return messages for the session (ownership-checked), ordered by id."""
    if _PG_OK:
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    # Ownership check
                    cur.execute(
                        "SELECT 1 FROM conversations WHERE session_id=%s AND user_id=%s",
                        (session_id, user_id),
                    )
                    if not cur.fetchone():
                        return []
                    cur.execute(
                        "SELECT role, content, metadata FROM conversation_messages "
                        "WHERE session_id=%s ORDER BY id",
                        (session_id,),
                    )
                    rows = cur.fetchall()
            return [
                {"role": r[0], "content": r[1], "metadata": json.loads(r[2]) if r[2] else None}
                for r in rows
            ]
        except Exception as e:
            logger.error(f"[CONV] load_messages: {e}")
            return []
    else:
        s = _SESSIONS.get(session_id)
        if not s or s["user_id"] != user_id:
            return []
        return _MESSAGES.get(session_id, [])


def delete_session(session_id: str, user_id: str) -> bool:
    if _PG_OK:
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM conversation_messages WHERE session_id=%s AND user_id=%s",
                        (session_id, user_id),
                    )
                    cur.execute(
                        "DELETE FROM conversations WHERE session_id=%s AND user_id=%s",
                        (session_id, user_id),
                    )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"[CONV] delete_session: {e}")
            return False
    else:
        _SESSIONS.pop(session_id, None)
        _MESSAGES.pop(session_id, None)
        return True


def share_conversation(session_id: str, user_id: str) -> Optional[str]:
    """Generate a permanent share token for the conversation (idempotent).

    Returns the share token, or None if the session does not belong to user_id.
    Calling this multiple times returns the same token (no regeneration).
    """
    if _PG_OK:
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    # Check ownership and fetch existing token in one query
                    cur.execute(
                        "SELECT share_token FROM conversations "
                        "WHERE session_id=%s AND user_id=%s",
                        (session_id, user_id),
                    )
                    row = cur.fetchone()
                    if row is None:
                        return None  # not found or not owned by this user
                    if row[0]:
                        return row[0]  # already shared — return existing token
                    # Generate and persist a fresh token
                    token = secrets.token_urlsafe(16)
                    cur.execute(
                        "UPDATE conversations SET share_token=%s "
                        "WHERE session_id=%s AND user_id=%s",
                        (token, session_id, user_id),
                    )
                conn.commit()
            return token
        except Exception as e:
            logger.error(f"[CONV] share_conversation: {e}")
            return None
    else:
        s = _SESSIONS.get(session_id)
        if not s or s["user_id"] != user_id:
            return None
        if not s.get("share_token"):
            s["share_token"] = secrets.token_urlsafe(16)
        return s["share_token"]


def get_shared_conversation(share_token: str) -> Optional[dict]:
    """Return {label, session_id, messages} for a shared conversation token.

    No authentication required — caller must have obtained the token via a
    share link.  Returns None if the token is unknown.
    """
    if _PG_OK:
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT session_id, label FROM conversations "
                        "WHERE share_token=%s",
                        (share_token,),
                    )
                    row = cur.fetchone()
                    if not row:
                        return None
                    session_id, label = row
                    cur.execute(
                        "SELECT role, content, metadata FROM conversation_messages "
                        "WHERE session_id=%s ORDER BY id",
                        (session_id,),
                    )
                    msgs = cur.fetchall()
            return {
                "session_id": session_id,
                "label":      label,
                "messages": [
                    {
                        "role":     m[0],
                        "content":  m[1],
                        "metadata": json.loads(m[2]) if m[2] else None,
                    }
                    for m in msgs
                ],
            }
        except Exception as e:
            logger.error(f"[CONV] get_shared_conversation: {e}")
            return None
    else:
        for s in _SESSIONS.values():
            if s.get("share_token") == share_token:
                sid = s["session_id"]
                return {
                    "session_id": sid,
                    "label":      s["label"],
                    "messages":   _MESSAGES.get(sid, []),
                }
        return None
