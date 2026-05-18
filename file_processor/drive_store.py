"""
drive_store.py — PostgreSQL-backed Drive file permission store.

Table created automatically on first use:
  drive_files(file_id PK, name, mime_type, owner_email,
              allowed_users TEXT[], is_public BOOL, crawled_at TIMESTAMPTZ)

Falls back to an in-process dict when PostgreSQL is unavailable.
The in-process store is lost on server restart — use PostgreSQL in production.

Public API:
  upsert_file(file_id, name, mime_type, owner_email, allowed_users, is_public)
  get_file(file_id)                   → dict | None
  get_allowed_users(file_id)          → list[str]
  is_file_accessible(file_id, email)  → bool
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)

_PG_PARAMS = {
    "host":     os.getenv("PG_HOST",     "localhost"),
    "port":     int(os.getenv("PG_PORT", "5432")),
    "dbname":   os.getenv("PG_DB",       "csvstore"),
    "user":     os.getenv("PG_USER",     "csvuser"),
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
                    CREATE TABLE IF NOT EXISTS drive_files (
                        file_id       TEXT PRIMARY KEY,
                        name          TEXT,
                        mime_type     TEXT,
                        owner_email   TEXT NOT NULL,
                        allowed_users TEXT[],
                        is_public     BOOLEAN DEFAULT false,
                        crawled_at    TIMESTAMPTZ DEFAULT now()
                    )
                """)
                # Safe migration for deployments that predate this table
                cur.execute("""
                    ALTER TABLE drive_files
                    ADD COLUMN IF NOT EXISTS allowed_users TEXT[]
                """)
                cur.execute("""
                    ALTER TABLE drive_files
                    ADD COLUMN IF NOT EXISTS is_public BOOLEAN DEFAULT false
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_drive_files_owner
                    ON drive_files(owner_email)
                """)
            conn.commit()
        logger.info("[DRIVE STORE] drive_files table ready")
        return True
    except Exception as e:
        logger.warning(f"[DRIVE STORE] PostgreSQL unavailable — using in-process fallback: {e}")
        return False


_PG_OK = _ensure_tables()

# In-process fallback: file_id → row dict
_FILES: dict[str, dict] = {}


# ── Public API ────────────────────────────────────────────────────────────────

def upsert_file(
    file_id:       str,
    name:          str,
    mime_type:     str,
    owner_email:   str,
    allowed_users: list[str],
    is_public:     bool,
) -> None:
    """Insert or update a Drive file's permission record."""
    if _PG_OK:
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO drive_files
                            (file_id, name, mime_type, owner_email,
                             allowed_users, is_public, crawled_at)
                        VALUES (%s, %s, %s, %s, %s, %s, now())
                        ON CONFLICT (file_id) DO UPDATE SET
                            name          = EXCLUDED.name,
                            mime_type     = EXCLUDED.mime_type,
                            owner_email   = EXCLUDED.owner_email,
                            allowed_users = EXCLUDED.allowed_users,
                            is_public     = EXCLUDED.is_public,
                            crawled_at    = now()
                    """, (file_id, name, mime_type, owner_email,
                          allowed_users, is_public))
                conn.commit()
            return
        except Exception as e:
            logger.error(f"[DRIVE STORE] upsert_file: {e}")

    # Fallback
    _FILES[file_id] = {
        "file_id":       file_id,
        "name":          name,
        "mime_type":     mime_type,
        "owner_email":   owner_email,
        "allowed_users": allowed_users,
        "is_public":     is_public,
        "crawled_at":    datetime.now(timezone.utc).isoformat(),
    }


def get_file(file_id: str) -> Optional[dict]:
    """Return the stored permission record for a file, or None if not found."""
    if _PG_OK:
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT file_id, name, mime_type, owner_email,
                               allowed_users, is_public, crawled_at
                        FROM drive_files WHERE file_id = %s
                    """, (file_id,))
                    row = cur.fetchone()
            if row is None:
                return None
            return {
                "file_id":       row[0],
                "name":          row[1],
                "mime_type":     row[2],
                "owner_email":   row[3],
                "allowed_users": row[4] or [],
                "is_public":     row[5],
                "crawled_at":    row[6].isoformat() if row[6] else None,
            }
        except Exception as e:
            logger.error(f"[DRIVE STORE] get_file: {e}")
            return None

    return _FILES.get(file_id)


def get_allowed_users(file_id: str) -> list[str]:
    """Return the list of emails allowed to access a file."""
    record = get_file(file_id)
    if record is None:
        return []
    return record.get("allowed_users") or []


def is_file_accessible(file_id: str, email: str) -> bool:
    """Return True if the given email can access the file (or the file is public)."""
    record = get_file(file_id)
    if record is None:
        return False
    if record.get("is_public"):
        return True
    return email in (record.get("allowed_users") or [])


def list_drive_files(owner_email: str, limit: int = 200) -> list[dict]:
    """
    Return up to *limit* Drive file records owned by *owner_email*,
    ordered by most-recently crawled first.

    Each record is a dict with keys:
      file_id, name, mime_type, owner_email, allowed_users, is_public, crawled_at
    """
    if _PG_OK:
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT file_id, name, mime_type, owner_email,
                               allowed_users, is_public, crawled_at
                        FROM drive_files
                        WHERE owner_email = %s
                        ORDER BY crawled_at DESC
                        LIMIT %s
                    """, (owner_email, limit))
                    rows = cur.fetchall()
            return [
                {
                    "file_id":       r[0],
                    "name":          r[1],
                    "mime_type":     r[2],
                    "owner_email":   r[3],
                    "allowed_users": r[4] or [],
                    "is_public":     r[5],
                    "crawled_at":    r[6].isoformat() if r[6] else None,
                }
                for r in rows
            ]
        except Exception as e:
            logger.error(f"[DRIVE STORE] list_drive_files: {e}")
            return []

    # In-process fallback
    results = [
        rec for rec in _FILES.values()
        if rec.get("owner_email") == owner_email
    ]
    results.sort(key=lambda r: r.get("crawled_at") or "", reverse=True)
    return results[:limit]
