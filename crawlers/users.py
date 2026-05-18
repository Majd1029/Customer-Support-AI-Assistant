"""
crawlers/users.py — Lightweight user registry for the crawler layer.

Backed by PostgreSQL (same PG_* env vars used by the rest of the project).
Falls back to an in-process dict when PostgreSQL is unavailable so the
server still starts in development mode without a database.

Schema (auto-created on first use):

    CREATE TABLE crawler_users (
        id           TEXT PRIMARY KEY,
        email        TEXT UNIQUE NOT NULL,
        is_admin     BOOLEAN     DEFAULT FALSE,
        crawl_status TEXT        DEFAULT 'pending',   -- pending|running|done|failed
        last_crawl   TIMESTAMPTZ,
        created_at   TIMESTAMPTZ DEFAULT NOW()
    );

CLI usage:
    python crawlers/users.py list
    python crawlers/users.py add alice alice@gmail.com
    python crawlers/users.py add alice alice@gmail.com --admin
    python crawlers/users.py delete alice
"""

from __future__ import annotations

import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

# ── Load .env so callers don't need to ────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

# ── PostgreSQL connection (optional) ─────────────────────────────────────────

_pg_conn = None          # module-level psycopg2 connection
_pg_lock = threading.Lock()
_PG_AVAILABLE = False

try:
    import psycopg2
    import psycopg2.extras

    _pg_conn = psycopg2.connect(
        host     = os.getenv("PG_HOST",     "localhost"),
        port     = int(os.getenv("PG_PORT", "5432")),
        dbname   = os.getenv("PG_DB",       "csvstore"),
        user     = os.getenv("PG_USER",     "csvuser"),
        password = os.getenv("PG_PASSWORD", ""),
    )
    _pg_conn.autocommit = True
    _PG_AVAILABLE = True

    # Auto-create table
    with _pg_conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS crawler_users (
                id           TEXT PRIMARY KEY,
                email        TEXT UNIQUE NOT NULL,
                is_admin     BOOLEAN     DEFAULT FALSE,
                crawl_status TEXT        DEFAULT 'pending',
                last_crawl   TIMESTAMPTZ,
                created_at   TIMESTAMPTZ DEFAULT NOW()
            )
        """)

except Exception as _pg_err:
    print(f"[USERS] PostgreSQL unavailable ({_pg_err}) — using in-process fallback")
    _PG_AVAILABLE = False


# ── In-process fallback store ─────────────────────────────────────────────────

_MEM_STORE: dict[str, dict] = {}   # id → user dict
_MEM_LOCK  = threading.Lock()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _pg_fetch_one(query: str, params: tuple) -> dict | None:
    with _pg_lock:
        with _pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            row = cur.fetchone()
    return dict(row) if row else None


def _pg_fetch_all(query: str, params: tuple = ()) -> list[dict]:
    with _pg_lock:
        with _pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def _pg_execute(query: str, params: tuple) -> None:
    with _pg_lock:
        with _pg_conn.cursor() as cur:
            cur.execute(query, params)


# ── Public API ────────────────────────────────────────────────────────────────

def get_user(user_id: str) -> dict | None:
    """Return one user as a dict, or None if not found."""
    if _PG_AVAILABLE:
        return _pg_fetch_one(
            "SELECT * FROM crawler_users WHERE id = %s", (user_id,)
        )
    with _MEM_LOCK:
        return dict(_MEM_STORE[user_id]) if user_id in _MEM_STORE else None


def get_user_by_email(email: str) -> dict | None:
    """Look up a user by their email address."""
    if _PG_AVAILABLE:
        return _pg_fetch_one(
            "SELECT * FROM crawler_users WHERE email = %s", (email,)
        )
    with _MEM_LOCK:
        for u in _MEM_STORE.values():
            if u.get("email") == email:
                return dict(u)
    return None


def load_users() -> dict[str, dict]:
    """Return all users as {id: user_dict}."""
    if _PG_AVAILABLE:
        rows = _pg_fetch_all("SELECT * FROM crawler_users ORDER BY id")
        return {r["id"]: r for r in rows}
    with _MEM_LOCK:
        return {k: dict(v) for k, v in _MEM_STORE.items()}


def is_admin(user_id: str) -> bool:
    user = get_user(user_id)
    return bool(user and user.get("is_admin"))


def upsert_user(
    user_id:  str,
    email:    str,
    admin:    bool = False,
) -> dict:
    """
    Insert or update a user.  Safe to call repeatedly (idempotent).
    Returns the saved user dict.
    """
    if _PG_AVAILABLE:
        _pg_execute("""
            INSERT INTO crawler_users (id, email, is_admin)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO UPDATE
                SET email    = EXCLUDED.email,
                    is_admin = EXCLUDED.is_admin
        """, (user_id, email, admin))
        return get_user(user_id)

    record = {
        "id":           user_id,
        "email":        email,
        "is_admin":     admin,
        "crawl_status": "pending",
        "last_crawl":   None,
    }
    with _MEM_LOCK:
        existing = _MEM_STORE.get(user_id, {})
        record["crawl_status"] = existing.get("crawl_status", "pending")
        record["last_crawl"]   = existing.get("last_crawl")
        _MEM_STORE[user_id] = record
    return dict(record)


def update_crawl_status(
    user_id:    str,
    status:     str,
    last_crawl: datetime | None = None,
) -> None:
    """Update crawl_status and optionally last_crawl timestamp."""
    if _PG_AVAILABLE:
        if last_crawl is not None:
            _pg_execute("""
                UPDATE crawler_users
                SET crawl_status = %s, last_crawl = %s
                WHERE id = %s
            """, (status, last_crawl, user_id))
        else:
            _pg_execute("""
                UPDATE crawler_users
                SET crawl_status = %s
                WHERE id = %s
            """, (status, user_id))
        return

    with _MEM_LOCK:
        if user_id in _MEM_STORE:
            _MEM_STORE[user_id]["crawl_status"] = status
            if last_crawl is not None:
                _MEM_STORE[user_id]["last_crawl"] = last_crawl.isoformat()


def delete_user(user_id: str) -> None:
    """
    Remove a user from the registry.
    Does NOT delete their OAuth token file or Qdrant data — do that separately.
    """
    if _PG_AVAILABLE:
        _pg_execute("DELETE FROM crawler_users WHERE id = %s", (user_id,))
        return
    with _MEM_LOCK:
        _MEM_STORE.pop(user_id, None)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"

    if cmd == "list":
        users = load_users()
        if not users:
            print("No users registered.")
        else:
            print(f"{'ID':<20} {'EMAIL':<35} {'ROLE':<8} {'STATUS':<10} LAST_CRAWL")
            print("-" * 90)
            for uid, u in users.items():
                role   = "ADMIN" if u.get("is_admin") else "user"
                crawl  = str(u.get("last_crawl") or "never")[:19]
                status = u.get("crawl_status", "?")
                print(f"  {uid:<20} {u['email']:<35} [{role}]   {status:<10} {crawl}")

    elif cmd == "add":
        if len(sys.argv) < 4:
            print("Usage: python crawlers/users.py add <id> <email> [--admin]")
            sys.exit(1)
        uid   = sys.argv[2]
        email = sys.argv[3]
        adm   = "--admin" in sys.argv
        user  = upsert_user(uid, email, admin=adm)
        print(f"Saved: {user}")

    elif cmd == "delete":
        if len(sys.argv) < 3:
            print("Usage: python crawlers/users.py delete <id>")
            sys.exit(1)
        delete_user(sys.argv[2])
        print(f"Deleted: {sys.argv[2]}")

    else:
        print("Commands:")
        print("  list")
        print("  add <id> <email> [--admin]")
        print("  delete <id>")
