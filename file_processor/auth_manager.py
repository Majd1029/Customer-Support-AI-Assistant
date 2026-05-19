"""
auth_manager.py — Server-side user authentication with role-based access control.

Provides:
  register_user(username, password, email, role) → (token | None, error | None)
  login_user(username_or_email, password)        → (user_dict | None, error | None)
  verify_token(token)                            → user_dict | None
  get_or_create_google_user(google_id, email, name) → user_dict | None
  get_user_by_id(user_id)                        → user_dict | None
  create_admin_user(username, password, email)   → (user_dict | None, error | None)

user_dict shape: {user_id, username, email, role, token}

Roles:
  "user"  — normal user, can only see their own docs + global KB
  "admin" — can upload to global KB, no retrieval restrictions
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)

from loguru import logger

# ── JWT ───────────────────────────────────────────────────────────────────────
JWT_SECRET       = os.getenv("JWT_SECRET", "dev-secret-change-me-in-production")
JWT_ALGO         = "HS256"
JWT_EXPIRY_HOURS = 720  # 30 days

try:
    import jwt as _jwt_lib
    _JWT_OK = True
except ImportError:
    _JWT_OK = False
    logger.warning("[AUTH] PyJWT not installed — install with: pip install PyJWT")


def _make_token(user_id: str, username: str, email: str = "", role: str = "user") -> str:
    if _JWT_OK:
        payload = {
            "sub":      user_id,
            "username": username,
            "email":    email,
            "role":     role,
            "exp":      datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
        }
        return _jwt_lib.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)
    # Fallback: simple HMAC-signed blob
    data = json.dumps({"user_id": user_id, "username": username, "email": email, "role": role})
    sig  = hmac.new(JWT_SECRET.encode(), data.encode(), hashlib.sha256).hexdigest()
    import base64
    b64 = base64.urlsafe_b64encode(data.encode()).decode().rstrip("=")
    return f"{b64}.{sig}"


def verify_token(token: str) -> Optional[dict]:
    """Return {user_id, username, email, role} or None if invalid/expired."""
    if _JWT_OK:
        try:
            p = _jwt_lib.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
            return {
                "user_id":  p["sub"],
                "username": p.get("username", ""),
                "email":    p.get("email", ""),
                "role":     p.get("role", "user"),
            }
        except Exception:
            return None
    try:
        import base64
        parts = token.split(".")
        if len(parts) != 2:
            return None
        padding = "=" * (-len(parts[0]) % 4)
        data = base64.urlsafe_b64decode(parts[0] + padding).decode()
        expected = hmac.new(JWT_SECRET.encode(), data.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(parts[1], expected):
            return None
        d = json.loads(data)
        return {
            "user_id":  d["user_id"],
            "username": d["username"],
            "email":    d.get("email", ""),
            "role":     d.get("role", "user"),
        }
    except Exception:
        return None


# ── Password hashing (PBKDF2-HMAC-SHA256) ────────────────────────────────────
def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk   = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
    return salt.hex() + "$" + dk.hex()


def _check_password(password: str, stored: str) -> bool:
    try:
        salt_hex, dk_hex = stored.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        dk   = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
        return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False


# ── PostgreSQL ────────────────────────────────────────────────────────────────
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
                    CREATE TABLE IF NOT EXISTS auth_users (
                        user_id       TEXT PRIMARY KEY,
                        username      TEXT UNIQUE NOT NULL,
                        email         TEXT UNIQUE,
                        password_hash TEXT,
                        google_id     TEXT UNIQUE,
                        role          TEXT NOT NULL DEFAULT 'user',
                        created_at    TIMESTAMPTZ DEFAULT now()
                    )
                """)
                # Migration: add role column to existing tables
                cur.execute("""
                    ALTER TABLE auth_users
                    ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'user'
                """)
            conn.commit()
        logger.info("[AUTH] auth_users table ready (role-aware)")
        return True
    except Exception as e:
        logger.warning(f"[AUTH] PostgreSQL unavailable — using in-process fallback: {e}")
        return False


_PG_OK = _ensure_tables()

# In-process fallback (single-worker dev)
_FALLBACK: dict[str, dict] = {}   # user_id → row dict


# ── Public API ────────────────────────────────────────────────────────────────

def _create_user_internal(
    username: str,
    password: str,
    email:    str,
    role:     str,
) -> tuple[Optional[str], Optional[str]]:
    """Internal helper that persists a new user with the given role.

    Returns (token, None) on success or (None, error_string) on failure.
    Never call this from untrusted request handlers — use the public functions.
    """
    uname = username.strip()
    if not uname or len(uname) < 2:
        return None, "Username must be at least 2 characters"
    if len(password) < 4:
        return None, "Password must be at least 4 characters"

    user_id = str(uuid.uuid4())
    ph      = _hash_password(password)
    em      = email.strip() or None

    if _PG_OK:
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 FROM auth_users WHERE lower(username)=lower(%s)", (uname,))
                    if cur.fetchone():
                        return None, "That username is already taken"
                    if em:
                        cur.execute("SELECT 1 FROM auth_users WHERE lower(email)=lower(%s)", (em,))
                        if cur.fetchone():
                            return None, "That email is already registered"
                    cur.execute(
                        "INSERT INTO auth_users (user_id, username, email, password_hash, role) "
                        "VALUES (%s,%s,%s,%s,%s)",
                        (user_id, uname, em, ph, role),
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"[AUTH] register error: {e}")
            return None, "Server error during registration"
    else:
        if any(u["username"].lower() == uname.lower() for u in _FALLBACK.values()):
            return None, "That username is already taken"
        _FALLBACK[user_id] = {
            "user_id": user_id, "username": uname,
            "email": em, "password_hash": ph, "google_id": None, "role": role,
        }

    return _make_token(user_id, uname, em or "", role), None


def register_user(
    username: str,
    password: str,
    email:    str = "",
    role:     str = "user",   # accepted but ALWAYS ignored — see security note
) -> tuple[Optional[str], Optional[str]]:
    """Register a new public user account.  Returns (token, None) on success.

    SECURITY: the ``role`` parameter is present only so call sites that
    previously passed ``role=`` do not raise a TypeError.  It is **always
    overridden to "user"** — the public registration path can never produce
    an admin account.  Use :func:`create_admin_user` for server-side admin
    creation.
    """
    return _create_user_internal(username, password, email, role="user")


def login_user(username_or_email: str, password: str) -> tuple[Optional[dict], Optional[str]]:
    """Returns (user_dict, None) on success or (None, error_string) on failure."""
    val = username_or_email.strip()

    if _PG_OK:
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT user_id, username, email, password_hash, role
                           FROM auth_users
                           WHERE lower(username)=lower(%s) OR lower(email)=lower(%s)""",
                        (val, val),
                    )
                    row = cur.fetchone()
        except Exception as e:
            logger.error(f"[AUTH] login error: {e}")
            return None, "Server error during login"
        if not row:
            return None, "Account not found"
        uid, uname, em, ph, role = row
        if not ph or not _check_password(password, ph):
            return None, "Incorrect password"
    else:
        found = next(
            (u for u in _FALLBACK.values()
             if u["username"].lower() == val.lower()
             or (u.get("email") and u["email"].lower() == val.lower())),
            None,
        )
        if not found:
            return None, "Account not found"
        if not _check_password(password, found["password_hash"]):
            return None, "Incorrect password"
        uid, uname, em = found["user_id"], found["username"], found.get("email") or ""
        role = found.get("role", "user")

    token = _make_token(uid, uname, em or "", role)
    return {"user_id": uid, "username": uname, "email": em or "", "role": role, "token": token}, None


def get_or_create_google_user(google_id: str, email: str, name: str) -> Optional[dict]:
    """Find or create a user for a Google account. Returns user_dict or None on error."""
    if _PG_OK:
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    # 1. Find by google_id
                    cur.execute(
                        "SELECT user_id, username, email, role FROM auth_users WHERE google_id=%s",
                        (google_id,),
                    )
                    row = cur.fetchone()
                    if row:
                        uid, uname, em, role = row
                    else:
                        # 2. Find by email (link google_id)
                        cur.execute(
                            "SELECT user_id, username, email, role FROM auth_users "
                            "WHERE lower(email)=lower(%s)", (email,)
                        )
                        row = cur.fetchone()
                        if row:
                            uid, uname, em, role = row
                            cur.execute(
                                "UPDATE auth_users SET google_id=%s WHERE user_id=%s", (google_id, uid)
                            )
                        else:
                            # 3. Create new user
                            uid  = str(uuid.uuid4())
                            base = (name or email.split("@")[0]).replace(" ", "_")[:28]
                            uname = base
                            cur.execute(
                                "SELECT 1 FROM auth_users WHERE lower(username)=lower(%s)", (uname,)
                            )
                            if cur.fetchone():
                                uname = f"{base}_{uid[:6]}"
                            em   = email
                            role = "user"
                            cur.execute(
                                "INSERT INTO auth_users (user_id, username, email, google_id, role) "
                                "VALUES (%s,%s,%s,%s,%s)",
                                (uid, uname, em, google_id, role),
                            )
                conn.commit()
        except Exception as e:
            logger.error(f"[AUTH] google_user error: {e}")
            return None
    else:
        found = next((u for u in _FALLBACK.values() if u.get("google_id") == google_id), None)
        if found:
            uid, uname, em = found["user_id"], found["username"], found.get("email", "")
            role = found.get("role", "user")
        else:
            uid   = str(uuid.uuid4())
            uname = (name or email.split("@")[0]).replace(" ", "_")[:28]
            em    = email
            role  = "user"
            _FALLBACK[uid] = {
                "user_id": uid, "username": uname, "email": em,
                "password_hash": None, "google_id": google_id, "role": role,
            }

    token = _make_token(uid, uname, em, role)
    return {"user_id": uid, "username": uname, "email": em, "role": role, "token": token}


def get_user_by_id(user_id: str) -> Optional[dict]:
    if _PG_OK:
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT user_id, username, email, role FROM auth_users WHERE user_id=%s",
                        (user_id,),
                    )
                    row = cur.fetchone()
            if row:
                return {"user_id": row[0], "username": row[1], "email": row[2] or "", "role": row[3] or "user"}
        except Exception:
            pass
    return _FALLBACK.get(user_id)


def create_admin_user(
    username: str,
    password: str,
    email:    str = "",
) -> tuple[Optional[dict], Optional[str]]:
    """Create a user with the 'admin' role. Returns (user_dict, None) or (None, error).

    This is the only legitimate way to create an admin account — it calls
    :func:`_create_user_internal` directly, bypassing the public-registration
    role lock.  Never expose this path to untrusted HTTP input without an
    existing admin check.
    """
    token, err = _create_user_internal(username, password, email, role="admin")
    if err:
        return None, err
    info = verify_token(token)  # type: ignore[arg-type]
    if not info:
        return None, "Token verification failed after admin registration"
    return {
        "user_id":  info["user_id"],
        "username": info["username"],
        "email":    info.get("email", ""),
        "role":     "admin",
        "token":    token,
    }, None
