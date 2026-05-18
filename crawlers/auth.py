"""
auth.py — Shared OAuth 2.0 helper for Gmail and Google Drive crawlers.

Two authentication paths are supported:

  1. Web OAuth (recommended)
     ─────────────────────────
     Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env using a
     "Web application" client from Google Cloud Console.  Users sign in
     via the "Sign in with Google" button in the React UI — the backend
     handles the OAuth redirect flow and stores tokens in memory.
     No credentials.json or manual browser flow needed at runtime.

  2. Desktop app flow (legacy / CLI)
     ─────────────────────────────────
     Download credentials.json (OAuth client type: Desktop app) from
     Google Cloud Console and place it at the project root.  The first
     run opens a browser consent screen; subsequent runs refresh the
     token silently from crawlers/token.json.

Setup (Web OAuth)
-----------------
1.  Go to https://console.cloud.google.com/
2.  Create (or select) a project and enable Gmail API + Google Drive API.
3.  APIs & Services → Credentials → Create Credentials
    → OAuth client ID → Web application
    → Add Authorised redirect URI: http://localhost:8000/auth/google/callback
    → Download JSON, note the Client ID and Client Secret.
4.  Add to .env:
        GOOGLE_CLIENT_ID=<your-client-id>
        GOOGLE_CLIENT_SECRET=<your-client-secret>
5.  APIs & Services → OAuth consent screen → add your email as a Test User.
6.  Click "Sign in with Google" in the React UI — credentials are stored
    automatically and used for all subsequent crawl jobs.

Setup (Desktop app / CLI)
--------------------------
1.  Go to https://console.cloud.google.com/
2.  Enable Gmail API + Google Drive API.
3.  APIs & Services → Credentials → Create Credentials
    → OAuth client ID → Desktop app → Download JSON → save as credentials.json
4.  Add your email as a Test User on the OAuth consent screen.
5.  Run either crawler — a browser tab opens for the one-time consent.
    After approval, token.json is written and re-used automatically.

Environment variables (.env)
----------------------------
GOOGLE_CLIENT_ID          Web OAuth client ID (preferred)
GOOGLE_CLIENT_SECRET      Web OAuth client secret (preferred)
GOOGLE_CREDENTIALS_PATH   Path to credentials.json (desktop flow fallback)
                          (default: <project_root>/credentials.json)
GOOGLE_TOKEN_PATH         Path to token.json (desktop flow fallback)
                          (default: <project_root>/crawlers/token.json)
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

# ── Path resolution ────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env so callers don't need to do it themselves
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

CREDENTIALS_PATH: Path = Path(
    os.getenv("GOOGLE_CREDENTIALS_PATH", str(_PROJECT_ROOT / "credentials.json"))
)
TOKEN_PATH: Path = Path(
    os.getenv("GOOGLE_TOKEN_PATH", str(_PROJECT_ROOT / "crawlers" / "token.json"))
)

# ── OAuth scopes ───────────────────────────────────────────────────────────────

GMAIL_SCOPES    = ["https://www.googleapis.com/auth/gmail.readonly"]
DRIVE_SCOPES    = ["https://www.googleapis.com/auth/drive.readonly"]
COMBINED_SCOPES = GMAIL_SCOPES + DRIVE_SCOPES

# Full scope set for the web OAuth flow (includes identity for sign-in)
WEB_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]


# ── Public API ─────────────────────────────────────────────────────────────────

def get_credentials(scopes: list[str]):
    """
    Return valid Google OAuth2 credentials for the given scopes.

    - If token.json exists and covers the requested scopes, loads and
      refreshes it (no browser interaction).
    - Otherwise runs the browser-based consent flow and saves the new
      token to token.json.

    Args:
        scopes: list of OAuth scope strings, e.g. GMAIL_SCOPES.

    Returns:
        google.oauth2.credentials.Credentials
    """
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise ImportError(
            "Google auth packages not installed.\n"
            "Run: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client"
        ) from exc

    if not CREDENTIALS_PATH.exists():
        raise FileNotFoundError(
            f"credentials.json not found at {CREDENTIALS_PATH}.\n"
            "Download it from Google Cloud Console → APIs & Services → Credentials\n"
            "and save it as credentials.json in the project root.\n"
            "See crawlers/auth.py docstring for full setup instructions."
        )

    creds: Credentials | None = None

    # ── Load cached token ──────────────────────────────────────────────────────
    if TOKEN_PATH.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), scopes)
        except Exception as e:
            logger.warning(f"[AUTH] Could not load token.json: {e} — re-authorising")
            creds = None

    # ── Refresh or re-authorise ────────────────────────────────────────────────
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("[AUTH] Refreshing expired Google token …")
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.warning(f"[AUTH] Refresh failed: {e} — re-authorising")
                creds = None

        if not creds or not creds.valid:
            logger.info("[AUTH] Opening browser for Google authorisation …")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_PATH), scopes
            )
            creds = flow.run_local_server(port=0)
            logger.info("[AUTH] Authorisation complete.")

        # ── Persist token ──────────────────────────────────────────────────────
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_PATH.write_text(creds.to_json())
        logger.info(f"[AUTH] Token saved to {TOKEN_PATH}")

    return creds


def get_gmail_service():
    """Return an authorised Gmail API service object (desktop flow)."""
    from googleapiclient.discovery import build
    creds = get_credentials(GMAIL_SCOPES)
    return build("gmail", "v1", credentials=creds)


def get_drive_service():
    """Return an authorised Google Drive API service object (desktop flow)."""
    from googleapiclient.discovery import build
    creds = get_credentials(DRIVE_SCOPES)
    return build("drive", "v3", credentials=creds)


# ── Web OAuth helpers ──────────────────────────────────────────────────────────

def build_credentials_from_tokens(
    token_info: dict,
    client_id: str,
    client_secret: str,
):
    """
    Build Google Credentials from a token dict stored after the web OAuth flow.

    Automatically refreshes the access token if it has expired.
    The ``token_info`` dict is updated in-place with the new access token so
    the caller's in-memory store stays current.

    Args:
        token_info    : dict with at least ``access_token`` and ``refresh_token``
        client_id     : Google OAuth web-app client ID
        client_secret : Google OAuth web-app client secret

    Returns:
        google.oauth2.credentials.Credentials (valid, refreshed if needed)
    """
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
    except ImportError as exc:
        raise ImportError(
            "Google auth packages not installed.\n"
            "Run: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client"
        ) from exc

    creds = Credentials(
        token         = token_info.get("access_token"),
        refresh_token = token_info.get("refresh_token"),
        token_uri     = "https://oauth2.googleapis.com/token",
        client_id     = client_id,
        client_secret = client_secret,
        scopes        = WEB_SCOPES,
    )

    if not creds.valid and creds.refresh_token:
        try:
            creds.refresh(Request())
            token_info["access_token"] = creds.token   # keep store current
            logger.debug(f"[AUTH] Access token refreshed for {token_info.get('email', '?')}")
        except Exception as exc:
            logger.warning(f"[AUTH] Token refresh failed: {exc}")

    return creds


def get_gmail_service_from_tokens(
    token_info: dict,
    client_id: str,
    client_secret: str,
):
    """Return an authorised Gmail API service using stored web OAuth tokens."""
    from googleapiclient.discovery import build
    creds = build_credentials_from_tokens(token_info, client_id, client_secret)
    return build("gmail", "v1", credentials=creds)


def get_drive_service_from_tokens(
    token_info: dict,
    client_id: str,
    client_secret: str,
):
    """Return an authorised Drive API service using stored web OAuth tokens."""
    from googleapiclient.discovery import build
    creds = build_credentials_from_tokens(token_info, client_id, client_secret)
    return build("drive", "v3", credentials=creds)
