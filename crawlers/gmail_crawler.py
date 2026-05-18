"""
gmail_crawler.py — Crawl Gmail and index emails into the RAG pipeline.

Each email thread is exported as a standard .eml file and POST-ed to the
FastAPI /upload endpoint, where the existing EML parser extracts body text,
attachments, and tables — identical to uploading a local .eml file.

Usage
-----
    # Crawl last 50 emails from INBOX
    python crawlers/gmail_crawler.py --label INBOX --max 50

    # Crawl a specific label, filter by date
    python crawlers/gmail_crawler.py --label "Work" --after 2024-01-01 --before 2025-01-01

    # Crawl by search query (same syntax as Gmail search bar)
    python crawlers/gmail_crawler.py --query "from:boss@example.com has:attachment"

    # Custom API endpoint (non-localhost)
    python crawlers/gmail_crawler.py --api-url http://192.168.1.10:8000 --label INBOX

    # Dry run — print what would be indexed without actually uploading
    python crawlers/gmail_crawler.py --label INBOX --dry-run

Requirements
------------
    pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client

    Also place credentials.json in the project root.
    See crawlers/auth.py for full setup instructions.
"""

from __future__ import annotations

import argparse
import base64
import email
import io
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import requests
from loguru import logger

# ── Project root on sys.path ───────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from crawlers.auth import get_gmail_service  # noqa: E402

# ── Default configuration ──────────────────────────────────────────────────────

DEFAULT_API_URL   = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_API_KEY   = os.getenv("API_KEY", "")
DEFAULT_MAX       = 100          # threads to crawl per run
RETRY_SLEEP       = 2.0          # seconds between retries on 429
MAX_RETRIES       = 3
_POLL_INTERVAL    = 3      # seconds between /tasks/{job_id} polls
_POLL_MAX_WAIT    = 600    # max total seconds to wait per file (10 minutes)


# ── Gmail helpers ──────────────────────────────────────────────────────────────

def _build_query(
    label:  str | None,
    query:  str | None,
    after:  str | None,
    before: str | None,
) -> str:
    """Build a Gmail search query string from CLI arguments."""
    parts: list[str] = []
    if label:
        parts.append(f"label:{label}")
    if after:
        parts.append(f"after:{after}")
    if before:
        parts.append(f"before:{before}")
    if query:
        parts.append(query)
    return " ".join(parts) if parts else "in:anywhere"


def _list_thread_ids(service, q: str, max_results: int) -> list[str]:
    """Return up to max_results thread IDs matching query q."""
    ids: list[str] = []
    page_token = None

    while len(ids) < max_results:
        batch = min(500, max_results - len(ids))
        kwargs: dict = {"userId": "me", "q": q, "maxResults": batch}
        if page_token:
            kwargs["pageToken"] = page_token

        resp = service.users().threads().list(**kwargs).execute()
        threads = resp.get("threads", [])
        ids.extend(t["id"] for t in threads)

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return ids[:max_results]


def _thread_to_eml(service, thread_id: str) -> tuple[str, bytes]:
    """
    Fetch a Gmail thread and convert it to a single .eml bytes object.

    Returns (subject_slug, eml_bytes).
    The EML is the raw RFC 2822 representation of the first message in the
    thread (Gmail stores each message in raw base64url form).
    """
    thread = service.users().threads().get(
        userId="me", id=thread_id, format="full"
    ).execute()

    messages = thread.get("messages", [])
    if not messages:
        raise ValueError(f"Thread {thread_id} has no messages")

    # Reconstruct EML from raw message bytes (preferred — most faithful)
    msg = messages[0]
    raw_resp = service.users().messages().get(
        userId="me", id=msg["id"], format="raw"
    ).execute()
    raw_bytes = base64.urlsafe_b64decode(raw_resp["raw"] + "==")

    # Extract subject for a friendly filename
    parsed = email.message_from_bytes(raw_bytes)
    subject = parsed.get("Subject", f"email_{thread_id}")
    # Sanitise subject for use as a filename
    safe_subject = "".join(
        c if c.isalnum() or c in "._- " else "_" for c in subject
    )[:60].strip()

    return safe_subject, raw_bytes


def iter_threads(
    service,
    q: str,
    max_results: int,
) -> Iterator[tuple[str, bytes]]:
    """
    Yield (filename_slug, eml_bytes) for each thread matching the query.
    Logs progress as it goes.
    """
    thread_ids = _list_thread_ids(service, q, max_results)
    total = len(thread_ids)
    logger.info(f"[Gmail] Found {total} threads matching: {q!r}")

    for i, tid in enumerate(thread_ids, 1):
        try:
            slug, eml_bytes = _thread_to_eml(service, tid)
            logger.debug(f"[Gmail] {i}/{total}  {slug}")
            yield slug, eml_bytes
        except Exception as e:
            logger.warning(f"[Gmail] Skipping thread {tid}: {e}")


# ── Upload helpers ─────────────────────────────────────────────────────────────

def _poll_task(api_url: str, job_id: str, headers: dict, filename: str) -> dict:
    """
    Poll GET /tasks/{job_id} until the async upload job finishes.

    Returns the job result dict on success, raises on failure or timeout.
    """
    deadline = time.monotonic() + _POLL_MAX_WAIT
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{api_url}/tasks/{job_id}", headers={k: v for k, v in headers.items() if k != "Content-Type"}, timeout=10)
            if r.ok:
                t = r.json()
                status = t.get("status", "pending")
                if status == "done":
                    logger.info(f"[Upload] {filename} — async processing done ✓")
                    return t.get("result", t)
                if status == "failed":
                    err = t.get("error", "unknown error")
                    raise RuntimeError(f"[Upload] Async processing failed for {filename!r}: {err}")
                # still pending/processing — keep polling
        except requests.exceptions.ConnectionError:
            logger.warning(f"[Upload] Poll connection lost for job {job_id}; retrying …")
        time.sleep(_POLL_INTERVAL)
    raise TimeoutError(
        f"[Upload] {filename!r} timed out after {_POLL_MAX_WAIT}s. "
        "The file may still be processing — check /tasks/{job_id} manually."
    )


def _upload(
    api_url:    str,
    api_key:    str,
    filename:   str,
    eml_bytes:  bytes,
    user_id:    str = "",
    user_email: str = "",
) -> dict:
    """
    POST a single .eml to /upload (async mode) and poll until indexing is done.

    Uses async_mode=true so the server returns a job_id immediately instead of
    blocking — prevents 504 gateway timeouts on large emails with attachments.
    """
    headers: dict = {}
    if api_key:
        headers["X-API-Key"] = api_key

    # Extra form fields so the server can stamp owner metadata on every chunk
    extra_data: dict = {"async_mode": "true"}   # avoids 504 on large emails
    if user_id:
        extra_data["owner_id"] = user_id
    if user_email:
        extra_data["owner_email"] = user_email

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{api_url}/upload",
                files={"file": (filename, io.BytesIO(eml_bytes), "message/rfc822")},
                data=extra_data,
                headers=headers,
                timeout=30,   # async POST returns quickly
            )
            if resp.status_code == 429:
                logger.warning(
                    f"[Upload] 429 rate-limited — waiting {RETRY_SLEEP}s "
                    f"(attempt {attempt}/{MAX_RETRIES})"
                )
                time.sleep(RETRY_SLEEP)
                continue
            if resp.status_code == 503:
                wait = RETRY_SLEEP * 5
                logger.warning(
                    f"[Upload] 503 worker pool full — waiting {wait}s "
                    f"(attempt {attempt}/{MAX_RETRIES})"
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()

            # Async path: poll /tasks/{job_id} until done
            job_id = data.get("job_id")
            if job_id:
                return _poll_task(api_url, job_id, headers, filename)
            # Sync fallback (should not happen when async_mode=true is accepted)
            return data

        except requests.exceptions.ConnectionError:
            logger.error(
                f"[Upload] Cannot connect to {api_url}. "
                "Is the FastAPI server running?  (uvicorn api:app --reload --port 8000)"
            )
            sys.exit(1)

    raise RuntimeError(f"Upload failed after {MAX_RETRIES} retries for {filename}")


# ── Main crawl loop ────────────────────────────────────────────────────────────

def crawl(
    label:      str | None  = "INBOX",
    query:      str | None  = None,
    after:      str | None  = None,
    before:     str | None  = None,
    max_results: int        = DEFAULT_MAX,
    api_url:    str         = DEFAULT_API_URL,
    api_key:    str         = DEFAULT_API_KEY,
    dry_run:    bool        = False,
    user_id:    str         = "",
    user_email: str         = "",
    service                 = None,     # pre-built Gmail service (web OAuth path)
) -> dict:
    """
    Main entry point for programmatic use.

    Pass ``user_id`` and ``user_email`` so that every indexed chunk is tagged
    with owner metadata in Qdrant — required for per-user retrieval filtering.

    Pass a pre-built ``service`` object when using the web OAuth flow
    (``crawlers.auth.get_gmail_service_from_tokens``); otherwise the desktop
    flow credentials.json / token.json are used automatically.

    Returns a summary dict:
        {"total": N, "indexed": N, "skipped": N, "errors": N}
    """
    if service is None:
        service = get_gmail_service()
    q = _build_query(label, query, after, before)

    stats = {"total": 0, "indexed": 0, "skipped": 0, "errors": 0}

    for slug, eml_bytes in iter_threads(service, q, max_results):
        stats["total"] += 1
        filename = f"{slug}.eml"

        if dry_run:
            logger.info(f"[DRY RUN] Would upload: {filename}  ({len(eml_bytes):,} bytes)")
            stats["skipped"] += 1
            continue

        try:
            result = _upload(api_url, api_key, filename, eml_bytes, user_id, user_email)
            chunks = result.get("stats", {}).get("total_chunks", "?")
            logger.info(f"[Gmail] ✓  {filename}  →  {chunks} chunks indexed")
            stats["indexed"] += 1
        except Exception as e:
            logger.error(f"[Gmail] ✗  {filename}: {e}")
            stats["errors"] += 1

    logger.info(
        f"[Gmail] Done — "
        f"total={stats['total']}  "
        f"indexed={stats['indexed']}  "
        f"skipped={stats['skipped']}  "
        f"errors={stats['errors']}"
    )
    return stats


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl Gmail and index emails into the RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Filter options
    parser.add_argument(
        "--label", default="INBOX",
        help="Gmail label to crawl (default: INBOX). Use 'all' for no label filter.",
    )
    parser.add_argument(
        "--query", default=None,
        help="Gmail search query (e.g. 'from:boss@example.com has:attachment'). "
             "Combined with --label / --after / --before with AND logic.",
    )
    parser.add_argument(
        "--after", default=None,
        metavar="YYYY-MM-DD",
        help="Only fetch emails after this date.",
    )
    parser.add_argument(
        "--before", default=None,
        metavar="YYYY-MM-DD",
        help="Only fetch emails before this date.",
    )
    parser.add_argument(
        "--max", type=int, default=DEFAULT_MAX,
        dest="max_results",
        help=f"Maximum number of threads to crawl (default: {DEFAULT_MAX}).",
    )

    # Server options
    parser.add_argument(
        "--api-url", default=DEFAULT_API_URL,
        help=f"FastAPI server URL (default: {DEFAULT_API_URL}).",
    )
    parser.add_argument(
        "--api-key", default=DEFAULT_API_KEY,
        help="X-API-Key header value (leave blank for open dev mode).",
    )

    # Misc
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be uploaded without actually calling /upload.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG-level logging.",
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if args.debug else "INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    label = None if args.label.lower() == "all" else args.label

    crawl(
        label       = label,
        query       = args.query,
        after       = args.after,
        before      = args.before,
        max_results = args.max_results,
        api_url     = args.api_url,
        api_key     = args.api_key,
        dry_run     = args.dry_run,
    )


if __name__ == "__main__":
    _cli()
