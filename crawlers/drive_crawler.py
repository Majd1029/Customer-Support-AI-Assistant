"""
drive_crawler.py — Crawl Google Drive and index files into the RAG pipeline.

Each file is downloaded and POST-ed to the FastAPI /upload endpoint, where
the existing parsers handle PDFs, DOCX, PPTX, XLSX, TXT, images, and more.

Usage
-----
    # Crawl My Drive root (all supported file types)
    python crawlers/drive_crawler.py

    # Crawl a specific folder (find the ID in the Drive URL)
    python crawlers/drive_crawler.py --folder-id 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms

    # Crawl recursively through sub-folders
    python crawlers/drive_crawler.py --folder-id <ID> --recursive

    # Filter by MIME type
    python crawlers/drive_crawler.py --types pdf docx pptx

    # Crawl files modified after a date
    python crawlers/drive_crawler.py --modified-after 2024-01-01

    # Dry run — print what would be indexed without actually uploading
    python crawlers/drive_crawler.py --dry-run

    # Custom API endpoint
    python crawlers/drive_crawler.py --api-url http://192.168.1.10:8000

Requirements
------------
    pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client

    Also place credentials.json in the project root.
    See crawlers/auth.py for full setup instructions.
"""

from __future__ import annotations

import argparse
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

from crawlers.auth import get_drive_service  # noqa: E402

# ── Drive permission store (optional — degrades gracefully if PG is down) ─────
try:
    from file_processor.drive_store import upsert_file as _drive_upsert_file
    _DRIVE_STORE_OK = True
except Exception:
    _drive_upsert_file = None  # type: ignore
    _DRIVE_STORE_OK = False

# ── MIME type mapping ──────────────────────────────────────────────────────────
# Maps extension → (Drive MIME type, export MIME type or None)
# When export MIME is set, the file is a Google Workspace doc that must be
# exported rather than downloaded directly.

_EXT_TO_MIME: dict[str, tuple[str, str | None]] = {
    "pdf":   ("application/pdf",                                        None),
    "docx":  ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", None),
    "pptx":  ("application/vnd.openxmlformats-officedocument.presentationml.presentation", None),
    "xlsx":  ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", None),
    "txt":   ("text/plain",                                             None),
    "md":    ("text/markdown",                                          None),
    "csv":   ("text/csv",                                               None),
    "png":   ("image/png",                                              None),
    "jpg":   ("image/jpeg",                                             None),
    "jpeg":  ("image/jpeg",                                             None),
    "webp":  ("image/webp",                                             None),
    # Google Workspace formats → exported as their Office equivalent
    "gdoc":  (
        "application/vnd.google-apps.document",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    "gslides": (
        "application/vnd.google-apps.presentation",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ),
    "gsheets": (
        "application/vnd.google-apps.spreadsheet",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
}

# Maps Google Workspace MIME → export MIME (for the reverse lookup used in file listing)
_GOOGLE_EXPORT: dict[str, tuple[str, str]] = {
    "application/vnd.google-apps.document":     (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx",
    ),
    "application/vnd.google-apps.presentation": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".pptx",
    ),
    "application/vnd.google-apps.spreadsheet":  (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
}

# Default set of MIME types to crawl (all supported by the RAG pipeline)
_DEFAULT_MIMES: set[str] = {
    mime for ext, (mime, _export) in _EXT_TO_MIME.items()
    if ext not in ("gdoc", "gslides", "gsheets")
} | set(_GOOGLE_EXPORT.keys())

# Extension → pipeline's supported extensions (from config.py)
_SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".rst",
    ".csv", ".tsv", ".png", ".jpg", ".jpeg", ".webp", ".eml",
}

# ── Config ─────────────────────────────────────────────────────────────────────

DEFAULT_API_URL  = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_API_KEY  = os.getenv("API_KEY", "")
DEFAULT_MAX      = 200
RETRY_SLEEP      = 2.0
MAX_RETRIES      = 3
_POLL_INTERVAL   = 3      # seconds between /tasks/{job_id} polls
_POLL_MAX_WAIT   = 600    # max total seconds to wait per file (10 minutes)
PAGE_SIZE        = 100    # files per Drive API page


# ── Drive helpers ──────────────────────────────────────────────────────────────

def _build_mime_filter(ext_filter: list[str] | None) -> set[str]:
    """Convert a list of extensions to a set of MIME types to include."""
    if not ext_filter:
        return _DEFAULT_MIMES
    mimes: set[str] = set()
    for ext in ext_filter:
        ext = ext.lstrip(".")
        if ext in _EXT_TO_MIME:
            mimes.add(_EXT_TO_MIME[ext][0])
        else:
            logger.warning(f"[Drive] Unknown extension filter: .{ext} — ignored")
    return mimes


def _list_files(
    service,
    folder_id:      str | None,
    allowed_mimes:  set[str],
    modified_after: str | None,
    recursive:      bool,
    max_results:    int,
) -> list[dict]:
    """
    List Drive files matching the given criteria.

    Returns list of file metadata dicts with: id, name, mimeType, size.
    """
    files: list[dict] = []

    def _query_folder(fid: str | None) -> None:
        nonlocal files
        if len(files) >= max_results:
            return

        # Build query
        q_parts = ["trashed = false"]
        if fid:
            q_parts.append(f"'{fid}' in parents")
        if allowed_mimes:
            mime_clauses = " or ".join(
                f"mimeType = '{m}'" for m in sorted(allowed_mimes)
            )
            q_parts.append(f"({mime_clauses})")
        if modified_after:
            q_parts.append(f"modifiedTime > '{modified_after}T00:00:00'")

        q = " and ".join(q_parts)
        page_token = None

        while len(files) < max_results:
            kwargs: dict = {
                "q":        q,
                "fields":   "nextPageToken, files(id, name, mimeType, size, parents)",
                "pageSize": min(PAGE_SIZE, max_results - len(files)),
            }
            if page_token:
                kwargs["pageToken"] = page_token

            resp = service.files().list(**kwargs).execute()
            batch = resp.get("files", [])

            # Separate folders from files
            sub_folders = []
            for f in batch:
                if f["mimeType"] == "application/vnd.google-apps.folder":
                    sub_folders.append(f)
                else:
                    files.append(f)
                    if len(files) >= max_results:
                        return

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

            # Recurse into sub-folders if requested
            if recursive:
                for sf in sub_folders:
                    if len(files) >= max_results:
                        return
                    _query_folder(sf["id"])

    _query_folder(folder_id)
    return files[:max_results]


def _download_file(service, file_meta: dict) -> tuple[str, bytes]:
    """
    Download a Drive file.  Google Workspace files are exported first.

    Returns (final_filename, content_bytes).
    """
    fid      = file_meta["id"]
    name     = file_meta["name"]
    mime     = file_meta["mimeType"]

    # Google Workspace doc — export as Office format
    if mime in _GOOGLE_EXPORT:
        export_mime, ext = _GOOGLE_EXPORT[mime]
        data = service.files().export_media(fileId=fid, mimeType=export_mime).execute()
        # Ensure the filename has the correct extension
        stem = Path(name).stem
        filename = stem + ext
        return filename, data

    # Regular binary file — download directly
    from googleapiclient.http import MediaIoBaseDownload
    buf = io.BytesIO()
    request = service.files().get_media(fileId=fid)
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _status, done = downloader.next_chunk()
    filename = name
    # Ensure the filename has an extension the pipeline recognises
    suffix = Path(filename).suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        logger.warning(
            f"[Drive] {filename} has unsupported extension '{suffix}' — skipping"
        )
        return filename, b""

    return filename, buf.getvalue()


def _resolve_permissions(
    service,
    file_id:     str,
    owner_email: str,
) -> tuple[bool, list[str]]:
    """
    Fetch Drive permissions for a single file.

    Returns (is_public, allowed_emails).
    The owner is always included in allowed_emails.
    Fails silently — returns (False, [owner_email]) on any API error so a
    permissions failure never aborts the crawl.
    """
    is_public     = False
    allowed_users = [owner_email]

    try:
        perms = (
            service.permissions()
            .list(fileId=file_id, fields="permissions(type, emailAddress, role)")
            .execute()
            .get("permissions", [])
        )
        for p in perms:
            if p.get("role") not in ("owner", "writer", "commenter", "reader"):
                continue
            if p.get("type") == "anyone":
                is_public = True
            elif p.get("type") in ("user", "group"):
                email = p.get("emailAddress", "")
                if email and email not in allowed_users:
                    allowed_users.append(email)
    except Exception as e:
        logger.warning(f"[Drive] Could not fetch permissions for {file_id}: {e}")

    return is_public, allowed_users


def iter_files(
    service,
    folder_id:      str | None,
    allowed_mimes:  set[str],
    modified_after: str | None,
    recursive:      bool,
    max_results:    int,
) -> Iterator[tuple[str, bytes, dict]]:
    """
    Yield (filename, content_bytes, file_meta) for each Drive file matching
    the criteria.  ``file_meta`` is the raw dict from the Drive API list call
    (keys: id, name, mimeType, size, parents) — used by the caller to resolve
    permissions and tag chunks.  Skips files that cannot be downloaded or have
    unsupported formats.
    """
    file_list = _list_files(
        service, folder_id, allowed_mimes, modified_after, recursive, max_results
    )
    total = len(file_list)
    logger.info(f"[Drive] Found {total} files to process")

    for i, meta in enumerate(file_list, 1):
        name = meta["name"]
        size = meta.get("size", "?")
        logger.debug(f"[Drive] {i}/{total}  {name}  ({size} bytes)")
        try:
            filename, content = _download_file(service, meta)
            if not content:
                continue
            yield filename, content, meta
        except Exception as e:
            logger.warning(f"[Drive] Skipping {name}: {e}")


# ── Upload helpers ─────────────────────────────────────────────────────────────

def _mime_for_filename(filename: str) -> str:
    """Return the appropriate Content-Type for the /upload endpoint."""
    ext = Path(filename).suffix.lower()
    mime_map = {
        ".pdf":  "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".txt":  "text/plain",
        ".md":   "text/markdown",
        ".csv":  "text/csv",
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    return mime_map.get(ext, "application/octet-stream")


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
    api_url:       str,
    api_key:       str,
    filename:      str,
    content:       bytes,
    user_id:       str       = "",
    user_email:    str       = "",
    drive_file_id: str       = "",
    is_public:     bool      = False,
    allowed_users: list[str] | None = None,
) -> dict:
    """
    POST a single file to /upload (async mode) and poll until indexing is done.

    Uses async_mode=true so the server returns a job_id immediately instead of
    blocking until OCR/extraction completes — prevents 504 gateway timeouts on
    large scanned PDFs.
    """
    headers: dict = {}
    if api_key:
        headers["X-API-Key"] = api_key

    # Extra form fields so the server can stamp owner + permission metadata
    # on every chunk stored in Qdrant.
    extra_data: dict = {"async_mode": "true"}   # avoids 504 on large/scanned files
    if user_id:
        extra_data["owner_id"] = user_id
    if user_email:
        extra_data["owner_email"] = user_email
    if drive_file_id:
        extra_data["drive_file_id"] = drive_file_id
    extra_data["is_public"] = str(is_public).lower()
    if allowed_users:
        # Comma-separated list; server splits on ","
        extra_data["allowed_users"] = ",".join(allowed_users)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{api_url}/upload",
                files={"file": (filename, io.BytesIO(content), _mime_for_filename(filename))},
                data=extra_data,
                headers=headers,
                timeout=30,   # async POST returns quickly — just needs to save file to disk
            )
            if resp.status_code == 429:
                logger.warning(
                    f"[Upload] 429 rate-limited — waiting {RETRY_SLEEP}s "
                    f"(attempt {attempt}/{MAX_RETRIES})"
                )
                time.sleep(RETRY_SLEEP)
                continue
            if resp.status_code == 503:
                # Upload worker pool is full — back off and retry
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
    folder_id:      str | None       = None,
    ext_filter:     list[str] | None = None,
    modified_after: str | None       = None,
    recursive:      bool             = False,
    max_results:    int              = DEFAULT_MAX,
    api_url:        str              = DEFAULT_API_URL,
    api_key:        str              = DEFAULT_API_KEY,
    dry_run:        bool             = False,
    user_id:        str              = "",
    user_email:     str              = "",
    service                          = None,   # pre-built Drive service (web OAuth path)
) -> dict:
    """
    Main entry point for programmatic use.

    Pass ``user_id`` and ``user_email`` so that every indexed chunk is tagged
    with owner metadata in Qdrant — required for per-user retrieval filtering.

    Pass a pre-built ``service`` object when using the web OAuth flow
    (``crawlers.auth.get_drive_service_from_tokens``); otherwise the desktop
    flow credentials.json / token.json are used automatically.

    Returns a summary dict:
        {"total": N, "indexed": N, "skipped": N, "errors": N}
    """
    if service is None:
        service = get_drive_service()
    allowed_mimes = _build_mime_filter(ext_filter)

    stats = {"total": 0, "indexed": 0, "skipped": 0, "errors": 0}

    for filename, content, file_meta in iter_files(
        service,
        folder_id      = folder_id,
        allowed_mimes  = allowed_mimes,
        modified_after = modified_after,
        recursive      = recursive,
        max_results    = max_results,
    ):
        stats["total"] += 1
        drive_file_id = file_meta.get("id", "")

        if dry_run:
            logger.info(
                f"[DRY RUN] Would upload: {filename}  ({len(content):,} bytes)"
                + (f"  [Drive ID: {drive_file_id}]" if drive_file_id else "")
            )
            stats["skipped"] += 1
            continue

        # ── Resolve file permissions ──────────────────────────────────────────
        is_public, allowed_users = _resolve_permissions(
            service, drive_file_id, user_email
        ) if drive_file_id and user_email else (False, [user_email] if user_email else [])

        # ── Persist permissions to drive_files table ──────────────────────────
        if _DRIVE_STORE_OK and drive_file_id and _drive_upsert_file is not None:
            try:
                # Store ``filename`` (the post-export name with extension, e.g.
                # "Q3 Report.docx") so it matches the ``source`` field in Qdrant
                # and the UI can join on the same string.
                _drive_upsert_file(
                    file_id       = drive_file_id,
                    name          = filename,
                    mime_type     = file_meta.get("mimeType", ""),
                    owner_email   = user_email,
                    allowed_users = allowed_users,
                    is_public     = is_public,
                )
            except Exception as e:
                logger.warning(f"[Drive] drive_store upsert failed for {filename}: {e}")

        # ── Upload to RAG pipeline ────────────────────────────────────────────
        try:
            result = _upload(
                api_url, api_key, filename, content,
                user_id       = user_id,
                user_email    = user_email,
                drive_file_id = drive_file_id,
                is_public     = is_public,
                allowed_users = allowed_users,
            )
            chunks = result.get("stats", {}).get("total_chunks", "?")
            logger.info(
                f"[Drive] ✓  {filename}  →  {chunks} chunks indexed"
                f"  (public={is_public}, shared_with={len(allowed_users)})"
            )
            stats["indexed"] += 1
        except Exception as e:
            logger.error(f"[Drive] ✗  {filename}: {e}")
            stats["errors"] += 1

    logger.info(
        f"[Drive] Done — "
        f"total={stats['total']}  "
        f"indexed={stats['indexed']}  "
        f"skipped={stats['skipped']}  "
        f"errors={stats['errors']}"
    )
    return stats


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl Google Drive and index files into the RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Filter options
    parser.add_argument(
        "--folder-id", default=None,
        metavar="FOLDER_ID",
        help="Google Drive folder ID to crawl (default: My Drive root). "
             "Find the ID in the folder URL: drive.google.com/drive/folders/<FOLDER_ID>",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Recurse into sub-folders (default: top-level only).",
    )
    parser.add_argument(
        "--types", nargs="+", default=None,
        metavar="EXT",
        help="File extensions to include, e.g. --types pdf docx pptx xlsx. "
             "Defaults to all supported types.",
    )
    parser.add_argument(
        "--modified-after", default=None,
        metavar="YYYY-MM-DD",
        help="Only crawl files modified after this date.",
    )
    parser.add_argument(
        "--max", type=int, default=DEFAULT_MAX,
        dest="max_results",
        help=f"Maximum number of files to crawl (default: {DEFAULT_MAX}).",
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

    crawl(
        folder_id      = args.folder_id,
        ext_filter     = args.types,
        modified_after = args.modified_after,
        recursive      = args.recursive,
        max_results    = args.max_results,
        api_url        = args.api_url,
        api_key        = args.api_key,
        dry_run        = args.dry_run,
    )


if __name__ == "__main__":
    _cli()
