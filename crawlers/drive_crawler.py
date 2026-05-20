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

# ── Drive permission + crawl-state store (optional — degrades gracefully) ─────
try:
    from file_processor.drive_store import upsert_file as _drive_upsert_file
    from file_processor.drive_store import (
        get_crawl_state    as _get_crawl_state,
        update_crawl_state as _update_crawl_state,
    )
    _DRIVE_STORE_OK = True
except Exception:
    _drive_upsert_file   = None   # type: ignore
    _get_crawl_state     = None   # type: ignore
    _update_crawl_state  = None   # type: ignore
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


_FOLDER_MIME = "application/vnd.google-apps.folder"


def _list_files(
    service,
    folder_id:      str | None,
    allowed_mimes:  set[str],
    modified_after: str | None,
    recursive:      bool,
    max_results:    int,
) -> list[dict]:
    """
    List Drive files matching the given criteria using BFS traversal.

    Root cause of previous "Found 0 files" bug: the MIME type filter excluded
    application/vnd.google-apps.folder, so sub-folders were never returned
    by the file-listing query and the recursion queue was always empty.

    Fix: use two separate queries per folder —
      1. Files query  — filtered by allowed_mimes (never includes folders)
      2. Folders query — filtered by folder MIME, only when recursive=True

    BFS ensures full-depth traversal without Python stack overflow on deep trees.

    Returns list of file metadata dicts: id, name, mimeType, size, parents.
    """
    files:           list[dict]      = []
    seen_file_ids:   set[str]        = set()   # prevent duplicates across shared files
    visited_folders: set[str | None] = set()   # prevent re-entering folders

    # BFS queue — None represents "My Drive root" (no parent filter)
    queue: list[str | None] = [folder_id]
    visited_folders.add(folder_id)

    while queue and len(files) < max_results:
        current_fid = queue.pop(0)

        folder_label = current_fid or "root"
        logger.debug(f"[Drive] Traversing folder: {folder_label}")

        # ── 1. Query files in this folder ─────────────────────────────────────
        file_q_parts = ["trashed = false"]
        if current_fid is not None:
            file_q_parts.append(f"'{current_fid}' in parents")
        if allowed_mimes:
            mime_clauses = " or ".join(
                f"mimeType = '{m}'" for m in sorted(allowed_mimes)
            )
            file_q_parts.append(f"({mime_clauses})")
        if modified_after:
            file_q_parts.append(f"modifiedTime > '{modified_after}T00:00:00'")

        file_q = " and ".join(file_q_parts)
        page_token: str | None = None

        while len(files) < max_results:
            kwargs: dict = {
                "q":        file_q,
                "fields":   "nextPageToken, files(id, name, mimeType, size, parents, modifiedTime)",
                "pageSize": min(PAGE_SIZE, max_results - len(files)),
            }
            if page_token:
                kwargs["pageToken"] = page_token

            try:
                resp = service.files().list(**kwargs).execute()
            except Exception as exc:
                logger.warning(
                    f"[Drive] Error listing files in folder {folder_label}: {exc}"
                )
                break

            for f in resp.get("files", []):
                fid = f["id"]
                if fid not in seen_file_ids:
                    seen_file_ids.add(fid)
                    files.append(f)
                    logger.debug(
                        f"[Drive]   Found file: {f['name']} "
                        f"({f.get('size', '?')} bytes, {f['mimeType']})"
                    )
                    if len(files) >= max_results:
                        break

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        # ── 2. Query sub-folders and enqueue for BFS ──────────────────────────
        if not recursive:
            continue

        folder_q_parts = [
            "trashed = false",
            f"mimeType = '{_FOLDER_MIME}'",
        ]
        if current_fid is not None:
            folder_q_parts.append(f"'{current_fid}' in parents")

        folder_q = " and ".join(folder_q_parts)
        folder_page_token: str | None = None

        while True:
            folder_kwargs: dict = {
                "q":        folder_q,
                "fields":   "nextPageToken, files(id, name, mimeType)",
                "pageSize": PAGE_SIZE,
            }
            if folder_page_token:
                folder_kwargs["pageToken"] = folder_page_token

            try:
                folder_resp = service.files().list(**folder_kwargs).execute()
            except Exception as exc:
                logger.warning(
                    f"[Drive] Error listing sub-folders of {folder_label}: {exc}"
                )
                break

            for sf in folder_resp.get("files", []):
                sf_id = sf["id"]
                if sf_id not in visited_folders:
                    visited_folders.add(sf_id)
                    queue.append(sf_id)
                    logger.debug(
                        f"[Drive]   Discovered sub-folder: {sf['name']} ({sf_id})"
                    )

            folder_page_token = folder_resp.get("nextPageToken")
            if not folder_page_token:
                break

    logger.info(
        f"[Drive] BFS complete — visited {len(visited_folders)} folder(s), "
        f"found {len(files)} file(s)"
    )
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


def _expected_indexed_name(meta: dict) -> str:
    """
    Return the filename that will be written to Qdrant's ``source`` field for
    this file.  Google Workspace files get an Office extension on export
    (e.g. "Quarterly Report.gdoc" → "Quarterly Report.docx"); all other files
    keep their current Drive name.
    """
    mime = meta.get("mimeType", "")
    name = meta.get("name", "")
    if mime in _GOOGLE_EXPORT:
        _, ext = _GOOGLE_EXPORT[mime]
        return Path(name).stem + ext
    return name


def iter_files(
    service,
    folder_id:      str | None,
    allowed_mimes:  set[str],
    modified_after: str | None,
    recursive:      bool,
    max_results:    int,
    incremental:    bool = True,
    on_skip:        "callable | None" = None,
) -> Iterator[tuple[str, bytes, dict]]:
    """
    Yield (filename, content_bytes, file_meta) for each Drive file that should
    be indexed.

    When ``incremental=True`` (default), files whose Drive ``modifiedTime``
    matches the value stored from the previous crawl are silently skipped —
    they are already up-to-date in Qdrant.

    ``file_meta`` is the Drive API metadata dict enriched with three extra keys
    that the caller uses after a successful upload:

      _action          — "new" | "modified" | "renamed" | "full"
      _old_indexed_name — the Qdrant source name from the previous crawl, set
                         only for "renamed" files so the caller can delete the
                         stale entries before re-indexing.
      _modified_time   — Drive modifiedTime string to persist to crawl state.
    """
    file_list = _list_files(
        service, folder_id, allowed_mimes, modified_after, recursive, max_results
    )
    total = len(file_list)
    logger.info(f"[Drive] Found {total} file(s) to evaluate")

    skipped = 0
    for i, meta in enumerate(file_list, 1):
        name           = meta.get("name", "")
        size           = meta.get("size", "?")
        file_id        = meta.get("id", "")
        modified_time  = meta.get("modifiedTime", "")
        indexed_name   = _expected_indexed_name(meta)

        # ── Incremental check: skip files that haven't changed ────────────────
        if incremental and _get_crawl_state is not None and file_id:
            state = _get_crawl_state(file_id)
            if state is not None:
                if state.get("modified_time") == modified_time:
                    logger.info(
                        f"[Drive] Skipping unchanged file: {name}  "
                        f"(modifiedTime={modified_time})"
                    )
                    skipped += 1
                    if on_skip is not None:
                        try:
                            on_skip(name)
                        except Exception:
                            pass
                    continue  # ← never download

                # File changed — detect rename
                old_name = state.get("indexed_name") or ""
                if old_name and old_name != indexed_name:
                    logger.info(
                        f"[Drive] Processing renamed file: {old_name!r} → {indexed_name!r}"
                    )
                    meta["_action"]           = "renamed"
                    meta["_old_indexed_name"] = old_name
                else:
                    logger.info(
                        f"[Drive] Processing modified file: {name}  "
                        f"(prev={state.get('modified_time')!r}, "
                        f"now={modified_time!r})"
                    )
                    meta["_action"] = "modified"
            else:
                logger.info(f"[Drive] Processing new file: {name}  ({size} bytes)")
                meta["_action"] = "new"
        else:
            logger.info(f"[Drive] Processing file: {name}  ({size} bytes)  [{i}/{total}]")
            meta["_action"] = "full"

        meta["_modified_time"] = modified_time
        meta["_indexed_name"]  = indexed_name

        logger.debug(f"[Drive] {i}/{total}  {name}  ({size} bytes)")
        try:
            filename, content = _download_file(service, meta)
            if not content:
                continue
            yield filename, content, meta
        except Exception as e:
            logger.warning(f"[Drive] Skipping {name}: {e}")

    if incremental and skipped:
        logger.info(f"[Drive] Skipped {skipped} unchanged file(s)")


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
    recursive:      bool             = True,
    max_results:    int              = DEFAULT_MAX,
    api_url:        str              = DEFAULT_API_URL,
    api_key:        str              = DEFAULT_API_KEY,
    dry_run:        bool             = False,
    user_id:        str              = "",
    user_email:     str              = "",
    service                          = None,   # pre-built Drive service (web OAuth path)
    incremental:    bool             = True,   # skip unchanged files by default
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
        incremental    = incremental,
    ):
        drive_file_id  = file_meta.get("id", "")
        action         = file_meta.get("_action", "full")
        modified_time  = file_meta.get("_modified_time", "")
        indexed_name   = file_meta.get("_indexed_name", filename)
        old_indexed    = file_meta.get("_old_indexed_name")  # set only for renames

        stats["total"] += 1

        if dry_run:
            tag = f"[{action.upper()}]" if action != "full" else ""
            logger.info(
                f"[DRY RUN] {tag} Would upload: {filename}  ({len(content):,} bytes)"
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

        # ── For renamed files: delete stale Qdrant entries by old source name ─
        # The /upload auto-index uses clean_first=True keyed on the NEW filename.
        # Old entries stored under the previous name would otherwise persist forever.
        if old_indexed:
            try:
                _delete_old_source(api_url, api_key, old_indexed)
                logger.info(
                    f"[Drive] Deleted stale Qdrant entries for renamed source: {old_indexed!r}"
                )
            except Exception as e:
                logger.warning(
                    f"[Drive] Could not delete old Qdrant entries for {old_indexed!r}: {e}"
                )

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
                f"[Drive] ✓  {filename}  →  {chunks} chunks indexed  "
                f"(action={action}, public={is_public}, shared_with={len(allowed_users)})"
            )
            stats["indexed"] += 1

            # ── Persist crawl state so the next run can skip this file ────────
            if _DRIVE_STORE_OK and drive_file_id and _update_crawl_state is not None and modified_time:
                try:
                    _update_crawl_state(drive_file_id, modified_time, indexed_name)
                except Exception as e:
                    logger.warning(f"[Drive] Could not save crawl state for {filename}: {e}")

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


def _delete_old_source(api_url: str, api_key: str, source: str) -> None:
    """
    Call DELETE /index/{source} on the RAG server to remove stale Qdrant entries
    for a renamed file.  Uses the same server the crawler uploads to so the
    Qdrant client config is always consistent.
    """
    headers: dict = {}
    if api_key:
        headers["X-API-Key"] = api_key
    resp = requests.delete(
        f"{api_url}/index/{source}",
        headers=headers,
        timeout=15,
    )
    if not resp.ok and resp.status_code != 404:
        resp.raise_for_status()


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
