"""
api.py — FastAPI server for the document extraction pipeline.

Usage:
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from file_processor.extract import _to_dict, process_file
from file_processor.parser import SUPPORTED

# ── Auth manager + conversation store (always available — PostgreSQL or in-process fallback)
try:
    from file_processor.auth_manager import (
        get_or_create_google_user as _auth_google_user,
        get_user_by_id as _auth_get_user,
        login_user as _auth_login,
        register_user as _auth_register,
        verify_token as _auth_verify_token,
    )
    from file_processor.conversation_store import (
        create_or_touch_session as _conv_touch,
        delete_session as _conv_delete_session,
        list_sessions as _conv_list_sessions,
        load_messages as _conv_load_messages,
        save_messages as _conv_save_messages,
        update_session_label as _conv_update_label,
        share_conversation as _conv_share,
        get_shared_conversation as _conv_get_shared,
    )
    AUTH_ENABLED = True
    logger.info("[AUTH] Auth manager + conversation store ready")
except Exception as _auth_err:
    AUTH_ENABLED = False
    _auth_register = _auth_login = _auth_verify_token = _auth_get_user = _auth_google_user = None  # type: ignore
    _conv_touch = _conv_list_sessions = _conv_load_messages = _conv_save_messages = None  # type: ignore
    _conv_update_label = _conv_delete_session = _conv_share = _conv_get_shared = None  # type: ignore
    logger.warning(f"[AUTH] Auth/conversation store unavailable: {_auth_err}")

# ── Drive permission store (optional — degrades gracefully if PG is down) ─────
try:
    from file_processor.drive_store import list_drive_files as _list_drive_files
    DRIVE_STORE_ENABLED = True
    logger.info("[DRIVE STORE] drive_store ready")
except Exception as _ds_err:
    _list_drive_files = None  # type: ignore
    DRIVE_STORE_ENABLED = False
    logger.warning(f"[DRIVE STORE] Drive store unavailable: {_ds_err}")

# ── Google OAuth config ───────────────────────────────────────────────────────
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")
FRONTEND_URL         = os.getenv("FRONTEND_URL", "http://localhost:5173")
GOOGLE_AUTH_ENABLED  = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)

# ── CSV executor (optional — requires PostgreSQL + GROQ_API_KEY) ──────────────
# We import from the 'csv' package. The root directory is already in the
# search path, and we've added an __init__.py to the csv folder.
try:
    from csv_query_engine.csv_executer import (
        get_table_info as _csv_table_info,
    )
    from csv_query_engine.csv_executer import (
        import_csv as _csv_import,
    )
    from csv_query_engine.csv_executer import (
        list_tables as _csv_list_tables,
    )
    from csv_query_engine.csv_executer import (
        query_auto as _csv_query_auto,
    )
    from csv_query_engine.csv_executer import (
        query_table as _csv_query_table,
    )
    CSV_ENABLED = True
except (ImportError, ModuleNotFoundError) as _csv_err:
    logger.warning(f"CSV executor not available ({_csv_err}). "
                   "CSV upload and query endpoints will return 503.")
    CSV_ENABLED = False

# ── Indexing (optional — requires FlagEmbedding + qdrant-client + Qdrant) ─────
# Imports from file_preparation/indexing/ (store + indexer pipeline).
# Embedder lives in file_preparation/embedding/ and is imported by indexer.
try:
    from file_preparation.indexing.indexer import index_chunks as _emb_index_chunks  # type: ignore[import]
    from file_preparation.indexing.store import (
        COLLECTION_NAME as _EMB_DEFAULT_COLLECTION,
    )
    from file_preparation.indexing.store import (
        create_snapshot as _emb_create_snapshot,
    )
    from file_preparation.indexing.store import (
        delete_by_source as _emb_delete_by_source,
    )
    from file_preparation.indexing.store import (
        delete_snapshot as _emb_delete_snapshot,
    )
    from file_preparation.indexing.store import (
        ensure_collection as _emb_ensure_collection,
    )
    from file_preparation.indexing.store import (
        get_chunks_by_doc_id as _emb_get_chunks_by_doc_id,
    )
    from file_preparation.indexing.store import (  # type: ignore[import]
        get_client as _emb_get_client,
    )
    from file_preparation.indexing.store import (
        get_collection_stats as _emb_collection_stats,
    )
    from file_preparation.indexing.store import (
        get_neighbors as _emb_get_neighbors,
    )
    from file_preparation.indexing.store import (
        list_indexed_documents as _emb_list_indexed_docs,
    )
    from file_preparation.indexing.store import (
        list_snapshots as _emb_list_snapshots,
    )
    from file_preparation.retrieval.retriever import (
        RetrievalResult as _RetrievalResult,
    )
    from file_preparation.retrieval.retriever import (
        RetrievedChunk as _RC,
    )
    from file_preparation.retrieval.retriever import (
        SourceFilter as _SourceFilter,
    )
    from file_preparation.retrieval.retriever import (
        multihop_retrieve as _retr_multihop,
    )
    from file_preparation.retrieval.retriever import (  # type: ignore[import]
        retrieve as _retr_retrieve,
    )
    from file_preparation.retrieval.retriever import (
        retrieve_evidence as _retr_retrieve_evidence,
    )
    from file_preparation.retrieval.retriever import (
        score_answer_confidence as _retr_score_confidence,
    )
    # Cache a single client instance — reused across all requests
    _qdrant_client = _emb_get_client()
    _qdrant_client.get_collections()   # verify connectivity at startup
    EMBEDDING_ENABLED = True
except Exception as _emb_err:
    import traceback
    _qdrant_client = None
    print(f"\n[EMBEDDING] Import failed — embedding endpoints will return 503.\nReason: {_emb_err}\n{traceback.format_exc()}")
    EMBEDDING_ENABLED = False

# ── Memory layer (conversation buffer + query rewriter) ──────────────────────
try:
    from file_preparation.memory import (
        MemoryContext as _MemoryContext,
    )
    from file_preparation.memory import (
        clear_session as _mem_clear_session,
    )
    from file_preparation.memory import (
        clear_user_facts as _mem_clear_user_facts,
    )
    from file_preparation.memory import (
        delete_all_preferences as _mem_delete_all_prefs,
    )
    from file_preparation.memory import (
        delete_fact as _mem_delete_fact,
    )
    from file_preparation.memory import (
        delete_preference as _mem_delete_pref,
    )
    from file_preparation.memory import (
        extract_and_store_facts as _mem_extract_facts,
    )
    from file_preparation.memory import (
        extract_and_store_preferences as _mem_extract_prefs,
    )
    from file_preparation.memory import (
        get_facts as _mem_get_facts,
    )
    from file_preparation.memory import (
        list_preferences as _mem_list_prefs,
    )
    from file_preparation.memory import (
        load_memory_context as _mem_load_context,
    )
    from file_preparation.memory import (
        read_all_turns as _mem_read_all_turns,
    )
    from file_preparation.memory import (
        read_buffer as _mem_read_buffer,
    )
    from file_preparation.memory import (
        recall_preferences as _mem_recall_prefs,
    )
    from file_preparation.memory import (
        # Sprint 5 — semantic user memory
        remember_preference as _mem_remember_pref,
    )
    from file_preparation.memory import (
        semantic_memory_available as _mem_semantic_available,
    )
    from file_preparation.memory import (
        # Sprint 4 — structured user facts
        set_fact as _mem_set_fact,
    )
    from file_preparation.memory import (  # type: ignore[import]
        write_turn as _mem_write_turn,
    )
    from file_preparation.memory.buffer import (
        TOKEN_BUDGET as _MEM_TOKEN_BUDGET,
    )
    from file_preparation.memory.buffer import (
        get_summary as _mem_get_summary,
    )
    from file_preparation.memory.buffer import (  # type: ignore[import]
        should_summarise as _mem_should_summarise,
    )
    from file_preparation.memory.summariser import summarise_session as _mem_summarise  # type: ignore[import]
    MEMORY_ENABLED = True
    print("[MEMORY] Conversation buffer + query rewriter + summariser + user memory available.")
except Exception as _mem_err:
    _mem_write_turn         = None   # type: ignore[assignment]
    _mem_read_buffer        = None   # type: ignore[assignment]
    _mem_read_all_turns     = None   # type: ignore[assignment]
    _mem_clear_session      = None   # type: ignore[assignment]
    _mem_load_context       = None   # type: ignore[assignment]
    _MemoryContext          = None   # type: ignore[assignment]
    _mem_should_summarise   = None   # type: ignore[assignment]
    _mem_get_summary        = None   # type: ignore[assignment]
    _MEM_TOKEN_BUDGET       = 1500   # type: ignore[assignment]
    _mem_summarise          = None   # type: ignore[assignment]
    _mem_set_fact           = None   # type: ignore[assignment]
    _mem_get_facts          = None   # type: ignore[assignment]
    _mem_delete_fact        = None   # type: ignore[assignment]
    _mem_clear_user_facts   = None   # type: ignore[assignment]
    _mem_extract_facts      = None   # type: ignore[assignment]
    _mem_extract_prefs      = None   # type: ignore[assignment]
    _mem_remember_pref      = None   # type: ignore[assignment]
    _mem_recall_prefs       = None   # type: ignore[assignment]
    _mem_delete_pref        = None   # type: ignore[assignment]
    _mem_delete_all_prefs   = None   # type: ignore[assignment]
    _mem_list_prefs         = None   # type: ignore[assignment]
    _mem_semantic_available = None   # type: ignore[assignment]
    MEMORY_ENABLED          = False
    print(f"[MEMORY] Unavailable ({_mem_err}) — memory_enabled requests will be ignored.")

# ── LLM-as-a-Judge (optional — requires groq SDK + JUDGE_API_KEY / GROQ_API_KEY) ──
try:
    from file_preparation.evaluation import judge_answer as _judge_answer  # type: ignore[import]
    from file_preparation.evaluation import stream_comparison as _stream_comparison  # type: ignore[import]
    JUDGE_ENABLED = True
    print("[JUDGE] LLM-as-a-Judge available (qwen/qwen3-32b via Groq).")
except Exception as _judge_err:
    _judge_answer      = None   # type: ignore[assignment]
    _stream_comparison = None   # type: ignore[assignment]
    JUDGE_ENABLED      = False
    print(f"[JUDGE] Unavailable ({_judge_err}) — judge=true requests will be skipped.")

# ── Answer generator (Qwen2.5-7B via Ollama only) ───────────────────────────
try:
    from file_preparation.generation.answer_generator import (
        _CITATION_RE as _CITATION_RE_SSE,  # reused in /ask/stream done event
    )
    from file_preparation.generation.answer_generator import (
        CUSTOMER_SUPPORT_PERSONA as _CS_PERSONA,
    )
    from file_preparation.generation.answer_generator import (  # type: ignore[import]
        AnswerGenerator as _AnswerGenerator,
    )
    from file_preparation.generation.answer_generator import (
        GenerationConfig as _GenerationConfig,
    )
    # GROQ_GENERATION_API_KEY  — dedicated key for answer generation (highest token usage).
    # Falls back to GROQ_API_KEY so existing setups need no change.
    _gen_api_key = os.environ.get("GROQ_GENERATION_API_KEY") or os.environ.get("GROQ_API_KEY")
    _answer_gen        = _AnswerGenerator(groq_api_key=_gen_api_key)
    # Seed the module-level singleton so rag_graph nodes calling get_generator()
    # receive an instance keyed with GROQ_GENERATION_API_KEY (not the base key).
    from file_preparation.generation import get_generator as _seed_gen
    _seed_gen(_gen_api_key)
    GENERATION_ENABLED = True
    _groq_model = os.environ.get("GROQ_GENERATION_MODEL", "qwen/qwen3-coder-480b-a35b-instruct")
    if _gen_api_key:
        _key_label = "GROQ_GENERATION_API_KEY" if os.environ.get("GROQ_GENERATION_API_KEY") else "GROQ_API_KEY"
        print(f"[GEN] AnswerGenerator initialised (primary: Groq {_groq_model}; key: {_key_label}; fallback: Qwen2.5-7B via Ollama).")
    else:
        print("[GEN] AnswerGenerator initialised (Ollama Qwen2.5-7B — set GROQ_API_KEY to enable Groq primary).")
except Exception as _gen_err:
    _answer_gen        = None
    GENERATION_ENABLED = False
    _CITATION_RE_SSE   = None   # type: ignore[assignment]
    _CS_PERSONA        = ""     # type: ignore[assignment]
    print(f"[GEN] AnswerGenerator unavailable ({_gen_err}) — /ask will be disabled.")

# ── Intent classifier (customer support pre-retrieval classification) ─────────
try:
    from file_preparation.intent import classify_intent as _classify_intent  # type: ignore[import]
    INTENT_ENABLED = True
    print("[INTENT] Customer support intent classifier available.")
except Exception as _intent_err:
    _classify_intent = None  # type: ignore[assignment]
    INTENT_ENABLED   = False
    print(f"[INTENT] Unavailable ({_intent_err}) — intent classification will be skipped.")

# ── Escalation handler ────────────────────────────────────────────────────────
try:
    from file_preparation.escalation import should_escalate as _should_escalate  # type: ignore[import]
    ESCALATION_ENABLED = True
    print("[ESCALATION] Escalation handler available.")
except Exception as _esc_err:
    _should_escalate   = None  # type: ignore[assignment]
    ESCALATION_ENABLED = False
    print(f"[ESCALATION] Unavailable ({_esc_err}) — escalation checks will be skipped.")

# ── Feedback store ────────────────────────────────────────────────────────────
try:
    from file_preparation.feedback import (
        FeedbackEntry as _FeedbackEntry,
    )
    from file_preparation.feedback import (
        get_feedback_summary as _get_feedback_summary,
    )
    from file_preparation.feedback import (
        list_feedback as _list_feedback,
    )
    from file_preparation.feedback import (  # type: ignore[import]
        store_feedback as _store_feedback,
    )
    FEEDBACK_ENABLED = True
    print("[FEEDBACK] Feedback store available.")
except Exception as _fb_err:
    _store_feedback       = None  # type: ignore[assignment]
    _get_feedback_summary = None  # type: ignore[assignment]
    _list_feedback        = None  # type: ignore[assignment]
    _FeedbackEntry        = None  # type: ignore[assignment]
    FEEDBACK_ENABLED      = False
    print(f"[FEEDBACK] Unavailable ({_fb_err}) — feedback endpoints will return 503.")

# ── Groq singleton ───────────────────────────────────────────────────────────
# A single Groq client is shared across /ask and any future Groq-backed routes.
# Re-instantiating Groq() on every request creates a new HTTP connection pool,
# adding measurable latency and wasting sockets under concurrent load.

_groq_client = None
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

try:
    from groq import Groq as _Groq  # type: ignore[import]
    # GROQ_MEMORY_API_KEY — dedicated key for memory/retrieval/intent/summariser calls.
    # These fire on every /ask request (query rewriting, HyDE, confidence scoring,
    # fact extraction, summarisation) and compete with generation on the same rate limit.
    # Falls back to GROQ_API_KEY so existing setups need no change.
    _groq_api_key = (
        os.environ.get("GROQ_MEMORY_API_KEY")
        or os.environ.get("GROQ_API_KEY")
    )
    if _groq_api_key:
        _groq_client = _Groq(api_key=_groq_api_key)
        _mem_key_label = "GROQ_MEMORY_API_KEY" if os.environ.get("GROQ_MEMORY_API_KEY") else "GROQ_API_KEY"
        print(f"[GROQ] Memory/retrieval client initialised (key: {_mem_key_label}).")
    else:
        print("[GROQ] GROQ_API_KEY not set — /ask will return 503.")
except Exception as _groq_err:
    print(f"[GROQ] Could not initialise Groq client: {_groq_err}")

# ── Share Groq singleton with the retriever module ───────────────────────────
# retriever.py has its own lazy _get_groq_client() that would create a second
# HTTP connection pool.  Injecting the already-initialised client prevents that.
if EMBEDDING_ENABLED and _groq_client is not None:
    try:
        import retriever as _retriever_mod  # type: ignore[import]
        _retriever_mod._groq_client = _groq_client
        print("[GROQ] Shared client injected into retriever module.")
    except Exception:
        pass   # retriever not importable — already reported above

# ── LangGraph RAG pipeline (optional — requires langgraph) ───────────────────
# rag_graph is the compiled StateGraph singleton.  Both /ask and /ask/stream
# delegate their entire orchestration to it; the handlers become thin wrappers
# that build an initial RAGState dict and forward it to the graph.
_rag_graph     = None
_RAGState_type = None
RAG_GRAPH_ENABLED = False
try:
    from file_preparation.rag_graph import rag_graph as _rag_graph  # type: ignore[import]
    from file_preparation.rag_graph import RAGState as _RAGState_type  # type: ignore[import]
    RAG_GRAPH_ENABLED = True
    print("[RAG_GRAPH] LangGraph RAG pipeline compiled and ready.")
except Exception as _rg_err:
    print(f"[RAG_GRAPH] Unavailable ({_rg_err}) — /ask will use legacy inline pipeline.")

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
UPLOAD_DIR  = BASE_DIR / "uploads"
IMAGES_DIR  = BASE_DIR / "extracted_images"
RESULTS_DIR = BASE_DIR / "results"
# Persistent store for imported CSVs. The CSV query engine re-reads the source
# file on demand (query_table loads it just-in-time), so the uploaded CSV must
# outlive the temporary upload file — which is deleted right after import.
CSV_STORE_DIR = BASE_DIR / "csv_store"

for d in (UPLOAD_DIR, IMAGES_DIR, RESULTS_DIR, CSV_STORE_DIR):
    d.mkdir(exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Document Extraction API")

# CORS — required when the React UI (http://localhost:5173) talks to this server.
# In production, replace allow_origins with your actual deployed frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


# ── Background upload pool ───────────────────────────────────────────────────
# Bounded thread pool for async file processing.  A fixed max_workers cap
# prevents unbounded memory growth when many large files are uploaded at once
# (each BGE-M3 batch can use 1–2 GB RAM; 3 concurrent jobs is safe on 8 GB).
# When all worker slots are busy, the upload endpoint returns HTTP 429 so the
# client knows to retry later rather than silently queuing work forever.

_UPLOAD_MAX_WORKERS = 3
_UPLOAD_EXECUTOR    = ThreadPoolExecutor(max_workers=_UPLOAD_MAX_WORKERS)
_UPLOAD_ACTIVE      = 0          # count of jobs currently running (approximate)
_UPLOAD_ACTIVE_LOCK = threading.Lock()

# ── /ask concurrency limiter ──────────────────────────────────────────────────
# Each /ask or /ask/stream call loads the embedding model, runs a Groq LLM call,
# and (optionally) a cross-encoder reranker pass — all CPU/GPU and API-bound.
# Limiting simultaneous requests prevents OOM and Groq rate-limit cascades.
_ASK_MAX_CONCURRENT  = 5          # tune per your hardware / Groq tier
_ASK_SEMAPHORE: asyncio.Semaphore | None = None   # created lazily in the event loop

async def _get_ask_semaphore() -> asyncio.Semaphore:
    global _ASK_SEMAPHORE
    if _ASK_SEMAPHORE is None:
        _ASK_SEMAPHORE = asyncio.Semaphore(_ASK_MAX_CONCURRENT)
    return _ASK_SEMAPHORE


async def ask_concurrency_limit():
    """
    FastAPI yield dependency — acquire the /ask semaphore before the handler
    runs and release it after the response is sent (even on exception).
    Returns HTTP 503 immediately when all slots are occupied.
    """
    sem = await _get_ask_semaphore()
    if sem.locked():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Server busy: {_ASK_MAX_CONCURRENT} concurrent requests already "
                "in progress. Please retry in a few seconds."
            ),
        )
    await sem.acquire()
    try:
        yield
    finally:
        sem.release()


def _upload_worker_wrap(
    job_id:          str,
    dest:            Path,
    original_name:   str,
    uid:             str,
    suffix:          str,
    caption:         bool,
    caption_backend: str,
    password:        str,
    owner_id:        str       = "",
    owner_email:     str       = "",
    drive_file_id:   str       = "",
    is_public:       bool      = False,
    allowed_users:   list[str] | None = None,
) -> None:
    """Thin wrapper that tracks active-worker count around the real worker."""
    global _UPLOAD_ACTIVE
    with _UPLOAD_ACTIVE_LOCK:
        _UPLOAD_ACTIVE += 1
    try:
        _process_upload_background(
            job_id, dest, original_name, uid, suffix, caption, caption_backend, password,
            owner_id=owner_id, owner_email=owner_email,
            drive_file_id=drive_file_id, is_public=is_public,
            allowed_users=allowed_users or [],
        )
    finally:
        with _UPLOAD_ACTIVE_LOCK:
            _UPLOAD_ACTIVE -= 1


# ── Async job store ───────────────────────────────────────────────────────────
# Lightweight in-memory store for background upload/processing jobs.
# Keys are job UUIDs; values are status dicts.
# Thread-safe via a per-entry lock-free approach — Python dict ops are GIL-safe
# for simple assignments, which is sufficient here.

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_JOB_TTL_SECONDS = 3600   # keep completed/failed jobs for 1 hour


def _job_create(job_id: str, filename: str) -> dict:
    """Create a new job entry in the in-memory store."""
    entry: dict[str, Any] = {
        "job_id":    job_id,
        "filename":  filename,
        "status":    "pending",    # pending | processing | done | failed
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "result":    None,
        "error":     None,
    }
    with _JOBS_LOCK:
        _jobs_evict_expired()
        _JOBS[job_id] = entry
    return entry


def _job_update(job_id: str, **kwargs: Any) -> None:
    """Update fields on a job entry (thread-safe)."""
    with _JOBS_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].update(kwargs)
            _JOBS[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()


_JOB_PROCESSING_TIMEOUT = 1800   # mark stuck "processing" jobs as failed after 30 min


def _jobs_evict_expired() -> None:
    """
    Remove stale job entries (called under _JOBS_LOCK).

    Two eviction rules:
      1. done/failed jobs older than _JOB_TTL_SECONDS (1 hour) → delete.
      2. processing jobs older than _JOB_PROCESSING_TIMEOUT (30 min) → mark
         failed.  This handles background threads that crash before updating
         the job status, preventing them from hanging at "processing" forever.
    """
    now = time.time()
    to_delete = []
    for jid, j in _JOBS.items():
        age = now - datetime.fromisoformat(j["updated_at"]).timestamp()
        if j["status"] in ("done", "failed") and age > _JOB_TTL_SECONDS:
            to_delete.append(jid)
        elif j["status"] == "processing" and age > _JOB_PROCESSING_TIMEOUT:
            j["status"]     = "failed"
            j["error"]      = "Job timed out (no response from worker after 30 min)."
            j["updated_at"] = datetime.now(timezone.utc).isoformat()
    for jid in to_delete:
        del _JOBS[jid]


# ── API key authentication ────────────────────────────────────────────────────
# Set API_KEY in .env to protect write/query endpoints.
# If API_KEY is not set the server runs in open dev mode (no auth required).

def _get_api_key() -> str | None:
    """Read API_KEY from environment (loaded from .env at startup)."""
    return os.environ.get("API_KEY")


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    FastAPI dependency — validate the X-API-Key request header.

    Skipped entirely when API_KEY is not configured (dev / local mode).
    Returns 401 when a key is configured but the header is missing or wrong.
    """
    server_key = _get_api_key()
    if not server_key:
        return   # dev mode — no auth required
    if x_api_key != server_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Set X-API-Key header.",
        )


# ── JWT user authentication dependencies ─────────────────────────────────────
# These extract the current user from the Authorization: Bearer <token> header.
# Use `require_user` on routes that need a logged-in user.
# Use `require_admin` on routes that need admin access.
# Use `optional_user` on routes where authentication is optional.

async def optional_user(authorization: str = Header(default="")) -> Optional[dict]:
    """Extract user from JWT if present; return None if missing/invalid.
    Never raises — callers decide what to do with None."""
    if not AUTH_ENABLED or _auth_verify_token is None or not authorization:
        return None
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        return None
    return _auth_verify_token(token)


async def require_user(user: Optional[dict] = Depends(optional_user)) -> dict:
    """Require a valid JWT. Raises 401 if missing or invalid."""
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required. Please log in.")
    return user


async def require_admin(user: dict = Depends(require_user)) -> dict:
    """Require the admin role. Raises 403 for normal users."""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main UI."""
    html_path = BASE_DIR / "ui.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/supported-formats")
async def supported_formats():
    """Return the list of supported file extensions."""
    return {"formats": sorted(SUPPORTED)}


@app.get("/health")
async def health():
    async def check_qdrant() -> dict:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
                r = await client.get(f"{qdrant_url}/healthz")
            return {"status": "ok"} if r.status_code == 200 else \
                   {"status": "down", "error": f"HTTP {r.status_code}"}
        except Exception as e:
            return {"status": "down", "error": str(e)}

    async def check_postgres() -> dict:
        try:
            # Re-use your existing connection.py logic
            import psycopg2
            conn = psycopg2.connect(
                host=os.getenv("PG_HOST", "localhost"),
                port=os.getenv("PG_PORT", 5432),
                dbname=os.getenv("PG_DB", "csvstore"),
                user=os.getenv("PG_USER", "csvuser"),
                password=os.getenv("PG_PASSWORD", ""),
                connect_timeout=3,
            )
            conn.close()
            return {"status": "ok"}
        except Exception as e:
            return {"status": "down", "error": str(e)}

    async def check_ollama() -> dict:
        try:
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{ollama_host}/api/tags")
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                return {"status": "ok", "models": models}
            return {"status": "down", "error": f"HTTP {r.status_code}"}
        except Exception as e:
            return {"status": "down", "error": str(e)}

    async def check_groq() -> dict:
        try:
            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                return {"status": "down", "error": "GROQ_API_KEY not set"}
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            return {"status": "ok"} if r.status_code == 200 else \
                   {"status": "down", "error": f"HTTP {r.status_code}"}
        except Exception as e:
            return {"status": "down", "error": str(e)}

    async def check_groq_ocr() -> dict:
        api_key = os.getenv("GROQ_OCR_API_KEY") or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            return {"status": "down", "error": "GROQ_OCR_API_KEY not set"}
        ocr_model = os.getenv("GROQ_OCR_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        # Key presence is sufficient — Groq connectivity is already tested by check_groq()
        return {"status": "ok", "ocr_model": ocr_model}

    # Run all checks in parallel — total latency = slowest single check
    results = await asyncio.gather(
        check_qdrant(),
        check_postgres(),
        check_ollama(),
        check_groq(),
        check_groq_ocr(),
        return_exceptions=True,
    )

    services = {
        "qdrant":    results[0] if not isinstance(results[0], BaseException) else {"status": "down", "error": str(results[0])},
        "postgres":  results[1] if not isinstance(results[1], BaseException) else {"status": "down", "error": str(results[1])},
        "ollama":    results[2] if not isinstance(results[2], BaseException) else {"status": "down", "error": str(results[2])},
        "groq":      results[3] if not isinstance(results[3], BaseException) else {"status": "down", "error": str(results[3])},
        "groq_ocr":  results[4] if not isinstance(results[4], BaseException) else {"status": "down", "error": str(results[4])},
    }

    impact = {
        "qdrant":    "search and indexing unavailable — /search, /index, /ask return 503",
        "postgres":  "CSV queries and user facts unavailable — /query, /user-facts return 503",
        "ollama":    "answer generation and OCR fallback unavailable — /ask, /ask/stream, and scanned-PDF upload degraded",
        "groq":      "HyDE, query rewriting, judge, summariser degraded — fallbacks active",
        "groq_ocr":  "primary OCR unavailable (GROQ_OCR_API_KEY not set) — falling back to Ollama/Gemma4 for scanned documents",
    }

    degraded = [k for k, v in services.items() if v.get("status") != "ok"]

    # Memory backend info (no network round-trip needed)
    try:
        from file_preparation.memory.buffer import backend_name as _buf_backend
        mem_backend = _buf_backend()
    except Exception:
        mem_backend = "unknown"

    # Ask semaphore occupancy (how many /ask slots are in use right now)
    sem = _ASK_SEMAPHORE
    if sem is not None:
        ask_slots_used = _ASK_MAX_CONCURRENT - sem._value  # type: ignore[attr-defined]
    else:
        ask_slots_used = 0

    return {
        "status":         "ok" if not degraded else "degraded",
        "services":       services,
        "impact":         {k: impact[k] for k in degraded},
        "memory_backend": mem_backend,   # "redis" or "dict"
        "ask_concurrency": {
            "slots_total": _ASK_MAX_CONCURRENT,
            "slots_used":  ask_slots_used,
            "slots_free":  _ASK_MAX_CONCURRENT - ask_slots_used,
        },
    }


MAX_UPLOAD_MB    = 500
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
PARSE_TIMEOUT_S  = 120

# Magic-byte signatures mapped to the extensions they may appear as.
# Only the extensions listed here are checked; others rely on the extension alone.
# Note: .docx / .pptx / .xlsx all share the PK\x03\x04 (ZIP) magic — they are grouped
# together so that any ZIP-based Office format is accepted for any of them. The individual
# parsers downstream will error gracefully if the content doesn't match the declared type.
_MAGIC: dict[bytes, set[str]] = {
    b"%PDF"            : {".pdf"},
    b"PK\x03\x04"     : {".docx", ".pptx", ".xlsx"},  # ZIP-based Office formats (no bare .zip)
    b"\xd0\xcf\x11\xe0": {".doc", ".xls", ".ppt"},    # OLE2 (legacy Office)
    b"\xff\xd8\xff"   : {".jpg", ".jpeg"},
    b"\x89PNG\r\n"    : {".png"},
    b"GIF8"            : {".gif"},
    b"RIFF"            : {".webp"},                    # RIFF container (WebP)
    b"BM"              : {".bmp"},
}


def _check_magic(path: Path, declared_suffix: str) -> None:
    """
    Raise HTTPException 415 if the file's magic bytes contradict its extension.

    ZIP-based Office formats (.docx, .pptx, .xlsx) all share the same PK\\x03\\x04
    header — cross-confusion between them is benign (parsers fail gracefully) and
    they are treated as one group. Plain .zip is not in the allowed set so a
    renamed zip archive is still rejected.
    """
    with path.open("rb") as fh:
        header = fh.read(8)
    for magic, allowed_exts in _MAGIC.items():
        if header.startswith(magic):
            if declared_suffix not in allowed_exts:
                path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=415,
                    detail=(
                        f"File content does not match declared extension '{declared_suffix}'. "
                        "Upload rejected."
                    ),
                )


# ── Background processing function ───────────────────────────────────────────

def _process_upload_background(
    job_id:          str,
    dest:            Path,
    original_name:   str,
    uid:             str,
    suffix:          str,
    caption:         bool,
    caption_backend: str,
    password:        str,
    owner_id:        str       = "",
    owner_email:     str       = "",
    drive_file_id:   str       = "",
    is_public:       bool      = False,
    allowed_users:   list[str] | None = None,
) -> None:
    """
    Run file extraction + chunking in a background thread.

    Updates _JOBS[job_id] with status="processing", then "done" or "failed".
    Saves result JSON to results/ on success (same path as synchronous upload).

    owner_id / owner_email / drive_file_id / is_public / allowed_users are
    stamped onto every chunk's metadata so that documents uploaded via async
    mode carry the same owner + permission data as synchronous uploads.

    The job record stores only a lightweight summary — NOT the full extracted
    payload.  Full data is persisted to results/ on disk and can be fetched via
    GET /results/{filename}.  This prevents multi-MB payloads from sitting in
    the in-memory _JOBS dict for up to an hour.

    A try/finally guarantees the status is always set to "failed" even if an
    unexpected exception escapes the inner try blocks, so jobs never get stuck
    at "processing" indefinitely.
    """
    _job_update(job_id, status="processing")

    try:
        # ── CSV path ──────────────────────────────────────────────────────────
        if suffix == ".csv":
            if not CSV_ENABLED:
                _job_update(job_id, status="failed",
                            error="CSV engine unavailable (PostgreSQL / GROQ_API_KEY missing).")
                dest.unlink(missing_ok=True)
                return
            try:
                # The query engine re-reads the source CSV on demand, so copy it
                # to a persistent location BEFORE the temp upload file is deleted.
                #
                # Use a STABLE filename derived from owner + original name (no random
                # upload id), so re-uploading the same file overwrites in place and
                # registers the SAME table (import_csv does ON CONFLICT DO UPDATE)
                # instead of accumulating a new "<uid>_name" table on every upload.
                owner_tag = "".join(
                    c if c.isalnum() else "_" for c in (owner_id or "anon")
                ).strip("_") or "anon"
                csv_persist = CSV_STORE_DIR / f"{owner_tag}_{original_name}"
                shutil.copy2(dest, csv_persist)

                table_name = _csv_import(str(csv_persist))
                info       = _csv_table_info(table_name)
                data = {
                    "type"       : "csv_import",
                    "source_file": original_name,
                    "table"      : table_name,
                    "rows"       : info["shape"][0],
                    "cols"       : info["shape"][1],
                    "columns"    : info["columns"],
                }
                result_filename = f"{uid}_{Path(original_name).stem}.json"
                result_path     = RESULTS_DIR / result_filename
                result_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                _job_update(job_id, status="done", result_file=result_filename,
                            stats={"type": "csv_import", "table": table_name,
                                   "rows": data["rows"], "cols": data["cols"]})
            except Exception as e:
                _job_update(job_id, status="failed", error=str(e))
            finally:
                # Only the temporary upload file is removed; csv_persist remains
                # so the table can be queried later.
                dest.unlink(missing_ok=True)
            return

        # ── Standard extraction path ──────────────────────────────────────────
        try:
            result = process_file(
                dest,
                caption=caption,
                caption_backend=caption_backend,
                pdf_pass=password or None,
            )
        except Exception as e:
            _job_update(job_id, status="failed", error=str(e))
            return
        finally:
            # Always clean up the temp upload file
            for _ in range(5):
                try:
                    dest.unlink(missing_ok=True)
                    break
                except PermissionError:
                    time.sleep(0.3)

        result.source_file = original_name
        from file_processor.extract import _save_images
        _save_images(result.images, IMAGES_DIR, original_name)
        data = _to_dict(result, doc_uid=uid)

        # Stamp owner + Drive permission metadata — same logic as the sync path
        _eff_allowed = allowed_users or []
        if owner_id or owner_email or drive_file_id or _eff_allowed:
            for chunk in data.get("chunks", []):
                meta = chunk.setdefault("metadata", {})
                if owner_id:
                    meta["owner_id"] = owner_id
                if owner_email:
                    meta["owner_email"] = owner_email
                if drive_file_id:
                    meta["drive_file_id"] = drive_file_id
                meta["is_public"] = is_public
                if _eff_allowed:
                    meta["allowed_users"] = _eff_allowed

        for chunk in data.get("chunks", []):
            if chunk.get("type") == "image" and chunk.get("image_file"):
                chunk["image_url"] = f"/images/{chunk['image_file']}"

        result_filename = f"{uid}_{Path(original_name).stem}.json"
        result_path     = RESULTS_DIR / result_filename
        result_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        # ── Auto-index into Qdrant (async path) ──────────────────────────────
        _bg_index_stats: dict = {}
        if EMBEDDING_ENABLED and _qdrant_client is not None:
            chunks = data.get("chunks", [])
            if chunks:
                try:
                    _emb_ensure_collection(_qdrant_client)
                    _bg_index_stats = _emb_index_chunks(
                        chunks,
                        _qdrant_client,
                        source=original_name,
                        clean_first=True,
                    )
                    logger.info(
                        f"[BG JOB {job_id}] Auto-indexed "
                        f"{_bg_index_stats.get('indexed', 0)} chunks "
                        f"from '{original_name}' (owner={owner_id or 'anon'})"
                    )
                except Exception as _idx_err:
                    logger.warning(
                        f"[BG JOB {job_id}] Auto-indexing failed for '{original_name}': {_idx_err}"
                    )

        # Store only a lightweight summary — full data is on disk in results/
        stats = data.get("stats", {})
        _job_update(
            job_id,
            status="done",
            result_file=result_filename,       # ← fetch full data via GET /results/{filename}
            stats={
                "total_chunks": stats.get("total_chunks", len(data.get("chunks", []))),
                "text_chunks":  stats.get("text_chunks",  0),
                "table_chunks": stats.get("table_chunks", 0),
                "image_chunks": stats.get("image_chunks", 0),
                "total_tokens": stats.get("total_tokens", 0),
                "indexed":      _bg_index_stats.get("indexed", 0),
            },
        )

    except Exception as e:
        # Safety net: any uncaught exception sets status=failed so the job
        # never stays stuck at "processing".
        logger.error(f"  Background job {job_id} crashed unexpectedly: {e}")
        _job_update(job_id, status="failed", error=f"Unexpected error: {e}")


# ── Task polling endpoint ────────────────────────────────────────────────────

@app.get("/tasks/{job_id}")
async def get_task_status(job_id: str):
    """
    Poll the status of a background upload/processing job.

    Returns:
        status="pending"    — job queued, not started yet
        status="processing" — extraction + chunking in progress
        status="done"       — job finished successfully
                              result_file: filename in results/ (use GET /results/{file})
                              stats: chunk count summary
        status="failed"     — job failed; error contains the reason

    When Celery is enabled, status is read from the Celery result backend (Redis DB 2)
    so it reflects the true worker state regardless of process isolation.
    Job records are kept for 1 hour after completion, then evicted.
    """
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found or expired.")
    return JSONResponse(content=job)


@app.post("/upload", dependencies=[Depends(require_api_key)])
async def upload(
    file:            UploadFile      = File(...),
    caption:         bool            = Form(True),
    caption_backend: str             = Form("groq"),
    password:        str             = Form(""),
    async_mode:      bool            = Form(False),
    owner_id:        str             = Form(""),   # override from crawler; JWT wins for UI uploads
    owner_email:     str             = Form(""),
    drive_file_id:   str             = Form(""),
    is_public:       bool            = Form(False),
    allowed_users:   str             = Form(""),
    current_user:    Optional[dict]  = Depends(optional_user),
):
    """
    Upload a file, extract its content and return chunks.

    **Synchronous mode** (default, async_mode=false):
      Blocks until extraction is complete and returns the full chunk payload.
      Suitable for small files (< ~5 MB) or interactive use.

    **Async mode** (async_mode=true):
      Streams the file to disk, then immediately returns a `job_id`.
      Processing runs in a background thread.
      Poll `GET /tasks/{job_id}` to check progress and retrieve the result.
      Use this for large PDFs, DOCX with many images, or batch uploads.

    caption=true (default) activates image captioning via Groq / Llama-4-Scout.
    caption_backend: "groq" (cloud, default) or "llava" (local Ollama).
    Disable captioning only for testing — uncaptioned image chunks embed poorly.
    """
    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format '{suffix}'. Supported: {sorted(SUPPORTED)}",
        )

    # Save uploaded file with a unique name to avoid collisions
    uid       = uuid.uuid4().hex[:8]
    safe_name = f"{uid}_{filename}"
    dest      = UPLOAD_DIR / safe_name

    # Stream to disk while checking size limit
    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 256):   # 256 KB chunks
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {MAX_UPLOAD_MB} MB.",
                )
            out.write(chunk)

    # Validate magic bytes — reject files whose content contradicts their extension
    _check_magic(dest, suffix)

    # ── Resolve owner_id for RBAC / global KB ────────────────────────────────
    # Priority:
    #   1. If a JWT user is present, use their user_id (UI uploads).
    #      Admins get owner_id = '__global__' so their docs enter the shared KB.
    #   2. If owner_id form field is set (crawler path), use that value.
    #   3. Otherwise leave empty (anonymous upload — visible to admins only).
    if current_user:
        _jwt_uid = current_user["user_id"]
        _is_admin_upload = current_user.get("role") == "admin"
        if not owner_id:
            # UI upload: stamp with '__global__' for admins, user_id for users
            owner_id = "__global__" if _is_admin_upload else _jwt_uid
        elif _is_admin_upload and owner_id == _jwt_uid:
            # Admin explicitly used their own ID — redirect to global scope
            owner_id = "__global__"

    # ── Async mode — submit to bounded thread pool and return immediately ────
    if async_mode:
        # Reject when the pool is already at capacity to avoid OOM from
        # BGE-M3 loading in too many concurrent workers.
        with _UPLOAD_ACTIVE_LOCK:
            active = _UPLOAD_ACTIVE
        if active >= _UPLOAD_MAX_WORKERS:
            dest.unlink(missing_ok=True)
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Server is busy ({active}/{_UPLOAD_MAX_WORKERS} jobs running). "
                    "Please retry in a few seconds."
                ),
            )

        job_id = uid
        _job_create(job_id, filename)
        _UPLOAD_EXECUTOR.submit(
            _upload_worker_wrap,
            job_id, dest, filename, uid, suffix,
            caption, caption_backend, password,
            owner_id, owner_email,
            drive_file_id, is_public,
            [e.strip() for e in allowed_users.split(",") if e.strip()],
        )
        return JSONResponse(
            status_code=202,
            content={
                "job_id":    job_id,
                "filename":  filename,
                "status":    "pending",
                "poll_url":  f"/tasks/{job_id}",
                "message":   "Processing started. Poll /tasks/{job_id} for status.",
            },
        )

    # ── CSV path — import into PostgreSQL instead of chunking ────────────────
    if suffix == ".csv":
        if not CSV_ENABLED:
            dest.unlink(missing_ok=True)
            raise HTTPException(
                status_code=503,
                detail=(
                    "CSV structured-query engine is not available. "
                    "Check that PostgreSQL is running and GROQ_API_KEY is set."
                ),
            )
        try:
            table_name = _csv_import(str(dest))
            info       = _csv_table_info(table_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"CSV import failed: {e}")
        finally:
            for _ in range(5):
                try:
                    dest.unlink(missing_ok=True)
                    break
                except PermissionError:
                    time.sleep(0.3)

        data = {
            "type"       : "csv_import",
            "source_file": filename,
            "table"      : table_name,
            "rows"       : info["shape"][0],
            "cols"       : info["shape"][1],
            "columns"    : info["columns"],
        }

        result_path = RESULTS_DIR / f"{uid}_{Path(filename).stem}.json"
        result_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return JSONResponse(content=data)

    # ── Standard synchronous path — extract + chunk ───────────────────────────
    # Run in a thread so we can enforce a wall-clock timeout without blocking
    # the event loop.  concurrent.futures.Future.result(timeout=) raises
    # TimeoutError which we surface as HTTP 504.
    try:
        future = _UPLOAD_EXECUTOR.submit(
            process_file,
            dest,
            caption=caption,
            caption_backend=caption_backend,
            pdf_pass=password or None,
        )
        result = future.result(timeout=PARSE_TIMEOUT_S)
    except TimeoutError:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=504,
            detail=f"Parsing timed out after {PARSE_TIMEOUT_S} s. Try async_mode=true for large files.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always delete the upload after processing — result is saved in results/
        # Retry a few times on Windows where files may still be locked briefly
        for _ in range(5):
            try:
                dest.unlink(missing_ok=True)
                break
            except PermissionError:
                time.sleep(0.3)

    # Use original filename for consistent naming across images, chunks and results
    result.source_file = filename

    # Save extracted images
    from file_processor.extract import _save_images
    _save_images(result.images, IMAGES_DIR, filename)

    # Build output dict (RAG-ready chunks) — uses result.source_file for chunk_id and image_file
    data = _to_dict(result, doc_uid=uid)

    # Stamp owner + Drive permission metadata on every chunk when the upload
    # came from a crawler.  Stored in Qdrant payload so retrieval can be
    # filtered by owner / access-control list at query time.
    _parsed_allowed = [e.strip() for e in allowed_users.split(",") if e.strip()]
    if owner_id or owner_email or drive_file_id or _parsed_allowed:
        for chunk in data.get("chunks", []):
            meta = chunk.setdefault("metadata", {})
            if owner_id:
                meta["owner_id"] = owner_id
            if owner_email:
                meta["owner_email"] = owner_email
            if drive_file_id:
                meta["drive_file_id"] = drive_file_id
            meta["is_public"] = is_public
            if _parsed_allowed:
                meta["allowed_users"] = _parsed_allowed

    # Add accessible image URLs
    for chunk in data.get("chunks", []):
        if chunk.get("type") == "image" and chunk.get("image_file"):
            chunk["image_url"] = f"/images/{chunk['image_file']}"

    # Persist result as JSON
    result_path = RESULTS_DIR / f"{uid}_{Path(filename).stem}.json"
    result_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Auto-index into Qdrant immediately after extraction ──────────────────
    # This ensures crawled documents appear in the knowledge base without
    # requiring a separate POST /index call.
    _index_stats: dict = {}
    if EMBEDDING_ENABLED and _qdrant_client is not None:
        chunks = data.get("chunks", [])
        if chunks:
            try:
                _emb_ensure_collection(_qdrant_client)
                _index_stats = _emb_index_chunks(
                    chunks,
                    _qdrant_client,
                    source=file.filename,
                    clean_first=True,
                )
                logger.info(
                    f"[UPLOAD] Auto-indexed {_index_stats.get('indexed', 0)} chunks "
                    f"from '{file.filename}' (owner={owner_id or 'anon'})"
                )
            except Exception as _idx_err:
                logger.warning(f"[UPLOAD] Auto-indexing failed for '{file.filename}': {_idx_err}")

    data["result_file"]   = result_path.name
    data["indexed"]       = _index_stats.get("indexed", 0)
    data["index_skipped"] = _index_stats.get("skipped", 0)
    return JSONResponse(content=data)


@app.get("/results")
async def list_results(offset: int = 0, limit: int = 50):
    """
    List previously processed files, sorted by modification time (newest first).

    Args:
        offset: Number of entries to skip (for pagination).
        limit:  Maximum number of entries to return (max 200).
    """
    limit = min(limit, 200)
    all_files = sorted(RESULTS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    total     = len(all_files)
    page      = all_files[offset : offset + limit]

    results = []
    for f in page:
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            if doc.get("type") == "csv_import":
                results.append({
                    "file"  : f.name,
                    "source": doc.get("source_file", f.stem),
                    "type"  : "csv_import",
                    "stats" : {"rows": doc.get("rows", 0), "cols": doc.get("cols", 0)},
                })
            else:
                results.append({
                    "file"  : f.name,
                    "source": doc.get("source_file", f.stem),
                    "stats" : doc.get("stats", {}),
                })
        except Exception:
            pass
    return {"total": total, "offset": offset, "limit": limit, "results": results}


@app.get("/results/{filename}")
async def get_result(filename: str):
    """Fetch a specific result JSON."""
    path = RESULTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))


@app.delete("/results/{filename}", dependencies=[Depends(require_api_key)])
async def delete_result(filename: str):
    """Delete a specific result."""
    path = RESULTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    path.unlink()
    return {"deleted": filename}


# ── CSV query endpoints ───────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    table: str | None = None   # None → auto-detect from question


@app.post("/query", dependencies=[Depends(require_api_key)])
async def query_csv(req: QueryRequest):
    """
    Ask a natural language question against an imported CSV table.

    If `table` is omitted, the LLM auto-detects which table to use.
    Requires CSV_ENABLED (PostgreSQL + GROQ_API_KEY configured).
    """
    if not CSV_ENABLED:
        raise HTTPException(
            status_code=503,
            detail=(
                "CSV query engine is not available. "
                "Check that PostgreSQL is running and GROQ_API_KEY is set."
            ),
        )

    # Reject an explicitly empty table name — callers must omit the field (or
    # pass null) to trigger auto-detection.  An empty string almost always means
    # a client bug; surfacing 400 is more helpful than silently running the
    # auto-detect path and returning a confusing answer.
    if req.table is not None and not req.table.strip():
        raise HTTPException(
            status_code=400,
            detail="table name cannot be empty; omit the field or pass null to auto-detect",
        )

    try:
        if req.table:
            result = _csv_query_table(req.table, req.question, verbose=False)
        else:
            result = _csv_query_auto(req.question, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content={
        "question": result["question"],
        "table"   : result["table"],
        "answer"  : result["answer"],
        "code"    : result["code"],
        "success" : result["execution"]["success"] if result["execution"] else False,
        "error"   : result["execution"]["error"]   if result["execution"] else None,
    })


@app.get("/csv-tables")
async def csv_tables():
    """List all CSV tables currently imported in PostgreSQL."""
    if not CSV_ENABLED:
        return {"tables": [], "enabled": False}
    try:
        tables = _csv_list_tables()
        info   = {}
        for t in tables:
            try:
                i = _csv_table_info(t)
                info[t] = {"rows": i["shape"][0], "cols": i["shape"][1], "columns": i["columns"]}
            except Exception:
                info[t] = {}
        return {"tables": tables, "info": info, "enabled": True}
    except Exception as e:
        return {"tables": [], "enabled": True, "error": str(e)}


# ── Embedding endpoints ───────────────────────────────────────────────────────

def _require_embedding():
    """Raise 503 if packages are missing or Qdrant is unreachable."""
    if not EMBEDDING_ENABLED or _qdrant_client is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding engine unavailable. Check FlagEmbedding install and Qdrant on localhost:6333.",
        )


class IndexRequest(BaseModel):
    result_file: str            # filename inside results/  e.g. "abc123_report.json"
    collection:  str = _EMB_DEFAULT_COLLECTION if EMBEDDING_ENABLED else "documents"


class SourceFilterModel(BaseModel):
    """
    Typed source / metadata filter for /search and /ask.

    All fields are optional.  Multiple values → any-of (OR) match within
    that field.  All specified fields are ANDed together.

    Examples:
        {"sources": ["report.pdf"]}
        {"languages": ["en", "ar"], "types": ["text", "table"]}
        {"sources": ["q3.pdf", "q4.pdf"], "languages": ["en"]}
    """
    sources:   list[str] | None = None   # filter by source filename(s)
    languages: list[str] | None = None   # filter by ISO-639-1 language code(s)
    types:     list[str] | None = None   # filter by chunk type: text / table / image


def _build_source_filter(sf: SourceFilterModel | None) -> _SourceFilter | None:
    """Convert the Pydantic SourceFilterModel to the retriever's SourceFilter dataclass."""
    if sf is None or not EMBEDDING_ENABLED:
        return None
    if not any([sf.sources, sf.languages, sf.types]):
        return None
    return _SourceFilter(
        sources   = sf.sources,
        languages = sf.languages,
        types     = sf.types,
    )


class SearchRequest(BaseModel):
    query:         str
    collection:    str                   = _EMB_DEFAULT_COLLECTION if EMBEDDING_ENABLED else "documents"
    limit:         int                   = Field(10,  ge=1,   le=100)
    filters:       dict | None           = None    # e.g. {"language": "en", "type": ["text","table"]}
    source_filter: SourceFilterModel | None = None  # typed source / language / type filter
    min_score:     float | None          = Field(None, ge=0.0, le=1.0)
    rerank:        bool                  = False   # apply cross-encoder reranking (jina-reranker-v2-base-multilingual)
    use_hyde:      bool                  = False   # expand query via HyDE before embedding (Groq)
    mmr:           bool                  = False   # Maximal Marginal Relevance diversification
    mmr_lambda:    float                 = Field(0.5, ge=0.0, le=1.0)
    decompose:     bool                  = False   # decompose compound query into sub-queries (Groq)


@app.get("/qdrant-status")
async def qdrant_status():
    """Check Qdrant connectivity using the cached client."""
    if not EMBEDDING_ENABLED or _qdrant_client is None:
        return {"enabled": False, "reason": "FlagEmbedding or qdrant-client not installed"}
    try:
        collections = _qdrant_client.get_collections().collections
        return {
            "enabled":     True,
            "collections": [c.name for c in collections],
        }
    except Exception as e:
        return {"enabled": False, "reason": str(e)}


@app.get("/collection-stats")
async def collection_stats(
    collection: str = _EMB_DEFAULT_COLLECTION if EMBEDDING_ENABLED else "documents",
):
    """
    Return point count, vector count, and status for the active Qdrant collection.

    Useful for monitoring index health without opening the Qdrant dashboard.
    """
    _require_embedding()
    assert _qdrant_client is not None
    try:
        stats = _emb_collection_stats(_qdrant_client, collection=collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collection stats: {e}")
    return JSONResponse(content=stats)


# ── Backup / snapshot endpoints ───────────────────────────────────────────────

@app.get("/backup")
async def list_backups(
    collection: str = _EMB_DEFAULT_COLLECTION if EMBEDDING_ENABLED else "documents",
):
    """
    List all available Qdrant snapshots for the collection.

    Snapshots are stored server-side in the Qdrant container.
    Mount /qdrant/snapshots to a host volume for persistence.
    """
    _require_embedding()
    assert _qdrant_client is not None
    try:
        snapshots = _emb_list_snapshots(_qdrant_client, collection=collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list snapshots: {e}")
    return JSONResponse(content={"collection": collection, "snapshots": snapshots})


@app.post("/backup", dependencies=[Depends(require_api_key)])
async def create_backup(
    collection: str = _EMB_DEFAULT_COLLECTION if EMBEDDING_ENABLED else "documents",
):
    """
    Create a Qdrant snapshot of the collection.

    The snapshot is stored server-side inside the Qdrant container at
    /qdrant/snapshots/<collection>/. Mount that path to persist across restarts.
    Returns the snapshot name and size.
    """
    _require_embedding()
    assert _qdrant_client is not None
    try:
        snapshot = _emb_create_snapshot(_qdrant_client, collection=collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create snapshot: {e}")
    return JSONResponse(content=snapshot)


@app.delete("/backup/{snapshot_name}", dependencies=[Depends(require_api_key)])
async def delete_backup(
    snapshot_name: str,
    collection:    str = _EMB_DEFAULT_COLLECTION if EMBEDDING_ENABLED else "documents",
):
    """Delete a named snapshot from Qdrant."""
    _require_embedding()
    assert _qdrant_client is not None
    if not snapshot_name.strip():
        raise HTTPException(status_code=422, detail="snapshot_name must not be empty.")
    try:
        _emb_delete_snapshot(_qdrant_client, snapshot_name=snapshot_name, collection=collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete snapshot: {e}")
    return JSONResponse(content={"deleted": snapshot_name, "collection": collection})


# ── Memory session endpoints ──────────────────────────────────────────────────

@app.post("/sessions", dependencies=[Depends(require_api_key)])
async def create_session():
    """
    Create a new conversation session.

    Returns a fresh session_id (UUID) that the client should pass in subsequent
    /ask requests as `session_id` with `memory_enabled: true`.

    NOTE: We intentionally do NOT pre-register the session in conversation_store
    here because the user_id is not known at this point (this endpoint is protected
    by an API key, not a JWT).  The first call to save_messages() (via
    POST /conversations/{id}/messages) will INSERT the row with the correct
    user_id.  Pre-inserting with "__system__" caused ownership checks in
    save_messages() to fail, silently dropping all message saves.
    """
    session_id = str(uuid.uuid4())
    return JSONResponse(content={"session_id": session_id})


# ══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    password: str = Field(..., min_length=4)
    email:    str = ""


class LoginRequest(BaseModel):
    username_or_email: str
    password:          str


@app.post("/auth/register")
async def auth_register(req: RegisterRequest):
    if not AUTH_ENABLED:
        raise HTTPException(503, "Auth service unavailable")
    assert _auth_register is not None and _auth_verify_token is not None
    token, err = _auth_register(req.username, req.password, req.email)
    if err or not token:
        raise HTTPException(400, err or "Unknown registration error")
    # decode token to get user_id
    info = _auth_verify_token(token)
    if not info:
        raise HTTPException(500, "Failed to verify token after registration")
    return {"token": token, "user_id": info["user_id"], "username": info["username"], "email": info.get("email", ""), "role": info.get("role", "user")}


@app.post("/auth/login")
async def auth_login(req: LoginRequest):
    if not AUTH_ENABLED:
        raise HTTPException(503, "Auth service unavailable")
    assert _auth_login is not None
    user, err = _auth_login(req.username_or_email, req.password)
    if err or not user:
        raise HTTPException(401, err or "Unknown login error")
    return user


@app.get("/auth/google")
async def auth_google_start():
    """
    Redirect browser to Google's OAuth consent screen.

    Uses manual PKCE (S256) so the flow works with both installed/desktop
    credentials.json (which Google now requires PKCE for) and web-app clients.
    The code_verifier is embedded in the `state` parameter so it is echoed back
    by Google in the callback without needing a server-side session.
    """
    if not GOOGLE_AUTH_ENABLED or not AUTH_ENABLED:
        raise HTTPException(503, "Google OAuth not configured or Auth unavailable")
    try:
        import hashlib as _hl
        import base64 as _b64
        import secrets as _sec
        import json as _json
        import urllib.parse as _up

        # PKCE: generate verifier + challenge
        code_verifier  = _b64.urlsafe_b64encode(_sec.token_bytes(32)).rstrip(b"=").decode()
        code_challenge = _b64.urlsafe_b64encode(
            _hl.sha256(code_verifier.encode()).digest()
        ).rstrip(b"=").decode()

        # Encode verifier in state so the callback can retrieve it without
        # a server-side session (Google echoes state back unchanged).
        state = _b64.urlsafe_b64encode(
            _json.dumps({"cv": code_verifier}).encode()
        ).decode()

        params = {
            "client_id":             GOOGLE_CLIENT_ID,
            "redirect_uri":          GOOGLE_REDIRECT_URI,
            "response_type":         "code",
            "scope":                 "openid email profile",
            "state":                 state,
            "code_challenge":        code_challenge,
            "code_challenge_method": "S256",
            "access_type":           "online",
            "prompt":                "select_account",
        }
        auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + _up.urlencode(params)
        return RedirectResponse(url=auth_url)
    except Exception as e:
        raise HTTPException(500, f"Google OAuth error: {e}")


@app.get("/auth/google/callback")
async def auth_google_callback(code: str = "", error: str = "", state: str = ""):
    """
    Exchange authorization code for tokens, find/create user, return HTML that
    postMessages the JWT back to the opener popup.

    Uses raw requests for the token exchange so the code_verifier from the PKCE
    flow (encoded in `state` by auth_google_start) can be passed explicitly.
    """
    if error:
        html = f"""<script>window.opener?.postMessage({{type:'google_auth',error:'{error}'}}, '{FRONTEND_URL}');window.close();</script>"""
        return HTMLResponse(html)
    if not GOOGLE_AUTH_ENABLED or not AUTH_ENABLED:
        html = f"""<script>window.opener?.postMessage({{type:'google_auth',error:'OAuth not configured'}}, '{FRONTEND_URL}');window.close();</script>"""
        return HTMLResponse(html)
    if not code:
        html = f"""<script>window.opener?.postMessage({{type:'google_auth',error:'No code returned'}}, '{FRONTEND_URL}');window.close();</script>"""
        return HTMLResponse(html)
    try:
        import base64 as _b64
        import json as _json
        import requests as _req

        # Recover code_verifier from state
        code_verifier = ""
        if state:
            try:
                # urlsafe_b64decode requires padding to be a multiple of 4
                padded = state + "=" * (-len(state) % 4)
                state_data  = _json.loads(_b64.urlsafe_b64decode(padded).decode())
                code_verifier = state_data.get("cv", "")
            except Exception as _se:
                logger.warning(f"[AUTH] Could not decode PKCE state: {_se}")

        # Exchange authorization code for access token (raw request — avoids
        # google_auth_oauthlib re-generating PKCE params that don't match).
        token_payload: dict = {
            "code":          code,
            "client_id":     GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri":  GOOGLE_REDIRECT_URI,
            "grant_type":    "authorization_code",
        }
        if code_verifier:
            token_payload["code_verifier"] = code_verifier

        token_resp = _req.post(
            "https://oauth2.googleapis.com/token",
            data=token_payload,
            timeout=15,
        )
        token_data   = token_resp.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise ValueError(f"Token exchange failed: {token_data}")

        # Fetch user profile
        user_info = _req.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        ).json()

        google_id = user_info.get("sub")
        email     = user_info.get("email", "")
        name      = user_info.get("name", email.split("@")[0])
        if not google_id:
            raise ValueError("No sub in userinfo")
        assert _auth_google_user is not None
        user = _auth_google_user(google_id, email, name)
        if not user:
            raise ValueError("Failed to create/find Google user")

        html = f"""<!DOCTYPE html><html><body><script>
            try {{
                window.opener.postMessage({{type:'google_auth', user:{_json.dumps(user)}}}, '{FRONTEND_URL}');
            }} catch(e) {{
                // fallback: store in sessionStorage and redirect
                sessionStorage.setItem('google_auth_result', JSON.stringify({{type:'google_auth', user:{_json.dumps(user)}}}));
                window.location.href = '{FRONTEND_URL}?google_auth=1';
            }}
            window.close();
        </script></body></html>"""
        return HTMLResponse(html)
    except Exception as e:
        logger.error(f"[AUTH] Google callback error: {e}")
        html = f"""<script>window.opener?.postMessage({{type:'google_auth',error:'Authentication failed'}}, '{FRONTEND_URL}');window.close();</script>"""
        return HTMLResponse(html)


@app.get("/auth/me")
async def auth_me(authorization: str = Header(default="")):
    """Validate JWT and return current user info."""
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(401, "No token")
    if not AUTH_ENABLED:
        raise HTTPException(503, "Auth service unavailable")
    assert _auth_verify_token is not None and _auth_get_user is not None
    info = _auth_verify_token(token)
    if not info:
        raise HTTPException(401, "Invalid or expired token")
    user = _auth_get_user(info["user_id"])
    if not user:
        raise HTTPException(404, "User not found")
    return user


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN MANAGEMENT ROUTES
# All routes require the "admin" role (enforced by require_admin dependency).
# Normal users receive HTTP 403 before any handler body executes.
# ══════════════════════════════════════════════════════════════════════════════

class CreateAdminRequest(BaseModel):
    username: str
    password: str
    email:    str = ""


class UpdateRoleRequest(BaseModel):
    role: str   # "user" | "admin"


@app.post("/admin/users", dependencies=[Depends(require_admin)])
async def admin_create_user(req: CreateAdminRequest):
    """Create a new admin account. Requires caller to be an admin."""
    if not AUTH_ENABLED:
        raise HTTPException(503, "Auth service unavailable")
    from file_processor.auth_manager import create_admin_user
    user, err = create_admin_user(req.username, req.password, req.email)
    if err or not user:
        raise HTTPException(400, detail=err or "Unknown error creating admin user")
    return {"message": "Admin user created", "user_id": user["user_id"],
            "username": user["username"], "role": user["role"]}


@app.get("/admin/users", dependencies=[Depends(require_admin)])
async def admin_list_users():
    """List all registered users (admin only)."""
    if not AUTH_ENABLED:
        raise HTTPException(503, "Auth service unavailable")
    from file_processor.auth_manager import _PG_OK, _get_conn, _FALLBACK
    users: list[dict] = []
    if _PG_OK:
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT user_id, username, email, role, created_at "
                        "FROM auth_users ORDER BY created_at DESC"
                    )
                    for row in cur.fetchall():
                        users.append({
                            "user_id":    row[0],
                            "username":   row[1],
                            "email":      row[2] or "",
                            "role":       row[3] or "user",
                            "created_at": row[4].isoformat() if row[4] else None,
                        })
        except Exception as e:
            raise HTTPException(500, detail=f"Failed to list users: {e}")
    else:
        users = [
            {"user_id": v["user_id"], "username": v["username"],
             "email": v.get("email", ""), "role": v.get("role", "user"),
             "created_at": None}
            for v in _FALLBACK.values()
        ]
    return {"users": users, "total": len(users)}


@app.put("/admin/users/{user_id}/role", dependencies=[Depends(require_admin)])
async def admin_update_role(user_id: str, req: UpdateRoleRequest, admin: dict = Depends(require_admin)):
    """Promote or demote a user. Admins cannot demote themselves."""
    if not AUTH_ENABLED:
        raise HTTPException(503, "Auth service unavailable")
    if req.role not in ("user", "admin"):
        raise HTTPException(400, detail="role must be 'user' or 'admin'")
    if user_id == admin["user_id"] and req.role != "admin":
        raise HTTPException(400, detail="You cannot remove your own admin role")
    from file_processor.auth_manager import _PG_OK, _get_conn, _FALLBACK
    if _PG_OK:
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE auth_users SET role=%s WHERE user_id=%s RETURNING username",
                        (req.role, user_id),
                    )
                    row = cur.fetchone()
                conn.commit()
            if not row:
                raise HTTPException(404, "User not found")
            return {"message": f"Role updated to '{req.role}'", "username": row[0]}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, detail=f"Role update failed: {e}")
    else:
        if user_id not in _FALLBACK:
            raise HTTPException(404, "User not found")
        _FALLBACK[user_id]["role"] = req.role
        return {"message": f"Role updated to '{req.role}'"}


@app.delete("/admin/users/{user_id}", dependencies=[Depends(require_admin)])
async def admin_delete_user(user_id: str, admin: dict = Depends(require_admin)):
    """Delete a user account (admin only). Admins cannot delete themselves."""
    if not AUTH_ENABLED:
        raise HTTPException(503, "Auth service unavailable")
    if user_id == admin["user_id"]:
        raise HTTPException(400, detail="You cannot delete your own account via the admin API")
    from file_processor.auth_manager import _PG_OK, _get_conn, _FALLBACK
    if _PG_OK:
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM auth_users WHERE user_id=%s RETURNING username",
                        (user_id,),
                    )
                    row = cur.fetchone()
                conn.commit()
            if not row:
                raise HTTPException(404, "User not found")
            return {"message": f"User '{row[0]}' deleted"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, detail=f"Delete failed: {e}")
    else:
        if user_id not in _FALLBACK:
            raise HTTPException(404, "User not found")
        del _FALLBACK[user_id]
        return {"message": "User deleted"}


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATION ROUTES
# All routes use JWT auth exclusively — user identity is NEVER accepted from
# query params or request body to prevent cross-user data access.
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/conversations")
async def list_conversations(current_user: dict = Depends(require_user)):
    """List the authenticated user's conversations (newest first)."""
    if not AUTH_ENABLED:
        return {"conversations": [], "total": 0}
    assert _conv_list_sessions is not None
    uid      = current_user["user_id"]
    sessions = _conv_list_sessions(uid)
    return {"conversations": sessions, "total": len(sessions)}


@app.get("/conversations/{session_id}/messages")
async def get_conversation_messages(
    session_id:   str,
    current_user: dict = Depends(require_user),
):
    """Load messages for a conversation the caller owns."""
    if not AUTH_ENABLED:
        return {"messages": []}
    assert _conv_load_messages is not None
    uid      = current_user["user_id"]
    messages = _conv_load_messages(session_id, uid)
    return {"messages": messages}


class SaveMessagesRequest(BaseModel):
    # user_id kept for backwards compat but is IGNORED — JWT always wins
    user_id:  str = ""
    messages: list[dict] = []
    label:    str = ""


@app.post("/conversations/{session_id}/messages")
async def save_conversation_messages(
    session_id:   str,
    req:          SaveMessagesRequest,
    current_user: dict = Depends(require_user),
):
    """Persist (full-overwrite) messages for a conversation the caller owns."""
    if not AUTH_ENABLED:
        return {"ok": True}
    assert _conv_update_label is not None and _conv_save_messages is not None
    uid = current_user["user_id"]
    # save_messages() itself validates ownership via INSERT…ON CONFLICT
    if req.label:
        _conv_update_label(session_id, uid, req.label)
    ok = _conv_save_messages(session_id, uid, req.messages)
    return {"ok": ok}


@app.put("/conversations/{session_id}/label")
async def update_conversation_label(
    session_id:   str,
    body:         dict,
    current_user: dict = Depends(require_user),
):
    label = body.get("label", "")
    if not label:
        raise HTTPException(400, "label required")
    if not AUTH_ENABLED:
        return {"ok": True}
    assert _conv_update_label is not None
    uid = current_user["user_id"]
    ok  = _conv_update_label(session_id, uid, label)
    return {"ok": ok}


@app.delete("/conversations/{session_id}")
async def delete_conversation(
    session_id:   str,
    current_user: dict = Depends(require_user),
):
    if not AUTH_ENABLED:
        return {"ok": True}
    assert _conv_delete_session is not None
    uid = current_user["user_id"]
    ok  = _conv_delete_session(session_id, uid)
    return {"ok": ok}


@app.post("/conversations/{session_id}/share")
async def share_conversation_endpoint(
    session_id:   str,
    current_user: dict = Depends(require_user),
):
    """Generate a permanent share token for a conversation the caller owns."""
    if not AUTH_ENABLED or not _conv_share:
        raise HTTPException(503, "Conversation store unavailable")
    uid   = current_user["user_id"]
    token = _conv_share(session_id, uid)
    if token is None:
        raise HTTPException(404, "Conversation not found or access denied")
    share_url = f"{FRONTEND_URL}/?share={token}"
    return {"share_token": token, "share_url": share_url}


@app.get("/shared/{share_token}")
async def get_shared_conversation_endpoint(share_token: str):
    """Return a shared conversation by its share token.

    No authentication required — any bearer of the token may read.
    Returns ``{label, session_id, messages}`` or 404.
    """
    if not AUTH_ENABLED or not _conv_get_shared:
        raise HTTPException(503, "Conversation store unavailable")
    data = _conv_get_shared(share_token)
    if data is None:
        raise HTTPException(404, "Shared conversation not found")
    return data


@app.delete("/sessions/{session_id}", dependencies=[Depends(require_api_key)])
async def delete_session(session_id: str, x_user_id: str = Header(default="")):
    """
    Clear the conversation buffer for a session.

    Pass the user's identifier in the X-User-Id header.
    Returns the number of turns deleted.
    """
    if not MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory layer not available.")
    assert _mem_clear_session is not None
    user_id = x_user_id or session_id
    try:
        deleted = _mem_clear_session(session_id, user_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    return JSONResponse(content={"session_id": session_id, "turns_deleted": deleted})


@app.get("/sessions/{session_id}", dependencies=[Depends(require_api_key)])
async def get_session(session_id: str, x_user_id: str = Header(default="")):
    """
    Inspect the conversation buffer for a session (debug / admin).

    Returns the trimmed turns that would currently be injected into the next
    request, plus diagnostic fields useful when tuning the memory layer:

    - total_tokens    : sum of token_count for all turns in the live buffer
    - budget_pct      : total_tokens as a percentage of TOKEN_BUDGET
    - should_summarise: True when un-absorbed tokens exceed SUMMARISE_THRESHOLD
    - has_summary     : True when a rolling summary exists for this session
    - summary_preview : first 200 chars of the rolling summary (or null)
    """
    if not MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory layer not available.")
    assert _mem_read_buffer is not None
    user_id      = x_user_id or session_id
    turns        = _mem_read_buffer(session_id, user_id)
    total_tokens = sum(t.token_count for t in turns)
    budget       = _MEM_TOKEN_BUDGET if _MEM_TOKEN_BUDGET else 1500
    summary      = _mem_get_summary(session_id, user_id) if _mem_get_summary else None

    return JSONResponse(content={
        "session_id":      session_id,
        "turn_count":      len(turns),
        "total_tokens":    total_tokens,
        "budget_pct":      round(total_tokens / budget * 100, 1),
        "should_summarise": bool(_mem_should_summarise and _mem_should_summarise(session_id)),
        "has_summary":     summary is not None,
        "summary_preview": summary[:200] if summary else None,
        "turns": [
            {"role": t.role, "content": t.content[:200], "token_count": t.token_count}
            for t in turns
        ],
    })


# ── Sprint 4: User fact endpoints ─────────────────────────────────────────────

class SetFactRequest(BaseModel):
    key:   str   # e.g. "preferred_language", "name"
    value: str


@app.get("/user-facts/{user_id}", dependencies=[Depends(require_api_key)])
async def get_user_facts(user_id: str):
    """
    Return all stored key-value facts for a user.

    Facts persist across sessions (stored in PostgreSQL or the in-process
    fallback dict).  They are automatically extracted from conversation turns
    and injected into the generation prompt on every /ask request.

    Response: { "user_id": str, "facts": { key: value, ... } }
    """
    if not MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory layer not available.")
    assert _mem_get_facts is not None
    facts = _mem_get_facts(user_id)
    return JSONResponse(content={"user_id": user_id, "facts": facts})


@app.post("/user-facts/{user_id}", dependencies=[Depends(require_api_key)])
async def set_user_fact(user_id: str, req: SetFactRequest):
    """
    Manually set or overwrite a single user fact.

    Useful for seeding known facts at session start (e.g. name, preferred
    language, project name) before the auto-extraction pipeline has run.

    Response: { "user_id": str, "key": str, "value": str }
    """
    if not MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory layer not available.")
    key   = req.key.strip().lower()
    value = req.value.strip()
    if not key or not value:
        raise HTTPException(status_code=422, detail="key and value must not be empty.")
    assert _mem_set_fact is not None
    _mem_set_fact(user_id, key, value)
    return JSONResponse(content={"user_id": user_id, "key": key, "value": value})


@app.delete("/user-facts/{user_id}/{key}", dependencies=[Depends(require_api_key)])
async def delete_user_fact(user_id: str, key: str):
    """
    Delete a single user fact by key.

    Response: { "user_id": str, "key": str, "deleted": bool }
    """
    if not MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory layer not available.")
    assert _mem_delete_fact is not None
    deleted = _mem_delete_fact(user_id, key.strip().lower())
    return JSONResponse(content={"user_id": user_id, "key": key, "deleted": deleted})


@app.delete("/user-facts/{user_id}", dependencies=[Depends(require_api_key)])
async def clear_all_user_facts(user_id: str):
    """
    Delete ALL facts for a user.

    Response: { "user_id": str, "deleted": int }
    """
    if not MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory layer not available.")
    assert _mem_clear_user_facts is not None
    count = _mem_clear_user_facts(user_id)
    return JSONResponse(content={"user_id": user_id, "deleted": count})


# ── Sprint 5: Semantic user memory endpoints ───────────────────────────────────

class StorePreferenceRequest(BaseModel):
    text: str   # Free-text preference, e.g. "I prefer concise answers with code examples."


@app.post("/user-memory/{user_id}", dependencies=[Depends(require_api_key)])
async def store_user_preference(user_id: str, req: StorePreferenceRequest):
    """
    Store a free-text preference statement for a user in semantic (Qdrant) memory.

    Preferences are retrieved by semantic similarity on each /ask request and
    injected into the generation prompt.  Storing the same statement twice is
    idempotent (deterministic UUID5 point ID).

    Requires Qdrant + BGE-M3 embedder.

    Response: { "user_id": str, "preference_id": str, "text": str }
    """
    if not MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory layer not available.")
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty.")
    if _mem_semantic_available and not _mem_semantic_available():
        raise HTTPException(
            status_code=503,
            detail="Semantic memory not available — check Qdrant and BGE-M3 embedder.",
        )
    pref_id = _mem_remember_pref(user_id, text) if _mem_remember_pref else ""
    return JSONResponse(content={"user_id": user_id, "preference_id": pref_id, "text": text})


@app.get("/user-memory/{user_id}", dependencies=[Depends(require_api_key)])
async def recall_user_preferences(
    user_id: str,
    query:   str = "",
    limit:   int = 5,
):
    """
    Recall the user's preferences most semantically relevant to ``query``.

    When ``query`` is omitted, lists all stored preferences (up to ``limit``).

    Response: { "user_id": str, "query": str, "preferences": [str, ...] }
    or, for list mode: { "user_id": str, "preferences": [{ preference_id, text, created_at }] }
    """
    if not MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory layer not available.")
    if query.strip():
        prefs = _mem_recall_prefs(user_id, query, limit=limit) if _mem_recall_prefs else []
        return JSONResponse(content={"user_id": user_id, "query": query, "preferences": prefs})
    else:
        prefs = _mem_list_prefs(user_id, limit=limit) if _mem_list_prefs else []
        return JSONResponse(content={"user_id": user_id, "preferences": prefs})


@app.delete("/user-memory/{user_id}/{preference_id}", dependencies=[Depends(require_api_key)])
async def delete_user_preference(user_id: str, preference_id: str):
    """
    Delete a single preference by its ID.

    Response: { "user_id": str, "preference_id": str, "deleted": bool }
    """
    if not MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory layer not available.")
    deleted = _mem_delete_pref(user_id, preference_id) if _mem_delete_pref else False
    return JSONResponse(content={"user_id": user_id, "preference_id": preference_id, "deleted": deleted})


@app.delete("/user-memory/{user_id}", dependencies=[Depends(require_api_key)])
async def delete_all_user_preferences(user_id: str):
    """
    Delete ALL preferences for a user from semantic memory.

    Response: { "user_id": str, "deleted": int }
    """
    if not MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory layer not available.")
    count = _mem_delete_all_prefs(user_id) if _mem_delete_all_prefs else 0
    return JSONResponse(content={"user_id": user_id, "deleted": count})


@app.post("/index", dependencies=[Depends(require_api_key)])
async def index_document(req: IndexRequest):
    """
    Embed and index a previously processed result JSON into Qdrant.

    The result_file must exist in the results/ directory (produced by POST /upload).
    CSV import results are skipped — they are queried via POST /query instead.

    Returns the number of chunks indexed and any skipped (e.g. uncaptioned images).
    """
    _require_embedding()

    result_path = RESULTS_DIR / req.result_file
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"Result file '{req.result_file}' not found.")

    try:
        doc = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read result file: {e}")

    if doc.get("type") == "csv_import":
        # CSV uploads are stored in the CSV query engine (queried via POST /query),
        # not embedded into Qdrant. This is expected, not an error — return 200 with
        # a "skipped" status so callers (React UI / curl / legacy UI) don't surface a
        # failure for a perfectly successful CSV import.
        return JSONResponse(content={
            "result_file":  req.result_file,
            "source":       doc.get("source_file", req.result_file),
            "collection":   req.collection,
            "indexed":      0,
            "skipped":      "csv_import",
            "total_chunks": 0,
            "table":        doc.get("table"),
            "detail":       "CSV import — queried via POST /query, not vector-indexed.",
        })

    chunks = doc.get("chunks", [])
    if not chunks:
        raise HTTPException(status_code=422, detail="Result file contains no chunks to index.")

    source = doc.get("source_file", req.result_file)


    try:
        assert _qdrant_client is not None
        _emb_ensure_collection(_qdrant_client, req.collection)
        stats = _emb_index_chunks(
            chunks,
            _qdrant_client,
            collection=req.collection,
            source=source,       # enables stale-chunk cleanup before upsert
            clean_first=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    return JSONResponse(content={
        "result_file":  req.result_file,
        "source":       source,
        "collection":   req.collection,
        "indexed":      stats["indexed"],
        "skipped":      stats["skipped"],
        "total_chunks": len(chunks),
        "by_type":      stats.get("by_type", {}),
    })


@app.post("/search", dependencies=[Depends(require_api_key)])
async def search_documents(req: SearchRequest):
    """
    Hybrid semantic search across indexed documents.

    Combines dense (BGE-M3 cosine) + sparse (SPLADE) retrieval fused via RRF.

    Optional filters narrow results by any metadata field, e.g.:
        {"language": "en"}
        {"source": "report.pdf", "type": "text"}

    Returns a ranked list of chunks with score and full metadata.
    """
    _require_embedding()
    assert _qdrant_client is not None

    if not req.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    sf = _build_source_filter(req.source_filter)
    merged_filters = sf.to_filters(req.filters) if sf else req.filters

    try:
        results = _retr_retrieve(
            req.query,
            _qdrant_client,
            collection = req.collection,
            limit      = req.limit,
            filters    = merged_filters,
            min_score  = req.min_score,
            rerank     = req.rerank,
            use_hyde   = req.use_hyde,
            mmr        = req.mmr,
            mmr_lambda = req.mmr_lambda,
            decompose  = req.decompose,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    return JSONResponse(content={
        "query":      req.query,
        "collection": req.collection,
        "total":      len(results),
        "results": [
            {
                "chunk_id": r["chunk_id"],
                "score":    round(r["score"], 4),
                "content":  r["payload"].get("content", ""),
                "metadata": {
                    k: v for k, v in r["payload"].items()
                    if k not in ("content", "chunk_id")
                },
            }
            for r in results
        ],
    })


_ALLOWED_LANGUAGE_HINTS: frozenset[str] = frozenset({"", "en", "fr", "ar"})


class AskRequest(BaseModel):
    question:        str
    collection:      str                    = _EMB_DEFAULT_COLLECTION if EMBEDDING_ENABLED else "documents"
    limit:           int                    = Field(5,    ge=1, le=20)   # number of chunks to retrieve
    context_window:  int                    = Field(0,    ge=0, le=5)   # neighbor chunks to expand each result
    filters:         dict | None            = None   # same filter syntax as /search
    source_filter:   SourceFilterModel | None = None # typed source / language / type filter
    min_score:       float | None           = Field(None, ge=0.0, le=1.0)
    rerank:          bool                   = False
    use_hyde:        bool                   = False
    mmr:             bool                   = False
    mmr_lambda:      float                  = Field(0.5, ge=0.0, le=1.0)
    decompose:       bool                   = False
    multi_hop:       bool                   = False  # two-hop retrieval (entity follow-up via Groq)
    max_tokens:      int                    = Field(1024, ge=64, le=4096)  # max answer tokens for the LLM
    language_hint:   str                    = ""     # ISO 639-1 code — "fr", "ar", etc. (reply in that language)
    judge:           bool                   = False  # run LLM-as-a-Judge after generation
    reference:       str | None             = None   # ground-truth answer for correctness scoring
    score_chunks:    bool                   = False  # score each retrieved chunk individually
    # ── Memory layer ──────────────────────────────────────────────────────────
    session_id:      str | None             = None   # caller-managed conversation ID (UUID)
    memory_enabled:  bool                   = False  # enable conversation buffer + query rewriting
    # ── Customer support features ─────────────────────────────────────────────
    support_mode:    bool                   = False  # enable intent classification + escalation + persona
    persona:         str                    = ""     # custom persona; overrides support_mode default when set
    # ── Evaluation helpers ────────────────────────────────────────────────────
    include_chunks:  bool                   = False  # include chunks_in_context in response (for LangSmith eval)

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty.")
        if len(v) > 4000:
            raise ValueError("question must not exceed 4 000 characters.")
        return v

    @field_validator("language_hint")
    @classmethod
    def language_hint_allowed(cls, v: str) -> str:
        if v not in _ALLOWED_LANGUAGE_HINTS:
            raise ValueError(
                f"language_hint must be one of {sorted(_ALLOWED_LANGUAGE_HINTS)!r}."
            )
        return v


def _build_rag_state(req: "AskRequest", current_user: Optional[dict]) -> dict:
    """
    Build the initial RAGState dict to pass to the LangGraph pipeline.

    Maps every field from AskRequest + the resolved caller identity into the
    typed state schema.  Runtime singletons (groq_client, qdrant_client) are
    injected here so graph nodes never need to import from api.py directly.
    """
    uid      = current_user["user_id"] if current_user else None
    is_admin = (current_user or {}).get("role") == "admin"

    # Server-side RBAC filter — merged with any caller-supplied filters
    rbac_filters = dict(req.filters or {})
    if uid and not is_admin:
        rbac_filters["owner_id"] = [uid, "__global__"]

    sf = _build_source_filter(req.source_filter) if EMBEDDING_ENABLED else None

    return {
        # ── Request inputs ────────────────────────────────────────────────
        "question":       req.question,
        "session_id":     req.session_id,
        "user_id":        uid,
        "is_admin":       is_admin,
        "collection":     req.collection,
        # Retrieval options
        "limit":          req.limit,
        "context_window": req.context_window,
        "rerank":         req.rerank,
        "use_hyde":       req.use_hyde,
        "mmr":            req.mmr,
        "mmr_lambda":     req.mmr_lambda,
        "decompose":      req.decompose,
        "multi_hop":      req.multi_hop,
        "min_score":      req.min_score,
        "filters":        rbac_filters or None,
        "source_filter":  sf,
        # Generation options
        "max_tokens":     req.max_tokens,
        "language_hint":  req.language_hint,
        # Feature flags
        "support_mode":   req.support_mode,
        "persona":        req.persona,
        "memory_enabled": req.memory_enabled,
        "judge":          req.judge,
        "reference":      req.reference,
        "score_chunks":   req.score_chunks,
        # ── Runtime singletons ────────────────────────────────────────────
        "groq_client":    _groq_client,
        "qdrant_client":  _qdrant_client,
        # ── Timing ───────────────────────────────────────────────────────
        "start_time":     __import__("time").monotonic(),
    }


@app.post("/ask", dependencies=[Depends(require_api_key), Depends(ask_concurrency_limit)])
async def ask(
    req:          AskRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[dict] = Depends(optional_user),
):
    """
    RAG question-answering endpoint.

    Retrieves the most relevant chunks for the question, optionally expands
    context by fetching neighbor chunks, then calls Qwen2.5-7B (via Ollama) to
    generate a grounded answer with source citations.

    Pipeline:
        0. Memory (optional) — load conversation buffer, rewrite query
        1. retrieve()  — hybrid RRF search (+ HyDE / reranking if enabled)
        2. get_neighbors() — context expansion around each hit
        3. Qwen2.5-7B (Ollama) — answer grounded in retrieved context
        4. Judge (optional) — LLM-as-a-Judge quality scoring
        5. Memory write-back (async) — update buffer after response

    Returns:
        {
            "question":  str,
            "answer":    str,
            "sources":   [{ "source", "page_start", "section", "score", "chunk_id" }],
            "chunks_used": int,
        }

    Orchestration is delegated to the LangGraph RAG pipeline (rag_graph) when
    available.  Falls back to the legacy inline pipeline if langgraph is not
    installed.
    """
    _require_embedding()

    if not GENERATION_ENABLED or _answer_gen is None:
        raise HTTPException(
            status_code=503,
            detail="Generation backend unavailable. Start Ollama and pull the model: `ollama serve && ollama pull qwen2.5:7b`",
        )

    # ── LangGraph path ────────────────────────────────────────────────────────
    if RAG_GRAPH_ENABLED and _rag_graph is not None:
        state = _build_rag_state(req, current_user)
        try:
            final: dict = await _rag_graph.ainvoke(state)
        except Exception as _rg_exc:
            _rg_err_s = str(_rg_exc)
            if "429" in _rg_err_s or "rate_limit_exceeded" in _rg_err_s or "tokens per day" in _rg_err_s.lower():
                raise HTTPException(status_code=429, detail="Daily AI usage limit reached. Please try again in a few minutes.")
            raise HTTPException(status_code=500, detail=f"RAG pipeline failed: {_rg_exc}")

        # Extract escalation result if present
        _esc_result = final.get("escalation_result")
        _esc_dict   = _esc_result.to_dict() if _esc_result and hasattr(_esc_result, "to_dict") else None
        _intent_r   = final.get("intent_result")
        _intent_dict = _intent_r.to_dict() if _intent_r and hasattr(_intent_r, "to_dict") else None
        _judge_r    = final.get("judge_result")
        _judge_dict = _judge_r.to_dict() if _judge_r and hasattr(_judge_r, "to_dict") else None

        _mem_ctx = final.get("memory_context")
        return JSONResponse(content={
            "question":            req.question,
            "rewritten_question":  (
                final.get("retrieval_question")
                if final.get("retrieval_question") != req.question
                else None
            ),
            "rewrite_tier":        _mem_ctx.rewrite_tier if _mem_ctx else None,
            "answer":              final.get("answer", ""),
            "sources":             final.get("sources", []),
            "chunks_used":         len([c for c in final.get("gen_chunks", []) if c.get("primary")]),
            "confidence":          final.get("confidence_score"),
            "hops":                (final.get("retrieval_result").hops
                                    if final.get("retrieval_result") else 1),
            "backend":             final.get("backend", ""),
            "model":               final.get("model", ""),
            "elapsed_ms":          final.get("elapsed_ms"),
            "retrieval_ms":        final.get("retrieval_ms"),
            "generation_ms":       final.get("generation_ms"),
            "citation_count":      final.get("citation_count", 0),
            "context_utilisation": (
                final.get("tokens_in_context", 0) / 8000
                if final.get("tokens_in_context") else None
            ),
            "no_answer":           final.get("no_answer", False),
            "token_counts":        final.get("token_counts"),
            "judge":               _judge_dict,
            "eval_verdict":        final.get("eval_verdict"),
            "eval_feedback":       final.get("eval_feedback"),
            "intent":              _intent_dict,
            "escalated":           final.get("escalated", False),
            "escalation":          _esc_dict,
            "chunks_in_context":   (
                [
                    {
                        "chunk_id":   c.get("chunk_id", ""),
                        "content":    c.get("content", ""),
                        "source":     c.get("source", c.get("metadata", {}).get("source", "")),
                        "page_start": c.get("page_start", c.get("metadata", {}).get("page_start")),
                        "section":    c.get("section", c.get("metadata", {}).get("section", "")),
                        "score":      c.get("score", 0.0),
                    }
                    for c in (final.get("ordered_chunks") or [])
                ]
                if req.include_chunks else None
            ),
        })

    # ── Legacy inline pipeline (fallback when langgraph is not installed) ─────
    _ask_uid      = current_user["user_id"] if current_user else None
    _ask_is_admin = (current_user or {}).get("role") == "admin"
    _ask_rbac_filters = dict(req.filters or {})
    if _ask_uid and not _ask_is_admin:
        _ask_rbac_filters["owner_id"] = [_ask_uid, "__global__"]

    _intent_result = None
    if req.support_mode and INTENT_ENABLED and _classify_intent is not None:
        try:
            _intent_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _classify_intent(req.question, groq_client=_groq_client),
            )
        except Exception as _int_exc:
            logger.warning(f"[INTENT] /ask classification failed: {_int_exc}")

    if _intent_result is not None and _should_escalate is not None:
        _pre_esc = _should_escalate(intent=_intent_result)
        if _pre_esc.should_escalate:
            return JSONResponse(content={
                "question":   req.question,
                "answer":     _pre_esc.message,
                "sources":    [],
                "chunks_used": 0,
                "confidence": None,
                "hops":       0,
                "escalated":  True,
                "escalation": _pre_esc.to_dict(),
                "intent":     _intent_result.to_dict(),
            })

    mem_ctx = None
    retrieval_question = req.question
    if req.memory_enabled and MEMORY_ENABLED and req.session_id:
        assert _mem_load_context is not None
        try:
            mem_ctx = await _mem_load_context(
                session_id=req.session_id,
                user_id=_ask_uid or req.session_id,
                raw_query=req.question,
                groq_client=_groq_client,
                memory_enabled=True,
            )
            retrieval_question = mem_ctx.rewritten_query
        except Exception as _mem_exc:
            logger.warning(f"[MEMORY] load_memory_context failed: {_mem_exc}")

    assert _qdrant_client is not None
    sf = _build_source_filter(req.source_filter)
    _retr_kw = dict(
        collection=req.collection, limit=req.limit, context_window=req.context_window,
        filters=_ask_rbac_filters if _ask_rbac_filters else req.filters,
        source_filter=sf, min_score=req.min_score, rerank=req.rerank,
        use_hyde=req.use_hyde, mmr=req.mmr, mmr_lambda=req.mmr_lambda, decompose=req.decompose,
    )
    try:
        retrieval = (_retr_multihop(retrieval_question, _qdrant_client, **_retr_kw)
                     if req.multi_hop
                     else _retr_retrieve_evidence(retrieval_question, _qdrant_client, **_retr_kw))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    if not retrieval.chunks:
        return JSONResponse(content={
            "question": req.question, "answer": "I could not find any relevant information in the indexed documents.",
            "sources": [], "chunks_used": 0, "confidence": None, "hops": 1,
        })

    seen: set[str] = set()
    gen_chunks: list[dict] = []
    for chunk in retrieval.chunks:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            gen_chunks.append({"content": chunk.content, "chunk_id": chunk.chunk_id,
                                "score": chunk.score, "hop": chunk.hop, "primary": True, "metadata": chunk.metadata})
        for nb in chunk.neighbors:
            nb_id = nb.get("chunk_id", "")
            if nb_id and nb_id not in seen:
                seen.add(nb_id)
                gen_chunks.append({"content": nb.get("content", ""), "chunk_id": nb_id,
                                    "score": 0.0, "hop": 1, "primary": False, "metadata": nb})

    _persona = req.persona or (_CS_PERSONA if req.support_mode else "")
    cfg = _GenerationConfig(
        temperature=0.2, max_tokens=req.max_tokens, language_hint=req.language_hint,
        conversation_summary=mem_ctx.summary_as_context() if mem_ctx else "",
        user_facts=mem_ctx.user_facts_as_context() if mem_ctx else "",
        user_preferences=mem_ctx.recalled_preferences_as_context() if mem_ctx else "",
        persona=_persona,
    )
    _gen_history = mem_ctx.prompt_messages() if (mem_ctx and mem_ctx.has_history) else None
    try:
        result = _answer_gen.generate(retrieval_question, gen_chunks, cfg, history=_gen_history)
    except Exception as e:
        _err_s = str(e)
        if "429" in _err_s or "rate_limit_exceeded" in _err_s or "tokens per day" in _err_s.lower():
            raise HTTPException(status_code=429, detail="Daily AI usage limit reached. Please try again in a few minutes.")
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {e}")
    result.retrieval_ms = retrieval.elapsed_ms

    confidence: float | None = None
    try:
        ctx_chunks = [_RC(chunk_id=c.get("chunk_id",""), content=c.get("content",""),
                          score=c.get("score",0.0), metadata={})
                      for c in result.chunks_in_context] if result.chunks_in_context else retrieval.chunks
        confidence = _retr_score_confidence(result.answer, ctx_chunks, groq_client=_groq_client)
    except Exception:
        pass

    judge_data: dict | None = None
    if req.judge and JUDGE_ENABLED and _judge_answer is not None:
        try:
            jr = _judge_answer(question=req.question, answer=result.answer,
                               chunks=result.chunks_in_context, no_answer=result.no_answer,
                               reference=req.reference or None, score_chunks=req.score_chunks,
                               retrieval_ms=result.retrieval_ms, generation_ms=result.generation_ms,
                               token_counts=result.token_counts)
            judge_data = jr.to_dict()
        except Exception as exc:
            logger.warning(f"[JUDGE] /ask evaluation failed: {exc}")

    if req.memory_enabled and MEMORY_ENABLED and req.session_id:
        _ask_sid = req.session_id
        _ask_mem_uid = _ask_uid or _ask_sid

        def _write_turns() -> None:
            assert _mem_write_turn is not None
            try:
                _mem_write_turn(_ask_sid, _ask_mem_uid, "user", req.question)
                _mem_write_turn(_ask_sid, _ask_mem_uid, "assistant", result.answer)
            except Exception as _wt_exc:
                logger.warning(f"[MEMORY] write_turn failed: {_wt_exc}")
        background_tasks.add_task(_write_turns)

        def _maybe_summarise() -> None:
            try:
                if _mem_should_summarise and _mem_should_summarise(_ask_sid):
                    assert _mem_summarise is not None
                    _mem_summarise(_ask_sid, _ask_mem_uid, _groq_client)
            except Exception as _s_exc:
                logger.warning(f"[MEMORY] summarise_session failed: {_s_exc}")
        background_tasks.add_task(_maybe_summarise)

        def _extract_facts() -> None:
            try:
                if _mem_extract_facts and _mem_read_all_turns and _groq_client:
                    all_turns = _mem_read_all_turns(req.session_id, _ask_mem_uid)
                    _mem_extract_facts(_ask_mem_uid, all_turns, _groq_client)
            except Exception as _ef_exc:
                logger.warning(f"[MEMORY] extract_and_store_facts failed: {_ef_exc}")
        background_tasks.add_task(_extract_facts)

        def _extract_preferences() -> None:
            try:
                if _mem_extract_prefs and _mem_read_all_turns and _groq_client:
                    all_turns = _mem_read_all_turns(req.session_id, _ask_mem_uid)
                    _mem_extract_prefs(_ask_mem_uid, all_turns, _groq_client)
            except Exception as _ep_exc:
                logger.warning(f"[MEMORY] extract_and_store_preferences failed: {_ep_exc}")
        background_tasks.add_task(_extract_preferences)

    _ask_eval_verdict: str | None = None
    if judge_data:
        _overall = judge_data.get("overall")
        if _overall is not None:
            _ask_eval_verdict = "fail" if _overall < 0.60 else "pass"

    _post_escalation: dict | None = None
    if req.support_mode and ESCALATION_ENABLED and _should_escalate is not None:
        try:
            _post_esc = _should_escalate(intent=_intent_result, no_answer=result.no_answer,
                                         eval_verdict=_ask_eval_verdict)
            if _post_esc.should_escalate:
                _post_escalation = _post_esc.to_dict()
        except Exception as _esc_exc:
            logger.warning(f"[ESCALATION] /ask post-generation check failed: {_esc_exc}")

    # Serialize chunks_in_context for eval (include_chunks=True only — keeps normal responses lean)
    _chunks_out: list | None = None
    if req.include_chunks and result.chunks_in_context:
        _chunks_out = [
            {
                "chunk_id":  c.get("chunk_id", ""),
                "content":   c.get("content", ""),
                "source":    c.get("source", c.get("metadata", {}).get("source", "")),
                "page_start": c.get("page_start", c.get("metadata", {}).get("page_start")),
                "section":   c.get("section", c.get("metadata", {}).get("section", "")),
                "score":     c.get("score", 0.0),
            }
            for c in result.chunks_in_context
        ]

    return JSONResponse(content={
        "question":            req.question,
        "rewritten_question":  retrieval_question if retrieval_question != req.question else None,
        "rewrite_tier":        mem_ctx.rewrite_tier if mem_ctx else None,
        "answer":              result.answer,
        "sources":             result.sources,
        "chunks_used":         result.chunks_used,
        "confidence":          confidence,
        "hops":                retrieval.hops,
        "backend":             result.backend,
        "model":               result.model,
        "elapsed_ms":          result.elapsed_ms,
        "retrieval_ms":        result.retrieval_ms,
        "generation_ms":       result.generation_ms,
        "citation_count":      result.citation_count,
        "context_utilisation": result.context_utilisation,
        "no_answer":           result.no_answer,
        "judge":               judge_data,
        "intent":              _intent_result.to_dict() if _intent_result else None,
        "escalated":           _post_escalation is not None,
        "escalation":          _post_escalation,
        "chunks_in_context":   _chunks_out,   # None unless include_chunks=True
    })


@app.post("/ask/stream", dependencies=[Depends(require_api_key), Depends(ask_concurrency_limit)])
async def ask_stream(
    req:          AskRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[dict] = Depends(optional_user),
):
    """
    Streaming RAG answer endpoint (Server-Sent Events).

    Performs the same retrieval pipeline as POST /ask but streams the LLM
    answer token-by-token.  The SSE stream contains three event types:

        {"type": "sources",  "sources": [...], "chunks_used": N, "hops": N}
            — emitted first, before any answer tokens, so the client can
              display citations while the answer is still being written.

        {"type": "token",    "content": "..."}
            — one event per answer token from Ollama's streaming API.

        {"type": "done",     "confidence": 0.85}
            — final event; confidence is null when Groq scoring is unavailable.

    Clients should subscribe with:
        const es = new EventSource("/ask/stream", {method: "POST", ...});
        es.onmessage = e => { const d = JSON.parse(e.data); ... };

    Returns 503 if Ollama or Qdrant is not available.

    Orchestration is delegated to the LangGraph RAG pipeline (rag_graph) when
    available.  Falls back to the legacy inline pipeline if langgraph is not
    installed.
    """
    _require_embedding()

    if not GENERATION_ENABLED or _answer_gen is None:
        raise HTTPException(
            status_code=503,
            detail="Generation backend unavailable. Start Ollama and pull the model: `ollama serve && ollama pull qwen2.5:7b`",
        )

    # ── LangGraph streaming path ──────────────────────────────────────────────
    if RAG_GRAPH_ENABLED and _rag_graph is not None:
        state = _build_rag_state(req, current_user)

        async def _graph_sse():
            async for event in _rag_graph.astream(state, stream_mode="custom"):
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(
            _graph_sse(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Legacy inline streaming pipeline (fallback) ───────────────────────────
    # Intent classification + pre-retrieval escalation (support_mode)
    _stream_intent_result = None
    if req.support_mode and INTENT_ENABLED and _classify_intent is not None:
        try:
            _stream_intent_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _classify_intent(req.question, groq_client=_groq_client),
            )
            logger.info(
                f"[INTENT] /ask/stream: intent={_stream_intent_result.intent} "
                f"strategy={_stream_intent_result.strategy}"
            )
        except Exception as _si_exc:
            logger.warning(f"[INTENT] /ask/stream classification failed: {_si_exc}")

    # Pre-retrieval escalation: explicit request or complaint → stream escalation event + done
    if _stream_intent_result is not None and _should_escalate is not None:
        _pre_esc_stream = _should_escalate(intent=_stream_intent_result)
        if _pre_esc_stream.should_escalate:
            async def _escalation_only_stream():
                yield f"data: {json.dumps({'type': 'sources', 'sources': [], 'chunks_used': 0, 'hops': 0})}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'content': _pre_esc_stream.message})}\n\n"
                yield f"data: {json.dumps({'type': 'escalation', **_pre_esc_stream.to_dict()})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'confidence': None, 'no_answer': False, 'escalated': True, 'intent': _stream_intent_result.to_dict()})}\n\n"
            return StreamingResponse(
                _escalation_only_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

    # ── Resolve caller identity + RBAC filter ────────────────────────────────
    # current_user comes from JWT; fall back to session_id as anonymous scope.
    _stream_uid      = current_user["user_id"] if current_user else None
    _stream_is_admin = (current_user or {}).get("role") == "admin"

    # Build RBAC-aware retrieval filter:
    #   admin   → no owner restriction (sees all documents)
    #   user    → owner_id must be [user_id, '__global__']  (own docs + shared KB)
    # If the request already carries an explicit filter dict, merge into it;
    # admin-provided explicit filters are respected as-is.
    _stream_rbac_filters = dict(req.filters or {})
    if _stream_uid and not _stream_is_admin:
        _stream_rbac_filters["owner_id"] = [_stream_uid, "__global__"]

    # ── Memory context ────────────────────────────────────────────────────────
    _stream_mem_ctx = None
    _stream_retrieval_question = req.question

    if req.memory_enabled and MEMORY_ENABLED and req.session_id:
        assert _mem_load_context is not None
        try:
            _stream_mem_ctx = await _mem_load_context(
                session_id     = req.session_id,
                # FIX: use actual user_id from JWT, not session_id as user scope
                user_id        = _stream_uid or req.session_id,
                raw_query      = req.question,
                groq_client    = _groq_client,
                memory_enabled = True,
            )
            _stream_retrieval_question = _stream_mem_ctx.rewritten_query
        except Exception as _sm_exc:
            logger.warning(f"[MEMORY] /ask/stream load_memory_context failed: {_sm_exc}")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    assert _qdrant_client is not None
    sf = _build_source_filter(req.source_filter)
    _retriever_kwargs = dict(
        collection     = req.collection,
        limit          = req.limit,
        context_window = req.context_window,
        filters        = _stream_rbac_filters if _stream_rbac_filters else req.filters,
        source_filter  = sf,
        min_score      = req.min_score,
        rerank         = req.rerank,
        use_hyde       = req.use_hyde,
        mmr            = req.mmr,
        mmr_lambda     = req.mmr_lambda,
        decompose      = req.decompose,
    )

    try:
        retrieval: _RetrievalResult = (
            _retr_multihop(_stream_retrieval_question, _qdrant_client, **_retriever_kwargs)
            if req.multi_hop
            else _retr_retrieve_evidence(_stream_retrieval_question, _qdrant_client, **_retriever_kwargs)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    # ── Flatten chunks + neighbors for AnswerGenerator ────────────────────────
    seen_chunk_ids: set[str] = set()
    gen_chunks: list[dict]   = []

    for chunk in retrieval.chunks:
        if chunk.chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk.chunk_id)
            gen_chunks.append({
                "content":  chunk.content,
                "chunk_id": chunk.chunk_id,
                "score":    chunk.score,
                "hop":      chunk.hop,
                "primary":  True,
                "metadata": chunk.metadata,
            })
        for nb in chunk.neighbors:
            nb_id = nb.get("chunk_id", "")
            if nb_id and nb_id not in seen_chunk_ids:
                seen_chunk_ids.add(nb_id)
                gen_chunks.append({
                    "content":  nb.get("content", ""),
                    "chunk_id": nb_id,
                    "score":    0.0,
                    "hop":      1,
                    "primary":  False,
                    "metadata": nb,
                })

    if not retrieval.chunks:
        # No evidence — stream a single done event with an empty answer
        def _empty_stream():
            yield f"data: {json.dumps({'type': 'sources', 'sources': [], 'chunks_used': 0, 'hops': 1})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'content': 'I could not find any relevant information in the indexed documents.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'confidence': None})}\n\n"

        return StreamingResponse(
            _empty_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Shared capture dict — the generator writes the completed answer here so the
    # BackgroundTask can read it after streaming finishes.  Using a dict (not a
    # bare variable) so the closure captures by reference rather than by value.
    _wb: dict = {"answer": None}   # None = generator did not complete (client disconnected)

    # Capture variables needed inside the generator closure
    _retrieval  = retrieval
    _gen        = _answer_gen
    _groq       = _groq_client
    _score_fn   = _retr_score_confidence
    _stream_persona = req.persona or (_CS_PERSONA if req.support_mode else "")
    _cfg        = _GenerationConfig(
        temperature=0.2,
        max_tokens=req.max_tokens,
        language_hint=req.language_hint,
        conversation_summary=_stream_mem_ctx.summary_as_context() if _stream_mem_ctx else "",
        user_facts=_stream_mem_ctx.user_facts_as_context() if _stream_mem_ctx else "",
        user_preferences=_stream_mem_ctx.recalled_preferences_as_context() if _stream_mem_ctx else "",
        persona=_stream_persona,
    )
    _stream_history = (
        _stream_mem_ctx.prompt_messages()
        if (_stream_mem_ctx and _stream_mem_ctx.has_history)
        else None
    )
    # Memory metadata forwarded to the client via the 'done' SSE event
    _rewrite_tier       = _stream_mem_ctx.rewrite_tier if _stream_mem_ctx else None
    _rewritten_question = (
        _stream_retrieval_question
        if _stream_retrieval_question != req.question
        else None
    )

    async def _sse_generator():
        # Build context once — reused for both the sources event and token streaming.
        # build_context_metadata returns (sources, tokens, context_text, ordered_chunks).
        # ordered_chunks is the token-budget-trimmed set the LLM will actually see;
        # use it for confidence scoring instead of the full retrieved set.
        sources_list, _, prebuilt_ctx, ctx_ordered_chunks = _gen.build_context_metadata(gen_chunks, _cfg)
        primary_count = sum(1 for c in gen_chunks if c["primary"])

        # Event 1 — sources emitted before any answer tokens
        yield (
            f"data: {json.dumps({'type': 'sources', 'sources': sources_list, 'chunks_used': primary_count, 'hops': _retrieval.hops})}\n\n"
        )

        # Events 2..N — answer tokens (timed for generation_ms)
        import time as _time
        answer_parts:  list[str] = []
        _t_gen = _time.monotonic()
        try:
            for token in _gen.stream(req.question, gen_chunks, _cfg, prebuilt_context=prebuilt_ctx, history=_stream_history):
                if token:
                    answer_parts.append(token)
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        except Exception as e:
            _err_str = str(e)
            if "429" in _err_str or "rate_limit_exceeded" in _err_str or "tokens per day" in _err_str.lower() or "tokens per minute" in _err_str.lower():
                err_token = "⚠️ Daily AI usage limit reached. Please try again in a few minutes, or tomorrow when the quota resets."
            elif "401" in _err_str or "invalid_api_key" in _err_str.lower():
                err_token = "⚠️ AI service authentication error. Please contact the administrator."
            elif "503" in _err_str or "unavailable" in _err_str.lower():
                err_token = "⚠️ AI service is temporarily unavailable. Please try again in a moment."
            else:
                err_token = f"[Generation error: {e}]"
            answer_parts.append(err_token)
            yield f"data: {json.dumps({'type': 'token', 'content': err_token})}\n\n"
        generation_ms = round((_time.monotonic() - _t_gen) * 1000, 1)
        # Capture token counts from the backend after the stream is fully consumed.
        stream_token_counts: dict = getattr(getattr(_gen, "_ollama", None), "last_token_counts", {}) or {}

        full_answer    = "".join(answer_parts)
        citation_count = len({m[0] for m in _CITATION_RE_SSE.findall(full_answer)}) if _CITATION_RE_SSE else 0
        _sse_no_answer = not full_answer.strip()

        confidence: float | None = None
        try:
            # Use ctx_ordered_chunks (token-budget-trimmed) so the scorer sees
            # exactly the same evidence the LLM used — not the wider retrieved set.
            _ctx_chunks_sse = [_RC(
                chunk_id = c.get("chunk_id", ""),
                content  = c.get("content", ""),
                score    = c.get("score", 0.0),
                metadata = {},
            ) for c in ctx_ordered_chunks] if ctx_ordered_chunks else _retrieval.chunks
            confidence = _score_fn(full_answer, _ctx_chunks_sse, groq_client=_groq)
        except Exception:
            pass

        # LLM-as-a-Judge (optional, best-effort)
        judge_data: dict | None = None
        if req.judge and JUDGE_ENABLED:
            assert _judge_answer is not None
            try:
                jr = _judge_answer(
                    question      = req.question,
                    answer        = full_answer,
                    chunks        = ctx_ordered_chunks or [],
                    no_answer     = _sse_no_answer,
                    reference     = req.reference or None,
                    score_chunks  = req.score_chunks,
                    retrieval_ms  = _retrieval.elapsed_ms,
                    generation_ms = generation_ms,
                    token_counts  = stream_token_counts or None,
                )
                judge_data = jr.to_dict()
            except Exception as _je:
                logger.warning(f"[JUDGE] /ask/stream evaluation failed: {_je}")

        # ── Streaming comparison graph (eval events) ──────────────────────────
        # Emitted between the last token and the done event.
        # Captures verdict + feedback for inclusion in the done payload.
        _eval_verdict:  str | None = None
        _eval_feedback: str | None = None
        if _stream_comparison is not None and full_answer:
            try:
                async for eval_event in _stream_comparison(
                    user_query       = req.question,
                    generated_answer = full_answer,
                    retrieved_chunks = ctx_ordered_chunks or [],
                ):
                    # Parse eval_done payload to extract verdict/feedback before forwarding.
                    try:
                        _ed = json.loads(eval_event.removeprefix("data: ").rstrip("\n"))
                        if _ed.get("type") == "eval_done":
                            _eval_verdict  = _ed.get("verdict")
                            _eval_feedback = _ed.get("feedback")
                    except Exception:
                        pass
                    yield eval_event
            except Exception as _eval_err:
                logger.warning(f"[EVAL] stream_comparison failed: {_eval_err}")
                yield f"data: {json.dumps({'type': 'eval_error', 'message': str(_eval_err)})}\n\n"

        # ── Post-generation escalation (support_mode) ─────────────────────────
        _sse_escalated   = False
        _sse_escalation: dict | None = None
        if req.support_mode and ESCALATION_ENABLED and _should_escalate is not None:
            try:
                _sse_esc = _should_escalate(
                    intent       = _stream_intent_result,
                    no_answer    = _sse_no_answer,
                    eval_verdict = _eval_verdict,
                )
                if _sse_esc.should_escalate:
                    _sse_escalated  = True
                    _sse_escalation = _sse_esc.to_dict()
                    yield f"data: {json.dumps({'type': 'escalation', **_sse_esc.to_dict()})}\n\n"
            except Exception as _sse_esc_err:
                logger.warning(f"[ESCALATION] /ask/stream post-generation check failed: {_sse_esc_err}")

        # ── Final done event ──────────────────────────────────────────────────
        yield f"data: {json.dumps({'type': 'done', 'confidence': confidence, 'retrieval_ms': _retrieval.elapsed_ms, 'generation_ms': generation_ms, 'citation_count': citation_count, 'no_answer': _sse_no_answer, 'token_counts': stream_token_counts or None, 'judge': judge_data, 'rewritten_question': _rewritten_question, 'rewrite_tier': _rewrite_tier, 'eval_verdict': _eval_verdict, 'eval_feedback': _eval_feedback, 'escalated': _sse_escalated, 'escalation': _sse_escalation, 'intent': _stream_intent_result.to_dict() if _stream_intent_result else None})}\n\n"

        # Signal successful completion to the write-back BackgroundTask.
        # This runs after the done event is yielded so the client always
        # receives the full response even if write-back somehow stalls.
        if full_answer:
            _wb["answer"] = full_answer

    # ── Memory write-back as a BackgroundTask (guaranteed to run) ─────────────
    # BackgroundTasks registered here fire after the StreamingResponse body
    # is fully consumed — whether the generator completed normally or was
    # stopped early because the client disconnected.  Reading _wb["answer"]
    # is safe: the generator sets it before exhausting, and the BackgroundTask
    # reads it only after the generator is done.
    if req.memory_enabled and MEMORY_ENABLED and req.session_id:
        # Use JWT user_id for memory scoping — fall back to session_id only when
        # the request is unauthenticated (guest / anonymous usage).
        _stream_sid = req.session_id
        _stream_mem_uid = _stream_uid or _stream_sid

        def _stream_write_turns() -> None:
            answer = _wb.get("answer")
            if not answer:
                # Generator didn't complete (client disconnected before done event).
                # Don't write a partial answer into the conversation buffer.
                logger.debug(
                    f"[MEMORY] /ask/stream: generator incomplete for session "
                    f"{_stream_sid[:8]}… — skipping write-back."
                )
                return
            assert _mem_write_turn is not None
            try:
                _mem_write_turn(_stream_sid, _stream_mem_uid, "user",      req.question)
                _mem_write_turn(_stream_sid, _stream_mem_uid, "assistant", answer)
            except Exception as _wt_exc:
                logger.warning(f"[MEMORY] /ask/stream write_turn failed: {_wt_exc}")
                return
            assert _mem_should_summarise is not None and _mem_summarise is not None
            try:
                if _mem_should_summarise(_stream_sid):
                    _mem_summarise(_stream_sid, _stream_mem_uid, _groq_client)
            except Exception as _s_exc:
                logger.warning(f"[MEMORY] /ask/stream summarise_session failed: {_s_exc}")
            # Sprint 4 — extract structured facts from the latest turns.
            # Uses read_all_turns (not read_buffer) so facts stated in older turns
            # that have been budget-trimmed from the live window are still seen.
            try:
                if _mem_extract_facts and _mem_read_all_turns and _groq_client:
                    all_turns = _mem_read_all_turns(_stream_sid, _stream_mem_uid)
                    _mem_extract_facts(_stream_mem_uid, all_turns, _groq_client)
            except Exception as _ef_exc:
                logger.warning(f"[MEMORY] /ask/stream extract_and_store_facts failed: {_ef_exc}")
            # Sprint 5 — auto-extract and persist semantic preferences.
            try:
                if _mem_extract_prefs and _mem_read_all_turns and _groq_client:
                    all_turns = _mem_read_all_turns(req.session_id, _stream_mem_uid)
                    _mem_extract_prefs(_stream_mem_uid, all_turns, _groq_client)
            except Exception as _ep_exc:
                logger.warning(f"[MEMORY] /ask/stream extract_and_store_preferences failed: {_ep_exc}")

        background_tasks.add_task(_stream_write_turns)

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Customer support feedback endpoints ───────────────────────────────────────

class FeedbackRequest(BaseModel):
    """Body for POST /feedback — submit a rating for a completed RAG interaction."""
    question:          str
    answer:            str
    session_id:        str  = ""
    user_id:           str  = ""
    intent:            str  = ""
    rating:            int  = 0     # 1=helpful, -1=not helpful, 0=not rated
    escalated:         bool = False
    escalation_reason: str  = ""
    eval_verdict:      str  = ""
    confidence:        float | None = None
    elapsed_ms:        float | None = None


@app.post("/feedback", dependencies=[Depends(require_api_key)])
async def submit_feedback(req: FeedbackRequest):
    """
    Submit feedback for a completed support interaction.

    Called after the customer rates the answer (thumbs up / down) or after
    an escalation occurs.  All fields except `question` and `answer` are optional.

    Returns: { "feedback_id": str }
    """
    if not FEEDBACK_ENABLED or _FeedbackEntry is None or _store_feedback is None:
        raise HTTPException(status_code=503, detail="Feedback store not available.")
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty.")

    entry = _FeedbackEntry(
        question          = req.question.strip(),
        answer            = req.answer.strip(),
        session_id        = req.session_id,
        user_id           = req.user_id,
        intent            = req.intent,
        rating            = req.rating,
        escalated         = req.escalated,
        escalation_reason = req.escalation_reason,
        eval_verdict      = req.eval_verdict,
        confidence        = req.confidence,
        elapsed_ms        = req.elapsed_ms,
    )
    fid = _store_feedback(entry)
    return JSONResponse(content={"feedback_id": fid})


@app.get("/feedback/summary")
async def feedback_summary():
    """
    Return aggregate statistics across all stored feedback.

    Useful for monitoring support quality over time:
    total interactions, satisfaction rate, escalation rate,
    breakdown by intent and eval verdict.

    Response:
        {
          "total":           int,
          "helpful":         int,
          "not_helpful":     int,
          "escalated":       int,
          "by_intent":       {intent: count},
          "by_verdict":      {verdict: count},
          "avg_confidence":  float | null,
          "satisfaction_pct": float | null
        }
    """
    if not FEEDBACK_ENABLED or _get_feedback_summary is None:
        raise HTTPException(status_code=503, detail="Feedback store not available.")
    try:
        summary = _get_feedback_summary()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to get feedback summary: {exc}")
    return JSONResponse(content=summary)


@app.get("/feedback")
async def list_feedback_entries(offset: int = 0, limit: int = 50):
    """
    Return recent feedback entries (newest first).

    Args:
        offset: Skip the first N entries (pagination).
        limit:  Max entries to return (max 200).
    """
    if not FEEDBACK_ENABLED or _list_feedback is None:
        raise HTTPException(status_code=503, detail="Feedback store not available.")
    limit = min(limit, 200)
    try:
        entries = _list_feedback(limit=limit, offset=offset)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list feedback: {exc}")
    return JSONResponse(content={"total": len(entries), "offset": offset, "limit": limit, "entries": entries})


@app.get("/index/documents")
async def list_indexed_documents(
    collection:   str  = _EMB_DEFAULT_COLLECTION if EMBEDDING_ENABLED else "documents",
    offset:       int  = 0,
    limit:        int  = 50,
    current_user: dict = Depends(require_user),
):
    """
    List documents indexed in Qdrant for the authenticated user.

    Admins see all documents (no owner filter). Regular users see only their
    own documents plus documents in the global knowledge base (owner_id='__global__').

    Returns a paginated list sorted by total chunk count descending:
        [{ "source": str, "chunks": int, "types": {"text": n, "table": n, ...} }]
    """
    _require_embedding()
    assert _qdrant_client is not None
    limit    = min(limit, 200)
    uid      = current_user["user_id"]
    is_admin = current_user.get("role") == "admin"

    # Admins see everything; regular users see their own + global KB
    filter_owner = "" if is_admin else str(uid)
    try:
        all_docs = _emb_list_indexed_docs(
            _qdrant_client, collection=collection, owner_id=filter_owner
        )
        # For regular users also include global KB documents
        if not is_admin:
            global_docs = _emb_list_indexed_docs(
                _qdrant_client, collection=collection, owner_id="__global__"
            )
            # Merge, deduplicate by source name
            seen = {d["source"] for d in all_docs}
            for d in global_docs:
                if d["source"] not in seen:
                    all_docs.append(d)
                    seen.add(d["source"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list indexed documents: {e}")

    total = len(all_docs)
    page  = all_docs[offset : offset + limit]
    return JSONResponse(content={
        "collection": collection,
        "total":      total,
        "offset":     offset,
        "limit":      limit,
        "documents":  page,
    })


@app.get("/documents/{doc_id}/chunks")
async def get_document_chunks(
    doc_id:     str,
    collection: str = _EMB_DEFAULT_COLLECTION if EMBEDDING_ENABLED else "documents",
    limit:      int = 1000,
):
    """
    Fetch all indexed chunks belonging to a document identified by its doc_id GUID.

    Every chunk carries metadata.doc_id — a UUID5 derived from the source
    filename — so this call returns every text, table, and image chunk from
    a specific document in document order (sorted by chunk_index).

    Args:
        doc_id:     UUID5 string (e.g. "3f2a1b4c-…").
        collection: Qdrant collection name.
        limit:      Maximum chunks to return (default 1 000, max 5 000).
    """
    _require_embedding()
    assert _qdrant_client is not None
    limit = min(limit, 5000)
    try:
        chunks = _emb_get_chunks_by_doc_id(
            _qdrant_client, doc_id=doc_id, collection=collection, limit=limit
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chunks: {e}")
    if not chunks:
        raise HTTPException(status_code=404, detail=f"No chunks found for doc_id '{doc_id}'.")
    return JSONResponse(content={
        "doc_id":     doc_id,
        "collection": collection,
        "total":      len(chunks),
        "chunks":     chunks,
    })


@app.delete("/index/{source:path}", dependencies=[Depends(require_api_key)])
async def delete_indexed_document(
    source:     str,
    collection: str = _EMB_DEFAULT_COLLECTION if EMBEDDING_ENABLED else "documents",
):
    """
    Remove all indexed chunks for a specific source document from Qdrant.

    `source` is the original filename (e.g. "report.pdf").
    This does NOT delete the result JSON in results/ — only the Qdrant points.
    """
    _require_embedding()
    assert _qdrant_client is not None
    if not source.strip():
        raise HTTPException(status_code=422, detail="source must not be empty.")
    try:
        _emb_delete_by_source(_qdrant_client, source=source, collection=collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete indexed document: {e}")
    return JSONResponse(content={
        "deleted":    source,
        "collection": collection,
        "message":    f"All indexed chunks for '{source}' have been removed.",
    })


# ── Google OAuth (web flow) ────────────────────────────────────────────────────
# Users sign in with Google via GET /auth/google → Google consent →
# GET /auth/google/callback → tokens stored in _GOOGLE_TOKENS keyed by email.
# Those tokens are used automatically when starting crawl jobs, so no
# credentials.json is needed for Google-authenticated users.

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID",     "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI",  "http://localhost:8000/auth/google/callback")
_FRONTEND_URL        = os.getenv("FRONTEND_URL",          "http://localhost:5173")

_GOOGLE_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

# email → {"access_token", "refresh_token", "email", "name"}
_GOOGLE_TOKENS: dict[str, dict] = {}
# state → True  (CSRF guard, cleared immediately after use)
_OAUTH_STATES:  dict[str, bool] = {}

# NOTE: /auth/google and /auth/google/callback are defined earlier in this file
# (around line 1547) using the PKCE flow that works with installed/desktop OAuth
# credentials.  The old google_auth_oauthlib-based definitions have been removed
# to avoid FastAPI duplicate-route warnings.


@app.get("/auth/google/status")
async def auth_google_status(user_id: str):
    """
    Check whether a user has valid stored Google OAuth tokens.

    Returns ``{"connected": true, "email": "..."}`` when tokens are present,
    or ``{"connected": false}`` otherwise.
    """
    if user_id in _GOOGLE_TOKENS:
        return {"connected": True, "email": _GOOGLE_TOKENS[user_id]["email"]}
    return {"connected": False}


@app.delete("/auth/google/{user_id}")
async def auth_google_revoke(user_id: str):
    """Remove stored Google OAuth tokens for a user (sign out from Google)."""
    removed = _GOOGLE_TOKENS.pop(user_id, None)
    if removed:
        logger.info(f"[GoogleAuth] Tokens cleared for {user_id}")
        return {"ok": True}
    return {"ok": False, "detail": "No tokens found for that user."}


# ── Google Crawlers ────────────────────────────────────────────────────────────
# These endpoints start Gmail / Drive crawl jobs in a background thread and
# expose SSE-based status polling so the UI can show live progress.
#
# The crawlers POST each file to /upload internally (same process), so all the
# existing parsing, chunking, and indexing logic is reused unchanged.
#
# In-memory job store: _CRAWL_JOBS  (separate from _JOBS used by async upload)
# History store:       _CRAWL_HISTORY  (last 50 completed/failed runs)

_CRAWL_JOBS:    dict[str, dict] = {}   # job_id → live status dict
_CRAWL_HISTORY: list[dict]      = []   # most-recent-first, capped at 50
_CRAWL_LOCK     = threading.Lock()
_CRAWL_MAX_HISTORY = 50
_CRAWL_HISTORY_FILE = BASE_DIR / "crawl_history.json"


def _load_crawl_history() -> None:
    """Load persisted crawl history from disk on startup (best-effort)."""
    global _CRAWL_HISTORY
    try:
        if _CRAWL_HISTORY_FILE.exists():
            data = json.loads(_CRAWL_HISTORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                _CRAWL_HISTORY = data[:_CRAWL_MAX_HISTORY]
                logger.info(f"[Crawl] Loaded {len(_CRAWL_HISTORY)} history entries from disk.")
    except Exception as exc:
        logger.warning(f"[Crawl] Could not load crawl history: {exc}")


def _save_crawl_history() -> None:
    """Persist crawl history to disk (best-effort, called inside _CRAWL_LOCK)."""
    try:
        _CRAWL_HISTORY_FILE.write_text(
            json.dumps(_CRAWL_HISTORY, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning(f"[Crawl] Could not save crawl history: {exc}")


# Load history at module import time (server start)
_load_crawl_history()

# ── Crawler availability check ───────────────────────────────────────────────

def _crawlers_available() -> bool:
    """Return True when the google-api-python-client package is installed."""
    try:
        import googleapiclient  # noqa: F401
        return True
    except ImportError:
        return False

# ── Job helpers ───────────────────────────────────────────────────────────────

def _crawl_job_create(job_id: str, source: str, config: dict) -> dict:
    entry: dict = {
        "job_id":     job_id,
        "source":     source,          # "gmail" or "drive"
        "config":     config,
        "status":     "running",       # running | done | failed
        "total":      0,
        "indexed":    0,
        "skipped":    0,
        "errors":     0,
        "current":    "",              # name of file being processed
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "error":      None,
    }
    with _CRAWL_LOCK:
        _CRAWL_JOBS[job_id] = entry
    return entry


def _crawl_job_update(job_id: str, **kwargs) -> None:
    with _CRAWL_LOCK:
        if job_id in _CRAWL_JOBS:
            _CRAWL_JOBS[job_id].update(kwargs)


def _crawl_job_finish(job_id: str, status: str, error: str | None = None) -> None:
    with _CRAWL_LOCK:
        job = _CRAWL_JOBS.get(job_id)
        if job:
            job["status"]      = status
            job["finished_at"] = datetime.now(timezone.utc).isoformat()
            job["error"]       = error
            # Move a copy to history
            _CRAWL_HISTORY.insert(0, dict(job))
            if len(_CRAWL_HISTORY) > _CRAWL_MAX_HISTORY:
                _CRAWL_HISTORY.pop()
            del _CRAWL_JOBS[job_id]
            # Persist to disk so history survives server restarts
            _save_crawl_history()

# ── Pydantic request models ──────────────────────────────────────────────────

class GmailCrawlRequest(BaseModel):
    label:       str | None  = "INBOX"
    query:       str | None  = None
    after:       str | None  = None    # YYYY-MM-DD
    before:      str | None  = None    # YYYY-MM-DD
    max_results: int         = 100
    # Owner identity — passed through to /upload so chunks are tagged in Qdrant.
    # When the user authenticated via "Sign in with Google", set user_id to their
    # session/account identifier and user_email to their Gmail address so that
    # retrieval can be filtered to their own data.
    user_id:    str = ""
    user_email: str = ""

class DriveCrawlRequest(BaseModel):
    folder_id:      str | None        = None
    types:          list[str] | None  = None   # ["pdf", "docx", ...]
    modified_after: str | None        = None   # YYYY-MM-DD
    recursive:      bool              = True   # default True — traverses sub-folders automatically
    incremental:    bool              = True   # skip unchanged files (compare Drive modifiedTime)
    max_results:    int               = 200
    # Owner identity — same as GmailCrawlRequest above.
    user_id:    str = ""
    user_email: str = ""

# ── Background worker ─────────────────────────────────────────────────────────

def _run_crawler(job_id: str, source: str, kwargs: dict) -> None:
    """
    Background thread: run a Gmail or Drive crawl and report progress back via
    the _CRAWL_JOBS dict.  Files are uploaded to /upload on localhost so the
    full parsing + chunking + indexing pipeline runs automatically.

    When ``kwargs["user_id"]`` is present and matches a key in ``_GOOGLE_TOKENS``
    (web OAuth flow), that token is used to build the service object.
    Otherwise falls back to the desktop credentials.json / token.json flow.
    """
    import time as _time

    api_url    = f"http://localhost:{os.environ.get('PORT', '8000')}"
    api_key    = os.environ.get("API_KEY", "")
    # Delay (seconds) between consecutive file uploads.  The async upload
    # worker pool (max 3) starts OCR/captioning immediately after each POST,
    # so submitting 78 files back-to-back saturates Windows' TCP socket
    # buffer pool (WinError 10055).  A 5-second gap ensures at most a few
    # uploads are actively processing at once.
    _INTER_FILE_DELAY = 5.0
    user_id    = kwargs.get("user_id",    "")
    user_email = kwargs.get("user_email", "")
    # Crawl jobs are admin-only — always feed the global knowledge base so
    # all users can query the crawled content.  The background thread does
    # not send a JWT, so the RBAC owner_id resolution in /upload would not
    # fire; we set it explicitly here instead.
    _crawl_owner_id = "__global__"

    # ── Resolve OAuth service ─────────────────────────────────────────────────
    def _build_gmail_service():
        if user_id and user_id in _GOOGLE_TOKENS:
            from crawlers.auth import get_gmail_service_from_tokens  # type: ignore[import]
            token_info = _GOOGLE_TOKENS[user_id]
            return get_gmail_service_from_tokens(
                token_info, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
            )
        from crawlers.auth import get_gmail_service  # type: ignore[import]
        return get_gmail_service()

    def _build_drive_service():
        if user_id and user_id in _GOOGLE_TOKENS:
            from crawlers.auth import get_drive_service_from_tokens  # type: ignore[import]
            token_info = _GOOGLE_TOKENS[user_id]
            return get_drive_service_from_tokens(
                token_info, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
            )
        from crawlers.auth import get_drive_service  # type: ignore[import]
        return get_drive_service()

    try:
        if source == "gmail":
            from crawlers.gmail_crawler import _build_query, _upload, iter_threads  # type: ignore[import]

            service = _build_gmail_service()
            q = _build_query(
                kwargs.get("label"),
                kwargs.get("query"),
                kwargs.get("after"),
                kwargs.get("before"),
            )
            for slug, eml_bytes in iter_threads(service, q, kwargs["max_results"]):
                _crawl_job_update(job_id, current=f"{slug}.eml", total=_CRAWL_JOBS.get(job_id, {}).get("total", 0) + 1)
                try:
                    _upload(api_url, api_key, f"{slug}.eml", eml_bytes, _crawl_owner_id, user_email)
                    _crawl_job_update(
                        job_id,
                        indexed=_CRAWL_JOBS.get(job_id, {}).get("indexed", 0) + 1,
                    )
                except Exception as e:
                    logger.warning(f"[Crawl:{job_id}] Upload error: {e}")
                    _crawl_job_update(
                        job_id,
                        errors=_CRAWL_JOBS.get(job_id, {}).get("errors", 0) + 1,
                    )
                # Throttle: give the server time to finish OCR/captioning
                # before the next file arrives (prevents socket exhaustion).
                _time.sleep(_INTER_FILE_DELAY)

        elif source == "drive":
            from crawlers.drive_crawler import _build_mime_filter, iter_files          # type: ignore[import]
            from crawlers.drive_crawler import _upload as _drive_upload                # type: ignore[import]
            from crawlers.drive_crawler import _delete_old_source as _drv_del_source  # type: ignore[import]

            # Optional crawl-state persistence (requires PostgreSQL drive_files table)
            try:
                from file_processor.drive_store import update_crawl_state as _drv_update_state
            except Exception:
                _drv_update_state = None  # type: ignore[assignment]

            service     = _build_drive_service()
            allowed     = _build_mime_filter(kwargs.get("types"))
            incremental = bool(kwargs.get("incremental", True))

            # Callback invoked by iter_files for every unchanged file that is
            # skipped — lets us keep the job's skipped counter current in
            # real time without modifying iter_files' yield contract.
            def _on_skip(_name: str) -> None:
                _crawl_job_update(
                    job_id,
                    skipped=_CRAWL_JOBS.get(job_id, {}).get("skipped", 0) + 1,
                )

            for filename, content, meta in iter_files(
                service,
                folder_id      = kwargs.get("folder_id"),
                allowed_mimes  = allowed,
                modified_after = kwargs.get("modified_after"),
                recursive      = bool(kwargs.get("recursive", True)),
                max_results    = kwargs["max_results"],
                incremental    = incremental,
                on_skip        = _on_skip,
            ):
                drive_file_id = meta.get("id", "")
                action        = meta.get("_action", "full")
                modified_time = meta.get("_modified_time", "")
                indexed_name  = meta.get("_indexed_name", filename)
                old_indexed   = meta.get("_old_indexed_name")   # set only for renames

                _crawl_job_update(
                    job_id,
                    current = filename,
                    total   = _CRAWL_JOBS.get(job_id, {}).get("total", 0) + 1,
                )

                # For renamed files: remove stale Qdrant entries under the old
                # source name before uploading under the new name.
                if old_indexed:
                    try:
                        _drv_del_source(api_url, api_key, old_indexed)
                        logger.info(
                            f"[Crawl:{job_id}] Deleted stale Qdrant entries "
                            f"for renamed source: {old_indexed!r}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"[Crawl:{job_id}] Could not delete old Qdrant "
                            f"entries for {old_indexed!r}: {e}"
                        )

                try:
                    _drive_upload(
                        api_url, api_key, filename, content, _crawl_owner_id, user_email,
                        drive_file_id = drive_file_id,
                    )
                    _crawl_job_update(
                        job_id,
                        indexed = _CRAWL_JOBS.get(job_id, {}).get("indexed", 0) + 1,
                    )

                    # Persist crawl state so the next incremental run can skip
                    # this file if it hasn't changed.
                    if _drv_update_state is not None and drive_file_id and modified_time:
                        try:
                            _drv_update_state(drive_file_id, modified_time, indexed_name)
                        except Exception as e:
                            logger.warning(
                                f"[Crawl:{job_id}] Could not save crawl state "
                                f"for {filename}: {e}"
                            )

                except Exception as e:
                    logger.warning(f"[Crawl:{job_id}] Upload error ({action}): {e}")
                    _crawl_job_update(
                        job_id,
                        errors = _CRAWL_JOBS.get(job_id, {}).get("errors", 0) + 1,
                    )
                # Throttle: give the server time to finish OCR/captioning
                # before the next file arrives (prevents socket exhaustion).
                _time.sleep(_INTER_FILE_DELAY)

        _crawl_job_finish(job_id, "done")

    except Exception as exc:
        logger.error(f"[Crawl:{job_id}] Fatal error: {exc}")
        _crawl_job_finish(job_id, "failed", error=str(exc))


# ── Crawler endpoints ─────────────────────────────────────────────────────────

@app.post("/crawl/gmail", dependencies=[Depends(require_admin)])
async def start_gmail_crawl(req: GmailCrawlRequest, background_tasks: BackgroundTasks):
    """
    Start a background Gmail crawl job.

    Fetches emails matching the given filters, converts them to .eml, and
    indexes them via the existing /upload pipeline.

    Returns a job_id that can be polled via GET /crawl/status/{job_id}.
    """
    if not _crawlers_available():
        raise HTTPException(
            status_code=503,
            detail="Google API client not installed. Run: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client",
        )
    job_id = str(uuid.uuid4())[:8]
    config = req.model_dump()
    _crawl_job_create(job_id, "gmail", config)
    background_tasks.add_task(_run_crawler, job_id, "gmail", config)
    return {"job_id": job_id, "status": "running", "poll_url": f"/crawl/status/{job_id}"}


@app.post("/crawl/drive", dependencies=[Depends(require_admin)])
async def start_drive_crawl(req: DriveCrawlRequest, background_tasks: BackgroundTasks):
    """
    Start a background Google Drive crawl job.

    Lists and downloads files from the specified folder (or My Drive root)
    and indexes them via the existing /upload pipeline.

    Returns a job_id that can be polled via GET /crawl/status/{job_id}.
    """
    if not _crawlers_available():
        raise HTTPException(
            status_code=503,
            detail="Google API client not installed. Run: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client",
        )
    job_id = str(uuid.uuid4())[:8]
    config = req.model_dump()
    _crawl_job_create(job_id, "drive", config)
    background_tasks.add_task(_run_crawler, job_id, "drive", config)
    return {"job_id": job_id, "status": "running", "poll_url": f"/crawl/status/{job_id}"}


@app.get("/crawl/status/{job_id}", dependencies=[Depends(require_admin)])
async def get_crawl_status(job_id: str):
    """
    Poll the status of a crawl job (admin only).

    While running, returns live counts of files indexed/skipped/errored and
    the name of the file currently being processed.

    After the job completes or fails, the result is moved to the history
    store and this endpoint returns 404.
    """
    with _CRAWL_LOCK:
        job = _CRAWL_JOBS.get(job_id)

    if job:
        return dict(job)

    # Check history for completed/failed jobs
    with _CRAWL_LOCK:
        for entry in _CRAWL_HISTORY:
            if entry.get("job_id") == job_id:
                return dict(entry)

    raise HTTPException(status_code=404, detail=f"Crawl job {job_id!r} not found")


@app.get("/crawl/history", dependencies=[Depends(require_admin)])
async def get_crawl_history():
    """Return the list of completed crawl runs, newest first (admin only)."""
    with _CRAWL_LOCK:
        return {"history": list(_CRAWL_HISTORY)}


@app.get("/crawl/active", dependencies=[Depends(require_admin)])
async def get_active_crawls():
    """Return all currently running crawl jobs (admin only)."""
    with _CRAWL_LOCK:
        return {"jobs": list(_CRAWL_JOBS.values())}
