"""
reranker.py — Cross-encoder reranking second stage (Jina Reranker v2).

Two backends, selected automatically:

  1. Jina AI API  (preferred when JINA_API_KEY is set in .env)
     POST https://api.jina.ai/v1/rerank
     - No local model download, no GPU/CPU overhead
     - Free tier: 1 M tokens / month
     - Same model: jina-reranker-v2-base-multilingual

  2. Local sentence-transformers  (fallback when no API key is present)
     jinaai/jina-reranker-v2-base-multilingual  (~560 MB download on first use)
     - 278 M parameters, sliding-window attention (up to 8 192 tokens)
     - Multilingual (100+ languages)
     - Loaded via sentence-transformers CrossEncoder with trust_remote_code=True
     - Scores normalized to [0, 1] via sigmoid (applied at model-load time)

Backend selection:
    JINA_API_KEY set  →  API backend  (fast, zero VRAM/RAM)
    JINA_API_KEY unset →  local model (requires sentence-transformers + torch)

Local load order:
    1. LOCAL_MODEL_DIR  — skips HuggingFace Hub if the directory exists.
    2. HuggingFace Hub  — automatic download on first call.

Typical usage:
    from reranker import rerank, reranker_available

    if reranker_available():
        results = rerank(query, candidates, top_k=10)
    else:
        results = candidates[:10]
"""

from __future__ import annotations

import os
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()

from loguru import logger

# ── CPU thread tuning — must happen BEFORE torch/model import ─────────────────
_CPU_COUNT = os.cpu_count() or 4
try:
    import torch as _torch
    _torch.set_num_threads(_CPU_COUNT)
except ImportError:
    pass

# ── Load environment ──────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_HERE.parent.parent.parent / ".env", override=False)
except ImportError:
    pass

# ── API backend config ────────────────────────────────────────────────────────
_JINA_API_KEY    = os.getenv("JINA_API_KEY", "")
_JINA_API_URL    = "https://api.jina.ai/v1/rerank"
_JINA_API_MODEL  = "jina-reranker-v2-base-multilingual"
_JINA_API_TIMEOUT = 30   # seconds

# ── Local model constants ─────────────────────────────────────────────────────
_RERANKER_HF_NAME  = "jinaai/jina-reranker-v2-base-multilingual"

_LOCAL_MODEL_DIR: str = os.getenv(
    "LOCAL_MODEL_DIR",
    r"C:\Users\majda\jina_model",
)

_MAX_SEQ_LENGTH  = 512
_MAX_CHUNK_CHARS = 1_000   # truncate before tokenisation for local model

# ── Device auto-detection (local model only) ──────────────────────────────────
def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"

_DEVICE = _detect_device()

# ── Local model singleton ─────────────────────────────────────────────────────
_reranker    = None
_reranker_ok = None   # None = not yet checked; True/False after first call


# ── Availability check ────────────────────────────────────────────────────────

def reranker_available() -> bool:
    """
    Return True if either the Jina API key is configured OR
    sentence-transformers is installed for local reranking.

    Does NOT trigger a model download.
    """
    if _JINA_API_KEY:
        return True
    global _reranker_ok
    if _reranker_ok is not None:
        return _reranker_ok
    try:
        from sentence_transformers import CrossEncoder   # noqa: F401
        _reranker_ok = True
    except ImportError:
        _reranker_ok = False
    return _reranker_ok


def _using_api() -> bool:
    """True when the Jina cloud API will be used (JINA_API_KEY is set)."""
    return bool(_JINA_API_KEY)


# ── API backend ───────────────────────────────────────────────────────────────

def _rerank_via_api(
    query:   str,
    results: list[dict[str, Any]],
    top_k:   int | None,
) -> list[dict[str, Any]]:
    """
    Re-rank using the Jina AI cloud API.

    POST https://api.jina.ai/v1/rerank
    Auth: Bearer {JINA_API_KEY}

    The API returns results in ranked order with relevance_score in [0, 1].
    """
    documents = [
        r["payload"].get("content", "")
        for r in results
    ]

    payload = json.dumps({
        "model":            _JINA_API_MODEL,
        "query":            query,
        "documents":        documents,
        "top_n":            top_k if top_k is not None else len(documents),
        "return_documents": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        _JINA_API_URL,
        data    = payload,
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {_JINA_API_KEY}",
        },
        method = "POST",
    )

    logger.debug(
        f"  [reranker/api] Sending {len(documents)} candidates to Jina API …"
    )

    try:
        with urllib.request.urlopen(req, timeout=_JINA_API_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Jina API rerank failed (HTTP {exc.code}): {body[:200]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Jina API unreachable: {exc.reason}"
        ) from exc

    # data["results"] is a list sorted by relevance descending:
    # [{"index": 0, "relevance_score": 0.95}, ...]
    ranked = data.get("results", [])

    reranked: list[dict[str, Any]] = []
    for item in ranked:
        idx   = item["index"]
        score = float(item["relevance_score"])
        entry = dict(results[idx])
        entry["first_stage_score"] = entry.get("score", 0.0)
        entry["score"]             = score
        reranked.append(entry)

    logger.debug(
        f"  [reranker/api] Done. "
        f"Top score: {reranked[0]['score']:.4f}, "
        f"Bottom: {reranked[-1]['score']:.4f}"
        if reranked else "  [reranker/api] No results returned."
    )
    return reranked


# ── Local model backend ───────────────────────────────────────────────────────

def _load_reranker():
    """Lazy-load the Jina cross-encoder model (local backend)."""
    global _reranker
    if _reranker is not None:
        return _reranker

    if not reranker_available():
        raise ImportError(
            "sentence-transformers is required for the local reranker. "
            "Install it with:  pip install sentence-transformers\n"
            "Or set JINA_API_KEY in .env to use the cloud API instead."
        )

    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    from sentence_transformers import CrossEncoder

    sigmoid = torch.nn.Sigmoid() if torch is not None else None

    local_path = Path(_LOCAL_MODEL_DIR)
    if local_path.exists() and local_path.is_dir():
        model_source = str(local_path)
        logger.info(
            f"  Loading Jina reranker from local directory: {model_source} "
            f"[device={_DEVICE}, max_length={_MAX_SEQ_LENGTH}] …"
        )
    else:
        model_source = _RERANKER_HF_NAME
        logger.info(
            f"  Local model dir not found — loading Jina reranker from HuggingFace: "
            f"{model_source} [device={_DEVICE}, max_length={_MAX_SEQ_LENGTH}] …"
        )

    import inspect as _inspect
    _sig = _inspect.signature(CrossEncoder.__init__)
    _act_kwarg = (
        {"activation_fn": sigmoid}
        if "activation_fn" in _sig.parameters
        else {"default_activation_function": sigmoid}
    )

    _reranker = CrossEncoder(
        model_source,
        trust_remote_code = True,
        device            = _DEVICE,
        max_length        = _MAX_SEQ_LENGTH,
        **_act_kwarg,
    )

    logger.info(
        f"  Jina reranker ready "
        f"(device={_DEVICE}, max_length={_MAX_SEQ_LENGTH}, scores=[0,1])."
    )
    return _reranker


def _rerank_via_local(
    query:   str,
    results: list[dict[str, Any]],
    top_k:   int | None,
    score_key: str,
) -> list[dict[str, Any]]:
    """Re-rank using the locally loaded cross-encoder."""
    reranker = _load_reranker()

    pairs = [
        [query, r["payload"].get("content", "")[:_MAX_CHUNK_CHARS]]
        for r in results
    ]

    avg_len = sum(len(p[1]) for p in pairs) / max(len(pairs), 1)
    logger.debug(
        f"  [reranker/local] Reranking {len(pairs)} candidates "
        f"(avg={avg_len:.0f} chars, trunc={_MAX_CHUNK_CHARS}) …"
    )

    scores = reranker.predict(
        pairs,
        convert_to_numpy  = True,
        show_progress_bar = False,
    )

    if not hasattr(scores, "__iter__"):
        scores = [scores]

    reranked: list[dict[str, Any]] = []
    for result, score in zip(results, scores):
        entry = dict(result)
        entry["first_stage_score"] = entry.get("score", 0.0)
        entry[score_key]           = float(score)
        reranked.append(entry)

    reranked.sort(key=lambda r: r[score_key], reverse=True)

    logger.debug(
        f"  [reranker/local] Done. "
        f"Top score: {reranked[0][score_key]:.4f}, "
        f"Bottom: {reranked[-1][score_key]:.4f}"
    )
    return reranked[:top_k] if top_k is not None else reranked


# ── Public API ────────────────────────────────────────────────────────────────

def rerank(
    query:      str,
    results:    list[dict[str, Any]],
    top_k:      int | None  = None,
    score_key:  str         = "score",
) -> list[dict[str, Any]]:
    """
    Re-score search results with the Jina cross-encoder and return sorted list.

    Automatically uses the cloud API if JINA_API_KEY is set, otherwise falls
    back to the locally loaded cross-encoder model.

    Each result dict has its "score" field replaced with the cross-encoder
    relevance score in [0, 1].  The original first-stage RRF score is
    preserved under "first_stage_score".

    Args:
        query:      The user's search query.
        results:    List of result dicts from retrieve() / search().
                    Each must have a "payload" dict with a "content" key.
        top_k:      Return only the top-k reranked results.  None = all.
        score_key:  Key written into each result dict for the reranker score.

    Returns:
        Results sorted by descending cross-encoder score.
    """
    if not results:
        return results

    if _using_api():
        reranked = _rerank_via_api(query, results, top_k)
        # Honour score_key alias if caller wants a different key name
        if score_key != "score":
            for r in reranked:
                r[score_key] = r.pop("score", 0.0)
        return reranked
    else:
        return _rerank_via_local(query, results, top_k, score_key)
