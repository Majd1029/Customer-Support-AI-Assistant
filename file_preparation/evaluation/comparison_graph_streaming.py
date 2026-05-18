"""
file_preparation/evaluation/comparison_graph_streaming.py

Streaming LangGraph query↔answer comparison graph.

Exposes a single public async generator:

    async def stream_comparison(
        user_query:       str,
        generated_answer: str,
        retrieved_chunks: list[dict],
    ) -> AsyncGenerator[str, None]:

It yields SSE-formatted strings as each LangGraph node completes:

    data: {"type": "eval_start",  "query": "..."}
    data: {"type": "eval_node",   "node": "semantic_alignment", "alignment_score": 0.82}
    data: {"type": "eval_node",   "node": "grounding_score",    "confidence_score": 0.74}
    data: {"type": "eval_node",   "node": "llm_judge",          "judge": {...}}
    data: {"type": "eval_done",   "verdict": "pass", "feedback": "...", ...}

The llm_judge node fires only when scores are borderline:
    alignment  0.30–0.65  OR  confidence 0.40–0.72

Every node degrades gracefully on failure — a Groq 429 or BGE-M3 error
emits {"type": "eval_error", "message": "..."} and never crashes the SSE stream.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, AsyncGenerator, Optional, TypedDict

import numpy as np
from loguru import logger

# ── Optional dependency: LangGraph ────────────────────────────────────────────
try:
    from langgraph.graph import StateGraph, END  # type: ignore[import]
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    logger.warning(
        "[EVAL] langgraph not installed — install with `pip install langgraph`. "
        "stream_comparison will emit eval_error events until then."
    )

# ── Optional dependency: encode_query (BGE-M3) ────────────────────────────────
try:
    from file_preparation.embedding import encode_query as _encode_query  # type: ignore[import]
    _EMBED_AVAILABLE = True
except Exception as _ee:
    logger.warning(f"[EVAL] encode_query unavailable ({_ee}) — alignment_score will be null.")
    _encode_query = None          # type: ignore[assignment]
    _EMBED_AVAILABLE = False

# ── Optional dependency: score_answer_confidence ─────────────────────────────
try:
    from file_preparation.retrieval import (  # type: ignore[import]
        score_answer_confidence as _score_confidence,
    )
    _CONFIDENCE_AVAILABLE = True
except Exception as _ce:
    logger.warning(f"[EVAL] score_answer_confidence unavailable ({_ce}) — confidence_score will be null.")
    _score_confidence = None      # type: ignore[assignment]
    _CONFIDENCE_AVAILABLE = False

# ── Optional dependency: judge_answer ────────────────────────────────────────
try:
    from file_preparation.evaluation.judge import (  # type: ignore[import]
        judge_answer as _judge_answer,
    )
    _JUDGE_AVAILABLE = True
except Exception as _je:
    logger.warning(f"[EVAL] judge_answer unavailable ({_je}) — llm_judge node will be skipped.")
    _judge_answer = None          # type: ignore[assignment]
    _JUDGE_AVAILABLE = False

# ── Groq client ───────────────────────────────────────────────────────────────
# api.py injects its singleton here after importing this module so that
# score_answer_confidence and judge_answer share the same connection pool.
_groq_client: Any = None

# ── Regex: strip memory-injection blocks from the raw prompt query ────────────
# _build_prompt() in answer_generator.py injects these blocks before the
# Question: line.  normalize_query strips them so semantic alignment operates
# on the clean user question, not the full augmented prompt.
_USER_FACTS_RE   = re.compile(r"\[USER FACTS\].*?\[/USER FACTS\]",               re.DOTALL | re.IGNORECASE)
_USER_PREFS_RE   = re.compile(r"\[USER PREFERENCES\].*?\[/USER PREFERENCES\]",   re.DOTALL | re.IGNORECASE)
_CONV_SUMMARY_RE = re.compile(r"\[CONVERSATION SUMMARY\].*?\[END SUMMARY\]",     re.DOTALL | re.IGNORECASE)
_QUESTION_RE     = re.compile(r"(?:Question|Q)\s*:\s*(.+)",                       re.IGNORECASE | re.DOTALL)


# ── LangGraph state ───────────────────────────────────────────────────────────
class ComparisonState(TypedDict):
    query:             str
    clean_query:       str
    answer:            str
    chunks:            list          # list[dict] — raw chunk dicts from api.py
    alignment_score:   Optional[float]
    confidence_score:  Optional[float]
    judge_result:      Optional[dict]
    verdict:           Optional[str]
    feedback:          Optional[str]


# ── Minimal chunk stub ────────────────────────────────────────────────────────
# score_answer_confidence expects objects with a .content attribute.
# We build lightweight stubs from the raw dicts rather than importing
# RetrievedChunk (avoids circular imports and heavier retrieval overhead).
class _ChunkStub:
    __slots__ = ("chunk_id", "content", "score", "metadata")

    def __init__(self, d: dict) -> None:
        self.chunk_id = d.get("chunk_id", "")
        self.content  = d.get("content", "")
        self.score    = float(d.get("score", 0.0))
        self.metadata = d.get("metadata", {})


# ── Node: normalize_query ─────────────────────────────────────────────────────
def _node_normalize_query(state: ComparisonState) -> dict:
    """
    Strip [USER FACTS], [USER PREFERENCES], [CONVERSATION SUMMARY]…[END SUMMARY]
    blocks that _build_prompt() injects.  Extract the clean question after
    "Question:" if present.
    """
    q = state["query"]
    q = _USER_FACTS_RE.sub("", q)
    q = _USER_PREFS_RE.sub("", q)
    q = _CONV_SUMMARY_RE.sub("", q)
    q = q.strip()

    m = _QUESTION_RE.search(q)
    if m:
        q = m.group(1).strip()

    q = re.sub(r"\s+", " ", q).strip()
    clean = q or state["query"]
    logger.debug(f"[EVAL] normalize_query: '{clean[:80]}'")
    return {"clean_query": clean}


# ── Node: semantic_alignment ──────────────────────────────────────────────────
def _node_semantic_alignment(state: ComparisonState) -> dict:
    """
    Cosine similarity between the BGE-M3 dense embeddings of the clean query
    and the first 512 characters of the generated answer.
    """
    score: Optional[float] = None
    try:
        if _EMBED_AVAILABLE and _encode_query is not None:
            q_emb = _encode_query(state["clean_query"])
            a_emb = _encode_query(state["answer"][:512])
            q_vec = np.array(q_emb.dense[0], dtype=np.float32)
            a_vec = np.array(a_emb.dense[0], dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            a_norm = np.linalg.norm(a_vec)
            if q_norm > 0.0 and a_norm > 0.0:
                score = round(float(np.dot(q_vec, a_vec) / (q_norm * a_norm)), 4)
    except Exception as _e:
        logger.warning(f"[EVAL] semantic_alignment failed: {_e}")
    logger.debug(f"[EVAL] alignment_score={score}")
    return {"alignment_score": score}


# ── Node: grounding_score ─────────────────────────────────────────────────────
def _node_grounding_score(state: ComparisonState) -> dict:
    """
    Score how well the generated answer is grounded in the retrieved chunks.
    Delegates to score_answer_confidence() in file_preparation.retrieval.
    """
    score: Optional[float] = None
    try:
        if _CONFIDENCE_AVAILABLE and _score_confidence is not None and state["chunks"]:
            stubs = [_ChunkStub(c) for c in state["chunks"]]
            raw = _score_confidence(state["answer"], stubs, groq_client=_groq_client)
            if raw is not None:
                score = round(float(raw), 4)
    except Exception as _e:
        logger.warning(f"[EVAL] grounding_score failed: {_e}")
    logger.debug(f"[EVAL] confidence_score={score}")
    return {"confidence_score": score}


# ── Conditional edge: should_run_judge ────────────────────────────────────────
def _should_run_judge(state: ComparisonState) -> str:
    """
    Route to llm_judge when scores are borderline OR when confidence is very
    low (so the judge can confirm or override a potential false low_confidence):
        alignment  0.30 – 0.65
        confidence <= 0.72   (includes 0.00 – the scorer sometimes underscores)
    Otherwise go straight to aggregate_verdict.
    """
    alignment  = state.get("alignment_score")
    confidence = state.get("confidence_score")
    run_judge  = False
    if alignment  is not None and 0.30 <= alignment <= 0.65:
        run_judge = True
    # Trigger judge for the full low-to-borderline confidence range so a
    # systematic scorer undercount (e.g. 0.00) doesn't lock in low_confidence
    # without a second opinion from the more capable LLM judge.
    if confidence is not None and confidence <= 0.72:
        run_judge = True
    return "llm_judge" if run_judge else "aggregate_verdict"


# ── Node: llm_judge ───────────────────────────────────────────────────────────
def _node_llm_judge(state: ComparisonState) -> dict:
    """
    Full LLM-as-a-Judge evaluation — fires only on borderline scores.
    Serialises JudgeResult with .to_dict() if available, else .__dict__.
    """
    result: Optional[dict] = None
    try:
        if _JUDGE_AVAILABLE and _judge_answer is not None:
            jr = _judge_answer(
                question  = state["clean_query"],
                answer    = state["answer"],
                chunks    = state["chunks"],
                no_answer = not state["answer"].strip(),
            )
            if hasattr(jr, "to_dict"):
                result = jr.to_dict()
            elif hasattr(jr, "__dict__"):
                result = jr.__dict__
            else:
                result = dict(jr)
    except Exception as _e:
        logger.warning(f"[EVAL] llm_judge failed: {_e}")
    return {"judge_result": result}


# ── Node: aggregate_verdict ───────────────────────────────────────────────────
def _node_aggregate_verdict(state: ComparisonState) -> dict:
    """
    Apply thresholds to produce a verdict and human-readable feedback string.

    Verdict priority:
        alignment < 0.30                          → "off_topic"
        judge_overall < 0.60  (when judge ran)    → "fail"
        confidence < 0.50     (no judge result)   → "low_confidence"
        otherwise                                 → "pass"

    When the LLM judge has run, its overall score is the authoritative signal
    and takes precedence over the lightweight confidence scorer.  This prevents
    a systematic scorer undercount from locking in a false "low_confidence".
    """
    alignment     = state.get("alignment_score")
    confidence    = state.get("confidence_score")
    judge_res     = state.get("judge_result")
    judge_overall: Optional[float] = None
    if isinstance(judge_res, dict):
        judge_overall = judge_res.get("overall")

    verdict: str     = "pass"
    reasons: list[str] = []

    # Verdict priority:
    # 1. Off-topic (semantic alignment) — highest signal, checked first.
    # 2. LLM judge overall — when the judge ran, it is the most authoritative
    #    signal and overrides the lighter confidence scorer.  This prevents a
    #    systematic confidence-scorer undercount (e.g. 0.00) from locking in
    #    "low_confidence" when the judge actually grades the answer as good.
    # 3. Raw confidence score — used only when the judge did NOT run.

    if alignment is not None and alignment < 0.30:
        verdict = "off_topic"
        reasons.append(
            f"semantic alignment {alignment:.2f} is below 0.30 — "
            "answer may not address the question"
        )
    elif judge_overall is not None:
        # Judge ran — use its overall score as the authoritative verdict.
        if judge_overall < 0.60:
            verdict = "fail"
            reasons.append(f"LLM judge overall score {judge_overall:.2f} is below 0.60")
        else:
            parts: list[str] = []
            if alignment     is not None: parts.append(f"alignment {alignment:.2f}")
            if confidence    is not None: parts.append(f"confidence {confidence:.2f}")
            parts.append(f"judge {judge_overall:.2f}")
            reasons.append(
                "scores within acceptable range"
                + (f" ({', '.join(parts)})" if parts else "")
            )
    elif confidence is not None and confidence < 0.50:
        # Judge did not run; fall back to raw confidence.
        verdict = "low_confidence"
        reasons.append(
            f"grounding confidence {confidence:.2f} is below 0.50 — "
            "answer may not be grounded in retrieved evidence"
        )
    else:
        parts2: list[str] = []
        if alignment  is not None: parts2.append(f"alignment {alignment:.2f}")
        if confidence is not None: parts2.append(f"confidence {confidence:.2f}")
        reasons.append(
            "scores within acceptable range"
            + (f" ({', '.join(parts2)})" if parts2 else "")
        )

    feedback = "; ".join(reasons) if reasons else "evaluation complete"
    logger.debug(f"[EVAL] verdict={verdict}  feedback={feedback}")
    return {"verdict": verdict, "feedback": feedback}


# ── Build the compiled graph (once at import time) ────────────────────────────
def _build_graph() -> Optional[Any]:
    if not _LANGGRAPH_AVAILABLE:
        return None
    try:
        sg: StateGraph = StateGraph(ComparisonState)

        sg.add_node("normalize_query",    _node_normalize_query)
        sg.add_node("semantic_alignment", _node_semantic_alignment)
        sg.add_node("grounding_score",    _node_grounding_score)
        sg.add_node("llm_judge",          _node_llm_judge)
        sg.add_node("aggregate_verdict",  _node_aggregate_verdict)

        sg.set_entry_point("normalize_query")
        sg.add_edge("normalize_query",    "semantic_alignment")
        sg.add_edge("semantic_alignment", "grounding_score")
        sg.add_conditional_edges(
            "grounding_score",
            _should_run_judge,
            {
                "llm_judge":         "llm_judge",
                "aggregate_verdict": "aggregate_verdict",
            },
        )
        sg.add_edge("llm_judge",         "aggregate_verdict")
        sg.add_edge("aggregate_verdict", END)

        compiled = sg.compile()
        logger.info("[EVAL] Comparison graph compiled successfully.")
        return compiled
    except Exception as _be:
        logger.warning(f"[EVAL] Failed to compile comparison graph: {_be}")
        return None


# Module-level compiled graph — instantiated once, reused on every request.
_comparison_graph: Optional[Any] = _build_graph()


# ── Public async generator ────────────────────────────────────────────────────
async def stream_comparison(
    user_query:       str,
    generated_answer: str,
    retrieved_chunks: list[dict],
) -> AsyncGenerator[str, None]:
    """
    Async generator that streams SSE-formatted evaluation events as each
    LangGraph node in the comparison graph completes.

    Args:
        user_query:       The original user question (may contain memory blocks).
        generated_answer: The full answer produced by the LLM.
        retrieved_chunks: Token-budget-trimmed chunks the LLM actually saw
                          (chunks_in_context from ContextBuilder).

    Yields:
        SSE strings ending with '\\n\\n', e.g.:
            'data: {"type": "eval_start", "query": "..."}\\n\\n'

    Never raises — any internal failure emits an eval_error event and
    the generator then still emits eval_done so the caller's done event
    can carry eval_verdict and eval_feedback.
    """
    if _comparison_graph is None:
        msg = (
            "LangGraph comparison graph is unavailable "
            "(langgraph not installed or graph failed to compile)"
        )
        logger.warning(f"[EVAL] {msg}")
        yield f"data: {json.dumps({'type': 'eval_error', 'message': msg})}\n\n"
        # Still emit eval_done with null fields so the caller's done event is clean
        yield f"data: {json.dumps({'type': 'eval_done', 'verdict': None, 'feedback': msg, 'alignment_score': None, 'confidence_score': None, 'judge': None})}\n\n"
        return

    # ── eval_start ────────────────────────────────────────────────────────────
    yield f"data: {json.dumps({'type': 'eval_start', 'query': user_query})}\n\n"
    await asyncio.sleep(0)  # flush to client

    initial_state: ComparisonState = {
        "query":            user_query,
        "clean_query":      user_query,
        "answer":           generated_answer,
        "chunks":           retrieved_chunks,
        "alignment_score":  None,
        "confidence_score": None,
        "judge_result":     None,
        "verdict":          None,
        "feedback":         None,
    }

    # Track final values for eval_done
    _final_alignment:  Optional[float] = None
    _final_confidence: Optional[float] = None
    _final_judge:      Optional[dict]  = None
    _final_verdict:    Optional[str]   = None
    _final_feedback:   Optional[str]   = None

    # astream_events yields deltas per-node; accumulate into a running state dict.
    _accumulated: dict = dict(initial_state)

    try:
        async for event in _comparison_graph.astream_events(initial_state, version="v2"):
            if event.get("event") != "on_chain_end":
                continue

            node_name: str = event.get("name", "")
            output: Any    = event.get("data", {}).get("output") or {}

            # Merge the node's output delta into the accumulated state
            if isinstance(output, dict):
                _accumulated.update(output)

            # ── Per-node SSE events ───────────────────────────────────────────
            if node_name == "semantic_alignment":
                score = _accumulated.get("alignment_score")
                _final_alignment = score
                payload = {
                    "type":            "eval_node",
                    "node":            "semantic_alignment",
                    "alignment_score": score,
                }
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0)

            elif node_name == "grounding_score":
                score = _accumulated.get("confidence_score")
                _final_confidence = score
                payload = {
                    "type":             "eval_node",
                    "node":             "grounding_score",
                    "confidence_score": score,
                }
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0)

            elif node_name == "llm_judge":
                judge = _accumulated.get("judge_result")
                _final_judge = judge
                payload = {
                    "type":  "eval_node",
                    "node":  "llm_judge",
                    "judge": judge,
                }
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0)

            elif node_name == "aggregate_verdict":
                _final_verdict  = _accumulated.get("verdict")
                _final_feedback = _accumulated.get("feedback")
                # aggregate_verdict does not get its own eval_node event;
                # its output goes directly into eval_done below.

    except Exception as _stream_err:
        logger.warning(f"[EVAL] astream_events error: {_stream_err}")
        yield f"data: {json.dumps({'type': 'eval_error', 'message': str(_stream_err)})}\n\n"
        await asyncio.sleep(0)

    # ── eval_done — always emitted, even after an error ───────────────────────
    done_payload = {
        "type":             "eval_done",
        "verdict":          _final_verdict,
        "feedback":         _final_feedback,
        "alignment_score":  _final_alignment,
        "confidence_score": _final_confidence,
        "judge":            _final_judge,
    }
    yield f"data: {json.dumps(done_payload)}\n\n"
    await asyncio.sleep(0)
