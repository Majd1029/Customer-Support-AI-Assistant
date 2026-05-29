"""
nodes.py — LangGraph node functions for the full RAG pipeline.

Each function accepts a RAGState dict and returns a partial dict of updated
state keys.  Nodes that emit SSE events call _writer(), which dispatches
LangGraph custom stream events when the graph runs in streaming mode and is a
no-op when called via ainvoke().

Streaming protocol (preserved from the original api.py SSE implementation):
    {"type": "sources",   "sources": [...], "chunks_used": N, "hops": N}
    {"type": "token",     "content": "..."}          — one per LLM token
    {"type": "eval_start","query": "..."}
    {"type": "eval_node", "node": "...", ...}         — one per eval node
    {"type": "eval_done", "verdict": "...", ...}
    {"type": "escalation","should_escalate": true, ...} — when applicable
    {"type": "done",      "confidence": ..., ...}    — final event

Import note: nodes import from file_preparation.* implementation modules
directly and never from file_processor.api, which avoids circular imports.
All api-level singletons (groq_client, qdrant_client) are passed in via state.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Helper: safe stream writer
# ─────────────────────────────────────────────────────────────────────────────

def _writer():
    """
    Return the LangGraph custom-stream writer for the current execution context.
    Falls back to a no-op lambda so the same node code works for both
    astream(stream_mode="custom") and ainvoke().
    """
    try:
        from langgraph.config import get_stream_writer  # type: ignore
        return get_stream_writer()
    except Exception:
        return lambda _: None


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Intent classification (customer-support mode only)
# ─────────────────────────────────────────────────────────────────────────────

async def intent_node(state: dict) -> dict:
    """
    Classify the user's intent via Groq llama-3.1-8b-instant.
    A regex fast-path detects explicit escalation phrases in < 1 ms.
    Skipped entirely when support_mode=False.
    """
    if not state.get("support_mode"):
        return {}

    try:
        from file_preparation.intent.classifier import classify_intent  # type: ignore
    except ImportError:
        logger.warning("[GRAPH][INTENT] classifier not available")
        return {}

    try:
        intent_result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: classify_intent(
                state["question"],
                groq_client=state.get("groq_client"),
            ),
        )
        logger.info(
            f"[GRAPH][INTENT] intent={intent_result.intent} "
            f"strategy={intent_result.strategy} "
            f"conf={intent_result.confidence:.2f} "
            f"escalate_now={intent_result.escalate_now}"
        )
        return {"intent_result": intent_result}
    except Exception as exc:
        logger.warning(f"[GRAPH][INTENT] classification failed: {exc}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Memory context + query rewriting
# ─────────────────────────────────────────────────────────────────────────────

async def memory_node(state: dict) -> dict:
    """
    Load the conversation buffer, apply 3-tier query rewriting, and assemble
    the MemoryContext (summary, user_facts, recalled_preferences).
    Skipped when memory_enabled=False or session_id is absent.
    """
    if not state.get("memory_enabled") or not state.get("session_id"):
        return {"retrieval_question": state.get("question", ""), "memory_context": None}

    try:
        from file_preparation.memory.orchestrator import load_memory_context  # type: ignore
    except ImportError:
        logger.warning("[GRAPH][MEMORY] orchestrator not available")
        return {"retrieval_question": state.get("question", ""), "memory_context": None}

    try:
        mem_ctx = await load_memory_context(
            session_id=state["session_id"],
            user_id=state.get("user_id") or state["session_id"],
            raw_query=state["question"],
            groq_client=state.get("groq_client"),
            memory_enabled=True,
        )
        rewritten = mem_ctx.rewritten_query
        logger.info(f"[GRAPH][MEMORY] rewrite_tier={mem_ctx.rewrite_tier} "
                    f"rewritten={rewritten != state['question']}")
        return {"memory_context": mem_ctx, "retrieval_question": rewritten}
    except Exception as exc:
        logger.warning(f"[GRAPH][MEMORY] load_memory_context failed: {exc}")
        return {"retrieval_question": state.get("question", ""), "memory_context": None}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Hybrid retrieval (single-hop or multi-hop)
# ─────────────────────────────────────────────────────────────────────────────

async def retrieve_node(state: dict) -> dict:
    """
    Run hybrid RRF retrieval (BGE-M3 dense + SPLADE sparse) with optional
    HyDE expansion, cross-encoder reranking, MMR diversification, and
    query decomposition.  Routes to multihop_retrieve when multi_hop=True.
    """
    try:
        from file_preparation.retrieval.retriever import (  # type: ignore
            retrieve_evidence,
            multihop_retrieve,
        )
    except ImportError as e:
        logger.error(f"[GRAPH][RETRIEVE] import failed: {e}")
        return {"retrieval_result": None, "retrieval_ms": 0.0}

    retrieval_question = state.get("retrieval_question") or state["question"]
    client = state.get("qdrant_client")
    if client is None:
        logger.error("[GRAPH][RETRIEVE] qdrant_client not in state")
        return {"retrieval_result": None, "retrieval_ms": 0.0}

    kwargs = dict(
        collection=state.get("collection", "documents"),
        limit=state.get("limit", 5),
        context_window=state.get("context_window", 1),
        filters=state.get("rbac_filters") or state.get("filters"),
        source_filter=state.get("source_filter"),
        min_score=state.get("min_score"),
        rerank=state.get("rerank", False),
        use_hyde=state.get("use_hyde", False),
        mmr=state.get("mmr", False),
        mmr_lambda=state.get("mmr_lambda", 0.5),
        decompose=state.get("decompose", False),
    )

    try:
        retrieval_fn = multihop_retrieve if state.get("multi_hop") else retrieve_evidence
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: retrieval_fn(retrieval_question, client, **kwargs),
        )
        logger.info(
            f"[GRAPH][RETRIEVE] {'multi-hop' if state.get('multi_hop') else 'single-hop'} "
            f"chunks={len(result.chunks)} hops={result.hops} "
            f"elapsed={result.elapsed_ms:.0f}ms"
        )
        return {"retrieval_result": result, "retrieval_ms": result.elapsed_ms}
    except Exception as exc:
        logger.error(f"[GRAPH][RETRIEVE] failed: {exc}")
        return {"retrieval_result": None, "retrieval_ms": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — Flatten chunks + context expansion neighbors
# ─────────────────────────────────────────────────────────────────────────────

def flatten_node(state: dict) -> dict:
    """
    Merge primary retrieved chunks with their context-expansion neighbors into
    a single deduplicated list for the ContextBuilder.
    """
    retrieval = state.get("retrieval_result")
    if retrieval is None or not retrieval.chunks:
        return {"gen_chunks": []}

    seen: set[str] = set()
    gen_chunks: list[dict] = []

    for chunk in retrieval.chunks:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
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
            if nb_id and nb_id not in seen:
                seen.add(nb_id)
                gen_chunks.append({
                    "content":  nb.get("content", ""),
                    "chunk_id": nb_id,
                    "score":    0.0,
                    "hop":      1,
                    "primary":  False,
                    "metadata": nb,
                })

    logger.debug(f"[GRAPH][FLATTEN] {len(gen_chunks)} gen_chunks "
                 f"({len(retrieval.chunks)} primary + neighbors)")
    return {"gen_chunks": gen_chunks}


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — Context building + sources event
# ─────────────────────────────────────────────────────────────────────────────

def build_context_node(state: dict) -> dict:
    """
    Assemble the numbered context block within the token budget.
    Emits the SSE 'sources' event so the client can display citations
    before the first answer token arrives.
    """
    from file_preparation.generation import GenerationConfig, get_generator  # type: ignore

    write = _writer()
    retrieval = state.get("retrieval_result")

    cfg = _make_generation_config(state)
    gen = get_generator()

    sources, tokens, context_text, ordered_chunks = gen.build_context_metadata(
        state.get("gen_chunks", []), cfg
    )

    hops = retrieval.hops if retrieval else 1

    # Emit sources event immediately — client renders citation chips while streaming
    write({
        "type":        "sources",
        "sources":     sources,
        # Only count primary retrieved chunks, not context-expansion neighbors
        "chunks_used": sum(1 for c in state.get("gen_chunks", []) if c.get("primary")),
        "hops":        hops,
    })

    return {
        "context_text":      context_text,
        "sources":           sources,
        "ordered_chunks":    ordered_chunks,
        "tokens_in_context": tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 6 — Generation (streaming token-by-token)
# ─────────────────────────────────────────────────────────────────────────────

async def generate_node(state: dict) -> dict:
    """
    Stream tokens from the active backend (Groq primary / Ollama fallback).

    The sync generator (GroqBackend.stream / OllamaBackend.stream) runs in a
    ThreadPoolExecutor.  Tokens are forwarded to the SSE layer via a
    asyncio.Queue + loop.call_soon_threadsafe pattern so the async event loop
    is never blocked.

    Post-generation: citation-filtered sources replace the pre-generation list,
    and the no_answer flag is detected via the sentinel regex.
    """
    from file_preparation.generation import GenerationConfig, get_generator  # type: ignore
    from file_preparation.generation.answer_generator import (  # type: ignore
        _build_prompt,
        _NO_ANSWER_RE,
        _extract_cited_sources,
    )

    write = _writer()
    gen = get_generator()
    cfg = _make_generation_config(state)

    context_text = state.get("context_text", "")
    if not context_text:
        write({"type": "token", "content": "I could not find any relevant information in the indexed documents."})
        return {
            "answer": "I could not find any relevant information in the indexed documents.",
            "no_answer": True, "citation_count": 0,
            "backend": "none", "model": "none",
            "generation_ms": 0.0, "token_counts": None,
        }

    retrieval_question = state.get("retrieval_question") or state["question"]
    mem_ctx = state.get("memory_context")
    history = mem_ctx.prompt_messages() if (mem_ctx and getattr(mem_ctx, "has_history", False)) else None

    system, user_msg = _build_prompt(retrieval_question, context_text, cfg)

    backend, backend_name = gen._pick_backend()
    logger.info(f"[GRAPH][GENERATE] streaming via {backend_name}/{backend.model}")

    # ── Token streaming via queue ─────────────────────────────────────────
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    full_answer_holder: list[str] = [""]
    t0 = time.monotonic()

    def _stream_worker() -> None:
        try:
            for token in backend.stream(system, user_msg, cfg, history=history):
                full_answer_holder[0] += token
                loop.call_soon_threadsafe(queue.put_nowait, token)
        except Exception as exc:
            logger.error(f"[GRAPH][GENERATE] backend stream error: {exc}")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)   # sentinel

    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = loop.run_in_executor(executor, _stream_worker)

    while True:
        token = await queue.get()
        if token is None:
            break
        write({"type": "token", "content": token})

    await future
    executor.shutdown(wait=False)

    generation_ms = (time.monotonic() - t0) * 1000
    full_answer = full_answer_holder[0].strip()
    token_counts = getattr(backend, "last_token_counts", None) or {}

    # ── No-answer detection + citation filtering ──────────────────────────
    no_answer = bool(_NO_ANSWER_RE.search(full_answer))
    ordered_chunks = state.get("ordered_chunks", [])

    if not no_answer:
        cited = _extract_cited_sources(full_answer, ordered_chunks)
        if cited:
            seen: set[tuple] = set()
            final_sources: list[dict] = []
            for c in cited:
                key = (c["source"], c["page_start"])
                if key not in seen:
                    seen.add(key)
                    final_sources.append({
                        "chunk_id":   c.get("chunk_id", ""),
                        "source":     c["source"],
                        "page_start": c["page_start"],
                        "section":    c.get("section", ""),
                        "score":      round(c.get("score", 0.0), 4),
                    })
        else:
            final_sources = state.get("sources", [])
    else:
        final_sources = []

    citation_count = len(final_sources)
    logger.info(
        f"[GRAPH][GENERATE] done in {generation_ms:.0f}ms | "
        f"no_answer={no_answer} | cited={citation_count}"
    )

    return {
        "answer":         full_answer,
        "sources":        final_sources,
        "backend":        backend_name,
        "model":          backend.model,
        "generation_ms":  round(generation_ms, 1),
        "token_counts":   token_counts if token_counts else None,
        "no_answer":      no_answer,
        "citation_count": citation_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 7 — Scoring (semantic alignment + confidence)
# ─────────────────────────────────────────────────────────────────────────────

async def score_node(state: dict) -> dict:
    """
    Fast-path evaluation:
      1. Semantic Alignment — BGE-M3 cosine similarity between query and answer.
      2. Confidence Score  — Groq-backed grounding scorer (single decimal, stop=\\n).

    Emits eval_start + two eval_node events.
    """
    write = _writer()
    question = state.get("retrieval_question") or state.get("question", "")
    answer   = state.get("answer", "")

    write({"type": "eval_start", "query": question})

    # ── Semantic alignment ────────────────────────────────────────────────
    alignment: Optional[float] = None
    try:
        from file_preparation.embedding.embedder import encode_query  # type: ignore
        import numpy as np

        eq = await asyncio.get_event_loop().run_in_executor(
            None, lambda: encode_query(question)
        )
        ea = await asyncio.get_event_loop().run_in_executor(
            None, lambda: encode_query(answer[:512])
        )
        dq = np.array(eq.dense[0])
        da = np.array(ea.dense[0])
        norm = (np.linalg.norm(dq) * np.linalg.norm(da))
        alignment = float(np.dot(dq, da) / norm) if norm > 0 else 0.0
        write({
            "type":            "eval_node",
            "node":            "semantic_alignment",
            "alignment_score": round(alignment, 4),
        })
        logger.debug(f"[GRAPH][SCORE] alignment={alignment:.4f}")
    except Exception as exc:
        logger.warning(f"[GRAPH][SCORE] alignment failed: {exc}")

    # ── Confidence / grounding score ──────────────────────────────────────
    confidence: Optional[float] = None
    try:
        from file_preparation.retrieval.retriever import score_answer_confidence  # type: ignore
        from file_preparation.retrieval.retriever import RetrievedChunk  # type: ignore

        ctx_chunks = [
            RetrievedChunk(
                chunk_id=c.get("chunk_id", ""),
                content =c.get("content", ""),
                score   =c.get("score", 0.0),
                metadata={},
                neighbors=[],
                hop     =1,
            )
            for c in (state.get("ordered_chunks") or [])
        ]
        if ctx_chunks:
            confidence = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: score_answer_confidence(
                    answer, ctx_chunks, groq_client=state.get("groq_client")
                ),
            )
        write({
            "type":             "eval_node",
            "node":             "grounding_score",
            "confidence_score": round(confidence, 4) if confidence is not None else None,
        })
        logger.debug(f"[GRAPH][SCORE] confidence={confidence}")
    except Exception as exc:
        logger.warning(f"[GRAPH][SCORE] confidence scoring failed: {exc}")

    return {"alignment_score": alignment, "confidence_score": confidence}


# ─────────────────────────────────────────────────────────────────────────────
# Node 8 — LLM judge (borderline or explicitly requested)
# ─────────────────────────────────────────────────────────────────────────────

async def judge_node(state: dict) -> dict:
    """
    Full five-dimension LLM-as-a-Judge evaluation via qwen/qwen3-32b.
    Only fires when:
      - judge=True is explicitly set in the request, OR
      - alignment_score < 0.65 or confidence_score <= 0.72 (borderline).
    """
    write = _writer()

    try:
        from file_preparation.evaluation.judge import judge_answer  # type: ignore
    except ImportError:
        logger.warning("[GRAPH][JUDGE] judge module not available")
        return {}

    try:
        jr = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: judge_answer(
                question     = state.get("question", ""),
                answer       = state.get("answer", ""),
                chunks       = state.get("ordered_chunks") or [],
                no_answer    = state.get("no_answer", False),
                reference    = state.get("reference"),
                score_chunks = state.get("score_chunks", False),
                retrieval_ms = state.get("retrieval_ms"),
                generation_ms= state.get("generation_ms"),
                token_counts = state.get("token_counts"),
            ),
        )
        write({
            "type":  "eval_node",
            "node":  "llm_judge",
            "judge": jr.to_dict(),
        })
        logger.info(f"[GRAPH][JUDGE] overall={jr.overall} error={jr.error}")
        return {"judge_result": jr}
    except Exception as exc:
        logger.warning(f"[GRAPH][JUDGE] evaluation failed: {exc}")
        write({"type": "eval_error", "message": str(exc)})
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Node 9 — Verdict aggregation
# ─────────────────────────────────────────────────────────────────────────────

def verdict_node(state: dict) -> dict:
    """
    Derive the final evaluation verdict and check post-generation escalation.
    Emits eval_done.  When the LLM judge ran its result is authoritative; the
    confidence score is only used as a fallback when the judge was skipped.
    """
    write = _writer()

    alignment  = state.get("alignment_score")
    confidence = state.get("confidence_score")
    jr         = state.get("judge_result")

    # Verdict priority (mirrors comparison_graph_streaming.py aggregate_verdict)
    if alignment is not None and alignment < 0.30:
        verdict  = "off_topic"
        feedback = "Answer is not aligned with the question."
    elif jr is not None:
        overall = getattr(jr, "overall", None)
        if overall is not None:
            verdict  = "fail" if overall < 0.60 else "pass"
            feedback = getattr(jr, "feedback", "") or ""
        else:
            verdict  = "pass"
            feedback = ""
    elif confidence is not None and confidence < 0.50:
        verdict  = "low_confidence"
        feedback = "Answer may not be fully grounded in the retrieved context."
    else:
        verdict  = "pass"
        feedback = ""

    judge_dict = jr.to_dict() if jr else None

    write({
        "type":             "eval_done",
        "verdict":          verdict,
        "feedback":         feedback,
        "alignment_score":  alignment,
        "confidence_score": confidence,
        "judge":            judge_dict,
    })

    # ── Post-generation escalation check ─────────────────────────────────
    escalated       = False
    escalation_obj  = None
    if state.get("support_mode"):
        try:
            from file_preparation.escalation.handler import should_escalate  # type: ignore
            esc = should_escalate(
                intent      =state.get("intent_result"),
                no_answer   =state.get("no_answer", False),
                eval_verdict=verdict,
            )
            if esc.should_escalate:
                escalated      = True
                escalation_obj = esc
                write({"type": "escalation", **esc.to_dict()})
        except Exception as exc:
            logger.warning(f"[GRAPH][VERDICT] escalation check failed: {exc}")

    return {
        "eval_verdict":     verdict,
        "eval_feedback":    feedback,
        "escalated":        escalated,
        "escalation_result":escalation_obj,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 10 — Pre-retrieval escalation (early exit)
# ─────────────────────────────────────────────────────────────────────────────

def pre_escalation_node(state: dict) -> dict:
    """
    Fires when the intent classifier flags an immediate escalation
    (explicit request or complaint intent).  Emits a minimal SSE stream
    and marks the pipeline as done so no further nodes run.
    """
    write = _writer()
    intent_result = state.get("intent_result")

    try:
        from file_preparation.escalation.handler import should_escalate  # type: ignore
        esc = should_escalate(intent=intent_result)
    except Exception:
        return {}

    write({"type": "sources", "sources": [], "chunks_used": 0, "hops": 0})
    write({"type": "token",   "content": esc.message})
    write({"type": "escalation", **esc.to_dict()})

    return {
        "answer":           esc.message,
        "sources":          [],
        "no_answer":        False,
        "escalated":        True,
        "escalation_result":esc,
        "citation_count":   0,
        "confidence_score": None,
        "alignment_score":  None,
        "eval_verdict":     None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 11 — Memory write-back + final done event
# ─────────────────────────────────────────────────────────────────────────────

async def finalize_node(state: dict) -> dict:
    """
    1. Emit the final SSE 'done' event with all metadata.
    2. Write conversation turns to the memory buffer (fire-and-forget).
    3. Trigger incremental summarisation if the buffer is > 80% full.
    4. Schedule background fact and preference extraction.

    Memory writes run as asyncio tasks so they don't block the SSE stream.
    """
    write = _writer()
    start  = state.get("start_time", time.monotonic())
    elapsed = round((time.monotonic() - start) * 1000, 1)

    retrieval = state.get("retrieval_result")
    hops      = retrieval.hops if retrieval else 1

    # Resolve optional rich objects → dicts for the done event
    _jr  = state.get("judge_result")
    _er  = state.get("escalation_result")
    _ir  = state.get("intent_result")

    done_event = {
        "type":                "done",
        "confidence":          state.get("confidence_score"),
        "retrieval_ms":        state.get("retrieval_ms"),
        "generation_ms":       state.get("generation_ms"),
        "citation_count":      state.get("citation_count", 0),
        "no_answer":           state.get("no_answer", False),
        "hops":                hops,
        "token_counts":        state.get("token_counts"),
        "eval_verdict":        state.get("eval_verdict"),
        "eval_feedback":       state.get("eval_feedback"),
        "escalated":           state.get("escalated", False),
        "rewritten_question":  (
            state.get("retrieval_question")
            if state.get("retrieval_question") != state.get("question")
            else None
        ),
        "rewrite_tier": (
            state["memory_context"].rewrite_tier
            if state.get("memory_context") else None
        ),
        # Fields required by the React UI for eval panel + support features
        "judge":               _jr.to_dict() if _jr and hasattr(_jr, "to_dict") else None,
        "escalation":          _er.to_dict() if _er and hasattr(_er, "to_dict") else None,
        "intent":              _ir.to_dict() if _ir and hasattr(_ir, "to_dict") else None,
    }
    write(done_event)

    # ── Memory write-back (fire-and-forget asyncio tasks) ─────────────────
    session_id = state.get("session_id")
    user_id    = state.get("user_id") or session_id
    answer     = state.get("answer", "")
    question   = state.get("question", "")

    if state.get("memory_enabled") and session_id:
        asyncio.create_task(
            _memory_write_task(session_id, user_id, question, answer, state)
        )

    return {"elapsed_ms": elapsed}


async def _memory_write_task(
    session_id: str,
    user_id:    str,
    question:   str,
    answer:     str,
    state:      dict,
) -> None:
    """Background async task: write turns, summarise, extract facts + prefs."""
    try:
        from file_preparation.memory import (  # type: ignore
            write_turn,
            should_summarise,
            summarise_session,
            read_all_turns,
        )
        loop = asyncio.get_event_loop()

        await loop.run_in_executor(None, lambda: write_turn(session_id, user_id, "user", question))
        await loop.run_in_executor(None, lambda: write_turn(session_id, user_id, "assistant", answer))

        if should_summarise(session_id):
            groq = state.get("groq_client")
            await loop.run_in_executor(None, lambda: summarise_session(session_id, user_id, groq))

        groq = state.get("groq_client")
        try:
            from file_preparation.memory import extract_and_store_facts  # type: ignore
            all_turns = await loop.run_in_executor(
                None, lambda: read_all_turns(session_id, user_id)
            )
            await loop.run_in_executor(
                None, lambda: extract_and_store_facts(user_id, all_turns, groq)
            )
        except ImportError:
            pass

        try:
            from file_preparation.memory import extract_and_store_preferences  # type: ignore
            all_turns = await loop.run_in_executor(
                None, lambda: read_all_turns(session_id, user_id)
            )
            await loop.run_in_executor(
                None, lambda: extract_and_store_preferences(user_id, all_turns, groq)
            )
        except ImportError:
            pass

    except Exception as exc:
        logger.warning(f"[GRAPH][MEMORY_WRITE] background task failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: build GenerationConfig from state
# ─────────────────────────────────────────────────────────────────────────────

def _make_generation_config(state: dict):
    from file_preparation.generation import GenerationConfig  # type: ignore
    from file_preparation.generation.answer_generator import CUSTOMER_SUPPORT_PERSONA  # type: ignore

    mem_ctx = state.get("memory_context")
    persona = state.get("persona") or (CUSTOMER_SUPPORT_PERSONA if state.get("support_mode") else "")

    return GenerationConfig(
        temperature=0.2,
        max_tokens=state.get("max_tokens", 1500),
        language_hint=state.get("language_hint", ""),
        conversation_summary=mem_ctx.summary_as_context() if mem_ctx else "",
        user_facts=mem_ctx.user_facts_as_context() if mem_ctx else "",
        user_preferences=mem_ctx.recalled_preferences_as_context() if mem_ctx else "",
        persona=persona,
    )
