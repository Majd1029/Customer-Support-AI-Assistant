"""
graph.py — LangGraph StateGraph definition for the full RAG pipeline.

Graph structure
───────────────
                         START
                           │
                      intent_node          ← classify intent (support_mode only)
                           │
             ┌─────────────┴──────────────┐
             │ pre_retrieval_escalate?     │ no
             ▼                            ▼
  pre_escalation_node              memory_node         ← load buffer + rewrite query
             │                            │
             │                      retrieve_node       ← hybrid RRF (+ HyDE / rerank / multi-hop)
             │                            │
             │                      flatten_node        ← dedup primary + neighbor chunks
             │                            │
             │               ┌────────────┴─────────────┐
             │               │ no chunks?                │ has chunks
             │               ▼                           ▼
             │           finalize_node          build_context_node  ← token-budget context + sources event
             │                                          │
             │                                   generate_node       ← stream tokens
             │                                          │
             │                                    score_node         ← alignment + confidence
             │                                          │
             │                         ┌────────────────┴───────────────┐
             │                         │ needs judge?                    │ no
             │                         ▼                                 ▼
             │                    judge_node                        verdict_node
             │                         │                                 │
             │                         └───────────────┬────────────────┘
             │                                         │
             │                                   verdict_node
             │                                         │
             └──────────────────────────┬──────────────┘
                                        │
                                  finalize_node          ← done event + memory write-back
                                        │
                                       END

Conditional edges
─────────────────
  after intent_node     → pre_escalation_node  if escalate_now=True and support_mode
                        → memory_node          otherwise

  after flatten_node    → finalize_node        if no chunks retrieved
                        → build_context_node   otherwise

  after score_node      → judge_node           if explicit judge=True OR borderline scores
                        → verdict_node         otherwise

  after verdict_node    → finalize_node        (always — verdict may include escalation flag)
  after pre_escalation  → finalize_node        (always — pipeline is done)
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, START, END  # type: ignore

from .state import RAGState
from .nodes import (
    intent_node,
    memory_node,
    retrieve_node,
    flatten_node,
    build_context_node,
    generate_node,
    score_node,
    judge_node,
    verdict_node,
    pre_escalation_node,
    finalize_node,
)


# ─────────────────────────────────────────────────────────────────────────────
# Conditional edge functions
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_intent(state: RAGState) -> str:
    """Route to pre-escalation when the intent classifier flags an immediate
    escalation, otherwise proceed to memory loading."""
    intent = state.get("intent_result")
    if (
        state.get("support_mode")
        and intent is not None
        and getattr(intent, "escalate_now", False)
    ):
        return "pre_escalation_node"
    return "memory_node"


def _route_after_flatten(state: RAGState) -> str:
    """Skip the generation pipeline entirely when retrieval returned nothing."""
    if not state.get("gen_chunks"):
        return "finalize_node"
    return "build_context_node"


def _route_after_score(state: RAGState) -> str:
    """
    Fire the full LLM judge when:
      - The caller explicitly requested it (judge=True), or
      - The fast-path scores are borderline (possible quality issue).
    Otherwise go straight to verdict aggregation.
    """
    if state.get("judge"):
        return "judge_node"

    alignment  = state.get("alignment_score")
    confidence = state.get("confidence_score")

    borderline_alignment  = alignment  is not None and alignment  < 0.65
    borderline_confidence = confidence is not None and confidence <= 0.72

    if borderline_alignment or borderline_confidence:
        return "judge_node"

    return "verdict_node"


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_graph() -> Any:
    """
    Compile and return the LangGraph StateGraph for the RAG pipeline.

    Returns a compiled graph with:
      - .ainvoke(state)  — for POST /ask  (non-streaming, returns final state)
      - .astream(state, stream_mode="custom")  — for POST /ask/stream (SSE)
    """
    g = StateGraph(RAGState)

    # ── Register nodes ────────────────────────────────────────────────────
    g.add_node("intent_node",         intent_node)
    g.add_node("pre_escalation_node", pre_escalation_node)
    g.add_node("memory_node",         memory_node)
    g.add_node("retrieve_node",       retrieve_node)
    g.add_node("flatten_node",        flatten_node)
    g.add_node("build_context_node",  build_context_node)
    g.add_node("generate_node",       generate_node)
    g.add_node("score_node",          score_node)
    g.add_node("judge_node",          judge_node)
    g.add_node("verdict_node",        verdict_node)
    g.add_node("finalize_node",       finalize_node)

    # ── Entry point ───────────────────────────────────────────────────────
    g.add_edge(START, "intent_node")

    # ── Conditional: after intent ─────────────────────────────────────────
    g.add_conditional_edges(
        "intent_node",
        _route_after_intent,
        {
            "pre_escalation_node": "pre_escalation_node",
            "memory_node":         "memory_node",
        },
    )

    # ── Linear: memory → retrieve → flatten ──────────────────────────────
    g.add_edge("memory_node",   "retrieve_node")
    g.add_edge("retrieve_node", "flatten_node")

    # ── Conditional: after flatten ────────────────────────────────────────
    g.add_conditional_edges(
        "flatten_node",
        _route_after_flatten,
        {
            "finalize_node":      "finalize_node",
            "build_context_node": "build_context_node",
        },
    )

    # ── Linear: build_context → generate → score ─────────────────────────
    g.add_edge("build_context_node", "generate_node")
    g.add_edge("generate_node",      "score_node")

    # ── Conditional: after score ──────────────────────────────────────────
    g.add_conditional_edges(
        "score_node",
        _route_after_score,
        {
            "judge_node":   "judge_node",
            "verdict_node": "verdict_node",
        },
    )

    # ── Judge → verdict → finalize ────────────────────────────────────────
    g.add_edge("judge_node",          "verdict_node")
    g.add_edge("verdict_node",        "finalize_node")
    g.add_edge("pre_escalation_node", "finalize_node")

    # ── Terminal ──────────────────────────────────────────────────────────
    g.add_edge("finalize_node", END)

    return g.compile()


# Module-level singleton — compiled once at import time
rag_graph = build_rag_graph()

__all__ = ["rag_graph", "build_rag_graph"]
