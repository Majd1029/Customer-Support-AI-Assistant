"""
state.py — Typed state schema for the LangGraph RAG pipeline.

All fields use total=False so partial updates (returning only changed keys
from a node) merge correctly with the existing state without overwriting
fields that the current node didn't touch.
"""
from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class RAGState(TypedDict, total=False):

    # ── Request inputs ────────────────────────────────────────────────────
    question:       str
    session_id:     Optional[str]
    user_id:        Optional[str]
    is_admin:       bool
    collection:     str

    # Retrieval options
    limit:          int
    context_window: int
    rerank:         bool
    use_hyde:       bool
    mmr:            bool
    mmr_lambda:     float
    decompose:      bool
    multi_hop:      bool
    min_score:      Optional[float]
    filters:        Optional[dict]
    source_filter:  Optional[Any]   # SourceFilter dataclass

    # Generation options
    max_tokens:     int
    language_hint:  str

    # Feature flags
    support_mode:   bool
    persona:        str
    memory_enabled: bool
    judge:          bool
    reference:      Optional[str]
    score_chunks:   bool

    # ── Runtime singletons (injected by the FastAPI handler) ─────────────
    # These are passed into the initial state dict by api.py so the graph
    # nodes never need to import from api.py directly.
    groq_client:    Optional[Any]   # groq.Groq instance
    qdrant_client:  Optional[Any]   # Qdrant QdrantClient instance

    # ── Pipeline state (written by individual nodes) ─────────────────────
    intent_result:       Optional[Any]   # IntentResult dataclass
    memory_context:      Optional[Any]   # MemoryContext dataclass
    retrieval_question:  str             # possibly rewritten by memory layer
    rbac_filters:        dict            # merged owner_id / user-supplied filters

    retrieval_result:    Optional[Any]   # RetrievalResult dataclass
    retrieval_ms:        float

    gen_chunks:          list            # flattened, deduplicated chunks for generation
    context_text:        str             # assembled context block (ContextBuilder output)
    sources:             list            # deduplicated source metadata
    ordered_chunks:      list            # token-budget-trimmed chunks the LLM will see
    tokens_in_context:   int

    # ── Generation output ─────────────────────────────────────────────────
    answer:         str
    backend:        str
    model:          str
    generation_ms:  float
    token_counts:   Optional[dict]   # {"prompt": n, "completion": n, "total": n}
    no_answer:      bool
    citation_count: int

    # ── Evaluation ───────────────────────────────────────────────────────
    alignment_score:  Optional[float]
    confidence_score: Optional[float]
    judge_result:     Optional[Any]    # JudgeResult dataclass
    eval_verdict:     Optional[str]    # "pass" | "fail" | "off_topic" | "low_confidence"
    eval_feedback:    Optional[str]

    # ── Escalation ───────────────────────────────────────────────────────
    escalated:          bool
    escalation_result:  Optional[Any]   # EscalationDecision dataclass

    # ── Timing ───────────────────────────────────────────────────────────
    start_time:   float
    elapsed_ms:   float
