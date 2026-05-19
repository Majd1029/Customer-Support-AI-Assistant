"""
answer_generator.py — Answer Generation for Secure AI Assistant RAG Pipeline
==============================================================================

Backend priority:
  1. Groq API  — meta-llama/llama-4-scout-17b-16e-instruct  (primary, fast, large context)
  2. Ollama    — qwen2.5:7b  (local fallback when Groq is unavailable / key not set)

Sits between the retrieval layer (file_preparation/retrieval/retriever.py) and
the API endpoints (/ask, /ask/stream).

Features
--------
  - Automatic backend selection: Groq first, Ollama fallback
  - Uses the `ollama` Python library (streaming, token counts, keep_alive)
  - `repeat_penalty`, `top_k`, `num_ctx` Ollama options wired in
  - Health check on Ollama fallback path
  - Token counts (prompt + completion) captured from every response
  - `no_answer` detection via sentinel regex
  - Citation-only source filtering
  - Per-type chunk labels: [TABLE], [IMAGE CAPTION]
  - Context block framed with --- CONTEXT EXCERPTS --- / --- END OF CONTEXT ---
  - 11-rule system prompt (grounding, citations, no hallucination, language, etc.)
  - Context token budget 8 000; token estimation improved to 3.8 chars/tok
  - AnswerResult extended with `no_answer: bool` and `token_counts: dict | None`

Usage
-----
    from file_preparation.generation.answer_generator import AnswerGenerator, GenerationConfig

    gen = AnswerGenerator()
    result = gen.generate(question, chunks)    # RetrievedChunk list from retrieve_evidence()

    # Streaming (yields str tokens)
    for token in gen.stream(question, chunks):
        print(token, end="", flush=True)

Environment variables (.env)
-----------------------------
    GROQ_API_KEY           = gsk_...         (required for primary Groq backend)
    GROQ_GENERATION_MODEL  = meta-llama/llama-4-scout-17b-16e-instruct  (override Groq model)
    OLLAMA_BASE_URL        = http://localhost:11434
    QWEN_MODEL             = qwen2.5:7b      (override Ollama fallback model)
    ANSWER_MAX_TOKENS      = 1500
    CONTEXT_TOKEN_CAP      = 8000
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

_OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_QWEN_MODEL:      str = os.getenv("QWEN_MODEL",       "qwen2.5:7b")
_ANSWER_MAX_TOKENS: int = int(os.getenv("ANSWER_MAX_TOKENS", "1500"))
_CONTEXT_TOKEN_CAP: int = int(os.getenv("CONTEXT_TOKEN_CAP", "8000"))

# Ollama-specific generation options
_OLLAMA_NUM_CTX        = 12288  # context window fed to Qwen2.5
# Must cover: CONTEXT_TOKEN_CAP (8000) + ANSWER_MAX_TOKENS (1500) + system prompt (~400)
# ≈ 9900 tokens minimum. 12288 gives a comfortable margin; Qwen2.5-7B supports 32K.
_OLLAMA_REPEAT_PENALTY = 1.1    # penalise repetition
_OLLAMA_TOP_K          = 40     # top-k sampling
_OLLAMA_KEEP_ALIVE     = "30m"  # keep model in VRAM/RAM between requests

# No-answer sentinel — matches rule 4 in the system prompt.
# Use a word-boundary regex to avoid false positives when the model quotes or
# paraphrases the sentinel within an actual answer (e.g. "Unlike a system that
# doesn't have enough information, this one does...").
_NO_ANSWER_SENTINEL = "i don't have enough information in the provided context"  # kept for logging
# Expanded regex: catches common Qwen2.5 refusal phrasings beyond the canonical sentinel.
# Word-boundary anchors prevent false positives inside real answers that quote these phrases.
_NO_ANSWER_RE = re.compile(
    r"\b("
    # Canonical: "don't have enough information"
    r"don['\u2019]?t\s+have\s+enough\s+information"
    # "cannot answer / determine / find / provide"
    r"|cannot\s+(?:answer|determine|find|provide|give)"
    # "no information available / provided / in the context"
    r"|no\s+(?:relevant\s+)?information\s+(?:available|provided|found|in\s+the)"
    # "not mentioned / discussed / covered / found in the (context|provided context)"
    r"|not\s+(?:mentioned|discussed|covered|addressed|found)\s+in\s+the"
    # "the context does not contain / include / mention / provide"
    r"|(?:the\s+)?(?:provided\s+)?context\s+does\s+not\s+(?:contain|include|mention|provide)"
    # "i am unable to answer / determine"
    r"|i\s+am\s+unable\s+to\s+(?:answer|determine|provide)"
    r")\b",
    re.IGNORECASE,
)

# Accurate chars-per-token ratio for Qwen2.5 (3.8 > the naive 4.0)
_CHARS_PER_TOKEN: float = 3.8

# Per-chunk-type content label prepended in the context block
_CHUNK_TYPE_LABELS: dict[str, str] = {
    "text":  "",
    "table": "[TABLE] ",
    "image": "[IMAGE CAPTION] ",
}

# Citation pattern: matches [Source: filename, page X] or [Source: filename]
_CITATION_RE = re.compile(
    r"\[Source:\s*([^,\]]+?)(?:,\s*page\s+(\d+))?\]",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Shared token counter (re-uses project helper when available)
# ---------------------------------------------------------------------------
try:
    import tiktoken as _tiktoken
    _enc = _tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))

except Exception:
    logger.warning("tiktoken unavailable — using len(text)/3.8 fallback for token counting")

    def _count_tokens(text: str) -> int:   # type: ignore[misc]
        return max(1, int(len(text) / _CHARS_PER_TOKEN))


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

def _source_stem(name: str) -> str:
    """Return the file stem (name without extension) in lower-case."""
    return Path(name).stem.lower()


def _extract_cited_sources(answer: str, used_chunks: list[dict]) -> list[dict]:
    """
    Parse [Source: filename, page X] citations from the generated answer
    and return the matching chunk dicts (deduplicated, order-preserving).

    Handles:
      [Source: report.pdf, page 3]
      [Source: report.pdf]                  (matches any page from that source)
      [Source: report, page 3]              (stem fallback — model dropped extension)
      Multiple citations in one sentence.
    """
    cited_keys: set[tuple] = set()   # (lowered_citation_text, page | None)
    for m in _CITATION_RE.finditer(answer):
        source = m.group(1).strip()
        page   = int(m.group(2)) if m.group(2) else None
        cited_keys.add((source.lower(), page))

    matched:    list[dict]   = []
    seen_keys:  set[tuple]   = set()

    for chunk in used_chunks:
        chunk_source = (chunk.get("source") or "").lower()
        chunk_stem   = _source_stem(chunk_source)
        chunk_page   = chunk.get("page_start")
        for cited_source, cited_page in cited_keys:
            # Exact filename match OR stem-only match (model dropped the extension)
            if chunk_source == cited_source or chunk_stem == _source_stem(cited_source):
                if cited_page is None or cited_page == chunk_page:
                    key = (chunk_source, chunk_page)
                    if key not in seen_keys:
                        matched.append(chunk)
                        seen_keys.add(key)
                    break

    return matched


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """Tunable parameters for a single generation call."""
    max_tokens:           int   = _ANSWER_MAX_TOKENS
    temperature:          float = 0.15         # low for factual grounding
    top_p:                float = 0.90
    context_token_cap:    int   = _CONTEXT_TOKEN_CAP
    cite_sources:         bool  = True
    language_hint:        str   = ""           # e.g. "fr" — instructs model to reply in that language
    conversation_summary: str   = ""           # rolling summary of old turns (Sprint 3)
    user_facts:           str   = ""           # [USER FACTS] block (Sprint 4) — key-value user facts
    user_preferences:     str   = ""           # [USER PREFERENCES] block (Sprint 5) — semantic preferences
    persona:              str   = ""           # persona/tone preamble prepended before STRICT RULES


@dataclass
class AnswerResult:
    """Final output of the generation step."""
    question:         str
    answer:           str
    sources:          list[dict]          # chunks actually cited (filtered by [Source:] references)
    chunks_used:      int
    tokens_in_context: int
    elapsed_ms:       float              # total wall-clock time for generate() (ms)
    model:            str
    backend:          str                # "ollama"
    confidence:       Optional[float] = None
    hops:             int               = 1
    no_answer:        bool              = False   # True when model hit the no-info sentinel
    token_counts:     Optional[dict]    = None    # {"prompt": n, "completion": n, "total": n}

    # ── Latency breakdown (ms) ──────────────────────────────────────────────────
    # retrieval_ms and rerank_ms are filled by the caller (api.py / test script)
    # after retrieve_evidence() completes; generation_ms is set by generate().
    retrieval_ms:     Optional[float]   = None   # time spent in hybrid RRF + reranking
    generation_ms:    Optional[float]   = None   # time spent in Ollama inference only

    # ── Pipeline quality metrics ────────────────────────────────────────────────
    citation_count:       int           = 0      # number of unique [Source:] citations found
    answer_length_chars:  int           = 0      # character length of the final answer
    context_utilisation:  float         = 0.0    # tokens_in_context / context_token_cap ∈ [0,1]

    # ── Evidence traceability ───────────────────────────────────────────────────
    # The exact chunk dicts that were placed in the LLM context (after token-budget
    # trimming).  Use these — not the full retrieval.chunks — when scoring
    # answer groundedness, so the scorer only evaluates evidence the model saw.
    chunks_in_context:    list          = field(default_factory=list)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

class ContextBuilder:
    """
    Converts a list of chunk dicts into:
      - A numbered context block for the system prompt
      - A deduplicated sources list
      - The ordered list of chunks as they appear in the prompt (for citation mapping)

    Respects CONTEXT_TOKEN_CAP by skipping chunks that don't fit (not stopping
    early — a later smaller chunk may still fit).
    """

    def build(
        self,
        chunks: list[dict],
        cap:    int = _CONTEXT_TOKEN_CAP,
    ) -> tuple[str, list[dict], int, list[dict]]:
        """
        Returns:
            context_text    — formatted string ready to inject into the prompt
            sources         — deduplicated source metadata list
            tokens_used     — actual token count of context_text
            ordered_chunks  — chunks as numbered in the context block (for citation mapping)
        """
        validated = self._validate_and_sort(chunks)
        kept, tokens_used = self._trim_to_budget(validated, cap)
        context_text = self._format_context(kept)
        sources      = self._build_sources(kept)
        return context_text, sources, tokens_used, kept

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_and_sort(self, chunks: list[dict]) -> list[dict]:
        """
        Accept both raw Qdrant payload dicts and RetrievedChunk dataclass dicts.
        Normalise to a common shape and sort: primary hop-1 chunks first, neighbours last.
        """
        normalised: list[dict] = []
        for c in chunks:
            if hasattr(c, "__dict__"):
                c = c.__dict__

            content: str  = c.get("content") or c.get("payload", {}).get("content", "")
            if not content.strip():
                continue

            metadata: dict = c.get("metadata") or c.get("payload") or {}
            hop_val = c.get("hop")
            if hop_val is None:
                hop_val = metadata.get("hop")
            hop: int = int(hop_val) if hop_val is not None else 1

            # Default to False so that chunks missing an explicit `primary` flag
            # (e.g. raw neighbor dicts from get_neighbors()) are treated as
            # context-expansion chunks rather than primary retrieval hits.
            # Callers that produce primary chunks always set primary=True explicitly.
            prim_val = c.get("primary")
            if prim_val is None:
                prim_val = metadata.get("primary")
            primary: bool = bool(prim_val) if prim_val is not None else False

            score_val = c.get("score")
            score: float = float(score_val) if score_val is not None else 0.0

            normalised.append({
                "content":     content.strip(),
                "chunk_id":    c.get("chunk_id") or metadata.get("chunk_id", ""),
                "source":      metadata.get("source", "unknown"),
                "page_start":  metadata.get("page_start", 1),
                "section":     metadata.get("section", ""),
                "type":        metadata.get("type", "text"),
                "retrieval":   c.get("retrieval") or metadata.get("retrieval", ""),
                "subject":     metadata.get("subject", ""),       # EML email subject
                "score":       score,
                "hop":         hop,
                "primary":     primary,
                # G6: prefer pre-computed token_count from the payload/metadata;
                # only call _count_tokens() as a fallback to avoid unnecessary
                # tiktoken encoding of chunks already counted at index time.
                "token_count": (
                    int(metadata.get("token_count", 0))
                    or int(c.get("token_count", 0))
                    or _count_tokens(content)
                ),
                "chunk_index": metadata.get("chunk_index", 0),
            })

        # Primary + hop-1 first; neighbours + hop-2 last; within group sort by score desc
        normalised.sort(key=lambda x: (not x["primary"], x["hop"], -x["score"]))
        return normalised

    # Overhead added per chunk for its header line, e.g.:
    #   "[1] (filename: report.pdf | page: 3 | section: Results | via: both)"
    # Measured at ~10–18 tokens depending on field lengths; 15 is a safe average.
    _HEADER_TOKENS_PER_CHUNK: int = 15
    # Fixed token overhead for the two section markers:
    #   "--- CONTEXT EXCERPTS ---" and "--- END OF CONTEXT ---" + surrounding newlines
    _CONTEXT_MARKER_TOKENS:   int = 20

    def _trim_to_budget(
        self, chunks: list[dict], cap: int
    ) -> tuple[list[dict], int]:
        """
        Greedy: skip chunks that don't fit rather than stopping the loop.
        Guarantees at least one chunk is always kept.
        Secondary trim removes from the end if accumulation still exceeds cap.

        Accounts for per-chunk header overhead and fixed context-marker tokens
        so the assembled context block never silently exceeds the LLM's context cap.

        Per-chunk cap: individual chunks are capped at cap // 2 tokens.  This
        prevents a single oversized chunk from monopolising the entire budget
        and crowding out all other evidence.
        """
        chunk_cap = cap // 2   # no single chunk may use more than half the budget
        total = self._CONTEXT_MARKER_TOKENS   # start with fixed marker overhead
        kept:  list[dict] = []
        for chunk in chunks:
            # Clamp the chunk's token count to chunk_cap for budget arithmetic.
            # (The actual content is passed unchanged — only the budget estimate
            # is capped so we don't incorrectly skip the chunk entirely.)
            effective_tokens = min(chunk["token_count"], chunk_cap)
            ct = effective_tokens + self._HEADER_TOKENS_PER_CHUNK
            if total + ct > cap:
                if not kept:          # always keep at least one
                    kept.append(chunk)
                    total += ct
                continue
            kept.append(chunk)
            total += ct
        while total > cap and len(kept) > 1:
            removed = kept.pop()
            effective_removed = min(removed["token_count"], chunk_cap)
            total  -= effective_removed + self._HEADER_TOKENS_PER_CHUNK
        logger.debug(f"Context: {len(kept)}/{len(chunks)} chunks, ~{total} tokens (incl. headers)")
        return kept, total

    def _format_context(self, chunks: list[dict]) -> str:
        """
        Build a numbered evidence block framed with section markers:

            --- CONTEXT EXCERPTS ---

            [1] (filename: report.pdf | page: 3 | section: Results | via: both)
            <content>

            [2] (filename: memo.docx | page: 1 | email subject: Q3 Update)
            [TABLE] | col1 | col2 | ...

            --- END OF CONTEXT ---
        """
        lines: list[str] = ["--- CONTEXT EXCERPTS ---"]
        for i, c in enumerate(chunks, 1):
            lines.append(self._build_chunk_header(i, c))
            label   = _CHUNK_TYPE_LABELS.get(c.get("type", "text") or "text", "")
            lines.append(f"{label}{c['content']}")
            lines.append("")
        lines.append("--- END OF CONTEXT ---")
        return "\n".join(lines).strip()

    def _build_chunk_header(self, idx: int, chunk: dict) -> str:
        """Pipe-separated metadata header — matches the format the model uses
        when constructing [Source: filename, page X] citations."""
        parts: list[str] = []
        if chunk.get("source"):
            parts.append(f"filename: {chunk['source']}")
        if chunk.get("page_start") is not None:
            parts.append(f"page: {chunk['page_start']}")
        if chunk.get("section"):
            parts.append(f"section: {chunk['section']}")
        if chunk.get("retrieval"):
            parts.append(f"via: {chunk['retrieval']}")
        if chunk.get("subject"):
            parts.append(f"email subject: {chunk['subject']}")

        meta = f" ({' | '.join(parts)})" if parts else ""
        return f"[{idx}]{meta}"

    def _build_sources(self, chunks: list[dict]) -> list[dict]:
        """Deduplicated source list (one entry per unique source+page pair)."""
        seen:    set[tuple]  = set()
        sources: list[dict]  = []
        for c in chunks:
            key = (c["source"], c["page_start"])
            if key not in seen:
                seen.add(key)
                sources.append({
                    "chunk_id":   c.get("chunk_id", ""),
                    "source":     c["source"],
                    "page_start": c["page_start"],
                    "section":    c["section"],
                    "score":      round(c["score"], 4),
                    "hop":        c["hop"],
                    "content":    (c.get("content") or "")[:400],
                })
        return sources


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Persona constants
# ---------------------------------------------------------------------------

CUSTOMER_SUPPORT_PERSONA = """\
You are a helpful, empathetic, and professional customer support assistant.
Your tone is warm, patient, and solution-focused at all times.
When a customer is frustrated, acknowledge their feelings before answering.
Keep answers clear and concise — avoid technical jargon unless the customer uses it.
Always end with an offer to help further if needed.
"""

# ---------------------------------------------------------------------------
# Language instructions
# ---------------------------------------------------------------------------

_LANG_INSTRUCTIONS: dict[str, str] = {
    "fr": "- Reply in French.",
    "ar": "- Reply in Arabic.",
    "de": "- Reply in German.",
    "es": "- Reply in Spanish.",
    "zh": "- Reply in Chinese.",
    "it": "- Reply in Italian.",
    "pt": "- Reply in Portuguese.",
    "nl": "- Reply in Dutch.",
    "ru": "- Reply in Russian.",
    "ja": "- Reply in Japanese.",
    "ko": "- Reply in Korean.",
}

_SYSTEM_PROMPT_TEMPLATE = """\
You are a precise, comprehensive question-answering assistant.
You are given numbered context excerpts retrieved from a document base.
Your task is to answer the user's question using ONLY the information in those excerpts.

STRICT RULES:
1. Ground every claim in the provided context. Do NOT use prior knowledge or external facts.
2. Cite sources inline using this exact format: [Source: filename, page X]
   - For multiple sources on one claim: [Source: file1.pdf, page 3] [Source: file2.pdf, page 5]
   - For chunks without a page number: [Source: filename]
3. Every factual claim must have a citation immediately after it.
4. If the context does not contain enough information, respond EXACTLY:
   "I don't have enough information in the provided context to answer this question."
5. Be thorough and detailed. Cover all relevant aspects from the context.
   Explain concepts fully, include examples and numbers from the text.
6. Avoid filler phrases like "Based on the context..." — go straight to the answer.
7. Preserve exact numbers, names, dates, and technical terms from the context.
8. If excerpts contradict each other, note the contradiction explicitly.
9. Preserve the original language of the question in your answer.{lang_rule}
10. [TABLE] excerpts contain structured data — extract exact values from them.
11. Structure long answers with paragraphs covering different aspects of the question.\
"""

# Variant without citation rules — used when cite_sources=False.
_SYSTEM_PROMPT_NO_CITE_TEMPLATE = """\
You are a precise, comprehensive question-answering assistant.
You are given numbered context excerpts retrieved from a document base.
Your task is to answer the user's question using ONLY the information in those excerpts.

STRICT RULES:
1. Ground every claim in the provided context. Do NOT use prior knowledge or external facts.
2. If the context does not contain enough information, respond EXACTLY:
   "I don't have enough information in the provided context to answer this question."
3. Be thorough and detailed. Cover all relevant aspects from the context.
   Explain concepts fully, include examples and numbers from the text.
4. Avoid filler phrases like "Based on the context..." — go straight to the answer.
5. Preserve exact numbers, names, dates, and technical terms from the context.
6. If excerpts contradict each other, note the contradiction explicitly.
7. Preserve the original language of the question in your answer.{lang_rule}
8. [TABLE] excerpts contain structured data — extract exact values from them.
9. Structure long answers with paragraphs covering different aspects of the question.\
"""


def _build_system_prompt(
    language_hint: str = "",
    cite_sources:  bool = True,
    persona:       str  = "",
) -> str:
    lang_rule = ""
    if language_hint:
        if language_hint in _LANG_INSTRUCTIONS:
            lang_rule = f"\n   {_LANG_INSTRUCTIONS[language_hint]}"
        else:
            # Generic fallback for language codes not in the lookup dict
            # (e.g. "tr", "pl", "sv" …) — the model understands ISO 639-1 codes.
            lang_rule = f"\n   - Reply in the language identified by ISO 639-1 code '{language_hint}'."
    template = _SYSTEM_PROMPT_TEMPLATE if cite_sources else _SYSTEM_PROMPT_NO_CITE_TEMPLATE
    base = template.format(lang_rule=lang_rule)
    if persona:
        # Persona preamble goes before STRICT RULES so the model adopts the
        # tone/role first, then the grounding rules constrain factual accuracy.
        return f"{persona.rstrip()}\n\n{base}"
    return base


def _build_prompt(
    question: str,
    context:  str,
    config:   GenerationConfig,
) -> tuple[str, str]:
    """
    Returns (system_prompt, user_message).

    Memory blocks are prepended to the user message in this order:
      1. [USER FACTS]          (Sprint 4) — persistent key-value facts about the user
      2. [USER PREFERENCES]    (Sprint 5) — semantically recalled preference statements
      3. [CONVERSATION SUMMARY](Sprint 3) — rolling summary of older absorbed turns
      4. Context excerpts
      5. Question
    """
    system = _build_system_prompt(
        config.language_hint,
        cite_sources=config.cite_sources,
        persona=config.persona,
    )

    # Sprint 4 — structured user facts (from PostgreSQL / in-process store)
    facts_block = ""
    if config.user_facts:
        facts_block = f"{config.user_facts.strip()}\n\n"

    # Sprint 5 — semantically recalled preferences (from Qdrant)
    prefs_block = ""
    if config.user_preferences:
        prefs_block = f"{config.user_preferences.strip()}\n\n"

    # Sprint 3 — rolling summary of absorbed older turns
    summary_block = ""
    if config.conversation_summary:
        summary_block = (
            "[CONVERSATION SUMMARY]\n"
            f"{config.conversation_summary.strip()}\n"
            "[END SUMMARY]\n\n"
        )

    preamble = f"{facts_block}{prefs_block}{summary_block}"

    if config.cite_sources:
        user = (
            f"{preamble}"
            f"{context}\n\n"
            f"Question: {question.strip()}\n\n"
            "Provide a thorough, well-cited answer using the format "
            "[Source: filename, page X] for citations:"
        )
    else:
        user = (
            f"{preamble}"
            f"{context}\n\n"
            f"Question: {question.strip()}\n\n"
            "Provide a thorough, grounded answer based on the context above:"
        )
    return system, user


# ---------------------------------------------------------------------------
# Backend: Ollama (Qwen2.5-7B) — uses the `ollama` Python library
# ---------------------------------------------------------------------------

class OllamaBackend:
    """
    Calls the local Ollama API via the `ollama` Python library.
    Adds keep_alive, repeat_penalty, top_k, num_ctx, and health check
    over the previous httpx-based implementation.
    """

    def __init__(
        self,
        base_url: str = _OLLAMA_BASE_URL,
        model:    str = _QWEN_MODEL,
    ) -> None:
        import ollama as _ollama_lib  # lazy — only fails if lib not installed
        self._lib    = _ollama_lib
        self._client = _ollama_lib.Client(host=base_url)
        self.model   = model
        self._healthy: Optional[bool] = None

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if Ollama is running and the model is pulled.
        After the first successful health check the result is cached so
        subsequent calls skip the HTTP round-trip entirely.
        """
        if self._healthy:           # cache hit — no HTTP call needed
            return True
        try:
            available = [m["model"] for m in self._client.list()["models"]]
            base_tags = [m.split(":")[0] for m in available]
            target    = self.model.split(":")[0]
            ok = target in base_tags or any(a.startswith(target) for a in available)
            if ok:
                self._healthy = True   # warm the cache for _ensure_healthy() too
            return ok
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False

    def _ensure_healthy(self) -> None:
        """Verify Ollama is reachable and the model is available. Result cached."""
        if self._healthy:
            return
        try:
            available  = [m["model"] for m in self._client.list()["models"]]
            base_tags  = [m.split(":")[0] for m in available]
            target     = self.model.split(":")[0]
            if target not in base_tags and not any(a.startswith(target) for a in available):
                raise RuntimeError(
                    f"Model '{self.model}' not found in Ollama.\n"
                    f"Pull it with: ollama pull {self.model}\n"
                    f"Available: {available}"
                )
            self._healthy = True
            logger.info(f"  Ollama health check OK — model '{self.model}' is ready.")
        except RuntimeError:
            raise
        except Exception as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {_OLLAMA_BASE_URL}. "
                f"Is `ollama serve` running?\n  → {exc}"
            ) from exc

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        system:  str,
        user:    str,
        config:  GenerationConfig,
        *,
        history: list[dict] | None = None,
    ) -> tuple[str, dict]:
        """
        Returns (answer_text, token_counts).
        token_counts = {"prompt": n, "completion": n, "total": n}

        When `history` is provided (list of {"role": ..., "content": ...} dicts
        from the conversation buffer), those turns are inserted between the system
        message and the current user message so the model sees prior context.
        """
        self._ensure_healthy()
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})
        response = self._client.chat(
            model      = self.model,
            messages   = messages,
            options    = self._options(config),
            keep_alive = _OLLAMA_KEEP_ALIVE,
        )
        answer  = response["message"]["content"].strip()
        tc      = self._extract_token_counts(response)
        return answer, tc

    def stream(
        self,
        system:  str,
        user:    str,
        config:  GenerationConfig,
        *,
        history: list[dict] | None = None,
    ) -> Generator[str, None, None]:
        """
        Yield tokens; capture token counts from the final done chunk.

        After the generator is exhausted, `self.last_token_counts` holds
        {"prompt": N, "completion": N, "total": N} — callers can read this
        to report token usage in SSE done events or log output.

        When `history` is provided, those turns are inserted between system
        and user messages — same behaviour as generate().
        """
        self._ensure_healthy()
        self.last_token_counts: dict = {}   # reset before each stream
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})
        stream = self._client.chat(
            model      = self.model,
            messages   = messages,
            options    = self._options(config),
            stream     = True,
            keep_alive = _OLLAMA_KEEP_ALIVE,
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token
            # Capture token counts from the final done chunk (no yield — metadata only)
            if chunk.get("done"):
                self.last_token_counts = self._extract_token_counts(chunk)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _options(self, config: GenerationConfig) -> dict:
        return {
            "num_ctx":        _OLLAMA_NUM_CTX,
            "temperature":    config.temperature,
            "top_p":          config.top_p,
            "top_k":          _OLLAMA_TOP_K,
            "repeat_penalty": _OLLAMA_REPEAT_PENALTY,
            "num_predict":    config.max_tokens,
        }

    @staticmethod
    def _extract_token_counts(response: Any) -> dict:
        p = response.get("prompt_eval_count", 0) or 0
        c = response.get("eval_count", 0)        or 0
        return {"prompt": p, "completion": c, "total": p + c}


# ---------------------------------------------------------------------------
# Groq Backend (primary — meta-llama/llama-4-scout-17b-16e-instruct)
# ---------------------------------------------------------------------------

_GROQ_MODEL_DEFAULT: str = os.getenv("GROQ_GENERATION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
"""
Primary Groq model for answer generation.
Override with GROQ_GENERATION_MODEL in .env.
llama-4-scout is a fast, high-quality model well-suited for RAG answers.
"""

_GROQ_STREAM_TIMEOUT: float = 120.0


class GroqBackend:
    """
    Groq-hosted LLM fallback for /ask and /ask/stream when Ollama is unavailable.

    Uses groq.chat.completions.create() for blocking generation and
    groq.chat.completions.create(stream=True) for true token streaming —
    the Groq path is now identical in user experience to the Ollama path.

    Token counts come from the final stream chunk's `usage` field.
    `last_token_counts` is populated after stream() is exhausted, matching
    the same pattern as OllamaBackend so callers don't need branching.
    """

    def __init__(self, api_key: str | None = None, model: str = _GROQ_MODEL_DEFAULT) -> None:
        self._api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model    = model
        self.last_token_counts: dict = {}

    def is_available(self) -> bool:
        """Return True if a GROQ_API_KEY is configured."""
        return bool(self._api_key)

    def _get_client(self):
        """Lazy-import groq and return a configured client."""
        try:
            import groq as _groq_lib
        except ImportError:
            raise RuntimeError(
                "groq package not installed — run: pip install groq"
            )
        return _groq_lib.Groq(api_key=self._api_key)

    def generate(
        self,
        system:  str,
        user:    str,
        config:  "GenerationConfig",
        *,
        history: list[dict] | None = None,
    ) -> tuple[str, dict]:
        """
        Blocking Groq generation.

        Returns (answer_text, token_counts).
        token_counts = {"prompt": n, "completion": n, "total": n}
        """
        client = self._get_client()
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})

        response = client.chat.completions.create(
            model       = self.model,
            messages    = messages,
            temperature = config.temperature,
            max_tokens  = config.max_tokens,
            top_p       = config.top_p,
            timeout     = _GROQ_STREAM_TIMEOUT,
        )
        answer = (response.choices[0].message.content or "").strip()
        tc = self._extract_token_counts(response.usage)
        return answer, tc

    def stream(
        self,
        system:  str,
        user:    str,
        config:  "GenerationConfig",
        *,
        history: list[dict] | None = None,
    ) -> Generator[str, None, None]:
        """
        Streaming Groq generation — yields str tokens.

        After the generator is exhausted, `self.last_token_counts` holds
        {"prompt": N, "completion": N, "total": N} from the final chunk's
        `x_groq.usage` field — same interface as OllamaBackend.
        """
        client = self._get_client()
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})

        self.last_token_counts = {}  # reset before each stream

        stream = client.chat.completions.create(
            model       = self.model,
            messages    = messages,
            temperature = config.temperature,
            max_tokens  = config.max_tokens,
            top_p       = config.top_p,
            stream      = True,
            timeout     = _GROQ_STREAM_TIMEOUT,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
            # Groq puts usage in the final chunk's x_groq attribute
            if hasattr(chunk, "x_groq") and chunk.x_groq and hasattr(chunk.x_groq, "usage"):
                usage = chunk.x_groq.usage
                self.last_token_counts = self._extract_token_counts(usage)

    @staticmethod
    def _extract_token_counts(usage) -> dict:
        if usage is None:
            return {}
        p = getattr(usage, "prompt_tokens",     0) or 0
        c = getattr(usage, "completion_tokens", 0) or 0
        return {"prompt": p, "completion": c, "total": p + c}


# ---------------------------------------------------------------------------
# Main AnswerGenerator
# ---------------------------------------------------------------------------

class AnswerGenerator:
    """
    Primary entry point for answer generation.

    Backend priority:
      1. Groq API  — meta-llama/llama-4-scout-17b-16e-instruct  (primary, requires GROQ_API_KEY)
      2. Ollama    — qwen2.5:7b  (local fallback, no API key needed)
    """

    def __init__(self, groq_api_key: str | None = None) -> None:
        self._ollama      = OllamaBackend()
        self._groq        = GroqBackend(api_key=groq_api_key)
        self._ctx_builder = ContextBuilder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        chunks:   list[dict],
        config:   Optional[GenerationConfig] = None,
        *,
        history:  list[dict] | None = None,
    ) -> AnswerResult:
        """
        Synchronous generation with citation-filtered sources.

        Parameters
        ----------
        question : str
        chunks   : list of chunk dicts from retrieve_evidence() or multihop_retrieve()
        config   : GenerationConfig (defaults applied if None)
        history  : prior conversation turns as [{"role": ..., "content": ...}].
                   When provided, these are inserted between the system message
                   and the current user message so the model has conversational
                   context (Sprint 1 memory layer).

        Returns
        -------
        AnswerResult with answer, filtered sources, no_answer flag, token_counts
        """
        config  = config or GenerationConfig()
        backend, backend_name = self._pick_backend()

        context, sources, tokens_used, ordered_chunks = self._ctx_builder.build(
            chunks, cap=config.context_token_cap
        )
        if not context:
            return self._empty_result(question, config)

        system, user = _build_prompt(question, context, config)

        t0 = time.monotonic()
        try:
            answer, token_counts = backend.generate(system, user, config, history=history)
        except Exception as e:
            logger.error(f"Generation failed on {backend_name}: {e}")
            raise
        generation_ms = (time.monotonic() - t0) * 1000
        elapsed = generation_ms  # total elapsed for this call = generation only

        # ── No-answer detection ──────────────────────────────────────────────
        no_answer = bool(_NO_ANSWER_RE.search(answer))

        # ── Citation-only source filtering ───────────────────────────────────
        # Parse [Source: filename, page X] references from the answer.
        # Falls back to the full context sources when the model cited nothing.
        # When no_answer=True, return an empty sources list — the model
        # explicitly said it couldn't answer, so attaching sources is misleading.
        if not no_answer:
            cited_chunks = _extract_cited_sources(answer, ordered_chunks)
            if cited_chunks:
                # Rebuild deduplicated sources from the cited chunks only
                seen: set[tuple] = set()
                sources = []
                for c in cited_chunks:
                    key = (c["source"], c["page_start"])
                    if key not in seen:
                        seen.add(key)
                        sources.append({
                            "chunk_id":   c.get("chunk_id", ""),
                            "source":     c["source"],
                            "page_start": c["page_start"],
                            "section":    c.get("section", ""),
                            "score":      round(c.get("score", 0.0), 4),
                        })
            # else: keep the full context sources as a fallback (model cited nothing)
        else:
            sources = []  # no_answer=True — don't attach sources to a non-answer

        hops = max((c.get("hop", 1) for c in chunks), default=1)

        # ── Pipeline quality metrics ─────────────────────────────────────────
        citation_count      = len(sources)
        answer_length_chars = len(answer)
        context_utilisation = round(tokens_used / max(config.context_token_cap, 1), 3)

        logger.info(
            f"  Generation done via {backend_name}/{backend.model} "
            f"in {generation_ms:.0f}ms | "
            f"ctx={tokens_used}tok ({context_utilisation:.0%}) | "
            f"cited={citation_count} sources | ans={answer_length_chars}ch | "
            f"no_answer={no_answer}"
        )

        return AnswerResult(
            question=question,
            answer=answer,
            sources=sources,
            chunks_used=len(chunks),
            tokens_in_context=tokens_used,
            elapsed_ms=round(elapsed, 1),
            model=backend.model,
            backend=backend_name,
            hops=int(hops),
            no_answer=no_answer,
            token_counts=token_counts if token_counts else None,
            generation_ms=round(generation_ms, 1),
            citation_count=citation_count,
            answer_length_chars=answer_length_chars,
            context_utilisation=context_utilisation,
            chunks_in_context=ordered_chunks,   # trimmed set the LLM actually saw
        )

    def stream(
        self,
        question:         str,
        chunks:           list[dict],
        config:           Optional[GenerationConfig] = None,
        *,
        prebuilt_context: Optional[str] = None,
        history:          list[dict] | None = None,
    ) -> Generator[str, None, None]:
        """
        Streaming generation — yields str tokens.

        Pass `prebuilt_context` (from build_context_metadata) to avoid rebuilding
        the context block a second time; if omitted, build() is called internally.

        Callers (/ask/stream in api.py) should emit a `sources` SSE event using
        build_context_metadata() BEFORE starting to iterate this generator, and
        pass the returned context string here to skip the redundant build pass.

        `history` — prior conversation turns as [{"role": ..., "content": ...}].
        Inserted between system and user messages so the model has conversational
        context (Sprint 1 memory layer).
        """
        config  = config or GenerationConfig()
        backend, backend_name = self._pick_backend()

        if prebuilt_context is not None:
            context = prebuilt_context
        else:
            context, _, _, _ = self._ctx_builder.build(chunks, cap=config.context_token_cap)

        if not context:
            yield "(No relevant context found in the indexed documents.)"
            return

        system, user = _build_prompt(question, context, config)
        logger.info(f"  Streaming answer via {backend_name}/{backend.model}")

        yield from backend.stream(system, user, config, history=history)

    def build_context_metadata(
        self,
        chunks: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> tuple[list[dict], int, str, list[dict]]:
        """
        Pre-flight: build context once and return (sources, tokens_used, context_text, ordered_chunks).

        The returned `context_text` should be passed to stream(..., prebuilt_context=)
        so the context block is not built a second time inside stream().

        `ordered_chunks` is the token-budget-trimmed list of chunks the LLM will
        actually see — use this for confidence scoring rather than the full
        retrieved set (which may be much larger).

        Note: sources are not citation-filtered here (answer not yet generated).
        """
        config = config or GenerationConfig()
        context, sources, tokens, ordered_chunks = self._ctx_builder.build(chunks, cap=config.context_token_cap)
        return sources, tokens, context, ordered_chunks

    def stream_with_sources(
        self,
        question: str,
        chunks:   list[dict],
        config:   Optional[GenerationConfig] = None,
        *,
        history:  list[dict] | None = None,
    ) -> "AnswerResult":
        """
        Stream tokens to stdout, then return a full AnswerResult with
        citation-filtered sources — mirrors the old generator.stream_with_metadata().

        Prints tokens as they arrive, then returns the complete result for
        the caller to display sources and stats.

        `history` — prior conversation turns as [{"role": ..., "content": ...}].
        """
        config = config or GenerationConfig()
        backend, backend_name = self._pick_backend()

        # Fail-fast health check for Ollama (no-op for Groq — it's stateless)
        if isinstance(backend, OllamaBackend):
            backend._ensure_healthy()

        context, _, tokens_used, ordered_chunks = self._ctx_builder.build(
            chunks, cap=config.context_token_cap
        )
        if not context:
            return self._empty_result(question, config)

        system, user = _build_prompt(question, context, config)

        import time as _time
        t0 = _time.monotonic()
        full_tokens: list[str] = []
        token_counts: dict = {}

        for token in backend.stream(system, user, config, history=history):
            if token:
                full_tokens.append(token)
                print(token, end="", flush=True)

        print()  # newline after stream ends
        generation_ms = (_time.monotonic() - t0) * 1000
        token_counts = getattr(backend, "last_token_counts", {})
        elapsed       = generation_ms
        answer        = "".join(full_tokens).strip()
        no_answer     = bool(_NO_ANSWER_RE.search(answer))

        if not no_answer:
            cited_chunks = _extract_cited_sources(answer, ordered_chunks)
            if cited_chunks:
                seen: set[tuple] = set()
                sources: list[dict] = []
                for c in cited_chunks:
                    key = (c["source"], c["page_start"])
                    if key not in seen:
                        seen.add(key)
                        sources.append({
                            "chunk_id":   c.get("chunk_id", ""),
                            "source":     c["source"],
                            "page_start": c["page_start"],
                            "section":    c.get("section", ""),
                            "score":      round(c.get("score", 0.0), 4),
                        })
            else:
                sources = self._ctx_builder._build_sources(ordered_chunks)
        else:
            sources = []

        hops = max((c.get("hop", 1) for c in chunks), default=1)

        citation_count      = len(sources)
        answer_length_chars = len(answer)
        context_utilisation = round(tokens_used / max(config.context_token_cap, 1), 3)

        logger.info(
            f"  Streaming done via {backend_name}/{backend.model} "
            f"in {generation_ms:.0f}ms | ctx={tokens_used}tok ({context_utilisation:.0%}) | "
            f"cited={citation_count} sources | ans={answer_length_chars}ch | no_answer={no_answer}"
        )
        return AnswerResult(
            question=question,
            answer=answer,
            sources=sources,
            chunks_used=len(chunks),
            tokens_in_context=tokens_used,
            elapsed_ms=round(elapsed, 1),
            model=backend.model,
            backend=backend_name,
            hops=int(hops),
            no_answer=no_answer,
            token_counts=token_counts or None,
            generation_ms=round(generation_ms, 1),
            citation_count=citation_count,
            answer_length_chars=answer_length_chars,
            context_utilisation=context_utilisation,
            chunks_in_context=ordered_chunks,   # trimmed set the LLM actually saw
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_backend(self) -> tuple:
        # ── Primary: Groq API ────────────────────────────────────────────────
        if self._groq.is_available():
            logger.info(f"  Using Groq backend: {self._groq.model}")
            return self._groq, "groq"
        # ── Fallback: local Ollama ───────────────────────────────────────────
        if self._ollama.is_available():
            logger.info(
                f"  Groq unavailable — falling back to Ollama backend: {_QWEN_MODEL}"
            )
            return self._ollama, "ollama"
        raise RuntimeError(
            "Neither Groq nor Ollama is available for answer generation.\n"
            f"  • Set GROQ_API_KEY in .env to use Groq ({_GROQ_MODEL_DEFAULT})\n"
            f"  • Or start Ollama: `ollama serve` then `ollama pull {_QWEN_MODEL}`"
        )

    @staticmethod
    def _empty_result(question: str, config: GenerationConfig) -> AnswerResult:
        return AnswerResult(
            question=question,
            answer="No relevant context was found in the indexed documents.",
            sources=[],
            chunks_used=0,
            tokens_in_context=0,
            elapsed_ms=0.0,
            model="none",
            backend="none",
            no_answer=True,
        )


# ---------------------------------------------------------------------------
# Convenience functions — drop-in for api.py /ask endpoint
# ---------------------------------------------------------------------------

_generator: Optional[AnswerGenerator] = None


def get_generator(groq_api_key: str | None = None) -> AnswerGenerator:
    """
    Module-level singleton, mirroring the pattern used for embedder.py.

    Pass `groq_api_key` on the first call to enable the Groq fallback backend.
    Subsequent calls return the cached instance regardless of the argument.
    api.py passes the GROQ_API_KEY at startup so all subsequent /ask calls
    automatically have the fallback available.
    """
    global _generator
    if _generator is None:
        _generator = AnswerGenerator(groq_api_key=groq_api_key)
    return _generator


def generate_answer(
    question: str,
    chunks:   list[dict],
    config:   Optional[GenerationConfig] = None,
) -> AnswerResult:
    """One-liner for non-streaming use in api.py."""
    return get_generator().generate(question, chunks, config)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test answer_generator.py")
    parser.add_argument("question", nargs="?", default="What is the main topic?")
    parser.add_argument("--stream", action="store_true", help="Stream output")
    args = parser.parse_args()

    dummy_chunks = [
        {
            "content": (
                "The Secure AI Assistant project is a fully self-contained RAG pipeline "
                "that ingests documents, extracts structured content, chunks for vector "
                "retrieval, and exposes a FastAPI server with streaming endpoints."
            ),
            "metadata": {
                "source":     "CLAUDE.md",
                "page_start": 1,
                "section":    "Project Overview",
                "type":       "text",
            },
            "score":   0.95,
            "hop":     1,
            "primary": True,
        },
        {
            "content": (
                "The embedding model is BAAI/bge-m3 which produces both dense (1024-dim) "
                "and sparse vectors. Hybrid RRF search is used in Qdrant with BM25 sparse "
                "vectors computed by bm25_encoder.py."
            ),
            "metadata": {
                "source":     "CLAUDE.md",
                "page_start": 2,
                "section":    "Embedding",
                "type":       "text",
            },
            "score":   0.82,
            "hop":     1,
            "primary": True,
        },
    ]

    gen = AnswerGenerator()

    if args.stream:
        print(f"\n[Streaming answer for: '{args.question}']\n")
        for token in gen.stream(args.question, dummy_chunks):
            print(token, end="", flush=True)
        print()
    else:
        result = gen.generate(args.question, dummy_chunks)
        print(f"\n[{result.backend}/{result.model}  {result.elapsed_ms:.0f}ms  "
              f"{result.tokens_in_context} ctx tokens  no_answer={result.no_answer}]")
        if result.token_counts:
            tc = result.token_counts
            print(f"[tokens: {tc['prompt']} prompt + {tc['completion']} completion = {tc['total']}]")
        print(f"\n{result.answer}")
        print(f"\n[Sources cited: {len(result.sources)}]")
        for s in result.sources:
            print(f"  - {s['source']} p.{s['page_start']}  score={s['score']}")
