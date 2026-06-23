"""
file_preparation/evaluation/judge.py

LLM-as-a-Judge evaluation layer for the Secure AI Assistant RAG pipeline.

Evaluates generated RAG answers across six quality dimensions using the
Groq API (qwen/qwen3-32b).  Uses the same `groq` SDK already installed for
OCR, HyDE query expansion, and confidence scoring — no new dependencies.

Configure via .env:
    JUDGE_MODEL   = qwen/qwen3-32b   # override to any Groq-hosted model
    JUDGE_API_KEY = gsk_...          # falls back to GROQ_API_KEY if not set

Dimensions
----------
faithfulness        0-1   Every claim in the answer is grounded in retrieved context.
answer_relevance    0-1   Answer actually addresses the user's question.
context_relevance   0-1   Retrieved chunks are relevant to the question.
completeness        0-1   Answer covers all aspects the question asks for.
citation_accuracy   0-1   [Source: ...] citations map to chunks that support the claim.
correctness         0-1   Answer matches a ground-truth reference (only when reference provided).

Each dimension is scored independently so a failure in one call does not
corrupt the others.  Results combine into a JudgeResult dataclass that is
JSON-serialisable.

Usage
-----
    from file_preparation.evaluation import judge_answer, JudgeResult

    # Basic evaluation
    result: JudgeResult = judge_answer(
        question  = "What are the key revenue figures?",
        answer    = answer_text,
        chunks    = rag_result.chunks_in_context,
        no_answer = rag_result.no_answer,
    )
    print(result.overall)     # weighted aggregate 0-1
    print(result.to_dict())   # full breakdown, JSON-ready

    # With ground-truth reference + per-chunk scoring
    result = judge_answer(
        question     = "What is the revenue?",
        answer       = answer_text,
        chunks       = chunks,
        reference    = "Revenue was $4.2B in Q3 2024.",
        score_chunks = True,
    )
    print(result.correctness.score)   # 0.0-1.0
    print(result.chunk_scores)        # per-chunk relevance list

    # Save to CSV for longitudinal tracking
    save_to_csv(result, "eval_log.csv")

Notes
-----
- Qwen3 returns <think>...</think> blocks before its JSON -- stripped automatically.
- All dimension calls run in parallel via ThreadPoolExecutor (default).
- Per-chunk calls always run sequentially to avoid rate limits.
- Returns score=None (not 0) for failed or skipped dimensions.
- run_parallel=False forces sequential execution (useful for debugging).
"""

from __future__ import annotations

import csv as _csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

_DEFAULT_MODEL = "qwen/qwen3-32b"

# Model: override via JUDGE_MODEL env var
_JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", _DEFAULT_MODEL)

# API key: prefer GROQ_EVAL_API_KEY (dedicated eval pool) → JUDGE_API_KEY → GROQ_API_KEY
_JUDGE_API_KEY: str = (
    os.getenv("GROQ_EVAL_API_KEY", "")
    or os.getenv("JUDGE_API_KEY", "")
    or os.getenv("GROQ_API_KEY", "")
)

# Generation settings
_MAX_TOKENS  = 2048  # Qwen3 <think> blocks can be 500-800 tokens; 2048 gives room for both thinking and JSON output
_TEMPERATURE = 0.0   # fully deterministic scoring

# Context sent to judge -- capped to stay well inside Groq context limits
_MAX_CONTEXT_CHARS = 6000
_MAX_ANSWER_CHARS  = 2000
_MAX_CHUNK_CHARS   = 800   # per-chunk cap for individual chunk scoring

# Rate-limit retry settings for Groq free tier (6 000 TPM limit)
# When a 429 is received, parse Groq's suggested wait time from the error message
# and sleep that long before retrying.  Up to _MAX_RETRIES attempts per dimension.
_MAX_RETRIES        = 3    # max retries per dimension call on 429
_RETRY_WAIT_DEFAULT = 35.0 # seconds -- used when we can't parse Groq's suggested wait
_RETRY_WAIT_EXTRA   = 2.0  # extra buffer added on top of Groq's suggested wait

# Regex to extract the suggested retry delay from Groq's 429 error message
# e.g. "Please try again in 32.18s."
_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)

# Strip Qwen3 (and other thinking-model) <think>...</think> blocks before JSON parse
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(raw: str) -> str:
    """
    Remove <think>...</think> reasoning blocks from a thinking-model response.

    Tolerates three failure modes that the bare `_THINK_RE.sub` does not:
      1. Properly closed <think>...</think> blocks   → stripped in full.
      2. Unclosed <think> blocks (model truncated   → everything from <think>
         before emitting </think>)                     onward is dropped.
      3. Malformed closing tags (e.g. "< /think>",  → also caught by the
         "</ think>", or HTML-escaped variants)        residual-<think> guard.

    After stripping, leading and trailing whitespace is removed so the
    downstream JSON parser sees only the JSON payload (if any).
    """
    cleaned = _THINK_RE.sub("", raw)
    # Catch any straggler opening tag (unclosed or malformed closing)
    lowered = cleaned.lower()
    if "<think>" in lowered:
        idx = lowered.find("<think>")
        cleaned = cleaned[:idx]
    return cleaned.strip()

# Dimension weights -- the _compute_overall normalises by total active weight,
# so these don't need to sum to 1.0. correctness only contributes when a
# reference answer is provided; all others contribute when score is not None.
_WEIGHTS: dict[str, float] = {
    "faithfulness":      0.35,
    "answer_relevance":  0.25,
    "context_relevance": 0.15,
    "completeness":      0.15,
    "citation_accuracy": 0.10,
    "correctness":       0.30,  # counted only when reference is provided
}

# Default path for CSV logging
_DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "eval_log.csv"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DimensionScore:
    score:     Optional[float]  # 0.0-1.0, or None if evaluation failed / skipped
    reasoning: str              # one-sentence rationale from the judge
    dimension: str


def _dim_to_dict(d: DimensionScore) -> dict:
    return {"score": d.score, "reasoning": d.reasoning}


def _skipped(dimension: str, reason: str = "skipped") -> DimensionScore:
    return DimensionScore(score=None, reasoning=reason, dimension=dimension)


@dataclass
class JudgeResult:
    # ── Required ──────────────────────────────────────────────────────────────
    question:          str
    faithfulness:      DimensionScore
    answer_relevance:  DimensionScore
    context_relevance: DimensionScore
    completeness:      DimensionScore
    citation_accuracy: DimensionScore

    # ── Optional — with defaults so callers don't have to supply them ─────────
    correctness:   DimensionScore  = field(
        default_factory=lambda: _skipped("correctness", "no reference provided")
    )
    overall:       Optional[float] = None
    elapsed_ms:    float           = 0.0
    model:         str             = _JUDGE_MODEL
    error:         Optional[str]   = None

    # Per-chunk relevance scores (populated when score_chunks=True)
    chunk_scores:  Optional[list]  = None   # list of {"chunk_idx", "score", "reasoning", "preview"}

    # Pipeline timing passed in from the caller (api.py)
    retrieval_ms:  Optional[float] = None
    generation_ms: Optional[float] = None
    token_counts:  Optional[dict]  = None   # {"prompt": N, "completion": N, "total": N}

    def __post_init__(self) -> None:
        self.overall = self._compute_overall()

    def _compute_overall(self) -> Optional[float]:
        total_weight = 0.0
        weighted_sum = 0.0
        for dim, weight in _WEIGHTS.items():
            score = getattr(self, dim).score
            if score is not None:
                weighted_sum += score * weight
                total_weight += weight
        if total_weight == 0:
            return None
        return round(weighted_sum / total_weight, 4)

    def to_dict(self) -> dict:
        d = {
            "question":    self.question,
            "overall":     self.overall,
            "dimensions": {
                "faithfulness":      _dim_to_dict(self.faithfulness),
                "answer_relevance":  _dim_to_dict(self.answer_relevance),
                "context_relevance": _dim_to_dict(self.context_relevance),
                "completeness":      _dim_to_dict(self.completeness),
                "citation_accuracy": _dim_to_dict(self.citation_accuracy),
                "correctness":       _dim_to_dict(self.correctness),
            },
            "chunk_scores":  self.chunk_scores,
            "elapsed_ms":    self.elapsed_ms,
            "retrieval_ms":  self.retrieval_ms,
            "generation_ms": self.generation_ms,
            "token_counts":  self.token_counts,
            "model":         self.model,
            "error":         self.error,
        }
        return d


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

def _get_groq_client(api_key: Optional[str] = None):
    """
    Return a groq.Groq client using the project's existing groq SDK.
    Raises ImportError if groq is not installed, ValueError if no API key.
    """
    try:
        import groq as groq_sdk
    except ImportError as exc:
        raise ImportError(
            "groq package not installed. Run: pip install groq"
        ) from exc

    key = api_key or _JUDGE_API_KEY
    if not key:
        raise ValueError(
            "No API key found for the judge. "
            "Set JUDGE_API_KEY or GROQ_API_KEY in .env."
        )
    return groq_sdk.Groq(api_key=key)


# ---------------------------------------------------------------------------
# Context block formatter
# ---------------------------------------------------------------------------

def _build_context_block(
    chunks:    list[dict],
    max_chars: int = _MAX_CONTEXT_CHARS,
) -> str:
    """Flatten retrieved chunks into a numbered evidence block for the judge."""
    lines = []
    total = 0
    for i, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "")
        meta    = chunk.get("metadata") or {}
        source  = meta.get("source") or chunk.get("source", "unknown")
        _ps     = meta.get("page_start") if meta.get("page_start") is not None else chunk.get("page_start")
        page    = _ps if _ps is not None else "?"
        entry   = f"[{i}] (source: {source} | page: {page})\n{content}"
        if total + len(entry) > max_chars:
            lines.append("[... additional chunks truncated for evaluation ...]")
            break
        lines.append(entry)
        total += len(entry)
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Individual dimension prompts
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """\
You are a strict factual auditor for a RAG system.

TASK
Evaluate whether every factual claim in ANSWER is supported by the CONTEXT.
A claim is "hallucinated" if it cannot be verified from or inferred from CONTEXT.

CONTEXT
{context}

QUESTION
{question}

ANSWER
{answer}

OUTPUT FORMAT
Output ONLY a single JSON object with this exact schema:
{{"score": <float in [0.0, 1.0]>, "reasoning": "<one sentence>"}}

CRITICAL RULES
- "score" MUST be a float between 0.0 and 1.0 (inclusive).
- 0.0 means entirely hallucinated, 1.0 means fully grounded.
- Do NOT use a 1-5, 1-10, or 1-100 scale.
- Do NOT output any text, commentary, or markdown before or after the JSON.

Example output:
{{"score": 0.85, "reasoning": "Most claims are supported but one revenue figure has no source in context."}}
"""

_ANSWER_RELEVANCE_PROMPT = """\
You are evaluating a RAG system's answer quality.

TASK
Assess how well ANSWER addresses the user's QUESTION.
A perfect answer (1.0) completely and directly answers all parts of the question.
A poor answer (0.0) ignores the question, answers a different question, or gives an unhelpful refusal.

QUESTION
{question}

ANSWER
{answer}

OUTPUT FORMAT
Output ONLY a single JSON object with this exact schema:
{{"score": <float in [0.0, 1.0]>, "reasoning": "<one sentence>"}}

CRITICAL RULES
- "score" MUST be a float between 0.0 and 1.0 (inclusive).
- 0.0 means the answer ignores the question, 1.0 means it fully addresses it.
- Do NOT use a 1-5, 1-10, or 1-100 scale.
- Do NOT output any text, commentary, or markdown before or after the JSON.

Example output:
{{"score": 0.9, "reasoning": "The answer directly addresses the question with specific figures."}}
"""

_CONTEXT_RELEVANCE_PROMPT = """\
You are evaluating a RAG retrieval system.

TASK
Assess how relevant the retrieved CONTEXT chunks are to answering the QUESTION.
Score 1.0 if all chunks are highly relevant, 0.0 if the chunks are entirely off-topic.
Penalise noise chunks that add no value.

QUESTION
{question}

CONTEXT
{context}

OUTPUT FORMAT
Output ONLY a single JSON object with this exact schema:
{{"score": <float in [0.0, 1.0]>, "reasoning": "<one sentence>"}}

CRITICAL RULES
- "score" MUST be a float between 0.0 and 1.0 (inclusive).
- 0.0 means all chunks are off-topic, 1.0 means all chunks are highly relevant.
- Do NOT use a 1-5, 1-10, or 1-100 scale.
- Do NOT output any text, commentary, or markdown before or after the JSON.

Example output:
{{"score": 0.75, "reasoning": "Most chunks are relevant but two discuss unrelated topics."}}
"""

_COMPLETENESS_PROMPT = """\
You are evaluating a RAG system's answer completeness.

TASK
Given the QUESTION and the available CONTEXT, assess how completely ANSWER covers
all aspects the question asks about. The context defines the ceiling -- if
information is not in the context, the answer cannot be expected to cover it.

QUESTION
{question}

CONTEXT
{context}

ANSWER
{answer}

OUTPUT FORMAT
Output ONLY a single JSON object with this exact schema:
{{"score": <float in [0.0, 1.0]>, "reasoning": "<one sentence>"}}

CRITICAL RULES
- "score" MUST be a float between 0.0 and 1.0 (inclusive).
- 0.0 means the answer ignores most aspects, 1.0 means it fully covers them.
- Do NOT use a 1-5, 1-10, or 1-100 scale.
- Do NOT output any text, commentary, or markdown before or after the JSON.

Example output:
{{"score": 0.6, "reasoning": "The answer covers revenue but omits the headcount figures present in context."}}
"""

_CITATION_ACCURACY_PROMPT = """\
You are auditing source citations in a RAG system answer.

TASK
Check each [Source: filename, page X] citation in ANSWER against CONTEXT.
A citation is accurate if the cited chunk genuinely supports the claim it annotates.
A citation is inaccurate if the chunk does not exist, does not match, or supports
a different claim.

CONTEXT
{context}

ANSWER
{answer}

OUTPUT FORMAT
Output ONLY a single JSON object with this exact schema:
{{"score": <float in [0.0, 1.0]>, "reasoning": "<one sentence>"}}

CRITICAL RULES
- "score" MUST be a float between 0.0 and 1.0 (inclusive).
- 0.0 means all citations are wrong or absent, 1.0 means all citations are correct.
- If there are NO [Source: ...] citations anywhere in ANSWER, return 0.0 — citations
  are required and their absence is a failure.
- Do NOT use a 1-5, 1-10, or 1-100 scale.
- Do NOT output any text, commentary, or markdown before or after the JSON.

Example output:
{{"score": 1.0, "reasoning": "All three citations map to chunks that directly support their claims."}}
"""

_CORRECTNESS_PROMPT = """\
You are evaluating a RAG system's factual correctness against a ground-truth reference.

TASK
Compare ANSWER against the REFERENCE (ground-truth) answer.
Score 1.0 if the answer is fully correct and consistent with the reference.
Score 0.0 if the answer contradicts or completely ignores the reference.
Penalise omissions, factual errors, and contradictions proportionally.

QUESTION
{question}

REFERENCE ANSWER (ground truth)
{reference}

ANSWER (to evaluate)
{answer}

OUTPUT FORMAT
Output ONLY a single JSON object with this exact schema:
{{"score": <float in [0.0, 1.0]>, "reasoning": "<one sentence>"}}

CRITICAL RULES
- "score" MUST be a float between 0.0 and 1.0 (inclusive).
- 0.0 means the answer contradicts the reference, 1.0 means it fully matches.
- Do NOT use a 1-5, 1-10, or 1-100 scale.
- Do NOT output any text, commentary, or markdown before or after the JSON.

Example output:
{{"score": 0.8, "reasoning": "Answer is mostly correct but omits the YoY growth figure present in the reference."}}
"""

_CHUNK_RELEVANCE_PROMPT = """\
You are evaluating retrieval quality for a RAG system.

TASK
Rate how relevant the CHUNK is for answering the QUESTION.
Score 1.0 if the chunk is perfectly relevant and directly useful.
Score 0.0 if the chunk is completely irrelevant or off-topic.

QUESTION
{question}

CHUNK [{chunk_idx}]
{chunk}

OUTPUT FORMAT
Output ONLY a single JSON object with this exact schema:
{{"score": <float in [0.0, 1.0]>, "reasoning": "<one sentence>"}}

CRITICAL RULES
- "score" MUST be a float between 0.0 and 1.0 (inclusive).
- 0.0 means the chunk is completely off-topic, 1.0 means it is perfectly relevant.
- Do NOT use a 1-5, 1-10, or 1-100 scale.
- Do NOT output any text, commentary, or markdown before or after the JSON.

Example output:
{{"score": 0.9, "reasoning": "Chunk directly contains the revenue figures the question asks about."}}
"""


# ---------------------------------------------------------------------------
# Single-dimension evaluation
# ---------------------------------------------------------------------------

def _call_judge(
    client,
    model:     str,
    prompt:    str,
    dimension: str,
) -> DimensionScore:
    """
    Fire one Groq completion for a single dimension.

    Handles:
    - Qwen3 <think>...</think> blocks (stripped before JSON parse)
    - Markdown code fences (stripped)
    - Out-of-range or missing score values (returns score=None)
    - Groq 429 rate-limit errors: retries up to _MAX_RETRIES times, sleeping
      for the duration Groq suggests in the error message.
    """
    raw = ""
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model       = model,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = _MAX_TOKENS,
                temperature = _TEMPERATURE,
                timeout     = 60,   # seconds — prevents indefinite hangs on slow Groq responses
            )
            raw = response.choices[0].message.content.strip()

            # Strip <think>...</think> blocks (Qwen3 thinking mode).
            # Uses the robust helper that also tolerates unclosed/malformed tags.
            raw = _strip_think(raw)
            if not raw:
                # Pure unclosed <think> with no JSON payload — surface cleanly
                logger.warning(
                    f"[JUDGE] {dimension}: empty response after stripping "
                    f"<think> block (model truncated or refused to emit JSON)"
                )
                return DimensionScore(
                    score     = None,
                    reasoning = "Empty response after <think> strip",
                    dimension = dimension,
                )

            # Strip markdown code fences if the model wraps its JSON
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```$",           "", raw, flags=re.MULTILINE)
            raw = raw.strip()

            data  = json.loads(raw)
            score = float(data.get("score", -1))
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"score out of range: {score}")
            reasoning = str(data.get("reasoning", "")).strip()

            return DimensionScore(
                score     = round(score, 4),
                reasoning = reasoning,
                dimension = dimension,
            )

        except json.JSONDecodeError as exc:
            logger.warning(f"[JUDGE] {dimension}: JSON parse error -- {exc}. Raw: {raw!r:.120}")
            return DimensionScore(score=None, reasoning=f"Parse error: {exc}", dimension=dimension)

        except Exception as exc:
            err_str = str(exc)
            is_rate_limit = "429" in err_str or "rate_limit_exceeded" in err_str

            if is_rate_limit and attempt < _MAX_RETRIES:
                m    = _RETRY_AFTER_RE.search(err_str)
                wait = (float(m.group(1)) + _RETRY_WAIT_EXTRA) if m else _RETRY_WAIT_DEFAULT
                logger.warning(
                    f"[JUDGE] {dimension}: rate limited (429), "
                    f"retrying in {wait:.0f}s (attempt {attempt + 1}/{_MAX_RETRIES})"
                )
                time.sleep(wait)
                continue  # retry the loop

            # Non-429 error, or retries exhausted
            logger.warning(f"[JUDGE] {dimension}: evaluation failed -- {exc}")
            return DimensionScore(score=None, reasoning=f"Error: {exc}", dimension=dimension)

    # Should never reach here, but satisfy the type checker
    return DimensionScore(score=None, reasoning="Max retries exceeded", dimension=dimension)


# ---------------------------------------------------------------------------
# Per-chunk relevance scoring
# ---------------------------------------------------------------------------

def _score_chunks_individually(
    client,
    model:     str,
    question:  str,
    chunks:    list[dict],
) -> list[dict]:
    """
    Score each chunk individually for relevance to the question.

    Runs sequentially (not in parallel) to avoid stacking rate-limit pressure
    on top of the main 5-dimension calls.

    Returns a list of dicts:
      {"chunk_idx": int, "score": float|None, "reasoning": str, "preview": str}
    """
    results = []
    for i, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "") or ""
        prompt  = _CHUNK_RELEVANCE_PROMPT.format(
            question  = question,
            chunk_idx = i,
            chunk     = content[:_MAX_CHUNK_CHARS],
        )
        ds = _call_judge(client, model, prompt, f"chunk_{i}")
        results.append({
            "chunk_idx": i,
            "score":     ds.score,
            "reasoning": ds.reasoning,
            "preview":   content[:120].replace("\n", " ") + ("…" if len(content) > 120 else ""),
        })
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def judge_answer(
    question:          str,
    answer:            str,
    chunks:            list[dict],
    *,
    no_answer:         bool           = False,
    reference:         Optional[str]  = None,
    score_chunks:      bool           = False,
    model:             str            = _JUDGE_MODEL,
    api_key:           Optional[str]  = None,
    run_parallel:      bool           = True,
    max_context_chars: int            = _MAX_CONTEXT_CHARS,
    max_answer_chars:  int            = _MAX_ANSWER_CHARS,
    retrieval_ms:      Optional[float] = None,
    generation_ms:     Optional[float] = None,
    token_counts:      Optional[dict]  = None,
) -> JudgeResult:
    """
    Evaluate a RAG answer across up to six quality dimensions via Groq.

    Parameters
    ----------
    question          : The user's original question.
    answer            : The generated answer text.
    chunks            : AnswerResult.chunks_in_context -- list of chunk dicts.
    no_answer         : Pass result.no_answer=True to skip citation_accuracy and
                        completeness scoring (meaningless on refusals).
    reference         : Ground-truth reference answer. When provided, the
                        correctness dimension is scored (0.0-1.0). When absent,
                        correctness is skipped (score=None).
    score_chunks      : When True, score each chunk individually for relevance.
                        Runs sequentially after the main dimensions. Adds
                        ~1-2 Groq calls per chunk — use with care on free tier.
    model             : Groq model identifier (default: JUDGE_MODEL or qwen/qwen3-32b).
    api_key           : Groq API key (falls back to JUDGE_API_KEY / GROQ_API_KEY).
    run_parallel      : Fire the main dimension calls concurrently (default True).
    max_context_chars : Character cap on the context block sent to the judge.
    max_answer_chars  : Character cap on the answer text sent to the judge.
    retrieval_ms      : Retrieval latency from the pipeline (stored for CSV logging).
    generation_ms     : Generation latency from the pipeline (stored for CSV logging).
    token_counts      : LLM token usage dict {"prompt", "completion", "total"}
                        (stored for CSV logging).

    Returns
    -------
    JudgeResult with per-dimension scores and a weighted overall score.
    """
    t0 = time.monotonic()

    try:
        client = _get_groq_client(api_key=api_key)
    except Exception as exc:
        logger.error(f"[JUDGE] Cannot initialise Groq client: {exc}")
        return JudgeResult(
            question          = question,
            faithfulness      = _skipped("faithfulness",      f"Client error: {exc}"),
            answer_relevance  = _skipped("answer_relevance",  f"Client error: {exc}"),
            context_relevance = _skipped("context_relevance", f"Client error: {exc}"),
            completeness      = _skipped("completeness",      f"Client error: {exc}"),
            citation_accuracy = _skipped("citation_accuracy", f"Client error: {exc}"),
            correctness       = _skipped("correctness",       f"Client error: {exc}"),
            elapsed_ms        = 0.0,
            model             = model,
            error             = str(exc),
            retrieval_ms      = retrieval_ms,
            generation_ms     = generation_ms,
            token_counts      = token_counts,
        )

    context_block = _build_context_block(chunks, max_chars=max_context_chars)
    ans           = answer[:max_answer_chars]

    # ── Build task map ──────────────────────────────────────────────────────
    # Skip citation_accuracy and completeness when no_answer=True.
    # Add correctness only when a reference answer is provided.
    tasks: dict[str, str] = {
        "faithfulness": _FAITHFULNESS_PROMPT.format(
            context=context_block, question=question, answer=ans),
        "answer_relevance": _ANSWER_RELEVANCE_PROMPT.format(
            question=question, answer=ans),
        "context_relevance": _CONTEXT_RELEVANCE_PROMPT.format(
            question=question, context=context_block),
    }
    if not no_answer:
        tasks["completeness"] = _COMPLETENESS_PROMPT.format(
            context=context_block, question=question, answer=ans)
        tasks["citation_accuracy"] = _CITATION_ACCURACY_PROMPT.format(
            context=context_block, answer=ans)
    if reference:
        tasks["correctness"] = _CORRECTNESS_PROMPT.format(
            question=question,
            reference=reference[:max_answer_chars],
            answer=ans,
        )

    # ── Run dimension evaluations ───────────────────────────────────────────
    scores: dict[str, DimensionScore] = {}

    if run_parallel:
        with ThreadPoolExecutor(max_workers=min(len(tasks), 2)) as pool:
            futures = {
                pool.submit(_call_judge, client, model, prompt, dim): dim
                for dim, prompt in tasks.items()
            }
            for future in as_completed(futures):
                dim = futures[future]
                scores[dim] = future.result()
    else:
        for dim, prompt in tasks.items():
            scores[dim] = _call_judge(client, model, prompt, dim)

    # Fill skipped dimensions
    if no_answer:
        scores["completeness"]      = _skipped("completeness",      "skipped: no_answer=True")
        scores["citation_accuracy"] = _skipped("citation_accuracy", "skipped: no_answer=True")
    if not reference:
        scores["correctness"] = _skipped("correctness", "no reference provided")

    # ── Per-chunk scoring (sequential, after main dims) ─────────────────────
    chunk_scores_out: Optional[list] = None
    if score_chunks and chunks:
        logger.info(f"[JUDGE] scoring {len(chunks)} chunk(s) individually …")
        chunk_scores_out = _score_chunks_individually(client, model, question, chunks)

    elapsed = round((time.monotonic() - t0) * 1000, 1)
    logger.info(
        f"[JUDGE] done in {elapsed:.0f} ms via {model} -- "
        f"F={scores['faithfulness'].score} "
        f"AR={scores['answer_relevance'].score} "
        f"CR={scores['context_relevance'].score} "
        f"C={scores['completeness'].score} "
        f"CA={scores['citation_accuracy'].score} "
        f"COR={scores['correctness'].score}"
    )

    return JudgeResult(
        question          = question,
        faithfulness      = scores["faithfulness"],
        answer_relevance  = scores["answer_relevance"],
        context_relevance = scores["context_relevance"],
        completeness      = scores["completeness"],
        citation_accuracy = scores["citation_accuracy"],
        correctness       = scores["correctness"],
        elapsed_ms        = elapsed,
        model             = model,
        chunk_scores      = chunk_scores_out,
        retrieval_ms      = retrieval_ms,
        generation_ms     = generation_ms,
        token_counts      = token_counts,
    )


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

def save_to_csv(result: "JudgeResult", filepath: str | Path = _DEFAULT_CSV_PATH) -> None:
    """
    Append one JudgeResult to a CSV log file (creates it if it doesn't exist).

    Columns (flattened for easy analysis):
      timestamp, question, overall,
      faithfulness, faithfulness_reasoning,
      answer_relevance, answer_relevance_reasoning,
      context_relevance, context_relevance_reasoning,
      completeness, completeness_reasoning,
      citation_accuracy, citation_accuracy_reasoning,
      correctness, correctness_reasoning,
      num_chunks, chunk_scores_json,
      retrieval_ms, generation_ms,
      prompt_tokens, completion_tokens, total_tokens,
      judge_elapsed_ms, model
    """
    filepath = Path(filepath)
    tc = result.token_counts or {}
    row = {
        "timestamp":                     datetime.now().isoformat(timespec="seconds"),
        "question":                      result.question[:200],
        "overall":                       result.overall,
        "faithfulness":                  result.faithfulness.score,
        "faithfulness_reasoning":        result.faithfulness.reasoning,
        "answer_relevance":              result.answer_relevance.score,
        "answer_relevance_reasoning":    result.answer_relevance.reasoning,
        "context_relevance":             result.context_relevance.score,
        "context_relevance_reasoning":   result.context_relevance.reasoning,
        "completeness":                  result.completeness.score,
        "completeness_reasoning":        result.completeness.reasoning,
        "citation_accuracy":             result.citation_accuracy.score,
        "citation_accuracy_reasoning":   result.citation_accuracy.reasoning,
        "correctness":                   result.correctness.score,
        "correctness_reasoning":         result.correctness.reasoning,
        "num_chunks":                    len(result.chunk_scores) if result.chunk_scores else "",
        "chunk_scores_json":             json.dumps(result.chunk_scores) if result.chunk_scores else "",
        "retrieval_ms":                  result.retrieval_ms if result.retrieval_ms is not None else "",
        "generation_ms":                 result.generation_ms if result.generation_ms is not None else "",
        "prompt_tokens":                 tc.get("prompt", ""),
        "completion_tokens":             tc.get("completion", ""),
        "total_tokens":                  tc.get("total", ""),
        "judge_elapsed_ms":              result.elapsed_ms,
        "model":                         result.model,
    }

    file_exists = filepath.exists()
    with filepath.open("a", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f"[JUDGE] evaluation saved → {filepath}")


# ---------------------------------------------------------------------------
# Batch evaluation (offline dataset benchmarking)
# ---------------------------------------------------------------------------

def batch_judge(
    records:            list[dict],
    *,
    model:              str            = _JUDGE_MODEL,
    api_key:            Optional[str]  = None,
    inter_record_sleep: float          = 2.0,
    run_parallel:       bool           = True,
    score_chunks:       bool           = False,
    csv_path:           Optional[str]  = None,
) -> list[dict]:
    """
    Evaluate a list of RAG records offline.

    Each record must have keys: "question", "answer", "chunks".
    Optional keys:
      "no_answer"    (bool)  — skips citation/completeness when True.
      "reference"    (str)   — ground-truth answer for correctness scoring.
      "retrieval_ms" (float) — passed through to JudgeResult.
      "generation_ms"(float) — passed through to JudgeResult.
      "token_counts" (dict)  — passed through to JudgeResult.

    Parameters
    ----------
    records             : List of record dicts.
    model               : Groq model identifier.
    api_key             : Groq API key (falls back to env vars).
    inter_record_sleep  : Seconds between records (default 2.0 -- respects Groq
                          free-tier rate limit of ~30 req/min; set to 0 on paid tier).
    run_parallel        : Fire all dimension calls concurrently per record (default True).
    score_chunks        : Score each chunk individually in every record (default False).
    csv_path            : If provided, append every result to this CSV file.

    Returns a list of dicts (one per record) with a "judge" key merged in.
    """
    output = []
    for i, rec in enumerate(records):
        logger.info(f"[JUDGE] batch {i + 1}/{len(records)}: {rec['question'][:60]}")
        jr = judge_answer(
            question      = rec["question"],
            answer        = rec["answer"],
            chunks        = rec.get("chunks", []),
            no_answer     = rec.get("no_answer", False),
            reference     = rec.get("reference"),
            score_chunks  = score_chunks,
            model         = model,
            api_key       = api_key,
            run_parallel  = run_parallel,
            retrieval_ms  = rec.get("retrieval_ms"),
            generation_ms = rec.get("generation_ms"),
            token_counts  = rec.get("token_counts"),
        )
        if csv_path:
            save_to_csv(jr, csv_path)
        merged          = dict(rec)
        merged["judge"] = jr.to_dict()
        output.append(merged)
        if inter_record_sleep > 0 and i < len(records) - 1:
            time.sleep(inter_record_sleep)
    return output
