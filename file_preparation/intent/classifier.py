"""
intent/classifier.py — Customer Support Intent Classifier
==========================================================

Pre-retrieval query classification for the RAG Customer Support pipeline.
Classifies every incoming question into one of 7 support intents and decides
whether to escalate immediately (before any retrieval is attempted).

Uses Groq llama-3.1-8b-instant — the fastest Groq model — to keep the
pre-retrieval latency under 300 ms on average.

Intents
-------
  how_to              — User wants step-by-step guidance ("How do I reset my password?")
  troubleshooting     — User is reporting a malfunction/error ("My login keeps failing")
  complaint           — User is expressing dissatisfaction/frustration
  billing             — Questions about charges, invoices, subscriptions, refunds
  account             — Account management: settings, deletion, access, security
  escalation_request  — User explicitly asks to speak to a human / escalate
  general             — Anything else (product info, feature questions, …)

Escalate-now intents: escalation_request, complaint (high-frustration)

Usage
-----
    from file_preparation.intent import classify_intent

    result = classify_intent("I want to speak to a manager", groq_client=client)
    if result.strategy == "escalate":
        # skip RAG, go straight to escalation handler
        ...
    else:
        # run normal RAG pipeline
        ...
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTENTS = {
    "how_to":             "User wants instructions or step-by-step guidance.",
    "troubleshooting":    "User is experiencing a bug, error, or malfunction.",
    "complaint":          "User is expressing frustration, dissatisfaction, or a complaint.",
    "billing":            "Question about charges, invoices, subscriptions, payments, or refunds.",
    "account":            "Question about account access, settings, deletion, security, or profile.",
    "escalation_request": "User explicitly requests to speak with a human agent or escalate the issue.",
    "general":            "General product/service question that does not fit the above categories.",
}

# Intents that trigger immediate escalation (before RAG)
_ESCALATE_NOW_INTENTS = {"escalation_request"}

# Regex patterns for fast zero-latency escalation detection (before Groq call)
_FAST_ESCALATE_PATTERNS = re.compile(
    r"\b("
    r"speak\s+to\s+(a\s+)?(human|agent|person|manager|representative|rep|supervisor)"
    r"|talk\s+to\s+(a\s+)?(human|agent|person|manager|representative|rep|supervisor)"
    r"|connect\s+me\s+(with|to)\s+(a\s+)?(human|agent|person|manager|representative)"
    r"|escalat(e|ing|ion)"
    r"|transfer\s+me"
    r"|live\s+(chat|agent|support)"
    r"|real\s+person"
    r")\b",
    re.IGNORECASE,
)

# Groq model for intent classification — use fastest model on hot path
_INTENT_MODEL = "llama-3.1-8b-instant"

# Qwen3 think-tag stripper (in case model emits them)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

_CLASSIFICATION_PROMPT = """\
You are a customer support query classifier. Classify the following customer query
into exactly ONE of these intent categories:

{intent_list}

Respond with a JSON object ONLY (no markdown, no explanation):
{{
  "intent": "<one of the exact intent names above>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}}

Customer query: {question}"""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class IntentResult:
    """Outcome of a single intent classification call."""
    intent:       str                   # one of the 7 intent keys
    confidence:   float                 # 0.0–1.0
    escalate_now: bool                  # True → skip RAG, go to escalation handler
    strategy:     str                   # "rag" | "escalate"
    elapsed_ms:   float
    reasoning:    str = ""              # one-sentence rationale from the model
    fast_path:    bool = False          # True when escalation detected via regex (no Groq call)
    error:        Optional[str] = None  # set when classification failed

    def to_dict(self) -> dict:
        return {
            "intent":       self.intent,
            "confidence":   self.confidence,
            "escalate_now": self.escalate_now,
            "strategy":     self.strategy,
            "elapsed_ms":   round(self.elapsed_ms, 1),
            "reasoning":    self.reasoning,
            "fast_path":    self.fast_path,
            "error":        self.error,
        }


# ---------------------------------------------------------------------------
# Groq client helper
# ---------------------------------------------------------------------------

def _get_groq_client(groq_client=None):
    """Return the provided client or create a new one.
    Reads GROQ_MEMORY_API_KEY first, falls back to GROQ_API_KEY."""
    if groq_client is not None:
        return groq_client
    import os
    from groq import Groq
    api_key = os.getenv("GROQ_MEMORY_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set — cannot run intent classification")
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

def classify_intent(
    question:    str,
    groq_client=None,
    model:       str = _INTENT_MODEL,
    timeout:     float = 8.0,
) -> IntentResult:
    """
    Classify a customer support query into one of 7 intents.

    Fast path: if the query matches a hard-coded escalation regex, the call
    returns immediately without hitting Groq (< 1 ms, no API cost).

    Slow path: sends the query to Groq llama-3.1-8b-instant for classification
    and parses the JSON response.

    Falls back gracefully to intent="general", strategy="rag" on any error,
    so the pipeline always continues rather than crashing.

    Args:
        question:    The user's raw question string.
        groq_client: Optional pre-initialised Groq client.
        model:       Groq model to use (default: llama-3.1-8b-instant).
        timeout:     HTTP timeout in seconds.

    Returns:
        IntentResult
    """
    t0 = time.perf_counter()

    # ── Fast-path: regex escalation detection ──────────────────────────────
    if _FAST_ESCALATE_PATTERNS.search(question):
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(f"[INTENT] fast-path escalation detected in {elapsed:.1f} ms")
        return IntentResult(
            intent="escalation_request",
            confidence=1.0,
            escalate_now=True,
            strategy="escalate",
            elapsed_ms=elapsed,
            reasoning="Regex pattern matched an explicit escalation request.",
            fast_path=True,
        )

    # ── Slow-path: Groq classification ─────────────────────────────────────
    intent_list_str = "\n".join(
        f"  - {k}: {v}" for k, v in INTENTS.items()
    )
    prompt = _CLASSIFICATION_PROMPT.format(
        intent_list=intent_list_str,
        question=question.strip(),
    )

    try:
        client = _get_groq_client(groq_client)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
            timeout=timeout,
        )
        raw = response.choices[0].message.content or ""
        # Strip Qwen3-style <think>…</think> blocks
        raw = _THINK_RE.sub("", raw).strip()

        # Extract JSON — handle markdown code fences
        json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON object found in response: {raw[:200]}")

        parsed = json.loads(json_match.group(0))
        intent     = parsed.get("intent", "general").strip().lower()
        confidence = float(parsed.get("confidence", 0.5))
        reasoning  = parsed.get("reasoning", "")

        # Validate intent — default to "general" if unknown
        if intent not in INTENTS:
            logger.warning(f"[INTENT] Unknown intent '{intent}' — defaulting to 'general'")
            intent = "general"

        escalate_now = intent in _ESCALATE_NOW_INTENTS
        strategy     = "escalate" if escalate_now else "rag"

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[INTENT] '{intent}' (conf={confidence:.2f}, strategy={strategy}) "
            f"in {elapsed:.1f} ms"
        )
        return IntentResult(
            intent=intent,
            confidence=confidence,
            escalate_now=escalate_now,
            strategy=strategy,
            elapsed_ms=elapsed,
            reasoning=reasoning,
        )

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.warning(f"[INTENT] Classification failed ({exc}) — defaulting to general/rag")
        return IntentResult(
            intent="general",
            confidence=0.0,
            escalate_now=False,
            strategy="rag",
            elapsed_ms=elapsed,
            error=str(exc),
        )
