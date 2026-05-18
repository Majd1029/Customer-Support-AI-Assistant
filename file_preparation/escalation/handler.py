"""
escalation/handler.py — Customer Support Escalation Handler
============================================================

Determines whether a support conversation should be escalated to a human agent
and produces a customer-facing message to accompany the escalation event.

Escalation can be triggered at two points in the pipeline:

  1. PRE-RETRIEVAL (intent-based)
     When classify_intent() returns escalate_now=True (e.g. the customer
     explicitly asked to speak to a human). RAG is skipped entirely.

  2. POST-GENERATION (quality-based)
     After the LLM answer is generated and the comparison graph evaluates it.
     Escalation fires when:
       - eval verdict is "fail" or "off_topic"
       - OR the answer was a no_answer refusal
       - OR the eval verdict is "low_confidence" (< 0.50 grounding score)

Usage
-----
    from file_preparation.escalation import should_escalate, EscalationDecision

    decision = should_escalate(
        intent=intent_result,       # IntentResult or None
        no_answer=True,
        eval_verdict="fail",
    )

    if decision.should_escalate:
        # yield escalation SSE event with decision.message
        ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Customer-facing escalation messages
# (keyed by trigger reason for easy localisation/customisation)
# ---------------------------------------------------------------------------

_MESSAGES: dict[str, str] = {
    "explicit_request": (
        "I'll connect you with a human support agent right away. "
        "A member of our team will be with you shortly. "
        "Please hold while we transfer your conversation."
    ),
    "no_answer": (
        "I wasn't able to find a satisfactory answer to your question in our knowledge base. "
        "I'm escalating this to our support team who can assist you directly. "
        "Please expect a response within 1 business day."
    ),
    "low_quality": (
        "I want to make sure you get the most accurate help possible. "
        "I'm escalating your question to a human specialist who can give you a definitive answer. "
        "Our team will follow up with you shortly."
    ),
    "complaint": (
        "I understand your frustration and I sincerely apologise for the experience you've had. "
        "I'm escalating this to a senior support agent who will prioritise your case. "
        "You can expect to hear from us very soon."
    ),
    "low_confidence": (
        "I have some information on this topic, but I'm not fully confident it covers your "
        "specific situation. I'm escalating to a specialist who can give you a precise answer. "
        "Our team will follow up shortly."
    ),
}

# Verdict values that trigger post-generation escalation
_ESCALATE_VERDICTS = {"fail", "off_topic", "low_confidence"}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class EscalationDecision:
    """Outcome of an escalation check."""
    should_escalate: bool
    reason:          str   # machine-readable reason key
    message:         str   # customer-facing message (empty when should_escalate=False)
    trigger:         str   # "pre_retrieval" | "post_generation" | "none"

    def to_dict(self) -> dict:
        return {
            "should_escalate": self.should_escalate,
            "reason":          self.reason,
            "message":         self.message,
            "trigger":         self.trigger,
        }


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def should_escalate(
    intent=None,           # IntentResult | None  — from pre-retrieval classifier
    no_answer:    bool  = False,
    eval_verdict: Optional[str] = None,  # "pass" | "fail" | "off_topic" | "low_confidence"
) -> EscalationDecision:
    """
    Decide whether to escalate the current support interaction to a human agent.

    Evaluation order:
      1. Intent says escalate_now (explicit user request)
      2. Intent is "complaint" (high-frustration case)
      3. Answer is a no_answer refusal
      4. Eval verdict is fail / off_topic / low_confidence

    Args:
        intent:       IntentResult from classify_intent(), or None if classification
                      was skipped (e.g. memory_enabled=False with no intent check).
        no_answer:    True when the LLM returned the no-information sentinel.
        eval_verdict: Verdict string from the LangGraph comparison graph.

    Returns:
        EscalationDecision
    """
    # ── 1. Explicit escalation request ──────────────────────────────────────
    if intent is not None and getattr(intent, "escalate_now", False):
        logger.info("[ESCALATION] Triggered by explicit escalation intent")
        return EscalationDecision(
            should_escalate=True,
            reason="explicit_request",
            message=_MESSAGES["explicit_request"],
            trigger="pre_retrieval",
        )

    # ── 2. Complaint intent ─────────────────────────────────────────────────
    if intent is not None and getattr(intent, "intent", "") == "complaint":
        logger.info("[ESCALATION] Triggered by complaint intent")
        return EscalationDecision(
            should_escalate=True,
            reason="complaint",
            message=_MESSAGES["complaint"],
            trigger="pre_retrieval",
        )

    # ── 3. No-answer refusal ────────────────────────────────────────────────
    if no_answer:
        logger.info("[ESCALATION] Triggered by no_answer sentinel")
        return EscalationDecision(
            should_escalate=True,
            reason="no_answer",
            message=_MESSAGES["no_answer"],
            trigger="post_generation",
        )

    # ── 4. Eval verdict quality gate ────────────────────────────────────────
    if eval_verdict is not None and eval_verdict in _ESCALATE_VERDICTS:
        reason = (
            "low_confidence"
            if eval_verdict == "low_confidence"
            else "low_quality"
        )
        logger.info(f"[ESCALATION] Triggered by eval_verdict='{eval_verdict}'")
        return EscalationDecision(
            should_escalate=True,
            reason=reason,
            message=_MESSAGES[reason],
            trigger="post_generation",
        )

    # ── No escalation ───────────────────────────────────────────────────────
    return EscalationDecision(
        should_escalate=False,
        reason="none",
        message="",
        trigger="none",
    )


def get_escalation_message(reason: str) -> str:
    """Return the customer-facing message for a given reason key."""
    return _MESSAGES.get(reason, _MESSAGES["low_quality"])
