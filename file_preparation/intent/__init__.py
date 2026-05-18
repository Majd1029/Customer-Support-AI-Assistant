"""
file_preparation/intent — Customer Support Intent Classification
"""

from .classifier import classify_intent, IntentResult, INTENTS

__all__ = [
    "classify_intent",
    "IntentResult",
    "INTENTS",
]
