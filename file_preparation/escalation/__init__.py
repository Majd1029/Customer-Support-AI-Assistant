"""
file_preparation/escalation — Customer Support Escalation Handler
"""

from .handler import should_escalate, EscalationDecision, get_escalation_message

__all__ = [
    "should_escalate",
    "EscalationDecision",
    "get_escalation_message",
]
