"""
file_preparation/evaluation/__init__.py

Public surface of the LLM-as-a-Judge evaluation package.
"""

from .judge import (
    judge_answer,
    batch_judge,
    JudgeResult,
    DimensionScore,
)

from .comparison_graph_streaming import stream_comparison

__all__ = [
    "judge_answer",
    "batch_judge",
    "JudgeResult",
    "DimensionScore",
    "stream_comparison",
]
