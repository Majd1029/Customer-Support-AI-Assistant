"""
groq_retry.py — Shared Groq API call helper with 429 retry logic.

Centralises the retry/back-off pattern that was previously copy-pasted across
summariser.py, user_memory.py, and semantic_memory.py.

Usage
-----
from file_preparation.utils.groq_retry import call_groq_with_retry

response_text = call_groq_with_retry(
    groq_client,
    model="qwen/qwen3-32b",
    prompt="Summarise this text…",
    max_tokens=512,
    temperature=0.0,
    timeout=30,
    label="[MEMORY] summariser",   # prefix shown in warning logs
)
"""

from __future__ import annotations

import re
import time

from loguru import logger

_MAX_RETRIES        = 3
_RETRY_DEFAULT_WAIT = 35.0  # seconds — used when Groq's body lacks a wait time
_RETRY_AFTER_RE     = re.compile(r"try again in (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
# Strip <think>…</think> blocks emitted by Qwen3 before the actual response
_THINK_RE           = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def call_groq_with_retry(
    groq_client,
    *,
    model: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: int = 60,
    label: str = "[GROQ]",
    strip_think: bool = True,
) -> str:
    """
    Call ``groq_client.chat.completions.create`` with a single user message
    and return the response text.

    Retries up to ``_MAX_RETRIES`` times on HTTP 429, parsing the exact
    retry-after duration from Groq's error body when available and falling
    back to ``_RETRY_DEFAULT_WAIT``.  Raises on any non-429 error or after
    exhausting retries.

    Parameters
    ----------
    groq_client:
        An initialised ``groq.Groq`` client instance.
    model:
        Groq model identifier, e.g. ``"qwen/qwen3-32b"``.
    prompt:
        The user-turn prompt text.
    max_tokens:
        Maximum tokens in the completion.
    temperature:
        Sampling temperature (0 = deterministic).
    timeout:
        Per-request HTTP timeout in seconds.
    label:
        Short prefix for log messages (helps identify which caller hit a 429).
    strip_think:
        When True, remove ``<think>…</think>`` blocks from the response before
        returning (needed for Qwen3 models that emit chain-of-thought XML).

    Returns
    -------
    str
        The stripped response text.
    """
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = groq_client.chat.completions.create(
                model       = model,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = max_tokens,
                temperature = temperature,
                timeout     = timeout,
            )
            text = resp.choices[0].message.content.strip()
            if strip_think:
                text = _THINK_RE.sub("", text).strip()
            return text

        except Exception as exc:
            exc_str      = str(exc)
            is_rate_limit = "429" in exc_str or "rate_limit" in exc_str.lower()

            if is_rate_limit and attempt < _MAX_RETRIES:
                m    = _RETRY_AFTER_RE.search(exc_str)
                wait = float(m.group(1)) + 2.0 if m else _RETRY_DEFAULT_WAIT
                logger.warning(
                    f"{label}: Groq 429 — waiting {wait:.1f}s "
                    f"(attempt {attempt + 1}/{_MAX_RETRIES})"
                )
                time.sleep(wait)
            else:
                raise
