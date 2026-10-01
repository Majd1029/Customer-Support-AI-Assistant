"""
vision_client.py — chooses the provider for image understanding (OCR + captions).

By default the vision call sites (gemma4.py, ocr.py, groq_client.py) use Groq,
exactly as before. When VISION_API_BASE, VISION_API_KEY and VISION_MODEL are
set, they instead call any OpenAI-compatible chat-completions endpoint that
accepts images — e.g. Google Gemini's free tier:

    VISION_API_BASE = https://generativelanguage.googleapis.com/v1beta/openai/
    VISION_API_KEY  = <key from https://aistudio.google.com/apikey>
    VISION_MODEL    = gemini-flash-latest

The returned client exposes the same ``client.chat.completions.create(...)``
call as the Groq SDK, so call sites only swap how the client is built.

    VISION_MIN_TOKENS  floor for max_tokens on the external provider (default 1024).
                       "Thinking" models count reasoning tokens against
                       max_tokens; the small caps tuned for Groq (e.g. 120 for
                       captions) would otherwise leave the answer empty.
"""

from __future__ import annotations

import os
from typing import Any

VISION_API_BASE   = os.getenv("VISION_API_BASE", "").strip()
VISION_API_KEY    = os.getenv("VISION_API_KEY", "").strip()
VISION_MODEL      = os.getenv("VISION_MODEL", "").strip()
VISION_MIN_TOKENS = int(os.getenv("VISION_MIN_TOKENS", "1024"))


def external_vision_enabled() -> bool:
    """True when an OpenAI-compatible vision provider is fully configured."""
    return bool(VISION_API_BASE and VISION_API_KEY and VISION_MODEL)


def vision_available(groq_api_key: str | None) -> bool:
    """True when either the external provider or Groq can serve vision calls."""
    return external_vision_enabled() or bool(groq_api_key)


def vision_provider_label(groq_model: str) -> str:
    return f"{VISION_MODEL} via {VISION_API_BASE}" if external_vision_enabled() else f"Groq {groq_model}"


class _Completions:
    def __init__(self, client: Any):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        kwargs["model"] = VISION_MODEL
        if kwargs.get("max_tokens") is not None:
            kwargs["max_tokens"] = max(int(kwargs["max_tokens"]), VISION_MIN_TOKENS)
        return self._client.chat.completions.create(**kwargs)


class _Chat:
    def __init__(self, client: Any):
        self.completions = _Completions(client)


class _ExternalVisionClient:
    """OpenAI-SDK client with the Groq-style ``.chat.completions.create`` surface."""

    def __init__(self) -> None:
        from openai import OpenAI
        self.chat = _Chat(OpenAI(base_url=VISION_API_BASE, api_key=VISION_API_KEY))


def make_vision_client(groq_api_key: str | None) -> Any:
    """
    Return a client for image requests: the external provider when configured,
    otherwise a Groq client (unchanged behaviour). The external client always
    uses VISION_MODEL, whatever ``model=`` the call site passes.
    """
    if external_vision_enabled():
        return _ExternalVisionClient()
    from groq import Groq
    return Groq(api_key=groq_api_key)
