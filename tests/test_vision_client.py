"""
tests/test_vision_client.py — provider selection for OCR / image captions.

file_processor/vision_client.py keeps Groq as the default and switches to any
OpenAI-compatible endpoint (e.g. Gemini) when VISION_API_BASE, VISION_API_KEY
and VISION_MODEL are all set. No network access: the OpenAI SDK is stubbed.
"""
from __future__ import annotations

import sys
import types

import pytest

from file_processor import vision_client as vc


@pytest.fixture
def external(monkeypatch):
    monkeypatch.setattr(vc, "VISION_API_BASE", "https://example.test/v1/")
    monkeypatch.setattr(vc, "VISION_API_KEY", "key-123")
    monkeypatch.setattr(vc, "VISION_MODEL", "vision-model-x")
    monkeypatch.setattr(vc, "VISION_MIN_TOKENS", 1024)

    calls: dict = {}

    class FakeOpenAI:
        def __init__(self, base_url, api_key):
            calls["init"] = (base_url, api_key)
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kw: calls.setdefault("create", kw) or "resp")
            )

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    return calls


def test_default_uses_groq(monkeypatch):
    monkeypatch.setattr(vc, "VISION_API_BASE", "")
    client = vc.make_vision_client("gsk_test")
    assert type(client).__module__.startswith("groq")
    assert not vc.external_vision_enabled()
    assert vc.vision_available("gsk_test") and not vc.vision_available("")


def test_partial_config_stays_on_groq(monkeypatch):
    monkeypatch.setattr(vc, "VISION_API_BASE", "https://example.test/v1/")
    monkeypatch.setattr(vc, "VISION_API_KEY", "")
    monkeypatch.setattr(vc, "VISION_MODEL", "m")
    assert not vc.external_vision_enabled()


def test_external_provider_overrides_model_and_raises_token_floor(external):
    client = vc.make_vision_client("gsk_ignored")
    client.chat.completions.create(model="meta-llama/llama-4-scout", messages=[], max_tokens=120)

    assert external["init"] == ("https://example.test/v1/", "key-123")
    sent = external["create"]
    assert sent["model"] == "vision-model-x"
    assert sent["max_tokens"] == 1024


def test_external_keeps_larger_token_limits(external):
    vc.make_vision_client(None).chat.completions.create(model="x", messages=[], max_tokens=4096)
    assert external["create"]["max_tokens"] == 4096


def test_external_counts_as_available_without_groq_key(external):
    assert vc.external_vision_enabled()
    assert vc.vision_available("")
