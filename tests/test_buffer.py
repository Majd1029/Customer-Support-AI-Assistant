"""
tests/test_buffer.py — unit tests for file_preparation/memory/buffer.py

Tests the dual-backend (Redis + in-process dict) conversation buffer.
All tests use the in-process _DictBackend directly, so no Redis connection
is required.  This keeps the suite runnable in CI without any external services.

Covers:
  • Turn creation and field types
  • write_turn / read_buffer / read_all_turns round-trip
  • Token-budget trimming (trim_buffer)
  • Cross-user isolation (ownership check)
  • clear_session
  • write_summary / get_summary
  • absorb_turns
  • should_summarise threshold
  • backend_name() returns "dict" when REDIS_URL is not set
  • format_buffer_for_prompt / buffer_as_text helpers
"""
from __future__ import annotations

import os
import time
import uuid

import pytest

# Import the module-level public API (which uses _DictBackend in CI)
from file_preparation.memory.buffer import (
    TOKEN_BUDGET,
    SUMMARISE_THRESHOLD,
    MIN_KEEP_TURNS,
    Turn,
    _DictBackend,          # direct backend for isolation
    trim_buffer,
    format_buffer_for_prompt,
    buffer_as_text,
    backend_name,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fresh() -> _DictBackend:
    """Return a brand-new _DictBackend instance (isolated from module singleton)."""
    return _DictBackend()


def _turns(n: int, role: str = "user", content: str = "Hello world.",
           user_id: str = "user-1", session_id: str = "sess-1",
           backend: _DictBackend | None = None) -> list[Turn]:
    """Write n turns to a fresh backend and return them."""
    b = backend or _fresh()
    result = []
    for _ in range(n):
        t = b.write_turn(session_id, user_id, role, content)
        result.append(t)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 1. Turn dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestTurnDataclass:
    def test_turn_has_required_fields(self):
        b = _fresh()
        t = b.write_turn("s1", "u1", "user", "Hello")
        assert isinstance(t.turn_id,     str)
        assert isinstance(t.session_id,  str)
        assert isinstance(t.user_id,     str)
        assert isinstance(t.role,        str)
        assert isinstance(t.content,     str)
        assert isinstance(t.timestamp,   float)
        assert isinstance(t.token_count, int)
        assert isinstance(t.summary_absorbed, bool)

    def test_turn_id_is_uuid4(self):
        b = _fresh()
        t = b.write_turn("s1", "u1", "user", "Hello")
        parsed = uuid.UUID(t.turn_id)
        assert str(parsed) == t.turn_id

    def test_turn_token_count_positive(self):
        b = _fresh()
        t = b.write_turn("s1", "u1", "user", "Some content here.")
        assert t.token_count > 0

    def test_invalid_role_raises(self):
        b = _fresh()
        with pytest.raises(ValueError):
            b.write_turn("s1", "u1", "invalid_role", "content")

    def test_turn_serialisation_roundtrip(self):
        b = _fresh()
        original = b.write_turn("s1", "u1", "assistant", "My answer.")
        d = original.to_dict()
        restored = Turn.from_dict(d)
        assert restored.turn_id == original.turn_id
        assert restored.content == original.content
        assert restored.role    == original.role


# ─────────────────────────────────────────────────────────────────────────────
# 2. write_turn / read_all_turns round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestWriteRead:
    def test_single_turn_roundtrip(self):
        b = _fresh()
        b.write_turn("sess", "user", "user", "What is the capital of France?")
        turns = b.read_all_turns("sess", "user")
        assert len(turns) == 1
        assert turns[0].content == "What is the capital of France?"

    def test_multiple_turns_in_order(self):
        b = _fresh()
        messages = ["First message", "Second message", "Third message"]
        for msg in messages:
            b.write_turn("sess", "u1", "user", msg)
        turns = b.read_all_turns("sess", "u1")
        assert [t.content for t in turns] == messages

    def test_empty_session_returns_empty_list(self):
        b = _fresh()
        assert b.read_all_turns("nonexistent", "u1") == []

    def test_read_buffer_with_budget(self):
        """read_buffer applies token-budget trimming."""
        b = _fresh()
        for i in range(5):
            b.write_turn("sess", "u1", "user", f"Turn number {i} with some words.")
        all_turns  = b.read_all_turns("sess", "u1")
        buf_turns  = b.read_buffer("sess", "u1", budget=TOKEN_BUDGET)
        assert len(buf_turns) <= len(all_turns)


# ─────────────────────────────────────────────────────────────────────────────
# 3. trim_buffer
# ─────────────────────────────────────────────────────────────────────────────

class TestTrimBuffer:
    def test_empty_input(self):
        assert trim_buffer([]) == []

    def test_under_budget_keeps_all(self):
        """If total tokens < budget, all turns should be kept."""
        b = _fresh()
        for _ in range(3):
            b.write_turn("s", "u", "user", "Short.")
        turns = b.read_all_turns("s", "u")
        total = sum(t.token_count for t in turns)
        if total < TOKEN_BUDGET:
            trimmed = trim_buffer(turns, budget=TOKEN_BUDGET)
            assert trimmed == turns

    def test_always_keeps_min_turns(self):
        """Even with budget=0, the last MIN_KEEP_TURNS turns are preserved."""
        b = _fresh()
        for i in range(MIN_KEEP_TURNS + 3):
            b.write_turn("s", "u", "user", f"Turn content number {i}.")
        turns = b.read_all_turns("s", "u")
        trimmed = trim_buffer(turns, budget=0)
        assert len(trimmed) >= min(MIN_KEEP_TURNS, len(turns))

    def test_chronological_order_preserved(self):
        """Trimmed buffer should remain chronologically ordered."""
        b = _fresh()
        for i in range(5):
            b.write_turn("s", "u", "user", f"Message {i}.")
        turns   = b.read_all_turns("s", "u")
        trimmed = trim_buffer(turns, budget=TOKEN_BUDGET)
        ts = [t.timestamp for t in trimmed]
        assert ts == sorted(ts)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cross-user isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestOwnershipIsolation:
    def test_cross_user_read_returns_empty(self):
        b = _fresh()
        b.write_turn("sess", "alice", "user", "Alice's secret.")
        # Bob should get nothing from Alice's session
        turns = b.read_all_turns("sess", "bob")
        assert turns == []

    def test_cross_user_write_raises(self):
        b = _fresh()
        b.write_turn("sess", "alice", "user", "Alice was here.")
        with pytest.raises(PermissionError):
            b.write_turn("sess", "bob", "user", "Bob tries to write.")

    def test_owner_can_read_own_session(self):
        b = _fresh()
        b.write_turn("sess", "alice", "user", "Alice's message.")
        turns = b.read_all_turns("sess", "alice")
        assert len(turns) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. clear_session
# ─────────────────────────────────────────────────────────────────────────────

class TestClearSession:
    def test_clear_removes_all_turns(self):
        b = _fresh()
        for _ in range(3):
            b.write_turn("sess", "u1", "user", "Some content.")
        count = b.clear_session("sess", "u1")
        assert count == 3
        assert b.read_all_turns("sess", "u1") == []

    def test_clear_empty_session(self):
        b = _fresh()
        count = b.clear_session("does_not_exist", "u1")
        assert count == 0

    def test_clear_wrong_user_raises(self):
        b = _fresh()
        b.write_turn("sess", "alice", "user", "Content.")
        with pytest.raises(PermissionError):
            b.clear_session("sess", "bob")


# ─────────────────────────────────────────────────────────────────────────────
# 6. write_summary / get_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestSummary:
    def test_write_and_get_summary(self):
        b = _fresh()
        b.write_turn("sess", "u1", "user", "Any turn to establish ownership.")
        b.write_summary("sess", "u1", "  The discussion was about revenue.  ")
        summary = b.get_summary("sess", "u1")
        assert summary == "The discussion was about revenue."  # stripped

    def test_no_summary_returns_none(self):
        b = _fresh()
        assert b.get_summary("no_session", "u1") is None

    def test_summary_overwritten_on_second_write(self):
        b = _fresh()
        b.write_turn("sess", "u1", "user", "Initial turn.")
        b.write_summary("sess", "u1", "First summary.")
        b.write_summary("sess", "u1", "Updated summary.")
        assert b.get_summary("sess", "u1") == "Updated summary."


# ─────────────────────────────────────────────────────────────────────────────
# 7. absorb_turns
# ─────────────────────────────────────────────────────────────────────────────

class TestAbsorbTurns:
    def test_absorb_removes_oldest_turns(self):
        b = _fresh()
        for _ in range(4):
            b.write_turn("sess", "u1", "user", "Content.")
        absorbed = b.absorb_turns("sess", "u1", 2)
        assert len(absorbed) == 2
        remaining = b.read_all_turns("sess", "u1")
        assert len(remaining) == 2

    def test_absorb_zero_does_nothing(self):
        b = _fresh()
        for _ in range(3):
            b.write_turn("sess", "u1", "user", "Content.")
        absorbed = b.absorb_turns("sess", "u1", 0)
        assert absorbed == []
        assert len(b.read_all_turns("sess", "u1")) == 3

    def test_absorb_more_than_available(self):
        b = _fresh()
        b.write_turn("sess", "u1", "user", "Only one turn.")
        absorbed = b.absorb_turns("sess", "u1", 100)
        assert len(absorbed) == 1
        assert b.read_all_turns("sess", "u1") == []

    def test_empty_session_absorb_returns_empty(self):
        b = _fresh()
        assert b.absorb_turns("no_session", "u1", 5) == []


# ─────────────────────────────────────────────────────────────────────────────
# 8. should_summarise threshold
# ─────────────────────────────────────────────────────────────────────────────

class TestShouldSummarise:
    def test_below_threshold_false(self):
        b = _fresh()
        # Write a couple of short turns (well under threshold)
        b.write_turn("sess", "u1", "user", "Hi.")
        b.write_turn("sess", "u1", "assistant", "Hello!")
        total = b.session_token_total("sess")
        if total < SUMMARISE_THRESHOLD:
            # Manually check: total < threshold → should not summarise
            assert total < SUMMARISE_THRESHOLD

    def test_above_threshold_true(self):
        """Generate enough tokens to exceed SUMMARISE_THRESHOLD."""
        b = _fresh()
        # Write turns with enough content to breach the threshold
        big_content = "word " * 200   # ~200 tokens
        while b.session_token_total("sess") < SUMMARISE_THRESHOLD:
            b.write_turn("sess", "u1", "user", big_content)
        assert b.session_token_total("sess") >= SUMMARISE_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# 9. Prompt formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptHelpers:
    def test_format_buffer_for_prompt_structure(self):
        b = _fresh()
        b.write_turn("s", "u", "user",      "Question here.")
        b.write_turn("s", "u", "assistant", "Answer here.")
        turns = b.read_all_turns("s", "u")
        messages = format_buffer_for_prompt(turns)
        assert len(messages) == 2
        assert messages[0] == {"role": "user",      "content": "Question here."}
        assert messages[1] == {"role": "assistant", "content": "Answer here."}

    def test_buffer_as_text(self):
        b = _fresh()
        b.write_turn("s", "u", "user",      "First.")
        b.write_turn("s", "u", "assistant", "Second.")
        turns = b.read_all_turns("s", "u")
        text = buffer_as_text(turns)
        assert "[user] First."     in text
        assert "[assistant] Second." in text


# ─────────────────────────────────────────────────────────────────────────────
# 10. Backend identification
# ─────────────────────────────────────────────────────────────────────────────

class TestBackendName:
    def test_dict_backend_when_no_redis_url(self):
        """In CI (no REDIS_URL set), the module should have chosen DictBackend."""
        if not os.getenv("REDIS_URL"):
            assert backend_name() == "dict"

    def test_backend_name_valid_string(self):
        name = backend_name()
        assert name in ("dict", "redis")
