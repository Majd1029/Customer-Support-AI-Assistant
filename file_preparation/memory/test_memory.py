"""
file_preparation/memory/test_memory.py

Smoke-test suite for the conversation memory layer.
Covers buffer.py, rewriter.py, summariser.py, orchestrator.py (Sprints 1–3)
and user_memory.py, semantic_memory.py, orchestrator edge cases (Sprints 4–5).
No Groq key, no PostgreSQL, no Qdrant required — all external calls are stubbed.

Run from the project root:
    cd Secure-AI-Assistant-main
    .\\venv\\Scripts\\activate
    python -m pytest file_preparation/memory/test_memory.py -v

Or run directly:
    python file_preparation/memory/test_memory.py
"""

from __future__ import annotations

import asyncio
import sys
import time
import types
import unittest
import unittest.mock
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow running as a script from any directory
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent.parent   # Secure-AI-Assistant-main/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


from file_preparation.memory.buffer import (
    Turn,
    TOKEN_BUDGET,
    SUMMARISE_THRESHOLD,
    MIN_KEEP_TURNS,
    SESSION_TTL,
    _STORE,
    _SUMMARIES,
    _lock,
    write_turn,
    read_buffer,
    clear_session,
    trim_buffer,
    format_buffer_for_prompt,
    buffer_as_text,
    write_summary,
    get_summary,
    absorb_turns,
    should_summarise,
)
from file_preparation.memory.rewriter import (
    rewrite_query,
    _is_self_contained,
    _entity_inject,
    _entity_injection_helped,
    _extract_entities,
)
from file_preparation.memory.summariser import summarise_session
from file_preparation.memory.orchestrator import MemoryContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_turn(
    session_id: str,
    user_id:    str,
    role:       str,
    content:    str,
    token_count: int = 10,
    timestamp:  float | None = None,
) -> Turn:
    """Build a Turn directly (bypasses write_turn's side-effects)."""
    import uuid
    return Turn(
        turn_id          = str(uuid.uuid4()),
        session_id       = session_id,
        user_id          = user_id,
        role             = role,
        content          = content,
        timestamp        = timestamp or time.time(),
        token_count      = token_count,
        summary_absorbed = False,
    )


def _clear_store() -> None:
    """Wipe the in-process store between tests."""
    with _lock:
        _STORE.clear()


# ---------------------------------------------------------------------------
# Buffer tests
# ---------------------------------------------------------------------------

class TestWriteTurn(unittest.TestCase):
    def setUp(self):
        _clear_store()

    def test_valid_user_role(self):
        t = write_turn("s1", "u1", "user", "Hello")
        self.assertEqual(t.role, "user")
        self.assertEqual(t.session_id, "s1")
        self.assertEqual(t.user_id, "u1")
        self.assertEqual(t.content, "Hello")

    def test_valid_assistant_role(self):
        t = write_turn("s2", "u1", "assistant", "Hi there")
        self.assertEqual(t.role, "assistant")

    def test_invalid_role_raises(self):
        with self.assertRaises(ValueError) as ctx:
            write_turn("s3", "u1", "system", "You are a bot")
        self.assertIn("system", str(ctx.exception))

    def test_invalid_role_tool_raises(self):
        with self.assertRaises(ValueError):
            write_turn("s4", "u1", "tool", "result")

    def test_token_count_positive(self):
        t = write_turn("s5", "u1", "user", "This is a test sentence with several words.")
        self.assertGreater(t.token_count, 0)

    def test_cross_user_write_raises(self):
        write_turn("s6", "u1", "user", "First message")
        with self.assertRaises(PermissionError):
            write_turn("s6", "u2", "user", "Hijack attempt")

    def test_multiple_turns_stored(self):
        write_turn("s7", "u1", "user", "Q1")
        write_turn("s7", "u1", "assistant", "A1")
        write_turn("s7", "u1", "user", "Q2")
        turns = read_buffer("s7", "u1")
        self.assertEqual(len(turns), 3)


class TestReadBuffer(unittest.TestCase):
    def setUp(self):
        _clear_store()

    def test_empty_session_returns_empty(self):
        result = read_buffer("no-such-session", "u1")
        self.assertEqual(result, [])

    def test_cross_user_read_returns_empty(self):
        write_turn("s1", "u1", "user", "Secret message")
        result = read_buffer("s1", "u2")   # different user — should be denied
        self.assertEqual(result, [])

    def test_returns_chronological_order(self):
        write_turn("s2", "u1", "user",      "First")
        write_turn("s2", "u1", "assistant", "Second")
        write_turn("s2", "u1", "user",      "Third")
        turns = read_buffer("s2", "u1")
        self.assertEqual([t.content for t in turns], ["First", "Second", "Third"])

    def test_returns_shallow_copy(self):
        write_turn("s3", "u1", "user", "Hello")
        turns_a = read_buffer("s3", "u1")
        turns_a.clear()
        turns_b = read_buffer("s3", "u1")
        self.assertEqual(len(turns_b), 1)


class TestTrimBuffer(unittest.TestCase):
    """Token-budget windowing logic."""

    def _turns(self, n: int, tokens_each: int = 200) -> list[Turn]:
        return [
            _make_turn("s", "u", "user", f"msg{i}", token_count=tokens_each)
            for i in range(n)
        ]

    def test_empty_returns_empty(self):
        self.assertEqual(trim_buffer([]), [])

    def test_all_fit_within_budget(self):
        turns = self._turns(4, tokens_each=200)   # 800 total < 1500 budget
        result = trim_buffer(turns, budget=TOKEN_BUDGET)
        self.assertEqual(len(result), 4)

    def test_oldest_dropped_when_over_budget(self):
        # 6 turns × 300 tokens = 1800 > 1500 budget
        turns = self._turns(6, tokens_each=300)
        result = trim_buffer(turns, budget=TOKEN_BUDGET)
        # Fewer turns returned, but still within budget (plus MIN_KEEP_TURNS guarantee)
        total_tokens = sum(t.token_count for t in result)
        self.assertGreater(total_tokens, 0)
        # Last MIN_KEEP_TURNS must always be present
        self.assertEqual(result[-1].content, "msg5")
        self.assertEqual(result[-2].content, "msg4")

    def test_min_keep_turns_always_preserved(self):
        # Single massive turn that exceeds the budget on its own
        turns = [_make_turn("s", "u", "user", "huge", token_count=TOKEN_BUDGET + 500)]
        result = trim_buffer(turns, budget=TOKEN_BUDGET)
        # Must still return that 1 turn (MIN_KEEP_TURNS=2, but only 1 turn exists)
        self.assertEqual(len(result), 1)

    def test_chronological_order_preserved(self):
        turns = self._turns(3, tokens_each=100)
        result = trim_buffer(turns, budget=TOKEN_BUDGET)
        contents = [t.content for t in result]
        self.assertEqual(contents, sorted(contents, key=lambda c: int(c[3:])))


class TestClearSession(unittest.TestCase):
    def setUp(self):
        _clear_store()

    def test_clear_own_session(self):
        write_turn("s1", "u1", "user", "Hello")
        count = clear_session("s1", "u1")
        self.assertEqual(count, 1)
        self.assertEqual(read_buffer("s1", "u1"), [])

    def test_clear_other_session_raises(self):
        write_turn("s2", "u1", "user", "Hello")
        with self.assertRaises(PermissionError):
            clear_session("s2", "u2")

    def test_clear_nonexistent_session_returns_zero(self):
        count = clear_session("ghost-session", "u1")
        self.assertEqual(count, 0)


class TestFormatHelpers(unittest.TestCase):
    def test_format_buffer_for_prompt(self):
        turns = [
            _make_turn("s", "u", "user",      "Hello"),
            _make_turn("s", "u", "assistant", "Hi"),
        ]
        messages = format_buffer_for_prompt(turns)
        self.assertEqual(messages, [
            {"role": "user",      "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ])

    def test_buffer_as_text(self):
        turns = [
            _make_turn("s", "u", "user",      "Hello"),
            _make_turn("s", "u", "assistant", "Hi"),
        ]
        text = buffer_as_text(turns)
        self.assertIn("[user] Hello", text)
        self.assertIn("[assistant] Hi", text)


# ---------------------------------------------------------------------------
# Rewriter tests
# ---------------------------------------------------------------------------

class TestIsSelfContained(unittest.TestCase):
    def test_long_standalone_query_is_self_contained(self):
        q = "What were the Q3 2024 revenue figures broken down by product line?"
        self.assertTrue(_is_self_contained(q))

    def test_short_query_is_not_self_contained(self):
        self.assertFalse(_is_self_contained("What about it?"))

    def test_query_with_pronoun_is_not_self_contained(self):
        q = "Can you explain this topic in more detail for the current period?"
        self.assertFalse(_is_self_contained(q))

    def test_punctuation_does_not_bloat_count(self):
        # 7 real words with trailing punctuation — should NOT be self-contained
        q = "What were those results? Tell me more!"
        self.assertFalse(_is_self_contained(q))

    def test_exactly_eight_words_no_signals(self):
        # Exactly at the threshold — should be self-contained
        q = "Revenue breakdown by region for fiscal year 2024"
        words = len(__import__('re').findall(r'\b\w+\b', q))
        self.assertGreaterEqual(words, 8)
        self.assertTrue(_is_self_contained(q))


class TestEntityInjection(unittest.TestCase):
    def _turns_with(self, *contents: str) -> list[Turn]:
        return [_make_turn("s", "u", "user", c) for c in contents]

    def test_entities_extracted_from_turns(self):
        turns = self._turns_with(
            "Q3 2024 revenue was $4.2B, up 12% YoY.",
            "Cloud Services drove most of the growth.",
        )
        entities = _extract_entities(turns)
        entity_lower = [e.lower() for e in entities]
        self.assertTrue(any("2024" in e or "q3" in e.lower() for e in entities))

    def test_stopwords_filtered(self):
        turns = self._turns_with(
            "Revenue results for the fiscal year are as expected."
        )
        entities = _extract_entities(turns)
        # Generic finance terms should be filtered
        for e in entities:
            self.assertNotIn(e.lower(), {"revenue", "results"})

    def test_entity_prefix_prepended(self):
        turns = self._turns_with("Cloud Services revenue was $4.2B in Q3 2024.")
        rewritten = _entity_inject("What about the margin?", turns)
        self.assertIn(": What about the margin?", rewritten)
        self.assertGreater(len(rewritten), len("What about the margin?"))

    def test_no_entities_returns_query_unchanged(self):
        turns = self._turns_with("the and or but in on at to for")  # all stopwords
        rewritten = _entity_inject("What happened?", turns)
        self.assertEqual(rewritten, "What happened?")


class TestEntityInjectionHelped(unittest.TestCase):
    def test_substantive_entity_helped(self):
        original  = "What about the margin?"
        rewritten = "Cloud Services, Q3 2024: What about the margin?"
        self.assertTrue(_entity_injection_helped(original, rewritten))

    def test_micro_entity_did_not_help(self):
        # "Q3" is only 2 chars — should not count as substantive
        original  = "What happened?"
        rewritten = "Q3: What happened?"
        self.assertFalse(_entity_injection_helped(original, rewritten))

    def test_entity_already_in_query_did_not_help(self):
        # "Cloud Services" already appears in the query
        original  = "What is the Cloud Services margin?"
        rewritten = "Cloud Services: What is the Cloud Services margin?"
        self.assertFalse(_entity_injection_helped(original, rewritten))

    def test_unchanged_did_not_help(self):
        q = "What happened to revenue?"
        self.assertFalse(_entity_injection_helped(q, q))


class TestRewriteQuery(unittest.TestCase):
    def setUp(self):
        _clear_store()

    def _turns(self) -> list[Turn]:
        return [
            _make_turn("s", "u", "user",      "Q3 2024 revenue was $4.2B, driven by Cloud Services."),
            _make_turn("s", "u", "assistant", "Yes, Cloud Services grew 18% YoY."),
        ]

    def test_tier_skip_no_history(self):
        rewritten, tier = rewrite_query(
            "What were the Q3 2024 revenue figures by region?", turns=[]
        )
        self.assertEqual(tier, "skip")
        self.assertEqual(rewritten, "What were the Q3 2024 revenue figures by region?")

    def test_tier_skip_self_contained(self):
        turns = self._turns()
        rewritten, tier = rewrite_query(
            "What were the Q3 2024 revenue figures by product line?",
            turns=turns,
            groq_client=None,
        )
        self.assertEqual(tier, "skip")

    def test_tier_entity_short_query(self):
        turns = self._turns()
        rewritten, tier = rewrite_query(
            "And the margin?",
            turns=turns,
            groq_client=None,   # no groq → stays at entity tier
        )
        # Should inject something from the history
        self.assertIn(tier, ("entity", "skip"))   # entity if injection helped, skip otherwise
        if tier == "entity":
            self.assertIn(": And the margin?", rewritten)

    def test_tier_entity_fallback_when_no_groq(self):
        """With groq_client=None, rewriter never reaches Tier 2."""
        turns = self._turns()
        _, tier = rewrite_query(
            "What about it?",
            turns=turns,
            groq_client=None,
        )
        self.assertIn(tier, ("entity", "skip"))
        self.assertNotEqual(tier, "llm")

    def test_force_llm_without_groq_falls_back_to_entity(self):
        turns = self._turns()
        rewritten, tier = rewrite_query(
            "it?",
            turns=turns,
            groq_client=None,
            force_llm=True,
        )
        # force_llm=True but no groq client → entity fallback
        self.assertEqual(tier, "entity")


# ---------------------------------------------------------------------------
# TTL eviction test
# ---------------------------------------------------------------------------

class TestTTLEviction(unittest.TestCase):
    def setUp(self):
        _clear_store()

    def test_expired_session_evicted_on_write(self):
        # Manually plant an expired session in the store
        old_turn = _make_turn(
            "old-session", "u1", "user", "Old message",
            timestamp=time.time() - SESSION_TTL - 60,  # 1 min past expiry
        )
        with _lock:
            _STORE["old-session"] = [old_turn]

        # Trigger eviction by writing a new turn to a different session
        write_turn("new-session", "u1", "user", "Hello")

        with _lock:
            self.assertNotIn("old-session", _STORE)


# ---------------------------------------------------------------------------
# Summary storage tests (Sprint 3)
# ---------------------------------------------------------------------------

class TestSummaryStorage(unittest.TestCase):
    def setUp(self):
        _clear_store()
        with _lock:
            _SUMMARIES.clear()

    def test_write_and_get_summary(self):
        write_turn("s1", "u1", "user", "Hello")   # establish ownership
        write_summary("s1", "u1", "The user asked about Q3 revenue.")
        result = get_summary("s1", "u1")
        self.assertEqual(result, "The user asked about Q3 revenue.")

    def test_get_summary_none_when_absent(self):
        write_turn("s1", "u1", "user", "Hello")
        result = get_summary("s1", "u1")
        self.assertIsNone(result)

    def test_write_summary_cross_user_raises(self):
        write_turn("s1", "u1", "user", "Hello")
        with self.assertRaises(PermissionError):
            write_summary("s1", "u2", "Hijack attempt")

    def test_get_summary_cross_user_returns_none(self):
        write_turn("s1", "u1", "user", "Hello")
        write_summary("s1", "u1", "Prior context.")
        result = get_summary("s1", "u2")   # different user — should be denied
        self.assertIsNone(result)

    def test_summary_overwritten_on_second_write(self):
        write_turn("s1", "u1", "user", "Hello")
        write_summary("s1", "u1", "First summary.")
        write_summary("s1", "u1", "Updated summary.")
        self.assertEqual(get_summary("s1", "u1"), "Updated summary.")

    def test_summary_cleared_with_session(self):
        write_turn("s1", "u1", "user", "Hello")
        write_summary("s1", "u1", "Some summary.")
        clear_session("s1", "u1")
        # After clear the session is gone; summary should also be gone
        with _lock:
            self.assertNotIn("s1", _SUMMARIES)


class TestAbsorbTurns(unittest.TestCase):
    def setUp(self):
        _clear_store()

    def test_absorb_removes_oldest_turns(self):
        for i in range(5):
            write_turn("s1", "u1", "user", f"msg{i}")
        absorbed = absorb_turns("s1", "u1", 2)
        self.assertEqual(len(absorbed), 2)
        remaining = read_buffer("s1", "u1")
        self.assertEqual(len(remaining), 3)

    def test_absorb_empty_session_returns_empty(self):
        result = absorb_turns("no-such-session", "u1", 3)
        self.assertEqual(result, [])

    def test_absorb_cross_user_raises(self):
        write_turn("s1", "u1", "user", "Hello")
        with self.assertRaises(PermissionError):
            absorb_turns("s1", "u2", 1)

    def test_absorbed_turns_not_re_absorb(self):
        """Absorbing more turns than exist just takes all available."""
        write_turn("s1", "u1", "user", "A")
        write_turn("s1", "u1", "user", "B")
        absorbed = absorb_turns("s1", "u1", 100)   # request more than exist
        self.assertLessEqual(len(absorbed), 2)


class TestShouldSummarise(unittest.TestCase):
    def setUp(self):
        _clear_store()

    def test_empty_session_returns_false(self):
        self.assertFalse(should_summarise("no-session"))

    def test_small_session_returns_false(self):
        write_turn("s1", "u1", "user", "Short message")   # tiny token count
        self.assertFalse(should_summarise("s1"))

    def test_large_session_returns_true(self):
        # Plant turns that exceed SUMMARISE_THRESHOLD directly in the store
        from uuid import uuid4
        heavy_turns = [
            Turn(
                turn_id=str(uuid4()), session_id="s2", user_id="u1",
                role="user", content=f"msg{i}",
                timestamp=time.time(),
                token_count=300,       # 300 × 5 = 1500 > SUMMARISE_THRESHOLD (1200)
                summary_absorbed=False,
            )
            for i in range(5)
        ]
        with _lock:
            _STORE["s2"] = heavy_turns
        self.assertTrue(should_summarise("s2"))


class TestSummariseSessionStub(unittest.TestCase):
    """
    Tests summarise_session() with a stub Groq client — no real API call.
    Verifies the full absorb + write_summary path without network I/O.
    """

    def setUp(self):
        _clear_store()
        with _lock:
            _SUMMARIES.clear()

    def _stub_groq(self, reply: str):
        """Return a minimal Groq client stub that returns `reply`."""
        import types

        msg   = types.SimpleNamespace(content=reply)
        choice = types.SimpleNamespace(message=msg)
        resp  = types.SimpleNamespace(choices=[choice])

        class _FakeGroq:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        return resp

        return _FakeGroq()

    def test_summarise_returns_false_without_groq(self):
        write_turn("s1", "u1", "user", "Hello")
        result = summarise_session("s1", "u1", groq_client=None)
        self.assertFalse(result)

    def test_summarise_returns_false_when_no_evictable_turns(self):
        # Only 1 small turn — trim_buffer keeps everything, nothing to summarise
        write_turn("s1", "u1", "user", "Short message")
        groq = self._stub_groq("Summary of prior conversation.")
        result = summarise_session("s1", "u1", groq_client=groq)
        self.assertFalse(result)   # nothing evictable

    def test_summarise_absorbs_evictable_turns(self):
        """Plant a session with many heavy turns so some are evictable."""
        from uuid import uuid4
        # 6 turns × 300 tokens = 1800 > TOKEN_BUDGET (1500), so 1-2 are evictable
        heavy_turns = [
            Turn(
                turn_id=str(uuid4()), session_id="s2", user_id="u1",
                role="user", content=f"Q3 2024 Cloud Services revenue was $4.2B msg{i}",
                timestamp=time.time(),
                token_count=300,
                summary_absorbed=False,
            )
            for i in range(6)
        ]
        with _lock:
            _STORE["s2"] = heavy_turns

        groq   = self._stub_groq("The user discussed Q3 2024 Cloud Services revenue of $4.2B.")
        result = summarise_session("s2", "u1", groq_client=groq)
        self.assertTrue(result)

        # Summary should be stored
        summary = get_summary("s2", "u1")
        self.assertIsNotNone(summary)
        self.assertGreater(len(summary), 0)

        # Fewer turns should remain in the buffer
        remaining = read_buffer("s2", "u1")
        self.assertLess(len(remaining), 6)

    def test_summarise_empty_reply_returns_false(self):
        """If Groq returns an empty string, summarise_session should bail."""
        from uuid import uuid4
        heavy_turns = [
            Turn(
                turn_id=str(uuid4()), session_id="s3", user_id="u1",
                role="user", content=f"msg{i}",
                timestamp=time.time(),
                token_count=300,
                summary_absorbed=False,
            )
            for i in range(6)
        ]
        with _lock:
            _STORE["s3"] = heavy_turns

        groq = self._stub_groq("")    # empty reply
        result = summarise_session("s3", "u1", groq_client=groq)
        self.assertFalse(result)


class TestMemoryContextSummary(unittest.TestCase):
    """MemoryContext.summary_as_context() and has_history with summary field."""

    def test_summary_as_context_empty_when_none(self):
        ctx = MemoryContext(session_id="s", user_id="u", rewritten_query="q")
        self.assertEqual(ctx.summary_as_context(), "")

    def test_summary_as_context_returns_stripped_text(self):
        ctx = MemoryContext(
            session_id="s", user_id="u", rewritten_query="q",
            summary="  Prior session summary.  "
        )
        self.assertEqual(ctx.summary_as_context(), "Prior session summary.")

    def test_to_dict_includes_has_summary(self):
        ctx = MemoryContext(session_id="s", user_id="u", rewritten_query="q", summary="S")
        d = ctx.to_dict()
        self.assertIn("has_summary", d)
        self.assertTrue(d["has_summary"])

    def test_to_dict_has_summary_false_when_none(self):
        ctx = MemoryContext(session_id="s", user_id="u", rewritten_query="q")
        self.assertFalse(ctx.to_dict()["has_summary"])


# ---------------------------------------------------------------------------
# Sprint 4 — user_memory.py (in-process dict backend; no PostgreSQL needed)
# ---------------------------------------------------------------------------

# Import the module-level internals we need to reset between tests.
from file_preparation.memory.user_memory import (
    _MEM_STORE,
    _MEM_LOCK,
    set_fact,
    get_facts,
    get_fact,
    delete_fact,
    clear_user_facts,
    extract_and_store_facts,
    format_facts_for_prompt,
)


def _clear_mem_store() -> None:
    with _MEM_LOCK:
        _MEM_STORE.clear()


class TestUserMemoryDict(unittest.TestCase):
    """CRUD against the in-process dict / PG backends."""

    def setUp(self):
        # Clear both the in-process dict AND PostgreSQL (when available)
        # so tests are isolated regardless of which backend is active.
        _clear_mem_store()
        clear_user_facts("u1")
        clear_user_facts("u2")

    def test_set_and_get_fact(self):
        set_fact("u1", "role", "ML engineer")
        self.assertEqual(get_fact("u1", "role"), "ML engineer")

    def test_get_facts_returns_all(self):
        set_fact("u1", "role", "ML engineer")
        set_fact("u1", "language", "Arabic")
        facts = get_facts("u1")
        self.assertEqual(facts.get("role"), "ML engineer")
        self.assertEqual(facts.get("language"), "Arabic")

    def test_get_fact_missing_key_returns_none(self):
        self.assertIsNone(get_fact("u1", "nonexistent"))

    def test_delete_fact_returns_true_when_exists(self):
        set_fact("u1", "role", "engineer")
        self.assertTrue(delete_fact("u1", "role"))
        self.assertIsNone(get_fact("u1", "role"))

    def test_delete_fact_returns_false_when_missing(self):
        self.assertFalse(delete_fact("u1", "ghost"))

    def test_clear_user_facts_removes_all(self):
        set_fact("u1", "role", "engineer")
        set_fact("u1", "language", "Arabic")
        count = clear_user_facts("u1")
        self.assertEqual(count, 2)
        self.assertEqual(get_facts("u1"), {})

    def test_clear_user_facts_on_empty_user_returns_zero(self):
        self.assertEqual(clear_user_facts("u_nobody"), 0)

    def test_keys_normalised_to_lowercase(self):
        set_fact("u1", "ROLE", "engineer")
        self.assertEqual(get_fact("u1", "role"), "engineer")

    def test_empty_key_or_value_is_ignored(self):
        set_fact("u1", "", "value")          # empty key — should not store
        set_fact("u1", "key", "")            # empty value — should not store
        self.assertEqual(get_facts("u1"), {})

    def test_users_are_isolated(self):
        set_fact("u1", "role", "engineer")
        set_fact("u2", "role", "manager")
        self.assertEqual(get_fact("u1", "role"), "engineer")
        self.assertEqual(get_fact("u2", "role"), "manager")


class TestFormatFactsForPrompt(unittest.TestCase):

    def test_empty_facts_returns_empty_string(self):
        self.assertEqual(format_facts_for_prompt({}), "")

    def test_nonempty_facts_returns_block(self):
        result = format_facts_for_prompt({"role": "engineer", "language": "Arabic"})
        self.assertIn("[USER FACTS]", result)
        self.assertIn("[END USER FACTS]", result)
        self.assertIn("role: engineer", result)
        self.assertIn("language: Arabic", result)

    def test_facts_sorted_alphabetically(self):
        result = format_facts_for_prompt({"z_key": "last", "a_key": "first"})
        self.assertLess(result.index("a_key"), result.index("z_key"))


class TestExtractAndStoreFacts(unittest.TestCase):
    """extract_and_store_facts() with a stub Groq client."""

    def setUp(self):
        # Clear both backends so PG state from a previous test doesn't
        # affect the longer-value-wins merge heuristic in the next one.
        _clear_mem_store()
        clear_user_facts("u1")

    # ── Shared stub factories ──────────────────────────────────────────────

    @staticmethod
    def _stub_groq(reply: str):
        """Groq stub that always returns `reply` as the assistant message."""
        msg    = types.SimpleNamespace(content=reply)
        choice = types.SimpleNamespace(message=msg)
        resp   = types.SimpleNamespace(choices=[choice])

        class _FakeGroq:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        return resp
        return _FakeGroq()

    @staticmethod
    def _retry_groq(fail_times: int, success_reply: str):
        """Groq stub that raises a 429 for the first `fail_times` calls."""
        state = {"calls": 0}
        msg    = types.SimpleNamespace(content=success_reply)
        choice = types.SimpleNamespace(message=msg)
        resp   = types.SimpleNamespace(choices=[choice])

        class _RetryGroq:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        state["calls"] += 1
                        if state["calls"] <= fail_times:
                            raise Exception(
                                "429 rate_limit: please try again in 1.0s"
                            )
                        return resp
        return _RetryGroq(), state

    # ── Tests ──────────────────────────────────────────────────────────────

    def test_no_groq_client_returns_empty(self):
        turns = [_make_turn("s", "u", "user", "I am a ML engineer.")]
        result = extract_and_store_facts("u1", turns, groq_client=None)
        self.assertEqual(result, {})

    def test_no_user_turns_returns_empty(self):
        turns = [_make_turn("s", "u", "assistant", "Sure, here is the answer.")]
        groq  = self._stub_groq('{"role": "engineer"}')
        result = extract_and_store_facts("u1", turns, groq_client=groq)
        self.assertEqual(result, {})

    def test_basic_extraction_stores_facts(self):
        turns = [_make_turn("s", "u", "user", "I am a senior ML engineer.")]
        groq  = self._stub_groq('{"role": "senior ML engineer"}')
        result = extract_and_store_facts("u1", turns, groq_client=groq)
        self.assertIn("role", result)
        self.assertEqual(get_fact("u1", "role"), "senior ML engineer")

    def test_longer_value_overwrites_shorter(self):
        """The 'longer value wins' merge heuristic — a new specific value replaces a vague one."""
        set_fact("u1", "role", "engineer")           # existing short value
        turns = [_make_turn("s", "u", "user", "I am a senior ML engineer.")]
        groq  = self._stub_groq('{"role": "senior ML engineer"}')   # longer
        extract_and_store_facts("u1", turns, groq_client=groq)
        self.assertEqual(get_fact("u1", "role"), "senior ML engineer")

    def test_shorter_value_does_not_overwrite_longer(self):
        """A vaguer new value should NOT replace a more specific existing one."""
        set_fact("u1", "role", "senior ML engineer")  # existing specific value
        turns = [_make_turn("s", "u", "user", "I am an engineer.")]
        groq  = self._stub_groq('{"role": "engineer"}')              # shorter
        extract_and_store_facts("u1", turns, groq_client=groq)
        # Should still be the original longer value
        self.assertEqual(get_fact("u1", "role"), "senior ML engineer")

    def test_extraction_sees_all_provided_turns(self):
        """
        Regression: extract_and_store_facts must be called with read_all_turns()
        (not read_buffer()) so facts in older, budget-trimmed turns are not lost.
        This test verifies that when all turns are passed in, facts from the
        oldest ones are still extracted correctly.
        """
        # Simulate a long session: 6 heavy turns, the first contains a key fact
        old_turn  = _make_turn("s", "u", "user", "My project is called SecureRAG.", token_count=300)
        new_turns = [_make_turn("s", "u", "user", f"msg{i}", token_count=300) for i in range(5)]
        all_turns = [old_turn] + new_turns   # 6 turns, oldest has the fact

        groq = self._stub_groq('{"project": "SecureRAG"}')
        result = extract_and_store_facts("u1", all_turns, groq_client=groq)
        self.assertIn("project", result)
        self.assertEqual(get_fact("u1", "project"), "SecureRAG")

    def test_groq_429_retry_succeeds(self):
        """
        Groq returns a 429 on the first attempt; the retry loop should wait
        and succeed on the second call.
        """
        groq, state = self._retry_groq(
            fail_times=1,
            success_reply='{"language": "Arabic"}',
        )
        turns = [_make_turn("s", "u", "user", "I work in Arabic NLP.")]

        # Patch time.sleep so the test does not actually wait
        with unittest.mock.patch(
            "file_preparation.memory.user_memory.time.sleep"
        ):
            result = extract_and_store_facts("u1", turns, groq_client=groq)

        self.assertIn("language", result)
        self.assertEqual(state["calls"], 2)   # confirm retry fired

    def test_invalid_json_from_groq_returns_empty(self):
        groq  = self._stub_groq("not valid json at all")
        turns = [_make_turn("s", "u", "user", "Hello.")]
        result = extract_and_store_facts("u1", turns, groq_client=groq)
        self.assertEqual(result, {})

    def test_think_block_stripped_before_parse(self):
        reply = "<think>Thinking...</think>\n{\"role\": \"engineer\"}"
        groq  = self._stub_groq(reply)
        turns = [_make_turn("s", "u", "user", "I am an engineer.")]
        result = extract_and_store_facts("u1", turns, groq_client=groq)
        self.assertIn("role", result)

    def test_code_fenced_json_is_tolerated(self):
        reply = "```json\n{\"role\": \"engineer\"}\n```"
        groq  = self._stub_groq(reply)
        turns = [_make_turn("s", "u", "user", "I am an engineer.")]
        result = extract_and_store_facts("u1", turns, groq_client=groq)
        self.assertIn("role", result)


# ---------------------------------------------------------------------------
# Sprint 5 — semantic_memory.py
# ---------------------------------------------------------------------------

import file_preparation.memory.semantic_memory as _sem_mod
from file_preparation.memory.semantic_memory import (
    format_preferences_for_prompt,
    extract_and_store_preferences,
)


class TestFormatPreferencesForPrompt(unittest.TestCase):

    def test_empty_list_returns_empty_string(self):
        self.assertEqual(format_preferences_for_prompt([]), "")

    def test_nonempty_returns_block(self):
        result = format_preferences_for_prompt([
            "I prefer concise answers.",
            "Respond in French.",
        ])
        self.assertIn("[USER PREFERENCES]", result)
        self.assertIn("[END USER PREFERENCES]", result)
        self.assertIn("- I prefer concise answers.", result)
        self.assertIn("- Respond in French.", result)

    def test_single_preference(self):
        result = format_preferences_for_prompt(["Use code examples."])
        self.assertIn("- Use code examples.", result)


class TestExtractAndStorePreferences(unittest.TestCase):
    """extract_and_store_preferences() with stubs — no Qdrant or BGE-M3."""

    @staticmethod
    def _stub_groq_prefs(reply: str):
        msg    = types.SimpleNamespace(content=reply)
        choice = types.SimpleNamespace(message=msg)
        resp   = types.SimpleNamespace(choices=[choice])

        class _FakeGroq:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        return resp
        return _FakeGroq()

    def test_no_groq_client_returns_empty(self):
        turns  = [_make_turn("s", "u", "user", "I prefer concise answers.")]
        result = extract_and_store_preferences("u1", turns, groq_client=None)
        self.assertEqual(result, [])

    def test_unavailable_semantic_memory_returns_empty(self):
        """When Qdrant/embedder is not available, extraction silently returns []."""
        turns = [_make_turn("s", "u", "user", "I prefer concise answers.")]
        groq  = self._stub_groq_prefs('["I prefer concise answers."]')
        # Patch _available() to return False (simulates missing Qdrant)
        with unittest.mock.patch.object(_sem_mod, "_available", return_value=False):
            result = extract_and_store_preferences("u1", turns, groq_client=groq)
        self.assertEqual(result, [])

    def test_no_user_turns_returns_empty(self):
        turns = [_make_turn("s", "u", "assistant", "Here is the answer.")]
        groq  = self._stub_groq_prefs('["I prefer concise answers."]')
        with unittest.mock.patch.object(_sem_mod, "_available", return_value=True):
            result = extract_and_store_preferences("u1", turns, groq_client=groq)
        self.assertEqual(result, [])

    def test_extracts_and_stores_preferences(self):
        """Preferences returned by Groq are stored via remember_preference."""
        turns = [_make_turn("s", "u", "user", "I prefer concise answers with code.")]
        groq  = self._stub_groq_prefs(
            '["I prefer concise answers with code.", "Use Python examples."]'
        )
        fake_pref_id = "fake-uuid-1234"

        with unittest.mock.patch.object(_sem_mod, "_available", return_value=True), \
             unittest.mock.patch.object(
                 _sem_mod, "remember_preference", return_value=fake_pref_id
             ) as mock_remember:
            result = extract_and_store_preferences("u1", turns, groq_client=groq)

        self.assertEqual(len(result), 2)
        self.assertIn("I prefer concise answers with code.", result)
        self.assertIn("Use Python examples.", result)
        self.assertEqual(mock_remember.call_count, 2)

    def test_groq_429_retry_fires(self):
        """A 429 on the first Groq call triggers the retry logic."""
        state = {"calls": 0}
        msg    = types.SimpleNamespace(content='["Respond in French."]')
        choice = types.SimpleNamespace(message=msg)
        resp   = types.SimpleNamespace(choices=[choice])

        class _RetryGroq:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        state["calls"] += 1
                        if state["calls"] == 1:
                            raise Exception("429 rate_limit: please try again in 1.0s")
                        return resp

        turns = [_make_turn("s", "u", "user", "Always respond in French.")]

        with unittest.mock.patch.object(_sem_mod, "_available", return_value=True), \
             unittest.mock.patch.object(_sem_mod, "remember_preference", return_value="id1"), \
             unittest.mock.patch("file_preparation.memory.semantic_memory.time.sleep"):
            result = extract_and_store_preferences("u1", turns, groq_client=_RetryGroq())

        self.assertIn("Respond in French.", result)
        self.assertEqual(state["calls"], 2)

    def test_uuid5_idempotency(self):
        """Storing the same preference text twice produces the same point ID."""
        from file_preparation.memory.semantic_memory import _make_id
        id1 = _make_id("user123", "I prefer concise answers.")
        id2 = _make_id("user123", "I prefer concise answers.")
        self.assertEqual(id1, id2)

    def test_uuid5_different_users_different_ids(self):
        """The same preference text for two different users yields different IDs."""
        from file_preparation.memory.semantic_memory import _make_id
        id1 = _make_id("user_a", "I prefer concise answers.")
        id2 = _make_id("user_b", "I prefer concise answers.")
        self.assertNotEqual(id1, id2)

    def test_recall_graceful_degradation_when_unavailable(self):
        """recall_preferences returns [] without raising when Qdrant is down."""
        from file_preparation.memory.semantic_memory import recall_preferences
        with unittest.mock.patch.object(_sem_mod, "_available", return_value=False):
            result = recall_preferences("u1", "any query")
        self.assertEqual(result, [])

    def test_remember_preference_graceful_degradation(self):
        """remember_preference returns '' without raising when unavailable."""
        from file_preparation.memory.semantic_memory import remember_preference
        with unittest.mock.patch.object(_sem_mod, "_available", return_value=False):
            result = remember_preference("u1", "I prefer short answers.")
        self.assertEqual(result, "")

    def test_invalid_json_from_groq_returns_empty(self):
        turns = [_make_turn("s", "u", "user", "I prefer X.")]
        groq  = self._stub_groq_prefs("not json at all")
        with unittest.mock.patch.object(_sem_mod, "_available", return_value=True):
            result = extract_and_store_preferences("u1", turns, groq_client=groq)
        self.assertEqual(result, [])

    def test_think_block_stripped_before_parse(self):
        reply = "<think>Let me think...</think>\n[\"Respond in English.\"]"
        turns = [_make_turn("s", "u", "user", "Always in English please.")]
        groq  = self._stub_groq_prefs(reply)
        with unittest.mock.patch.object(_sem_mod, "_available", return_value=True), \
             unittest.mock.patch.object(_sem_mod, "remember_preference", return_value="id1"):
            result = extract_and_store_preferences("u1", turns, groq_client=groq)
        self.assertIn("Respond in English.", result)


# ---------------------------------------------------------------------------
# Orchestrator edge cases (Sprints 4 + 5 wiring)
# ---------------------------------------------------------------------------

from file_preparation.memory.orchestrator import load_memory_context


class TestOrchestratorEdgeCases(unittest.TestCase):

    def _run(self, coro):
        """Run a coroutine synchronously (works on Python 3.7+)."""
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_memory_disabled_returns_immediately(self):
        """memory_enabled=False must return an empty MemoryContext without any I/O."""
        ctx = self._run(
            load_memory_context(
                session_id      = "s1",
                user_id         = "u1",
                raw_query       = "What is hybrid search?",
                memory_enabled  = False,
            )
        )
        self.assertFalse(ctx.enabled)
        self.assertEqual(ctx.turns, [])
        self.assertEqual(ctx.rewritten_query, "What is hybrid search?")
        self.assertEqual(ctx.rewrite_tier, "skip")
        self.assertIsNone(ctx.summary)
        self.assertEqual(ctx.user_facts, {})
        self.assertEqual(ctx.recalled_preferences, [])

    def test_all_reads_failing_returns_safe_empty_context(self):
        """
        If every I/O read raises, load_memory_context must still return a valid
        MemoryContext with safe empty defaults — the caller must never see an
        exception from the memory layer.
        """
        _boom = RuntimeError("simulated infrastructure failure")

        with unittest.mock.patch(
            "file_preparation.memory.orchestrator.read_buffer",
            side_effect=_boom
        ), unittest.mock.patch(
            "file_preparation.memory.orchestrator.get_summary",
            side_effect=_boom
        ), unittest.mock.patch(
            "file_preparation.memory.orchestrator.get_facts",
            side_effect=_boom
        ), unittest.mock.patch(
            "file_preparation.memory.orchestrator.recall_preferences",
            side_effect=_boom
        ):
            ctx = self._run(
                load_memory_context(
                    session_id     = "s2",
                    user_id        = "u2",
                    raw_query      = "What is BM25?",
                    memory_enabled = True,
                )
            )

        # Must return a valid, usable object despite all reads failing
        self.assertTrue(ctx.enabled)
        self.assertEqual(ctx.turns, [])
        self.assertIsNone(ctx.summary)
        self.assertEqual(ctx.user_facts, {})
        self.assertEqual(ctx.recalled_preferences, [])
        # Query must still be reachable (raw query unchanged since no turns to rewrite)
        self.assertEqual(ctx.rewritten_query, "What is BM25?")

    def test_user_facts_and_preferences_surfaced_in_context(self):
        """
        When reads succeed, user_facts and recalled_preferences must appear
        in the returned MemoryContext and be accessible via helper methods.
        """
        fake_facts = {"role": "ML engineer", "language": "Arabic"}
        fake_prefs = ["I prefer concise answers.", "Respond in French."]

        with unittest.mock.patch(
            "file_preparation.memory.orchestrator.read_buffer", return_value=[]
        ), unittest.mock.patch(
            "file_preparation.memory.orchestrator.get_summary", return_value=None
        ), unittest.mock.patch(
            "file_preparation.memory.orchestrator.get_facts", return_value=fake_facts
        ), unittest.mock.patch(
            "file_preparation.memory.orchestrator.recall_preferences",
            return_value=fake_prefs
        ):
            ctx = self._run(
                load_memory_context(
                    session_id     = "s3",
                    user_id        = "u3",
                    raw_query      = "What is SPLADE?",
                    memory_enabled = True,
                )
            )

        self.assertEqual(ctx.user_facts, fake_facts)
        self.assertEqual(ctx.recalled_preferences, fake_prefs)
        self.assertIn("[USER FACTS]",       ctx.user_facts_as_context())
        self.assertIn("[USER PREFERENCES]", ctx.recalled_preferences_as_context())
        self.assertIn("role: ML engineer",  ctx.user_facts_as_context())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running memory layer smoke tests...")
    unittest.main(verbosity=2)
