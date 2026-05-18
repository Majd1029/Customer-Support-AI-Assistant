"""
file_preparation/memory/__init__.py

Public surface of the memory layer.

    from file_preparation.memory import (
        # Sprint 1-3: conversation buffer, rewriter, summariser
        write_turn, read_buffer, read_all_turns, clear_session,
        write_summary, get_summary, absorb_turns, should_summarise,
        rewrite_query,
        MemoryContext, load_memory_context,
        summarise_session,
        # Sprint 4: structured user facts
        set_fact, get_facts, get_fact, delete_fact, clear_user_facts,
        extract_and_store_facts, format_facts_for_prompt,
        # Sprint 5: semantic user memory
        remember_preference, recall_preferences, delete_preference,
        delete_all_preferences, list_preferences,
        format_preferences_for_prompt, semantic_memory_available,
        extract_and_store_preferences,
    )
"""

from .buffer import (
    Turn,
    write_turn,
    read_buffer,
    read_all_turns,
    clear_session,
    format_buffer_for_prompt,
    buffer_as_text,
    write_summary,
    get_summary,
    absorb_turns,
    should_summarise,
    list_sessions,
    backend_name,
    TOKEN_BUDGET,
    SUMMARISE_THRESHOLD,
)

from .rewriter import rewrite_query

from .orchestrator import MemoryContext, load_memory_context

from .summariser import summarise_session

# Sprint 4 — structured user facts
from .user_memory import (
    set_fact,
    get_facts,
    get_fact,
    delete_fact,
    clear_user_facts,
    extract_and_store_facts,
    format_facts_for_prompt,
)

# Sprint 5 — semantic user memory
from .semantic_memory import (
    remember_preference,
    recall_preferences,
    delete_preference,
    delete_all_preferences,
    list_preferences,
    format_preferences_for_prompt,
    semantic_memory_available,
    extract_and_store_preferences,
)

__all__ = [
    # buffer (Sprint 1)
    "Turn",
    "write_turn",
    "read_buffer",
    "read_all_turns",
    "clear_session",
    "format_buffer_for_prompt",
    "buffer_as_text",
    "write_summary",
    "get_summary",
    "absorb_turns",
    "should_summarise",
    "list_sessions",
    "backend_name",
    "TOKEN_BUDGET",
    "SUMMARISE_THRESHOLD",
    # rewriter (Sprint 2)
    "rewrite_query",
    # orchestrator
    "MemoryContext",
    "load_memory_context",
    # summariser (Sprint 3)
    "summarise_session",
    # user facts (Sprint 4)
    "set_fact",
    "get_facts",
    "get_fact",
    "delete_fact",
    "clear_user_facts",
    "extract_and_store_facts",
    "format_facts_for_prompt",
    # semantic memory (Sprint 5)
    "remember_preference",
    "recall_preferences",
    "delete_preference",
    "delete_all_preferences",
    "list_preferences",
    "format_preferences_for_prompt",
    "semantic_memory_available",
    "extract_and_store_preferences",
]
