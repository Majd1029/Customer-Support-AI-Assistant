"""
rag_graph — LangGraph-based orchestration layer for the RAG pipeline.

Public API
──────────
    from file_preparation.rag_graph import rag_graph, RAGState

    # Non-streaming (POST /ask)
    final_state = await rag_graph.ainvoke(initial_state)

    # Streaming (POST /ask/stream)
    async for event in rag_graph.astream(initial_state, stream_mode="custom"):
        # event is a dict already — emit as SSE:  data: {json.dumps(event)}
        ...
"""
from .state import RAGState
from .graph import rag_graph, build_rag_graph

__all__ = ["rag_graph", "build_rag_graph", "RAGState"]
