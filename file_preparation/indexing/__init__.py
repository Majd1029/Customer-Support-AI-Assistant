"""
file_preparation/indexing — Qdrant store + embedding indexing pipeline.

Retrieval logic (retrieve, HyDE, MMR, decompose, reranking) lives in
file_preparation/retrieval/ — import from there for any search/query work.

Public API
----------
get_client(host, port, ...)        → QdrantClient
ensure_collection(client, name)    → None
upsert(client, points, collection) → int
search(client, dense, sparse, ...) → list[dict]
build_point(chunk_id, dense, ...)  → PointStruct
delete_by_source(client, source)   → None
list_indexed_documents(client)     → list[dict]
get_collection_stats(client)       → dict
get_neighbors(client, source, ...) → list[dict]
get_chunks_by_doc_id(client, doc_id) → list[dict]
create_snapshot(client, ...)       → str
list_snapshots(client, ...)        → list[dict]
delete_snapshot(client, ...)       → bool

index_chunks(chunks, client, ...)  → stats dict
index_document(path, client, ...)  → stats dict
normalize_for_embedding(text)      → str
"""

from .store   import (                                                   # noqa: E402
    get_client,
    ensure_collection,
    upsert,
    search,
    build_point,
    delete_by_source,
    list_indexed_documents,
    get_collection_stats,
    get_neighbors,
    get_chunks_by_doc_id,
    create_snapshot,
    list_snapshots,
    delete_snapshot,
    COLLECTION_NAME,
)
from .indexer import (                                                   # noqa: E402
    index_chunks,
    index_document,
    normalize_for_embedding,
)

__all__ = [
    # store
    "get_client",
    "ensure_collection",
    "upsert",
    "search",
    "build_point",
    "delete_by_source",
    "list_indexed_documents",
    "get_collection_stats",
    "get_neighbors",
    "get_chunks_by_doc_id",
    "create_snapshot",
    "list_snapshots",
    "delete_snapshot",
    "COLLECTION_NAME",
    # indexer
    "index_chunks",
    "index_document",
    "normalize_for_embedding",
]
