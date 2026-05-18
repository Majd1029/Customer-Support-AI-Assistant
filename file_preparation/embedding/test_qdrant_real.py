"""
End-to-end Qdrant test — requires a running Qdrant instance.

Start Qdrant first:
    docker run -p 6333:6333 qdrant/qdrant

Then run:
    python file_preparation\embedding\test_qdrant_real.py

Tests:
  1. Connect to Qdrant
  2. Create (or recreate) a test collection
  3. Embed + upsert 5 chunks
  4. Hybrid search — verify top result is semantically correct
  5. Filter search — verify metadata filters work
  6. Cleanup — delete test collection
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from embedder import encode, encode_query
from store    import get_client, ensure_collection, build_point, upsert, search

COLLECTION = "test_e2e"

print("=" * 60)
print("  End-to-end: BGE-M3 + Qdrant")
print("=" * 60)

# ── 1. Connect ────────────────────────────────────────────────────────────────
print("\n[1] Connecting to Qdrant at localhost:6333 …")
client = get_client()
info   = client.get_collections()
print(f"  Connected  ✓  ({len(info.collections)} existing collection(s))")

# ── 2. Create collection ──────────────────────────────────────────────────────
print(f"\n[2] Creating collection '{COLLECTION}' …")
ensure_collection(client, COLLECTION, recreate=True)
print(f"  Collection ready  ✓")

# ── 3. Embed + upsert ─────────────────────────────────────────────────────────
print("\n[3] Embedding and upserting 5 chunks …")

chunks = [
    {"chunk_id": "c1", "type": "text",  "content": "Paris is the capital of France.",
     "language": "en", "source": "geo.pdf"},
    {"chunk_id": "c2", "type": "text",  "content": "The Eiffel Tower is a famous landmark in Paris.",
     "language": "en", "source": "geo.pdf"},
    {"chunk_id": "c3", "type": "text",  "content": "Machine learning is a subset of artificial intelligence.",
     "language": "en", "source": "ai.pdf"},
    {"chunk_id": "c4", "type": "text",  "content": "Neural networks are inspired by the human brain.",
     "language": "en", "source": "ai.pdf"},
    {"chunk_id": "c5", "type": "table", "content": "Country: France | Capital: Paris | Population: 68M",
     "language": "en", "source": "geo.pdf"},
]

texts = [c["content"] for c in chunks]
emb   = encode(texts)

points = [
    build_point(c["chunk_id"], d, s, {**c})
    for c, d, s in zip(chunks, emb.dense, emb.sparse)
]

total = upsert(client, points, collection=COLLECTION)
assert total == 5
print(f"  Upserted {total} points  ✓")

# ── 4. Hybrid search ──────────────────────────────────────────────────────────
print("\n[4] Hybrid search: 'What is the capital of France?' …")
q       = encode_query("What is the capital of France?")
results = search(client, q.dense[0], q.sparse[0], collection=COLLECTION, limit=3)

print(f"  Top {len(results)} results:")
for r in results:
    print(f"    [{r['score']:.4f}]  ({r['payload']['source']})  {r['payload']['content']}")

top_id = results[0]["payload"]["chunk_id"]
assert top_id in ("c1", "c5"), f"Expected c1 or c5 on top, got {top_id}"
print("  Top result is correct  ✓")

# ── 5. Filter search ──────────────────────────────────────────────────────────
print("\n[5] Filtered search: only source=ai.pdf …")
results_filtered = search(
    client, q.dense[0], q.sparse[0],
    collection=COLLECTION,
    limit=5,
    filters={"source": "ai.pdf"},
)

print(f"  Results ({len(results_filtered)}):")
for r in results_filtered:
    print(f"    [{r['score']:.4f}]  ({r['payload']['source']})  {r['payload']['content']}")

assert all(r["payload"]["source"] == "ai.pdf" for r in results_filtered), \
    "Filter failed — non-ai.pdf result returned"
print("  All results from ai.pdf  ✓")

# ── 6. Cleanup ────────────────────────────────────────────────────────────────
print(f"\n[6] Deleting test collection '{COLLECTION}' …")
try:
    client.delete_collection(COLLECTION)
    print("  Cleanup done  ✓")
except Exception:
    print("  In-memory client — no cleanup needed  ✓")

print("\n" + "=" * 60)
print("  All checks passed.")
print("=" * 60)
