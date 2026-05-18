"""
Real embedder test — loads the actual BGE-M3 model.
No Qdrant needed. First run downloads ~2 GB from HuggingFace.

Run from anywhere:
    python file_preparation\embedding\test_embedder_real.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from embedder import encode, encode_query

print("=" * 60)
print("  BGE-M3 real embedder test")
print("  (model will be downloaded on first run ~2 GB)")
print("=" * 60)

# ── 1. Encode a batch of passages ─────────────────────────────────────────────
print("\n[1] Encoding 3 passages …")
texts = [
    "Paris is the capital of France.",
    "Machine learning is a subset of artificial intelligence.",
    "The Eiffel Tower is located in Paris.",
]
emb = encode(texts)

assert len(emb.dense)  == 3,    f"Expected 3 dense vectors, got {len(emb.dense)}"
assert len(emb.dense[0]) == 1024, f"Expected 1024 dims, got {len(emb.dense[0])}"
assert len(emb.sparse) == 3,    f"Expected 3 sparse vectors, got {len(emb.sparse)}"
assert all(isinstance(k, int) for k in emb.sparse[0]), "Sparse keys should be ints"

print(f"  Dense  : {len(emb.dense)} vectors × {len(emb.dense[0])} dims  ✓")
print(f"  Sparse : {len(emb.sparse)} vectors, "
      f"~{sum(len(s) for s in emb.sparse) // len(emb.sparse)} non-zero entries/vec  ✓")

# ── 2. Encode a query ─────────────────────────────────────────────────────────
print("\n[2] Encoding query …")
q = encode_query("What is the capital of France?")

assert len(q.dense)  == 1
assert len(q.dense[0]) == 1024
assert len(q.sparse) == 1

print(f"  Dense  : {len(q.dense[0])} dims  ✓")
print(f"  Sparse : {len(q.sparse[0])} non-zero entries  ✓")

# ── 3. Cosine similarity — related texts should score higher ──────────────────
print("\n[3] Cosine similarity check …")
import math

def cosine(a, b):
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)

q_vec   = q.dense[0]
scores  = [(cosine(q_vec, emb.dense[i]), texts[i]) for i in range(len(texts))]
scores.sort(reverse=True)

print("  Query: 'What is the capital of France?'")
for score, text in scores:
    print(f"    {score:.4f}  {text}")

# Top result should be the Paris/France sentence
assert scores[0][1] == "Paris is the capital of France.", \
    f"Expected Paris sentence on top, got: {scores[0][1]}"
print("  Top result is correct  ✓")

# ── 4. Empty input ────────────────────────────────────────────────────────────
print("\n[4] Empty input …")
empty = encode([])
assert empty.dense  == []
assert empty.sparse == []
print("  encode([]) → empty Embeddings  ✓")

print("\n" + "=" * 60)
print("  All checks passed.")
print("=" * 60)
