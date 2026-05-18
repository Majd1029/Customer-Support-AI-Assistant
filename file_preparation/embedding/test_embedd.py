"""
Smoke test for the embedding module.

Uses a mock embedder so the BGE-M3 model is never loaded.
Tests: encode shape, Qdrant upsert, hybrid search, indexer retrieve.

Requirements: qdrant-client, a running Qdrant instance on localhost:6333
Start Qdrant with:  docker run -p 6333:6333 qdrant/qdrant
"""

import random
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# ── Ensure embedding/ siblings are importable ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Mock FlagEmbedding before any import tries to load it ─────────────────────
DENSE_DIM = 1024

def _fake_dense(n=1):
    return [[random.uniform(-0.1, 0.1) for _ in range(DENSE_DIM)] for _ in range(n)]

def _fake_sparse(n=1):
    return [{random.randint(0, 30000): random.uniform(0.0, 1.0) for _ in range(50)} for _ in range(n)]

class _FakeBGEM3:
    def encode(self, texts, **kwargs):
        n = len(texts)
        return {
            "dense_vecs":     [MagicMock(tolist=lambda v=v: v) for v in _fake_dense(n)],
            "lexical_weights": _fake_sparse(n),
        }

_flag_mod = types.ModuleType("FlagEmbedding")
_flag_mod.BGEM3FlagModel = lambda *a, **kw: _FakeBGEM3()
sys.modules["FlagEmbedding"] = _flag_mod

# ── Now safe to import ────────────────────────────────────────────────────────
from embedder import encode, encode_query, Embeddings  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# 1. Embedder unit tests (no Qdrant needed)
# ─────────────────────────────────────────────────────────────────────────────
class TestEmbedder(unittest.TestCase):

    def test_encode_shape(self):
        texts = ["Hello world", "Machine learning", "Paris is in France"]
        emb = encode(texts)
        self.assertIsInstance(emb, Embeddings)
        self.assertEqual(len(emb.dense), 3)
        self.assertEqual(len(emb.dense[0]), DENSE_DIM)
        self.assertEqual(len(emb.sparse), 3)
        self.assertTrue(all(isinstance(k, int) for k in emb.sparse[0]))
        print(f"\n  encode()       OK — {len(emb.dense)} × {len(emb.dense[0])}d, "
              f"sparse keys: {len(emb.sparse[0])}")

    def test_encode_empty(self):
        emb = encode([])
        self.assertEqual(emb.dense, [])
        self.assertEqual(emb.sparse, [])
        print("  encode([])     OK — returns empty Embeddings")

    def test_encode_query(self):
        q = encode_query("What is the capital of France?")
        self.assertEqual(len(q.dense), 1)
        self.assertEqual(len(q.dense[0]), DENSE_DIM)
        print(f"  encode_query() OK — dense dim={len(q.dense[0])}, "
              f"sparse keys={len(q.sparse[0])}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Store unit tests (mocked Qdrant — no server needed)
# ─────────────────────────────────────────────────────────────────────────────
class TestStoreMocked(unittest.TestCase):

    def setUp(self):
        """Patch QdrantClient so no real server is needed."""
        import store
        self.patcher = patch("store.QdrantClient", autospec=True)
        self.MockClient = self.patcher.start()
        self.client = self.MockClient.return_value
        # get_collections returns an empty list
        self.client.get_collections.return_value = MagicMock(collections=[])

    def tearDown(self):
        self.patcher.stop()

    def test_ensure_collection_creates(self):
        from store import ensure_collection
        ensure_collection(self.client, "test_col")
        self.client.create_collection.assert_called_once()
        print("  ensure_collection() OK — create_collection called")

    def test_ensure_collection_skip_existing(self):
        from store import ensure_collection
        existing = MagicMock()
        existing.name = "test_col"
        self.client.get_collections.return_value = MagicMock(collections=[existing])
        ensure_collection(self.client, "test_col")
        self.client.create_collection.assert_not_called()
        print("  ensure_collection() OK — skips existing collection")

    def test_build_point(self):
        from store import build_point
        dense  = _fake_dense(1)[0]
        sparse = _fake_sparse(1)[0]
        pt = build_point("chunk_001", dense, sparse, {"content": "test"})
        self.assertIsNotNone(pt.id)
        self.assertIn("dense",  pt.vector)
        self.assertIn("sparse", pt.vector)
        print("  build_point()       OK — point has dense + sparse vectors")

    def test_upsert(self):
        from store import build_point, upsert
        points = [
            build_point(f"c_{i}", _fake_dense(1)[0], _fake_sparse(1)[0], {"content": f"text {i}"})
            for i in range(5)
        ]
        total = upsert(self.client, points, collection="test_col")
        self.assertEqual(total, 5)
        self.client.upsert.assert_called()
        print(f"  upsert()            OK — {total} points upserted")


# ─────────────────────────────────────────────────────────────────────────────
# 3. index_chunks unit test (mocked Qdrant)
# ─────────────────────────────────────────────────────────────────────────────
class TestIndexerMocked(unittest.TestCase):

    def setUp(self):
        import store
        self.patcher = patch("store.QdrantClient", autospec=True)
        self.MockClient = self.patcher.start()
        self.client = self.MockClient.return_value
        self.client.get_collections.return_value = MagicMock(collections=[])

    def tearDown(self):
        self.patcher.stop()

    def test_index_chunks(self):
        from indexer import index_chunks
        chunks = [
            {
                "chunk_id": f"doc_p1_t{i}",
                "type": "text",
                "content": f"This is sentence number {i} about something interesting.",
                "metadata": {"source": "test.pdf", "page_start": 1, "token_count": 10},
            }
            for i in range(4)
        ]
        stats = index_chunks(chunks, self.client, collection="test_col")
        self.assertEqual(stats["indexed"], 4)
        self.assertEqual(stats["skipped"], 0)
        print(f"  index_chunks()      OK — indexed={stats['indexed']}, skipped={stats['skipped']}")

    def test_index_chunks_skips_empty(self):
        from indexer import index_chunks
        chunks = [
            {"chunk_id": "c1", "type": "image", "content": "",     "metadata": {}},
            {"chunk_id": "c2", "type": "text",  "content": "Hello", "metadata": {}},
        ]
        stats = index_chunks(chunks, self.client, collection="test_col")
        self.assertEqual(stats["indexed"], 1)
        self.assertEqual(stats["skipped"], 1)
        print(f"  index_chunks()      OK — empty content skipped correctly")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Embedding module smoke tests (mock embedder + mock Qdrant)")
    print("=" * 60)
    unittest.main(verbosity=0)
