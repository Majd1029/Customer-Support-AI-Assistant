"""
tests/test_remote_embedder.py — unit tests for the Hugging Face Inference API
embedding backend (file_preparation/embedding/remote_embedder.py) and the
EMBEDDING_BACKEND selector (file_preparation/embedding/backend.py).

No network access: HTTP calls go through an httpx.MockTransport.
"""
from __future__ import annotations

import importlib
import json
import math

import httpx
import pytest

from file_preparation.embedding import remote_embedder


def _vec(seed: float, dim: int = 1024) -> list[float]:
    """Deterministic, un-normalised test vector."""
    return [seed + i * 0.001 for i in range(dim)]


@pytest.fixture
def mock_api(monkeypatch):
    """Route remote_embedder's HTTP client to a handler the test controls."""
    calls: list[dict] = []
    state = {"handler": None}

    def dispatch(request: httpx.Request) -> httpx.Response:
        calls.append({"headers": dict(request.headers), "body": json.loads(request.content)})
        return state["handler"](request, len(calls))

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(remote_embedder, "_client", httpx.Client(transport=httpx.MockTransport(dispatch)))
    monkeypatch.setattr(remote_embedder.time, "sleep", lambda _s: None)

    def set_handler(fn):
        state["handler"] = fn

    return set_handler, calls


def _ok(request: httpx.Request, _n: int) -> httpx.Response:
    inputs = json.loads(request.content)["inputs"]
    return httpx.Response(200, json=[_vec(float(i + 1)) for i in range(len(inputs))])


class TestRemoteEncode:
    def test_returns_normalised_dense_and_bm25_sparse(self, mock_api):
        set_handler, calls = mock_api
        set_handler(_ok)

        emb = remote_embedder.encode(["refund policy", "shipping times"])

        assert len(emb.dense) == 2 and len(emb.sparse) == 2
        for v in emb.dense:
            assert len(v) == 1024
            assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-6)
        assert all(isinstance(s, dict) and s for s in emb.sparse)

        body = calls[0]["body"]
        assert body["inputs"] == ["refund policy", "shipping times"]
        assert body["normalize"] is True and body["truncate"] is True
        assert calls[0]["headers"]["authorization"] == "Bearer hf_test"

    def test_batches_requests(self, mock_api):
        set_handler, calls = mock_api
        set_handler(_ok)

        emb = remote_embedder.encode([f"text {i}" for i in range(5)], batch_size=2)

        assert len(emb.dense) == 5
        assert [len(c["body"]["inputs"]) for c in calls] == [2, 2, 1]

    def test_encode_query_single_vector(self, mock_api):
        set_handler, _ = mock_api
        set_handler(_ok)

        emb = remote_embedder.encode_query("where is my order?")

        assert len(emb.dense) == 1 and len(emb.dense[0]) == remote_embedder.dense_dim() == 1024

    def test_empty_input_makes_no_request(self, mock_api):
        _, calls = mock_api
        emb = remote_embedder.encode([])
        assert emb.dense == [] and emb.sparse == [] and calls == []

    def test_retries_on_cold_start_then_succeeds(self, mock_api):
        set_handler, calls = mock_api
        set_handler(lambda req, n: httpx.Response(503, text="loading") if n < 3 else _ok(req, n))

        emb = remote_embedder.encode(["hello"])

        assert len(emb.dense) == 1 and len(calls) == 3

    def test_client_error_raises_without_retry(self, mock_api):
        set_handler, calls = mock_api
        set_handler(lambda req, n: httpx.Response(401, text="Invalid credentials"))

        with pytest.raises(RuntimeError, match="HTTP 401"):
            remote_embedder.encode(["hello"])
        assert len(calls) == 1

    def test_vector_count_mismatch_raises(self, mock_api):
        set_handler, _ = mock_api
        set_handler(lambda req, n: httpx.Response(200, json=[_vec(1.0)]))

        with pytest.raises(RuntimeError, match="1 vectors for 2 inputs"):
            remote_embedder.encode(["a", "b"])

    def test_missing_token_raises(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
        with pytest.raises(RuntimeError, match="HF_TOKEN"):
            remote_embedder.encode(["hello"])


class TestBackendSelection:
    def _reload(self):
        from file_preparation.embedding import backend
        return importlib.reload(backend)

    def test_default_is_local(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
        backend = self._reload()
        from file_preparation.embedding import embedder
        assert backend.BACKEND == "local"
        assert backend.encode is embedder.encode

    def test_hf_api_selects_remote(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_BACKEND", "hf_api")
        backend = self._reload()
        assert backend.encode is remote_embedder.encode
        assert backend.encode_query is remote_embedder.encode_query
        monkeypatch.delenv("EMBEDDING_BACKEND")
        self._reload()
