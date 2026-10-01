"""
tests/test_payload_indexes.py — payload indexes required for filtered search.

Qdrant Cloud rejects filters on unindexed payload fields ("Index required but
not found for owner_id"), so every field /ask and /search filter on must be
indexed — including on collections created before the field was added.

Uses a fake client; no Qdrant connection required.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from file_preparation.indexing import store


class FakeClient:
    def __init__(self, existing: list[str] | None = None):
        self.existing = list(existing or [])
        self.created: list[str] = []
        self.indexes: list[tuple[str, str, object]] = []

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self.existing])

    def create_collection(self, collection_name, **_kw):
        self.created.append(collection_name)
        self.existing.append(collection_name)

    def delete_collection(self, name):
        self.existing.remove(name)

    def create_payload_index(self, collection_name, field_name, field_schema):
        self.indexes.append((collection_name, field_name, field_schema))


@pytest.fixture(autouse=True)
def _reset_cache():
    store._INDEXES_ENSURED.clear()
    yield
    store._INDEXES_ENSURED.clear()


def _fields(client, collection="documents"):
    return {f for c, f, _ in client.indexes if c == collection}


def test_rbac_fields_are_indexed():
    for field in ("owner_id", "owner_email", "drive_file_id", "allowed_users", "source", "doc_id"):
        assert field in store._INDEXED_PAYLOAD_FIELDS


def test_new_collection_gets_all_indexes():
    client = FakeClient()
    store.ensure_collection(client, "documents")

    assert client.created == ["documents"]
    assert _fields(client) == set(store._INDEXED_PAYLOAD_FIELDS) | set(store._INDEXED_INTEGER_FIELDS)


def test_existing_collection_is_backfilled_once():
    client = FakeClient(existing=["documents"])

    store.ensure_collection(client, "documents")
    store.ensure_collection(client, "documents")   # second call: cached, no extra requests

    assert client.created == []
    assert "owner_id" in _fields(client)
    expected = len(store._INDEXED_PAYLOAD_FIELDS) + len(store._INDEXED_INTEGER_FIELDS)
    assert len(client.indexes) == expected


def test_index_errors_do_not_abort(monkeypatch):
    client = FakeClient(existing=["documents"])
    calls = []

    def flaky(collection_name, field_name, field_schema):
        calls.append(field_name)
        if field_name == "language":
            raise RuntimeError("already exists with another type")

    client.create_payload_index = flaky
    store.ensure_payload_indexes(client, "documents")

    assert "owner_id" in calls and "chunk_index" in calls
