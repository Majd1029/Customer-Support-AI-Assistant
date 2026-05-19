"""
tests/test_api.py — Integration tests for the FastAPI application.

Uses FastAPI's ``TestClient`` (wraps ``httpx`` with ``requests``-like API).
External services (Qdrant, Ollama, PostgreSQL, Groq) are NOT required:
  • Embedding / Qdrant routes return 503 (EMBEDDING_ENABLED=False) in CI.
  • Auth uses the in-process dict fallback when PostgreSQL is unavailable.
  • CSV endpoints are tested at the metadata level only (no PG import).

Run:
    pytest tests/test_api.py -v
"""
from __future__ import annotations

import json
import io
from pathlib import Path

import pytest

# Guard: if FastAPI / starlette are not installed, skip the whole module
pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient


# ── App fixture ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    Return a TestClient for the FastAPI app.

    Imported here (not at module level) so that tests without FastAPI
    installed still get collected and skipped via importorskip above.
    """
    from file_processor.api import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def admin_token(client):
    """Register an admin user and return their token."""
    resp = client.post("/auth/register", json={
        "username": "api_test_admin",
        "password": "adminpass123",
        "email":    "api_admin@test.com",
    })
    # If already registered (module-scope re-use), try login
    if resp.status_code == 400:
        resp = client.post("/auth/login", json={
            "username_or_email": "api_test_admin",
            "password":          "adminpass123",
        })
    # Promote to admin via auth_manager directly (test setup only)
    if resp.status_code in (200, 201):
        data = resp.json()
        token = data.get("token")
        if token:
            import file_processor.auth_manager as am
            info = am.verify_token(token)
            if info:
                uid = info["user_id"]
                if am._PG_OK:
                    try:
                        with am._get_conn() as conn:
                            with conn.cursor() as cur:
                                cur.execute(
                                    "UPDATE auth_users SET role='admin' WHERE user_id=%s", (uid,)
                                )
                            conn.commit()
                    except Exception:
                        pass
                else:
                    if uid in am._FALLBACK:
                        am._FALLBACK[uid]["role"] = "admin"
                # Re-issue token with admin role
                token = am._make_token(uid, info["username"], info.get("email", ""), "admin")
        return token
    return None


@pytest.fixture(scope="module")
def user_token(client):
    """Register a regular test user and return their token."""
    resp = client.post("/auth/register", json={
        "username": "api_test_user",
        "password": "userpass123",
        "email":    "api_user@test.com",
    })
    if resp.status_code == 400:
        resp = client.post("/auth/login", json={
            "username_or_email": "api_test_user",
            "password":          "userpass123",
        })
    if resp.status_code in (200, 201):
        return resp.json().get("token")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Health endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_has_status_key(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data

    def test_supported_formats_returns_list(self, client):
        resp = client.get("/supported-formats")
        assert resp.status_code == 200
        assert "formats" in resp.json()
        assert isinstance(resp.json()["formats"], list)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Auth endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestAuthEndpoints:
    def test_register_new_user(self, client):
        resp = client.post("/auth/register", json={
            "username": "newuser_test",
            "password": "pass1234",
        })
        assert resp.status_code in (200, 201)
        data = resp.json()
        assert "token" in data
        assert data["username"] == "newuser_test"

    def test_register_duplicate_username_returns_400(self, client):
        client.post("/auth/register", json={"username": "dup_user", "password": "pass1234"})
        resp = client.post("/auth/register", json={"username": "dup_user", "password": "pass5678"})
        assert resp.status_code == 400

    def test_login_with_correct_credentials(self, client):
        client.post("/auth/register", json={"username": "login_test", "password": "pass1234"})
        resp = client.post("/auth/login", json={
            "username_or_email": "login_test",
            "password":          "pass1234",
        })
        assert resp.status_code == 200
        assert "token" in resp.json()

    def test_login_wrong_password_returns_401(self, client):
        client.post("/auth/register", json={"username": "badpass_test", "password": "correct"})
        resp = client.post("/auth/login", json={
            "username_or_email": "badpass_test",
            "password":          "wrong",
        })
        assert resp.status_code == 401

    def test_auth_me_with_valid_token(self, client, user_token):
        if not user_token:
            pytest.skip("Could not obtain user token")
        resp = client.get("/auth/me", headers={"Authorization": f"Bearer {user_token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert "user_id" in data

    def test_auth_me_without_token_returns_401(self, client):
        resp = client.get("/auth/me")
        assert resp.status_code == 401

    def test_auth_me_with_invalid_token_returns_401(self, client):
        resp = client.get("/auth/me", headers={"Authorization": "Bearer invalid.token.here"})
        assert resp.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# 3. Admin management endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestAdminEndpoints:
    def test_list_users_requires_admin(self, client, user_token):
        if not user_token:
            pytest.skip("No user token")
        resp = client.get("/admin/users", headers={"Authorization": f"Bearer {user_token}"})
        assert resp.status_code == 403

    def test_list_users_unauthenticated_returns_401(self, client):
        resp = client.get("/admin/users")
        assert resp.status_code == 401

    def test_list_users_as_admin(self, client, admin_token):
        if not admin_token:
            pytest.skip("No admin token")
        resp = client.get("/admin/users", headers={"Authorization": f"Bearer {admin_token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert "users" in data
        assert isinstance(data["users"], list)

    def test_create_admin_requires_admin(self, client, user_token):
        if not user_token:
            pytest.skip("No user token")
        resp = client.post(
            "/admin/users",
            json={"username": "newadmin", "password": "pass123"},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert resp.status_code == 403

    def test_update_role_requires_admin(self, client, user_token):
        if not user_token:
            pytest.skip("No user token")
        resp = client.put(
            "/admin/users/some-id/role",
            json={"role": "admin"},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert resp.status_code == 403

    def test_delete_user_requires_admin(self, client, user_token):
        if not user_token:
            pytest.skip("No user token")
        resp = client.delete(
            "/admin/users/some-id",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert resp.status_code == 403


# ─────────────────────────────────────────────────────────────────────────────
# 4. Conversation endpoints (require auth)
# ─────────────────────────────────────────────────────────────────────────────

class TestConversationEndpoints:
    def test_list_conversations_requires_auth(self, client):
        resp = client.get("/conversations")
        assert resp.status_code == 401

    def test_list_conversations_authenticated(self, client, user_token):
        if not user_token:
            pytest.skip("No user token")
        resp = client.get(
            "/conversations",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "conversations" in data

    def test_save_and_load_messages(self, client, user_token):
        if not user_token:
            pytest.skip("No user token")
        headers  = {"Authorization": f"Bearer {user_token}"}
        sid      = "test-session-abc123"
        messages = [
            {"role": "user",      "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        # Save
        save_resp = client.post(
            f"/conversations/{sid}/messages",
            json={"messages": messages, "label": "Test conversation"},
            headers=headers,
        )
        assert save_resp.status_code in (200, 201)

        # Load
        load_resp = client.get(
            f"/conversations/{sid}/messages",
            headers=headers,
        )
        assert load_resp.status_code == 200
        loaded = load_resp.json().get("messages", [])
        assert len(loaded) == 2

    def test_delete_conversation(self, client, user_token):
        if not user_token:
            pytest.skip("No user token")
        headers = {"Authorization": f"Bearer {user_token}"}
        sid     = "test-session-delete"
        # Create
        client.post(
            f"/conversations/{sid}/messages",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=headers,
        )
        # Delete
        del_resp = client.delete(
            f"/conversations/{sid}",
            headers=headers,
        )
        assert del_resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# 5. Upload endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestUploadEndpoint:
    def test_upload_txt_file(self, client, user_token):
        """Upload a small text file and verify the response shape."""
        if not user_token:
            pytest.skip("No user token")
        content = b"This is a test document with multiple sentences. It should be chunked properly."
        resp = client.post(
            "/upload",
            files={"file": ("test_doc.txt", io.BytesIO(content), "text/plain")},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        # 200 on success, 503 if embedder is offline in CI
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert "source_file" in data or "chunks" in data or "type" in data

    def test_upload_without_auth_uses_anonymous_owner(self, client):
        """Uploads without JWT should succeed (open dev mode) and use empty owner_id."""
        content = b"Anonymous upload test content."
        resp = client.post(
            "/upload",
            files={"file": ("anon.txt", io.BytesIO(content), "text/plain")},
        )
        assert resp.status_code in (200, 503)

    def test_upload_rejects_unsupported_extension(self, client, user_token):
        """Uploading an unsupported file type should return 415 or 400."""
        content = b"<exe content>"
        resp = client.post(
            "/upload",
            files={"file": ("malware.exe", io.BytesIO(content), "application/octet-stream")},
            headers={"Authorization": f"Bearer {user_token}"} if user_token else {},
        )
        assert resp.status_code in (400, 415, 422)


# ─────────────────────────────────────────────────────────────────────────────
# 6. CSV query endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestCsvEndpoints:
    def test_csv_tables_endpoint_exists(self, client):
        resp = client.get("/csv-tables")
        # 200 if CSV_ENABLED, 503 if not
        assert resp.status_code in (200, 503)

    def test_query_endpoint_without_table_returns_400_or_503(self, client):
        resp = client.post("/query", json={"question": "What is the total?", "table": ""})
        assert resp.status_code in (400, 422, 503)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Index / documents endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestIndexEndpoints:
    def test_list_indexed_documents_returns_200_or_503(self, client, user_token):
        if not user_token:
            pytest.skip("No user token")
        resp = client.get(
            "/index/documents",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert resp.status_code in (200, 503)

    def test_search_endpoint_returns_200_or_503(self, client):
        resp = client.post("/search", json={"query": "test query"})
        assert resp.status_code in (200, 503)
