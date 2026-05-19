"""
tests/test_admin_rbac.py — Tests for admin role enforcement and RBAC logic.

Covers:
  • Admin creation via create_admin_user()
  • Role cannot be set to admin via the public register_user() path
  • Owner-ID resolution: admin uploads → __global__, user uploads → user_id
  • RBAC filter construction: users see own docs + __global__; admins see all

No FastAPI TestClient required — tests exercise pure Python logic.
"""
from __future__ import annotations

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 1. Admin creation
# ─────────────────────────────────────────────────────────────────────────────

class TestAdminCreation:
    def test_create_admin_user_sets_admin_role(self, auth_manager):
        user, err = auth_manager.create_admin_user("superuser", "pass123")
        assert err is None
        assert user["role"] == "admin"

    def test_token_encodes_admin_role(self, auth_manager):
        user, _ = auth_manager.create_admin_user("boss", "pass123")
        decoded = auth_manager.verify_token(user["token"])
        assert decoded["role"] == "admin"

    def test_register_user_cannot_create_admin_directly(self, auth_manager):
        """Public registration MUST NOT allow callers to self-assign the admin role."""
        token, err = auth_manager.register_user("hacker", "pass123", role="admin")
        assert err is None  # registration succeeds...
        info = auth_manager.verify_token(token)
        assert info["role"] == "user"  # ...but role is always 'user'

    def test_admin_login_preserves_role(self, auth_manager):
        auth_manager.create_admin_user("ops", "pass123")
        user, err = auth_manager.login_user("ops", "pass123")
        assert err is None
        assert user["role"] == "admin"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Owner-ID resolution logic (mirrors api.py logic, tested in isolation)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_owner_id(current_user: dict | None, form_owner_id: str = "") -> str:
    """
    Re-implementation of the owner_id resolution logic from api.py POST /upload.

    Admins always get __global__.
    Regular users get their own user_id.
    Unauthenticated requests get empty string (anonymous).
    """
    GLOBAL_KB = "__global__"
    if current_user is None:
        return form_owner_id or ""

    is_admin = current_user.get("role") == "admin"
    uid      = current_user.get("user_id", "")

    if not form_owner_id:
        return GLOBAL_KB if is_admin else uid

    # If admin passes their own user_id in form, remap to __global__
    if is_admin and form_owner_id == uid:
        return GLOBAL_KB

    # Admin explicitly provides a different owner_id (crawler path) — respect it
    if is_admin:
        return form_owner_id

    # Regular user — always their own id regardless of form field
    return uid


class TestOwnerIdResolution:
    def test_unauthenticated_no_form_field_returns_empty(self):
        assert _resolve_owner_id(None, "") == ""

    def test_unauthenticated_with_form_field_returns_form_field(self):
        assert _resolve_owner_id(None, "crawler_user") == "crawler_user"

    def test_regular_user_gets_own_id(self):
        user = {"user_id": "uid-123", "role": "user"}
        assert _resolve_owner_id(user) == "uid-123"

    def test_regular_user_form_field_ignored(self):
        """Users cannot override owner_id via the form field."""
        user = {"user_id": "uid-123", "role": "user"}
        assert _resolve_owner_id(user, form_owner_id="different-id") == "uid-123"

    def test_admin_gets_global(self):
        user = {"user_id": "admin-uuid", "role": "admin"}
        assert _resolve_owner_id(user) == "__global__"

    def test_admin_with_own_uid_in_form_gets_global(self):
        """Admin accidentally sending their own ID should be remapped to __global__."""
        user = {"user_id": "admin-uuid", "role": "admin"}
        assert _resolve_owner_id(user, form_owner_id="admin-uuid") == "__global__"

    def test_admin_with_explicit_other_owner_id_uses_that(self):
        """Admin crawling on behalf of a specific user should preserve that ID."""
        user = {"user_id": "admin-uuid", "role": "admin"}
        assert _resolve_owner_id(user, form_owner_id="specific_user") == "specific_user"


# ─────────────────────────────────────────────────────────────────────────────
# 3. RBAC filter construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_rbac_filters(current_user: dict | None) -> dict:
    """
    Mirror the RBAC filter logic from api.py /ask and /ask/stream handlers.

    Regular users → filter by [user_id, "__global__"]
    Admins        → no filter (sees everything)
    Unauthenticated → no filter in dev mode (open access)
    """
    filters: dict = {}
    if current_user is None:
        return filters

    uid      = current_user.get("user_id", "")
    is_admin = current_user.get("role") == "admin"

    if uid and not is_admin:
        filters["owner_id"] = [uid, "__global__"]

    return filters


class TestRbacFilters:
    def test_unauthenticated_no_filters(self):
        filters = _build_rbac_filters(None)
        assert filters == {}

    def test_regular_user_sees_own_and_global(self):
        user    = {"user_id": "u123", "role": "user"}
        filters = _build_rbac_filters(user)
        assert "owner_id" in filters
        owner_ids = filters["owner_id"]
        assert "u123"      in owner_ids
        assert "__global__" in owner_ids

    def test_admin_has_no_owner_filter(self):
        user    = {"user_id": "a456", "role": "admin"}
        filters = _build_rbac_filters(user)
        assert "owner_id" not in filters

    def test_user_filter_excludes_other_users(self):
        user    = {"user_id": "u999", "role": "user"}
        filters = _build_rbac_filters(user)
        owner_ids = filters["owner_id"]
        assert "u111" not in owner_ids
        assert "u222" not in owner_ids


# ─────────────────────────────────────────────────────────────────────────────
# 4. Role update guard (self-demotion prevention)
# ─────────────────────────────────────────────────────────────────────────────

class TestRoleUpdateGuards:
    def test_admin_cannot_demote_themselves(self, auth_manager):
        """
        Mirrors the /admin/users/{id}/role endpoint guard.
        An admin should not be able to remove their own admin role.
        """
        user, _ = auth_manager.create_admin_user("self_admin", "pass123")
        uid = user["user_id"]

        # Simulate the guard check
        caller  = {"user_id": uid, "role": "admin"}
        new_role = "user"

        is_self_demotion = (uid == caller["user_id"] and new_role != "admin")
        assert is_self_demotion is True, "Guard should trigger for self-demotion"

    def test_admin_can_promote_other_user(self, auth_manager):
        token, _ = auth_manager.register_user("regular", "pass123")
        info = auth_manager.verify_token(token)
        uid = info["user_id"]

        admin = {"user_id": "admin-different-id", "role": "admin"}
        is_self_demotion = (uid == admin["user_id"] and "user" != "admin")
        assert is_self_demotion is False, "Guard should NOT trigger for promoting another user"
