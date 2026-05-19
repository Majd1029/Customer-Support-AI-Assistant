"""
tests/test_auth.py — Unit tests for file_processor/auth_manager.py.

All tests use the in-process dict fallback so no PostgreSQL connection is
required.  The ``auth_manager`` fixture (defined in conftest.py) patches
``_PG_OK = False`` and clears ``_FALLBACK`` before/after each test.
"""
from __future__ import annotations

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 1. User registration
# ─────────────────────────────────────────────────────────────────────────────

class TestRegisterUser:
    def test_successful_registration_returns_token(self, auth_manager):
        token, err = auth_manager.register_user("alice", "password123")
        assert err is None
        assert token is not None
        assert len(token) > 10

    def test_registered_user_has_default_role(self, auth_manager):
        token, _ = auth_manager.register_user("bob", "password123")
        info = auth_manager.verify_token(token)
        assert info["role"] == "user"

    def test_duplicate_username_rejected(self, auth_manager):
        auth_manager.register_user("charlie", "pass1")
        token2, err2 = auth_manager.register_user("charlie", "pass2")
        assert token2 is None
        assert "already taken" in (err2 or "").lower()

    def test_too_short_username_rejected(self, auth_manager):
        token, err = auth_manager.register_user("x", "password123")
        assert token is None
        assert err is not None

    def test_too_short_password_rejected(self, auth_manager):
        token, err = auth_manager.register_user("validuser", "abc")
        assert token is None
        assert err is not None

    def test_email_stored_in_token(self, auth_manager):
        token, _ = auth_manager.register_user("dave", "pass1234", "dave@example.com")
        info = auth_manager.verify_token(token)
        assert info["email"] == "dave@example.com"

    def test_role_forced_to_user_even_if_invalid_value_passed(self, auth_manager):
        """register_user should silently normalise invalid role to 'user'."""
        token, _ = auth_manager.register_user("eve", "pass1234", role="superuser")
        info = auth_manager.verify_token(token)
        assert info["role"] == "user"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Login
# ─────────────────────────────────────────────────────────────────────────────

class TestLoginUser:
    def test_correct_credentials_return_user_dict(self, auth_manager):
        auth_manager.register_user("frank", "mypassword")
        user, err = auth_manager.login_user("frank", "mypassword")
        assert err is None
        assert user is not None
        assert user["username"] == "frank"
        assert "token" in user

    def test_wrong_password_rejected(self, auth_manager):
        auth_manager.register_user("grace", "correct")
        user, err = auth_manager.login_user("grace", "wrong")
        assert user is None
        assert "incorrect" in (err or "").lower()

    def test_unknown_username_rejected(self, auth_manager):
        user, err = auth_manager.login_user("nobody", "pass")
        assert user is None
        assert "not found" in (err or "").lower()

    def test_case_insensitive_username(self, auth_manager):
        auth_manager.register_user("Henry", "pass1234")
        user, err = auth_manager.login_user("henry", "pass1234")
        assert err is None
        assert user is not None

    def test_login_with_email(self, auth_manager):
        auth_manager.register_user("iris", "pass1234", "iris@example.com")
        user, err = auth_manager.login_user("iris@example.com", "pass1234")
        assert err is None
        assert user["username"] == "iris"

    def test_login_preserves_role(self, auth_manager):
        user, _ = auth_manager.create_admin_user("jadmin", "adminpass")
        logged, _ = auth_manager.login_user("jadmin", "adminpass")
        assert logged["role"] == "admin"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Token verification
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyToken:
    def test_valid_token_returns_user_info(self, auth_manager):
        token, _ = auth_manager.register_user("kyle", "pass1234")
        info = auth_manager.verify_token(token)
        assert info is not None
        assert info["username"] == "kyle"
        assert info["role"] == "user"

    def test_invalid_token_returns_none(self, auth_manager):
        result = auth_manager.verify_token("not.a.real.token")
        assert result is None

    def test_empty_string_returns_none(self, auth_manager):
        assert auth_manager.verify_token("") is None

    def test_token_contains_role(self, auth_manager):
        user, _ = auth_manager.create_admin_user("liz", "adminpass")
        info = auth_manager.verify_token(user["token"])
        assert info["role"] == "admin"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Admin user creation
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateAdminUser:
    def test_creates_user_with_admin_role(self, auth_manager):
        user, err = auth_manager.create_admin_user("mgr", "adminpass")
        assert err is None
        assert user["role"] == "admin"

    def test_admin_token_encodes_role(self, auth_manager):
        user, _ = auth_manager.create_admin_user("boss", "adminpass")
        info = auth_manager.verify_token(user["token"])
        assert info["role"] == "admin"

    def test_duplicate_admin_username_rejected(self, auth_manager):
        auth_manager.create_admin_user("ceo", "adminpass1")
        user2, err2 = auth_manager.create_admin_user("ceo", "adminpass2")
        assert user2 is None
        assert err2 is not None

    def test_returned_user_dict_has_all_fields(self, auth_manager):
        user, _ = auth_manager.create_admin_user("cto", "adminpass", "cto@example.com")
        assert all(k in user for k in ("user_id", "username", "email", "role", "token"))


# ─────────────────────────────────────────────────────────────────────────────
# 5. get_user_by_id
# ─────────────────────────────────────────────────────────────────────────────

class TestGetUserById:
    def test_returns_user_for_valid_id(self, auth_manager):
        token, _ = auth_manager.register_user("nat", "pass1234")
        info = auth_manager.verify_token(token)
        user = auth_manager.get_user_by_id(info["user_id"])
        assert user is not None
        assert user["username"] == "nat"

    def test_returns_none_for_unknown_id(self, auth_manager):
        result = auth_manager.get_user_by_id("00000000-0000-0000-0000-000000000000")
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Password hashing (implementation detail sanity check)
# ─────────────────────────────────────────────────────────────────────────────

class TestPasswordHashing:
    def test_different_salts_produce_different_hashes(self, auth_manager):
        h1 = auth_manager._hash_password("same_password")
        h2 = auth_manager._hash_password("same_password")
        assert h1 != h2, "Each hash must use a unique random salt"

    def test_correct_password_verifies(self, auth_manager):
        stored = auth_manager._hash_password("mysecret")
        assert auth_manager._check_password("mysecret", stored) is True

    def test_wrong_password_fails(self, auth_manager):
        stored = auth_manager._hash_password("mysecret")
        assert auth_manager._check_password("wrong", stored) is False
