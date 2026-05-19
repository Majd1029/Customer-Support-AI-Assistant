"""
tests/conftest.py — shared fixtures and path setup for the pytest suite.

Run the full suite:
    pip install -e .
    pytest

Run a specific file:
    pytest tests/test_auth.py -v

Run with coverage:
    pytest --cov=file_processor --cov=file_preparation --cov=csv_query_engine
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest

# ── Ensure project root is importable without editable install ────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Keep file_processor/ on path for within-package sibling imports used by api.py
_FP = _ROOT / "file_processor"
if str(_FP) not in sys.path:
    sys.path.insert(0, str(_FP))


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_csv_content() -> str:
    """A minimal CSV string with numeric and text columns."""
    return (
        "id,name,category,revenue,active\n"
        "1,Alpha,sales,1000.50,true\n"
        "2,Beta,marketing,2500.00,false\n"
        "3,Gamma,sales,750.25,true\n"
        "4,Delta,engineering,3200.00,true\n"
        "5,Epsilon,marketing,980.75,false\n"
    )


@pytest.fixture
def sample_csv_file(sample_csv_content: str, tmp_path: Path) -> Path:
    """Write sample_csv_content to a temp file and return the path."""
    path = tmp_path / "sample.csv"
    path.write_text(sample_csv_content, encoding="utf-8")
    return path


@pytest.fixture
def large_csv_file(tmp_path: Path) -> Path:
    """Write a 25 000-row CSV to tmp_path for chunking tests."""
    rows = ["id,value,category"]
    for i in range(25_000):
        rows.append(f"{i},{i * 1.5},cat{i % 10}")
    path = tmp_path / "large.csv"
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


@pytest.fixture
def sample_dataframe(sample_csv_content: str) -> pd.DataFrame:
    """Return the sample CSV as a pandas DataFrame."""
    return pd.read_csv(io.StringIO(sample_csv_content))


@pytest.fixture
def auth_manager():
    """Import and return auth_manager module (uses in-process fallback)."""
    # Patch PG_OK to False so tests never touch a real database
    import file_processor.auth_manager as am
    original_pg_ok = am._PG_OK
    am._PG_OK = False
    am._FALLBACK.clear()
    yield am
    am._PG_OK = original_pg_ok
    am._FALLBACK.clear()


@pytest.fixture
def registered_user(auth_manager):
    """Return (token, user_info) for a freshly registered test user."""
    token, err = auth_manager.register_user("testuser", "testpass123", "test@example.com")
    assert err is None, f"Registration failed: {err}"
    info = auth_manager.verify_token(token)
    return token, info


@pytest.fixture
def registered_admin(auth_manager):
    """Return (token, user_info) for a freshly registered admin user."""
    user, err = auth_manager.create_admin_user("adminuser", "adminpass123", "admin@example.com")
    assert err is None, f"Admin creation failed: {err}"
    return user["token"], user
