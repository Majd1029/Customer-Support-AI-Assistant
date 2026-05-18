"""
tests/conftest.py — shared fixtures and path setup for the pytest suite.

Recommended (fastest) way to run:
    pip install -e .          # editable install — adds project root to sys.path via .pth
    pytest

Fallback without the editable install (used in some CI / sandbox environments):
    pytest                    # conftest.py adds project root + file_processor/ to sys.path

The editable install makes `file_processor` and `file_preparation`
importable without any sys.path manipulation inside tests.

For modules that still rely on file_processor/ being in sys.path
(within-package sibling imports used by api.py), we add it here so
the test runner matches the runtime environment of the server.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Project root is two levels up from this file (tests/ → root)
_ROOT = Path(__file__).resolve().parent.parent

# Add the project root so `file_preparation` and `file_processor` are importable
# as top-level packages when pip install -e . has NOT been run.
# When the editable install IS active the root is already on sys.path via the
# .pth file in site-packages — the guard below keeps it idempotent.
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Ensure file_processor/ is on sys.path so within-package sibling imports
# (e.g. `from models import ExtractionResult`) work during tests.
_FP = _ROOT / "file_processor"
if str(_FP) not in sys.path:
    sys.path.insert(0, str(_FP))
