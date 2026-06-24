"""
clean_csv_registry.py — Remove stale CSV table registrations.

Early uploads registered each CSV under a per-upload table name (e.g.
"b1d1d56a_test123") whose source file lived in the temporary uploads/ dir and
was deleted right after import. Those registry rows can never be queried
(query_table fails with "source file not found"), and they confuse the CSV
auto-detect / table-matching logic.

This script deletes every _csv_registry row whose file_path no longer exists
on disk, keeping only tables whose source CSV is still available (i.e. the ones
now persisted under csv_store/).

Usage:
    python scripts/clean_csv_registry.py            # delete stale rows
    python scripts/clean_csv_registry.py --dry-run  # just report, change nothing
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when run as `python scripts/clean_csv_registry.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from csv_query_engine.csv_executer import get_engine, _ensure_registry  # noqa: E402


def main(dry_run: bool = False) -> None:
    _ensure_registry()
    engine = get_engine()

    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT table_name, file_path FROM _csv_registry ORDER BY table_name")
        ).fetchall()

        if not rows:
            print("Registry is empty — nothing to clean.")
            return

        stale, kept = [], []
        for table_name, file_path in rows:
            exists = bool(file_path) and Path(file_path).exists()
            (kept if exists else stale).append((table_name, file_path))

        print(f"Registered tables: {len(rows)}  |  valid: {len(kept)}  |  stale: {len(stale)}\n")

        if kept:
            print("KEEPING (source file present):")
            for t, p in kept:
                print(f"  ✓ {t}")
            print()

        if not stale:
            print("No stale rows found.")
            return

        print("STALE (source file missing — will be removed):")
        for t, p in stale:
            print(f"  ✗ {t}   [{p}]")

        if dry_run:
            print("\n--dry-run: no changes made.")
            return

        conn.execute(
            text("DELETE FROM _csv_registry WHERE table_name = ANY(:names)"),
            {"names": [t for t, _ in stale]},
        )
        print(f"\nRemoved {len(stale)} stale registration(s).")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
