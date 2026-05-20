"""
reset_drive_cache.py — Wipe the Drive incremental state and delete all Qdrant
documents that were indexed from Drive crawls, so the next crawl processes
every file fresh and re-indexes with correct owner_id = "__global__".

Usage:
    python scripts/reset_drive_cache.py               # dry run (shows what would be deleted)
    python scripts/reset_drive_cache.py --confirm     # actually delete
    python scripts/reset_drive_cache.py --qdrant-only # only clear Qdrant docs
    python scripts/reset_drive_cache.py --pg-only     # only clear DB cache

After running with --confirm:
    1. Go to the Data Sources panel → Drive tab
    2. Make sure "Incremental" is toggled OFF (or the cache is now empty so it
       makes no difference) and re-run the crawl.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reset_pg(dry_run: bool) -> int:
    """Null-out modified_time and indexed_name for all drive_files rows.
    Returns the number of rows affected."""
    try:
        import psycopg2
        params = {
            "host":     os.getenv("PG_HOST",     "localhost"),
            "port":     int(os.getenv("PG_PORT", "5432")),
            "dbname":   os.getenv("PG_DB",       "csvstore"),
            "user":     os.getenv("PG_USER",     "csvuser"),
            "password": os.getenv("PG_PASSWORD", ""),
        }
        with psycopg2.connect(**params) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM drive_files WHERE modified_time IS NOT NULL")
                count = cur.fetchone()[0]
                if dry_run:
                    print(f"  [DRY RUN] Would clear modified_time / indexed_name on {count} row(s) in drive_files")
                else:
                    cur.execute("UPDATE drive_files SET modified_time = NULL, indexed_name = NULL")
                    conn.commit()
                    print(f"  [PG] Cleared incremental state on {count} row(s)")
                return count
    except Exception as e:
        print(f"  [PG] Error: {e}")
        return 0


def _list_drive_sources(dry_run: bool) -> list[str]:
    """Get indexed_name values from drive_files (= Qdrant source filenames)."""
    sources = []
    try:
        import psycopg2
        params = {
            "host":     os.getenv("PG_HOST",     "localhost"),
            "port":     int(os.getenv("PG_PORT", "5432")),
            "dbname":   os.getenv("PG_DB",       "csvstore"),
            "user":     os.getenv("PG_USER",     "csvuser"),
            "password": os.getenv("PG_PASSWORD", ""),
        }
        with psycopg2.connect(**params) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT indexed_name FROM drive_files WHERE indexed_name IS NOT NULL")
                sources = [row[0] for row in cur.fetchall()]
        if sources:
            print(f"  [PG] Found {len(sources)} indexed_name(s) in drive_files:")
            for s in sources:
                print(f"       • {s}")
        else:
            print("  [PG] No indexed_name entries found — Qdrant delete will use collection scroll instead")
    except Exception as e:
        print(f"  [PG] Could not fetch indexed names: {e}")
    return sources


def _delete_qdrant_by_owner(dry_run: bool) -> int:
    """Delete all Qdrant points whose payload has owner_id != '__global__'.
    Targets old crawl data indexed before the global-KB fix."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue, IsNullCondition

        url     = os.getenv("QDRANT_URL",     "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY", "") or None
        coll    = os.getenv("QDRANT_COLLECTION", "documents")

        client = QdrantClient(url=url, api_key=api_key)

        # Find all distinct owner_id values so we can report what will be deleted
        # Scroll a sample to identify non-global entries
        print(f"  [Qdrant] Scanning collection '{coll}' for non-global owner_id points …")

        deleted = 0
        offset  = None
        batch   = 200

        while True:
            result, next_offset = client.scroll(
                collection_name = coll,
                limit           = batch,
                offset          = offset,
                with_payload    = True,
                with_vectors    = False,
            )
            if not result:
                break

            ids_to_delete = []
            for pt in result:
                oid = (pt.payload or {}).get("owner_id")
                if oid is not None and oid != "__global__":
                    ids_to_delete.append(pt.id)

            if ids_to_delete:
                if dry_run:
                    print(f"  [DRY RUN] Would delete {len(ids_to_delete)} point(s) with non-global owner_id")
                else:
                    from qdrant_client.models import PointIdsList
                    client.delete(
                        collection_name = coll,
                        points_selector = PointIdsList(points=ids_to_delete),
                    )
                    print(f"  [Qdrant] Deleted {len(ids_to_delete)} point(s)")
                deleted += len(ids_to_delete)

            if next_offset is None:
                break
            offset = next_offset

        print(f"  [Qdrant] {'Would delete' if dry_run else 'Deleted'} {deleted} point(s) total")
        return deleted

    except ImportError:
        print("  [Qdrant] qdrant_client not installed — skipping Qdrant cleanup")
        return 0
    except Exception as e:
        print(f"  [Qdrant] Error: {e}")
        return 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Reset Drive crawl cache and clean old Qdrant docs")
    parser.add_argument("--confirm",    action="store_true", help="Actually delete (default: dry run)")
    parser.add_argument("--qdrant-only", action="store_true", help="Only clean Qdrant, skip PG")
    parser.add_argument("--pg-only",    action="store_true", help="Only reset PG cache, skip Qdrant")
    args = parser.parse_args()

    dry_run = not args.confirm

    if dry_run:
        print("=" * 60)
        print("DRY RUN — pass --confirm to actually delete")
        print("=" * 60)
    else:
        print("=" * 60)
        print("LIVE RUN — deleting data")
        print("=" * 60)

    if not args.qdrant_only:
        print("\n[1] PostgreSQL drive_files cache:")
        _reset_pg(dry_run)

    if not args.pg_only:
        print("\n[2] Qdrant — removing points with non-global owner_id:")
        _delete_qdrant_by_owner(dry_run)

    print("\nDone.")
    if dry_run:
        print("\nRe-run with --confirm to apply changes:")
        print("  python scripts/reset_drive_cache.py --confirm")
    else:
        print("\nNext steps:")
        print("  1. Go to the Data Sources panel → Drive tab")
        print("  2. Run the Drive crawl (Incremental toggle doesn't matter now — cache is empty)")
        print("  3. Files will be re-indexed with owner_id = '__global__'")


if __name__ == "__main__":
    main()
