"""
reset_qdrant.py — Delete every collection in the configured Qdrant instance so
the deployment starts empty and each user uploads their own documents.

The app recreates its collections ("documents", "user_preferences") on the next
upload, so nothing else needs to be set up afterwards.

Connection comes from QDRANT_URL / QDRANT_API_KEY (env or .env), falling back
to localhost:6333.

Usage:
    python scripts/reset_qdrant.py              # dry run: list collections and point counts
    python scripts/reset_qdrant.py --confirm    # actually delete them
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)

from file_preparation.indexing.store import get_client  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confirm", action="store_true", help="delete the collections (default: dry run)")
    args = parser.parse_args()

    client = get_client()
    collections = [c.name for c in client.get_collections().collections]
    if not collections:
        print("Qdrant is already empty — nothing to delete.")
        return 0

    for name in collections:
        count = client.count(collection_name=name, exact=True).count
        print(f"  {name}: {count} point(s)")

    if not args.confirm:
        print(f"\nDry run: {len(collections)} collection(s) would be deleted. Re-run with --confirm.")
        return 0

    for name in collections:
        client.delete_collection(collection_name=name)
        print(f"  deleted {name}")
    print("\nDone. Qdrant is empty; collections are recreated on the next upload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
