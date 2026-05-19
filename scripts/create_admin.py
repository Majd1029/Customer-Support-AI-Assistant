#!/usr/bin/env python3
"""
scripts/create_admin.py — Bootstrap the first admin user.

Run once after setting up the database:

    python scripts/create_admin.py

Or with explicit credentials (useful in CI/CD):

    python scripts/create_admin.py --username admin --password secret --email admin@example.com

  Admin user created successfully!
  username : admin
  user_id  : e3b54878-8abe-48f3-895e-cdba20b6a9f4
  role     : admin
  email    : admin@example.com

  JWT token (30-day expiry):
  eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJlM2I1NDg3OC04YWJlLTQ4ZjMtODk1ZS1jZGJhMjBiNmE5ZjQiLCJ1c2VybmFtZSI6ImFkbWluIiwiZW1haWwiOiJhZG1pbkBleGFtcGxlLmNvbSIsInJvbGUiOiJhZG1pbiIsImV4cCI6MTc4MTc3OTY2OH0.JR1_O7yTg3paNEX8aoIXb4bwFrTQgqTjpoBhLV1yvJk

Use this token in the Authorization header:
  Authorization: Bearer <token>

Or log in via POST /auth/login with your credentials.

The script prints the generated JWT token so you can test authenticated
admin endpoints immediately.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from file_processor.auth_manager import create_admin_user, login_user


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the first admin user for the Customer Support AI Assistant.",
    )
    parser.add_argument("--username", default="admin",          help="Admin username (default: admin)")
    parser.add_argument("--password", default=None,             help="Admin password (prompted if omitted)")
    parser.add_argument("--email",    default="",               help="Admin email (optional)")
    args = parser.parse_args()

    password = args.password
    if not password:
        import getpass
        password = getpass.getpass(f"Password for '{args.username}': ")
        confirm  = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("ERROR: Passwords do not match.", file=sys.stderr)
            sys.exit(1)

    if len(password) < 4:
        print("ERROR: Password must be at least 4 characters.", file=sys.stderr)
        sys.exit(1)

    print(f"\nCreating admin user '{args.username}'...")
    user, err = create_admin_user(args.username, password, args.email)

    if err or not user:
        # If already exists, try logging in to confirm it is indeed admin
        if "already taken" in (err or ""):
            print(f"Username '{args.username}' already exists — attempting login to verify role...")
            logged_in, login_err = login_user(args.username, password)
            if login_err or not logged_in:
                print(f"ERROR: {login_err or 'Unknown login error'}", file=sys.stderr)
                sys.exit(1)
                
            if logged_in.get("role") != "admin":
                print(f"WARNING: User '{args.username}' exists but has role '{logged_in['role']}', not 'admin'.")
                print("Use the /admin/users/{id}/role endpoint to promote them.")
            else:
                print(f"Admin user '{args.username}' already exists.")
                print(f"  role    : {logged_in['role']}")
                print(f"  user_id : {logged_in['user_id']}")
                print(f"  token   : {logged_in['token']}")
        else:
            print(f"ERROR: {err or 'Unknown error creating user'}", file=sys.stderr)
            sys.exit(1)
        return

    print("\n✓ Admin user created successfully!")
    print(f"  username : {user['username']}")
    print(f"  user_id  : {user['user_id']}")
    print(f"  role     : {user['role']}")
    print(f"  email    : {user.get('email') or '(none)'}")
    print(f"\n  JWT token (30-day expiry):")
    print(f"  {user['token']}")
    print(
        "\nUse this token in the Authorization header:\n"
        "  Authorization: Bearer <token>\n"
        "\nOr log in via POST /auth/login with your credentials."
    )


if __name__ == "__main__":
    main()
