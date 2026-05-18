"""
crawlers/ — Google data-source crawlers for the Secure AI Assistant pipeline.

Available crawlers:
  gmail_crawler.py  — Fetches Gmail threads and indexes them via /upload
  drive_crawler.py  — Downloads Google Drive files and indexes them via /upload

Both crawlers use OAuth 2.0 (shared via auth.py).  On first run a browser
window opens for the user to authorise access; credentials are cached in
token.json for subsequent runs.

Quick start:
    cd Secure-AI-Assistant-main
    pip install -r crawlers/requirements.txt

    # Gmail — last 50 emails from INBOX
    python crawlers/gmail_crawler.py --label INBOX --max 50

    # Drive — crawl a specific folder (or My Drive root)
    python crawlers/drive_crawler.py --folder-id <FOLDER_ID>

Run with --help for full options.
"""
