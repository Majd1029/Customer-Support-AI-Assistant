#!/usr/bin/env python
"""
scripts/test_openrouter_ocr.py — CLI smoke-test for the OpenRouter OCR backend.

Exercises the same code path used by the production pipeline
(``file_processor/gemma4.py`` → ``_ocr_via_openrouter``).

Usage
-----
    # Test with a real image
    python scripts/test_openrouter_ocr.py path/to/image.png

    # Generate a synthetic invoice and OCR it (no image file needed)
    python scripts/test_openrouter_ocr.py --demo

    # List all free vision/OCR models available on OpenRouter
    python scripts/test_openrouter_ocr.py --list-models

    # Use a specific model
    python scripts/test_openrouter_ocr.py path/to/image.png --model qwen/qwen2.5-vl-72b-instruct:free

    # Verbose — also print the full raw API response
    python scripts/test_openrouter_ocr.py --demo --verbose

Environment
-----------
Set ``OPENROUTER_API_KEY`` in ``.env`` (or export it) before running.
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

# ── Ensure project root is importable ────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=_ROOT / ".env")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = os.getenv("OPENROUTER_OCR_MODEL", "baidu/qianfan-ocr-fast:free")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Free vision models known to work well for OCR tasks.
FREE_VISION_MODELS = [
    ("baidu/qianfan-ocr-fast:free",          "Dedicated OCR model — fastest, best for text extraction"),
    ("qwen/qwen2.5-vl-72b-instruct:free",    "Qwen2.5-VL 72B — strong general vision + OCR"),
    ("qwen/qwen2.5-vl-7b-instruct:free",     "Qwen2.5-VL 7B  — lighter weight, still solid"),
    ("google/gemma-3-27b-it:free",           "Gemma 3 27B — good for structured extraction"),
    ("mistralai/pixtral-12b:free",           "Pixtral 12B  — Mistral's vision model"),
    ("meta-llama/llama-4-scout:free",        "Llama-4-Scout — Meta's latest multimodal"),
]


def _get_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        print("ERROR: OPENROUTER_API_KEY is not set.")
        print("  Add it to your .env file or export it before running.")
        sys.exit(1)
    return key


def _image_to_base64(path: Path) -> tuple[str, str]:
    """Return (base64_string, mime_type) for an image file."""
    data = path.read_bytes()
    # Detect MIME from magic bytes
    if data[:2] == b"\xff\xd8":
        mime = "image/jpeg"
    elif data[:4] == b"\x89PNG":
        mime = "image/png"
    elif data[:6] in (b"GIF87a", b"GIF89a"):
        mime = "image/gif"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        mime = "image/png"  # safest default
    return base64.b64encode(data).decode(), mime


def _make_demo_image() -> Path:
    """Generate a simple synthetic invoice PNG using Pillow and return its path."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("ERROR: Pillow is required for --demo mode.")
        print("  Install with: pip install Pillow --break-system-packages")
        sys.exit(1)

    img = Image.new("RGB", (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Try to use a basic font; fall back to default if not available
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_body  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font_title = ImageFont.load_default()
        font_body  = font_title

    lines = [
        ("INVOICE #INV-2025-0042",     40,  font_title, (30, 30, 30)),
        ("Date: 19 May 2025",          80,  font_body,  (60, 60, 60)),
        ("From: Acme Corp",            110, font_body,  (60, 60, 60)),
        ("To:   Client Inc.",          135, font_body,  (60, 60, 60)),
        ("",                           0,   font_body,  (0, 0, 0)),
        ("Item              Qty  Price", 175, font_body, (30, 30, 30)),
        ("-" * 42,                     195, font_body,  (150, 150, 150)),
        ("Widget A           10   $12.00", 215, font_body, (60, 60, 60)),
        ("Widget B            5   $25.00", 235, font_body, (60, 60, 60)),
        ("Consulting          2  $150.00", 255, font_body, (60, 60, 60)),
        ("-" * 42,                     275, font_body,  (150, 150, 150)),
        ("TOTAL                       $695.00", 295, font_title, (30, 30, 30)),
        ("Payment due: 30 days",       340, font_body,  (100, 100, 100)),
        ("Bank: IBAN FR76 0000 0000 0042", 360, font_body, (100, 100, 100)),
    ]

    for text, y, font, color in lines:
        if text:
            draw.text((40, y), text, fill=color, font=font)

    out_path = Path("/tmp/demo_invoice.png")
    img.save(out_path)
    return out_path


def call_openrouter_ocr(
    image_path: Path,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> str:
    """Send an image to OpenRouter and return the extracted text."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai SDK is required.")
        print("  Install with: pip install openai --break-system-packages")
        sys.exit(1)

    api_key = _get_api_key()
    b64, mime = _image_to_base64(image_path)

    client = OpenAI(
        base_url=OPENROUTER_BASE,
        api_key=api_key,
    )

    prompt = (
        "Extract all text from this image. "
        "Preserve the original layout as closely as possible. "
        "For tables, use a markdown table. "
        "For headings, use markdown headings. "
        "Output only the extracted text — no commentary."
    )

    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text",      "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ],
        max_tokens=2048,
    )
    elapsed = time.time() - t0

    if verbose:
        print("\n── Raw API response ─────────────────────────────────────────")
        print(response)
        print("─────────────────────────────────────────────────────────────\n")

    if not response or not getattr(response, "choices", None):
        return f"[ERROR] OpenRouter returned no choices (rate-limited or model error)."

    text = response.choices[0].message.content or ""
    tokens = getattr(getattr(response, "usage", None), "total_tokens", "?")
    print(f"\n[INFO] Model: {model}  |  {elapsed:.1f}s  |  ~{tokens} tokens\n")
    return text


def list_models() -> None:
    """Print the curated list of free vision models."""
    print("\nFree vision / OCR models on OpenRouter:")
    print("=" * 65)
    for model_id, description in FREE_VISION_MODELS:
        marker = " ← default" if model_id == DEFAULT_MODEL else ""
        print(f"  {model_id}{marker}")
        print(f"    {description}")
    print()
    print("Set OPENROUTER_OCR_MODEL in .env to override the default.")
    print("Full model list: https://openrouter.ai/models?modality=vision\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the OpenRouter OCR backend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "image", nargs="?", type=Path,
        help="Path to an image file (PNG, JPG, WEBP …)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Generate a synthetic invoice and OCR it",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available free vision models and exit",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"OpenRouter model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Also print the raw API response object",
    )
    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    if args.demo:
        print("[DEMO] Generating synthetic invoice image …")
        image_path = _make_demo_image()
        print(f"[DEMO] Saved to {image_path}")
    elif args.image:
        image_path = args.image
        if not image_path.exists():
            print(f"ERROR: File not found: {image_path}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)

    print(f"[OCR]  Sending {image_path.name} → {args.model} …")
    result = call_openrouter_ocr(image_path, model=args.model, verbose=args.verbose)

    print("── Extracted text ───────────────────────────────────────────")
    print(result)
    print("─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
