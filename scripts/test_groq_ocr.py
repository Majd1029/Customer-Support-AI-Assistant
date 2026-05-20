#!/usr/bin/env python
"""
scripts/test_openrouter_ocr.py — CLI smoke-test for the Groq OCR backend.

Exercises the same code path used by the production pipeline
(``file_processor/gemma4.py`` → ``_ocr_via_groq``).

Usage
-----
    # Test with a real image or PDF
    python scripts/test_openrouter_ocr.py path/to/image.png
    python scripts/test_openrouter_ocr.py path/to/document.pdf

    # Generate a synthetic invoice and OCR it (no image file needed)
    python scripts/test_openrouter_ocr.py --demo

    # List supported Groq vision models
    python scripts/test_openrouter_ocr.py --list-models

    # Use a specific model
    python scripts/test_openrouter_ocr.py path/to/image.png --model meta-llama/llama-4-scout-17b-16e-instruct

    # Verbose — also print the full raw API response
    python scripts/test_openrouter_ocr.py --demo --verbose

Environment
-----------
Set ``GROQ_OCR_API_KEY`` (or ``GROQ_API_KEY``) in ``.env`` before running.
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
# Config
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = os.getenv("GROQ_OCR_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# Groq vision models that accept image inputs
GROQ_VISION_MODELS = [
    ("meta-llama/llama-4-scout-17b-16e-instruct", "Llama-4-Scout 17B — Groq multimodal, current default"),
    ("meta-llama/llama-4-maverick-17b-128e-instruct", "Llama-4-Maverick 17B — larger context variant"),
]


def _get_api_key() -> str:
    key = os.getenv("GROQ_OCR_API_KEY") or os.getenv("GROQ_API_KEY", "")
    if not key:
        print("ERROR: GROQ_OCR_API_KEY (or GROQ_API_KEY) is not set.")
        print("  Add it to your .env file:  GROQ_OCR_API_KEY=gsk_...")
        sys.exit(1)
    return key


def _image_to_base64(path: Path) -> tuple[str, str]:
    """Return (base64_string, mime_type) for an image file."""
    data = path.read_bytes()
    if data[:2] == b"\xff\xd8":
        mime = "image/jpeg"
    elif data[:4] == b"\x89PNG":
        mime = "image/png"
    elif data[:6] in (b"GIF87a", b"GIF89a"):
        mime = "image/gif"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        mime = "image/png"
    return base64.b64encode(data).decode(), mime


def _pdf_first_page_to_base64(path: Path) -> tuple[str, str]:
    """Rasterise the first page of a PDF and return (base64_png, mime)."""
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("ERROR: pdf2image is required for PDF testing.")
        print("  Install with: pip install pdf2image")
        sys.exit(1)

    poppler = os.getenv("POPPLER_PATH")
    imgs = convert_from_path(str(path), dpi=150, first_page=1, last_page=1,
                             poppler_path=poppler)
    if not imgs:
        print("ERROR: Could not rasterise PDF.")
        sys.exit(1)

    import io
    from PIL import Image
    buf = io.BytesIO()
    imgs[0].save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), "image/png"


def _make_demo_image() -> Path:
    """Generate a simple synthetic invoice PNG using Pillow and return its path."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("ERROR: Pillow is required for --demo mode.")
        print("  Install with: pip install Pillow")
        sys.exit(1)

    img  = Image.new("RGB", (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_body  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font_title = ImageFont.load_default()
        font_body  = font_title

    lines = [
        ("INVOICE #INV-2025-0042",           40,  font_title, (30, 30, 30)),
        ("Date: 19 May 2025",                80,  font_body,  (60, 60, 60)),
        ("From: Acme Corp",                  110, font_body,  (60, 60, 60)),
        ("To:   Client Inc.",                135, font_body,  (60, 60, 60)),
        ("Item              Qty  Price",     175, font_body,  (30, 30, 30)),
        ("-" * 42,                           195, font_body,  (150, 150, 150)),
        ("Widget A           10   $12.00",   215, font_body,  (60, 60, 60)),
        ("Widget B            5   $25.00",   235, font_body,  (60, 60, 60)),
        ("Consulting          2  $150.00",   255, font_body,  (60, 60, 60)),
        ("-" * 42,                           275, font_body,  (150, 150, 150)),
        ("TOTAL                   $695.00",  295, font_title, (30, 30, 30)),
        ("Payment due: 30 days",             340, font_body,  (100, 100, 100)),
    ]
    for text, y, font, color in lines:
        if text:
            draw.text((40, y), text, fill=color, font=font)

    out_path = Path("/tmp/demo_invoice.png")
    img.save(out_path)
    return out_path


def call_groq_ocr(
    image_path: Path,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> str:
    """Send an image (or first page of a PDF) to Groq and return extracted text."""
    try:
        from groq import Groq
    except ImportError:
        print("ERROR: groq SDK is required.")
        print("  Install with: pip install groq")
        sys.exit(1)

    api_key = _get_api_key()

    # PDF → rasterise first page; image → read directly
    suffix = image_path.suffix.lower()
    if suffix == ".pdf":
        print(f"[OCR]  PDF detected — rasterising first page at 150 DPI …")
        b64, mime = _pdf_first_page_to_base64(image_path)
    else:
        b64, mime = _image_to_base64(image_path)

    client = Groq(api_key=api_key)

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
        messages=[{
            "role": "user",
            "content": [
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }],
        max_tokens=2048,
        temperature=0.0,
    )
    elapsed = time.time() - t0

    if verbose:
        print("\n── Raw API response ─────────────────────────────────────────")
        print(response)
        print("─────────────────────────────────────────────────────────────\n")

    if not response or not getattr(response, "choices", None):
        return "[ERROR] Groq returned no choices (rate-limited or model error)."

    text   = response.choices[0].message.content or ""
    tokens = getattr(getattr(response, "usage", None), "total_tokens", "?")
    print(f"\n[INFO] Model: {model}  |  {elapsed:.1f}s  |  ~{tokens} tokens\n")
    return text


def list_models() -> None:
    """Print supported Groq vision models."""
    print("\nGroq vision models supported for OCR:")
    print("=" * 65)
    for model_id, description in GROQ_VISION_MODELS:
        marker = " ← default" if model_id == DEFAULT_MODEL else ""
        print(f"  {model_id}{marker}")
        print(f"    {description}")
    print()
    print("Set GROQ_OCR_MODEL in .env to override the default.")
    print("Full model list: https://console.groq.com/docs/models\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the Groq OCR backend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "image", nargs="?", type=Path,
        help="Path to an image or PDF file",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Generate a synthetic invoice and OCR it",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List supported Groq vision models and exit",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Groq model ID (default: {DEFAULT_MODEL})",
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
    result = call_groq_ocr(image_path, model=args.model, verbose=args.verbose)

    print("── Extracted text ───────────────────────────────────────────")
    print(result)
    print("─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
