"""
test_together_ocr.py — Test OpenRouter vision API as an OCR backend.

OpenRouter gives access to many vision models, including several free ones.
It uses the OpenAI-compatible API format.

Usage:
    python test_together_ocr.py image.png
    python test_together_ocr.py scanned_doc.pdf --page 1
    python test_together_ocr.py --demo          # runs with a generated test image
    python test_together_ocr.py image.png --model "qwen/qwen2.5-vl-72b-instruct:free"

Setup:
    1. Get a free key at https://openrouter.ai
    2. Add to .env:  OPENROUTER_API_KEY=sk-or-v1-...
    3. pip install openai

Free vision models on OpenRouter (no credits needed):
    qwen/qwen2.5-vl-72b-instruct:free             ← best OCR quality (recommended)
    qwen/qwen2.5-vl-32b-instruct:free             ← faster, still excellent
    google/gemma-3-27b-it:free                    ← Google Gemma 3 multimodal
    google/gemma-3-12b-it:free                    ← lighter Gemma 3
    meta-llama/llama-3.2-11b-vision-instruct:free ← Llama 3.2 vision
    moonshotai/kimi-vl-a3b-thinking:free          ← Moonshot reasoning model
    mistralai/mistral-small-3.1-24b-instruct:free ← Mistral multimodal
    baidu/qianfan-ocr-fast:free                   ← dedicated OCR model (fast, 66K ctx)
"""

import sys
import os
import base64
import argparse
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
except ImportError:
    pass

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Best free model for OCR on OpenRouter (currently active)
DEFAULT_MODEL = "baidu/qianfan-ocr-fast:free"

OCR_PROMPT = (
    "You are an OCR engine. Extract ALL text from this image exactly as it appears. "
    "Preserve headings, tables, bullet points, and numbers. "
    "Output only the extracted text — no commentary, no preamble."
)

FREE_MODELS = [
    ("baidu/qianfan-ocr-fast:free",                   "dedicated OCR, 66K ctx ← WORKING"),
    ("qwen/qwen2.5-vl-72b-instruct:free",             "best quality when available"),
    ("qwen/qwen2.5-vl-32b-instruct:free",             "faster Qwen VL"),
    ("google/gemma-3-27b-it:free",                    "Google Gemma 3 multimodal"),
    ("google/gemma-3-12b-it:free",                    "lighter Gemma 3"),
    ("meta-llama/llama-3.2-11b-vision-instruct:free", "Llama 3.2 vision"),
    ("moonshotai/kimi-vl-a3b-thinking:free",          "Moonshot reasoning"),
    ("mistralai/mistral-small-3.1-24b-instruct:free", "Mistral multimodal"),
]


# ── OpenRouter call ───────────────────────────────────────────────────────────

def ocr_with_openrouter(
    image_b64: str,
    model:     str = DEFAULT_MODEL,
    mime:      str = "image/png",
) -> str:
    """
    Send a base64-encoded image to OpenRouter and return the OCR text.
    Uses the OpenAI-compatible API with a custom base URL.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set.\n"
            "Get a free key at https://openrouter.ai\n"
            "Then add to .env:  OPENROUTER_API_KEY=sk-or-v1-..."
        )

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package not installed. Run:  pip install openai"
        )

    client = OpenAI(
        api_key  = OPENROUTER_API_KEY,
        base_url = OPENROUTER_BASE_URL,
    )

    response = client.chat.completions.create(
        model    = model,
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type":      "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": OCR_PROMPT,
                    },
                ],
            }
        ],
        max_tokens  = 2048,
        temperature = 0.0,
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenRouter returned an empty response (content=None).")
    return content.strip()


# ── Helpers ───────────────────────────────────────────────────────────────────

def image_to_b64(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    mime = {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif":  "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    return base64.b64encode(path.read_bytes()).decode("utf-8"), mime


def pdf_page_to_b64(pdf_path: Path, page_num: int = 1) -> tuple[str, str]:
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("ERROR: pdf2image not installed.  Run:  pip install pdf2image")
        sys.exit(1)

    poppler_path = os.getenv("POPPLER_PATH") or None
    pages = convert_from_path(
        str(pdf_path),
        first_page   = page_num,
        last_page    = page_num,
        dpi          = 200,
        poppler_path = poppler_path,
    )
    if not pages:
        raise ValueError(f"PDF page {page_num} not found in {pdf_path}")

    import io
    buf = io.BytesIO()
    pages[0].save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/png"


def make_demo_image() -> tuple[str, str]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("ERROR: Pillow not installed.  Run:  pip install pillow")
        sys.exit(1)

    img  = Image.new("RGB", (600, 300), color="white")
    draw = ImageDraw.Draw(img)

    try:
        font_big   = ImageFont.truetype("arial.ttf", 28)
        font_small = ImageFont.truetype("arial.ttf", 18)
    except (IOError, OSError):
        font_big   = ImageFont.load_default()
        font_small = font_big

    draw.text((30, 30),  "Invoice #INV-2024-0042",         fill="black", font=font_big)
    draw.text((30, 80),  "Date: 2024-11-15",                fill="black", font=font_small)
    draw.text((30, 110), "Client: Acme Corporation",        fill="black", font=font_small)
    draw.text((30, 150), "Item          Qty    Unit    Total", fill="black", font=font_small)
    draw.line([(30, 170), (570, 170)], fill="black", width=1)
    draw.text((30, 175), "Consulting    10h    $150    $1,500", fill="black", font=font_small)
    draw.text((30, 200), "Hosting        1     $200      $200",  fill="black", font=font_small)
    draw.line([(30, 225), (570, 225)], fill="black", width=1)
    draw.text((30, 235), "TOTAL                           $1,700", fill="black", font=font_big)

    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/png"


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test OpenRouter vision API for OCR"
    )
    parser.add_argument("image",   nargs="?",            help="Image or PDF file path")
    parser.add_argument("--page",  type=int, default=1,  help="PDF page number (default: 1)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model ID")
    parser.add_argument("--demo",  action="store_true",  help="Use a generated test image")
    parser.add_argument("--list-models", action="store_true", help="Show available free models")
    args = parser.parse_args()

    if args.list_models:
        print("Free vision models on OpenRouter:")
        for model_id, desc in FREE_MODELS:
            marker = "  ← default" if model_id == DEFAULT_MODEL else ""
            print(f"  {model_id:<55} {desc}{marker}")
        return

    if not args.demo and not args.image:
        parser.print_help()
        print("\nTip: run with --demo to test without an image file.")
        sys.exit(1)

    # ── Prepare image ─────────────────────────────────────────────────────────
    if args.demo:
        print("Generating demo invoice image …")
        b64, mime = make_demo_image()
        label = "demo-invoice.png"
    else:
        path = Path(args.image)
        if not path.exists():
            print(f"ERROR: file not found: {path}")
            sys.exit(1)
        if path.suffix.lower() == ".pdf":
            print(f"Rasterising PDF page {args.page} …")
            b64, mime = pdf_page_to_b64(path, args.page)
        else:
            b64, mime = image_to_b64(path)
        label = path.name

    # ── Call OpenRouter ───────────────────────────────────────────────────────
    import time

    print(f"\nSending '{label}' to OpenRouter …")
    print(f"  API key: {'set ✓' if OPENROUTER_API_KEY else 'NOT SET ✗'}\n")

    # If the user explicitly passed --model, try only that model.
    # If using the default, auto-fallback through the full FREE_MODELS list.
    user_specified = args.model != DEFAULT_MODEL or "--model" in sys.argv
    if user_specified:
        models_to_try = [args.model]
    else:
        models_to_try = [m for m, _ in FREE_MODELS]

    last_error = None
    for model_id in models_to_try:
        print(f"  Trying : {model_id}")
        t0 = time.time()
        try:
            result  = ocr_with_openrouter(b64, model=model_id, mime=mime)
            elapsed = time.time() - t0
            print("─" * 60)
            print(result)
            print("─" * 60)
            print(f"\n✓  Done in {elapsed:.1f}s  ({len(result)} chars extracted)")
            print(f"   Model used: {model_id}")
            sys.exit(0)
        except Exception as e:
            last_error = e
            msg = str(e)
            if "404" in msg or "No endpoints" in msg:
                print(f"  ✗  Model unavailable (404) — trying next …")
            else:
                # Non-404 errors (auth, rate limit, etc.) — stop immediately
                print(f"✗  OCR failed: {e}")
                sys.exit(1)

    print(f"\n✗  All models failed. Last error: {last_error}")
    print("\nRun  python test_together_ocr.py --list-models  to see available options.")
    sys.exit(1)


if __name__ == "__main__":
    main()
