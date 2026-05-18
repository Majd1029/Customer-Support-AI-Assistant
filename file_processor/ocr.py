from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from PIL import Image, ImageEnhance, ImageFilter
from tenacity import retry, stop_after_attempt, wait_exponential

# Load .env so GROQ_API_KEY is available when running directly
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


# ── Config ────────────────────────────────────────────────────────────────────

# Poppler binary directory for pdf2image.
# Resolution order:
#   1. POPPLER_PATH environment variable  (set in .env or system env)
#   2. None → pdf2image uses the system PATH  (Linux/macOS, Docker, CI)
# On Windows, set POPPLER_PATH in .env pointing to the Poppler bin/ folder.
POPPLER_PATH: str | None = os.environ.get("POPPLER_PATH") or None

_GROQ_OCR_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"
_GROQ_OCR_PROMPT  = (
    "Extract ALL content from this image exactly as it appears.\n\n"
    "Rules:\n"
    "1. Text: output as clean markdown — preserve headings (#, ##, ###), "
    "bullet lists, numbered lists, tables (as markdown tables), and paragraph breaks.\n"
    "2. If the text is in Arabic or mixed Arabic/English, preserve both scripts.\n"
    "3. Visual elements: for any chart, graph, diagram, figure, photograph, "
    "illustration, or non-text visual element you see, insert a description "
    "in the form [Figure: concise description of what the visual shows, "
    "including key data points or labels if visible] at the position where "
    "it appears in the page flow.\n"
    "4. Output ONLY the extracted content — no meta-commentary, "
    "no explanations about the extraction process."
)
_GROQ_MAX_TOKENS  = 4096


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_for_ocr(img: Image.Image, handwriting: bool = True) -> Image.Image:
    """
    Optional preprocessing pipeline — useful for very low-quality images.
    Llama 4 Scout handles most image conditions natively, so this is only
    applied when explicitly requested (preprocess=True).

    For handwriting:
      - Upscale to 2000 px wide  (larger strokes for the vision encoder)
      - Remove ruled lines       (lined-paper blue/grey lines)
      - High contrast + sharpness

    For print:
      - Upscale to 1000 px wide
      - Standard contrast + sharpness
    """
    img = img.convert("RGB")
    w, h = img.size

    target_w = 2000 if handwriting else 1000
    if w < target_w:
        scale = target_w / w
        img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        logger.debug(f"  preprocess: upscaled {w}×{h} → {img.size}")

    if handwriting:
        import numpy as np
        arr       = np.array(img.convert("L"), dtype=np.float32)
        row_means = arr.mean(axis=1)
        thresh    = float(np.percentile(row_means, 85))
        mask      = row_means > thresh
        arr[mask] = 255.0
        img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")

    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.5 if handwriting else 1.8)
    img = ImageEnhance.Sharpness(img).enhance(3.0 if handwriting else 2.0)
    img = img.filter(
        ImageFilter.UnsharpMask(radius=2, percent=150, threshold=2)
        if handwriting else
        ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3)
    )
    return img.convert("RGB")


# ── Groq client singleton ─────────────────────────────────────────────────────

_GROQ_CLIENT = None


def _get_groq_client():
    """
    Lazy singleton for the Groq client.
    Reads GROQ_API_KEY from the environment / .env file.
    Returns None if the groq package is missing or the key is not set.
    """
    global _GROQ_CLIENT
    if _GROQ_CLIENT is not None:
        return _GROQ_CLIENT
    try:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not set — add it to your .env file")
            return None
        _GROQ_CLIENT = Groq(api_key=api_key)
        logger.info(f"Groq OCR client ready (model: {_GROQ_OCR_MODEL})")
        return _GROQ_CLIENT
    except ImportError:
        logger.warning("groq package not installed. Run: pip install groq")
        return None
    except Exception as e:
        logger.warning(f"Groq client init failed: {e}")
        return None


def glmocr_available() -> bool:
    """Returns True if the Groq client is usable for OCR."""
    try:
        from groq import Groq  # noqa: F401
        return bool(os.getenv("GROQ_API_KEY"))
    except ImportError:
        return False


# ── Core OCR call ─────────────────────────────────────────────────────────────

def _pil_to_base64(pil_img: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _run_glm_ocr_api(client, data_url: str) -> str:
    """Inner call — isolated so tenacity retries only the network hop."""
    response = client.chat.completions.create(
        model=_GROQ_OCR_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text",      "text": _GROQ_OCR_PROMPT},
                ],
            }
        ],
        max_tokens=_GROQ_MAX_TOKENS,
    )
    return (response.choices[0].message.content or "").strip()


def _run_glm_ocr(pil_img: Image.Image) -> str:
    """
    Send a PIL image to Groq (Llama 4 Scout) and return extracted text as markdown.

    Llama 4 Scout is a native multimodal model — it handles Arabic, English,
    mixed scripts, tables, handwriting, and structured documents without any
    language hints.

    Returns empty string on failure.  Retries up to 4 times with exponential
    backoff (2 s → 4 s → 8 s → 16 s) to survive Groq rate-limit (429) spikes.
    """
    client = _get_groq_client()
    if client is None:
        logger.warning("  Groq OCR not available — no text extracted")
        return ""

    try:
        b64      = _pil_to_base64(pil_img)
        data_url = f"data:image/png;base64,{b64}"
        text     = _run_glm_ocr_api(client, data_url)
        logger.debug(f"  Groq OCR: {len(text)} chars extracted")
        return text

    except Exception as e:
        logger.warning(f"  Groq OCR API call failed: {e}")
        return ""


# ── Public API ────────────────────────────────────────────────────────────────

def ocr_image_bytes(
    image_bytes: bytes,
    preprocess: bool = False,
    handwriting: bool = True,
) -> str:
    """
    Run OCR on raw image bytes and return the extracted text as markdown.

    Uses Groq / Llama 4 Scout — handles Arabic, English, mixed scripts,
    tables, and handwriting natively.

    Args:
        image_bytes : raw PNG / JPEG / WEBP / etc. bytes
        preprocess  : apply preprocessing pipeline before OCR (default False —
                      Llama 4 Scout handles most image conditions natively;
                      enable for very noisy or low-contrast scans)
        handwriting : use handwriting-tuned preprocessing when preprocess=True
    """
    pil_img = Image.open(io.BytesIO(image_bytes))

    if preprocess:
        pil_img = preprocess_for_ocr(pil_img, handwriting=handwriting)
    else:
        pil_img = pil_img.convert("RGB")

    return _run_glm_ocr(pil_img)


def ocr_pil_image(
    pil_img: Image.Image,
    preprocess: bool = False,
    handwriting: bool = True,
) -> str:
    """Run OCR on a PIL Image object."""
    if preprocess:
        pil_img = preprocess_for_ocr(pil_img, handwriting=handwriting)
    else:
        pil_img = pil_img.convert("RGB")

    return _run_glm_ocr(pil_img)


def is_scanned_pdf(text_from_pypdf: str, page_count: int) -> bool:
    """Returns True when a PDF yields less than 50 chars/page from pypdf (i.e. it is scanned)."""
    if page_count == 0:
        return False
    return len(text_from_pypdf.strip()) / page_count < 50


def ocr_pdf_pages(
    pdf_path: str | Path,
    dpi: int = 200,
    preprocess: bool = False,
    handwriting: bool = False,
) -> list[str]:
    """
    Rasterise a PDF and run OCR on each page via Groq / Llama 4 Scout.

    Returns a list of markdown strings, one per page.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "pdf2image not installed. Run: pip install pdf2image\n"
            "Windows: https://github.com/oschwartz10612/poppler-windows/releases"
        )

    path = Path(pdf_path)
    logger.info(f"Rasterising '{path.name}' at {dpi} DPI ...")

    try:
        pages = convert_from_path(str(path), dpi=dpi, poppler_path=POPPLER_PATH)
    except Exception as e:
        logger.error(f"Rasterisation failed for '{path.name}': {e}")
        return []

    logger.info(f"Groq OCR on {len(pages)} page(s) ...")
    results = []
    for i, page_img in enumerate(pages, 1):
        logger.info(f"  Page {i}/{len(pages)} ...")
        text = ocr_pil_image(page_img, preprocess=preprocess, handwriting=handwriting)
        results.append(text)
        logger.info(f"  → {len(text)} characters extracted")

    return results


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Test the OCR pipeline directly from the terminal.

    Usage
    -----
    # Image (default — no preprocessing):
        python ocr.py image.png

    # Enable preprocessing (for noisy / low-contrast images):
        python ocr.py image.png --preprocess

    # Handwriting mode (preprocessing tuned for handwriting):
        python ocr.py image.png --preprocess --handwriting

    # PDF (rasterise + OCR per page):
        python ocr.py document.pdf

    # PDF with higher DPI:
        python ocr.py document.pdf --dpi 300
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="OCR runner — Groq / Llama 4 Scout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("file",           help="Image or PDF file to OCR")
    parser.add_argument("--preprocess",   action="store_true",
                        help="Apply preprocessing pipeline before OCR")
    parser.add_argument("--handwriting",  action="store_true",
                        help="Use handwriting-tuned preprocessing (implies --preprocess)")
    parser.add_argument("--dpi",          type=int, default=200,
                        help="DPI for PDF rasterisation (default 200)")
    args = parser.parse_args()

    target       = Path(args.file)
    _preprocess  = args.preprocess or args.handwriting
    _handwriting = args.handwriting

    if not target.exists():
        print(f"Error: file not found — {target}", file=sys.stderr)
        sys.exit(1)

    print(f"\nFile       : {target}")
    print(f"Preprocess : {_preprocess}")
    print(f"Handwriting: {_handwriting}")
    print("-" * 50)

    suffix = target.suffix.lower()

    if suffix == ".pdf":
        pages = ocr_pdf_pages(target, dpi=args.dpi,
                               preprocess=_preprocess, handwriting=_handwriting)
        for i, page_text in enumerate(pages, 1):
            print(f"\n=== Page {i} ===")
            print(page_text if page_text.strip() else "(no text detected)")
    else:
        raw  = target.read_bytes()
        text = ocr_image_bytes(raw, preprocess=_preprocess, handwriting=_handwriting)
        print()
        print(text if text.strip() else "(no text detected)")
        print()
