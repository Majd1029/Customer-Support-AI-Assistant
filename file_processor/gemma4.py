"""
OCR with Ollama + Gemma (vision model)
Supports: PNG, JPG, WEBP, BMP, TIFF, PDF

Requirements:
    pip install ollama pillow pdf2image
    # PDF support also needs poppler:
    # Windows  → download from https://github.com/oschwartz10612/poppler-windows/releases
    #            extract ZIP and add the bin\ folder to your system PATH
    # Linux    → sudo apt install poppler-utils
    # macOS    → brew install poppler
"""

import base64
import json
import os
import re
import sys
import io
from pathlib import Path
from enum import Enum

import ollama
from PIL import Image
from loguru import logger

# ── Load .env ──────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gemma4:e4b"
OLLAMA_HOST   = "http://localhost:11434"
PDF_DPI       = 150   # higher = better quality but slower (150 is a good balance)

# ─── OpenRouter config ────────────────────────────────────────────────────────

OPENROUTER_API_KEY   = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL  = "https://openrouter.ai/api/v1"
OPENROUTER_OCR_MODEL = os.getenv("OPENROUTER_OCR_MODEL", "baidu/qianfan-ocr-fast:free")

# ─── OCR Modes ────────────────────────────────────────────────────────────────

class OCRMode(str, Enum):
    EXTRACT     = "extract"
    STRUCTURED  = "structured"
    TABLE       = "table"
    HANDWRITING = "handwriting"
    CUSTOM      = "custom"


PROMPTS = {
    OCRMode.EXTRACT: (
        "Extract ALL text visible in this image exactly as it appears. "
        "Preserve formatting, line breaks, and spacing. "
        "Output only the extracted text with no commentary."
    ),
    OCRMode.STRUCTURED: (
        "Extract all text from this image and return it as structured JSON. "
        "Include fields like 'title', 'body', 'labels', 'numbers', and any "
        "other logical groupings you detect. Return only valid JSON, no markdown."
    ),
    OCRMode.TABLE: (
        "Detect and extract any tables in this image. "
        "Format them as markdown tables with aligned columns. "
        "If no table is found, extract all visible text instead."
    ),
    OCRMode.HANDWRITING: (
        "This image may contain handwritten text. Carefully transcribe ALL "
        "handwritten and printed text you can see, character by character. "
        "Output only the transcribed text."
    ),
}


# ─── OpenRouter backend ───────────────────────────────────────────────────────

def openrouter_available() -> bool:
    """
    True when OPENROUTER_API_KEY is set in .env and the openai SDK is installed.
    Does not make a network call.
    """
    if not OPENROUTER_API_KEY:
        return False
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        logger.warning(
            "  [OCR] OPENROUTER_API_KEY is set but 'openai' is not installed. "
            "Run: pip install openai"
        )
        return False


def _ocr_via_openrouter(
    img_b64: str,
    mode: OCRMode = OCRMode.EXTRACT,
    mime: str = "image/png",
) -> str:
    """
    Send a base64-encoded image to OpenRouter and return the OCR text.

    Uses the OpenAI-compatible chat API.  Raises on any error so the caller
    can fall back to Gemma4/Ollama.
    """
    from openai import OpenAI

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    prompt = PROMPTS.get(mode, PROMPTS[OCRMode.EXTRACT])

    response = client.chat.completions.create(
        model=OPENROUTER_OCR_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                {"type": "text",      "text": prompt},
            ],
        }],
        max_tokens=2048,
        temperature=0.0,
    )
    
    if not response or not getattr(response, "choices", None):
        raise ValueError("OpenRouter returned no choices (rate-limited or model error).")
    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenRouter returned an empty response (content=None).")
    return content.strip()


# ─── PDF → images ─────────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: str | Path, dpi: int = PDF_DPI) -> list[Image.Image]:
    """Convert every page of a PDF to a PIL Image."""
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "pdf2image is not installed.\n"
            "Run: pip install pdf2image\n\n"
            "Also install Poppler:\n"
            "  Windows → https://github.com/oschwartz10612/poppler-windows/releases\n"
            "            Extract the ZIP and add the bin\\ folder to your system PATH\n"
            "  Linux   → sudo apt install poppler-utils\n"
            "  macOS   → brew install poppler"
        )
    return convert_from_path(str(pdf_path), dpi=dpi)


# ─── Image loader ─────────────────────────────────────────────────────────────

def _pil_to_base64(img: Image.Image) -> str:
    """Convert a PIL Image to a base64 string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _load_as_pil(source) -> list[Image.Image]:
    """
    Accept a file path, bytes, or PIL Image.
    Always returns a list of PIL Images (one per page for PDFs).
    """
    if isinstance(source, Image.Image):
        return [source]

    if isinstance(source, bytes):
        return [Image.open(io.BytesIO(source))]

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".pdf":
        return pdf_to_images(path)

    return [Image.open(path)]


# ─── Core OCR ─────────────────────────────────────────────────────────────────

def _ocr_single_image(
    img: Image.Image,
    prompt: str,
    model: str,
    stream: bool,
    client: ollama.Client,
) -> str:
    """Run OCR on a single PIL Image.

    Handles both old Ollama SDK (dict responses) and new SDK (object responses).
    """
    img_b64 = _pil_to_base64(img)
    messages = [{"role": "user", "content": prompt, "images": [img_b64]}]

    def _extract_content(obj) -> str:
        """Extract content string from either a dict or an SDK object."""
        if isinstance(obj, dict):
            return obj["message"]["content"]
        return obj.message.content  # newer Ollama SDK returns ChatResponse objects

    if stream:
        full = ""
        for chunk in client.chat(model=model, messages=messages, stream=True):
            token = _extract_content(chunk)
            logger.debug(token)
            full += token
        return full
    else:
        response = client.chat(model=model, messages=messages)
        return _extract_content(response)


def ocr(
    image,
    mode: OCRMode = OCRMode.EXTRACT,
    custom_prompt: str = "",
    model: str = DEFAULT_MODEL,
    stream: bool = False,
) -> str:
    """
    Run OCR on an image or PDF.

    Args:
        image         : file path (str/Path), bytes, or PIL.Image
                        PDFs are automatically split into pages.
        mode          : OCRMode enum value
        custom_prompt : used when mode=OCRMode.CUSTOM
        model         : Ollama model name (must support vision)
        stream        : stream tokens to stdout while generating

    Returns:
        Extracted text. For multi-page PDFs, pages are separated by
        '--- Page N ---' markers.
    """
    if mode == OCRMode.CUSTOM:
        if not custom_prompt:
            raise ValueError("custom_prompt is required when mode=OCRMode.CUSTOM")
        prompt = custom_prompt
    else:
        prompt = PROMPTS[mode]

    pages = _load_as_pil(image)
    client = ollama.Client(host=OLLAMA_HOST)

    if len(pages) == 1:
        return _ocr_single_image(pages[0], prompt, model, stream, client)

    # Multi-page PDF — process page by page
    results = []
    for i, page_img in enumerate(pages, 1):
        logger.info(f"  Gemma OCR — page {i}/{len(pages)} ...")
        text = _ocr_single_image(page_img, prompt, model, stream, client)
        results.append(f"--- Page {i} ---\n{text}")

    return "\n\n".join(results)


# ─── Convenience wrappers ─────────────────────────────────────────────────────

def extract_text(image, model=DEFAULT_MODEL) -> str:
    """Extract plain text from an image or PDF."""
    return ocr(image, mode=OCRMode.EXTRACT, model=model)


def extract_json(image, model=DEFAULT_MODEL) -> dict:
    """Extract structured data as a Python dict."""
    raw = ocr(image, mode=OCRMode.STRUCTURED, model=model)
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    return json.loads(clean)


def extract_table(image, model=DEFAULT_MODEL) -> str:
    """Extract tables as markdown."""
    return ocr(image, mode=OCRMode.TABLE, model=model)


def extract_handwriting(image, model=DEFAULT_MODEL) -> str:
    """Transcribe handwritten text."""
    return ocr(image, mode=OCRMode.HANDWRITING, model=model)


def ask(image, question: str, model=DEFAULT_MODEL) -> str:
    """Ask a free-form question about text in an image or PDF."""
    return ocr(image, mode=OCRMode.CUSTOM, custom_prompt=question, model=model)


# ─── Batch processing ─────────────────────────────────────────────────────────

def batch_ocr(
    paths: list,
    mode: OCRMode = OCRMode.EXTRACT,
    model: str = DEFAULT_MODEL,
    on_progress=None,
) -> dict[str, str]:
    """
    Run OCR on multiple images/PDFs.

    Returns:
        dict mapping file path -> extracted text
    """
    results = {}
    total = len(paths)

    for i, path in enumerate(paths, 1):
        try:
            text = ocr(path, mode=mode, model=model)
            results[str(path)] = text
        except Exception as e:
            results[str(path)] = f"ERROR: {e}"

        if on_progress:
            on_progress(i, total, path, results[str(path)])

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _cli():
    
    global PDF_DPI

    import argparse

    parser = argparse.ArgumentParser(
        description="OCR via Ollama vision model — supports images and PDFs"
    )
    parser.add_argument("image", help="Path to image or PDF file")
    parser.add_argument(
        "--mode", choices=[m.value for m in OCRMode],
        default=OCRMode.EXTRACT.value,
    )
    parser.add_argument("--prompt", default="", help="Custom prompt (with --mode custom)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--output", default="output.txt", help="Save result to file")
    parser.add_argument("--dpi", type=int, default=PDF_DPI,
                        help=f"DPI for PDF rendering (default: {PDF_DPI})")

    args = parser.parse_args()

    PDF_DPI = args.dpi

    print(f"[OCR] model={args.model}  mode={args.mode}  file={args.image}\n")

    result = ocr(
        image=args.image,
        mode=OCRMode(args.mode),
        custom_prompt=args.prompt,
        model=args.model,
        stream=args.stream,
    )

    if not args.stream:
        print(result)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
        print(f"\n[saved to {args.output}]")


# ─── Availability check ───────────────────────────────────────────────────────

def gemma4_available(model: str = DEFAULT_MODEL) -> bool:
    """
    Check if Ollama is running and the Gemma vision model is loaded.
    Returns False silently if Ollama is not reachable.
    """
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        models_resp = client.list()
        # Handle both old API (list of dicts) and new API (object with .models)
        raw = models_resp if isinstance(models_resp, dict) else vars(models_resp)
        model_list = raw.get("models", [])
        loaded = []
        for m in model_list:
            name = m["name"] if isinstance(m, dict) else getattr(m, "model", getattr(m, "name", ""))
            loaded.append(name)
        logger.debug(f"  Ollama models available: {loaded}")
        result = any(model in m for m in loaded)
        if not result:
            logger.warning(f"  gemma4_available: model '{model}' not found in {loaded}")
        return result
    except Exception as e:
        logger.warning(f"  gemma4_available: Ollama unreachable — {e}")
        return False


# ─── Pipeline-compatible bridge ───────────────────────────────────────────────

def ocr_image_bytes_gemma(
    image_bytes: bytes,
    mode: OCRMode = OCRMode.EXTRACT,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    OCR from raw image bytes with automatic backend selection.

    Backend priority:
      1. OpenRouter cloud API  — primary when OPENROUTER_API_KEY is set in .env
                                 Model: OPENROUTER_OCR_MODEL (default: baidu/qianfan-ocr-fast:free)
      2. Gemma4 via Ollama     — local fallback (or primary when no API key)

    Args:
        image_bytes : raw PNG / JPEG bytes
        mode        : OCRMode (default EXTRACT; use HANDWRITING for handwritten docs)
        model       : Ollama model name (used only by the Gemma4 fallback path)
    """
    # ── Primary: OpenRouter ───────────────────────────────────────────────────
    if openrouter_available():
        try:
            # Detect MIME from magic bytes
            mime = "image/jpeg" if image_bytes[:2] == b"\xff\xd8" else "image/png"
            img_b64 = base64.b64encode(image_bytes).decode("utf-8")
            result = _ocr_via_openrouter(img_b64, mode=mode, mime=mime)
            logger.info(f"  [OCR] OpenRouter ({OPENROUTER_OCR_MODEL}) ✓")
            return result
        except Exception as e:
            logger.warning(f"  [OCR] OpenRouter failed — falling back to Gemma4: {e}")

    # ── Fallback: Gemma4 / Ollama ─────────────────────────────────────────────
    logger.info(f"  [OCR] Gemma4 via Ollama ({model})")
    pil_img = Image.open(io.BytesIO(image_bytes))
    return ocr(pil_img, mode=mode, model=model, stream=False)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        _cli()
    else:
        # Quick demo
        IMG = "sample.png"
        print(extract_text(IMG))