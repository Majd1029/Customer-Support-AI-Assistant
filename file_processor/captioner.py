"""
captioner.py — generates a textual description for each extracted image.

Backend priority:
  1. Groq  — meta-llama/llama-4-scout-17b-16e-instruct  (primary, cloud API)
  2. LLaVA — local Ollama                                (automatic fallback)

If Groq fails for any reason (API error, rate-limit, key missing), the
pipeline immediately retries the same image with LLaVA.  If LLaVA is also
unavailable, caption is set to None and the image chunk is silently dropped
by the quality filter downstream.

The fallback is per-image, not per-batch, so a single rate-limit error on
image 3 does not force the whole batch onto LLaVA.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from file_processor.models import ExtractedImage

# ── Concurrency ──────────────────────────────────────────────────────────────
# Groq is I/O-bound (HTTP) — 4 parallel calls are safe and ~4× faster.
# LLaVA runs in a separate Ollama process — 2 parallel calls are a safe default.
_CAPTION_WORKERS: dict[str, int] = {"groq": 4, "llava": 2}

# ── LLaVA (Ollama) ───────────────────────────────────────────────────────────
OLLAMA_MODEL = "llava"

LLAVA_PROMPT = (
    "Document image for a search index. "
    "Reply in exactly 2 sentences: "
    "(1) Visual type and main topic. "
    "(2) Key data, labels, or text visible. "
    "Factual only. No intro, no opinion. Max 60 words."
)

# ── Backend availability checks ───────────────────────────────────────────────

def _ollama_available() -> bool:
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


def _groq_available() -> bool:
    try:
        import os
        if not os.getenv("GROQ_API_KEY", ""):
            return False
        from file_processor.groq_client import caption_image_groq   # noqa: F401
        return True
    except Exception:
        return False


# ── LLaVA call ───────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
def _call_llava(b64: str) -> str:
    import ollama
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": LLAVA_PROMPT, "images": [b64]}],
    )
    return resp["message"]["content"].strip()


# ── Single-image captioner with automatic fallback ────────────────────────────

def caption_image(image: ExtractedImage, backend: str = "groq") -> ExtractedImage:
    """
    Caption a single image.

    Tries the requested backend first.  If it fails and the backend is "groq",
    automatically retries with LLaVA (Ollama) before giving up.

    backend: "groq"  → Groq (primary); falls back to LLaVA on failure
             "llava" → LLaVA only (no further fallback)
    """
    t0 = time.time()

    # ── Primary attempt ───────────────────────────────────────────────────────
    try:
        if backend == "groq":
            from file_processor.groq_client import caption_image_groq
            image.caption = caption_image_groq(image.base64_data)
        else:
            image.caption = _call_llava(image.base64_data)

        logger.info(
            f"  [{backend}] caption in {time.time()-t0:.1f}s "
            f"(page={image.page_number}, idx={image.image_index})"
        )
        return image

    except Exception as primary_err:
        logger.warning(
            f"  [{backend}] captioning failed "
            f"(page={image.page_number}, idx={image.image_index}): {primary_err}"
        )

    # ── Fallback to LLaVA (only when primary was Groq) ────────────────────────
    if backend == "groq":
        if _ollama_available():
            logger.info(
                f"  [groq→llava] Retrying with LLaVA fallback "
                f"(page={image.page_number}, idx={image.image_index}) …"
            )
            try:
                image.caption = _call_llava(image.base64_data)
                logger.info(
                    f"  [llava/fallback] caption in {time.time()-t0:.1f}s "
                    f"(page={image.page_number}, idx={image.image_index})"
                )
                return image
            except Exception as fallback_err:
                logger.warning(
                    f"  [llava/fallback] also failed "
                    f"(page={image.page_number}, idx={image.image_index}): {fallback_err}"
                )
        else:
            logger.warning(
                "  [llava/fallback] Ollama not available — caption set to None."
            )

    # Both backends failed (or LLaVA-only path failed)
    image.caption = None
    return image


# ── Batch captioner ───────────────────────────────────────────────────────────

def caption_all(
    images: list[ExtractedImage],
    *,
    backend: str = "groq",
    skip_if_unavailable: bool = True,
) -> list[ExtractedImage]:
    """
    Generates captions for all images.

    backend: "groq" (default, Llama-4-Scout via Groq API)
             "llava" (LLaVA via local Ollama)

    When backend="groq" and Groq is unavailable, automatically demotes the
    whole batch to LLaVA before starting rather than skipping all images.

    skip_if_unavailable=True → if BOTH backends are unavailable, log a warning
    and return images unchanged (captions remain None).
    """
    if not images:
        return images

    # ── Auto-demote to LLaVA when Groq is requested but unavailable ───────────
    if backend == "groq" and not _groq_available():
        if _ollama_available():
            logger.warning(
                "Groq not available — demoting entire batch to LLaVA fallback."
            )
            backend = "llava"
        elif skip_if_unavailable:
            logger.warning(
                "Neither Groq nor Ollama is available — image captions skipped."
            )
            return images

    # ── LLaVA-only path availability check ────────────────────────────────────
    if backend == "llava" and skip_if_unavailable and not _ollama_available():
        logger.warning(
            "Ollama not available — captions skipped. "
            "Run 'ollama serve' then 'ollama pull llava' to enable."
        )
        return images

    backend_label = "Groq (llama-4-scout)" if backend == "groq" else "LLaVA (Ollama)"
    workers = _CAPTION_WORKERS.get(backend, 2)
    logger.info(
        f"Captioning {len(images)} image(s) with {backend_label} "
        f"(max_workers={workers}) …"
    )

    if len(images) == 1 or workers == 1:
        for i, img in enumerate(images, 1):
            logger.info(f"  [{i}/{len(images)}] page={img.page_number}, idx={img.image_index}")
            caption_image(img, backend=backend)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_img = {
                pool.submit(caption_image, img, backend): (i, img)
                for i, img in enumerate(images, 1)
            }
            for future in as_completed(future_to_img):
                i, img = future_to_img[future]
                try:
                    future.result()
                    logger.debug(
                        f"  [{i}/{len(images)}] done — page={img.page_number}, "
                        f"idx={img.image_index}"
                    )
                except Exception as e:
                    logger.warning(
                        f"  [{i}/{len(images)}] unexpected error "
                        f"(page={img.page_number}, idx={img.image_index}): {e}"
                    )

    done = sum(1 for img in images if img.caption)
    logger.info(f"Captioning complete — {done}/{len(images)} successful")
    return images
