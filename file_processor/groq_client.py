"""
groq.py — vision captioning via Groq API (llama-4-scout multimodal).

Usage:
    from groq import caption_image_groq
    caption = caption_image_groq(img_base64)   # returns str
"""
from __future__ import annotations

import os

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

# ── Model ────────────────────────────────────────────────────────────────────
GROQ_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "groq_api_key")

# RAG-optimised: structured, 2-sentence output, hard 80-token cap.
# - System turn keeps the model on-task without repeating instructions.
# - User turn is a single, unambiguous request (no bullet lists = less padding).
_SYSTEM = (
    "You are a document-analysis assistant. "
    "Reply only with factual descriptions. "
    "No preamble, no opinion, no filler words."
)

_USER = (
    "Describe this document image for a search index. "
    "Use exactly 2 sentences: "
    "sentence 1 — visual type (chart/table/diagram/photo/screenshot/map/logo) and main topic; "
    "sentence 2 — key data values, labels, or text visible in the image. "
    "Max 70 words total."
)

# ── Client (lazy singleton) ───────────────────────────────────────────────────
_client = None

def _get_client():
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


# ── Public function ───────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def caption_image_groq(img_base64: str, mime: str = "image/png") -> str:
    """
    Sends *img_base64* to Groq's multimodal endpoint and returns a short caption.

    Retries up to 4 times with exponential backoff (2 s → 4 s → 8 s → 16 s)
    to handle Groq rate-limit (429) and transient network errors.
    Raises on final failure (caller sets caption=None).
    """
    client = _get_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{img_base64}"},
                    },
                    {"type": "text", "text": _USER},
                ],
            },
        ],
        max_tokens=120,          # 70 words ≈ 90 tokens — hard ceiling
        temperature=0.1,         # near-deterministic: more consistent captions
    )

    caption = response.choices[0].message.content.strip()
    used    = response.usage.total_tokens if response.usage else "?"
    logger.info(f"  Groq caption — {used} tokens used")
    return caption
