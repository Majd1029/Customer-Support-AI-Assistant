import os
from pathlib import Path

# ── PROJECT PATHS ─────────────────────────────────────
BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# ── POPPLER ───────────────────────────────────────────
POPPLER_PATH = r"C:\poppler-25.12.0\Library\bin"
HF_HUB_OFFLINE=1

from dotenv import load_dotenv
load_dotenv()


# ── OLLAMA (Moondream2) ───────────────────────────────
OLLAMA_HOST         = "http://localhost:11434"
OLLAMA_VISION_MODEL = "llava"

# ── GROQ ─────────────────────────────────────────────
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY")
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── FILE TYPES ────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    "pdf"  : [".pdf"],
    "image": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
    "txt"  : [".txt"],
    "docx" : [".docx"],
    "csv"  : [".csv"],
    "pptx" : [".pptx"],
    "eml"  : [".eml"],
}

# ── GEMINI PROMPT SETTINGS ────────────────────────────
SCANNED_PDF_PROMPT = """
Extract all content from this scanned page IN ORDER as it appears:
- Text → as plain text
- Tables → as markdown table with columns and rows
- Charts/Images → as detailed text description
Preserve the reading flow and structure exactly.
"""

IMAGE_PROMPT = """
Describe this image in detail:
- If it is a chart → describe type, values, trends
- If it is a figure → describe what it shows
- If it is a diagram → explain the components
Be precise and detailed.
"""

# ── DEBUG ─────────────────────────────────────────────
DEBUG = True

# ── CAMELOT SETTINGS ──────────────────────────────────
CAMELOT_FLAVOR = "stream"