"""
models.py — résultats de l'extraction
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExtractedImage:
    page_number: int
    image_index: int
    base64_data: str
    mime_type: str = "image/png"
    caption: Optional[str] = None   


@dataclass
class ExtractedTable:
    page_number: int
    table_index: int
    markdown: str
    raw_rows: list[list[str]] = field(default_factory=list)
    # Nearest heading that preceded this table in document order.
    # Populated by parsers that can track heading context (DOCX, PDF).
    # Used by the chunker to label the table chunk instead of the
    # generic "[Table, page N]" fallback.
    section: Optional[str] = None


@dataclass
class ExtractionResult:
    """Tout ce qu'on a tiré d'un fichier."""
    source_file: str
    text_blocks: list[tuple[int, str]]   # (page_number, text)
    tables: list[ExtractedTable]
    images: list[ExtractedImage]
    doc_metadata: dict = field(default_factory=dict)  # document-level metadata (e.g. email headers)

    @property
    def stats(self) -> dict:
        return {
            "text_blocks": len(self.text_blocks),   # count of (page, text) tuples
            "tables":      len(self.tables),
            "images":      len(self.images),
            "images_captioned": sum(1 for i in self.images if i.caption),
        }