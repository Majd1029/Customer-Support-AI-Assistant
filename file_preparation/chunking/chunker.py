"""
chunker.py — multi-strategy document chunking for hybrid RAG retrieval.

Chunking strategy
─────────────────
Text  : Paragraph-first with heading prefix injection.
        1. Split text on paragraph breaks (\\n\\n) — each paragraph is usually
           one complete thought.
        2. Track the nearest markdown/text heading above each paragraph and
           prepend it as context → retrieval accuracy improves significantly
           because the chunk carries its own section label.
        3. If a paragraph exceeds MAX_TOKENS (300) → split at sentence
           boundaries with NO overlap.
        4. If a paragraph is below MIN_PARA_TOKENS (30) → merge with the
           next paragraph before chunking.
        5. TXT / scanned-PDF blocks run a 6-stage semantic pipeline
           (sentence splitting → BGE-M3 embeddings → valley detection →
           token-budget grouping).

Tables: Each table is one chunk with a section/page prefix.

Images: Each image is one chunk whose text is the Ollama / Groq caption.

Strategy selection
──────────────────
- Contextual  : narrative docs (prose-heavy, low heading ratio)
- Hierarchical: structured docs with clear heading trees (DOCX, XLSX, PPTX)
- TXT semantic: plain-text and scanned-PDF pages (6-stage pipeline)
- EML         : header chunk + body chunks + per-attachment chunks

Public API
──────────
    from file_preparation.chunking import chunk_document

    data = chunk_document(result, doc_uid="a1b2c3d4")
    # → { "source_file": ..., "stats": {...}, "chunks": [...] }
    #
    # Every chunk carries metadata.doc_id — a UUID5 derived from the filename.
    # This is deterministic (same file → same doc_id across re-indexing runs)
    # and present on all file types: PDF, DOCX, PPTX, XLSX, TXT, EML, images.
    # EML chunks additionally carry metadata.email_id (identical value, kept for
    # backward-compatibility).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from file_processor.models import ExtractionResult

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_TOKENS       = 300   # Target chunk size — raised from 200 (customer-support default)
                         # to 300 for research/document RAG where queries are richer and
                         # longer context windows improve semantic completeness.
                         # DOCX uses 400 (its own override in chunk_document).
MIN_PARA_TOKENS  = 30    # Paragraphs smaller than this are merged with the next
MIN_CHUNK_TOKENS = 5     # Final gate — chunks below this are dropped entirely
                         # (lowered from 10 → 5 to preserve short PPTX title slides)

# ── Token counter ─────────────────────────────────────────────────────────────

# Module-level encoder cache.  Loaded once at import time so that:
#   (a) repeated calls never re-attempt the vocab download, and
#   (b) proxy/network failures only surface once (at startup) rather than
#       silently corrupting every single token-count estimate at runtime.
_tiktoken_enc = None

def _load_enc():
    """Load and cache the cl100k_base tiktoken encoder (idempotent)."""
    global _tiktoken_enc
    if _tiktoken_enc is None:
        try:
            import tiktoken
            _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
            logger.debug("  tiktoken cl100k_base encoder loaded")
        except Exception as _e:
            logger.warning(
                f"  tiktoken unavailable ({_e}); token counts will use "
                "len(text)//4 fallback — chunk sizes will be approximate"
            )
    return _tiktoken_enc

# Eagerly load at import time so the first actual call is free.
_load_enc()


# ── Language detector ─────────────────────────────────────────────────────────

# Minimum characters of content needed before attempting detection.
# Below this the detector is unreliable (e.g. a 3-word table cell).
_LANG_MIN_CHARS = 30

_langdetect_available: bool | None = None   # None = not yet checked

def _detect_language(text: str) -> str | None:
    """
    Returns an ISO 639-1 language code (e.g. "en", "fr", "ar") for `text`,
    or None if detection fails or the text is too short to be reliable.

    Uses langdetect with a fixed seed (0) for deterministic results.
    Failures are silently swallowed so a single undetectable chunk never
    aborts the whole document.
    """
    global _langdetect_available
    if _langdetect_available is False:
        return None

    # Sample the first 300 characters — enough for reliable detection,
    # cheap enough to run on every chunk.
    sample = text.strip()[:300]
    if len(sample) < _LANG_MIN_CHARS:
        return None

    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0   # deterministic output across runs
        _langdetect_available = True
        return detect(sample)
    except Exception:
        if _langdetect_available is None:
            logger.warning(
                "  langdetect unavailable — metadata.language will not be set. "
                "Install with: pip install langdetect"
            )
            _langdetect_available = False
        return None


def count_tokens(text: str) -> int:
    """
    Counts tokens using tiktoken (cl100k_base).
    Falls back to len(text) // 4 if tiktoken is unavailable or the vocab
    file could not be downloaded (e.g. sandboxed / offline environments).
    """
    enc = _load_enc()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


# ── Heading detector ──────────────────────────────────────────────────────────

# Pattern 1 — markdown headings:  "# Title", "## Section"
_MD_HEADING_FULL_RE = re.compile(r'^(#{1,6})\s+(.+)', re.M)

# Section divider marker emitted by the PPTX parser for section-layout slides.
# Format: "[SECTION] Section Name"
_SECTION_MARKER_RE = re.compile(r'^\[SECTION\]\s*(.+)', re.I)

# Pattern 2 — ALL CAPS short lines common in PDFs / DOCX exports
_CAPS_HEADING_RE = re.compile(r'^[A-Z][A-Z\s\-:]{3,60}$')

# Pattern 3 — mixed-case short standalone lines (most common in .txt / plain PDFs)
# Rules:
#   - Single line (no \n inside)
#   - 3–80 characters
#   - Starts with a letter (any language, including French accented / Arabic)
#   - Does NOT end with sentence-ending punctuation (. ? !) — those are sentences
#   - Does NOT start with a bullet marker (● • ○ - * 1.)
#   - Does NOT contain a colon mid-line (likely a label:value pair, not a heading)
_MIXED_HEADING_RE = re.compile(
    r'^'
    r'(?![●•○\-\*\d])'         # not a bullet or numbered list item
    r'[\w\u00C0-\u024F\u0600-\u06FF]'  # starts with letter (Latin ext. + Arabic)
    r'[^\n]{2,78}'              # 3–80 chars total, no internal newline
    r'(?<![.?!:,;])'            # does not end with punctuation
    r'$'
)


def _get_heading_level(text: str) -> tuple[int, str] | None:
    """
    Returns (level, heading_text) if `text` is a heading, else None.

    Levels mirror HTML heading semantics:
      1 → most important  (# H1, ALL-CAPS section titles)
      2 → sub-section     (## H2, mixed-case standalone lines)
      3+→ deeper levels   (### H3 and beyond, markdown only)

    Used by the hierarchical chunker to maintain a proper heading stack.
    """
    stripped = text.strip()
    if '\n' in stripped:
        return None

    # Markdown headings — level comes directly from the # count
    m = _MD_HEADING_FULL_RE.match(stripped)
    if m:
        return len(m.group(1)), m.group(2).strip()

    # ALL-CAPS lines → top-level section heading (level 1)
    if _CAPS_HEADING_RE.match(stripped) and len(stripped) <= 80:
        return 1, stripped

    # Guard A: CamelCase concatenation artifact (no spaces + internal uppercase)
    if ' ' not in stripped and any(c.isupper() for c in stripped[1:]):
        return None

    # Guard B: contains 3+ consecutive digits (IDs, postal codes, citations)
    if re.search(r'\d{3,}', stripped):
        return None

    # Mixed-case short standalone lines → subsection heading (level 2)
    if 3 <= len(stripped) <= 80 and _MIXED_HEADING_RE.match(stripped):
        return 2, stripped

    return None


def _extract_heading(text: str) -> str | None:
    """
    Returns the heading text if the paragraph looks like a heading, else None.

    Detection priority:
      1. Markdown heading  (# / ## / ###…)
      2. ALL-CAPS line     (PDF section titles)
      3. Mixed-case short standalone line with no terminal punctuation
         (most common in .txt, plain PDFs, DOCX exports)
    """
    stripped = text.strip()

    # Must be a single line to be a heading
    if '\n' in stripped:
        return None

    # 1. Markdown
    m = _MD_HEADING_FULL_RE.match(stripped)
    if m:
        return m.group(2).strip()

    # 2. ALL CAPS
    if _CAPS_HEADING_RE.match(stripped) and len(stripped) <= 80:
        return stripped

    # 3. Mixed-case short standalone line
    if 3 <= len(stripped) <= 80 and _MIXED_HEADING_RE.match(stripped):
        # Guard A — CamelCase / concatenated words artifact (no spaces + internal uppercase)
        # e.g. "UniversityofTexasatElPaso", "FederatedLearning", "RelatedWork"
        # Real single-word headings like "Introduction" have no internal uppercase after pos 0.
        if ' ' not in stripped and any(c.isupper() for c in stripped[1:]):
            return None

        # Guard B — contains 3+ consecutive digits (postal codes, IDs, citation refs)
        # e.g. "79968", "1v24250.8042:viXra"
        if re.search(r'\d{3,}', stripped):
            return None

        return stripped

    return None


# ── Sentence splitter ─────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """
    Splits text into sentences at .?!… boundaries.
    Protects common abbreviations to avoid false splits.
    """
    abbrevs  = r'\b(M|Mme|Mlle|Dr|Pr|Prof|Sr|Jr|vs|etc|Fig|fig|No|no|pp|vol|approx|cf|ibid|op)\.'
    protected = re.sub(abbrevs, r'\1<DOT>', text)
    parts     = re.split(r'(?<=[.?!…])\s+(?=[A-ZÀ-Üa-zà-ü0-9"«\(])', protected)
    return [p.replace('<DOT>', '.').strip() for p in parts if p.strip()]


def _split_by_sentences(text: str, max_tokens: int = MAX_TOKENS) -> list[str]:
    """
    Fallback: splits a long paragraph at sentence boundaries without overlap.
    No overlap because support answers are self-contained — overlap adds noise.
    """
    sentences = split_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks:  list[str] = []
    current: list[str] = []
    current_tok: int   = 0

    for sent in sentences:
        t = count_tokens(sent)
        if t >= max_tokens:
            if current:
                chunks.append(" ".join(current))
                current, current_tok = [], 0
            chunks.append(sent)
            continue
        if current_tok + t > max_tokens and current:
            chunks.append(" ".join(current))
            current, current_tok = [], 0
        current.append(sent)
        current_tok += t

    if current:
        chunks.append(" ".join(current))
    return chunks


# ── Text cleaning helpers ─────────────────────────────────────────────────────

_UNICODE_NOISE_RE = re.compile(
    '[\u00a0\ufeff\u200b\u200c\u200d\u00ad\u2028\u2029\u00b7'
    '\x00-\x08\x0b\x0c\x0e-\x1f\x7f]'
)
_MD_BOLD_RE    = re.compile(r'\*{1,3}(.+?)\*{1,3}')
_MD_HEADING_RE = re.compile(r'^#{1,6}\s+', re.M)
_MD_LINK_RE    = re.compile(r'\[([^\]]*)\]\([^)]*\)')
_MD_HR_RE      = re.compile(r'^[-*_]{3,}\s*$', re.M)
_MD_QUOTE_RE   = re.compile(r'^>\s+', re.M)
_OCR_NOISE_RE  = re.compile(r'[-=.|]{4,}')
_MULTI_WS_RE   = re.compile(r'[ \t]{2,}')
_MULTI_NL_RE   = re.compile(r'\n{3,}')
_ATTACH_TAG_RE = re.compile(r'^\[attachment:[^\]]+\]\n?', re.M)


def clean_chunk_text(text: str) -> str:
    """
    Strips unicode noise, markdown syntax, OCR border noise,
    and normalises whitespace.
    Headings are stripped from the body text (they become the prefix instead).
    """
    text = _UNICODE_NOISE_RE.sub(' ', text)
    text = _MD_HEADING_RE.sub('', text)
    text = _MD_BOLD_RE.sub(r'\1', text)
    text = _MD_LINK_RE.sub(r'\1', text)
    text = _MD_HR_RE.sub('', text)
    text = _MD_QUOTE_RE.sub('', text)
    text = _OCR_NOISE_RE.sub('', text)
    text = _MULTI_WS_RE.sub(' ', text)
    text = _MULTI_NL_RE.sub('\n\n', text)
    return text.strip()


def clean_metadata(meta: dict) -> dict:
    """Strips None, empty string, and empty list values."""
    return {k: v for k, v in meta.items() if v is not None and v != "" and v != []}


def slugify(text: str) -> str:
    """Converts a filename stem to a URL/DB-safe chunk_id component."""
    nfkd      = unicodedata.normalize('NFKD', text)
    ascii_str = nfkd.encode('ascii', 'ignore').decode('ascii')
    slug      = re.sub(r'[^a-zA-Z0-9]+', '_', ascii_str).strip('_').lower()
    return slug or "file"


def clean_table_markdown(markdown: str) -> str:
    """Removes empty rows and separator-only rows from table markdown."""
    lines   = markdown.splitlines()
    cleaned = []
    for line in lines:
        if not line.strip().startswith('|'):
            cleaned.append(line)
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if all(c == '' or re.match(r'^[-: ]+$', c) for c in cells):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned).strip()


def extract_attachment_source(text: str) -> tuple[str, str | None]:
    """Peels off an [attachment:fname] prefix. Returns (clean_text, fname|None)."""
    match = _ATTACH_TAG_RE.match(text)
    if match:
        tag   = match.group(0)
        fname = tag.strip()[len('[attachment:'):-1]
        return text[len(tag):].strip(), fname
    return text, None


# ── Paragraph-aware text chunker ──────────────────────────────────────────────

def _chunk_block(
    text: str,
    heading: str | None,
    max_tokens: int = MAX_TOKENS,
) -> list[str]:
    """
    Chunks a single text block (paragraph or merged paragraphs).
    Prepends the nearest heading when present.
    If the block fits in max_tokens → one chunk.
    Otherwise → sentence-boundary split, heading prepended to each sub-chunk.
    """
    prefix    = f"[{heading}]\n" if heading else ""
    body      = clean_chunk_text(text)
    if not body:
        return []

    full_text = prefix + body
    if count_tokens(full_text) <= max_tokens:
        return [full_text]

    # Too long → split at sentence boundaries, repeat heading prefix on each
    sub_chunks = _split_by_sentences(body, max_tokens=max_tokens - count_tokens(prefix))
    return [prefix + s for s in sub_chunks if s.strip()]


def _chunk_block_with_prefix(
    text: str,
    prefix: str,
    max_tokens: int = MAX_TOKENS,
) -> list[str]:
    """
    Like `_chunk_block` but accepts a pre-built prefix string (e.g. a full
    hierarchical path "[Chapter 1 > Background]\n").  Used by the hierarchical
    chunker so the prefix can span more than one heading level.
    """
    body = clean_chunk_text(text)
    if not body:
        return []
    full = prefix + body
    if count_tokens(full) <= max_tokens:
        return [full]
    prefix_tok  = count_tokens(prefix)
    sub_chunks  = _split_by_sentences(body, max_tokens=max_tokens - prefix_tok)
    return [prefix + s for s in sub_chunks if s.strip()]


# ── Document type classifier ──────────────────────────────────────────────────

# File types that always have a clear structure (slides, spreadsheets)
_ALWAYS_STRUCTURED = {".pptx", ".xlsx", ".csv"}
# File types that are always narrative / prose
_ALWAYS_NARRATIVE  = {".eml"}

def classify_document_type(
    text_blocks: list[tuple[int, str]],
    tables: list,
    file_type: str,
) -> tuple[str, str, str]:
    """
    Analyses the document and returns (doc_type, strategy, reason).

    doc_type : "narrative" | "structured" | "mixed"
    strategy : "contextual"   — simple paragraph chunking with nearest-heading prefix
               "hierarchical" — full heading-path prefix, maintains heading stack
    reason   : human-readable explanation (logged at INFO level)

    Selection logic
    ───────────────
    File-type shortcuts
      • PPTX / XLSX / CSV  → structured  → hierarchical  (always slide/table based)
      • EML                → narrative   → contextual    (email prose)

    Content-based analysis (PDF, DOCX, TXT, images)
      Metrics computed:
        heading_ratio      = heading blocks / total blocks
        md_heading_count   = blocks that start with #/##/###
        avg_block_tokens   = mean token count per text block
        table_density      = tables / (total_tokens / MAX_TOKENS)

      Rules (in priority order):
        1. md_heading_count ≥ 2  OR  heading_ratio > 0.20
               → structured, hierarchical
        2. avg_block_tokens > 60  AND  heading_ratio < 0.08
               → narrative, contextual
        3. heading_ratio > 0.12  OR  table_density > 0.15
               → mixed, hierarchical
        4. fallback
               → mixed, contextual

    Size override: total_tokens < 150 → always contextual (fits in ≤1 chunk)
    """
    ext = file_type.lower() if file_type else ""

    # ── File-type shortcuts ───────────────────────────────────────────────────
    if ext in _ALWAYS_STRUCTURED:
        return "structured", "hierarchical", f"file type '{ext}' is always structured"
    if ext in _ALWAYS_NARRATIVE:
        return "narrative", "contextual", f"file type '{ext}' is always narrative"

    # ── Content analysis ──────────────────────────────────────────────────────
    total_blocks = len(text_blocks)
    if total_blocks == 0:
        reason = "no text blocks" + (" — has tables" if tables else "")
        return ("structured" if tables else "narrative"), "contextual", reason

    heading_count    = 0
    md_heading_count = 0
    total_tokens_sum = 0
    list_item_count  = 0
    question_count   = 0   # blocks that look like FAQ questions

    for _, raw in text_blocks:
        clean, _ = extract_attachment_source(raw)
        # Skip section markers — they're structural, not content
        if _SECTION_MARKER_RE.match(clean.strip()):
            continue
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', clean) if p.strip()]
        for para in paragraphs:
            tok = count_tokens(para)
            total_tokens_sum += tok
            lvl_info = _get_heading_level(para)
            if lvl_info:
                heading_count += 1
                if _MD_HEADING_FULL_RE.match(para.strip()):
                    md_heading_count += 1
            elif re.match(r'^[●•○\-\*]|\d+[.)]\s', para):
                list_item_count += 1
            # FAQ/Q&A signal: block ends with "?" or starts with "Q:" / "Q."
            if para.endswith("?") or re.match(r'^Q[.:\s]', para, re.IGNORECASE):
                question_count += 1

    total_para_blocks = total_blocks  # proxy
    heading_ratio    = heading_count    / max(1, total_para_blocks)
    avg_block_tokens = total_tokens_sum / max(1, total_para_blocks)
    table_density    = len(tables)      / max(1, total_tokens_sum / MAX_TOKENS)
    question_ratio   = question_count   / max(1, total_para_blocks)

    # ── Size override ─────────────────────────────────────────────────────────
    if total_tokens_sum < 150:
        return "narrative", "contextual", (
            f"tiny document ({total_tokens_sum} tokens) — contextual regardless of structure"
        )

    # ── FAQ/Q&A override — takes priority over heading-based rules ────────────
    # FAQ documents often have short question blocks (low avg_block_tokens) which
    # would otherwise look "structured" and trigger hierarchical chunking.
    # Hierarchical would treat each question as a heading prefix, producing tiny
    # one-sentence chunks.  Contextual is better: it merges the question with
    # its answer into a single retrievable unit.
    if question_ratio > 0.15:
        return "narrative", "contextual", (
            f"FAQ/Q&A pattern — question_ratio={question_ratio:.2f} "
            f"({question_count}/{total_para_blocks} question blocks)"
        )

    # ── Classification rules ──────────────────────────────────────────────────
    if md_heading_count >= 2 or heading_ratio > 0.20:
        return "structured", "hierarchical", (
            f"md_headings={md_heading_count}, heading_ratio={heading_ratio:.2f}"
        )

    if avg_block_tokens > 60 and heading_ratio < 0.08:
        return "narrative", "contextual", (
            f"avg_block_tokens={avg_block_tokens:.0f}, heading_ratio={heading_ratio:.2f}"
        )

    if heading_ratio > 0.12 or table_density > 0.15:
        return "mixed", "hierarchical", (
            f"heading_ratio={heading_ratio:.2f}, table_density={table_density:.2f}"
        )

    return "mixed", "contextual", (
        f"avg_block_tokens={avg_block_tokens:.0f}, heading_ratio={heading_ratio:.2f} "
        f"(no dominant signal)"
    )


# ── Simple contextual chunker (narrative / small docs) ────────────────────────

def chunk_text_blocks(
    text_blocks: list[tuple[int, str]],
    max_tokens: int = MAX_TOKENS,
) -> list[tuple[int, str, str | None, str | None, str | None]]:
    """
    Converts raw (page, text) blocks into
    (page, chunk_text, att_source, slide_title, section) tuples using
    paragraph-first splitting with heading prefix injection.

    slide_title is only set for PPTX files (where titles are emitted as
    '# heading' blocks by the parser). For all other file types it is None.

    section is set when a '[SECTION] Name' marker block is encountered
    (emitted by _parse_pptx() for section-divider slides). It persists
    across all subsequent slides until a new section marker is found.

    Returns list of (page_number, chunk_text, attachment_source_or_None,
                     slide_title_or_None, section_or_None).
    """
    result_chunks: list[tuple[int, str, str | None, str | None, str | None]] = []

    # Track the most recent heading seen across blocks
    current_heading:    str | None = None

    # Track current page — heading resets when page changes (critical for PPTX
    # where each slide is its own topic and headings must not bleed across slides)
    current_page:       int | None = None

    # Track the slide title (set when a markdown "# heading" is detected,
    # which is emitted by _parse_pptx() for title placeholder shapes)
    current_slide_title: str | None = None

    # Track the current section (set when a "[SECTION] Name" marker block is
    # detected, emitted by _parse_pptx() for section-divider slides).
    # Unlike current_heading, this persists across slide boundaries — it
    # applies to all chunks until the next section divider is found.
    current_section: str | None = None

    # Buffer for merging short paragraphs
    pending_pg:   int | None = None
    pending_text: str        = ""
    pending_tok:  int        = 0
    pending_att:  str | None = None

    def _flush_pending() -> None:
        nonlocal pending_pg, pending_text, pending_tok, pending_att
        if not pending_text.strip():
            pending_pg, pending_text, pending_tok, pending_att = None, "", 0, None
            return
        for chunk in _chunk_block(pending_text, current_heading, max_tokens=max_tokens):
            if count_tokens(chunk) >= MIN_CHUNK_TOKENS:
                result_chunks.append((
                    pending_pg or 0, chunk, pending_att,
                    current_slide_title, current_section,
                ))
        pending_pg, pending_text, pending_tok, pending_att = None, "", 0, None

    for pg, raw_text in text_blocks:
        clean_text, att_src = extract_attachment_source(raw_text)

        # ── Section marker — emitted by parser for section-divider slides ─────
        # Format: "[SECTION] Section Name"
        # These blocks are never emitted as chunks; they update current_section
        # which is then injected as metadata on all subsequent chunks.
        sec_match = _SECTION_MARKER_RE.match(clean_text.strip())
        if sec_match:
            _flush_pending()
            current_section = sec_match.group(1).strip()
            logger.debug(f"  chunker: section → '{current_section}' (slide {pg + 1})")
            continue

        # ── Page boundary — reset heading state ───────────────────────────────
        # For PPTX each page = one slide. A heading from slide N must not
        # propagate to slide N+1 since slides are independent topics.
        # For PDFs/DOCX this still makes sense: section headings rarely span
        # more than one page boundary in practice.
        if pg != current_page:
            _flush_pending()
            current_heading = None
            current_page    = pg
            # Note: current_slide_title is NOT reset here — it persists until
            # a new "# heading" is seen (i.e. the next slide's title block).
            # Note: current_section is NOT reset here — sections span multiple
            # slides and persist until a new [SECTION] marker is encountered.

        # Split each block into paragraphs on double newlines
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', clean_text) if p.strip()]
        if not paragraphs:
            continue

        for para in paragraphs:
            # Check if this paragraph is itself a heading
            heading = _extract_heading(para)
            if heading:
                # Flush whatever was pending before updating the heading
                _flush_pending()
                current_heading = heading

                # If this came from a markdown "# heading" block (emitted by
                # the PPTX parser for title placeholders), also record it as
                # the slide title for metadata injection.
                if _MD_HEADING_FULL_RE.match(para):
                    current_slide_title = heading

                continue   # Headings are not emitted as chunks themselves

            tok = count_tokens(para)

            if tok < MIN_PARA_TOKENS:
                # Too short — merge with pending
                if pending_text:
                    pending_text += " " + para
                    pending_tok  += tok
                else:
                    pending_pg   = pg
                    pending_text = para
                    pending_tok  = tok
                    pending_att  = att_src

                # If merged result is now large enough, flush it
                if pending_tok >= MIN_PARA_TOKENS:
                    _flush_pending()
            else:
                # Normal paragraph — flush any pending short para first
                if pending_text:
                    # Merge if combined still fits in one chunk
                    combined_tok = pending_tok + tok
                    if combined_tok <= max_tokens:
                        pending_text += " " + para
                        pending_tok   = combined_tok
                        _flush_pending()
                    else:
                        _flush_pending()
                        pending_pg   = pg
                        pending_text = para
                        pending_tok  = tok
                        pending_att  = att_src
                        _flush_pending()
                else:
                    pending_pg   = pg
                    pending_text = para
                    pending_tok  = tok
                    pending_att  = att_src
                    _flush_pending()

    # Flush any remaining content
    _flush_pending()

    return result_chunks  # list of (page, chunk_text, att_src, slide_title, section)


# ══════════════════════════════════════════════════════════════════════════════
# TXT SEMANTIC CHUNKING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
#
#  Automatically selected when file_type == ".txt" (or plain-text variants).
#  Mirrors the logic from the reference implementation but:
#    • Strips the dead-code embedding pass in "late chunking" (embeddings were
#      computed but never stored — wasted GPU/CPU for zero benefit).
#    • Uses our existing count_tokens() / tiktoken instead of a second tokenizer.
#    • Sentence-embedding model (sentence-transformers) is optional; falls back
#      to token-budget-only sentence grouping when not installed.
#    • Output is a plain list[str] that chunk_document() wraps into our standard
#      5-tuple format (page, text, att_src, slide_title, section).
#
#  Pipeline overview
#  ─────────────────
#    STEP 1 — detect TXT doc type
#             bullet_ratio and heading_ratio across all lines
#             → "narrative"  : bullet < 5%  and heading < 2%
#             → "mixed"      : bullet 5–20% or heading 2–5%
#             → "structured" : bullet > 20% or heading > 5%
#
#    STEP 2 — estimate total tokens (tiktoken cl100k_base)
#
#    STEP 3 — route to strategy:
#
#      narrative                → SENTENCE GROUPING
#        • split into sentences
#        • group by token budget (MIN=80, MAX=512)
#        • merge leftover groups < MIN tokens into the previous chunk
#
#      structured / mixed       → HIERARCHICAL + SLIDING WINDOW
#        • split on ___ / === / # separators (structured)
#          or on double-newline before capital/digit/# (mixed)
#        • per section: split into sentences
#          → embed (sentence-transformers) if available
#          → cosine similarity between adjacent sentences → similarity curve
#          → detect valley boundaries (local minima below mean − 0.5 × std)
#          → build chunks respecting MIN/MAX token budget
#        • fallback (no embedding model): token-budget grouping only
#
# ══════════════════════════════════════════════════════════════════════════════

# Token budget constants for TXT — wider than the generic MAX_TOKENS (200)
# because plain-text documents are typically retrieved as standalone passages.
_TXT_MIN_CHUNK_TOKENS = 80
_TXT_MAX_CHUNK_TOKENS = 512
_TXT_VALLEY_STD_FACTOR = 0.5   # boundary threshold = mean − 0.5 × std


# ── Step 1: doc-type detection ────────────────────────────────────────────────

def _detect_txt_doc_type(text: str) -> str:
    """
    Classifies a plain-text document as 'narrative', 'mixed', or 'structured'
    by measuring bullet and heading line density.
    """
    lines = text.splitlines()
    total = max(len(lines), 1)

    bullet_lines = sum(
        1 for ln in lines if re.match(r"^\s*[\*\-•]\s+", ln)
    )
    heading_lines = sum(
        1 for ln in lines
        if re.match(r"^\s*#{1,6}\s+", ln)
        or re.match(r"^\d+[\)\.]\s+[A-Z]", ln)
        or re.match(r"^[A-Z][A-Z\s\-]{3,}$", ln.rstrip())
        or re.match(r"^_{3,}$",  ln.strip())
        or re.match(r"^={3,}$",  ln.strip())
        or re.match(r"^-{3,}$",  ln.strip())
    )

    bullet_ratio  = bullet_lines  / total
    heading_ratio = heading_lines / total

    if bullet_ratio > 0.20 or heading_ratio > 0.05:
        return "structured"
    elif bullet_ratio > 0.05 or heading_ratio > 0.02:
        return "mixed"
    return "narrative"


# ── Step 3a: sentence splitter ────────────────────────────────────────────────

def _split_txt_sentences(text: str) -> list[str]:
    """
    Splits plain text into sentences, handling:
      • structural separators (___  ===  ---)       → hard breaks
      • ALL-CAPS headings                            → hard breaks around them
      • bullet / numbered list items                → hard breaks
      • common abbreviations (Dr. Mr. etc.)         → protected dots
      • standard .?! sentence boundaries
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Structural separators → hard break marker
    text = re.sub(r"_{3,}|={3,}|-{3,}", " <BREAK> ", text)
    text = re.sub(r"\n{2,}", " <BREAK> ", text)

    # ALL-CAPS standalone lines → wrap with breaks
    text = re.sub(
        r"(?m)^([A-Z][A-Z\s\-]{3,})\s*$",
        r" <BREAK> \1 <BREAK> ",
        text,
    )

    # List items → hard break
    text = re.sub(r"\n\*\s+",    " <BREAK> ", text)
    text = re.sub(r"\n\d+\.\s+", " <BREAK> ", text)
    text = re.sub(r"\n\-\s+",    " <BREAK> ", text)

    # Protect ordinal numbers (1. 2. etc.) so they don't split sentences
    text = re.sub(r"(\d+)\.\s", r"\1<DOT> ", text)

    # Protect common abbreviations
    for abbr in ["Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Jr",
                 "etc", "e.g", "i.e", "M", "Mme", "Mlle"]:
        text = text.replace(f"{abbr}.", f"{abbr}<DOT>")

    # Split on sentence-ending punctuation
    raw = re.split(r"(?<=[.!?])\s+", text)

    sentences: list[str] = []
    for piece in raw:
        parts = piece.split("<BREAK>")
        sentences.extend(
            p.replace("<DOT>", ".").strip()
            for p in parts if p.strip()
        )
    return sentences


# ── Step 3b: section splitter ─────────────────────────────────────────────────

def _split_txt_sections(text: str, doc_type: str) -> list[str]:
    """
    Splits a TXT document into top-level sections before sentence-level
    processing.  Only used for structured and mixed documents.
    """
    if doc_type == "narrative":
        return [text]

    if doc_type == "structured":
        parts = re.split(r"(?m)_{3,}|={3,}|^#{1,3}\s", text)
    else:  # mixed
        parts = re.split(r"\n{2,}(?=[A-Z\d#])", text)

    return [s.strip() for s in parts if s.strip()]


# ── Step 3c: sentence grouping by token budget ────────────────────────────────

def _group_sentences_by_budget(
    sentences: list[str],
    boundaries: set[int] | None = None,
) -> list[str]:
    """
    Groups sentences into text chunks respecting _TXT_MIN/_TXT_MAX token budget.
    If `boundaries` is provided (set of sentence indices), forces a split there
    whenever the current group is already at or above _TXT_MIN_CHUNK_TOKENS.
    """
    groups:        list[str]  = []
    current:       list[str]  = []
    current_tok:   int             = 0
    boundaries = boundaries or set()

    for i, sent in enumerate(sentences):
        tok = count_tokens(sent)

        force_split = (
            i in boundaries
            and current_tok >= _TXT_MIN_CHUNK_TOKENS
        )
        size_split = (
            current_tok + tok > _TXT_MAX_CHUNK_TOKENS
            and current_tok >= _TXT_MIN_CHUNK_TOKENS
        )

        if (force_split or size_split) and current:
            groups.append(" ".join(current))
            current, current_tok = [], 0

        current.append(sent)
        current_tok += tok

    if current:
        leftover = " ".join(current)
        # Merge tiny leftovers into the previous group
        if groups and count_tokens(leftover) < _TXT_MIN_CHUNK_TOKENS:
            groups[-1] += " " + leftover
        else:
            groups.append(leftover)

    return groups


# ── Step 3d: optional semantic embedding + valley detection ───────────────────

_txt_embed_model = None   # lazy singleton

def _get_txt_embed_model():
    """
    Returns a sentence-transformers model for TXT semantic boundary detection.
    Uses a small, fast model (all-MiniLM-L6-v2, ~22 MB).
    Returns None if sentence-transformers is not installed.
    """
    global _txt_embed_model
    if _txt_embed_model is not None:
        return _txt_embed_model if _txt_embed_model else None
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("  [txt-chunker] Loading sentence-transformers model ...")
        try:
            # Prefer the local cache — avoids HuggingFace Hub timeout on
            # every load when the model is already downloaded.
            _txt_embed_model = SentenceTransformer(
                "all-MiniLM-L6-v2", local_files_only=True
            )
            logger.info("  [txt-chunker] Model ready (loaded from local cache).")
        except Exception:
            # Cache miss — download from HuggingFace (first run only).
            logger.info("  [txt-chunker] Cache miss — downloading model from HuggingFace ...")
            _txt_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("  [txt-chunker] Model ready (downloaded).")
    except Exception:
        _txt_embed_model = False   # mark as unavailable
    return _txt_embed_model if _txt_embed_model else None


def _embed_sentences_txt(sentences: list[str]):
    """
    Embeds a list of sentences.  Returns an (N, D) float32 numpy array,
    or None if the embedding model is unavailable.
    """
    model = _get_txt_embed_model()
    if model is None:
        return None
    try:
        import numpy as np
        vecs = model.encode(sentences, batch_size=32, show_progress_bar=False,
                            normalize_embeddings=True)
        return np.array(vecs, dtype="float32")
    except Exception as e:
        logger.warning(f"  [txt-chunker] embedding failed: {e}")
        return None


def _detect_valley_boundaries(embeddings) -> set[int]:
    """
    Finds topic-shift boundaries: indices i where the cosine similarity
    between sentence[i-1] and sentence[i] is a local minimum below the
    adaptive threshold  mean − VALLEY_STD_FACTOR × std.

    Returns a set of sentence indices at which a new chunk should start.
    Returns an empty set if embeddings is None (model unavailable).
    """
    if embeddings is None or len(embeddings) < 3:
        return set()

    import numpy as np

    sims = np.array([
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ])
    threshold = float(sims.mean() - _TXT_VALLEY_STD_FACTOR * sims.std())
    logger.debug(
        f"  [txt-chunker] sim mean={sims.mean():.3f} "
        f"std={sims.std():.3f} threshold={threshold:.3f}"
    )

    boundaries: set[int] = set()
    for i in range(1, len(sims) - 1):
        if sims[i] < sims[i - 1] and sims[i] < sims[i + 1] and sims[i] < threshold:
            boundaries.add(i + 1)   # boundary before sentence i+1
    return boundaries


# ── Main TXT dispatcher ───────────────────────────────────────────────────────

def _merge_small_txt_chunks(chunks: list[str]) -> list[str]:
    """
    Post-processing pass over the final chunk list.

    Any chunk whose token count is below _TXT_MIN_CHUNK_TOKENS is merged into
    its neighbour — preferring the previous chunk, falling back to the next —
    as long as the combined size stays within _TXT_MAX_CHUNK_TOKENS.

    Runs iteratively until no more merges are possible, so chains of small
    chunks (e.g. two 40-token sections in a row) are fully resolved.
    """
    if len(chunks) <= 1:
        return chunks

    changed = True
    merged  = list(chunks)

    while changed:
        changed = False
        output: list[str] = []
        i = 0

        while i < len(merged):
            tok = count_tokens(merged[i])

            if tok < _TXT_MIN_CHUNK_TOKENS:
                # ── Try merging with the previous chunk ───────────────────────
                if output:
                    combined     = output[-1] + " " + merged[i]
                    combined_tok = count_tokens(combined)
                    if combined_tok <= _TXT_MAX_CHUNK_TOKENS:
                        output[-1] = combined
                        changed = True
                        i += 1
                        continue

                # ── Try merging with the next chunk ───────────────────────────
                if i + 1 < len(merged):
                    combined     = merged[i] + " " + merged[i + 1]
                    combined_tok = count_tokens(combined)
                    if combined_tok <= _TXT_MAX_CHUNK_TOKENS:
                        output.append(combined)
                        changed = True
                        i += 2
                        continue

            # No merge possible (or already at budget) — keep as-is
            output.append(merged[i])
            i += 1

        merged = output

    return merged


def chunk_txt_content(text: str) -> list[str]:
    """
    Full TXT chunking pipeline.  Returns a list of chunk strings ready to be
    wrapped into chunk dicts by chunk_document().

    Strategy selection:
      narrative  → sentence grouping with token budget
                   (clean and fast; no structural noise to split on)
      structured → section split → per-section semantic sliding window
      mixed      → same as structured with looser section boundaries
    """
    doc_type = _detect_txt_doc_type(text)
    total_tok = count_tokens(text)
    logger.info(
        f"  [txt-chunker] doc_type={doc_type!r}, tokens={total_tok}"
    )

    # ── Narrative: sentence grouping only ────────────────────────────────────
    if doc_type == "narrative":
        logger.info("  [txt-chunker] strategy=SENTENCE_GROUPING")
        sentences = _split_txt_sentences(text)
        if len(sentences) <= 1:
            return [text.strip()] if text.strip() else []
        return _merge_small_txt_chunks(_group_sentences_by_budget(sentences))

    # ── Structured / Mixed: hierarchical + sliding window ────────────────────
    logger.info(
        f"  [txt-chunker] strategy=HIERARCHICAL+SLIDING_WINDOW ({doc_type})"
    )
    sections = _split_txt_sections(text, doc_type)
    logger.debug(f"  [txt-chunker] {len(sections)} section(s)")

    result: list[str] = []

    for section in sections:
        sentences = _split_txt_sentences(section)

        if len(sentences) <= 1:
            if section.strip():
                result.append(section.strip())
            continue

        # Try semantic boundary detection
        embeddings  = _embed_sentences_txt(sentences)
        boundaries  = _detect_valley_boundaries(embeddings)
        sub_chunks  = _group_sentences_by_budget(sentences, boundaries)

        label = section.strip().splitlines()[0][:40]
        logger.debug(
            f"  [txt-chunker] section '{label}' → "
            f"{len(sub_chunks)} chunk(s)"
            + (" (semantic)" if embeddings is not None else " (token-budget fallback)")
        )
        result.extend(sub_chunks)

    return _merge_small_txt_chunks(result)


# ── Hierarchical chunker (structured / mixed docs) ────────────────────────────

def chunk_text_blocks_hierarchical(
    text_blocks: list[tuple[int, str]],
    file_type: str = "",
    max_tokens: int = MAX_TOKENS,
) -> list[tuple[int, str, str | None, str | None, str | None]]:
    """
    Hierarchical chunking strategy for structured and mixed documents.

    Maintains a heading *stack* instead of a single nearest-heading.
    Each chunk prefix is the full section path, e.g.:

        [Introduction > Background > Related Work]
        The transformer architecture was first proposed …

    This gives the retriever far richer context than a flat nearest-heading
    prefix, especially for long documents with nested sections.

    Key differences from the simple contextual chunker
    ───────────────────────────────────────────────────
    • Heading stack  — levels H1→H2→H3 are tracked; a new H2 pops the previous
      H2/H3 but keeps H1, so the path always reflects real document nesting.
    • section metadata — the root (shallowest) heading in the stack at chunk
      time is recorded as `section` for metadata filtering.
    • PPTX slide isolation — when file_type == ".pptx", the heading stack is
      reset on each page boundary (slides are independent topics).
    • [SECTION] markers — parsed the same way as in the simple chunker; they
      inject a virtual level-0 root that replaces everything in the stack.

    Returns the same 5-tuple format as chunk_text_blocks():
        (page, chunk_text, att_source, slide_title, section)
    """
    result_chunks: list[tuple[int, str, str | None, str | None, str | None]] = []

    # heading_stack: list of (level, heading_text)
    # Invariant: levels are strictly increasing top→bottom.
    heading_stack:       list[tuple[int, str]] = []
    current_section:     str | None            = None   # root of heading stack
    current_slide_title: str | None            = None   # PPTX only (level-1 heading)
    current_page:        int | None            = None
    is_pptx             = file_type.lower() == ".pptx"

    pending_pg:   int | None = None
    pending_text: str        = ""
    pending_tok:  int        = 0
    pending_att:  str | None = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_prefix() -> str:
        if not heading_stack:
            return ""
        path = " > ".join(h for _, h in heading_stack)
        return f"[{path}]\n"

    def _push_heading(level: int, heading_text: str) -> None:
        nonlocal current_section, current_slide_title
        # Pop any existing heading at the same or deeper level
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, heading_text))
        # The root of the stack (first element) is the section
        current_section = heading_stack[0][1]
        # Track slide title for PPTX (level-1 markdown heading)
        if level == 1:
            current_slide_title = heading_text

    def _flush_pending() -> None:
        nonlocal pending_pg, pending_text, pending_tok, pending_att
        if not pending_text.strip():
            pending_pg, pending_text, pending_tok, pending_att = None, "", 0, None
            return
        prefix = _build_prefix()
        for chunk in _chunk_block_with_prefix(pending_text, prefix, max_tokens=max_tokens):
            if count_tokens(chunk) >= MIN_CHUNK_TOKENS:
                result_chunks.append((
                    pending_pg or 0,
                    chunk,
                    pending_att,
                    current_slide_title,
                    current_section,
                ))
        pending_pg, pending_text, pending_tok, pending_att = None, "", 0, None

    # ── Main loop ─────────────────────────────────────────────────────────────

    for pg, raw_text in text_blocks:
        clean_text, att_src = extract_attachment_source(raw_text)

        # PPTX [SECTION] marker — acts as a virtual level-0 root heading
        sec_match = _SECTION_MARKER_RE.match(clean_text.strip())
        if sec_match:
            _flush_pending()
            sec_name = sec_match.group(1).strip()
            heading_stack.clear()
            heading_stack.append((0, sec_name))
            current_section = sec_name
            logger.debug(f"  chunker[hier]: section → '{sec_name}' (slide {pg + 1})")
            continue

        # Page boundary
        if pg != current_page:
            _flush_pending()
            current_page = pg
            if is_pptx:
                # Each slide is an independent topic — reset the heading stack.
                # current_section (from [SECTION] marker) is intentionally kept:
                # it should persist across slides within the same PPTX section.
                heading_stack.clear()
                current_slide_title = None

        # Split into paragraphs
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', clean_text) if p.strip()]
        if not paragraphs:
            continue

        for para in paragraphs:
            lvl_info = _get_heading_level(para)
            if lvl_info:
                level, heading_text = lvl_info
                _flush_pending()
                _push_heading(level, heading_text)
                continue

            tok = count_tokens(para)

            if tok < MIN_PARA_TOKENS:
                # Short paragraph — merge with pending
                if pending_text:
                    pending_text += " " + para
                    pending_tok  += tok
                else:
                    pending_pg   = pg
                    pending_text = para
                    pending_tok  = tok
                    pending_att  = att_src
                if pending_tok >= MIN_PARA_TOKENS:
                    _flush_pending()
            else:
                if pending_text:
                    combined_tok = pending_tok + tok
                    if combined_tok <= max_tokens:
                        pending_text += " " + para
                        pending_tok   = combined_tok
                        _flush_pending()
                    else:
                        _flush_pending()
                        pending_pg   = pg
                        pending_text = para
                        pending_tok  = tok
                        pending_att  = att_src
                        _flush_pending()
                else:
                    pending_pg   = pg
                    pending_text = para
                    pending_tok  = tok
                    pending_att  = att_src
                    _flush_pending()

    _flush_pending()
    return result_chunks


# ── Scanned-PDF table / figure extraction ────────────────────────────────────

# Matches a separator row like |---|---| or |:---:|---|
_MD_TABLE_SEP_RE = re.compile(r'^\|[\s:]*\-[\s\-:|]*\|')

# Matches one or more consecutive lines that start with '|'
_MD_TABLE_BLOCK_RE = re.compile(r'(?m)(?:^\|.+\n?)+')

# Matches [Figure: ...] markers inserted by the OCR prompt
_FIGURE_RE = re.compile(r'\[Figure:\s*([^\]]+)\]')


def _clean_table_block(raw: str) -> tuple[str, str]:
    """
    Removes trailing prose that the OCR joined onto the last table row.

    Example OCR artefact:
        | Point E | 93 | 43 | — | Next, we see a table with special...

    Returns (clean_table_text, rescued_prose) where rescued_prose is any
    text found after the last '|' on any row.  The rescued text is returned
    so the caller can prepend it to the following text segment.
    """
    lines = raw.splitlines()
    clean_lines: list[str] = []
    rescued: list[str]     = []

    for line in lines:
        stripped = line.rstrip()
        if not stripped.startswith("|"):
            clean_lines.append(line)
            continue
        last_pipe = stripped.rfind("|")
        trailing  = stripped[last_pipe + 1:].strip()
        if trailing:
            clean_lines.append(stripped[: last_pipe + 1])
            rescued.append(trailing)
        else:
            clean_lines.append(line)

    return "\n".join(clean_lines).strip(), " ".join(rescued).strip()


def _extract_md_tables_from_text(text: str) -> list[tuple[str, str]]:
    """
    Split a text string that may contain embedded markdown tables into
    alternating ("text"|"table", content) segments.

    Used on scanned-PDF OCR chunks: the vision model returns all content as
    plain text, including tables formatted as markdown.  Detecting and
    re-typing those blocks as "table" produces proper table chunks downstream.

    A table block is a run of one or more consecutive lines starting with '|'.
    Any prose text between (or before/after) table blocks becomes a "text" segment.
    """
    segments: list[tuple[str, str]] = []
    last_end = 0

    for m in _MD_TABLE_BLOCK_RE.finditer(text):
        before = text[last_end : m.start()].strip()
        if before:
            segments.append(("text", before))

        table_raw            = m.group(0).strip()
        clean_tbl, rescued   = _clean_table_block(table_raw)
        if clean_tbl:
            segments.append(("table", clean_tbl))
        # Any prose rescued from inside the table block becomes a text segment
        if rescued:
            segments.append(("text", rescued))

        last_end = m.end()

    after = text[last_end:].strip()
    if after:
        segments.append(("text", after))

    return segments or [("text", text)]


def _parse_md_table_rows(md_table: str) -> list[list[str]]:
    """
    Parse a markdown table string into a list of row lists.

    Skips separator lines (|---|---|).  For lines that have trailing prose
    after the last '|' (an OCR artefact), only the table portion is kept.
    """
    rows: list[list[str]] = []
    for line in md_table.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        # Trim trailing prose after the last pipe (OCR join artefact)
        last_pipe = stripped.rfind("|")
        table_part = stripped[: last_pipe + 1] if last_pipe > 0 else stripped
        # Skip separator rows
        if _MD_TABLE_SEP_RE.match(table_part):
            continue
        cells = [c.strip() for c in table_part.strip("|").split("|")]
        if any(cells):
            rows.append(cells)
    return rows


# ── XLSX helpers ──────────────────────────────────────────────────────────────

def _xlsx_schema_text(summary: str, sheet_name: str) -> str:
    """
    Builds a clean English schema description from the parser's sheet summary.

    The parser writes summaries in French:
        "Feuille 'Sales': 120 lignes, 8 colonnes. Colonnes : A, B, C, D…"

    We normalise to English and structure the output so embeddings capture
    column semantics:
        [Sheet: Sales] Schema
        Rows: 120  |  Columns (8): A, B, C, D…

    Falls back gracefully when the French pattern is not found (e.g. custom
    parser summary format).
    """
    rows_m    = re.search(r'(\d+)\s+lignes',    summary)
    cols_m    = re.search(r'(\d+)\s+colonnes',  summary)
    col_list_m = re.search(r'Colonnes\s*:\s*(.+)', summary, re.IGNORECASE)

    rows_str = rows_m.group(1)                              if rows_m     else "?"
    cols_str = cols_m.group(1)                              if cols_m     else "?"
    col_list = col_list_m.group(1).strip().rstrip('.')      if col_list_m else summary

    return (
        f"[Sheet: {sheet_name}] Schema\n"
        f"Rows: {rows_str}  |  Columns ({cols_str}): {col_list}"
    )


# ── EML helpers ───────────────────────────────────────────────────────────────

def _strip_quoted_reply(text: str) -> str:
    """
    Cleans up email body text in two passes:

    Pass 1 — Signature stripping
        Cuts everything at or after the RFC 3676 email signature delimiter.
        The delimiter is a line containing exactly '--' (with optional trailing
        whitespace).  Email parsers sometimes deliver it mid-paragraph or with
        encoding artefacts, so we apply both a regex pre-pass (to handle
        embedded delimiters) and a line-by-line check.

        Pre-pass  — regex splits on  \\n--\\s*\\n  or  \\n-- \\n  patterns
                    that may survive quoted-printable decoding.
        Line pass — for the common case where the delimiter is a clean line.

        Everything after the first match is discarded, removing the sender's
        name/title/phone/LinkedIn block and any tracking footers.

    Pass 2 — Quoted reply collapsing
        Collapses consecutive '>' quoted-reply lines into a single
        '[quoted reply]' placeholder.  Quoted reply blocks contain the
        previous email(s) in a thread and add noise to retrieval; collapsing
        them preserves the signal (there was a prior exchange) without
        polluting the embedding space.
    """
    # ── Pre-pass: regex signature cut ─────────────────────────────────────────
    # Handles cases where the parser joins `-- \n` mid-paragraph, leaving
    # no clean line boundary for the line-by-line check below.
    # Pattern: optional \r, newline, exactly "--" with optional trailing
    # whitespace, then another newline — RFC 3676 § 4.3.
    _SIG_RE = re.compile(r'\r?\n--\s*\r?\n')
    sig_match = _SIG_RE.search(text)
    if sig_match:
        text = text[:sig_match.start()].strip()

    # ── Line-by-line pass ─────────────────────────────────────────────────────
    lines    = text.splitlines()
    result:  list[str] = []
    in_quote = False

    for line in lines:
        # RFC 3676 signature delimiter — line is exactly "--" or "-- "
        if line.rstrip() == "--":
            break

        # Quoted reply collapsing
        if line.startswith(">"):
            if not in_quote:
                result.append("[quoted reply]")
                in_quote = True
        else:
            in_quote = False
            result.append(line)

    return "\n".join(result)


# Token threshold below which an email body chunk is flagged as potentially
# boilerplate.  Chunks at or under this limit get  boilerplate_risk=True  in
# their metadata so the retrieval layer can de-prioritize or filter them.
# The chunker never suppresses them — that decision belongs downstream.
_EML_BOILERPLATE_TOKENS = 60

# Common boilerplate phrases found in short email bodies.  If ALL non-empty
# sentences in a chunk match at least one of these patterns, the chunk is
# considered high-confidence boilerplate regardless of its token count.
_EML_BOILERPLATE_RE = re.compile(
    r'(?i)'
    r'(hope (you are|this email finds) (you )?(\w+ ?)+)'
    r'|(please find (the |this )?(revised |updated |)?(document|file|report|version)? ?(attached|below))'
    r'|(best regards|kind regards|warm regards|yours sincerely|yours faithfully)'
    r'|(thank you for (your )?(support|time|consideration|feedback|patience))'
    r'|(do not hesitate to (contact|reach out|get back))'
    r'|(looking forward to (hearing|meeting|your (reply|response|feedback)))'
    r'|(have a (great|good|nice|wonderful) (day|week|weekend))'
    r'|(feel free to (contact|reach out|let me know))'
)


def _is_boilerplate_body(text: str) -> bool:
    """
    Returns True if the body chunk looks like pure email pleasantries with
    no substantive informational content.

    Two conditions either of which triggers the flag:
      1. Token count ≤ _EML_BOILERPLATE_TOKENS  (very short body)
      2. Every non-empty sentence matches a known boilerplate pattern
         (regardless of length — catches polite but content-free emails)
    """
    tok = count_tokens(text)
    if tok <= _EML_BOILERPLATE_TOKENS:
        return True
    # Check if every sentence is boilerplate
    sentences = [s.strip() for s in re.split(r'[.!?\n]+', text) if s.strip()]
    if not sentences:
        return True
    return all(_EML_BOILERPLATE_RE.search(s) for s in sentences)


def _build_eml_header_text(meta: dict) -> str:
    """
    Builds an embeddable header string from email metadata fields.

    Output format:
        [Email Header]
        Subject: Q3 Sales Report — please review
        From: alice@example.com
        To: bob@example.com, carol@example.com
        Date: Mon, 14 Apr 2025 09:15:00 +0000
        Attachments: sales_q3.xlsx, notes.pdf

    This chunk lets a retriever answer "who sent the email about X?"
    or "which attachments came with the email from Y?"
    without needing to embed the full body text.
    """
    parts = ["[Email Header]"]
    for field in ("subject", "from", "to", "cc", "date"):
        val = (meta.get(field) or meta.get(field.capitalize()) or "").strip()
        if val:
            parts.append(f"{field.capitalize()}: {val}")
    attachments = meta.get("attachments") or []
    if attachments:
        att_names = ", ".join(
            # Parser stores attachment info as dicts with key "name" or "filename"
            (a.get("filename") or a.get("name") or str(a)) if isinstance(a, dict) else str(a)
            for a in attachments
        )
        parts.append(f"Attachments: {att_names}")
    return "\n".join(parts)


# ── Main entry point ──────────────────────────────────────────────────────────

def chunk_document(
    result: "ExtractionResult",
    *,
    include_images_b64: bool = False,
    doc_uid: str = "",
) -> dict:
    """
    Converts an ExtractionResult into a RAG-ready chunk dict.

    Automatically classifies the document as narrative / structured / mixed
    and selects the matching chunking strategy:

      narrative  → contextual  — paragraph-first with nearest-heading prefix
      structured → hierarchical — full section-path prefix [H1 > H2 > H3]
      mixed      → hierarchical (degrades gracefully to contextual when no
                                  headings are detected)

    For scanned PDFs the same text strategy applies; tables and images are
    always handled independently with their own chunk types.

    Returns:
        {
            "source_file": str,
            "stats":       { total_chunks, text_chunks, doc_type, strategy, ... },
            "chunks":      [ { chunk_id, source_file, page, type,
                               text, token_count, metadata?, rows?,
                               image_file?, base64_data? }, ... ]
        }
    """
    # ── Filename normalisation ────────────────────────────────────────────────
    raw_name   = result.source_file
    clean_name = re.sub(r'^[0-9a-f]{8}_', '', raw_name)
    stem_raw   = Path(clean_name).stem
    stem       = slugify(stem_raw)
    uid        = doc_uid or hashlib.md5(raw_name.encode()).hexdigest()[:8]
    file_ext   = Path(clean_name).suffix.lower()

    # ── Document-level GUID ───────────────────────────────────────────────────
    # UUID5 derived from the clean filename so re-indexing the same file always
    # produces the same doc_id (idempotent Qdrant upserts).  Every chunk from
    # this document — text, table, image — carries metadata.doc_id so retrieval
    # can pull every chunk belonging to a document via a single payload filter:
    #
    #   client.scroll(filter={"doc_id": doc_id})
    #
    # For EML files the same value is also written to metadata.email_id for
    # backward-compatibility with code that already filters on that key.
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, clean_name))

    # ── Document-level metadata ───────────────────────────────────────────────
    doc_meta = clean_metadata(result.doc_metadata) if result.doc_metadata else None

    # ── Auto-classify and select chunking strategy ────────────────────────────
    doc_type, strategy, reason = classify_document_type(
        result.text_blocks, result.tables, file_ext
    )
    logger.info(
        f"  chunker: doc_type={doc_type!r}, strategy={strategy!r} — {reason}"
    )

    # ── Chunk ID counter ──────────────────────────────────────────────────────
    _counter: dict[int, int] = {}

    def _next_id(pg: int, kind: str) -> str:
        _counter[pg] = _counter.get(pg, 0)
        cid = f"{uid}_{stem}_p{pg + 1}_{kind}{_counter[pg]}"
        _counter[pg] += 1
        return cid

    def _attach_meta(chunk: dict, extra: dict | None = None) -> None:
        combined: dict = {}
        if doc_meta:
            combined.update(doc_meta)
        if extra:
            combined.update(extra)
        if combined:
            chunk["metadata"] = combined

    chunks: list[dict] = []

    # Shared EML document GUID — set inside the EML branch, read by the image
    # and table loops that run after all the elif branches complete.
    _is_eml:       bool = False
    _eml_email_id: str  = ""

    # ── 1. Text blocks ────────────────────────────────────────────────────────
    # Plain-text files (.txt, .md, .rst …) and image files get their own
    # semantic pipeline: sentence splitting + optional embedding-based boundary
    # detection.  All other types use the generic contextual / hierarchical
    # chunkers.
    _plain_text_exts = {".txt", ".md", ".rst", ".text"}
    _image_file_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp"}
    _is_scanned_pdf  = (
        file_ext == ".pdf"
        and bool((result.doc_metadata or {}).get("scanned"))
    )
    _is_xlsx       = file_ext == ".xlsx"
    _is_pptx       = file_ext == ".pptx"
    _is_docx       = file_ext == ".docx"
    _is_image_file = file_ext in _image_file_exts

    # ── DOCX token budget ─────────────────────────────────────────────────────
    # DOCX paragraphs are typically denser and more self-contained than the
    # short Q&A snippets MAX_TOKENS=200 was tuned for.  Using 400 tokens lets
    # the chunker keep intact paragraphs that would otherwise be split mid-
    # sentence, while still staying well within most embedding model limits.
    _DOCX_MAX_TOKENS = 400
    _effective_max_tokens = _DOCX_MAX_TOKENS if _is_docx else MAX_TOKENS

    # ── XLSX: strip bulk / noisy fields from per-chunk metadata ──────────────
    # `_attach_meta()` merges the entire doc_meta onto every chunk.  For XLSX
    # files this includes fields that are either document-level aggregates
    # (sheet_names, sheet_count) or library artefacts (author="openpyxl",
    # openpyxl-generated timestamps) — none of which have value on individual
    # chunks when each chunk already carries sheet_name + sheet_index.
    #
    # `_attach_meta` captures `doc_meta` by name (Python closure), so
    # reassigning the variable here affects all subsequent chunk emissions.
    if _is_xlsx and doc_meta:
        _XLSX_CHUNK_META_EXCLUDE = {"sheet_names", "sheet_count", "created", "modified"}
        doc_meta = {
            k: v for k, v in doc_meta.items()
            if k not in _XLSX_CHUNK_META_EXCLUDE
            # Also drop "author" when it is the openpyxl library default
            and not (k == "author" and v == "openpyxl")
        }
        if not doc_meta:
            doc_meta = None

    # ── DOCX: strip library-generated metadata artifacts ─────────────────────
    # python-docx writes itself as "last_modified_by" when it opens a document
    # for metadata inspection. Word also stores a generic "Word Document" title
    # when no real title was set. Neither value has retrieval value.
    if _is_docx and doc_meta:
        _DOCX_META_EXCLUDE = {"paragraph_count"}
        doc_meta = {
            k: v for k, v in doc_meta.items()
            if k not in _DOCX_META_EXCLUDE
            and not (k == "last_modified_by" and v == "python-docx")
            and not (k == "title"           and v in ("Word Document", "Microsoft Word Document"))
        }
        if not doc_meta:
            doc_meta = None

    # For image files: count OCR tokens upfront — used later to decide whether
    # to suppress the image chunk (OCR is sufficient) or emit it as a caption
    # fallback (OCR produced too little text to be useful on its own).
    _IMG_MIN_OCR_TOKENS = 30   # below this → OCR failed / image is non-textual
    _img_ocr_tokens = (
        sum(count_tokens(t) for _, t in result.text_blocks)
        if _is_image_file else 0
    )

    # Whether OCR produced enough text for the image file path.
    # Used both here (strategy logging) and in the image chunk section below.
    _img_ocr_sufficient = _is_image_file and _img_ocr_tokens >= _IMG_MIN_OCR_TOKENS

    if (file_ext in _plain_text_exts or _is_scanned_pdf or _is_image_file) and result.text_blocks:
        # ── TXT / Scanned-PDF / Image path ────────────────────────────────────
        # Reconstruct the full OCR / plain text from parser blocks, then run
        # the same semantic pipeline used for .txt files:
        #   narrative              → sentence grouping by token budget
        #   structured / mixed     → section split → embedding → valley detection
        #
        # Image files use this path because their OCR output is structurally
        # identical to scanned PDF OCR — raw plain text with no paragraph
        # structure guarantee.  The generic paragraph chunker produces worse
        # boundaries for this kind of content.
        full_txt   = "\n\n".join(t for _, t in result.text_blocks)
        doc_type   = _detect_txt_doc_type(full_txt)

        if _is_scanned_pdf or _is_image_file:
            strategy = "txt_semantic_scanned_pdf" if _is_scanned_pdf else "txt_semantic_image"
            # Process each (page_num, text) block independently to preserve page
            # attribution. Previously all pages were joined into one flat string
            # and page numbers were discarded — every chunk ended up with page=1.
            tagged_txt_chunks: list[tuple[int, str]] = []
            for pnum, page_text in result.text_blocks:
                for cs in chunk_txt_content(page_text):
                    tagged_txt_chunks.append((pnum, cs))
        else:
            strategy = "txt_semantic"
            tagged_txt_chunks = [(0, cs) for cs in chunk_txt_content(full_txt)]
        logger.info(
            f"  chunker: "
            f"{'scanned PDF' if _is_scanned_pdf else 'image' if _is_image_file else 'plain text'}"
            f" → doc_type={doc_type!r}, strategy={strategy!r}"
            + (f", ocr_tokens={_img_ocr_tokens}" if _is_image_file else "")
        )

        if _is_scanned_pdf or _is_image_file:
            # ── Scanned-PDF / Image extra steps ──────────────────────────────
            # The vision OCR model (Llama 4 Scout) returns all content as plain
            # text, including [Figure: ...] markers and markdown tables.
            # We do two passes over each chunk:
            #   Pass 1 — extract [Figure: ...] markers → image chunks
            #   Pass 2 — extract markdown table blocks  → table chunks
            # Remaining prose becomes normal text chunks.
            para_chunks = []
            for pnum, chunk_str in tagged_txt_chunks:
                page_1 = pnum + 1  # Qdrant payload — 1-based page numbers

                # ── Pass 1: figure markers ────────────────────────────────────
                # The OCR prompt asks Llama 4 Scout to insert [Figure: caption]
                # wherever it sees a chart / diagram / non-text visual element.
                # Extract as standalone image chunks; strip from prose so they
                # don't pollute text embeddings.
                figure_matches = list(_FIGURE_RE.finditer(chunk_str))
                if figure_matches:
                    for fm in figure_matches:
                        caption = fm.group(1).strip()
                        fig_chunk: dict = {
                            "chunk_id":    _next_id(pnum, "fig_ocr"),
                            "source_file": clean_name,
                            "page":        page_1,
                            "type":        "image",
                            "caption":     caption,
                            "token_count": count_tokens(caption),
                        }
                        _attach_meta(fig_chunk, {"ocr_figure": True})
                        chunks.append(fig_chunk)
                    chunk_str = _FIGURE_RE.sub("", chunk_str).strip()
                    if not chunk_str:
                        continue

                # ── Pass 2: markdown table blocks ─────────────────────────────
                segs = _extract_md_tables_from_text(chunk_str)
                if len(segs) == 1 and segs[0][0] == "text":
                    para_chunks.append((pnum, chunk_str, None, None, None))
                else:
                    for seg_type, seg_content in segs:
                        tok = count_tokens(seg_content)
                        if tok < MIN_CHUNK_TOKENS:
                            continue
                        if seg_type == "table":
                            tbl_rows = _parse_md_table_rows(seg_content)
                            tc_inline: dict = {
                                "chunk_id":    _next_id(pnum, "tbl_ocr"),
                                "source_file": clean_name,
                                "page":        page_1,
                                "type":        "table",
                                "text":        seg_content,
                                "token_count": tok,
                                "rows":        tbl_rows,
                            }
                            _attach_meta(tc_inline, None)
                            chunks.append(tc_inline)
                        else:
                            para_chunks.append((pnum, seg_content, None, None, None))
        else:
            # Plain text (.txt / .md / .rst) — wrap as-is
            para_chunks = [(pnum, chunk_str, None, None, None)
                           for pnum, chunk_str in tagged_txt_chunks]
    elif _is_xlsx and result.text_blocks:
        # ── XLSX path ─────────────────────────────────────────────────────────
        # Each text_block is a parser-generated sheet summary (French prose).
        # We normalise it to an English schema description and emit one schema
        # chunk per sheet.  Actual data lives in the table chunks below.
        xlsx_sheet_names = (result.doc_metadata or {}).get("sheet_names", [])
        para_chunks = []
        for sidx, summary in result.text_blocks:
            sname = (xlsx_sheet_names[sidx]
                     if sidx < len(xlsx_sheet_names)
                     else f"Sheet{sidx + 1}")
            schema_text = _xlsx_schema_text(summary, sname)
            # Reuse the (page, text, att_src, slide_title, section) 5-tuple.
            # slide_title carries the sheet name so it lands in chunk metadata.
            para_chunks.append((sidx, schema_text, None, sname, "xlsx_schema"))
        doc_type = "structured"
        strategy = "xlsx_schema"
        logger.info(
            f"  chunker: xlsx → {len(para_chunks)} schema chunk(s) "
            f"+ {len(result.tables)} table chunk(s)"
        )
    elif file_ext == ".eml":
        # ── EML path ──────────────────────────────────────────────────────────
        # EML text blocks produced by _parse_eml() in parser.py:
        #   page=0        → email body paragraphs (possibly multiple blocks)
        #   page=1000×n   → attachment text (n = 1-based attachment index),
        #                    each block prefixed with [attachment:fname]\n
        #
        # Output: three logical sections in para_chunks —
        #   1. Header chunk     — subject / from / to / date / attachment names
        #                         One embeddable chunk for header-level search.
        #   2. Body chunks      — body text through TXT semantic pipeline.
        #                         Consecutive '>' quoted-reply lines are
        #                         collapsed to '[quoted reply]' to reduce noise.
        #   3. Attachment chunks — each attachment's text through TXT pipeline,
        #                          `attachment_source` set to the filename so
        #                          retrieval can filter by attachment.
        doc_type = "narrative"
        strategy = "eml"

        # ── email_id — deterministic GUID linking all chunks of this EML ──────
        # UUID5 from the EML filename so re-indexing the same file produces the
        # same email_id (idempotent Qdrant upserts).  Every chunk emitted in this
        # branch (header, body text, body tables, attachment text, attachment
        # tables, attachment images) carries metadata.email_id so a retriever
        # can answer "find the body of the email that contains attachment X" by
        # filtering all chunks with the same email_id.
        _eml_email_id = str(uuid.uuid5(uuid.NAMESPACE_URL, clean_name))
        logger.info("  chunker: EML → header + body + attachment chunks")
        # EML emits all chunks directly into `chunks` — para_chunks stays
        # empty so the generic 5-tuple emission loop below is a no-op.
        para_chunks = []

        # ── 1. Header chunk ───────────────────────────────────────────────────
        if doc_meta:
            header_text = _build_eml_header_text(doc_meta)
            tok_h = count_tokens(header_text)
            if tok_h >= MIN_CHUNK_TOKENS:
                c_hdr: dict = {
                    "chunk_id":    _next_id(0, "txt"),
                    "source_file": clean_name,
                    "page":        1,
                    "type":        "text",
                    "text":        header_text,
                    "token_count": tok_h,
                }
                _attach_meta(c_hdr, {"section": "email_header", "email_id": _eml_email_id})
                chunks.append(c_hdr)

        # ── 2. Body chunks ────────────────────────────────────────────────────
        # The body is split into (type, content) segments first so that
        # markdown tables embedded in the email body are extracted as proper
        # table chunks instead of being swallowed by chunk_txt_content().
        body_blocks = [t for pg, t in result.text_blocks if pg == 0]
        if body_blocks:
            body_text = _strip_quoted_reply("\n\n".join(body_blocks)).strip()
            if body_text:
                # Detect embedded markdown tables (pipe-delimited lines)
                body_segs = _extract_md_tables_from_text(body_text)

                for seg_type, seg_content in body_segs:
                    seg_content = seg_content.strip()
                    if not seg_content:
                        continue

                    if seg_type == "table":
                        # ── Inline body table ──────────────────────────────
                        tok = count_tokens(seg_content)
                        if tok < MIN_CHUNK_TOKENS:
                            continue
                        tbl_rows = _parse_md_table_rows(seg_content)
                        c_tbl: dict = {
                            "chunk_id":    _next_id(0, "tbl"),
                            "source_file": clean_name,
                            "page":        1,
                            "type":        "table",
                            "text":        seg_content,
                            "token_count": tok,
                            "rows":        tbl_rows,
                        }
                        _attach_meta(c_tbl, {"section": "email_body", "email_id": _eml_email_id})
                        chunks.append(c_tbl)

                    else:
                        # ── Prose segment — TXT semantic pipeline ──────────
                        for bc in chunk_txt_content(seg_content):
                            tok = count_tokens(bc)
                            if tok < MIN_CHUNK_TOKENS:
                                continue
                            boilerplate = _is_boilerplate_body(bc)
                            c_body: dict = {
                                "chunk_id":    _next_id(0, "txt"),
                                "source_file": clean_name,
                                "page":        1,
                                "type":        "text",
                                "text":        bc,
                                "token_count": tok,
                            }
                            body_extra: dict = {"section": "email_body", "email_id": _eml_email_id}
                            if boilerplate:
                                body_extra["boilerplate_risk"] = True
                            _attach_meta(c_body, body_extra)
                            chunks.append(c_body)

        # ── 3. Attachment chunks ──────────────────────────────────────────────
        # Group blocks by attachment filename.  Assign sequential page numbers
        # (1, 2, 3 …) so attachment chunks sort cleanly after the body in the
        # final output (page = pg + 1, so body=1, att1=2, att2=3 …).
        # Email-level metadata (subject, from, date) is propagated onto every
        # attachment chunk so retrieval can answer "find the PDF Majd sent in
        # March" even when the attachment text doesn't mention sender/date.
        eml_from    = (doc_meta or {}).get("from",    "")
        eml_subject = (doc_meta or {}).get("subject", "")
        eml_date    = (doc_meta or {}).get("date",    "")

        # Store (pg, text) pairs per attachment so page/slide numbers survive.
        att_blocks:   dict[str, list[tuple[int, str]]] = {}
        att_page_map: dict[str, int]                   = {}
        next_att_pg = 1

        for pg, raw in result.text_blocks:
            if pg == 0:
                continue
            clean_t, att_src = extract_attachment_source(raw)
            if not clean_t.strip():
                continue
            key = att_src or f"attachment_{pg}"
            if key not in att_blocks:
                att_blocks[key]   = []
                att_page_map[key] = next_att_pg
                next_att_pg += 1
            att_blocks[key].append((pg, clean_t))   # preserve pg

        for att_key, att_pairs in att_blocks.items():
            if not att_pairs:
                continue
            att_pg = att_page_map[att_key]

            # Derive file type from the attachment name stored in att_key.
            _att_ext = Path(att_key).suffix.lower() if "." in att_key else ""

            # Normalise page numbers: the parser offsets pages by 1000×N so
            # that different attachments don't collide.  Strip that base so
            # the page-aware chunkers see 0-indexed local page numbers.
            _base_page  = att_pairs[0][0] // 1000 * 1000
            local_blocks: list[tuple[int, str]] = [
                (pg - _base_page, text) for pg, text in att_pairs
            ]

            # Choose the right chunking strategy for this attachment type.
            if _att_ext == ".pptx":
                att_chunk_tuples = chunk_text_blocks_hierarchical(
                    local_blocks, file_type=".pptx"
                )
            elif _att_ext in (".docx", ".xlsx"):
                # DOCX/XLSX keep their per-block structure via the paragraph
                # chunker — blocks are already well-sized from the parsers.
                att_chunk_tuples = chunk_text_blocks(local_blocks)
            elif _att_ext == ".pdf":
                # PDF parser produces many small paragraph-level blocks
                # (often 40-60 tokens each).  Joining and running the TXT
                # semantic pipeline produces properly sized chunks (80-512 tok)
                # with semantic boundary detection instead of hundreds of tiny
                # fragments.
                att_full = "\n\n".join(text for _, text in local_blocks).strip()
                att_chunk_tuples = [
                    (0, ac, None, None, None)
                    for ac in chunk_txt_content(att_full)
                    if ac.strip()
                ]
            else:
                # Plain-text / unknown → join and run TXT semantic pipeline.
                att_full = "\n\n".join(text for _, text in local_blocks).strip()
                att_chunk_tuples = [
                    (0, ac, None, None, None)
                    for ac in chunk_txt_content(att_full)
                    if ac.strip()
                ]

            for local_pg, ac_text, _att_src2, slide_title, section in att_chunk_tuples:
                tok = count_tokens(ac_text)
                if tok < MIN_CHUNK_TOKENS:
                    continue
                c_att: dict = {
                    "chunk_id":    _next_id(att_pg, "txt"),
                    "source_file": clean_name,
                    "page":        att_pg + 1,
                    "type":        "text",
                    "text":        ac_text,
                    "token_count": tok,
                }
                att_extra: dict = {
                    "section":           "email_attachment",
                    "attachment_source": att_key,
                    "email_id":          _eml_email_id,
                }
                # For PPTX attachments expose the 1-indexed slide number so
                # retrievers can answer "find the slide about X in the deck".
                if _att_ext == ".pptx":
                    att_extra["slide"] = local_pg + 1
                if slide_title:
                    att_extra["slide_title"] = slide_title
                if eml_from:    att_extra["email_from"]    = eml_from
                if eml_subject: att_extra["email_subject"] = eml_subject
                if eml_date:    att_extra["email_date"]    = eml_date
                _attach_meta(c_att, att_extra)
                chunks.append(c_att)

    elif strategy == "hierarchical":
        para_chunks = chunk_text_blocks_hierarchical(
            result.text_blocks, file_type=file_ext,
            max_tokens=_effective_max_tokens,
        )
    else:
        para_chunks = chunk_text_blocks(
            result.text_blocks, max_tokens=_effective_max_tokens
        )

    for pg, chunk_text_val, att_src, slide_title, section in para_chunks:
        tok = count_tokens(chunk_text_val)
        if tok < MIN_CHUNK_TOKENS:
            continue
        c: dict = {
            "chunk_id":    _next_id(pg, "txt"),
            "source_file": clean_name,
            "page":        pg + 1,
            "type":        "text",
            "text":        chunk_text_val,
            "token_count": tok,
        }
        extra: dict = {}
        if att_src:
            extra["attachment_source"] = att_src
        if section == "xlsx_schema":
            # For xlsx schema chunks the "slide_title" slot carries the sheet
            # name — store it under the correct key and add sheet_index.
            if slide_title:
                extra["sheet_name"]  = slide_title
                extra["sheet_index"] = pg          # pg == sheet index for xlsx
        else:
            if slide_title:
                extra["slide_title"] = slide_title
        if section:
            extra["section"] = section
        _attach_meta(c, extra if extra else None)
        chunks.append(c)

    logger.debug(f"  chunker: {sum(1 for c in chunks if c['type'] == 'text')} text chunks")

    # ── 2. Tables ─────────────────────────────────────────────────────────────
    # Build a page→(section, slide_title) lookup from the text chunks so that
    # table chunks can carry the same section/slide context as nearby text.
    _page_ctx: dict[int, tuple[str | None, str | None]] = {}
    for c in chunks:
        if c["type"] == "text":
            pg0  = c["page"] - 1
            meta = c.get("metadata", {})
            _page_ctx.setdefault(pg0, (meta.get("section"), meta.get("slide_title")))

    # Pre-build xlsx sheet-name lookup (used in both small and large table paths)
    _xlsx_sheet_names: list[str] = (
        (result.doc_metadata or {}).get("sheet_names", [])
        if _is_xlsx else []
    )

    for t in result.tables:
        clean_md = clean_table_markdown(t.markdown)
        if not clean_md.strip():
            continue

        # For xlsx the parser already prepends "Sheet: {name}\n" to the
        # markdown.  Strip it now — the chunker writes its own [Sheet: {name}]
        # label and having both produces a visible duplicate in the output.
        if _is_xlsx:
            md_lines = clean_md.splitlines()
            if md_lines and re.match(r'^Sheet\s*:', md_lines[0]):
                clean_md = "\n".join(md_lines[1:]).lstrip("\n")

        header_lines = [ln for ln in clean_md.splitlines() if ln.strip().startswith('|')]
        header_row   = header_lines[0] if header_lines else ""

        # ── Context label ──────────────────────────────────────────────────────
        # xlsx: use sheet name as the primary label so chunks are self-contained
        #       and downstream retrieval can filter by sheet.
        # other: use section / slide_title from nearby text chunks.
        if _is_xlsx:
            tbl_sheet_name = (
                _xlsx_sheet_names[t.page_number]
                if t.page_number < len(_xlsx_sheet_names)
                else f"Sheet{t.page_number + 1}"
            )
            tbl_ctx, tbl_slide = None, None
            base_label = f"[Sheet: {tbl_sheet_name}]\n"
        else:
            # For DOCX, prefer the section stored on the table by the parser
            # (nearest heading in document order) over the page-based lookup.
            # _page_ctx works well for PPTX/PDF where pages separate content,
            # but DOCX puts everything on page 0 so page lookup is useless.
            _tbl_parser_section = getattr(t, "section", None)
            if _is_docx and _tbl_parser_section:
                tbl_ctx   = _tbl_parser_section
                tbl_slide = None
            else:
                tbl_ctx, tbl_slide = _page_ctx.get(t.page_number, (None, None))
            tbl_sheet_name = None
            ctx_prefix = ""
            if _is_pptx:
                # PPTX: use the [SECTION] name as context label — it spans
                # multiple slides and is more stable than a per-slide title.
                # slide_title is intentionally excluded from table metadata.
                if tbl_ctx:
                    ctx_prefix = f"[{tbl_ctx}]\n"
            else:
                if tbl_slide:
                    ctx_prefix = f"[{tbl_slide}]\n"
                elif tbl_ctx:
                    ctx_prefix = f"[{tbl_ctx}]\n"

            if ctx_prefix and _is_docx:
                # DOCX: section label is fully self-describing.
                # Page numbers are meaningless (everything is page 1) so
                # suppress the "[Table, page N]" suffix to avoid noise.
                base_label = ctx_prefix
            else:
                # PDF / PPTX / other: page numbers are meaningful; keep both.
                base_label = f"{ctx_prefix}[Table, page {t.page_number + 1}]\n"

        embeddable = base_label + clean_md
        tok        = count_tokens(embeddable)

        if tok <= MAX_TOKENS:
            # Table fits in one chunk
            if tok >= MIN_CHUNK_TOKENS:
                tc: dict = {
                    "chunk_id":    _next_id(t.page_number, "tbl"),
                    "source_file": clean_name,
                    "page":        t.page_number + 1,
                    "type":        "table",
                    "text":        embeddable,
                    "token_count": tok,
                    "rows":        t.raw_rows,
                }
                tbl_extra: dict = {}
                if tbl_ctx:                      tbl_extra["section"]     = tbl_ctx
                if tbl_slide and not _is_pptx:   tbl_extra["slide_title"] = tbl_slide
                if tbl_sheet_name:               tbl_extra["sheet_name"]  = tbl_sheet_name
                if _is_xlsx:                     tbl_extra["sheet_index"] = t.page_number
                if _is_eml and _eml_email_id:    tbl_extra["email_id"]    = _eml_email_id
                _attach_meta(tc, tbl_extra if tbl_extra else None)
                chunks.append(tc)
        else:
            # Large table — split into row-group chunks, each with header row
            data_rows = t.raw_rows[1:] if len(t.raw_rows) > 1 else t.raw_rows
            col_names = t.raw_rows[0] if t.raw_rows else []

            # Build markdown row groups that each fit within MAX_TOKENS
            label_tok  = count_tokens(base_label)
            header_tok = count_tokens(header_row + "\n") if header_row else 0
            budget     = MAX_TOKENS - label_tok - header_tok

            group:     list[str] = []
            group_tok: int       = 0
            group_raw: list[list] = []
            tbl_chunk_idx = 0

            md_rows = clean_md.splitlines()
            # Collect data rows from markdown (skip header and separator lines)
            md_data_rows = [
                ln for ln in md_rows
                if ln.strip().startswith('|') and not re.match(r'^\|[\s\-:|]+\|', ln)
                and ln != header_row
            ]

            def _emit_group(grp_lines: list[str], grp_raw: list[list], idx: int) -> None:
                nonlocal tbl_chunk_idx
                if not grp_lines:
                    return
                body = header_row + "\n" + "\n".join(grp_lines) if header_row else "\n".join(grp_lines)
                text = base_label + body
                t_tok = count_tokens(text)
                if t_tok < MIN_CHUNK_TOKENS:
                    return
                tc2: dict = {
                    "chunk_id":    _next_id(t.page_number, f"tbl{idx}"),
                    "source_file": clean_name,
                    "page":        t.page_number + 1,
                    "type":        "table",
                    "text":        text,
                    "token_count": t_tok,
                    "rows":        ([col_names] + grp_raw) if col_names else grp_raw,
                }
                tbl_extra2: dict = {}
                if tbl_ctx:                    tbl_extra2["section"]     = tbl_ctx
                if tbl_slide and not _is_pptx: tbl_extra2["slide_title"] = tbl_slide
                if tbl_sheet_name:             tbl_extra2["sheet_name"]  = tbl_sheet_name
                if _is_xlsx:                   tbl_extra2["sheet_index"] = t.page_number
                if _is_eml and _eml_email_id:  tbl_extra2["email_id"]    = _eml_email_id
                _attach_meta(tc2, tbl_extra2 if tbl_extra2 else None)
                chunks.append(tc2)

            for i, md_row in enumerate(md_data_rows):
                row_tok = count_tokens(md_row)
                if group_tok + row_tok > budget and group:
                    _emit_group(group, group_raw, tbl_chunk_idx)
                    tbl_chunk_idx += 1
                    group, group_tok, group_raw = [], 0, []
                group.append(md_row)
                group_tok += row_tok
                if i < len(data_rows):
                    group_raw.append(data_rows[i])
            if group:
                _emit_group(group, group_raw, tbl_chunk_idx)

    logger.debug(f"  chunker: {sum(1 for c in chunks if c['type'] == 'table')} table chunks")

    # ── 3. Images ─────────────────────────────────────────────────────────────
    # Suppression logic for standalone image files:
    #   OCR produced enough text  → skip the image chunk (text chunks are the
    #                               primary output; the caption adds nothing new)
    #   OCR produced minimal text → emit the Groq caption as an image chunk
    #                               (the file is non-textual: diagram, chart,
    #                               logo — the caption is the only useful output)
    # For all other file types (PDF, PPTX, DOCX, EML) the image chunks carry
    # embedded visuals and are always emitted.
    skip_img_chunk = _is_image_file and _img_ocr_sufficient

    # Minimum decoded image size in bytes below which an image is considered
    # decorative (tracking pixel, logo, divider, etc.) and skipped entirely.
    # base64_data length × 3/4 ≈ decoded bytes.  500 decoded bytes → ~667 b64 chars.
    _EML_MIN_IMG_B64_LEN = 700   # proxy for ~500 decoded bytes

    # EML email-level metadata to propagate onto every image chunk so that
    # filtered retrieval can find images by sender/subject/date/email_id.
    _is_eml = file_ext == ".eml"
    _eml_img_meta: dict = {}
    if _is_eml and doc_meta:
        for _field in ("from", "subject", "date"):
            _val = doc_meta.get(_field, "").strip()
            if _val:
                _eml_img_meta[f"email_{_field}"] = _val
    if _is_eml and _eml_email_id:
        _eml_img_meta["email_id"] = _eml_email_id

    for img in result.images:
        if skip_img_chunk:
            continue   # OCR text is sufficient — image chunk would be redundant

        # For image files where OCR was insufficient, log that we're emitting
        # the caption as a fallback so it's visible in the processing log.
        if _is_image_file and not _img_ocr_sufficient:
            logger.info(
                f"  chunker: image file OCR insufficient ({_img_ocr_tokens} tokens) "
                f"— emitting caption chunk as fallback"
            )

        # ── EML: skip decorative / tracking images ────────────────────────────
        # Images whose base64 payload is under the minimum threshold are almost
        # certainly tracking pixels, 1×1 spacers, or tiny decorative elements.
        # Skipping them avoids wasted Groq caption calls and index noise.
        if _is_eml and len(img.base64_data or "") < _EML_MIN_IMG_B64_LEN:
            logger.debug(
                f"  chunker: skipping tiny EML image "
                f"(b64_len={len(img.base64_data or '')}, likely decorative)"
            )
            continue

        img_ext  = img.mime_type.split("/")[-1]
        img_file = f"{stem_raw}_p{img.page_number + 1}_i{img.image_index + 1}.{img_ext}"

        # Caption text: use LLaVA/Groq caption if available, else build context
        if img.caption:
            caption_text = clean_chunk_text(img.caption)
        else:
            page_context = " ".join(
                t for pg, t in result.text_blocks if pg == img.page_number
            )
            if page_context:
                preview      = clean_chunk_text(page_context[:200].rstrip())
                caption_text = f"Image on page {img.page_number + 1}. Context: {preview}"
            else:
                caption_text = f"Image on page {img.page_number + 1} of {clean_name}."

        tok = count_tokens(caption_text)
        if tok < MIN_CHUNK_TOKENS:
            continue

        img_ctx, img_slide = _page_ctx.get(img.page_number, (None, None))
        item: dict = {
            "chunk_id":    _next_id(img.page_number, "img"),
            "source_file": clean_name,
            "page":        img.page_number + 1,
            "type":        "image",
            "text":        caption_text,
            "token_count": tok,
            "image_file":  img_file,
        }
        if include_images_b64:
            item["base64_data"] = img.base64_data

        img_extra: dict = {}
        if img_ctx:                    img_extra["section"]     = img_ctx
        if img_slide and not _is_pptx: img_extra["slide_title"] = img_slide

        # ── EML-specific image metadata ───────────────────────────────────────
        if _is_eml:
            # Propagate sender/subject/date so retrieval can filter by email
            img_extra.update(_eml_img_meta)

            # Inline body images (page=0) that have no caption are likely
            # decorative (logos, signature graphics).  Flag them so retrieval
            # can de-prioritise without dropping them from the index.
            if img.page_number == 0 and not img.caption:
                img_extra["decorative_risk"] = True

            # Attachment images (page ≥ 1000) carry their source filename
            if img.page_number >= 1000:
                # Recover attachment filename from the [From attachment: fname] caption prefix
                att_prefix = "[From attachment:"
                if img.caption and img.caption.startswith(att_prefix):
                    att_fname = img.caption[len(att_prefix):].split("]")[0].strip()
                    img_extra["attachment_source"] = att_fname
                img_extra["section"] = "email_attachment"

        _attach_meta(item, img_extra if img_extra else None)
        chunks.append(item)

    logger.debug(f"  chunker: {sum(1 for c in chunks if c['type'] == 'image')} image chunks")

    # ── Sort by page (document order) ─────────────────────────────────────────
    chunks.sort(key=lambda c: c["page"])

    # ── Schema normalisation ───────────────────────────────────────────────────
    # Three-phase pass so filtering happens before index assignment:
    #
    #   Phase 1 — Field renames + language detection
    #     text / image caption  →  content
    #     token_count           →  metadata.token_count
    #     page                  →  metadata.page_start / page_end
    #     source_file           →  metadata.source  (+ metadata.eml_source for EML)
    #     email_from/subject/date → from/subject/date
    #     table rows            →  prose content + metadata.display (markdown)
    #     language              →  metadata.language  (ISO 639-1, e.g. "en", "fr", "ar")
    #
    #   Phase 2 — Quality filters (applied before indexing so indices stay
    #     contiguous and chunk_total is accurate)
    #     • Drop chunks with empty or whitespace-only content — these embed as
    #       near-zero vectors and pollute the vector store with useless entries.
    #     • Drop chunks flagged boilerplate_risk=True — EML body noise (repeated
    #       headers, footers, automated signatures) that survived the cleaner;
    #       they are low-information and increase false-positive retrieval rates.
    #
    #   Phase 3 — chunk_index / chunk_total assignment
    _is_eml_doc = file_ext == ".eml"

    # ── Phase 1: field renames ────────────────────────────────────────────────
    for c in chunks:
        # 1a. Rename primary content field → "content"
        if "text" in c:
            c["content"] = c.pop("text")

        meta = c.setdefault("metadata", {})

        # 1b. Move token_count into metadata
        if "token_count" in c:
            meta["token_count"] = c.pop("token_count")

        # 1c. page → page_start / page_end
        _pg = c.pop("page", None)
        if _pg is not None:
            meta["page_start"] = _pg
            meta["page_end"]   = _pg

        # 1d. source_file → metadata.source (+ eml_source for EML docs)
        _src_file = c.pop("source_file", None)
        if _is_eml_doc:
            att_src = meta.pop("attachment_source", None)
            if att_src:
                meta["source"]     = att_src
                meta["eml_source"] = _src_file or clean_name
            else:
                meta.setdefault("source", _src_file or clean_name)
                meta["eml_source"] = _src_file or clean_name
            # email_from / email_subject / email_date → short keys
            for _old, _new in [
                ("email_from",    "from"),
                ("email_subject", "subject"),
                ("email_date",    "date"),
            ]:
                if _old in meta:
                    meta[_new] = meta.pop(_old)
        else:
            meta.setdefault("source", _src_file or clean_name)

        # 1e-0. Stamp doc_id on every chunk from every file type.
        # UUID5 is deterministic from clean_name, so re-indexing is idempotent.
        # EML files also write the same value to email_id (kept for compat).
        meta["doc_id"] = doc_id

        # 1f. Table chunks: convert markdown to prose for better embedding quality
        #     Pipe characters in raw markdown add noise to embedding similarity.
        #     Keep the original markdown in metadata.display for rendering;
        #     put a "header: value, …" prose row summary in content.
        if c["type"] == "table":
            original_md = c.get("content", "")
            rows = c.pop("rows", None)   # remove raw rows from top-level
            meta["display"] = original_md  # preserve markdown for rendering

            if rows and len(rows) >= 2:
                headers   = [str(h).strip() for h in rows[0]]
                prose_rows: list[str] = []
                for row in rows[1:]:
                    parts = [
                        f"{h}: {str(v).strip()}"
                        for h, v in zip(headers, row)
                        if str(v).strip()
                    ]
                    if parts:
                        prose_rows.append(", ".join(parts))
                if prose_rows:
                    # Preserve any context prefix that precedes the first "|"
                    md_start = original_md.find("|")
                    prefix   = original_md[:md_start].strip() if md_start > 0 else ""
                    prose    = "\n".join(prose_rows)
                    c["content"] = f"{prefix}\n{prose}".strip() if prefix else prose

        # 1g. Language detection ───────────────────────────────────────────────
        # Detect on the first 300 chars of content so the embedding layer can
        # route to the right model (multilingual vs English-only) and so
        # retrieval filters can restrict results to a specific language.
        # Detection runs on the content AFTER all renames and prose conversion
        # so it sees exactly what will be embedded.
        _lang = _detect_language(c.get("content", ""))
        if _lang:
            meta["language"] = _lang

    # ── Phase 2: quality filters ──────────────────────────────────────────────
    _before_filter = len(chunks)
    chunks = [
        c for c in chunks
        # Drop chunks with no useful content — they embed as near-zero vectors
        if c.get("content", "").strip()
        # Drop EML boilerplate (repeated headers/footers/signatures that
        # survived the cleaner) — low information, high false-positive rate
        and not (c.get("metadata") or {}).get("boilerplate_risk")
    ]
    _dropped = _before_filter - len(chunks)
    if _dropped:
        logger.info(f"  chunker: {_dropped} chunk(s) dropped by quality filter "
                    f"(empty content or boilerplate)")

    # ── Phase 3: chunk_index / chunk_total ────────────────────────────────────
    _total_chunks = len(chunks)
    for _idx, c in enumerate(chunks):
        c["metadata"]["chunk_index"] = _idx + 1
        c["metadata"]["chunk_total"] = _total_chunks

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats = {
        "total_chunks":       len(chunks),
        "text_chunks":        sum(1 for c in chunks if c["type"] == "text"),
        "table_chunks":       sum(1 for c in chunks if c["type"] == "table"),
        "image_chunks":       sum(1 for c in chunks if c["type"] == "image"),
        "total_tokens":       sum((c.get("metadata") or {}).get("token_count", 0) for c in chunks),
        "doc_type":           doc_type,
        "strategy":           strategy,
        "images_captioned":   result.stats["images_captioned"],
        "parsed_text_blocks": result.stats["text_blocks"],
        "parsed_tables":      result.stats["tables"],
        "parsed_images":      result.stats["images"],
    }

    logger.info(
        f"  chunker: {stats['total_chunks']} chunks total "
        f"({stats['text_chunks']} text, {stats['table_chunks']} table, "
        f"{stats['image_chunks']} image) — {stats['total_tokens']} tokens"
    )

    return {
        "source_file": clean_name,
        "stats":       stats,
        "chunks":      chunks,
    }