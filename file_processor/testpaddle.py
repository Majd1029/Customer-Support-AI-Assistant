import sys
import os
import re
import json

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PPOCR_LOG_LEVEL"] = "ERROR"

from paddleocr import PaddleOCR
from pathlib import Path

# ── INIT TWO ENGINES ─────────────────────────────────
print("Loading Arabic engine...")
_ocr_ar = PaddleOCR(
    lang="ar",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

print("Loading French engine...")
_ocr_fr = PaddleOCR(
    lang="fr",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

def extract_lines(result):
    """Flatten PaddleOCR result into list of (y_center, x_center, score, text)."""
    lines = []
    for page in result:
        for item in page:
            if isinstance(item, dict):
                text  = item.get("rec_text", "").strip()
                score = item.get("rec_score", 0.0)
                poly  = item.get("dt_polys", [[0,0],[0,0],[0,0],[0,0]])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                poly  = item[0]
                text  = item[1][0].strip() if item[1] else ""
                score = item[1][1] if item[1] else 0.0
            else:
                continue

            if text:
                ys = [pt[1] for pt in poly] if poly else [0]
                xs = [pt[0] for pt in poly] if poly else [0]
                y_center = sum(ys) / len(ys)
                x_center = sum(xs) / len(xs)
                lines.append((y_center, x_center, score, text))
    return lines

def merge_results(lines_ar, lines_fr, score_threshold=0.6):
    """Merge arabic and french pass results, keeping best score per line."""
    combined = [(y, x, s, t, "ar") for y, x, s, t in lines_ar] + \
               [(y, x, s, t, "fr") for y, x, s, t in lines_fr]

    combined.sort(key=lambda item: item[0])

    merged = []
    used = set()

    for i, (y, x, score, text, lang) in enumerate(combined):
        if i in used:
            continue
        best = (y, x, score, text, lang)
        for j, (y2, x2, score2, text2, lang2) in enumerate(combined):
            if j == i or j in used:
                continue
            if abs(y - y2) < 10 and lang2 != lang:
                if score2 > best[2]:
                    best = (y2, x2, score2, text2, lang2)
                used.add(j)
        used.add(i)
        if best[2] >= score_threshold:
            merged.append(best)

    return merged

def is_arabic(word):
    return any('\u0600' <= c <= '\u06FF' for c in word)

def fix_arabic_line(words):
    """Fix RTL word order and character order for Arabic."""
    # Reverse word order for RTL
    words = list(reversed(words))
    line = " ".join(words)

    # Fix merged date patterns e.g. 197023 → 1970 23
    line = re.sub(r'\b((?:19|20)\d{2})(\d{2})\b', r'\1 \2', line)

    # Reverse characters within each Arabic word
    fixed_words = []
    for word in line.split():
        if is_arabic(word):
            fixed_words.append(word[::-1])
        else:
            fixed_words.append(word)

    # Reverse word order again for correct RTL reading
    fixed_words = list(reversed(fixed_words))

    return " ".join(fixed_words)

def group_into_lines(merged, y_tolerance=15):
    """Group nearby words into lines, sorted by X position RTL."""
    if not merged:
        return []

    lines = []
    current_line = [merged[0]]

    for item in merged[1:]:
        if abs(item[0] - current_line[-1][0]) <= y_tolerance:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
    lines.append(current_line)

    result = []
    for line in lines:
        # Sort words right-to-left by X position
        line_sorted = sorted(line, key=lambda item: item[1], reverse=True)
        words = [item[3] for item in line_sorted]
        fixed = fix_arabic_line(words)
        result.append(fixed)

    return result

def test_ocr_two_pass(image_path: str):
    path = Path(image_path)
    assert path.exists(), f"Image not found: {image_path}"
    print(f"\n── TWO-PASS OCR on: {path.name} ────────────────")

    result_ar = _ocr_ar.ocr(str(path))
    result_fr = _ocr_fr.ocr(str(path))

    lines_ar = extract_lines(result_ar)
    lines_fr = extract_lines(result_fr)

    print(f"\n── Arabic pass  → {len(lines_ar)} lines")
    print(f"── French pass  → {len(lines_fr)} lines")

    merged = merge_results(lines_ar, lines_fr)
    lines_grouped = group_into_lines(merged)

    # Save to file
    output_path = path.parent / "ocr_output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines_grouped:
            f.write(line + "\n")
    print(f"Saved to {output_path}")

    print("\n── GROUPED LINES ──────────────────────────────")
    for line in lines_grouped:
        print(line)

    print(f"\n── MERGED RESULT ({len(merged)} lines) ──────────────────")
    for y, x, score, text, lang in merged:  # fixed: unpack 5 values
        print(f"[{lang}][{score:.2f}] {text}")

    print("\n── RAW Arabic pass ──────────────────────────────")
    print(json.dumps(result_ar, ensure_ascii=False, indent=2, default=str))

    print("\n── RAW French pass ──────────────────────────────")
    print(json.dumps(result_fr, ensure_ascii=False, indent=2, default=str))

if __name__ == "__main__":
    image_path = r"C:\Users\majda\OneDrive\Desktop\Secure-AI-Assistant-main\file_processor\image.jpg"
    test_ocr_two_pass(image_path)