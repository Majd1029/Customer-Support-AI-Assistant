"""
file_preparation/evaluation/metrics.py

Automatic reference-based evaluation metrics for the RAG pipeline.

Implements ROUGE (1, 2, L) and BLEU alongside the LLM-as-a-Judge so that
answers can be measured without an API call — useful for CI regression tests,
batch benchmarking, and rapid A/B comparisons.

Why each metric:
  ROUGE-1 recall   — how many key words from the reference appear in the answer
  ROUGE-2 recall   — how many bigrams (2-word phrases) are shared
  ROUGE-L F1       — longest common subsequence; captures sentence-level fluency
  BLEU-1..4        — n-gram precision; penalises hallucinated content

The preferred single number for RAG evaluation is ROUGE-L F1.
A BLEU score below 0.10 with a ROUGE-L above 0.50 typically indicates the
model is paraphrasing correctly but adding spurious phrases.

Install dependencies once:
    pip install rouge-score nltk
    python -c "import nltk; nltk.download('punkt_tab')"

Usage
-----
    from file_preparation.evaluation.metrics import score_answer, batch_score, MetricResult

    result = score_answer(
        answer    = "Revenue was $4.2B, up 12% year-over-year.",
        reference = "Revenue reached $4.2B in Q3, a 12% YoY increase.",
    )
    print(result.rouge_l_f1)   # e.g. 0.73
    print(result.bleu)         # e.g. 0.51
    print(result.to_dict())    # full breakdown

    # Batch
    rows = batch_score(records, csv_path="metrics_log.csv")
"""

from __future__ import annotations

import csv
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from loguru import logger

# ── Optional heavy imports ─────────────────────────────────────────────────────

try:
    from rouge_score import rouge_scorer as _rouge_scorer_mod
    _ROUGE_OK = True
except ImportError:
    _ROUGE_OK = False
    logger.warning("[METRICS] rouge-score not installed — run: pip install rouge-score")

try:
    from nltk.translate.bleu_score import (
        sentence_bleu,
        SmoothingFunction,
        corpus_bleu,
    )
    from nltk.tokenize import word_tokenize
    _BLEU_OK = True
except ImportError:
    _BLEU_OK = False
    logger.warning(
        "[METRICS] nltk not installed or punkt_tab missing — "
        "run: pip install nltk && python -c \"import nltk; nltk.download('punkt_tab')\""
    )


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    """ROUGE + BLEU scores for a single answer / reference pair."""

    question:    str
    answer:      str
    reference:   str

    # ROUGE scores (None when rouge-score not installed)
    rouge1_precision: float | None = None
    rouge1_recall:    float | None = None
    rouge1_f1:        float | None = None

    rouge2_precision: float | None = None
    rouge2_recall:    float | None = None
    rouge2_f1:        float | None = None

    rougel_precision: float | None = None
    rougel_recall:    float | None = None
    rouge_l_f1:       float | None = None   # primary metric for RAG

    # BLEU scores (None when nltk not installed)
    bleu:    float | None = None  # corpus BLEU (1–4 gram average)
    bleu_1:  float | None = None
    bleu_2:  float | None = None
    bleu_3:  float | None = None
    bleu_4:  float | None = None

    elapsed_ms: float = 0.0
    error:      str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def summary(self) -> str:
        """One-line human-readable summary."""
        r = f"ROUGE-L={self.rouge_l_f1:.3f}" if self.rouge_l_f1 is not None else "ROUGE-L=n/a"
        b = f"BLEU={self.bleu:.3f}"           if self.bleu is not None          else "BLEU=n/a"
        r1 = f"R1={self.rouge1_f1:.3f}"        if self.rouge1_f1 is not None     else "R1=n/a"
        return f"{r}  {r1}  {b}"


# ── Core scorer ────────────────────────────────────────────────────────────────

def score_answer(
    answer:    str,
    reference: str,
    question:  str = "",
) -> MetricResult:
    """
    Compute ROUGE (1, 2, L) and BLEU scores for a single answer against
    a reference string.

    Parameters
    ----------
    answer    : The RAG-generated answer to evaluate.
    reference : Ground-truth reference answer.
    question  : Optional — stored in the result for tracking purposes only.

    Returns
    -------
    MetricResult with all metric fields populated (or None when the
    corresponding library is not installed).
    """
    t0     = time.monotonic()
    result = MetricResult(question=question, answer=answer, reference=reference)

    if not answer.strip() or not reference.strip():
        result.error = "answer or reference is empty"
        result.elapsed_ms = (time.monotonic() - t0) * 1000
        return result

    # ── ROUGE ────────────────────────────────────────────────────────────────
    if _ROUGE_OK:
        try:
            scorer = _rouge_scorer_mod.RougeScorer(
                ["rouge1", "rouge2", "rougeL"],
                use_stemmer=True,   # normalises inflections: "running" ≈ "run"
            )
            scores = scorer.score(reference, answer)

            result.rouge1_precision = round(scores["rouge1"].precision, 4)
            result.rouge1_recall    = round(scores["rouge1"].recall,    4)
            result.rouge1_f1        = round(scores["rouge1"].fmeasure,  4)

            result.rouge2_precision = round(scores["rouge2"].precision, 4)
            result.rouge2_recall    = round(scores["rouge2"].recall,    4)
            result.rouge2_f1        = round(scores["rouge2"].fmeasure,  4)

            result.rougel_precision = round(scores["rougeL"].precision, 4)
            result.rougel_recall    = round(scores["rougeL"].recall,    4)
            result.rouge_l_f1       = round(scores["rougeL"].fmeasure,  4)

        except Exception as exc:
            logger.warning(f"[METRICS] ROUGE scoring failed: {exc}")
            result.error = str(exc)

    # ── BLEU ─────────────────────────────────────────────────────────────────
    if _BLEU_OK:
        try:
            smooth   = SmoothingFunction().method1   # avoids zero for short texts
            ref_toks = [word_tokenize(reference.lower())]
            hyp_toks = word_tokenize(answer.lower())

            result.bleu_1 = round(sentence_bleu(ref_toks, hyp_toks, weights=(1, 0, 0, 0), smoothing_function=smooth), 4)
            result.bleu_2 = round(sentence_bleu(ref_toks, hyp_toks, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth), 4)
            result.bleu_3 = round(sentence_bleu(ref_toks, hyp_toks, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth), 4)
            result.bleu_4 = round(sentence_bleu(ref_toks, hyp_toks, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth), 4)
            # Geometric mean of 1–4 gram BLEU
            result.bleu   = round(sentence_bleu(ref_toks, hyp_toks, smoothing_function=smooth), 4)

        except Exception as exc:
            logger.warning(f"[METRICS] BLEU scoring failed: {exc}")
            if not result.error:
                result.error = str(exc)

    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
    result.elapsed_ms = elapsed_ms if elapsed_ms > 0 else 0.1
    return result


# ── Thresholds / interpretation ────────────────────────────────────────────────

# Suggested pass/fail thresholds for a RAG system.
# These are starting points — tune on your own eval set.
ROUGE_L_PASS    = 0.40   # below this → answer misses key content
ROUGE_L_WARN    = 0.25   # below this → answer is seriously off
BLEU_PASS       = 0.15   # below this → high precision n-gram mismatch
ROUGE1_PASS     = 0.50   # unigram recall; easier bar


def interpret(result: MetricResult) -> str:
    """
    Return a simple verdict string: 'pass' | 'warn' | 'fail'.

    Uses ROUGE-L F1 as the primary signal (most informative for RAG).
    Falls back to ROUGE-1 if rougeL is unavailable.
    """
    score = result.rouge_l_f1 if result.rouge_l_f1 is not None else result.rouge1_f1
    if score is None:
        return "unknown"
    if score >= ROUGE_L_PASS:
        return "pass"
    if score >= ROUGE_L_WARN:
        return "warn"
    return "fail"


# ── Batch scoring ──────────────────────────────────────────────────────────────

def batch_score(
    records:  list[dict],
    csv_path: str | Path | None = None,
) -> list[MetricResult]:
    """
    Score a list of records and optionally append results to a CSV file.

    Each record must have 'answer' and 'reference' keys; 'question' is optional.

    Parameters
    ----------
    records  : list of dicts with keys 'question', 'answer', 'reference'.
    csv_path : If given, results are appended to this CSV file (created with
               headers on first use).  Pass None to skip CSV logging.

    Returns
    -------
    list[MetricResult]

    Example
    -------
        records = [
            {"question": "What is revenue?",
             "answer":   "Revenue was $4.2B.",
             "reference": "Revenue reached $4.2B in Q3."},
        ]
        results = batch_score(records, csv_path="metrics_log.csv")
        for r in results:
            print(r.summary, interpret(r))
    """
    results: list[MetricResult] = []
    for rec in records:
        r = score_answer(
            answer    = rec.get("answer", ""),
            reference = rec.get("reference", ""),
            question  = rec.get("question", ""),
        )
        results.append(r)
        verdict = interpret(r)
        logger.info(f"[METRICS] {r.summary}  verdict={verdict}  q={r.question[:60]!r}")

    if csv_path and results:
        save_to_csv(results, csv_path)

    return results


# ── CSV logging ────────────────────────────────────────────────────────────────

_CSV_COLS = [
    "timestamp", "question",
    "rouge1_precision", "rouge1_recall", "rouge1_f1",
    "rouge2_precision", "rouge2_recall", "rouge2_f1",
    "rougel_precision", "rougel_recall", "rouge_l_f1",
    "bleu", "bleu_1", "bleu_2", "bleu_3", "bleu_4",
    "verdict", "elapsed_ms", "error",
    "answer", "reference",
]


def save_to_csv(
    results:  list[MetricResult] | MetricResult,
    filepath: str | Path = "metrics_log.csv",
) -> None:
    """
    Append one or more MetricResult rows to a CSV file.
    Creates the file with headers if it does not exist.
    """
    if isinstance(results, MetricResult):
        results = [results]

    filepath = Path(filepath)
    write_header = not filepath.exists() or filepath.stat().st_size == 0

    with open(filepath, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for r in results:
            row = r.to_dict()
            row["timestamp"] = datetime.now(timezone.utc).isoformat()
            row["verdict"]   = interpret(r)
            writer.writerow(row)

    logger.info(f"[METRICS] Saved {len(results)} row(s) to {filepath}")


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    _answer    = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Revenue was $4.2 billion in Q3 2024, representing a 12% year-over-year increase. "
        "Operating expenses grew 8% to $1.1B."
    )
    _reference = (
        "Q3 2024 revenue reached $4.2B, up 12% compared to Q3 2023. "
        "Operating expenses were $1.1 billion, an 8% increase year-over-year."
    )

    print("\n── Reference ──────────────────────────────────────────────")
    print(_reference)
    print("\n── Answer ─────────────────────────────────────────────────")
    print(_answer)
    print("\n── Scores ─────────────────────────────────────────────────")

    r = score_answer(_answer, _reference, question="What were the Q3 financials?")

    print(f"  ROUGE-1  precision={r.rouge1_precision}  recall={r.rouge1_recall}  F1={r.rouge1_f1}")
    print(f"  ROUGE-2  precision={r.rouge2_precision}  recall={r.rouge2_recall}  F1={r.rouge2_f1}")
    print(f"  ROUGE-L  precision={r.rougel_precision}  recall={r.rougel_recall}  F1={r.rouge_l_f1}")
    print(f"  BLEU-1={r.bleu_1}  BLEU-2={r.bleu_2}  BLEU-3={r.bleu_3}  BLEU-4={r.bleu_4}")
    print(f"  BLEU (avg)={r.bleu}  elapsed={r.elapsed_ms}ms")
    print(f"\n  Verdict: {interpret(r).upper()}")
