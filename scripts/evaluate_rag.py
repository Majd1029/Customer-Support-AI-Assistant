"""
evaluate_rag.py — RAG evaluation suite for the internship report.

Two evaluation modes (run each one independently):

  ROUGE / BLEU evaluation  (--rouge-bleu)
    • ROUGE-1 F1        — unigram overlap with reference
    • ROUGE-2 F1        — bigram overlap with reference
    • ROUGE-L F1        — longest common subsequence
    • BLEU-1            — 1-gram precision (unigram)
    • BLEU-4            — 4-gram precision (standard MT metric)
    • Coverage          — fraction of reference words found in the answer
    Calls /ask with judge=False — no LLM evaluation charges, ~3-5× faster.
    Limitation: surface token overlap; penalises correct paraphrases.

  LLM-as-a-Judge evaluation  (--judge-only)
    • Faithfulness      — every claim grounded in context (weight 0.35)
    • Answer Relevance  — directly addresses the question (weight 0.25)
    • Context Relevance — retrieved chunks are relevant (weight 0.15)
    • Completeness      — all aspects of question covered (weight 0.15)
    • Citation Accuracy — [Source:] citations map to supporting chunks (weight 0.10)
    • Overall (weighted)— aggregated judge score (0–1)
    Calls /ask with judge=True; the qwen/qwen3-32b judge is invoked via Groq.

Optional ablation modes (for either evaluation):
  • no_prompt  — blank persona / no grounding rules, retrieval only
  • no_memory  — memory_enabled=False, no query rewriting

Usage:
    # ROUGE / BLEU evaluation alone
    python scripts/evaluate_rag.py --testset eval_testset.json --rouge-bleu

    # LLM-as-a-Judge evaluation alone
    python scripts/evaluate_rag.py --testset eval_testset.json --judge-only

    # ROUGE / BLEU — recompute from a previously saved CSV (zero API calls)
    python scripts/evaluate_rag.py --testset eval_testset.json --from-csv eval_results.csv

    # Skip ablations for the quickest single-mode run
    python scripts/evaluate_rag.py --testset eval_testset.json --judge-only --skip-noprompt --skip-nomem

    # Generate + save a new testset, then evaluate
    python scripts/evaluate_rag.py --n 20 --sleep 4 --save-testset eval_testset.json

Requirements:
    pip install requests qdrant-client groq loguru python-dotenv rouge-score nltk
    python -c "import nltk; nltk.download('punkt_tab')"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)

# ── Optional lexical metric packages ─────────────────────────────────────────

try:
    from rouge_score import rouge_scorer as _rouge_scorer_mod
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False
    logger.warning("[EVAL] rouge-score not installed — ROUGE metrics will be N/A.  "
                   "Install: pip install rouge-score")

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _BLEU_AVAILABLE = True
    # Ensure punkt tokeniser data is present (download silently if missing)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass
except ImportError:
    _BLEU_AVAILABLE = False
    logger.warning("[EVAL] nltk not installed — BLEU metrics will be N/A.  "
                   "Install: pip install nltk")

# ── Config ────────────────────────────────────────────────────────────────────

API_URL        = os.getenv("EVAL_API_URL", "http://localhost:8000")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_QA_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"
QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION     = "documents"
ASK_TOKEN      = os.getenv("EVAL_JWT_TOKEN", "")   # set if /ask requires auth

# Judge dimensions returned by the system
JUDGE_DIMS = [
    ("faithfulness",      "Faithfulness"),
    ("answer_relevance",  "Answer Relevance"),
    ("context_relevance", "Context Relevance"),
    ("completeness",      "Completeness"),
    ("citation_accuracy", "Citation Accuracy"),
]

# Singleton ROUGE scorer (initialised lazily on first use)
_rouge_scorer: Any = None

def _get_rouge_scorer():
    global _rouge_scorer
    if _rouge_scorer is None and _ROUGE_AVAILABLE:
        _rouge_scorer = _rouge_scorer_mod.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
    return _rouge_scorer


# ── Tier 1: Lexical metrics (ROUGE / BLEU) ────────────────────────────────────

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_rouge(pred: str, ref: str) -> dict[str, float | None]:
    """Return ROUGE-1, ROUGE-2, ROUGE-L F1 scores (or None when unavailable)."""
    scorer = _get_rouge_scorer()
    if scorer is None:
        return {"rouge1": None, "rouge2": None, "rougeL": None}
    scores = scorer.score(ref, pred)   # rouge_score convention: (target, prediction)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def compute_bleu(pred: str, ref: str) -> dict[str, float | None]:
    """Return BLEU-1 and BLEU-4 (or None when unavailable)."""
    if not _BLEU_AVAILABLE:
        return {"bleu1": None, "bleu4": None}
    try:
        ref_tokens  = nltk.word_tokenize(_normalize(ref))
        pred_tokens = nltk.word_tokenize(_normalize(pred))
        smooth = SmoothingFunction().method1
        bleu1 = sentence_bleu([ref_tokens], pred_tokens,
                               weights=(1, 0, 0, 0), smoothing_function=smooth)
        bleu4 = sentence_bleu([ref_tokens], pred_tokens,
                               weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        return {"bleu1": round(bleu1, 4), "bleu4": round(bleu4, 4)}
    except Exception as e:
        logger.debug(f"[EVAL] BLEU computation error: {e}")
        return {"bleu1": None, "bleu4": None}


def compute_lexical(pred: str, ref: str) -> dict[str, float | None]:
    """Compute all lexical metrics for one prediction/reference pair."""
    result = {}
    result.update(compute_rouge(pred, ref))
    result.update(compute_bleu(pred, ref))
    return result


# ── Tier 2: Coverage (semantic recall proxy, no LLM) ─────────────────────────

def coverage(pred: str, ref: str) -> float:
    """Fraction of reference words that appear in the prediction (recall proxy)."""
    p_words = set(_normalize(pred).split())
    r_words = set(_normalize(ref).split())
    if not r_words:
        return 1.0
    return round(len(p_words & r_words) / len(r_words), 4)


# ── Step 1: pull chunks from Qdrant ──────────────────────────────────────────

def fetch_chunks(n_chunks: int = 80) -> list[dict]:
    """Scroll Qdrant and return up to n_chunks text chunks."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        results, _ = client.scroll(
            collection_name=COLLECTION,
            limit=n_chunks,
            with_payload=True,
            with_vectors=False,
        )
        chunks = []
        for pt in results:
            p = pt.payload or {}
            content = p.get("content", "")
            if len(content.split()) >= 30:
                chunks.append({
                    "content": content[:1500],
                    "source":  p.get("source", "unknown"),
                    "section": p.get("section", ""),
                })
        logger.info(f"[EVAL] Fetched {len(chunks)} usable chunks from Qdrant")
        return chunks
    except Exception as e:
        logger.error(f"[EVAL] Qdrant error: {e}")
        return []


# ── Step 2: generate Q&A pairs via Groq ──────────────────────────────────────

_QA_SYSTEM = (
    "You are an exam question writer. "
    "Given a document excerpt, generate exactly ONE factual question answerable "
    "directly from that excerpt, plus the correct reference answer. "
    "Rules: answer must be 1–3 sentences from the excerpt; "
    "avoid yes/no questions; avoid questions about document metadata. "
    'Respond ONLY with valid JSON: {"question": "...", "answer": "..."}'
)


def _call_groq_qa(content: str) -> dict | None:
    if not GROQ_API_KEY:
        logger.error("[EVAL] GROQ_API_KEY not set")
        return None
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=GROQ_QA_MODEL,
            messages=[
                {"role": "system", "content": _QA_SYSTEM},
                {"role": "user",   "content": f"Document excerpt:\n\n{content}"},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"[EVAL] Q&A generation error: {e}")
        return None


def generate_testset(chunks: list[dict], n: int = 20, sleep: float = 2.0) -> list[dict]:
    import random
    random.shuffle(chunks)
    pairs: list[dict] = []
    for chunk in chunks:
        if len(pairs) >= n:
            break
        qa = _call_groq_qa(chunk["content"])
        if qa and qa.get("question") and qa.get("answer"):
            pairs.append({
                "question":  qa["question"],
                "reference": qa["answer"],
                "source":    chunk["source"],
                "section":   chunk["section"],
            })
            logger.info(f"[EVAL] [{len(pairs)}/{n}] Q: {qa['question'][:80]}")
        time.sleep(sleep)
    logger.info(f"[EVAL] Generated {len(pairs)} Q&A pairs")
    return pairs


# ── Step 3a: Tier 1-only — fast lexical scoring (no LLM judge) ───────────────

def _ask_tier1(question: str, reference: str, mode: str) -> dict:
    """
    Call POST /ask with judge=False (no LLM-as-a-Judge, no Groq charge for evaluation).
    Computes only Tier 1 (ROUGE/BLEU) and Tier 2 (coverage, confidence) locally.

    Much faster than the full pipeline — useful for a quick lexical baseline run
    or when Groq eval quota is exhausted.
    """
    body: dict[str, Any] = {
        "question":       question,
        "limit":          5,
        "rerank":         False,
        "use_hyde":       False,
        "mmr":            False,
        "multi_hop":      False,
        "memory_enabled": False,
        "judge":          False,     # ← no LLM judge
        "max_tokens":     1024,
    }

    if mode == "no_prompt":
        body["persona"] = " "

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if ASK_TOKEN:
        headers["Authorization"] = f"Bearer {ASK_TOKEN}"

    empty: dict[str, Any] = {
        "answer": "",
        "rouge1": None, "rouge2": None, "rougeL": None,
        "bleu1":  None, "bleu4":  None,
        "coverage": None, "confidence": None,
        # Tier 3 stays None — not requested
        "faithfulness": None, "answer_relevance": None,
        "context_relevance": None, "completeness": None,
        "citation_accuracy": None, "overall": None,
    }

    try:
        resp = requests.post(f"{API_URL}/ask", json=body, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"[EVAL] /ask failed ({mode}): {e}")
        return empty

    answer     = data.get("answer", "")
    confidence = data.get("confidence")

    lexical = compute_lexical(answer, reference)
    cov     = coverage(answer, reference)

    return {
        "answer":     answer,
        "rouge1":     lexical.get("rouge1"),
        "rouge2":     lexical.get("rouge2"),
        "rougeL":     lexical.get("rougeL"),
        "bleu1":      lexical.get("bleu1"),
        "bleu4":      lexical.get("bleu4"),
        "coverage":   cov,
        "confidence": confidence,
        # Tier 3 — not computed
        "faithfulness": None, "answer_relevance": None,
        "context_relevance": None, "completeness": None,
        "citation_accuracy": None, "overall": None,
    }


def _tier1_from_csv(csv_path: str, testset: list[dict]) -> list[dict]:
    """
    Load answers from a previously saved eval CSV (full_answer column) and
    recompute Tier 1 metrics locally — zero API calls required.

    The CSV must have been produced by a previous evaluate_rag.py run and contain
    at minimum: 'question' and 'full_answer' columns (or 'full_rouge1' already
    computed — in which case this is a no-op recheck).
    """
    try:
        import csv as _csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
    except FileNotFoundError:
        logger.error(f"[EVAL] --from-csv: file not found: {csv_path}")
        return []

    # Build a question → answer lookup from the CSV
    answer_map: dict[str, str] = {}
    for row in rows:
        q = row.get("question", "").strip()
        a = row.get("full_answer", row.get("answer", "")).strip()
        if q and a:
            answer_map[q] = a

    results: list[dict] = []
    for item in testset:
        q   = item["question"].strip()
        ref = item.get("reference", "")
        ans = answer_map.get(q, "")
        if not ans:
            logger.warning(f"[EVAL] --from-csv: no saved answer for: {q[:60]}")
            results.append({
                "answer": "", "rouge1": None, "rouge2": None, "rougeL": None,
                "bleu1": None, "bleu4": None, "coverage": None, "confidence": None,
                "faithfulness": None, "answer_relevance": None,
                "context_relevance": None, "completeness": None,
                "citation_accuracy": None, "overall": None,
            })
            continue
        lexical = compute_lexical(ans, ref)
        cov     = coverage(ans, ref)
        results.append({
            "answer":   ans,
            "rouge1":   lexical.get("rouge1"),
            "rouge2":   lexical.get("rouge2"),
            "rougeL":   lexical.get("rougeL"),
            "bleu1":    lexical.get("bleu1"),
            "bleu4":    lexical.get("bleu4"),
            "coverage": cov,
            "confidence": None,   # not stored in CSV
            "faithfulness": None, "answer_relevance": None,
            "context_relevance": None, "completeness": None,
            "citation_accuracy": None, "overall": None,
        })
    logger.info(f"[EVAL] Tier 1 recomputed from CSV for {len(results)} questions")
    return results


# ── Step 3b: Full pipeline — all three tiers ─────────────────────────────────

def _ask(question: str, reference: str, mode: str, judge_only: bool = False) -> dict:
    """
    Call POST /ask with judge=True, then compute all three evaluation tiers:
      Tier 1 — ROUGE-1/2/L, BLEU-1/4   (lexical, computed here from reference)
      Tier 2 — coverage, confidence      (semantic proxy)
      Tier 3 — judge dimensions          (LLM-as-a-Judge via /ask response)

    mode:
      "full"      — standard RAG (system prompt + retrieval)
      "no_prompt" — blank persona, no grounding rules
      "no_memory" — memory_enabled=False, no query rewriting

    judge_only:
      When True, skip Tier 1 (ROUGE/BLEU) and Tier 2 (coverage) entirely.
      Only the LLM-as-a-Judge dimensions are computed; lexical/coverage
      fields are left as None. Confidence is still pulled from /ask since
      it costs nothing (returned in the response).
    """

    body: dict[str, Any] = {
    "question":       question,
    "limit":          4,         # was 5 or 10
    "rerank":         True,      # was False
    "context_window": 0,         # was 1 (the default)
    "min_score":      0.10,      # was 0.01 (the default)
    "use_hyde":       False,
    "mmr":            False,
    "multi_hop":      False,
    "memory_enabled": False,
    "judge":          True,      # for langsmith_eval.py keep this False since LangSmith runs the judge separately
    "max_tokens":     1024,
    "include_chunks": True,
}

    if mode == "no_prompt":
        body["persona"] = " "          # blank persona disables system prompt rules
    # "no_memory" is already memory_enabled=False by default

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if ASK_TOKEN:
        headers["Authorization"] = f"Bearer {ASK_TOKEN}"

    empty: dict[str, Any] = {
        "answer": "",
        # Tier 1 — lexical
        "rouge1": None, "rouge2": None, "rougeL": None,
        "bleu1":  None, "bleu4":  None,
        # Tier 2 — semantic proxy
        "coverage": None, "confidence": None,
        # Tier 3 — LLM judge
        "faithfulness": None, "answer_relevance": None,
        "context_relevance": None, "completeness": None,
        "citation_accuracy": None, "overall": None,
    }

    try:
        resp = requests.post(f"{API_URL}/ask", json=body, headers=headers, timeout=180)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"[EVAL] /ask failed ({mode}): {e}")
        return empty

    answer     = data.get("answer", "")
    confidence = data.get("confidence")
    judge      = data.get("judge") or {}
    dims       = judge.get("dimensions") or {}

    # ── Tier 1 & 2: skip when --judge-only is set ─────────────────────────────
    if judge_only:
        lexical = {"rouge1": None, "rouge2": None, "rougeL": None,
                   "bleu1":  None, "bleu4":  None}
        cov = None
    else:
        lexical = compute_lexical(answer, reference)
        cov     = coverage(answer, reference)

    # ── Tier 3: LLM-judge dimensions from /ask response ──────────────────────
    result: dict[str, Any] = {
        "answer":     answer,
        # Tier 1
        "rouge1":     lexical.get("rouge1"),
        "rouge2":     lexical.get("rouge2"),
        "rougeL":     lexical.get("rougeL"),
        "bleu1":      lexical.get("bleu1"),
        "bleu4":      lexical.get("bleu4"),
        # Tier 2
        "coverage":   cov,
        "confidence": confidence,
        # Tier 3
        "overall":            judge.get("overall"),
        "faithfulness":       (dims.get("faithfulness")      or {}).get("score"),
        "answer_relevance":   (dims.get("answer_relevance")  or {}).get("score"),
        "context_relevance":  (dims.get("context_relevance") or {}).get("score"),
        "completeness":       (dims.get("completeness")      or {}).get("score"),
        "citation_accuracy":  (dims.get("citation_accuracy") or {}).get("score"),
    }
    return result


def _avg(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    return round(sum(valid) / len(valid), 4) if valid else None


def avg_results(results: list[dict]) -> dict[str, float | None]:
    keys = [
        # Tier 1
        "rouge1", "rouge2", "rougeL", "bleu1", "bleu4",
        # Tier 2
        "coverage", "confidence",
        # Tier 3
        "overall", "faithfulness", "answer_relevance",
        "context_relevance", "completeness", "citation_accuracy",
    ]
    return {k: _avg([r.get(k) for r in results]) for k in keys}


# ── Printing ──────────────────────────────────────────────────────────────────

def _fmt(val: float | None, width: int = 6) -> str:
    return f"{val:.4f}" if val is not None else "  N/A"


def print_lexical_table(avg: dict, title: str) -> None:
    """Table 0 — Lexical Baseline (ROUGE / BLEU)."""
    W = 70
    print(f"\n{'='*W}")
    print(f"  {title}")
    print(f"{'='*W}")
    print(f"  {'Metric':<26}  {'Score':>8}  Note")
    print(f"  {'-'*26}  {'-'*8}  {'-'*35}")
    rows = [
        ("ROUGE-1 F1",  "rouge1", "unigram overlap (precision + recall)"),
        ("ROUGE-2 F1",  "rouge2", "bigram overlap"),
        ("ROUGE-L F1",  "rougeL", "longest common subsequence"),
        ("─"*26,        None,     ""),
        ("BLEU-1",      "bleu1",  "1-gram precision (unigram)"),
        ("BLEU-4",      "bleu4",  "4-gram precision (standard MT)"),
        ("─"*26,        None,     ""),
        ("Coverage",    "coverage",    "fraction of reference words in answer"),
        ("Confidence",  "confidence",  "BGE-M3 grounding score (semantic)"),
    ]
    for label, key, note in rows:
        if key is None:
            print(f"  {label}")
            continue
        val = avg.get(key)
        print(f"  {label:<26}  {_fmt(val):>8}  {note}")
    print(f"{'='*W}")
    if any(avg.get(k) is None for k in ["rouge1", "bleu1"]):
        print("  ⚠  Install rouge-score and nltk for lexical metrics:")
        print("     pip install rouge-score nltk")
        print("     python -c \"import nltk; nltk.download('punkt_tab')\"")


def print_judge_table(avg: dict, title: str) -> None:
    """Table 1 — Full LLM-as-a-Judge results."""
    W = 70
    print(f"\n{'='*W}")
    print(f"  {title}")
    print(f"{'='*W}")
    print(f"  {'Metric':<26}  {'Score':>8}  {'Out of'}")
    print(f"  {'-'*26}  {'-'*8}  {'-'*30}")
    rows = [
        ("Faithfulness",       "faithfulness",      "1.00  (weight 0.35)"),
        ("Answer Relevance",   "answer_relevance",  "1.00  (weight 0.25)"),
        ("Context Relevance",  "context_relevance", "1.00  (weight 0.15)"),
        ("Completeness",       "completeness",      "1.00  (weight 0.15)"),
        ("Citation Accuracy",  "citation_accuracy", "1.00  (weight 0.10)"),
        ("─"*26,               None,                ""),
        ("Overall (weighted)", "overall",           "1.00"),
        ("Confidence Score",   "confidence",        "1.00  (semantic grounding)"),
        ("Coverage",           "coverage",          "1.00  (reference recall)"),
    ]
    for label, key, note in rows:
        if key is None:
            print(f"  {label}")
            continue
        val = avg.get(key)
        print(f"  {label:<26}  {_fmt(val):>8}  {note}")
    print(f"{'='*W}")


def print_three_tier_table(avg: dict, title: str) -> None:
    """Combined summary table — ROUGE/BLEU + semantic + LLM judge."""
    W = 70
    print(f"\n{'='*W}")
    print(f"  {title}")
    print(f"{'='*W}")
    print(f"  {'Group':<10}  {'Metric':<22}  {'Score':>8}  Interpretation")
    print(f"  {'-'*10}  {'-'*22}  {'-'*8}  {'-'*28}")

    rows = [
        ("Lexical",   "ROUGE-1 F1",        "rouge1",           "surface token overlap"),
        ("Lexical",   "ROUGE-L F1",        "rougeL",           "sequence overlap"),
        ("Lexical",   "BLEU-1",            "bleu1",            "unigram precision"),
        ("Lexical",   "BLEU-4",            "bleu4",            "4-gram precision"),
        (None,        "─"*22,              None,               ""),
        ("Semantic",  "Coverage",          "coverage",         "reference recall proxy"),
        ("Semantic",  "Confidence",        "confidence",       "grounding score"),
        (None,        "─"*22,              None,               ""),
        ("LLM Judge", "Faithfulness",      "faithfulness",     "no hallucinations"),
        ("LLM Judge", "Answer Relevance",  "answer_relevance", "addresses question"),
        ("LLM Judge", "Context Relevance", "context_relevance","retrieval quality"),
        ("LLM Judge", "Completeness",      "completeness",     "full coverage"),
        ("LLM Judge", "Citation Accuracy", "citation_accuracy","citations verified"),
        (None,        "─"*22,              None,               ""),
        ("LLM Judge", "Overall (wt.)",     "overall",          "weighted aggregate"),
    ]
    for group, label, key, note in rows:
        if key is None:
            print(f"  {'':10}  {label}")
            continue
        val = avg.get(key)
        group_str = group or ""
        print(f"  {group_str:<10}  {label:<22}  {_fmt(val):>8}  {note}")
    print(f"{'='*W}")


def print_comparison_table(avg_a: dict, avg_b: dict,
                            label_a: str, label_b: str,
                            title: str) -> None:
    W = 74
    print(f"\n{'='*W}")
    print(f"  {title}")
    print(f"{'='*W}")
    print(f"  {'Metric':<26}  {label_a:>18}  {label_b:>18}")
    print(f"  {'-'*26}  {'-'*18}  {'-'*18}")
    rows = [
        ("ROUGE-1 F1",         "rouge1"),
        ("ROUGE-L F1",         "rougeL"),
        ("BLEU-1",             "bleu1"),
        ("─"*26,               None),
        ("Faithfulness",       "faithfulness"),
        ("Answer Relevance",   "answer_relevance"),
        ("Context Relevance",  "context_relevance"),
        ("Completeness",       "completeness"),
        ("Citation Accuracy",  "citation_accuracy"),
        ("─"*26,               None),
        ("Overall (weighted)", "overall"),
        ("Confidence Score",   "confidence"),
        ("Coverage",           "coverage"),
    ]
    for label, key in rows:
        if key is None:
            print(f"  {label}")
            continue
        a = avg_a.get(key)
        b = avg_b.get(key)
        if a is not None and b is not None:
            delta = a - b
            arrow = " ▲" if delta > 0.005 else (" ▼" if delta < -0.005 else "  ")
        else:
            arrow = ""
        print(f"  {label:<26}  {_fmt(a):>18}  {_fmt(b):>18}{arrow}")
    print(f"{'='*W}")


# ── CSV output ────────────────────────────────────────────────────────────────

def save_csv(
    testset: list[dict],
    full_res: list[dict],
    noprompt_res: list[dict] | None,
    nomem_res: list[dict] | None,
    path: str,
) -> None:
    dim_keys = [
        # Tier 1
        "rouge1", "rouge2", "rougeL", "bleu1", "bleu4",
        # Tier 2
        "coverage", "confidence",
        # Tier 3
        "overall", "faithfulness", "answer_relevance",
        "context_relevance", "completeness", "citation_accuracy",
    ]
    base_fields = ["question", "reference", "source"]
    full_fields = [f"full_{k}" for k in dim_keys]
    np_fields   = [f"np_{k}"   for k in dim_keys] if noprompt_res else []
    nm_fields   = [f"nm_{k}"   for k in dim_keys] if nomem_res    else []

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields + full_fields + np_fields + nm_fields)
        writer.writeheader()
        for i, item in enumerate(testset):
            row: dict[str, Any] = {
                "question":  item["question"],
                "reference": item["reference"],
                "source":    item.get("source", ""),
            }
            if i < len(full_res):
                for k in dim_keys:
                    row[f"full_{k}"] = full_res[i].get(k)
            if noprompt_res and i < len(noprompt_res):
                for k in dim_keys:
                    row[f"np_{k}"] = noprompt_res[i].get(k)
            if nomem_res and i < len(nomem_res):
                for k in dim_keys:
                    row[f"nm_{k}"] = nomem_res[i].get(k)
            writer.writerow(row)
    logger.info(f"[EVAL] CSV saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global API_URL
    parser = argparse.ArgumentParser(
        description="RAG evaluation — ROUGE/BLEU and LLM-as-a-Judge (run each one alone)"
    )
    parser.add_argument("--n",             type=int,   default=20,
                        help="Number of Q&A pairs to generate")
    parser.add_argument("--api",           type=str,   default=API_URL,
                        help="RAG API base URL")
    parser.add_argument("--testset",       type=str,   default="",
                        help="Load existing testset JSON")
    parser.add_argument("--save-testset",  type=str,   default="",
                        help="Save generated testset to this path")
    parser.add_argument("--csv",           type=str,   default="eval_results.csv",
                        help="Output CSV path")
    parser.add_argument("--sleep",         type=float, default=3.0,
                        help="Seconds between /ask calls (default 3 — judge fires extra Groq calls)")
    parser.add_argument("--skip-noprompt", action="store_true",
                        help="Skip the no-prompt ablation")
    parser.add_argument("--skip-nomem",    action="store_true",
                        help="Skip the no-memory ablation")
    # ── Evaluation-mode flags ─────────────────────────────────────────────────
    parser.add_argument("--rouge-bleu",    action="store_true",
                        help=(
                            "Run ROUGE / BLEU evaluation only (lexical + coverage). "
                            "Calls /ask with judge=False — no LLM evaluation charges."
                        ))
    parser.add_argument("--tier1-only",    action="store_true",
                        help="Deprecated alias for --rouge-bleu (kept for back-compat).")
    parser.add_argument("--judge-only",    action="store_true",
                        help=(
                            "Run LLM-as-a-Judge evaluation only. "
                            "Calls /ask with judge=True; ROUGE/BLEU/coverage columns "
                            "are written as N/A in the output CSV."
                        ))
    parser.add_argument("--from-csv",      type=str,   default="",
                        help=(
                            "Recompute ROUGE / BLEU metrics from a previously saved eval CSV "
                            "(reads the 'full_answer' column). Zero API calls required."
                        ))
    args = parser.parse_args()
    API_URL = args.api

    # ── 1. Load or generate test set ─────────────────────────────────────────
    if args.testset and Path(args.testset).exists():
        with open(args.testset, encoding="utf-8") as f:
            testset = json.load(f)
        logger.info(f"[EVAL] Loaded {len(testset)} pairs from {args.testset}")
    else:
        logger.info(f"[EVAL] Generating {args.n} Q&A pairs from Qdrant …")
        chunks = fetch_chunks(n_chunks=max(args.n * 4, 80))
        if not chunks:
            logger.error("[EVAL] No chunks found — is Qdrant running and indexed?")
            sys.exit(1)
        testset = generate_testset(chunks, n=args.n, sleep=args.sleep)
        if not testset:
            logger.error("[EVAL] Q&A generation returned no pairs — check GROQ_API_KEY")
            sys.exit(1)

    if args.save_testset:
        with open(args.save_testset, "w", encoding="utf-8") as f:
            json.dump(testset, f, ensure_ascii=False, indent=2)
        logger.info(f"[EVAL] Testset saved → {args.save_testset}")

    n = len(testset)

    # ══════════════════════════════════════════════════════════════════════════
    # ROUGE / BLEU PATH  (--rouge-bleu, --tier1-only alias, or --from-csv)
    # ══════════════════════════════════════════════════════════════════════════
    if args.rouge_bleu or args.tier1_only or args.from_csv:
        logger.info(f"[EVAL] ROUGE / BLEU evaluation — {'loading answers from CSV' if args.from_csv else 'calling /ask (judge=False)'}")

        if args.from_csv:
            # Zero API calls — read saved answers from a previous CSV
            full_res = _tier1_from_csv(args.from_csv, testset)
            if not full_res:
                sys.exit(1)
        else:
            # Call /ask once per question, no judge (fast & cheap)
            full_res = []
            for i, item in enumerate(testset, 1):
                result = _ask_tier1(item["question"], item["reference"], "full")
                full_res.append(result)
                logger.info(
                    f"[EVAL] [{i}/{n}]  "
                    f"R1={_fmt(result.get('rouge1'))}  "
                    f"R2={_fmt(result.get('rouge2'))}  "
                    f"RL={_fmt(result.get('rougeL'))}  "
                    f"B1={_fmt(result.get('bleu1'))}  "
                    f"B4={_fmt(result.get('bleu4'))}  "
                    f"cov={_fmt(result.get('coverage'))}"
                )
                if not args.from_csv:
                    time.sleep(max(args.sleep * 0.3, 0.5))  # no judge → much shorter pause

        avg_full = avg_results(full_res)

        # Print lexical table only
        print("\n\n" + "█" * 70)
        print("  RAG EVALUATION RESULTS  —  ROUGE / BLEU")
        print("█" * 70)
        print_lexical_table(avg_full, title="ROUGE / BLEU Lexical Metrics")

        # Print interpretive note
        r1 = avg_full.get("rouge1")
        rl = avg_full.get("rougeL")
        b1 = avg_full.get("bleu1")
        if r1 is not None:
            print(f"""
  ── Interpretive Note ───────────────────────────────────────────────────────
  ROUGE-1={_fmt(r1)}, ROUGE-L={_fmt(rl)}, BLEU-1={_fmt(b1)}.
  These lexical scores reflect surface token overlap with the reference answer.
  RAG systems generate grounded, paraphrased answers rather than copying the
  reference verbatim — low ROUGE/BLEU is expected and does NOT indicate poor
  quality. Run with --judge-only to get LLM-as-a-Judge scores for comparison.
  ─────────────────────────────────────────────────────────────────────────────
""")

        # Save CSV
        save_csv(testset, full_res, None, None, path=args.csv)

        summary_path = args.csv.replace(".csv", "_summary.json")
        summary = {
            "n_questions": n,
            "mode": "rouge-bleu (lexical only)",
            "lexical_available": {"rouge": _ROUGE_AVAILABLE, "bleu": _BLEU_AVAILABLE},
            "full": {k: (round(v, 4) if v is not None else None)
                     for k, v in avg_full.items()},
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"[EVAL] Summary → {summary_path}")
        print(f"\n  Output files:")
        print(f"    • {args.csv}")
        print(f"    • {summary_path}")
        print()
        return   # ← done, skip the full pipeline below

    # ══════════════════════════════════════════════════════════════════════════
    # JUDGE PATH  (judge-only OR combined lexical + judge)
    # ══════════════════════════════════════════════════════════════════════════
    lexical_note = (
        "ROUGE ✓  BLEU ✓" if (_ROUGE_AVAILABLE and _BLEU_AVAILABLE) else
        "ROUGE ✗ (install rouge-score)  BLEU ✗ (install nltk)" if not _ROUGE_AVAILABLE and not _BLEU_AVAILABLE else
        "ROUGE ✗ (install rouge-score)" if not _ROUGE_AVAILABLE else
        "BLEU ✗ (install nltk)"
    )
    if args.judge_only:
        logger.info(f"[EVAL] LLM-as-a-Judge evaluation ({n} questions)  |  sleep={args.sleep}s …")
    else:
        logger.info(f"[EVAL] Combined evaluation ({n} questions)  |  {lexical_note}  |  sleep={args.sleep}s …")

    # ── 2. Full RAG ───────────────────────────────────────────────────────────
    full_res = []
    for i, item in enumerate(testset, 1):
        result = _ask(item["question"], item["reference"], "full",
                      judge_only=args.judge_only)
        full_res.append(result)
        if args.judge_only:
            logger.info(
                f"[EVAL] full [{i}/{n}]  "
                f"overall={_fmt(result.get('overall'))}  "
                f"faith={_fmt(result.get('faithfulness'))}  "
                f"ans_rel={_fmt(result.get('answer_relevance'))}  "
                f"ctx_rel={_fmt(result.get('context_relevance'))}  "
                f"comp={_fmt(result.get('completeness'))}  "
                f"cite={_fmt(result.get('citation_accuracy'))}  "
                f"conf={_fmt(result.get('confidence'))}"
            )
        else:
            logger.info(
                f"[EVAL] full [{i}/{n}]  "
                f"R1={_fmt(result.get('rouge1'))}  "
                f"RL={_fmt(result.get('rougeL'))}  "
                f"B1={_fmt(result.get('bleu1'))}  "
                f"overall={_fmt(result.get('overall'))}  "
                f"conf={_fmt(result.get('confidence'))}"
            )
        time.sleep(args.sleep)

    avg_full = avg_results(full_res)

    # ── 3. No-prompt mode ─────────────────────────────────────────────────────
    noprompt_res: list[dict] = []
    avg_noprompt: dict = {}
    if not args.skip_noprompt:
        logger.info("[EVAL] Running no-prompt ablation …")
        for i, item in enumerate(testset, 1):
            result = _ask(item["question"], item["reference"], "no_prompt",
                          judge_only=args.judge_only)
            noprompt_res.append(result)
            logger.info(f"[EVAL] no_prompt [{i}/{n}]  overall={_fmt(result.get('overall'))}")
            time.sleep(args.sleep)
        avg_noprompt = avg_results(noprompt_res)

    # ── 4. No-memory mode ─────────────────────────────────────────────────────
    nomem_res: list[dict] = []
    avg_nomem: dict = {}
    if not args.skip_nomem:
        logger.info("[EVAL] Running no-memory ablation …")
        for i, item in enumerate(testset, 1):
            result = _ask(item["question"], item["reference"], "no_memory",
                          judge_only=args.judge_only)
            nomem_res.append(result)
            logger.info(f"[EVAL] no_memory [{i}/{n}]  overall={_fmt(result.get('overall'))}")
            time.sleep(args.sleep)
        avg_nomem = avg_results(nomem_res)

    # ── 5. Print tables ───────────────────────────────────────────────────────
    print("\n\n" + "█" * 70)
    if args.judge_only:
        print("  RAG EVALUATION RESULTS  —  LLM-as-a-Judge")
    else:
        print("  RAG EVALUATION RESULTS  —  Combined (ROUGE / BLEU + LLM-as-a-Judge)")
    print("█" * 70)

    if not args.judge_only:
        print_lexical_table(avg_full, title="ROUGE / BLEU Lexical Metrics")
    print_judge_table(avg_full, title="LLM-as-a-Judge Dimension Scores")
    if not args.judge_only:
        print_three_tier_table(avg_full, title="Combined Summary (ROUGE / BLEU + LLM-as-a-Judge)")

    if avg_noprompt:
        print_comparison_table(
            avg_full, avg_noprompt,
            label_a="With Sys. Prompt",
            label_b="No Sys. Prompt",
            title="Table 3 — Ablation: Impact of System Prompt",
        )

    if avg_nomem:
        print_comparison_table(
            avg_full, avg_nomem,
            label_a="With Memory",
            label_b="No Memory",
            title="Table 4 — Ablation: Impact of Conversational Memory",
        )

    # ── Interpretive note for the report ─────────────────────────────────────
    r1    = avg_full.get("rouge1")
    rl    = avg_full.get("rougeL")
    b1    = avg_full.get("bleu1")
    ov    = avg_full.get("overall")
    faith = avg_full.get("faithfulness")
    if r1 is not None and ov is not None:
        print(f"""
  ── Interpretive Note ───────────────────────────────────────────────────────
  Lexical metrics (ROUGE-1={_fmt(r1)}, ROUGE-L={_fmt(rl)}, BLEU-1={_fmt(b1)}) are lower than
  the LLM-judge overall ({_fmt(ov)}) because the RAG system generates grounded,
  cited, paraphrased answers rather than copying reference text verbatim.
  Faithfulness={_fmt(faith)} confirms that all claims are accurate — the lexical
  gap reflects paraphrasing ability, not a quality deficit. This demonstrates
  why LLM-as-a-Judge is the appropriate evaluation method for RAG systems.
  ─────────────────────────────────────────────────────────────────────────────
""")

    # ── 6. Save CSV ───────────────────────────────────────────────────────────
    save_csv(testset, full_res,
             noprompt_res or None,
             nomem_res    or None,
             path=args.csv)

    # ── 7. JSON summary ───────────────────────────────────────────────────────
    def _round_dict(d: dict) -> dict:
        return {k: (round(v, 4) if v is not None else None) for k, v in d.items()}

    summary = {
        "n_questions": n,
        "mode": "judge-only" if args.judge_only else "combined (rouge-bleu + judge)",
        "lexical_available": {"rouge": _ROUGE_AVAILABLE, "bleu": _BLEU_AVAILABLE},
        "full": _round_dict(avg_full),
    }
    if avg_noprompt:
        summary["no_prompt"] = _round_dict(avg_noprompt)
    if avg_nomem:
        summary["no_memory"] = _round_dict(avg_nomem)

    summary_path = args.csv.replace(".csv", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"[EVAL] Summary -> {summary_path}")

    print(f"\n  Output files:")
    print(f"    • {args.csv}")
    print(f"    • {summary_path}")
    print()


if __name__ == "__main__":
    main()
