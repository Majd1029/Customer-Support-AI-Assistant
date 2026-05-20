"""
evaluate_rag.py — Full RAG evaluation suite for the internship report.

Pipeline:
  1. Pull document chunks from Qdrant (or load a saved test set)
  2. Generate Q&A pairs via Groq (question + reference answer)
  3. Run each question through the RAG /ask endpoint in 3 modes:
       • full  — system prompt + retrieval (standard)
       • no_prompt — no persona/system prompt, retrieval only
       • no_memory — no conversation memory or query rewriting
  4. Compute: EM, ESM, BLEU, ROUGE-1 F1, ROUGE-2 F1, ROUGE-L F1
  5. Print formatted tables + save CSV for Excel/report

Usage:
    # Full run — generates 20 Q&A pairs then evaluates all modes
    python scripts/evaluate_rag.py

    # Use a saved test set (skip generation)
    python scripts/evaluate_rag.py --testset eval_testset.json

    # Control number of pairs and API endpoint
    python scripts/evaluate_rag.py --n 30 --api http://localhost:8000

    # Save generated test set for reuse
    python scripts/evaluate_rag.py --save-testset eval_testset.json

Requirements (already in requirements.txt):
    pip install nltk rouge-score requests qdrant-client
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
"""

from __future__ import annotations

import argparse
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

# ── Config ────────────────────────────────────────────────────────────────────

API_URL        = os.getenv("EVAL_API_URL", "http://localhost:8000")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_GEN_MODEL = os.getenv("GROQ_GENERATION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_QA_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"   # used for Q&A generation
QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION     = "documents"

ASK_TOKEN      = os.getenv("EVAL_JWT_TOKEN", "")   # set if your /ask endpoint requires auth


# ── Metric helpers ────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lower-case, strip accents, collapse whitespace, remove punctuation."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(ref) else 0.0


def exact_set_match(pred: str, ref: str) -> float:
    """Bag-of-words Jaccard: |pred_words ∩ ref_words| / |pred_words ∪ ref_words|"""
    p_words = set(_normalize(pred).split())
    r_words = set(_normalize(ref).split())
    if not p_words and not r_words:
        return 1.0
    if not p_words or not r_words:
        return 0.0
    return len(p_words & r_words) / len(p_words | r_words)


def bleu_score(pred: str, ref: str) -> float:
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize
        hyp = word_tokenize(_normalize(pred))
        ref_tokens = [word_tokenize(_normalize(ref))]
        if not hyp:
            return 0.0
        sf = SmoothingFunction().method1
        return sentence_bleu(ref_tokens, hyp, smoothing_function=sf)
    except LookupError:
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        return bleu_score(pred, ref)


def rouge_scores(pred: str, ref: str) -> dict[str, float]:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        scores = scorer.score(_normalize(ref), _normalize(pred))
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }
    except ImportError:
        logger.warning("rouge-score not installed: pip install rouge-score")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def case_insensitive_accuracy(pred: str, ref: str) -> float:
    return 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0


def extended_match(pred: str, ref: str) -> float:
    """Reference words covered by prediction (recall-oriented)."""
    p_words = set(_normalize(pred).split())
    r_words = set(_normalize(ref).split())
    if not r_words:
        return 1.0
    return len(p_words & r_words) / len(r_words)


def compute_all_metrics(pred: str, ref: str) -> dict[str, float]:
    r = rouge_scores(pred, ref)
    return {
        "EM":      exact_match(pred, ref),
        "ESM":     exact_set_match(pred, ref),
        "BLEU":    bleu_score(pred, ref),
        "ROUGE-1": r["rouge1"],
        "ROUGE-2": r["rouge2"],
        "ROUGE-L": r["rougeL"],
        "CI_ACC":  case_insensitive_accuracy(pred, ref),
        "EXT_ACC": extended_match(pred, ref),
    }


def avg_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    if not results:
        return {}
    keys = results[0].keys()
    return {k: sum(r[k] for r in results) / len(results) for k in keys}


# ── Step 1: pull chunks from Qdrant ──────────────────────────────────────────

def fetch_chunks(n_chunks: int = 60) -> list[dict]:
    """Scroll Qdrant and return up to n_chunks text chunks with content."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        results, _ = client.scroll(
            collection_name = COLLECTION,
            limit           = n_chunks,
            with_payload    = True,
            with_vectors    = False,
        )
        chunks = []
        for pt in results:
            p = pt.payload or {}
            content = p.get("content", "")
            if len(content.split()) >= 30:   # skip tiny chunks
                chunks.append({
                    "content": content[:1500],
                    "source":  p.get("source", "unknown"),
                    "section": p.get("section", ""),
                })
        logger.info(f"[EVAL] Fetched {len(chunks)} chunks from Qdrant")
        return chunks
    except Exception as e:
        logger.error(f"[EVAL] Qdrant unavailable: {e}")
        return []


# ── Step 2: generate Q&A pairs via Groq ──────────────────────────────────────

_QA_SYSTEM = (
    "You are an exam question writer. "
    "Given a document excerpt, generate exactly ONE question that can be answered "
    "directly and completely from that excerpt alone, plus the correct reference answer. "
    "Rules: the question must be answerable in 1–3 sentences from the excerpt; "
    "avoid yes/no questions; avoid questions about document metadata. "
    'Respond ONLY with valid JSON: {"question": "...", "answer": "..."}'
)


def _call_groq_qa(content: str) -> dict | None:
    if not GROQ_API_KEY:
        logger.error("[EVAL] GROQ_API_KEY not set — cannot generate Q&A pairs")
        return None
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model       = GROQ_QA_MODEL,
            messages    = [
                {"role": "system", "content": _QA_SYSTEM},
                {"role": "user",   "content": f"Document excerpt:\n\n{content}"},
            ],
            max_tokens  = 256,
            temperature = 0.3,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip Qwen3/Llama thinking blocks if any
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
        # Strip markdown fences
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"[EVAL] Q&A generation failed: {e}")
        return None


def generate_testset(chunks: list[dict], n: int = 20, sleep: float = 1.5) -> list[dict]:
    """Generate up to n Q&A pairs from the chunk list."""
    pairs: list[dict] = []
    import random
    random.shuffle(chunks)
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
        time.sleep(sleep)   # respect Groq TPM
    logger.info(f"[EVAL] Generated {len(pairs)} Q&A pairs")
    return pairs


# ── Step 3: query /ask in different modes ─────────────────────────────────────

def _ask(question: str, mode: str) -> str:
    """
    Call POST /ask/stream (or /ask) and return the answer text.

    mode:
      "full"      — standard RAG with system prompt + retrieval
      "no_prompt" — cite_sources=False, no persona
      "no_memory" — memory_enabled=False (no rewriting)
    """
    body: dict[str, Any] = {
        "question":       question,
        "limit":          5,
        "rerank":         False,
        "use_hyde":       False,
        "mmr":            False,
        "multi_hop":      False,
        "memory_enabled": False,
        "judge":          False,
        "max_tokens":     512,
    }

    if mode == "full":
        body["support_mode"] = False
    elif mode == "no_prompt":
        body["persona"] = " "    # blank persona = no CS persona
        # cite_sources handled server-side when persona is blank
    elif mode == "no_memory":
        body["memory_enabled"] = False

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if ASK_TOKEN:
        headers["Authorization"] = f"Bearer {ASK_TOKEN}"

    try:
        resp = requests.post(
            f"{API_URL}/ask",
            json    = body,
            headers = headers,
            timeout = 120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("answer", "")
    except Exception as e:
        logger.warning(f"[EVAL] /ask failed ({mode}): {e}")
        return ""


# ── Step 4 & 5: evaluate + print tables ──────────────────────────────────────

def _table_line(label: str, val: float, width: int = 22) -> str:
    return f"  {label:<{width}}  {val:.2f}"


def print_single_table(avg: dict[str, float], title: str = "Results") -> None:
    print(f"\n{'='*45}")
    print(f"  {title}")
    print(f"{'='*45}")
    print(f"  {'Metric':<22}  {'Score':>6}")
    print(f"  {'-'*22}  {'-'*6}")
    rows = [
        ("Exact Match (EM)",          avg.get("EM",      0)),
        ("Exact Set Match (ESM)",     avg.get("ESM",     0)),
        ("BLEU",                      avg.get("BLEU",    0)),
        ("ROUGE-1 F1",                avg.get("ROUGE-1", 0)),
        ("ROUGE-2 F1",                avg.get("ROUGE-2", 0)),
        ("ROUGE-L F1",                avg.get("ROUGE-L", 0)),
    ]
    for label, val in rows:
        print(_table_line(label, val))
    print(f"{'='*45}")


def print_accuracy_table(avg: dict[str, float], title: str = "Accuracy") -> None:
    print(f"\n{'='*45}")
    print(f"  {title}")
    print(f"{'='*45}")
    print(f"  {'Metric':<35}  {'Score':>6}")
    print(f"  {'-'*35}  {'-'*6}")
    rows = [
        ("Exact Match Accuracy",                avg.get("EM",      0) * 100),
        ("Case-Insensitive Accuracy",            avg.get("CI_ACC",  0) * 100),
        ("Extended Match Accuracy",              avg.get("EXT_ACC", 0) * 100),
    ]
    for label, val in rows:
        print(f"  {label:<35}  {val:.2f}")
    print(f"{'='*45}")


def print_comparison_table(
    avg_a: dict[str, float],
    avg_b: dict[str, float],
    label_a: str = "With prompt",
    label_b: str = "Without prompt",
    title: str   = "Comparison",
) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  {'Metric':<22}  {label_a:>14}  {label_b:>14}")
    print(f"  {'-'*22}  {'-'*14}  {'-'*14}")
    rows = [
        ("Exact Match (EM)",     "EM"),
        ("Exact Set Match (ESM)","ESM"),
        ("BLEU",                 "BLEU"),
        ("ROUGE-1 F1",           "ROUGE-1"),
        ("ROUGE-2 F1",           "ROUGE-2"),
        ("ROUGE-L F1",           "ROUGE-L"),
    ]
    for label, key in rows:
        a = avg_a.get(key, 0)
        b = avg_b.get(key, 0)
        print(f"  {label:<22}  {a:>14.2f}  {b:>14.2f}")
    print(f"{'='*60}")


def save_csv(
    testset:      list[dict],
    full_results: list[dict[str, float]],
    noprompt_results: list[dict[str, float]] | None = None,
    nomem_results:    list[dict[str, float]] | None = None,
    path: str = "eval_results.csv",
) -> None:
    import csv
    fieldnames = ["question", "reference", "source",
                  "full_EM", "full_ESM", "full_BLEU", "full_ROUGE1", "full_ROUGE2", "full_ROUGEL"]
    if noprompt_results:
        fieldnames += ["np_EM", "np_ESM", "np_BLEU", "np_ROUGE1", "np_ROUGE2", "np_ROUGEL"]
    if nomem_results:
        fieldnames += ["nm_EM", "nm_ESM", "nm_BLEU", "nm_ROUGE1", "nm_ROUGE2", "nm_ROUGEL"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, item in enumerate(testset):
            row: dict[str, Any] = {
                "question":  item["question"],
                "reference": item["reference"],
                "source":    item.get("source", ""),
            }
            if i < len(full_results):
                m = full_results[i]
                row.update({
                    "full_EM": m["EM"], "full_ESM": m["ESM"], "full_BLEU": m["BLEU"],
                    "full_ROUGE1": m["ROUGE-1"], "full_ROUGE2": m["ROUGE-2"], "full_ROUGEL": m["ROUGE-L"],
                })
            if noprompt_results and i < len(noprompt_results):
                m = noprompt_results[i]
                row.update({
                    "np_EM": m["EM"], "np_ESM": m["ESM"], "np_BLEU": m["BLEU"],
                    "np_ROUGE1": m["ROUGE-1"], "np_ROUGE2": m["ROUGE-2"], "np_ROUGEL": m["ROUGE-L"],
                })
            if nomem_results and i < len(nomem_results):
                m = nomem_results[i]
                row.update({
                    "nm_EM": m["EM"], "nm_ESM": m["ESM"], "nm_BLEU": m["BLEU"],
                    "nm_ROUGE1": m["ROUGE-1"], "nm_ROUGE2": m["ROUGE-2"], "nm_ROUGEL": m["ROUGE-L"],
                })
            writer.writerow(row)
    logger.info(f"[EVAL] Saved per-question CSV → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG evaluation for the internship report")
    parser.add_argument("--n",             type=int,  default=20,              help="Number of Q&A pairs to generate")
    parser.add_argument("--api",           type=str,  default=API_URL,         help="RAG API base URL")
    parser.add_argument("--testset",       type=str,  default="",              help="Load existing testset JSON instead of generating")
    parser.add_argument("--save-testset",  type=str,  default="",              help="Save generated testset to this path")
    parser.add_argument("--csv",           type=str,  default="eval_results.csv", help="Output CSV path")
    parser.add_argument("--sleep",         type=float, default=2.0,            help="Seconds between /ask calls (avoid TPM)")
    parser.add_argument("--skip-noprompt", action="store_true",                help="Skip the no-prompt comparison run")
    parser.add_argument("--skip-nomem",    action="store_true",                help="Skip the no-memory comparison run")
    args = parser.parse_args()

    global API_URL
    API_URL = args.api

    # ── 1. Load or generate test set ─────────────────────────────────────────
    if args.testset and Path(args.testset).exists():
        with open(args.testset, encoding="utf-8") as f:
            testset = json.load(f)
        logger.info(f"[EVAL] Loaded {len(testset)} pairs from {args.testset}")
    else:
        logger.info(f"[EVAL] Fetching chunks from Qdrant and generating {args.n} Q&A pairs …")
        chunks  = fetch_chunks(n_chunks=max(args.n * 4, 80))
        if not chunks:
            logger.error("[EVAL] No chunks found — is Qdrant running and the collection indexed?")
            sys.exit(1)
        testset = generate_testset(chunks, n=args.n, sleep=args.sleep)
        if not testset:
            logger.error("[EVAL] Q&A generation returned no pairs — check GROQ_API_KEY")
            sys.exit(1)

    if args.save_testset:
        with open(args.save_testset, "w", encoding="utf-8") as f:
            json.dump(testset, f, ensure_ascii=False, indent=2)
        logger.info(f"[EVAL] Saved testset → {args.save_testset}")

    n = len(testset)
    logger.info(f"[EVAL] Evaluating {n} questions across modes …")

    # ── 2. Full RAG ───────────────────────────────────────────────────────────
    full_results: list[dict[str, float]] = []
    for i, item in enumerate(testset, 1):
        pred = _ask(item["question"], "full")
        metrics = compute_all_metrics(pred, item["reference"])
        full_results.append(metrics)
        logger.info(f"[EVAL] full [{i}/{n}] EM={metrics['EM']:.0f}  ROUGE-L={metrics['ROUGE-L']:.2f}")
        time.sleep(args.sleep)

    avg_full = avg_metrics(full_results)

    # ── 3. No-prompt mode ─────────────────────────────────────────────────────
    avg_noprompt: dict[str, float] = {}
    noprompt_results: list[dict[str, float]] = []
    if not args.skip_noprompt:
        logger.info("[EVAL] Running no-prompt mode …")
        for i, item in enumerate(testset, 1):
            pred = _ask(item["question"], "no_prompt")
            metrics = compute_all_metrics(pred, item["reference"])
            noprompt_results.append(metrics)
            logger.info(f"[EVAL] no_prompt [{i}/{n}] ROUGE-L={metrics['ROUGE-L']:.2f}")
            time.sleep(args.sleep)
        avg_noprompt = avg_metrics(noprompt_results)

    # ── 4. No-memory mode ─────────────────────────────────────────────────────
    avg_nomem: dict[str, float] = {}
    nomem_results: list[dict[str, float]] = []
    if not args.skip_nomem:
        logger.info("[EVAL] Running no-memory mode …")
        for i, item in enumerate(testset, 1):
            pred = _ask(item["question"], "no_memory")
            metrics = compute_all_metrics(pred, item["reference"])
            nomem_results.append(metrics)
            logger.info(f"[EVAL] no_memory [{i}/{n}] ROUGE-L={metrics['ROUGE-L']:.2f}")
            time.sleep(args.sleep)
        avg_nomem = avg_metrics(nomem_results)

    # ── 5. Print tables ───────────────────────────────────────────────────────
    print("\n\n" + "█" * 60)
    print("  RAG EVALUATION RESULTS")
    print("█" * 60)

    print_single_table(avg_full,  title="Table 1 — Full RAG System (NLP Metrics)")
    print_accuracy_table(avg_full, title="Table 2 — Accuracy Metrics")

    if avg_noprompt:
        print_comparison_table(
            avg_full, avg_noprompt,
            label_a = "With prompt",
            label_b = "Without prompt",
            title   = "Table 3 — Impact of System Prompt",
        )

    if avg_nomem:
        print_comparison_table(
            avg_full, avg_nomem,
            label_a = "With memory",
            label_b = "Without memory",
            title   = "Table 4 — Impact of Memory / Query Rewriting",
        )

    # ── 6. Save CSV ───────────────────────────────────────────────────────────
    save_csv(
        testset,
        full_results,
        noprompt_results or None,
        nomem_results    or None,
        path = args.csv,
    )

    # ── 7. JSON summary ───────────────────────────────────────────────────────
    summary = {
        "n_questions": n,
        "full":        {k: round(v, 4) for k, v in avg_full.items()},
    }
    if avg_noprompt:
        summary["no_prompt"] = {k: round(v, 4) for k, v in avg_noprompt.items()}
    if avg_nomem:
        summary["no_memory"] = {k: round(v, 4) for k, v in avg_nomem.items()}

    summary_path = args.csv.replace(".csv", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"[EVAL] Summary JSON → {summary_path}")

    print(f"\n  Output files:")
    print(f"    • {args.csv}           (per-question detail)")
    print(f"    • {summary_path}  (aggregated averages)")
    print()


if __name__ == "__main__":
    main()
