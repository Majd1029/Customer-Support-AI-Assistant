"""
langsmith_eval.py — LangSmith-backed evaluation for CustomerAssist.

What this does
──────────────
1. Connects to LangSmith and creates (or reuses) a dataset named
   LANGSMITH_DATASET (default: "customerassist-golden-set").
2. Optionally seeds the dataset from a local testset JSON file
   (same format used by evaluate_rag.py).
3. Runs the /ask endpoint against every example in the dataset.
4. Scores each run with the existing 6-dimension LLM-as-a-Judge
   (judge.py) as a custom LangSmith evaluator — results appear in
   the LangSmith Experiments UI for comparison across runs.

Prerequisites
─────────────
    pip install langsmith requests groq python-dotenv loguru

Environment variables (.env)
─────────────────────────────
    LANGCHAIN_API_KEY=ls__...          # LangSmith API key
    LANGCHAIN_PROJECT=customer-assist  # optional, defaults below
    LANGSMITH_DATASET=customerassist-golden-set  # dataset name
    GROQ_EVAL_API_KEY=gsk_...          # dedicated eval key (own TPM pool)
    EVAL_API_URL=http://localhost:8000  # RAG server
    EVAL_JWT_TOKEN=...                 # JWT if /ask requires auth

Usage
─────
    # Run eval against existing dataset (most common)
    python scripts/langsmith_eval.py

    # Seed dataset from a local testset JSON first, then eval
    python scripts/langsmith_eval.py --seed eval_testset.json

    # Name a specific experiment (shows in LangSmith UI)
    python scripts/langsmith_eval.py --experiment v2-langgraph-pipeline

    # Dry-run: seed only, skip evaluation
    python scripts/langsmith_eval.py --seed eval_testset.json --seed-only

    # Limit to first N examples (useful for quick smoke-tests)
    python scripts/langsmith_eval.py --n 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)

# ── Config ─────────────────────────────────────────────────────────────────────

LANGSMITH_API_KEY  = os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT  = os.getenv("LANGCHAIN_PROJECT", "customer-assist")
LANGSMITH_DATASET  = os.getenv("LANGSMITH_DATASET", "customerassist-golden-set")
GROQ_EVAL_API_KEY  = (
    os.getenv("GROQ_EVAL_API_KEY", "")
    or os.getenv("JUDGE_API_KEY", "")
    or os.getenv("GROQ_API_KEY", "")
)
API_URL            = os.getenv("EVAL_API_URL", "http://localhost:8000")
ASK_TOKEN          = os.getenv("EVAL_JWT_TOKEN", "")
SLEEP_BETWEEN_RUNS = float(os.getenv("EVAL_SLEEP", "3.0"))  # seconds


# ── RAG pipeline wrapper ────────────────────────────────────────────────────────

def _ask(question: str) -> dict[str, Any]:
    """Call POST /ask and return the full response dict."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if ASK_TOKEN:
        headers["Authorization"] = f"Bearer {ASK_TOKEN}"

    body = {
        "question":       question,
        "limit":          5,
        "rerank":         False,   # keep off during eval — reranker may cause timeouts on CPU
        "use_hyde":       False,
        "mmr":            False,
        "multi_hop":      False,
        "memory_enabled": False,
        "judge":          False,   # LangSmith evaluator runs judge separately
        "max_tokens":     1024,    # 512 truncated answers before citations were written
        "include_chunks": True,    # return chunk content so context_relevance can be scored
    }

    try:
        resp = requests.post(
            f"{API_URL}/ask", json=body, headers=headers, timeout=180
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning(f"[LS] /ask failed: {exc}")
        return {"answer": "", "sources": [], "chunks_in_context": []}


# ── Custom LangSmith evaluator (wraps judge.py) ─────────────────────────────────

def _make_evaluators() -> list:
    """
    Return a list of LangSmith evaluator functions.

    Each evaluator receives:
      run    — LangSmith Run object  (run.outputs = what _ask() returned)
      example — LangSmith Example   (example.inputs["question"],
                                     example.outputs["reference"])
    and returns {"key": str, "score": float}.

    We produce one evaluator per judge dimension + one for overall.
    """
    # Import here so the script doesn't crash if judge deps are missing
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from file_preparation.evaluation.judge import judge_answer
        from groq import Groq
        groq_client = Groq(api_key=GROQ_EVAL_API_KEY) if GROQ_EVAL_API_KEY else None
    except ImportError as exc:
        logger.error(f"[LS] Cannot import judge: {exc}")
        return []

    def _run_judge(run, example):
        """Core: call judge_answer() and cache result on run object."""
        # Cache so we don't call Groq 6× per example
        if hasattr(run, "_judge_cache"):
            return run._judge_cache  # type: ignore[attr-defined]

        question  = example.inputs.get("question", "")
        reference = (example.outputs or {}).get("reference")
        answer    = (run.outputs or {}).get("answer", "")
        # chunks_in_context is now returned by /ask when include_chunks=True
        # It's a list of {chunk_id, content, source, page_start, section, score}
        chunks    = (run.outputs or {}).get("chunks_in_context") or []

        try:
            result = judge_answer(
                question  = question,
                answer    = answer,
                chunks    = chunks,
                reference = reference,
                no_answer = not bool(answer.strip()),
            )
        except Exception as exc:
            logger.warning(f"[LS] judge_answer failed: {exc}")
            result = None

        run._judge_cache = result  # type: ignore[attr-defined]
        return result

    # ── One evaluator function per dimension ────────────────────────────────────

    def overall(run, example):
        r = _run_judge(run, example)
        return {"key": "overall", "score": r.overall if r else None}

    def faithfulness(run, example):
        r = _run_judge(run, example)
        s = r.faithfulness.score if r else None
        return {"key": "faithfulness", "score": s}

    def answer_relevance(run, example):
        r = _run_judge(run, example)
        s = r.answer_relevance.score if r else None
        return {"key": "answer_relevance", "score": s}

    def context_relevance(run, example):
        r = _run_judge(run, example)
        s = r.context_relevance.score if r else None
        return {"key": "context_relevance", "score": s}

    def completeness(run, example):
        r = _run_judge(run, example)
        s = r.completeness.score if r else None
        return {"key": "completeness", "score": s}

    def citation_accuracy(run, example):
        r = _run_judge(run, example)
        s = r.citation_accuracy.score if r else None
        return {"key": "citation_accuracy", "score": s}

    def correctness(run, example):
        """Only meaningful when the example has a reference answer."""
        r = _run_judge(run, example)
        if r is None:
            return {"key": "correctness", "score": None}
        s = r.correctness.score if r.correctness else None
        return {"key": "correctness", "score": s}

    def confidence(run, example):
        score = (run.outputs or {}).get("confidence")
        return {"key": "confidence", "score": score}

    return [
        overall, faithfulness, answer_relevance, context_relevance,
        completeness, citation_accuracy, correctness, confidence,
    ]


# ── Dataset helpers ─────────────────────────────────────────────────────────────

def _get_or_create_dataset(client, name: str):
    """Return an existing dataset or create a new empty one."""
    try:
        return client.read_dataset(dataset_name=name)
    except Exception:
        logger.info(f"[LS] Creating dataset '{name}' …")
        return client.create_dataset(dataset_name=name)


def _seed_dataset(client, dataset, testset_path: str) -> int:
    """
    Add examples from a local JSON testset to the LangSmith dataset.
    Skips duplicates (checked by question text).
    Returns the number of new examples added.
    """
    with open(testset_path, encoding="utf-8") as f:
        testset = json.load(f)

    # Collect existing questions to avoid duplicates
    existing = {
        ex.inputs.get("question", "")
        for ex in client.list_examples(dataset_id=dataset.id)
    }

    new_examples = [
        item for item in testset
        if item.get("question") not in existing
    ]

    if not new_examples:
        logger.info("[LS] All examples already in dataset — nothing to seed.")
        return 0

    client.create_examples(
        inputs  = [{"question": item["question"]} for item in new_examples],
        outputs = [{"reference": item.get("reference", "")} for item in new_examples],
        dataset_id = dataset.id,
    )
    logger.info(f"[LS] Seeded {len(new_examples)} new examples into '{dataset.name}'.")
    return len(new_examples)


# ── Pipeline target function ────────────────────────────────────────────────────

def rag_pipeline(inputs: dict) -> dict:
    """
    The function LangSmith calls for each dataset example.
    inputs = {"question": "..."}
    Returns the full /ask response dict.
    """
    question = inputs.get("question", "")
    logger.info(f"[LS] → {question[:80]}")
    result = _ask(question)
    time.sleep(SLEEP_BETWEEN_RUNS)   # respect Groq rate limits
    return result


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LangSmith evaluation for CustomerAssist")
    parser.add_argument("--seed",        type=str, default="", help="Path to testset JSON to seed the dataset")
    parser.add_argument("--seed-only",   action="store_true",  help="Seed dataset and exit without running eval")
    parser.add_argument("--experiment",  type=str, default="", help="Experiment prefix shown in LangSmith UI")
    parser.add_argument("--n",           type=int, default=0,  help="Limit to first N examples (0 = all)")
    parser.add_argument("--dataset",     type=str, default=LANGSMITH_DATASET, help="LangSmith dataset name")
    args = parser.parse_args()

    # ── Validate env ────────────────────────────────────────────────────────────
    if not LANGSMITH_API_KEY:
        logger.error("[LS] LANGCHAIN_API_KEY is not set. Add it to .env.")
        sys.exit(1)
    if not GROQ_EVAL_API_KEY:
        logger.warning("[LS] GROQ_EVAL_API_KEY not set — judge will use GROQ_API_KEY fallback.")

    # ── Connect to LangSmith ────────────────────────────────────────────────────
    try:
        from langsmith import Client
        from langsmith import evaluate as ls_evaluate
    except ImportError:
        logger.error("[LS] langsmith not installed. Run: pip install langsmith")
        sys.exit(1)

    client  = Client(api_key=LANGSMITH_API_KEY)
    dataset = _get_or_create_dataset(client, args.dataset)
    logger.info(f"[LS] Dataset: '{dataset.name}'  (id={dataset.id})")

    # ── Seed ────────────────────────────────────────────────────────────────────
    if args.seed:
        _seed_dataset(client, dataset, args.seed)

    if args.seed_only:
        logger.info("[LS] --seed-only: exiting after seed.")
        return

    # ── Check dataset is non-empty ──────────────────────────────────────────────
    examples = list(client.list_examples(dataset_id=dataset.id))
    if not examples:
        logger.error(
            f"[LS] Dataset '{dataset.name}' is empty.\n"
            "     Use --seed <testset.json> to populate it first, or\n"
            "     add examples manually at smith.langchain.com."
        )
        sys.exit(1)

    if args.n:
        examples = examples[: args.n]
        logger.info(f"[LS] Limited to first {len(examples)} examples (--n {args.n})")

    logger.info(f"[LS] Running evaluation on {len(examples)} examples …")

    # ── Build evaluators ────────────────────────────────────────────────────────
    evaluators = _make_evaluators()
    if not evaluators:
        logger.error("[LS] No evaluators available — check judge.py imports.")
        sys.exit(1)

    # ── Run evaluation ──────────────────────────────────────────────────────────
    experiment_prefix = args.experiment or "customerassist"

    results = ls_evaluate(
        rag_pipeline,
        data               = args.dataset,
        evaluators         = evaluators,
        experiment_prefix  = experiment_prefix,
        client             = client,
        # Metadata shown in the LangSmith UI
        metadata           = {
            "api_url":     API_URL,
            "judge_model": os.getenv("JUDGE_MODEL", "qwen/qwen3-32b"),
            "sleep":       SLEEP_BETWEEN_RUNS,
        },
    )

    # ── Print summary ───────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  LANGSMITH EVALUATION COMPLETE")
    print("═" * 60)
    print(f"  Dataset:    {args.dataset}")
    print(f"  Examples:   {len(examples)}")
    print(f"  Experiment: {experiment_prefix}")
    print(f"\n  View results at:")
    print(f"  https://smith.langchain.com/o/<org>/projects/p/{LANGSMITH_PROJECT}")
    print("═" * 60 + "\n")

    # Print aggregate scores if available
    try:
        agg = results.to_pandas().select_dtypes("number").mean()
        print("  Aggregate scores:")
        for col, val in agg.items():
            if "score" in col.lower() or any(
                d in col for d in [
                    "overall", "faithfulness", "answer_relevance",
                    "context_relevance", "completeness", "citation_accuracy",
                    "correctness", "confidence",
                ]
            ):
                print(f"    {col:<30} {val:.4f}")
        print()
    except Exception:
        pass  # pandas optional — results still visible in UI


if __name__ == "__main__":
    main()
