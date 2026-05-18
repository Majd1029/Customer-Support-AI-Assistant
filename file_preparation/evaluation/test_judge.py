"""
file_preparation/evaluation/test_judge.py

CLI smoke-test for the LLM-as-a-Judge module.

Usage
-----
# Activate venv first
.\\venv\\Scripts\\activate

# Quick self-test with built-in sample (uses GROQ_API_KEY from .env)
python file_preparation/evaluation/test_judge.py

# Use a specific model
python file_preparation/evaluation/test_judge.py --model qwen/qwen3-32b

# Test with a custom question / answer
python file_preparation/evaluation/test_judge.py \\
    --question "What is the revenue?" \\
    --answer "Revenue was $4.2B in Q3 2024." \\
    --chunks '[]'

# Add a ground-truth reference for correctness scoring
python file_preparation/evaluation/test_judge.py \\
    --reference "Revenue was $4.2B in Q3 2024, a 12% YoY increase."

# Score each retrieved chunk individually
python file_preparation/evaluation/test_judge.py --score-chunks

# Save results to CSV for longitudinal tracking
python file_preparation/evaluation/test_judge.py --csv eval_log.csv

# Mark as a no-answer refusal (skips citation_accuracy + completeness)
python file_preparation/evaluation/test_judge.py --no-answer

# Batch evaluation from a JSONL file
#   Each line: {"question":"...","answer":"...","chunks":[...],"reference":"..."}
python file_preparation/evaluation/test_judge.py --batch results.jsonl

# Batch with inter-record sleep (recommended 2.0 s on Groq free tier)
python file_preparation/evaluation/test_judge.py --batch results.jsonl --sleep 2.0 --csv eval_log.csv

# Sequential calls (useful for debugging one dimension at a time)
python file_preparation/evaluation/test_judge.py --sequential
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "file_processor"))

from file_preparation.evaluation import judge_answer, batch_judge  # noqa: E402
from file_preparation.evaluation.judge import (                     # noqa: E402
    _JUDGE_MODEL,
    save_to_csv,
)


# ---------------------------------------------------------------------------
# Built-in sample data
# ---------------------------------------------------------------------------

_SAMPLE_CHUNKS = [
    {
        "content": (
            "The company reported revenue of $4.2 billion in Q3 2024, "
            "a 12% increase year-over-year driven primarily by cloud services growth."
        ),
        "metadata": {"source": "earnings_report.pdf", "page_start": 4},
    },
    {
        "content": (
            "Operating expenses rose to $2.1 billion in Q3 2024, "
            "mainly due to increased R&D investment in AI infrastructure."
        ),
        "metadata": {"source": "earnings_report.pdf", "page_start": 5},
    },
    {
        "content": (
            "Net income for the quarter was $890 million, compared to $740 million "
            "in Q3 2023, an improvement of 20%."
        ),
        "metadata": {"source": "earnings_report.pdf", "page_start": 6},
    },
]

_SAMPLE_ANSWER = (
    "In Q3 2024, the company achieved revenue of $4.2 billion, representing a "
    "12% year-over-year increase driven by cloud services "
    "[Source: earnings_report.pdf, page 4]. "
    "Operating expenses for the same period reached $2.1 billion, primarily "
    "reflecting higher R&D investment in AI infrastructure "
    "[Source: earnings_report.pdf, page 5]. "
    "Net income improved by 20% to $890 million compared to the prior year "
    "[Source: earnings_report.pdf, page 6]."
)

_SAMPLE_QUESTION = (
    "What were the Q3 2024 revenue, operating expenses, and net income figures?"
)

_SAMPLE_REFERENCE = (
    "In Q3 2024, revenue was $4.2 billion (up 12% YoY), operating expenses were "
    "$2.1 billion, and net income was $890 million (up 20% from $740M in Q3 2023)."
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM-as-a-Judge smoke-test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--question",     default=None, help="Question to evaluate")
    p.add_argument("--answer",       default=None, help="Answer to evaluate")
    p.add_argument("--chunks",       default=None, help="JSON array of chunk dicts")
    p.add_argument("--reference",    default=None,
                   help="Ground-truth reference answer (enables correctness scoring)")
    p.add_argument("--no-answer",    action="store_true",
                   help="Mark as a no-answer refusal (skips citation/completeness)")
    p.add_argument("--score-chunks", action="store_true",
                   help="Score each retrieved chunk individually for relevance")
    p.add_argument("--batch",        default=None, help="Path to JSONL file for batch eval")
    p.add_argument("--csv",          default=None,
                   help="Path to CSV file for logging results (appends if exists)")
    p.add_argument(
        "--model",
        default=_JUDGE_MODEL,
        help=f"Judge model (default: {_JUDGE_MODEL})",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Groq API key (falls back to JUDGE_API_KEY / GROQ_API_KEY env var)",
    )
    p.add_argument(
        "--sequential",
        action="store_true",
        help="Disable parallel judge calls (useful for debugging one dimension at a time)",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between batch records (default 0 — use 2.0 for Groq free tier)",
    )
    return p.parse_args()


def _pretty(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def main() -> None:
    args = _parse_args()

    # ── Batch mode ──────────────────────────────────────────────────────────────
    if args.batch:
        path = Path(args.batch)
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        records = []
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f"Batch evaluating {len(records)} records with {args.model} …\n")
        results = batch_judge(
            records,
            model              = args.model,
            api_key            = args.api_key,
            inter_record_sleep = args.sleep,
            run_parallel       = not args.sequential,
            score_chunks       = args.score_chunks,
            csv_path           = args.csv,
        )
        if args.csv:
            print(f"Results logged → {args.csv}")

        out_path = path.with_suffix(".judged.jsonl")
        with out_path.open("w") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Results written to {out_path}")
        print(f"\n{'Question (truncated)':<60} {'Overall':>8} {'Correct':>8}")
        print("-" * 80)
        for r in results:
            q   = r["question"][:57] + "…" if len(r["question"]) > 57 else r["question"]
            ov  = r["judge"]["overall"]
            cor = r["judge"]["dimensions"]["correctness"]["score"]
            print(
                f"{q:<60} "
                f"{f'{ov:.3f}' if ov is not None else '  N/A':>8} "
                f"{f'{cor:.3f}' if cor is not None else '  N/A':>8}"
            )
        return

    # ── Single evaluation mode ───────────────────────────────────────────────────
    if args.question or args.answer or args.chunks:
        question  = args.question or _SAMPLE_QUESTION
        answer    = args.answer   or _SAMPLE_ANSWER
        chunks    = json.loads(args.chunks) if args.chunks else _SAMPLE_CHUNKS
    else:
        print("No arguments provided — running built-in sample evaluation.\n")
        question = _SAMPLE_QUESTION
        answer   = _SAMPLE_ANSWER
        chunks   = _SAMPLE_CHUNKS

    reference = args.reference
    if reference is None and not args.question:
        # Use the built-in reference for the built-in sample
        reference = _SAMPLE_REFERENCE
        print("Using built-in reference answer for correctness scoring.\n")

    print(f"Question  : {question}")
    print(f"Model     : {args.model}")
    print(f"Chunks    : {len(chunks)}")
    print(f"No-answer : {args.no_answer}")
    print(f"Reference : {'yes' if reference else 'no'}")
    print(f"Chunks    : {len(chunks)} {'(will score individually)' if args.score_chunks else ''}")
    print("-" * 60)

    result = judge_answer(
        question     = question,
        answer       = answer,
        chunks       = chunks,
        no_answer    = args.no_answer,
        reference    = reference,
        score_chunks = args.score_chunks,
        model        = args.model,
        api_key      = args.api_key,
        run_parallel = not args.sequential,
    )

    if args.csv:
        save_to_csv(result, args.csv)
        print(f"Result logged → {args.csv}\n")

    data = result.to_dict()

    print(f"\n{'OVERALL':<30} {data['overall']}")
    print(f"{'Elapsed':<30} {data['elapsed_ms']} ms")
    print(f"{'Model':<30} {data['model']}")
    if data.get("error"):
        print(f"\nERROR: {data['error']}")

    print("\nDimension Scores")
    print("-" * 60)
    for dim, info in data["dimensions"].items():
        score_str = f"{info['score']:.4f}" if info["score"] is not None else "  N/A "
        print(f"  {dim:<22} {score_str}   {info['reasoning']}")

    if data.get("chunk_scores"):
        print("\nPer-Chunk Relevance")
        print("-" * 60)
        for c in data["chunk_scores"]:
            score = c["score"]
            bar   = ("█" * round((score or 0) * 5)).ljust(5, "░") if score is not None else "?" * 5
            print(f"  Chunk {c['chunk_idx']} [{bar}] {f'{score:.2f}' if score is not None else 'N/A'} — {c['reasoning']}")
            print(f"          \"{c['preview']}\"")

    print("\nFull JSON:")
    print(_pretty(data))


if __name__ == "__main__":
    main()
