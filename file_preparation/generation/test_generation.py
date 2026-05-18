# test_generation.py — run from any working directory with the venv active
# Location: file_preparation/generation/test_generation.py
import sys
from pathlib import Path

# Resolve project root: generation/ -> file_preparation/ -> project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


import argparse
from file_preparation.retrieval import retrieve_evidence
from file_preparation.generation import generate_answer, GenerationConfig, get_generator
from file_preparation.indexing import get_client

# Setup CLI arguments
parser = argparse.ArgumentParser(description="Test RAG generation pipeline")
parser.add_argument("question", nargs="?", help="Question to ask (if empty, you will be prompted)")
parser.add_argument("--stream", action="store_true", help="Enable streaming output")
args = parser.parse_args()

client = get_client()

# Get question from CLI or prompt
question = args.question
if not question:
    question = input("Enter your question: ")

# Step 1: retrieve
retrieval = retrieve_evidence(question, client, limit=5, rerank=True, context_window=0)
print(f"Retrieved {retrieval.total} chunks in {retrieval.elapsed_ms:.0f}ms")

# Step 2: flatten chunks + neighbours into the shape AnswerGenerator expects
chunks = [
    {"content": c.content, "score": c.score, "hop": c.hop,
     "primary": True, "metadata": c.metadata}
    for c in retrieval.chunks
]
chunks += [
    {"content": n.get("content",""), "score": n.get("score",0.0),
     "hop": 1, "primary": False, "metadata": n}
    for c in retrieval.chunks for n in c.neighbors
]

# Step 3: generate
# max_tokens raised 512 → 2000 to prevent answer truncation.
# context_token_cap kept at 8000 (matches answer_generator.py default).
cfg = GenerationConfig(temperature=0.2, max_tokens=2000)

SEP = "=" * 70

if args.stream:
    gen = get_generator()
    print(f"\n{SEP}")
    print(f"  ANSWER (streaming)")
    print(f"{SEP}\n")
    result = gen.stream_with_sources(question, chunks, cfg)
    result.retrieval_ms = retrieval.elapsed_ms   # wire in retrieval timing
    print(f"\n{SEP}")
else:
    result = generate_answer(question, chunks, cfg)
    result.retrieval_ms = retrieval.elapsed_ms   # wire in retrieval timing
    print(f"\n{SEP}")
    print(f"  ANSWER")
    print(f"{SEP}\n")
    print(result.answer)
    print(f"\n{SEP}")

if result.sources:
    print(f"  SOURCES CITED ({len(result.sources)})")
    print(f"{SEP}")
    for i, src in enumerate(result.sources, 1):
        location = src["source"]
        if src.get("page_start"):
            location += f", page {src['page_start']}"
        section = src.get("section", "")
        # Only show section if it looks like a real heading (short, no mid-sentence markers)
        if section and len(section) <= 80 and not any(
            c in section for c in ["[", ".", ",", "—"]
        ):
            location += f" | {section}"
        print(f"  [{i}] {location}")
    print(f"{SEP}\n")

print(f"  Backend  : {result.backend}/{result.model}")
print(f"  Latency  : {result.elapsed_ms:.0f}ms total"
      + (f"  |  gen={result.generation_ms:.0f}ms" if result.generation_ms else ""))
if result.retrieval_ms is not None:
    print(f"  Retrieval: {result.retrieval_ms:.0f}ms")
print(f"  Ctx tok  : {result.tokens_in_context} tokens ({result.context_utilisation:.0%} of cap)"
      f", {len(chunks)} chunks")
if result.token_counts:
    tc = result.token_counts
    print(f"  Tokens   : {tc['prompt']} prompt + {tc['completion']} completion = {tc['total']}")
print(f"  Citations: {result.citation_count}  |  Answer: {result.answer_length_chars} chars")
print(f"  No-answer: {result.no_answer}")
print(f"{SEP}\n")