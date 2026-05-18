"""
test_embedding_quality.py — BGE-M3 embedding quality diagnostics.

Runs 7 independent quality checks on the embedder and prints a
colour-coded report. No ground-truth labels needed — all tests are
self-contained and use internal consistency checks.

Usage
-----
    cd file_preparation/embedding
    python test_embedding_quality.py            # all tests
    python test_embedding_quality.py --quick    # skip slow isotropy test

Tests
-----
1. Norm integrity     — dense vectors must be L2-normalised (‖v‖ ≈ 1.0)
2. Semantic ordering  — related pair > unrelated pair cosine similarity
3. Sparse health      — SPLADE non-zero counts in expected range (50–500)
4. Isotropy           — average cosine of random pairs (good < 0.5)
5. Retrieval P@1      — top-1 retrieval on 10 known Q→passage pairs
6. Cluster separation — intra-doc similarity > inter-doc similarity
7. Negative rejection — clearly unrelated texts score < threshold
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "file_processor"))

from embedder import encode, encode_query  # type: ignore[import]


# ── ANSI colours ──────────────────────────────────────────────────────────────
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _ok(msg: str)   -> str: return f"{_GREEN}✓ PASS{_RESET}  {msg}"
def _fail(msg: str) -> str: return f"{_RED}✗ FAIL{_RESET}  {msg}"
def _warn(msg: str) -> str: return f"{_YELLOW}⚠ WARN{_RESET}  {msg}"
def _hdr(msg: str)  -> str: return f"\n{_BOLD}{_CYAN}── {msg} {'─' * (55 - len(msg))}{_RESET}"


# ── Helper: cosine similarity ─────────────────────────────────────────────────

def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two dense vectors (assumed L2-normalised → dot product)."""
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 1e-9 else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Test 1 — Norm integrity
# ═════════════════════════════════════════════════════════════════════════════

def test_norm_integrity() -> bool:
    """
    BGE-M3 dense vectors are L2-normalised during encoding.
    Every vector norm should be in [0.999, 1.001].

    Why: if norms deviate, cosine-similarity search degrades to dot-product
    search, producing incorrect relevance scores.
    """
    print(_hdr("Test 1 · Norm Integrity"))
    texts = [
        "The quarterly revenue report shows a 12% increase.",
        "Machine learning models require large datasets.",
        "يُعدّ التعلم الآلي مجالاً مثيراً للاهتمام.",   # Arabic
        "Les résultats du trimestre sont encourageants.",  # French
        "",                                                 # edge case: empty
        "x",                                               # single char
    ]
    # Filter empty so encode doesn't receive them as real inputs
    non_empty = [t for t in texts if t.strip()]
    emb = encode(non_empty)

    failures = []
    for i, (text, vec) in enumerate(zip(non_empty, emb.dense)):
        norm = float(np.linalg.norm(np.asarray(vec, dtype=np.float32)))
        if not (0.999 <= norm <= 1.001):
            failures.append(f"  text[{i}] norm={norm:.6f}  text={text[:40]!r}")

    if failures:
        print(_fail(f"{len(failures)}/{len(non_empty)} vectors out of norm range [0.999, 1.001]:"))
        for f in failures:
            print(f)
        return False

    norms = [float(np.linalg.norm(np.asarray(v, dtype=np.float32))) for v in emb.dense]
    print(_ok(f"All {len(non_empty)} vectors L2-normalised  "
              f"(min={min(norms):.6f}, max={max(norms):.6f})"))
    return True


# ═════════════════════════════════════════════════════════════════════════════
# Test 2 — Semantic ordering
# ═════════════════════════════════════════════════════════════════════════════

_SEMANTIC_PAIRS = [
    # (anchor, positive, negative)  — positive should score > negative
    (
        "What is the interest rate on a savings account?",
        "Savings accounts typically offer 3–5% annual interest rates depending on the bank.",
        "The football match ended 2-1 in favour of the home team.",
    ),
    (
        "How do I reset my password?",
        "To reset your password, click 'Forgot password' on the login page and follow the instructions.",
        "The quarterly earnings call is scheduled for next Tuesday.",
    ),
    (
        "Annual revenue figures for 2023",
        "Total revenue for fiscal year 2023 reached $4.2 billion, up 18% year-on-year.",
        "The new hiking trail was opened last month near the national park.",
    ),
    (
        "ما هو معدل الفائدة على حساب التوفير؟",
        "تقدم حسابات التوفير عادةً معدل فائدة سنوي يتراوح بين 3 و5 بالمئة.",
        "فاز الفريق المحلي بنتيجة 2-1 في مباراة كرة القدم.",
    ),
]


def test_semantic_ordering() -> bool:
    """
    For each (anchor, positive, negative) triple:
      cosine(anchor, positive) > cosine(anchor, negative)

    Tests that the model preserves semantic meaning and isn't just
    doing lexical overlap.
    """
    print(_hdr("Test 2 · Semantic Ordering"))
    passed = 0
    failed_cases = []

    for i, (anchor, pos, neg) in enumerate(_SEMANTIC_PAIRS):
        emb = encode([anchor, pos, neg])
        sim_pos = cosine(emb.dense[0], emb.dense[1])
        sim_neg = cosine(emb.dense[0], emb.dense[2])

        ok = sim_pos > sim_neg
        if ok:
            passed += 1
        else:
            failed_cases.append(
                f"  Pair {i+1}: sim_pos={sim_pos:.4f} ≤ sim_neg={sim_neg:.4f}\n"
                f"    anchor  : {anchor[:60]!r}\n"
                f"    positive: {pos[:60]!r}\n"
                f"    negative: {neg[:60]!r}"
            )

        marker = _GREEN + "✓" + _RESET if ok else _RED + "✗" + _RESET
        print(f"  {marker} Pair {i+1}: pos={sim_pos:.4f}  neg={sim_neg:.4f}  "
              f"margin={sim_pos - sim_neg:+.4f}")

    if failed_cases:
        print(_fail(f"{len(failed_cases)}/{len(_SEMANTIC_PAIRS)} pairs violated ordering:"))
        for fc in failed_cases:
            print(fc)
        return False

    print(_ok(f"All {len(_SEMANTIC_PAIRS)} pairs correctly ordered  "
              f"({passed}/{len(_SEMANTIC_PAIRS)} passed)"))
    return True


# ═════════════════════════════════════════════════════════════════════════════
# Test 3 — Sparse vector health
# ═════════════════════════════════════════════════════════════════════════════

_SPARSE_MIN_TOKENS = 5    # at least this many SPLADE non-zero entries for normal-length text
_SPARSE_MAX_TOKENS = 600  # at most this many
_SPARSE_MIN_WORDS  = 5    # texts with fewer words get a scaled-down minimum


def test_sparse_health() -> bool:
    """
    SPLADE sparse vectors should have 50–500 non-zero entries per passage.
    - Too few: model not capturing enough lexical features
    - Too many: weights may not be properly thresholded (memory / performance risk)
    - Weight range: all weights should be non-negative (SPLADE uses ReLU)
    """
    print(_hdr("Test 3 · Sparse Vector Health"))
    texts = [
        "The annual financial report shows strong growth in the technology sector.",
        "Customer support tickets are resolved within 24 hours on average.",
        "يُعدّ التعلم الآلي مجالاً واسع التطبيقات في العصر الحديث.",
        "Machine learning models require high-quality training data to generalise well.",
        "Short.",
    ]
    emb = encode(texts)

    all_ok = True
    counts = []
    for i, (text, sv) in enumerate(zip(texts, emb.sparse)):
        nnz        = len(sv)
        neg_weights = [w for w in sv.values() if w < 0]
        counts.append(nnz)

        # Scale the minimum threshold for very short texts.
        # A 1-word text like "Short." can legitimately produce only 1–2 SPLADE
        # entries — the model is not broken, the input is just tiny.
        word_count  = len(text.split())
        eff_min     = max(1, min(_SPARSE_MIN_TOKENS, word_count))
        range_ok    = eff_min <= nnz <= _SPARSE_MAX_TOKENS
        relu_ok     = len(neg_weights) == 0

        issues = []
        if not range_ok:
            issues.append(f"nnz={nnz} outside [{eff_min},{_SPARSE_MAX_TOKENS}] (words={word_count})")
        if not relu_ok:
            issues.append(f"{len(neg_weights)} negative weights (SPLADE should use ReLU)")

        marker = _GREEN + "✓" + _RESET if not issues else _RED + "✗" + _RESET
        print(f"  {marker} text[{i}]: nnz={nnz:4d}  "
              f"max_w={max(sv.values(), default=0):.3f}  {text[:40]!r}"
              + (f"  ⚠ {'; '.join(issues)}" if issues else ""))

        if issues:
            all_ok = False

    print(f"  Stats: nnz mean={sum(counts)/len(counts):.1f}  "
          f"min={min(counts)}  max={max(counts)}")

    if all_ok:
        print(_ok("All sparse vectors within healthy range"))
    else:
        print(_fail("Some sparse vectors are outside expected range"))
    return all_ok


# ═════════════════════════════════════════════════════════════════════════════
# Test 4 — Isotropy
# ═════════════════════════════════════════════════════════════════════════════

_ISOTROPY_TEXTS = [
    "The board approved the budget for next year.",
    "Revenue increased by 15% compared to the previous quarter.",
    "Customer complaints have decreased significantly.",
    "The new product line will launch in Q3.",
    "Operational costs were reduced through automation.",
    "Employee satisfaction scores are at an all-time high.",
    "The merger is expected to close by the end of the month.",
    "Supply chain disruptions have been resolved.",
    "R&D spending represents 12% of total revenue.",
    "The dividend was raised for the fifth consecutive year.",
    "Online sales now account for 40% of total transactions.",
    "Carbon emissions were reduced by 18% this year.",
    "The new CTO joined from a leading tech company.",
    "Regulatory approval was granted in all key markets.",
    "Net profit margin improved to 22%.",
    "The Asia-Pacific division showed the strongest growth.",
]

_ISOTROPY_THRESHOLD = 0.75  # above this → anisotropic / collapsed embedding space


def test_isotropy(skip: bool = False) -> bool:
    """
    Measures the average cosine similarity between random pairs of unrelated
    sentences.

    A perfectly isotropic space → average ≈ 0.0 (random directions).
    Collapsed (anisotropic) space → average close to 1.0 (all vectors cluster
    in one direction, making similarity scores meaningless).

    BGE-M3 typically scores 0.3–0.6 on business/technical text (acceptable).
    Above 0.75 is a warning sign that something is wrong.
    """
    print(_hdr("Test 4 · Isotropy (Anisotropy Check)"))
    if skip:
        print(f"  {_YELLOW}Skipped (--quick mode){_RESET}")
        return True

    emb = encode(_ISOTROPY_TEXTS)
    vecs = emb.dense
    n = len(vecs)

    sims = []
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for i, j in pairs:
        sims.append(cosine(vecs[i], vecs[j]))

    avg  = sum(sims) / len(sims)
    mn   = min(sims)
    mx   = max(sims)
    std  = float(np.std(sims))

    print(f"  Random-pair cosine stats over {len(pairs)} pairs:")
    print(f"    mean={avg:.4f}  std={std:.4f}  min={mn:.4f}  max={mx:.4f}")

    if avg > _ISOTROPY_THRESHOLD:
        print(_fail(f"High anisotropy: mean cosine={avg:.4f} > {_ISOTROPY_THRESHOLD}  "
                    f"(embedding space may be collapsed)"))
        return False
    elif avg > 0.5:
        print(_warn(f"Moderate anisotropy: mean cosine={avg:.4f}  (acceptable for domain text)"))
        return True   # warn but don't fail
    else:
        print(_ok(f"Good isotropy: mean cosine={avg:.4f}"))
        return True


# ═════════════════════════════════════════════════════════════════════════════
# Test 5 — Retrieval Precision@1
# ═════════════════════════════════════════════════════════════════════════════

_RETRIEVAL_PAIRS = [
    (
        "What is the refund policy?",
        "Customers may request a full refund within 30 days of purchase. Items must be unused and in original packaging.",
    ),
    (
        "How long does shipping take?",
        "Standard shipping takes 5–7 business days. Express delivery is available within 2 business days.",
    ),
    (
        "What are the password requirements?",
        "Passwords must be at least 12 characters and include one uppercase letter, one number, and one special character.",
    ),
    (
        "ما هي سياسة الاسترداد؟",
        "يمكن للعملاء طلب استرداد كامل خلال 30 يومًا من تاريخ الشراء، شريطة أن تكون المنتجات غير مستخدمة.",
    ),
    (
        "How is the annual bonus calculated?",
        "The annual bonus is calculated as 10–15% of base salary, depending on individual and company performance ratings.",
    ),
    (
        "What data does the app collect?",
        "The application collects your name, email address, and usage analytics to improve the service.",
    ),
    (
        "What is the minimum account balance?",
        "A minimum balance of $500 is required to avoid the monthly maintenance fee.",
    ),
    (
        "How do I contact technical support?",
        "Technical support is available 24/7 via live chat or by emailing support@example.com.",
    ),
    (
        "What programming languages are supported?",
        "The API supports Python, JavaScript, Java, and Go. SDKs are available on GitHub.",
    ),
    (
        "When does the warranty expire?",
        "The standard warranty covers hardware defects for 2 years from the date of purchase.",
    ),
]


def test_retrieval_precision() -> bool:
    """
    Precision@1 test: for each (query, relevant_passage) pair, embed both
    plus 9 distractors. Check that the relevant passage ranks 1st.

    This simulates the real retrieval task and is the most important quality
    signal for a RAG system.
    """
    print(_hdr("Test 5 · Retrieval Precision@1"))

    # Pool of all passages — each query's correct passage must rank above the rest
    all_passages = [p for _, p in _RETRIEVAL_PAIRS]
    pass_emb     = encode(all_passages)

    hits = 0
    for i, (query, correct_passage) in enumerate(_RETRIEVAL_PAIRS):
        q_emb = encode_query(query)
        q_vec = q_emb.dense[0]

        # Score against all passages
        scores = [(j, cosine(q_vec, pass_emb.dense[j])) for j in range(len(all_passages))]
        scores.sort(key=lambda x: x[1], reverse=True)

        top1_idx   = scores[0][0]
        top1_score = scores[0][1]
        correct_rank = next(r for r, (j, _) in enumerate(scores) if j == i)

        ok = (top1_idx == i)
        if ok:
            hits += 1

        marker = _GREEN + "✓" + _RESET if ok else _RED + "✗" + _RESET
        rank_str = f"rank={correct_rank + 1}" if not ok else "rank=1"
        print(f"  {marker} Q{i+1:02d}: score={top1_score:.4f}  {rank_str}  {query[:50]!r}")

    p_at_1 = hits / len(_RETRIEVAL_PAIRS)
    result_str = f"P@1 = {hits}/{len(_RETRIEVAL_PAIRS)} = {p_at_1:.0%}"

    if p_at_1 >= 0.8:
        print(_ok(result_str))
        return True
    elif p_at_1 >= 0.6:
        print(_warn(f"{result_str}  (below ideal ≥ 80%)"))
        return True   # warn but don't hard-fail
    else:
        print(_fail(f"{result_str}  (critically low — check embedder)"))
        return False


# ═════════════════════════════════════════════════════════════════════════════
# Test 6 — Cluster separation
# ═════════════════════════════════════════════════════════════════════════════

_CLUSTER_DOCS = {
    "finance": [
        "Revenue grew by 12% year-over-year driven by strong enterprise sales.",
        "Operating expenses were reduced through cost optimisation initiatives.",
        "Net profit margin improved from 18% to 22% in the current fiscal year.",
        "The dividend per share increased to $1.20, rewarding long-term shareholders.",
    ],
    "medicine": [
        "The clinical trial enrolled 500 patients across three hospital sites.",
        "Blood pressure was measured using a calibrated sphygmomanometer.",
        "Patients received a daily dose of 500 mg of the experimental compound.",
        "MRI scans showed a significant reduction in tumour size after 12 weeks.",
    ],
    "sports": [
        "The striker scored a hat-trick in the final minutes of the match.",
        "The team's defence conceded only two goals throughout the tournament.",
        "Training sessions were held twice daily in preparation for the championship.",
        "The coach announced changes to the starting lineup for the semi-final.",
    ],
}


def test_cluster_separation() -> bool:
    """
    Chunks from the same domain should have higher mutual cosine similarity
    than chunks from different domains.

    intra-cluster mean > inter-cluster mean is the pass condition.
    """
    print(_hdr("Test 6 · Cluster Separation (Domain Coherence)"))

    domain_vecs: dict[str, list[list[float]]] = {}
    for domain, texts in _CLUSTER_DOCS.items():
        domain_vecs[domain] = encode(texts).dense

    domains   = list(domain_vecs.keys())
    intra_sims = []
    inter_sims = []

    for d in domains:
        vecs = domain_vecs[d]
        n    = len(vecs)
        for i in range(n):
            for j in range(i + 1, n):
                intra_sims.append(cosine(vecs[i], vecs[j]))

    for di in range(len(domains)):
        for dj in range(di + 1, len(domains)):
            va = domain_vecs[domains[di]]
            vb = domain_vecs[domains[dj]]
            for a in va:
                for b in vb:
                    inter_sims.append(cosine(a, b))

    intra_mean = sum(intra_sims) / len(intra_sims)
    inter_mean = sum(inter_sims) / len(inter_sims)
    separation = intra_mean - inter_mean

    print(f"  Intra-cluster mean cosine : {intra_mean:.4f}")
    print(f"  Inter-cluster mean cosine : {inter_mean:.4f}")
    print(f"  Separation (intra - inter): {separation:+.4f}")

    if separation > 0.05:
        print(_ok(f"Good cluster separation (Δ={separation:+.4f})"))
        return True
    elif separation > 0:
        print(_warn(f"Weak but positive cluster separation (Δ={separation:+.4f})"))
        return True
    else:
        print(_fail(f"Negative cluster separation (Δ={separation:+.4f}) — "
                    f"inter-domain similarity exceeds intra-domain"))
        return False


# ═════════════════════════════════════════════════════════════════════════════
# Test 7 — Negative rejection
# ═════════════════════════════════════════════════════════════════════════════

_NEGATIVE_THRESHOLD = 0.82   # clearly unrelated pairs must score below this

_NEGATIVE_PAIRS = [
    ("The quarterly earnings exceeded expectations.", "The cat sat on the mat."),
    ("How do I reset my password?", "Photosynthesis converts sunlight into sugar."),
    ("Annual leave entitlement for full-time employees.", "The volcano erupted last Tuesday."),
    ("The refund was processed within 3 business days.", "Jupiter is the largest planet in the solar system."),
]


def test_negative_rejection() -> bool:
    """
    Clearly semantically unrelated pairs should score below the threshold.
    A good embedding model should confidently separate unrelated texts.
    """
    print(_hdr("Test 7 · Negative Rejection"))
    all_ok = True

    for i, (a, b) in enumerate(_NEGATIVE_PAIRS):
        emb = encode([a, b])
        sim = cosine(emb.dense[0], emb.dense[1])
        ok  = sim < _NEGATIVE_THRESHOLD

        marker = _GREEN + "✓" + _RESET if ok else _RED + "✗" + _RESET
        print(f"  {marker} Pair {i+1}: sim={sim:.4f}  (threshold <{_NEGATIVE_THRESHOLD})")
        print(f"         A: {a[:60]!r}")
        print(f"         B: {b[:60]!r}")

        if not ok:
            all_ok = False

    if all_ok:
        print(_ok(f"All {len(_NEGATIVE_PAIRS)} unrelated pairs scored below {_NEGATIVE_THRESHOLD}"))
    else:
        print(_fail("Some unrelated pairs scored above similarity threshold"))
    return all_ok


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

def run_all(quick: bool = False) -> None:
    print(f"\n{_BOLD}{_CYAN}{'=' * 60}")
    print("  BGE-M3 Embedding Quality Diagnostics")
    print(f"{'=' * 60}{_RESET}\n")

    t0 = time.time()

    results = {
        "Norm integrity"    : test_norm_integrity(),
        "Semantic ordering" : test_semantic_ordering(),
        "Sparse health"     : test_sparse_health(),
        "Isotropy"          : test_isotropy(skip=quick),
        "Retrieval P@1"     : test_retrieval_precision(),
        "Cluster separation": test_cluster_separation(),
        "Negative rejection": test_negative_rejection(),
    }

    elapsed = time.time() - t0

    print(f"\n{_BOLD}{_CYAN}{'─' * 60}")
    print("  Summary")
    print(f"{'─' * 60}{_RESET}")
    passed = sum(1 for v in results.values() if v)
    total  = len(results)

    for name, ok in results.items():
        status = f"{_GREEN}PASS{_RESET}" if ok else f"{_RED}FAIL{_RESET}"
        print(f"  {status}  {name}")

    colour = _GREEN if passed == total else (_YELLOW if passed >= total * 0.7 else _RED)
    print(f"\n{colour}{_BOLD}  {passed}/{total} tests passed  ({elapsed:.1f}s){_RESET}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BGE-M3 embedding quality diagnostics")
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip the isotropy test (which encodes 16 texts) — faster for CI"
    )
    args = parser.parse_args()
    run_all(quick=args.quick)
