"""
tests/test_metrics.py

Automated tests for ROUGE and BLEU evaluation metrics.

Tests run without any external services (no Groq, no Qdrant).
They do require rouge-score and nltk to be installed:

    pip install rouge-score nltk
    python -c "import nltk; nltk.download('punkt_tab')"

Run:
    pytest tests/test_metrics.py -v
"""

import csv
import math
import pytest
from pathlib import Path

from file_preparation.evaluation.metrics import (
    score_answer,
    batch_score,
    interpret,
    save_to_csv,
    MetricResult,
    ROUGE_L_PASS,
    ROUGE_L_WARN,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

REFERENCE = (
    "Revenue reached $4.2 billion in Q3 2024, representing a 12% "
    "year-over-year increase. Operating expenses were $1.1 billion, "
    "an 8% increase compared to the prior year."
)

PERFECT_ANSWER = REFERENCE                           # identical → scores ≈ 1.0

GOOD_ANSWER = (
    "Q3 2024 revenue was $4.2B, up 12% year-over-year. "
    "Operating expenses grew 8% to $1.1B."
)                                                     # paraphrase → moderate scores

PARTIAL_ANSWER = "Revenue was $4.2 billion."          # misses opex → lower recall

HALLUCINATED_ANSWER = (
    "Revenue was $9.8 billion, a 45% drop. "
    "The company announced record losses of $2.3B."
)                                                     # wrong facts → low scores

EMPTY_ANSWER = ""
EMPTY_REFERENCE = ""


# ── Basic smoke tests ──────────────────────────────────────────────────────────

class TestScoreAnswer:

    def test_returns_metric_result(self):
        r = score_answer(GOOD_ANSWER, REFERENCE)
        assert isinstance(r, MetricResult)

    def test_perfect_answer_near_one(self):
        r = score_answer(PERFECT_ANSWER, REFERENCE)
        # Stemmer and tokenisation are not byte-perfect, but should be >= 0.95
        if r.rouge_l_f1 is not None:
            assert r.rouge_l_f1 >= 0.95, f"Expected ≥ 0.95, got {r.rouge_l_f1}"
        if r.bleu is not None:
            assert r.bleu >= 0.85, f"Expected BLEU ≥ 0.85, got {r.bleu}"

    def test_good_answer_moderate(self):
        r = score_answer(GOOD_ANSWER, REFERENCE)
        if r.rouge_l_f1 is not None:
            assert r.rouge_l_f1 >= 0.30, f"Paraphrase should score ≥ 0.30, got {r.rouge_l_f1}"

    def test_partial_answer_lower_than_good(self):
        r_good    = score_answer(GOOD_ANSWER,    REFERENCE)
        r_partial = score_answer(PARTIAL_ANSWER, REFERENCE)
        if r_good.rouge_l_f1 is not None and r_partial.rouge_l_f1 is not None:
            assert r_partial.rouge_l_f1 <= r_good.rouge_l_f1, (
                f"Partial ({r_partial.rouge_l_f1}) should score ≤ good ({r_good.rouge_l_f1})"
            )

    def test_hallucinated_answer_low(self):
        r = score_answer(HALLUCINATED_ANSWER, REFERENCE)
        if r.rouge_l_f1 is not None:
            assert r.rouge_l_f1 < 0.30, f"Hallucinated answer should score < 0.30, got {r.rouge_l_f1}"

    def test_empty_answer_returns_error(self):
        r = score_answer(EMPTY_ANSWER, REFERENCE)
        assert r.error is not None
        assert r.rouge_l_f1 is None
        assert r.bleu is None

    def test_empty_reference_returns_error(self):
        r = score_answer(GOOD_ANSWER, EMPTY_REFERENCE)
        assert r.error is not None

    def test_elapsed_ms_populated(self):
        r = score_answer(GOOD_ANSWER, REFERENCE)
        assert r.elapsed_ms > 0

    def test_question_stored(self):
        r = score_answer(GOOD_ANSWER, REFERENCE, question="What were Q3 revenues?")
        assert r.question == "What were Q3 revenues?"

    def test_answer_reference_stored(self):
        r = score_answer(GOOD_ANSWER, REFERENCE)
        assert r.answer    == GOOD_ANSWER
        assert r.reference == REFERENCE


# ── ROUGE specific tests ───────────────────────────────────────────────────────

class TestROUGE:

    def test_rouge1_fields_present(self):
        r = score_answer(GOOD_ANSWER, REFERENCE)
        if r.rouge1_f1 is not None:   # only if rouge-score installed
            assert 0.0 <= r.rouge1_f1        <= 1.0
            assert 0.0 <= r.rouge1_precision  <= 1.0
            assert 0.0 <= r.rouge1_recall     <= 1.0

    def test_rouge2_fields_present(self):
        r = score_answer(GOOD_ANSWER, REFERENCE)
        if r.rouge2_f1 is not None:
            assert 0.0 <= r.rouge2_f1 <= 1.0

    def test_rougel_fields_present(self):
        r = score_answer(GOOD_ANSWER, REFERENCE)
        if r.rouge_l_f1 is not None:
            assert 0.0 <= r.rouge_l_f1 <= 1.0

    def test_rouge1_ge_rouge2(self):
        """Unigram overlap is always >= bigram overlap."""
        r = score_answer(GOOD_ANSWER, REFERENCE)
        if r.rouge1_f1 is not None and r.rouge2_f1 is not None:
            assert r.rouge1_f1 >= r.rouge2_f1, (
                f"ROUGE-1 ({r.rouge1_f1}) should be >= ROUGE-2 ({r.rouge2_f1})"
            )

    def test_recall_higher_for_partial(self):
        """
        A short answer that states a correct fact should have high precision
        but lower recall (it misses other content in the reference).
        """
        r = score_answer(PARTIAL_ANSWER, REFERENCE)
        if r.rouge1_precision is not None and r.rouge1_recall is not None:
            # Short answer → precision should be reasonable, recall lower
            # (partial answer covers only ~1/3 of the reference content)
            assert r.rouge1_recall < r.rouge1_precision or r.rouge1_recall < 0.60


# ── BLEU specific tests ────────────────────────────────────────────────────────

class TestBLEU:

    def test_bleu_fields_present(self):
        r = score_answer(GOOD_ANSWER, REFERENCE)
        if r.bleu is not None:
            assert 0.0 <= r.bleu   <= 1.0
            assert 0.0 <= r.bleu_1 <= 1.0
            assert 0.0 <= r.bleu_2 <= 1.0
            assert 0.0 <= r.bleu_3 <= 1.0
            assert 0.0 <= r.bleu_4 <= 1.0

    def test_bleu1_ge_bleu4(self):
        """Lower n-gram BLEU is always >= higher n-gram BLEU."""
        r = score_answer(GOOD_ANSWER, REFERENCE)
        if r.bleu_1 is not None and r.bleu_4 is not None:
            assert r.bleu_1 >= r.bleu_4, (
                f"BLEU-1 ({r.bleu_1}) should be >= BLEU-4 ({r.bleu_4})"
            )

    def test_hallucinated_bleu_low(self):
        r = score_answer(HALLUCINATED_ANSWER, REFERENCE)
        if r.bleu is not None:
            assert r.bleu < 0.20, f"Hallucinated BLEU should be < 0.20, got {r.bleu}"


# ── Interpret / verdict tests ──────────────────────────────────────────────────

class TestInterpret:

    def test_perfect_is_pass(self):
        r = score_answer(PERFECT_ANSWER, REFERENCE)
        assert interpret(r) in ("pass", "unknown")   # unknown if libs missing

    def test_hallucinated_is_fail_or_warn(self):
        r = score_answer(HALLUCINATED_ANSWER, REFERENCE)
        assert interpret(r) in ("fail", "warn", "unknown")

    def test_empty_is_unknown(self):
        r = score_answer(EMPTY_ANSWER, REFERENCE)
        assert interpret(r) == "unknown"

    def test_thresholds_consistent(self):
        assert ROUGE_L_WARN < ROUGE_L_PASS, "WARN threshold must be below PASS"

    def test_result_with_exact_pass_threshold(self):
        """A result exactly at the PASS threshold should be 'pass'."""
        r = MetricResult(question="q", answer="a", reference="r", rouge_l_f1=ROUGE_L_PASS)
        assert interpret(r) == "pass"

    def test_result_just_below_pass_threshold(self):
        r = MetricResult(question="q", answer="a", reference="r", rouge_l_f1=ROUGE_L_PASS - 0.01)
        assert interpret(r) == "warn"

    def test_result_just_below_warn_threshold(self):
        r = MetricResult(question="q", answer="a", reference="r", rouge_l_f1=ROUGE_L_WARN - 0.01)
        assert interpret(r) == "fail"


# ── Batch scoring tests ────────────────────────────────────────────────────────

class TestBatchScore:

    def test_returns_list(self):
        records = [
            {"question": "Q1", "answer": GOOD_ANSWER,         "reference": REFERENCE},
            {"question": "Q2", "answer": HALLUCINATED_ANSWER, "reference": REFERENCE},
        ]
        results = batch_score(records)
        assert len(results) == 2
        assert all(isinstance(r, MetricResult) for r in results)

    def test_order_preserved(self):
        records = [
            {"answer": GOOD_ANSWER,         "reference": REFERENCE, "question": "good"},
            {"answer": HALLUCINATED_ANSWER, "reference": REFERENCE, "question": "bad"},
        ]
        results = batch_score(records)
        assert results[0].question == "good"
        assert results[1].question == "bad"

    def test_higher_score_for_better_answer(self):
        records = [
            {"answer": GOOD_ANSWER,         "reference": REFERENCE},
            {"answer": HALLUCINATED_ANSWER, "reference": REFERENCE},
        ]
        results = batch_score(records)
        if results[0].rouge_l_f1 is not None and results[1].rouge_l_f1 is not None:
            assert results[0].rouge_l_f1 > results[1].rouge_l_f1

    def test_csv_output(self, tmp_path: Path):
        csv_file = tmp_path / "metrics.csv"
        records  = [
            {"question": "Q", "answer": GOOD_ANSWER, "reference": REFERENCE},
        ]
        batch_score(records, csv_path=csv_file)
        assert csv_file.exists()
        rows = list(csv.DictReader(csv_file.open()))
        assert len(rows) == 1
        assert "rouge_l_f1" in rows[0]
        assert "bleu"       in rows[0]
        assert "verdict"    in rows[0]

    def test_missing_question_key_ok(self):
        """Records without a 'question' key should not crash."""
        records = [{"answer": GOOD_ANSWER, "reference": REFERENCE}]
        results = batch_score(records)
        assert len(results) == 1
        assert results[0].question == ""


# ── CSV persistence tests ──────────────────────────────────────────────────────

class TestSaveToCSV:

    def test_creates_file(self, tmp_path: Path):
        p = tmp_path / "out.csv"
        r = score_answer(GOOD_ANSWER, REFERENCE, question="test")
        save_to_csv(r, p)
        assert p.exists()

    def test_header_written_once(self, tmp_path: Path):
        p = tmp_path / "out.csv"
        r = score_answer(GOOD_ANSWER, REFERENCE)
        save_to_csv(r, p)
        save_to_csv(r, p)
        rows   = list(csv.DictReader(p.open()))
        assert len(rows) == 2, "Two data rows expected, header should appear once"

    def test_verdict_column_present(self, tmp_path: Path):
        p = tmp_path / "out.csv"
        r = score_answer(PERFECT_ANSWER, REFERENCE)
        save_to_csv(r, p)
        rows = list(csv.DictReader(p.open()))
        assert "verdict" in rows[0]

    def test_accepts_list(self, tmp_path: Path):
        p = tmp_path / "out.csv"
        results = [
            score_answer(GOOD_ANSWER,         REFERENCE),
            score_answer(HALLUCINATED_ANSWER, REFERENCE),
        ]
        save_to_csv(results, p)
        rows = list(csv.DictReader(p.open()))
        assert len(rows) == 2

    def test_timestamp_column_present(self, tmp_path: Path):
        p = tmp_path / "out.csv"
        save_to_csv(score_answer(GOOD_ANSWER, REFERENCE), p)
        rows = list(csv.DictReader(p.open()))
        assert rows[0].get("timestamp", "") != ""


# ── to_dict / summary tests ────────────────────────────────────────────────────

class TestMetricResultHelpers:

    def test_to_dict_is_dict(self):
        r = score_answer(GOOD_ANSWER, REFERENCE)
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "rouge_l_f1" in d
        assert "bleu"       in d

    def test_summary_is_string(self):
        r = score_answer(GOOD_ANSWER, REFERENCE)
        s = r.summary
        assert isinstance(s, str)
        assert "ROUGE-L=" in s
        assert "BLEU="    in s

    def test_summary_for_empty_has_na(self):
        r = score_answer(EMPTY_ANSWER, REFERENCE)
        assert "n/a" in r.summary


# ── Multilingual smoke test ────────────────────────────────────────────────────

class TestMultilingual:

    def test_arabic_text(self):
        """ROUGE scorer should handle Arabic without crashing."""
        ref = "الإيرادات في الربع الثالث بلغت 4.2 مليار دولار."
        ans = "بلغت الإيرادات 4.2 مليار دولار في الفترة الثالثة."
        r = score_answer(ans, ref)
        # Should not raise; scores may be 0 for short Arabic text
        assert r.error is None or "empty" not in (r.error or "")

    def test_french_text(self):
        ref = "Le chiffre d'affaires du T3 a atteint 4,2 milliards de dollars."
        ans = "Au T3, le revenu était de 4,2 milliards, soit une hausse de 12%."
        r = score_answer(ans, ref)
        assert r.error is None or "empty" not in (r.error or "")
