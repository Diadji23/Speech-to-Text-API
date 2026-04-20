"""Tests pour src/evaluation/metrics.py."""
import pytest

from src.evaluation.metrics import compute_cer, compute_wer, evaluate


class TestWER:
    """Tests du Word Error Rate."""

    def test_perfect_match(self):
        assert compute_wer("bonjour le monde", "bonjour le monde") == 0.0

    def test_case_insensitive(self):
        assert compute_wer("Bonjour Le Monde", "bonjour le monde") == 0.0

    def test_one_substitution(self):
        wer = compute_wer("le chat dort", "le chien dort")
        assert abs(wer - 1 / 3) < 1e-6

    def test_one_deletion(self):
        wer = compute_wer("le chat dort", "le dort")
        assert abs(wer - 1 / 3) < 1e-6

    def test_one_insertion(self):
        wer = compute_wer("le chat dort", "le gros chat dort")
        assert abs(wer - 1 / 3) < 1e-6

    def test_completely_wrong(self):
        wer = compute_wer("bonjour le monde", "au revoir terre")
        assert wer == 1.0

    def test_empty_reference(self):
        assert compute_wer("", "bonjour") == float("inf")

    def test_both_empty(self):
        assert compute_wer("", "") == 0.0


class TestCER:
    """Tests du Character Error Rate."""

    def test_perfect_match(self):
        assert compute_cer("bonjour", "bonjour") == 0.0

    def test_one_char_substitution(self):
        cer = compute_cer("chat", "chal")
        assert abs(cer - 1 / 4) < 1e-6

    def test_one_char_deletion(self):
        cer = compute_cer("chat", "cha")
        assert abs(cer - 1 / 4) < 1e-6

    def test_empty_reference(self):
        assert compute_cer("", "abc") == float("inf")

    def test_both_empty(self):
        assert compute_cer("", "") == 0.0


class TestEvaluate:
    """Tests de la fonction evaluate complète."""

    def test_returns_eval_result(self):
        result = evaluate("le chat dort", "le chien dort")
        assert result.wer > 0
        assert result.cer > 0
        assert result.substitutions == 1
        assert result.insertions == 0
        assert result.deletions == 0
        assert result.num_ref_words == 3
        assert result.num_hyp_words == 3

    def test_perfect_transcription(self):
        result = evaluate("bonjour le monde", "bonjour le monde")
        assert result.wer == 0.0
        assert result.cer == 0.0
        assert result.substitutions == 0

    def test_deletion_counted(self):
        result = evaluate("le chat dort bien", "le dort bien")
        assert result.deletions == 1
        assert result.num_ref_words == 4
        assert result.num_hyp_words == 3
