"""Tests pour src/transcription/postprocessor.py."""
import pytest

from src.transcription.postprocessor import Postprocessor


class TestFillerRemoval:
    """Tests de la suppression des mots de remplissage."""

    def test_removes_default_fillers(self):
        proc = Postprocessor(remove_filler_words=True)
        result = proc.process("euh je pense que hum c'est bon")
        assert "euh" not in result
        assert "hum" not in result
        assert "je pense que" in result

    def test_custom_fillers(self):
        proc = Postprocessor(filler_words={"alors", "donc"})
        result = proc.process("alors je suis donc parti")
        assert "alors" not in result
        assert "donc" not in result
        assert "je suis parti" in result

    def test_no_removal_when_disabled(self):
        proc = Postprocessor(remove_filler_words=False)
        result = proc.process("euh je pense que hum c'est bon")
        assert "euh" in result
        assert "hum" in result


class TestWhitespace:
    """Tests de la normalisation des espaces."""

    def test_multiple_spaces(self):
        proc = Postprocessor(remove_filler_words=False)
        result = proc.process("bonjour    le     monde")
        assert result == "bonjour le monde"

    def test_leading_trailing(self):
        proc = Postprocessor(remove_filler_words=False)
        result = proc.process("   bonjour   ")
        assert result == "bonjour"

    def test_empty_string(self):
        proc = Postprocessor()
        assert proc.process("") == ""
        assert proc.process("   ") == ""


class TestPunctuation:
    """Tests de la correction de ponctuation."""

    def test_removes_space_before_period(self):
        proc = Postprocessor(remove_filler_words=False)
        result = proc.process("bonjour . comment ça va ?")
        assert result == "bonjour. comment ça va?"

    def test_adds_space_after_period(self):
        proc = Postprocessor(remove_filler_words=False)
        result = proc.process("bonjour.comment")
        assert result == "bonjour. comment"


class TestSegments:
    """Tests du traitement par segments."""

    def test_cleans_segments(self):
        proc = Postprocessor()
        segments = [
            {"start": 0.0, "end": 1.0, "text": "euh bonjour"},
            {"start": 1.0, "end": 2.0, "text": "hum"},
            {"start": 2.0, "end": 3.0, "text": "ça va bien"},
        ]
        result = proc.process_segments(segments)
        # Le segment "hum" seul doit être supprimé
        assert len(result) == 2
        assert result[0]["text"] == "bonjour"
        assert result[1]["text"] == "ça va bien"
