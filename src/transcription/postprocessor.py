"""
Postprocessor.

Nettoie le texte brut sorti de Whisper : suppression des mots
de remplissage (fillers), normalisation des espaces, et
formatage des timestamps.
"""
import logging
import re

logger = logging.getLogger(__name__)

DEFAULT_FILLERS_FR = {"euh", "hum", "bah", "ben", "genre", "voilà", "quoi", "hein"}


class Postprocessor:
    """
    Nettoyage post-transcription.

    Args:
        remove_filler_words: Supprimer les mots de remplissage.
        filler_words: Set de fillers à supprimer.
    """

    def __init__(
        self,
        remove_filler_words: bool = True,
        filler_words: set[str] | None = None,
    ):
        self.remove_filler_words = remove_filler_words
        self.filler_words = filler_words or DEFAULT_FILLERS_FR

    def process(self, text: str) -> str:
        """
        Applique le pipeline de nettoyage au texte.

        Args:
            text: Texte brut de Whisper.

        Returns:
            Texte nettoyé.
        """
        if not text.strip():
            return ""

        original = text

        if self.remove_filler_words:
            text = self._remove_fillers(text)

        text = self._normalize_whitespace(text)
        text = self._fix_punctuation_spacing(text)

        if text != original:
            logger.debug("Postprocessed: '%s...' → '%s...'", original[:40], text[:40])

        return text

    def _remove_fillers(self, text: str) -> str:
        """Supprime les mots de remplissage."""
        words = text.split()
        cleaned = [w for w in words if w.lower().strip(",.!?;:") not in self.filler_words]
        return " ".join(cleaned)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalise les espaces multiples et le trim."""
        return re.sub(r"\s+", " ", text).strip()

    def _fix_punctuation_spacing(self, text: str) -> str:
        """Corrige les espaces avant la ponctuation."""
        # Supprime l'espace avant . , ! ? ; :
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        # Assure un espace après la ponctuation (sauf fin de phrase)
        text = re.sub(r"([.,!?;:])([A-Za-zÀ-ÿ])", r"\1 \2", text)
        return text

    def process_segments(self, segments: list[dict]) -> list[dict]:
        """
        Applique le nettoyage à une liste de segments timestampés.

        Args:
            segments: Liste de {"start", "end", "text"}.

        Returns:
            Segments nettoyés (segments vides supprimés).
        """
        cleaned = []
        for seg in segments:
            text = self.process(seg["text"])
            if text:
                cleaned.append({**seg, "text": text})
        return cleaned
