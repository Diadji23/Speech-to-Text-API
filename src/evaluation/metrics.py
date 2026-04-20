"""
Evaluation metrics.

Implémente le Word Error Rate (WER) et le Character Error Rate (CER)
pour mesurer la qualité de transcription. Utilise l'algorithme
de distance d'édition de Levenshtein.
"""
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Résultat d'évaluation d'une transcription."""

    wer: float           # Word Error Rate (0-1, plus bas = meilleur)
    cer: float           # Character Error Rate (0-1)
    num_ref_words: int   # Nombre de mots dans la référence
    num_hyp_words: int   # Nombre de mots dans l'hypothèse
    substitutions: int
    insertions: int
    deletions: int


def _levenshtein(ref: list, hyp: list) -> tuple[int, int, int, int]:
    """
    Distance d'édition de Levenshtein.

    Calcule le nombre minimum d'opérations (substitution, insertion,
    deletion) pour transformer hyp en ref.

    Args:
        ref: Séquence de référence (ground truth).
        hyp: Séquence hypothèse (prédiction).

    Returns:
        Tuple (distance, substitutions, insertions, deletions).
    """
    n, m = len(ref), len(hyp)

    # Matrice de programmation dynamique
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # substitution
                    dp[i - 1][j],       # deletion
                    dp[i][j - 1],       # insertion
                )

    # Backtrack pour compter S, I, D
    i, j = n, m
    subs, ins, dels = 0, 0, 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            dels += 1
            i -= 1

    return dp[n][m], subs, ins, dels


def _normalize_text(text: str) -> str:
    """Normalise le texte pour l'évaluation (lowercase, strip)."""
    return text.lower().strip()


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Calcule le Word Error Rate.

    WER = (S + I + D) / N
    où S=substitutions, I=insertions, D=deletions, N=mots dans ref.

    Args:
        reference: Texte de référence (ground truth).
        hypothesis: Texte transcrit (prédiction).

    Returns:
        WER entre 0 et +inf (>1 possible si beaucoup d'insertions).
    """
    ref_words = _normalize_text(reference).split()
    hyp_words = _normalize_text(hypothesis).split()

    if not ref_words:
        return 0.0 if not hyp_words else float("inf")

    distance, _, _, _ = _levenshtein(ref_words, hyp_words)
    return distance / len(ref_words)


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Calcule le Character Error Rate.

    Même formule que WER mais au niveau caractère.

    Args:
        reference: Texte de référence.
        hypothesis: Texte transcrit.

    Returns:
        CER entre 0 et +inf.
    """
    ref_chars = list(_normalize_text(reference))
    hyp_chars = list(_normalize_text(hypothesis))

    if not ref_chars:
        return 0.0 if not hyp_chars else float("inf")

    distance, _, _, _ = _levenshtein(ref_chars, hyp_chars)
    return distance / len(ref_chars)


def evaluate(reference: str, hypothesis: str) -> EvalResult:
    """
    Évaluation complète d'une transcription.

    Args:
        reference: Texte de référence (ground truth).
        hypothesis: Texte transcrit (prédiction).

    Returns:
        EvalResult avec WER, CER et détail des erreurs.
    """
    ref_words = _normalize_text(reference).split()
    hyp_words = _normalize_text(hypothesis).split()

    if not ref_words:
        wer = 0.0 if not hyp_words else float("inf")
        cer = compute_cer(reference, hypothesis)
        return EvalResult(
            wer=wer, cer=cer,
            num_ref_words=0, num_hyp_words=len(hyp_words),
            substitutions=0, insertions=len(hyp_words), deletions=0,
        )

    distance, subs, ins, dels = _levenshtein(ref_words, hyp_words)
    wer = distance / len(ref_words)
    cer = compute_cer(reference, hypothesis)

    result = EvalResult(
        wer=round(wer, 4),
        cer=round(cer, 4),
        num_ref_words=len(ref_words),
        num_hyp_words=len(hyp_words),
        substitutions=subs,
        insertions=ins,
        deletions=dels,
    )

    logger.info(
        "Eval: WER=%.2f%% CER=%.2f%% (S=%d I=%d D=%d)",
        result.wer * 100,
        result.cer * 100,
        subs,
        ins,
        dels,
    )

    return result
