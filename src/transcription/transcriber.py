"""
Whisper transcriber.

Encapsule le modèle Whisper d'OpenAI avec gestion du device,
des options d'inférence, et retour structuré (texte, segments,
langue détectée, durée d'inférence).
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import whisper

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Résultat structuré d'une transcription."""

    text: str
    language: str
    segments: list[dict] = field(default_factory=list)
    inference_time: float = 0.0
    audio_duration: float = 0.0
    realtime_factor: float = 0.0  # inference_time / audio_duration


class Transcriber:
    """
    Wrapper Whisper avec gestion du device et options d'inférence.

    Args:
        model_size: Taille du modèle (tiny, base, small, medium, large).
        language: Code langue ISO (fr, en...) ou None pour auto-detect.
        device: "auto", "cpu", ou "cuda".
        fp16: Utiliser float16 (GPU only, plus rapide).
        beam_size: Largeur du beam search.
        temperature: 0 = greedy decoding, >0 = sampling.
    """

    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = "fr",
        device: str = "auto",
        fp16: bool = False,
        beam_size: int = 5,
        temperature: float = 0.0,
    ):
        self.model_size = model_size
        self.language = language
        self.beam_size = beam_size
        self.temperature = temperature

        # Résolution du device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.fp16 = fp16 and self.device == "cuda"

        logger.info(
            "Loading Whisper '%s' on %s (fp16=%s)",
            model_size,
            self.device,
            self.fp16,
        )
        t0 = time.time()
        self.model = whisper.load_model(model_size, device=self.device)
        logger.info("Model loaded in %.2fs", time.time() - t0)

    def transcribe(
        self,
        audio: np.ndarray | str,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """
        Transcrit un signal audio ou un fichier.

        Args:
            audio: Numpy float32 array ou chemin vers fichier audio.
            sample_rate: Fréquence d'échantillonnage (ignoré si path).

        Returns:
            TranscriptionResult avec texte, segments et métriques.

        Raises:
            RuntimeError: Si l'inférence Whisper échoue.
        """
        # Calculer durée audio
        if isinstance(audio, np.ndarray):
            audio_duration = len(audio) / sample_rate
        else:
            audio_duration = 0.0  # Sera calculé par Whisper

        logger.info("Transcribing %.2fs audio...", audio_duration)

        try:
            t0 = time.time()

            result = self.model.transcribe(
                audio,
                language=self.language,
                fp16=self.fp16,
                beam_size=self.beam_size,
                temperature=self.temperature,
            )

            inference_time = time.time() - t0

        except Exception as e:
            logger.error("Transcription failed: %s", e)
            raise RuntimeError(f"Whisper inference failed: {e}") from e

        # Construire le résultat structuré
        segments = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            }
            for seg in result.get("segments", [])
        ]

        # Durée réelle si on avait un fichier
        if audio_duration == 0.0 and segments:
            audio_duration = segments[-1]["end"]

        rtf = inference_time / audio_duration if audio_duration > 0 else 0.0

        transcription = TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language", self.language or "unknown"),
            segments=segments,
            inference_time=round(inference_time, 3),
            audio_duration=round(audio_duration, 3),
            realtime_factor=round(rtf, 3),
        )

        logger.info(
            "Transcribed: '%s...' (%.2fs inference, RTF=%.2f)",
            transcription.text[:50],
            inference_time,
            rtf,
        )

        return transcription
