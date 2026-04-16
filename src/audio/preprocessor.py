"""
Audio preprocessor.

Applique le Voice Activity Detection (VAD) et la normalisation
avant de passer l'audio au transcriber. Filtre les silences
pour réduire le temps d'inférence et améliorer la précision.
"""
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """Segment audio avec ses métadonnées."""

    samples: np.ndarray
    sample_rate: int
    duration: float
    is_speech: bool = True


class Preprocessor:
    """
    Pipeline de preprocessing audio.

    Étapes :
        1. Chargement et conversion en numpy
        2. Normalisation du volume (peak normalization)
        3. Voice Activity Detection (VAD basé sur l'énergie RMS)
        4. Découpage en segments speech-only

    Args:
        vad_enabled: Activer la détection d'activité vocale.
        vad_threshold: Seuil d'énergie RMS (0-1) pour considérer un frame comme voix.
        normalize: Normaliser le volume audio.
        target_db: Niveau cible en dB pour la normalisation.
        frame_duration_ms: Durée d'un frame VAD en millisecondes.
    """

    def __init__(
        self,
        vad_enabled: bool = True,
        vad_threshold: float = 0.5,
        normalize: bool = True,
        target_db: float = -20.0,
        frame_duration_ms: int = 30,
    ):
        self.vad_enabled = vad_enabled
        self.vad_threshold = vad_threshold
        self.normalize = normalize
        self.target_db = target_db
        self.frame_duration_ms = frame_duration_ms

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Applique le pipeline complet de preprocessing.

        Args:
            audio: Signal audio en numpy float32 [-1, 1].
            sample_rate: Fréquence d'échantillonnage en Hz.

        Returns:
            Audio preprocessé (numpy float32).

        Raises:
            ValueError: Si l'audio est vide ou le sample_rate invalide.
        """
        if audio.size == 0:
            raise ValueError("Audio array is empty")
        if sample_rate <= 0:
            raise ValueError(f"Invalid sample_rate: {sample_rate}")

        logger.info(
            "Preprocessing: %.2fs audio at %dHz",
            len(audio) / sample_rate,
            sample_rate,
        )

        # Convertir en mono si stéréo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Normalisation
        if self.normalize:
            audio = self._normalize(audio)

        # VAD
        if self.vad_enabled:
            audio = self._apply_vad(audio, sample_rate)

        logger.info("Preprocessing done: %.2fs output", len(audio) / sample_rate)
        return audio

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Peak normalization vers le niveau cible en dB.

        Ramène le pic audio au target_db pour que Whisper
        reçoive un signal de volume constant, quel que soit
        le micro ou le fichier d'entrée.
        """
        peak = np.max(np.abs(audio))
        if peak == 0:
            logger.warning("Silent audio detected, skipping normalization")
            return audio

        target_amplitude = 10 ** (self.target_db / 20)
        gain = target_amplitude / peak
        normalized = audio * gain

        logger.debug("Normalization: peak=%.4f, gain=%.2f", peak, gain)
        return normalized

    def _apply_vad(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Voice Activity Detection basé sur l'énergie RMS.

        Découpe l'audio en frames de frame_duration_ms,
        calcule le RMS de chaque frame, et ne garde que les frames
        au-dessus du seuil. Le seuil est relatif au RMS max.
        """
        frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        num_frames = len(audio) // frame_size

        if num_frames == 0:
            return audio

        # Calcul RMS par frame
        frames = audio[: num_frames * frame_size].reshape(num_frames, frame_size)
        rms_values = np.sqrt(np.mean(frames ** 2, axis=1))

        # Seuil adaptatif relatif au max RMS
        max_rms = np.max(rms_values)
        if max_rms == 0:
            return audio

        threshold = self.vad_threshold * max_rms
        speech_mask = rms_values > threshold

        # Garder les frames speech
        speech_frames = frames[speech_mask]
        speech_ratio = speech_mask.sum() / num_frames

        logger.info(
            "VAD: %d/%d frames kept (%.0f%% speech)",
            speech_mask.sum(),
            num_frames,
            speech_ratio * 100,
        )

        if speech_frames.size == 0:
            logger.warning("VAD removed all audio — returning original")
            return audio

        return speech_frames.flatten()

    def compute_rms(self, audio: np.ndarray) -> float:
        """Calcule le RMS global du signal."""
        return float(np.sqrt(np.mean(audio ** 2)))

    def compute_duration(self, audio: np.ndarray, sample_rate: int) -> float:
        """Calcule la durée en secondes."""
        return len(audio) / sample_rate
