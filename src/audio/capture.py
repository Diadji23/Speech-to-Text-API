"""
Audio capture.

Capture audio depuis le microphone via PyAudio.
Supporte l'enregistrement avec durée fixe et le mode streaming
(callback) pour le temps réel.
"""
import logging
import struct
import tempfile
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class AudioCapture:
    """
    Capture audio depuis le microphone.

    Args:
        sample_rate: Fréquence d'échantillonnage (16000 pour Whisper).
        channels: Nombre de canaux (1 = mono).
        chunk_size: Nombre d'échantillons par buffer callback.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self._pa = None
        self._stream = None

    def _init_pyaudio(self):
        """Initialise PyAudio de façon lazy."""
        if self._pa is None:
            import pyaudio
            self._pa = pyaudio.PyAudio()

    def record(self, duration: float, output_path: str | None = None) -> np.ndarray:
        """
        Enregistre un segment audio de durée fixe.

        Args:
            duration: Durée en secondes.
            output_path: Si fourni, sauvegarde en .wav.

        Returns:
            Audio en numpy float32 [-1, 1].

        Raises:
            RuntimeError: Si le micro n'est pas accessible.
        """
        import pyaudio

        self._init_pyaudio()
        logger.info("Recording %.1fs at %dHz...", duration, self.sample_rate)

        try:
            stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )

            num_chunks = int(self.sample_rate / self.chunk_size * duration)
            frames = []

            for _ in range(num_chunks):
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)

            stream.stop_stream()
            stream.close()

        except OSError as e:
            raise RuntimeError(f"Microphone access failed: {e}")

        raw_data = b"".join(frames)
        audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
        audio /= 32768.0

        logger.info("Recorded %d samples (%.2fs)", len(audio), len(audio) / self.sample_rate)

        if output_path:
            self._save_wav(raw_data, output_path)

        return audio

    def _save_wav(self, raw_data: bytes, output_path: str) -> None:
        """Sauvegarde les données brutes en fichier WAV."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(raw_data)

        logger.info("Saved WAV: %s", path)

    def close(self) -> None:
        """Libère les ressources PyAudio."""
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
            logger.info("PyAudio terminated")
