"""
Audio file loader.

Charge un fichier audio depuis le disque et le convertit
en numpy float32 à la fréquence cible (16kHz pour Whisper).
Supporte wav, mp3, flac, ogg, m4a via ffmpeg.
"""
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}


def load_audio(file_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Charge un fichier audio et le convertit en mono float32.

    Utilise ffmpeg pour la conversion, ce qui supporte tous
    les formats courants sans dépendance lourde.

    Args:
        file_path: Chemin vers le fichier audio.
        target_sr: Fréquence d'échantillonnage cible en Hz.

    Returns:
        Tuple (audio_array, sample_rate).

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le format n'est pas supporté.
        RuntimeError: Si ffmpeg échoue.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {path.suffix}. "
            f"Supported: {SUPPORTED_FORMATS}"
        )

    logger.info("Loading audio: %s", path.name)

    try:
        cmd = [
            "ffmpeg",
            "-i", str(path),
            "-f", "s16le",         # PCM 16-bit signed little-endian
            "-acodec", "pcm_s16le",
            "-ac", "1",            # Mono
            "-ar", str(target_sr), # Resample
            "-loglevel", "error",
            "pipe:1",              # Output to stdout
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
        )

        audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
        audio /= 32768.0  # Normaliser int16 → float32 [-1, 1]

        duration = len(audio) / target_sr
        logger.info("Loaded: %.2fs, %d samples at %dHz", duration, len(audio), target_sr)

        return audio, target_sr

    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install it: sudo apt install ffmpeg"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode()}")
