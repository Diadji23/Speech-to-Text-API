"""
Benchmark tool.

Mesure la latence et la précision du pipeline STT de manière
reproductible. Produit un rapport avec statistiques (moyenne,
écart-type, min, max) sur N runs.
"""
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.evaluation.metrics import compute_cer, compute_wer

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Résultat d'un benchmark."""

    model_size: str
    device: str
    num_runs: int
    audio_duration: float

    # Latence (secondes)
    latency_mean: float
    latency_std: float
    latency_min: float
    latency_max: float

    # Realtime factor
    rtf_mean: float  # < 1 = plus rapide que temps réel

    # Précision (si référence fournie)
    wer: float | None = None
    cer: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"Model: whisper-{self.model_size} on {self.device}",
            f"Audio: {self.audio_duration:.2f}s | Runs: {self.num_runs}",
            f"Latency: {self.latency_mean:.3f}s ± {self.latency_std:.3f}s "
            f"(min={self.latency_min:.3f}s, max={self.latency_max:.3f}s)",
            f"RTF: {self.rtf_mean:.3f}x",
        ]
        if self.wer is not None:
            lines.append(f"WER: {self.wer:.2%} | CER: {self.cer:.2%}")
        return "\n".join(lines)


class Benchmark:
    """
    Benchmark du pipeline STT.

    Args:
        transcriber: Instance de Transcriber.
        num_runs: Nombre de runs pour moyenner la latence.
    """

    def __init__(self, transcriber, num_runs: int = 3):
        self.transcriber = transcriber
        self.num_runs = num_runs

    def run(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        reference: str | None = None,
    ) -> BenchmarkResult:
        """
        Exécute le benchmark.

        Args:
            audio: Signal audio numpy float32.
            sample_rate: Fréquence d'échantillonnage.
            reference: Texte de référence pour WER/CER (optionnel).

        Returns:
            BenchmarkResult avec statistiques.
        """
        audio_duration = len(audio) / sample_rate
        latencies = []
        last_text = ""

        logger.info(
            "Benchmarking %d runs on %.2fs audio...",
            self.num_runs,
            audio_duration,
        )

        for i in range(self.num_runs):
            t0 = time.time()
            result = self.transcriber.transcribe(audio, sample_rate)
            latency = time.time() - t0
            latencies.append(latency)
            last_text = result.text
            logger.info("Run %d/%d: %.3fs", i + 1, self.num_runs, latency)

        latencies = np.array(latencies)

        wer = compute_wer(reference, last_text) if reference else None
        cer = compute_cer(reference, last_text) if reference else None

        bench = BenchmarkResult(
            model_size=self.transcriber.model_size,
            device=self.transcriber.device,
            num_runs=self.num_runs,
            audio_duration=round(audio_duration, 3),
            latency_mean=round(float(latencies.mean()), 3),
            latency_std=round(float(latencies.std()), 3),
            latency_min=round(float(latencies.min()), 3),
            latency_max=round(float(latencies.max()), 3),
            rtf_mean=round(float(latencies.mean() / audio_duration), 3),
            wer=round(wer, 4) if wer is not None else None,
            cer=round(cer, 4) if cer is not None else None,
        )

        logger.info("Benchmark done:\n%s", bench.summary())
        return bench

    def save_report(self, result: BenchmarkResult, output_path: str) -> None:
        """Sauvegarde le rapport en JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info("Report saved: %s", path)
