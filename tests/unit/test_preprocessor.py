"""Tests pour src/audio/preprocessor.py."""
import numpy as np
import pytest

from src.audio.preprocessor import Preprocessor


class TestNormalization:
    """Tests de la normalisation audio."""

    def test_normalizes_loud_audio(self, sine_audio):
        """Un signal fort est atténué."""
        proc = Preprocessor(vad_enabled=False, normalize=True, target_db=-20.0)
        output = proc.process(sine_audio, sample_rate=16000)
        assert np.max(np.abs(output)) < np.max(np.abs(sine_audio))

    def test_silent_audio_unchanged(self, silence_audio):
        """Le silence reste du silence."""
        proc = Preprocessor(vad_enabled=False, normalize=True)
        output = proc.process(silence_audio, sample_rate=16000)
        assert np.allclose(output, silence_audio)

    def test_stereo_to_mono(self):
        """Un signal stéréo est converti en mono."""
        stereo = np.random.randn(16000, 2).astype(np.float32) * 0.1
        proc = Preprocessor(vad_enabled=False, normalize=False)
        output = proc.process(stereo, sample_rate=16000)
        assert output.ndim == 1


class TestVAD:
    """Tests du Voice Activity Detection."""

    def test_keeps_speech_frames(self, mixed_audio):
        """Le VAD garde les frames avec signal et coupe le silence."""
        proc = Preprocessor(vad_enabled=True, vad_threshold=0.3, normalize=False)
        output = proc.process(mixed_audio, sample_rate=16000)
        # L'output doit être plus court (silences supprimés)
        assert len(output) < len(mixed_audio)
        # Mais pas vide (le signal est gardé)
        assert len(output) > 0

    def test_all_silence_returns_original(self, silence_audio):
        """Si tout est silence, retourne l'original (fallback)."""
        proc = Preprocessor(vad_enabled=True, vad_threshold=0.3, normalize=False)
        output = proc.process(silence_audio, sample_rate=16000)
        assert len(output) == len(silence_audio)

    def test_vad_disabled(self, mixed_audio):
        """Sans VAD, l'audio garde sa longueur."""
        proc = Preprocessor(vad_enabled=False, normalize=False)
        output = proc.process(mixed_audio, sample_rate=16000)
        assert len(output) == len(mixed_audio)


class TestEdgeCases:
    """Tests des cas limites."""

    def test_empty_audio_raises(self):
        """Un array vide lève ValueError."""
        proc = Preprocessor()
        with pytest.raises(ValueError, match="empty"):
            proc.process(np.array([], dtype=np.float32), sample_rate=16000)

    def test_invalid_sample_rate_raises(self, sine_audio):
        """Un sample_rate <= 0 lève ValueError."""
        proc = Preprocessor()
        with pytest.raises(ValueError, match="sample_rate"):
            proc.process(sine_audio, sample_rate=0)

    def test_very_short_audio(self):
        """Un audio plus court qu'un frame VAD passe quand même."""
        short = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        proc = Preprocessor(vad_enabled=True, normalize=False)
        output = proc.process(short, sample_rate=16000)
        assert len(output) == 3
