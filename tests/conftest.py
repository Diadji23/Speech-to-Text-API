"""
Fixtures partagées pour les tests.

Fournit des mini samples audio synthétiques et des configs de test
pour éviter de charger Whisper dans les tests unitaires.
"""
import numpy as np
import pytest

from src.config import load_config


@pytest.fixture
def test_config():
    """Config par défaut pour les tests."""
    return load_config()


@pytest.fixture
def silence_audio():
    """1 seconde de silence à 16kHz."""
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def sine_audio():
    """1 seconde de sinusoïde 440Hz à 16kHz (simule de la voix)."""
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def noisy_audio():
    """1 seconde de bruit blanc (pas de voix)."""
    rng = np.random.default_rng(42)
    return rng.uniform(-0.1, 0.1, 16000).astype(np.float32)


@pytest.fixture
def mixed_audio():
    """2 secondes : 0.5s silence + 1s signal + 0.5s silence."""
    sr = 16000
    silence = np.zeros(int(0.5 * sr), dtype=np.float32)
    t = np.linspace(0, 1, sr, dtype=np.float32)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    return np.concatenate([silence, signal, silence])
