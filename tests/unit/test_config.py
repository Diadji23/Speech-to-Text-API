"""Tests pour src/config.py."""
import pytest

from src.config import load_config


def test_load_default_config():
    """Vérifie le chargement de default.yaml."""
    cfg = load_config()
    assert cfg.audio.sample_rate == 16000
    assert cfg.transcription.model_size == "base"
    assert cfg.evaluation.num_benchmark_runs == 3


def test_config_dot_notation():
    """Vérifie l'accès dot-notation imbriqué."""
    cfg = load_config()
    assert isinstance(cfg.transcription.language, str)
    assert cfg.preprocessing.vad_enabled is True


def test_config_overrides():
    """Vérifie que les overrides fonctionnent."""
    cfg = load_config(overrides={
        "transcription.model_size": "tiny",
        "audio.sample_rate": 8000,
    })
    assert cfg.transcription.model_size == "tiny"
    assert cfg.audio.sample_rate == 8000


def test_config_to_dict():
    """Vérifie la serialisation en dict."""
    cfg = load_config()
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert d["audio"]["sample_rate"] == 16000


def test_config_missing_key_raises():
    """Vérifie qu'une clé inexistante lève AttributeError."""
    cfg = load_config()
    with pytest.raises(AttributeError):
        _ = cfg.audio.nonexistent_key


def test_config_file_not_found():
    """Vérifie qu'un fichier inexistant lève FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("path/that/does/not/exist.yaml")
