"""
Configuration loader.

Charge le fichier YAML et expose les paramètres via dot-notation.
Supporte les overrides pour les expériences.
"""
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


class ConfigNode:
    """Accès dot-notation sur un dictionnaire imbriqué."""

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        return f"ConfigNode({vars(self)})"

    def to_dict(self) -> dict:
        result = {}
        for key, value in vars(self).items():
            if isinstance(value, ConfigNode):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(
    path: Optional[str] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> ConfigNode:
    """
    Charge une config YAML et applique les overrides.

    Args:
        path: Chemin vers le fichier YAML. None = default.yaml
        overrides: Dict {"section.key": value} pour surcharger.

    Returns:
        ConfigNode avec accès dot-notation.

    Raises:
        FileNotFoundError: Si le fichier YAML n'existe pas.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if overrides:
        for dotted_key, value in overrides.items():
            keys = dotted_key.split(".")
            target = raw
            for k in keys[:-1]:
                target = target[k]
            target[keys[-1]] = value

    logger.info("Config loaded from %s", config_path)
    return ConfigNode(raw)
