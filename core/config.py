"""
config.py
Utility helpers for saving and loading model configuration from JSON files.
The configuration captures model choices, hyperparameters, forecast horizon,
frequency, and preprocessing options so users can persist and reload settings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    """Save the provided configuration dictionary to disk as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load configuration JSON from disk; returns empty dict if missing."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def default_config() -> Dict[str, Any]:
    """Return a base configuration with sensible defaults."""
    return {
        "horizon": 12,
        "frequency": None,
        "preprocess": {"missing": "ffill"},
        "models": {
            "statsforecast": ["AutoARIMA", "AutoETS"],
            "mlforecast": ["RandomForest"],
            "neuralforecast": ["RNN"],
        },
        "backtest": {"windows": 2},
    }
