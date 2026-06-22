"""I/O helpers and settings snapshots."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_npy(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, arr)


def load_npy(path: str | Path) -> np.ndarray:
    return np.load(path)


def save_settings_snapshot(output_dir: str | Path, settings_module: Any) -> Path:
    """Copy relevant settings vars to JSON for reproducibility."""
    out = Path(output_dir)
    ensure_dir(out)
    skip = {"__name__", "__doc__", "__package__", "__loader__", "__spec__", "__file__", "__cached__", "__builtins__"}
    data = {k: getattr(settings_module, k) for k in dir(settings_module) if not k.startswith("_") and k not in skip}
    # JSON-serialize non-primitives
    serializable = {}
    for k, v in data.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            serializable[k] = v
        elif isinstance(v, (list, tuple)):
            serializable[k] = list(v)
        elif isinstance(v, dict):
            serializable[k] = v
        else:
            serializable[k] = str(v)
    path = out / "settings_snapshot.json"
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    return path
