"""Cache existence checks."""

from __future__ import annotations

from pathlib import Path


def exists(path: str | Path) -> bool:
    return Path(path).exists()


def should_skip(path: str | Path, force_recompute: bool) -> bool:
    return exists(path) and not force_recompute
