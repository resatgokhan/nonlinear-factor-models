"""Shared Matplotlib cache configuration for repo-local runs."""

from __future__ import annotations

import os
from pathlib import Path


def ensure_mplconfigdir() -> Path:
    """Point Matplotlib at a persistent cache directory inside the repo."""
    configured = os.environ.get("MPLCONFIGDIR")
    if configured:
        return Path(configured)

    cache_dir = Path(__file__).resolve().parents[3] / ".mplconfig"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    return cache_dir


def configure_matplotlib() -> None:
    """Configure cache and a headless backend before renderer imports."""
    ensure_mplconfigdir()
    if not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
        os.environ["MPLBACKEND"] = "Agg"


configure_matplotlib()
