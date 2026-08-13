"""Plot artifact output service."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure


def save_figure(figure: Figure, path: Path, *, dpi: int | None = None) -> Path:
    """Persist one figure, creating only its immediate artifact directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    return path
