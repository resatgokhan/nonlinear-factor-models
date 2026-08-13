"""Registry for dataset preprocessors."""

from __future__ import annotations

from pathlib import Path

from bcd_lab._registry import ComponentRegistry

from .base import DataPreprocessor

_PREPROCESSORS = ComponentRegistry[DataPreprocessor]("Preprocessor", "preprocessors")


def register_preprocessor(preprocessor: DataPreprocessor) -> DataPreprocessor:
    """Register and return a dataset preprocessor."""
    return _PREPROCESSORS.register(preprocessor.name, preprocessor)


def get_preprocessor(name: str) -> DataPreprocessor:
    """Return a registered preprocessor by name."""
    return _PREPROCESSORS.get(name)


def list_preprocessors() -> tuple[str, ...]:
    """List registered preprocessor names."""
    return _PREPROCESSORS.names()


def preprocess_dataset(
    name: str,
    raw_dir: Path,
    out_dir: Path,
    **kwargs: object,
) -> Path:
    """Run the named preprocessor and return the clean CSV path."""
    preprocessor = get_preprocessor(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    return preprocessor.preprocess(raw_dir, out_dir, **kwargs)
