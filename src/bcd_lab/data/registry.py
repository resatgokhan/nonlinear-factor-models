"""Registry for dataset loaders."""

from __future__ import annotations

from typing import Any

from bcd_lab._registry import ComponentRegistry
from bcd_lab.types import DatasetBundle

from .base import DatasetLoader

_LOADERS = ComponentRegistry[DatasetLoader]("Dataset loader", "dataset loaders")


def register_data_loader(loader: DatasetLoader) -> DatasetLoader:
    """Register a dataset loader."""
    return _LOADERS.register(loader.name, loader)


def get_data_loader(name: str) -> DatasetLoader:
    """Get a dataset loader by name."""
    return _LOADERS.get(name)


def list_data_loaders() -> tuple[str, ...]:
    """List registered dataset loader names."""
    return _LOADERS.names()


def load_dataset(name: str, params: dict[str, Any], seed: int) -> DatasetBundle:
    """Load a dataset by loader name, params, and seed."""
    loader = get_data_loader(name)
    dataset = loader.load(params=params, seed=seed)
    try:
        dataset.validate()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Dataset loader {name!r} returned an invalid bundle: {exc}") from exc
    return dataset
