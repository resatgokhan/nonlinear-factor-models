"""Base protocol and interfaces for data loading."""

from __future__ import annotations

from typing import Any, Protocol

from bcd_lab.types import DatasetBundle


class DatasetLoader(Protocol):
    """Protocol for dataset loaders to register with the data registry."""

    name: str

    def load(
        self,
        *,
        params: dict[str, Any],
        seed: int,
    ) -> DatasetBundle:
        """Load or generate a dataset bundle.

        Args:
            params: Dict of loader-specific parameters.
            seed: Random seed for reproducibility.

        Returns:
            A DatasetBundle containing train/validation sample sets and metadata.
        """
        ...
