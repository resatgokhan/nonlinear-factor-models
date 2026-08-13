"""Algorithm-neutral, read-only inputs published for one diagnostic pass."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


def readonly_array(values: object) -> np.ndarray:
    """Return a detached NumPy array that diagnostics cannot modify."""
    result = np.asarray(values).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ObservedSplit:
    """Aligned observed values and model predictions for one data split."""

    observed: np.ndarray
    predictions: np.ndarray

    def __post_init__(self) -> None:
        observed = readonly_array(self.observed)
        predictions = readonly_array(self.predictions)
        if observed.shape != predictions.shape:
            raise ValueError(
                "observed values and predictions must have matching shapes, got "
                f"{observed.shape} and {predictions.shape}"
            )
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "predictions", predictions)


@dataclass(frozen=True)
class RunSnapshot:
    """Immutable public view of an algorithm run at one iteration.

    Algorithms retain their private mutable state and publish only detached values.
    ``capabilities`` holds optional, named payloads for specialized diagnostics.
    Publishers are responsible for copying mutable capability values; NumPy arrays
    passed directly as capability values are detached and made read-only here.
    """

    iteration: int
    elapsed_seconds: float
    observed_splits: Mapping[str, ObservedSplit] = field(default_factory=dict)
    objective_value: float | None = None
    objective_name: str | None = None
    capabilities: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        splits = MappingProxyType(dict(self.observed_splits))
        capabilities = {
            name: readonly_array(value) if isinstance(value, np.ndarray) else value
            for name, value in self.capabilities.items()
        }
        object.__setattr__(self, "observed_splits", splits)
        object.__setattr__(self, "capabilities", MappingProxyType(capabilities))

    def capability(self, name: str, default: Any = None) -> Any:
        """Return an optional named capability published by the algorithm."""
        return self.capabilities.get(name, default)
