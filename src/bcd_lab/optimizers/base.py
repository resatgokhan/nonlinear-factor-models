"""Public contracts for optimizers over fixed-coordinate NumPy arrays."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class OptimizerSpec:
    """Configured optimizer name and component-owned parameters."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ArrayOptimizer(Protocol):
    """Stateful optimizer for one fixed-shape floating-point array."""

    name: str
    state_version: int
    learning_rate: float

    def compute_update(self, gradient: np.ndarray) -> np.ndarray:
        """Advance optimizer state and return the update to subtract."""

    def checkpoint_state(self) -> Mapping[str, Any]:
        """Return the optimizer-owned versioned state payload."""

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        version: int,
    ) -> None:
        """Restore state for the parameter shape used at construction."""


class OptimizerBuilder(Protocol):
    """Construct a fresh optimizer instance for one parameter array."""

    def __call__(
        self,
        params: Mapping[str, Any],
        parameter: np.ndarray,
    ) -> ArrayOptimizer: ...
