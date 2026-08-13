"""Algorithm implementations and registries."""

from __future__ import annotations

from .bcd import BcdAlgorithm
from .bcd_loop import run_bcd_iterations
from .numerics import BcdNumericalError
from .registry import (
    Algorithm,
    AlgorithmContext,
    CheckpointableAlgorithm,
    DatasetCapabilityValidator,
    SelectedStateExporter,
    get_algorithm,
    list_algorithms,
    register_algorithm,
)

# Built-ins are wired here in one visible, deterministic order. External
# components use the public register_algorithm() API.
register_algorithm(BcdAlgorithm())

__all__ = [
    "Algorithm",
    "AlgorithmContext",
    "CheckpointableAlgorithm",
    "DatasetCapabilityValidator",
    "SelectedStateExporter",
    "BcdNumericalError",
    "run_bcd_iterations",
    "register_algorithm",
    "get_algorithm",
    "list_algorithms",
]
