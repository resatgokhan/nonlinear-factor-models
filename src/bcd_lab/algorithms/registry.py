"""Registry for algorithms, context, and protocol types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from bcd_lab._registry import ComponentRegistry
from bcd_lab.config import ModelConfig
from bcd_lab.diagnostics.registry import DiagnosticRequest
from bcd_lab.function_models import ScalarFunction
from bcd_lab.losses import Objective
from bcd_lab.types import (
    AlgorithmCheckpoint,
    AlgorithmResult,
    DatasetBundle,
    SelectionPolicy,
)


@dataclass
class AlgorithmContext:
    """Read-only algorithm run context container."""

    config: ModelConfig
    dataset: DatasetBundle
    objective: Objective
    plot_dir: str | None = None


class Algorithm(Protocol):
    """Protocol that all solver algorithms in bcd-lab must implement."""

    name: str

    def build_config(
        self,
        *,
        data: Mapping[str, Any],
        link: Mapping[str, Any],
        runtime: Mapping[str, Any],
        params: Mapping[str, Any],
    ) -> Any:
        """Build this algorithm's private run configuration."""

    def run(
        self,
        *,
        config: Any,
        dataset: DatasetBundle,
        objective: Objective,
        z_init: np.ndarray | None = None,
        phi_init: ScalarFunction | None = None,
        diagnostics_request: DiagnosticRequest | None = None,
        selection_policy: SelectionPolicy | None = None,
        rng: np.random.Generator,
    ) -> AlgorithmResult:
        """Run the optimization algorithm.

        Args:
            config: Solver-facing config.
            dataset: The input dataset bundle.
            objective: Configured data-fit objective.
            z_init: Optional pre-initialized factor matrix.
            phi_init: Optional pre-initialized function model.
            diagnostics_request: Optional diagnostic request and failure policy.
            selection_policy: Optional rule for retaining one observed run state.
            rng: Run-local random generator owned by the algorithm invocation.

        Returns:
            An AlgorithmResult envelope containing the final state, history, checkpoints, etc.
        """
        ...


@runtime_checkable
class DatasetCapabilityValidator(Protocol):
    """Optional algorithm hook for validating a loaded dataset before setup."""

    def validate_dataset(self, *, config: Any, dataset: DatasetBundle) -> None:
        """Raise a configuration error when a required capability is absent."""


@runtime_checkable
class CheckpointableAlgorithm(Protocol):
    """Algorithm-owned checkpoint capture and restoration contract."""

    name: str
    checkpoint_version: int

    def capture_checkpoint(
        self,
        state: Any,
        elapsed_seconds: float,
    ) -> AlgorithmCheckpoint: ...

    def restore_checkpoint(
        self,
        checkpoint: AlgorithmCheckpoint,
        *,
        config: Any,
        dataset: DatasetBundle | None = None,
    ) -> Any: ...


@runtime_checkable
class SelectedStateExporter(Protocol):
    """Optional algorithm hook for exporting a selected checkpoint for artifacts."""

    def export_selected_state(
        self,
        checkpoint: AlgorithmCheckpoint,
        *,
        config: Any,
        dataset: DatasetBundle,
    ) -> Mapping[str, Any]: ...


_ALGORITHMS = ComponentRegistry[Algorithm]("Algorithm", "algorithms")


def register_algorithm(algo: Algorithm) -> Algorithm:
    """Register an algorithm."""
    return _ALGORITHMS.register(algo.name, algo)


def get_algorithm(name: str) -> Algorithm:
    """Get an algorithm by name."""
    return _ALGORITHMS.get(name)


def list_algorithms() -> tuple[str, ...]:
    """List registered algorithm names."""
    return _ALGORITHMS.names()
