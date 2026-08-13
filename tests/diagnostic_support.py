"""Shared construction helpers for diagnostic boundary tests."""

from __future__ import annotations

import dataclasses

import numpy as np

from bcd_lab.algorithms.bcd_snapshot import publish_bcd_snapshot
from bcd_lab.algorithms.registry import AlgorithmContext
from bcd_lab.config import (
    LinkConfig,
    ModelConfig,
    PhiUpdateConfig,
    RegularizationConfig,
    RuntimeConfig,
    ShapeConfig,
)
from bcd_lab.diagnostics import (
    DiagnosticContext,
    DiagnosticRequest,
    compute_requested_diagnostics,
)
from bcd_lab.losses import get_objective
from bcd_lab.optimizers import OptimizerSpec
from bcd_lab.types import DatasetBundle, SampleSet


def sample_set() -> SampleSet:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    rows = np.array([0, 0, 1, 1])
    columns = np.array([0, 1, 0, 1])
    return values, (rows, columns)


def model_config(*, link_kind: str = "identity") -> ModelConfig:
    return ModelConfig(
        shape=ShapeConfig(n_rows=2, n_cols=2, rank=1, n_observations=4),
        z_optimizer=OptimizerSpec("gradient_descent", {"learning_rate": 0.1}),
        phi_update=PhiUpdateConfig(learning_rate=0.1),
        regularization=RegularizationConfig(lambda_=0.0, mu=1.0, alpha=0.0),
        link=LinkConfig(kind=link_kind, sigma=1.0),
        runtime=RuntimeConfig(max_iter=10, tol=1.0e-6),
    )


def diagnostic_context(
    state: object,
    config: ModelConfig,
    dataset: DatasetBundle,
    *,
    iteration: int = 0,
    elapsed: float = 0.0,
    objective_name: str | None = None,
    diagnostics_request: dict[str, set[str] | None] | None = None,
) -> DiagnosticContext:
    algorithm_context = AlgorithmContext(
        config=config,
        dataset=dataset,
        objective=get_objective("squared_error"),
    )
    if objective_name is not None:
        algorithm_context = dataclasses.replace(
            algorithm_context,
            objective=get_objective(objective_name),
        )
    snapshot = publish_bcd_snapshot(
        state,  # type: ignore[arg-type]
        algorithm_context,
        iteration,
        elapsed,
        0.0,
        0.0,
        diagnostics_request=DiagnosticRequest(
            families=diagnostics_request
            or {
                "objective": None,
                "truth_z": None,
                "truth_matrix": None,
                "timing": None,
                "gradients": None,
                "reconstruction": None,
                "phi": None,
                "phi_sharp": None,
            },
        ),
    )
    return DiagnosticContext(snapshot=snapshot, dataset=dataset)


def compute_metrics(
    context: DiagnosticContext,
    families: dict[str, set[str] | None],
) -> dict[str, float | int | str | None]:
    return compute_requested_diagnostics(
        context,
        DiagnosticRequest(families=families),
    ).metrics
