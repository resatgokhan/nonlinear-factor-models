"""Diagnostics that compare learned, reference, and linked functions."""

from __future__ import annotations

import numpy as np

from bcd_lab.algorithms.state import init_state
from bcd_lab.diagnostics import DiagnosticContext, RunSnapshot
from bcd_lab.function_models import IdentityFunction
from bcd_lab.function_models.piecewise_linear_rkhs import PiecewiseLinearRKHS
from bcd_lab.function_models.rkhs import GaussianRKHS
from bcd_lab.types import (
    DatasetBundle,
    DatasetCapabilities,
    FunctionEvaluationDomain,
    FunctionReferences,
)

from .diagnostic_support import compute_metrics, diagnostic_context, model_config, sample_set


def test_linked_truth_metrics_are_restricted_to_the_function_domain() -> None:
    dataset = DatasetBundle(
        source="toy",
        n_rows=2,
        n_cols=2,
        rank=1,
        train=sample_set(),
        dense_truth=np.array([[-10.0, 0.0], [0.5, 10.0]]),
        capabilities=DatasetCapabilities(
            function_domain=FunctionEvaluationDomain(
                grid=np.array([0.0, 0.5]),
                source="test_domain",
                central_mass=0.5,
            ),
            linked_truth=np.ones((2, 2)),
        ),
    )
    snapshot = RunSnapshot(
        iteration=0,
        elapsed_seconds=0.0,
        capabilities={
            "inner_matrix": np.zeros((2, 2)),
            "linked_inner_matrix": np.array([[100.0, 1.0], [3.0, 100.0]]),
        },
    )

    metrics = compute_metrics(
        DiagnosticContext(snapshot=snapshot, dataset=dataset),
        {"truth_matrix": {"phi_x_error", "phi_x_rmse"}},
    )

    assert np.isclose(metrics["phi_x_error"], np.sqrt(2.0))
    assert np.isclose(metrics["phi_x_rmse"], np.sqrt(2.0))


def test_phi_diagnostics_compare_reference_functions() -> None:
    z = np.array([[1.0], [2.0], [1.0], [1.0]])
    truth = GaussianRKHS(1.0, np.array([0.0]), np.array([0.0]))
    phi = GaussianRKHS(1.0, np.array([0.0]), np.array([0.0]))
    dataset = DatasetBundle(
        source="toy",
        n_rows=2,
        n_cols=2,
        rank=1,
        train=sample_set(),
        capabilities=DatasetCapabilities(
            functions=FunctionReferences(truth=truth, initial=truth, sample_count=8),
        ),
    )
    config = model_config(link_kind="gaussian_rkhs")
    state = init_state(sample_set(), config, z_init=z, phi_init=phi)

    metrics = compute_metrics(
        diagnostic_context(state, config, dataset),
        {"phi": {"phi_diff", "phi_diff_0", "phi_sample_mse", "phi_d_in_star_bounds"}},
    )

    assert np.isclose(metrics["phi_diff"], 0.0)
    assert np.isclose(metrics["phi_diff_0"], 0.0)
    assert np.isclose(metrics["phi_sample_mse"], 0.0)
    assert np.isclose(metrics["phi_d_in_star_bounds"], 1.0)


def test_phi_diagnostics_use_dataset_evaluation_domain() -> None:
    dataset = DatasetBundle(
        source="toy",
        n_rows=2,
        n_cols=2,
        rank=1,
        train=sample_set(),
        capabilities=DatasetCapabilities(
            functions=FunctionReferences(truth=IdentityFunction()),
            function_domain=FunctionEvaluationDomain(
                grid=np.array([0.0, 0.5, 1.0]),
                source="test_domain",
                central_mass=0.95,
            ),
        ),
    )
    zero = GaussianRKHS(1.0, np.array([0.0]), np.array([0.0]))
    snapshot = RunSnapshot(
        iteration=0,
        elapsed_seconds=0.0,
        capabilities={"link_function": zero},
    )

    metrics = compute_metrics(
        DiagnosticContext(snapshot=snapshot, dataset=dataset),
        {"phi": {"phi_diff", "phi_sample_mse"}},
    )

    assert np.isclose(metrics["phi_diff"], np.sqrt(0.625))
    assert np.isclose(metrics["phi_sample_mse"], 1.25 / 3.0)


def test_piecewise_link_gaussian_projection_diagnostic() -> None:
    centers = np.array([-1.0, 0.0, 1.0])
    truth = GaussianRKHS(1.0, centers, np.array([0.2, -0.1, 0.3]))
    phi = PiecewiseLinearRKHS(1.0, (centers, truth.eval_array(centers)))
    dataset = DatasetBundle(
        source="toy",
        n_rows=2,
        n_cols=2,
        rank=1,
        train=sample_set(),
        capabilities=DatasetCapabilities(functions=FunctionReferences(truth=truth)),
    )
    config = model_config(link_kind="piecewise_linear_rkhs")
    state = init_state(sample_set(), config, z_init=np.ones((4, 1)), phi_init=phi)

    metrics = compute_metrics(
        diagnostic_context(state, config, dataset),
        {"phi": {"phi_gaussian_rkhs_error", "phi_gaussian_rkhs_norm"}},
    )

    assert np.isclose(metrics["phi_gaussian_rkhs_error"], 0.0, atol=1.0e-7)
    assert np.isclose(metrics["phi_gaussian_rkhs_norm"], truth.rkhs_norm(ridge=0.0))


def test_phi_diagnostics_require_typed_reference_capability() -> None:
    phi = GaussianRKHS(1.0, np.array([0.0]), np.array([0.0]))
    dataset = DatasetBundle(
        source="toy",
        n_rows=2,
        n_cols=2,
        rank=1,
        train=sample_set(),
        metadata={"phi_truth": phi},
    )
    config = model_config(link_kind="gaussian_rkhs")
    state = init_state(sample_set(), config, z_init=np.ones((4, 1)), phi_init=phi)

    metrics = compute_metrics(
        diagnostic_context(state, config, dataset),
        {"phi": {"phi_diff", "phi_sample_mse"}},
    )

    assert metrics == {"phi_diff": None, "phi_sample_mse": None}
