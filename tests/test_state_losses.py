from __future__ import annotations

import numpy as np

from bcd_lab.algorithms.state import init_state
from bcd_lab.config import (
    LinkConfig,
    ModelConfig,
    PhiUpdateConfig,
    RegularizationConfig,
    RuntimeConfig,
    ShapeConfig,
)
from bcd_lab.function_models import IdentityFunction
from bcd_lab.initializers import build_zero_phi, svd_z
from bcd_lab.losses import get_objective, list_objectives
from bcd_lab.optimizers import OptimizerSpec
from bcd_lab.types import SampleSet


def _config(*, link_kind: str = "identity") -> ModelConfig:
    return ModelConfig(
        shape=ShapeConfig(n_rows=2, n_cols=2, rank=1, n_observations=4),
        z_optimizer=OptimizerSpec("gradient_descent", {"learning_rate": 0.1}),
        phi_update=PhiUpdateConfig(learning_rate=0.1),
        regularization=RegularizationConfig(lambda_=0.0, mu=1.0, alpha=0.0),
        link=LinkConfig(kind=link_kind, sigma=1.0),
        runtime=RuntimeConfig(max_iter=10, tol=1.0e-6),
    )


def _sample_set() -> SampleSet:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    return values, (rows, cols)


def test_init_state_accepts_explicit_z() -> None:
    z = np.array([[1.0], [2.0], [3.0], [4.0]])
    state = init_state(
        _sample_set(),
        _config(),
        z_init=z,
        phi_init=IdentityFunction(),
    )

    assert state.z.shape == (4, 1)
    assert np.allclose(state.centers, np.array([3.0, 4.0, 6.0, 8.0]))
    assert state.values.shape == (4,)


def test_svd_initializer_builds_z_outside_state() -> None:
    z = svd_z(_sample_set(), _config(), seed=0)
    state = init_state(
        _sample_set(),
        _config(link_kind="gaussian_rkhs"),
        z_init=z,
        phi_init=build_zero_phi(_config(link_kind="gaussian_rkhs")),
    )

    assert state.z.shape == (4, 1)
    assert state.phi is not None


def test_objective_registry_contains_builtin_losses() -> None:
    assert "squared_error" in list_objectives()
    assert "half_squared" in list_objectives()


def test_squared_error_matches_observed_residuals() -> None:
    z = np.array([[1.0], [2.0], [1.0], [1.0]])
    state = init_state(
        _sample_set(),
        _config(),
        z_init=z,
        phi_init=IdentityFunction(),
    )

    loss = get_objective("squared_error").value(state.values, state.centers)

    expected_predictions = np.array([1.0, 1.0, 2.0, 2.0])
    expected = float(np.sum((_sample_set()[0] - expected_predictions) ** 2))
    assert np.isclose(loss, expected)
