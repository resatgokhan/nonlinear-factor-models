"""Grid preparation must precede residual evaluation, including after large drift."""

from __future__ import annotations

import dataclasses

import numpy as np

from bcd_lab.algorithms.bcd import _update_phi
from bcd_lab.algorithms.registry import AlgorithmContext
from bcd_lab.algorithms.state import init_state
from bcd_lab.config import LinkConfig, PhiUpdateConfig
from bcd_lab.data.finance import return_dataset
from bcd_lab.function_models import (
    FixedGridBounds,
    GridRebuildPolicy,
    InputQuantileGridBounds,
    PiecewiseLinearRKHS,
    PiecewiseLinearRKHSConfig,
)
from bcd_lab.losses import get_objective
from tests.test_algorithms import _config


def test_phi_residuals_use_rebuilt_grid_before_decay() -> None:
    dataset = return_dataset(np.array([[1.0, -1.0], [0.0, 2.0]]), rank=1)
    config = dataclasses.replace(
        _config(),
        link=LinkConfig("piecewise_linear_rkhs"),
        phi_update=PhiUpdateConfig(0.02),
    )
    function = PiecewiseLinearRKHS(
        1.0,
        (np.linspace(-0.5, 0.5, 5), np.array([0.0, 0.2, 1.0, 0.2, 0.0])),
        config=PiecewiseLinearRKHSConfig(
            count=5,
            bounds=InputQuantileGridBounds(),
            rebuild=GridRebuildPolicy("every_iteration"),
        ),
    )
    z = np.array([[2.0], [0.2], [1.0], [-1.0]])
    state = init_state(dataset.train, config, z_init=z, phi_init=function)
    expected = state.phi.copy()
    before = expected.eval_array(state.centers)
    expected.prepare_update(state.centers)
    rebuilt = expected.eval_array(state.centers)
    assert not np.allclose(before, rebuilt)
    residuals = 2 * (state.values - rebuilt)
    expected.scale_parameters(1 - config.phi_update.learning_rate * config.regularization.alpha)
    direction = expected.gradient(inputs=state.centers, residuals=residuals)
    expected.apply_gradient(direction, step_size=config.phi_update.learning_rate)
    _update_phi(
        state,
        AlgorithmContext(config=config, dataset=dataset, objective=get_objective("squared_error")),
        1,
        rng=np.random.default_rng(0),
    )
    np.testing.assert_allclose(state.phi.coefs, expected.coefs)
    np.testing.assert_array_equal(state.phi.scalars, expected.scalars)


def test_fixed_grid_preparation_is_a_noop() -> None:
    function = PiecewiseLinearRKHS(
        1.0,
        (np.linspace(-1, 1, 4), np.array([0.0, 1.0, 2.0, 3.0])),
        config=PiecewiseLinearRKHSConfig(count=4, bounds=FixedGridBounds(-1, 1)),
    )
    old_grid, old_values = function.scalars.copy(), function.coefs.copy()
    function.prepare_update(np.array([-100.0, 100.0]))
    np.testing.assert_array_equal(function.scalars, old_grid)
    np.testing.assert_array_equal(function.coefs, old_values)
