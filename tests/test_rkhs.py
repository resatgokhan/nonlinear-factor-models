from __future__ import annotations

import numpy as np
import pytest

from bcd_lab.function_models import (
    BallRadiusResolver,
    DifferentiableFunction,
    ProjectableFunction,
    TrainableFunction,
)
from bcd_lab.function_models.rkhs import GaussianRKHS, gaussian_kernel_matrix


def test_gaussian_rkhs_evaluates_and_differentiates() -> None:
    phi = GaussianRKHS(
        sigma=1.0,
        scalars=np.array([-1.0, 1.0]),
        coefs=np.array([0.5, -0.25]),
    )

    assert isinstance(phi, DifferentiableFunction)
    assert isinstance(phi, TrainableFunction)
    assert isinstance(phi, ProjectableFunction)
    assert isinstance(phi, BallRadiusResolver)

    xs = np.array([-0.5, 0.0, 0.5])
    values, derivs = phi.eval_and_d_array(xs)

    assert values.shape == xs.shape
    assert derivs.shape == xs.shape
    assert np.allclose(values, phi.eval_array(xs))
    assert np.allclose(derivs, phi.d_array(xs))


def test_model_projection_preserves_values_on_same_centers() -> None:
    centers = np.array([-1.0, 0.0, 1.0])
    phi = GaussianRKHS(
        sigma=0.8,
        scalars=centers,
        coefs=np.array([0.2, -0.1, 0.4]),
    )

    expected = phi.eval_array(centers)
    phi.project(centers=centers, rng=np.random.default_rng(0))

    assert np.allclose(phi.eval_array(centers), expected, atol=1.0e-6)


def test_gaussian_kernel_matrix_shape() -> None:
    kernel = gaussian_kernel_matrix(np.array([0.0, 1.0]), np.array([-1.0, 2.0, 3.0]), 1.0)
    assert kernel.shape == (2, 3)


def test_gaussian_rkhs_config() -> None:
    from bcd_lab.function_models.rkhs import GaussianRKHSConfig

    config = GaussianRKHSConfig(
        sigma=0.75,
        unconstrained_ridge=1.0e-5,
        constrained_percentage=0.5,
        constrained_lam=1.0e-3,
        lower=-2.0,
        upper=2.0,
    )

    phi = GaussianRKHS(
        sigma=0.75,
        scalars=np.array([0.0]),
        coefs=np.array([1.0]),
        config=config,
    )

    assert phi.config is not None
    assert phi.config.sigma == 0.75
    assert phi.config.unconstrained_ridge == 1.0e-5
    assert phi.config.constrained_percentage == 0.5
    assert phi.config.constrained_lam == 1.0e-3
    assert phi.config.lower == -2.0
    assert phi.config.upper == 2.0

    # Test copy preserves config
    phi_copied = phi.copy()
    assert phi_copied.config == config


def test_model_projection_uses_configured_ball() -> None:
    from bcd_lab.function_models.rkhs import GaussianRKHS, GaussianRKHSConfig

    phi_0 = GaussianRKHS(
        sigma=1.0,
        scalars=np.array([0.0]),
        coefs=np.array([0.0]),
    )

    config = GaussianRKHSConfig(
        sigma=1.0,
        unconstrained_ridge=1e-5,
        lower=-1.0,
        upper=1.0,
        constrained_lam=1e-4,
        ball_enabled=True,
        ball_center_phi_0=phi_0,
        ball_radius=0.2,
    )

    phi = GaussianRKHS(
        sigma=1.0,
        scalars=np.array([-0.5, 0.5]),
        coefs=np.array([1.0, -1.0]),
        config=config,
    )

    assert phi.diff_norm(phi_0) > config.ball_radius
    phi.project(centers=np.array([0.0]), rng=np.random.default_rng(0))

    assert phi.diff_norm(phi_0) <= config.ball_radius + 1.0e-8


def test_model_projection_enforces_configured_derivative_bounds() -> None:
    pytest.importorskip("cvxpy")
    from bcd_lab.function_models.rkhs import GaussianRKHSConfig

    config = GaussianRKHSConfig(
        sigma=1.0,
        constrained_percentage=1.0,
        lower=-1.0,
        upper=1.0,
    )
    phi = GaussianRKHS(
        sigma=1.0,
        scalars=np.array([-0.5, 0.5]),
        coefs=np.array([4.0, -4.0]),
        config=config,
    )
    rng_seed = 12
    expected_rng = np.random.default_rng(rng_seed)
    expected_rng.random()
    constrained_points = expected_rng.uniform(-2.5, 2.5, 200)

    phi.project(
        centers=np.array([-0.5, 0.5]),
        rng=np.random.default_rng(rng_seed),
    )

    derivatives = phi.d_array(constrained_points)
    assert np.all(derivatives >= config.lower - 1.0e-5)
    assert np.all(derivatives <= config.upper + 1.0e-5)
