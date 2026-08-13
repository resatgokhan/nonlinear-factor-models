from __future__ import annotations

import numpy as np
import pytest

from bcd_lab.function_models import (
    BallRadiusResolver,
    BinnedPiecewiseLinearRKHS,
    CopyableFunction,
    DifferentiableFunction,
    GridRebuildPolicy,
    IdentityFunction,
    InputQuantileGridBounds,
    PiecewiseLinearRKHS,
    PiecewiseLinearRKHSConfig,
    ProjectableFunction,
    ScalarFunction,
    SerializableFunctionModel,
    SigmoidFunction,
    TrainableFunction,
)
from bcd_lab.function_models.distances import function_distance
from bcd_lab.function_models.rkhs import GaussianRKHSConfig, gaussian_kernel_matrix


def test_identity_function_model() -> None:
    phi = IdentityFunction()
    xs = np.array([-1.0, 0.0, 2.0])

    assert np.allclose(phi.eval_array(xs), xs)
    assert np.allclose(phi.d_array(xs), np.ones_like(xs))
    assert isinstance(phi, ScalarFunction)
    assert isinstance(phi, CopyableFunction)
    assert isinstance(phi, DifferentiableFunction)
    assert isinstance(phi, SerializableFunctionModel)
    assert not isinstance(phi, TrainableFunction)
    assert not isinstance(phi, ProjectableFunction)
    assert not hasattr(phi, "fit")
    assert not hasattr(phi, "gradient")
    assert not hasattr(phi, "project")
    assert phi.to_payload() == {"kind": "identity"}


def test_piecewise_linear_function_model() -> None:
    phi = PiecewiseLinearRKHS(
        sigma=0.8,
        grid_values=(np.array([-1.0, 0.0, 1.0]), np.array([-2.0, 0.0, 2.0])),
    )
    xs = np.array([-0.5, 0.5])

    values, derivs = phi.eval_and_d_array(xs)

    assert np.allclose(values, np.array([-1.0, 1.0]))
    assert np.allclose(derivs, np.array([2.0, 2.0]))
    assert np.isclose(phi.distance_to(phi.copy()), 0.0)
    assert np.isclose(phi.rkhs_norm(), np.sqrt(8.0 + 8.0))
    assert isinstance(phi, TrainableFunction)
    assert isinstance(phi, ProjectableFunction)
    assert isinstance(phi, BallRadiusResolver)


def test_piecewise_linear_gradient_and_projection() -> None:
    cfg = PiecewiseLinearRKHSConfig(
        sigma=1.0,
        monotone_enabled=True,
        monotone_direction="increasing",
        clamp_enabled=True,
        clamp_min=0.0,
        clamp_max=3.0,
    )
    phi = PiecewiseLinearRKHS(
        sigma=1.0,
        grid_values=(np.array([0.0, 1.0, 2.0]), np.array([0.0, 2.0, 1.0])),
        config=cfg,
    )

    grad = phi.gradient(inputs=np.array([0.5, 1.5]), residuals=np.array([2.0, 2.0]))
    expected_grad = gaussian_kernel_matrix(
        phi.scalars,
        np.array([0.5, 1.5]),
        phi.sigma,
    ) @ np.array([2.0, 2.0])
    assert np.allclose(grad, expected_grad)

    phi.apply_gradient(grad, step_size=0.5)
    phi.project()

    assert np.all(np.diff(phi.values) >= -1.0e-12)
    assert np.all(phi.values >= 0.0)
    assert np.all(phi.values <= 3.0)


def test_piecewise_linear_l2_ball_projection_matches_speed_phi_semantics() -> None:
    center = PiecewiseLinearRKHS(
        sigma=1.0,
        grid_values=(np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0])),
    )
    cfg = PiecewiseLinearRKHSConfig(
        sigma=1.0,
        ball_enabled=True,
        ball_center_phi_0=center,
        ball_radius=1.0,
    )
    phi = PiecewiseLinearRKHS(
        sigma=1.0,
        grid_values=(np.array([0.0, 1.0, 2.0]), np.array([2.0, 2.0, 2.0])),
        config=cfg,
    )

    phi.project()

    assert np.isclose(phi.diff_norm(center), 1.0)


def test_piecewise_linear_config_from_link_config_uses_config_defaults() -> None:
    from bcd_lab.config import LinkConfig

    cfg = PiecewiseLinearRKHSConfig.from_link_config(
        LinkConfig(
            kind="piecewise_linear_rkhs",
            sigma=2.0,
            params={
                "grid": {
                    "count": 50,
                    "bounds": {"name": "input_quantiles", "params": {}},
                    "rebuild": {"name": "never", "params": {}},
                }
            },
        ),
    )

    assert cfg.count == 50
    assert cfg.bounds == InputQuantileGridBounds()
    assert cfg.rebuild == GridRebuildPolicy(name="never")


def test_gaussian_rkhs_config_from_link_config_uses_config_defaults() -> None:
    from bcd_lab.config import LinkConfig

    cfg = GaussianRKHSConfig.from_link_config(
        LinkConfig(kind="gaussian_rkhs", sigma=2.0, params={}),
    )
    defaults = GaussianRKHSConfig(sigma=2.0)

    assert cfg == defaults


def test_function_model_config_reads_ball_settings_from_link_params() -> None:
    from bcd_lab.config import LinkConfig

    center = PiecewiseLinearRKHS(
        sigma=1.0,
        grid_values=(np.array([0.0, 1.0]), np.array([0.0, 0.0])),
    )
    cfg = PiecewiseLinearRKHSConfig.from_link_config(
        LinkConfig(
            kind="piecewise_linear_rkhs",
            sigma=1.0,
            params={
                "ball_enabled": True,
                "ball_radius": 2.0,
                "phi_0": center,
            },
        ),
    )

    assert cfg.ball_enabled is True
    assert cfg.ball_radius == 2.0
    assert cfg.ball_center_phi_0 is center


def test_parametric_function_model_distance() -> None:
    phi = SigmoidFunction()
    assert np.isclose(function_distance(phi, phi.copy()), 0.0)


def test_function_model_registry_resolution() -> None:
    from bcd_lab.config import (
        LinkConfig,
        ModelConfig,
        PhiType,
        PhiUpdateConfig,
        RegularizationConfig,
        RuntimeConfig,
        ShapeConfig,
    )
    from bcd_lab.function_models import get_function_model_builder
    from bcd_lab.optimizers import OptimizerSpec

    config = ModelConfig(
        shape=ShapeConfig(n_rows=2, n_cols=2, rank=1, n_observations=4),
        z_optimizer=OptimizerSpec("gradient_descent", {"learning_rate": 0.1}),
        phi_update=PhiUpdateConfig(learning_rate=0.1),
        regularization=RegularizationConfig(lambda_=0.0, mu=1.0, alpha=0.0),
        link=LinkConfig(kind="identity", sigma=1.0),
        runtime=RuntimeConfig(max_iter=10, tol=1.0e-6),
    )

    for kind in [
        "identity",
        "sigmoid",
        "exponential",
        "piecewise_linear_rkhs",
        "binned_piecewise_linear_rkhs",
        "gaussian_rkhs",
    ]:
        builder = get_function_model_builder(kind)
        assert builder is not None
        phi = builder(config)
        assert phi is not None

    with pytest.raises(KeyError):
        get_function_model_builder("speed")
    with pytest.raises(KeyError):
        get_function_model_builder("piecewise_linear_binned_rkhs")

    assert PhiType.from_str("piecewise_linear_rkhs") == PhiType.PIECEWISE_LINEAR_RKHS
    assert PhiType.from_str("binned_piecewise_linear_rkhs") == PhiType.BINNED_PIECEWISE_LINEAR_RKHS
    assert PhiType.from_str("piecewise_linear_binned_rkhs") == "piecewise_linear_binned_rkhs"


def test_trainable_function_fit() -> None:
    phi_pl = PiecewiseLinearRKHS(
        sigma=1.0,
        grid_values=(np.array([-1.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0])),
    )
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([-2.0, 1.5, 3.0])
    phi_pl.fit(x, y)
    assert np.allclose(phi_pl.eval_array(x), y)

    from bcd_lab.function_models.rkhs import GaussianRKHS

    phi_rkhs = GaussianRKHS(sigma=1.0, scalars=np.array([0.0]), coefs=np.array([0.0]))
    x_rkhs = np.array([-0.5, 0.5])
    y_rkhs = np.array([1.0, 2.0])
    phi_rkhs.fit(x_rkhs, y_rkhs)
    assert np.allclose(phi_rkhs.eval_array(x_rkhs), y_rkhs)


def test_piecewise_linear_input_quantile_grid_can_remain_fixed_once() -> None:
    cfg = PiecewiseLinearRKHSConfig(
        sigma=1.0,
        count=5,
        bounds=InputQuantileGridBounds(
            lower_quantile=0.01,
            upper_quantile=0.99,
            pad_fraction=0.25,
            min_padding=0.5,
        ),
        rebuild=GridRebuildPolicy(name="never"),
    )
    phi = PiecewiseLinearRKHS(sigma=1.0, config=cfg)

    inputs = np.arange(101, dtype=float)
    phi.prepare_grid(inputs)
    initial_grid = phi.scalars.copy()
    assert np.isclose(initial_grid[0], -23.5)
    assert np.isclose(initial_grid[-1], 123.5)

    phi.prepare_grid(inputs + 1000.0)
    assert np.array_equal(phi.scalars, initial_grid)


def test_piecewise_linear_every_iteration_rebuild_interpolates_values() -> None:
    cfg = PiecewiseLinearRKHSConfig(
        sigma=1.0,
        count=21,
        bounds=InputQuantileGridBounds(
            lower_quantile=0.0,
            upper_quantile=1.0,
            min_padding=0.1,
        ),
        rebuild=GridRebuildPolicy(name="every_iteration"),
    )
    phi = PiecewiseLinearRKHS(sigma=1.0, config=cfg)

    phi.prepare_grid(np.array([0.0, 1.0]))
    phi.coefs[:] = phi.scalars
    phi.prepare_grid(np.array([10.0, 11.0]))

    assert np.isclose(phi.scalars[0], 9.75)
    assert np.isclose(phi.scalars[-1], 11.25)
    assert np.allclose(phi.coefs, 1.25)


def test_piecewise_linear_adaptive_grid_rebuilds_when_resolution_is_lost() -> None:
    cfg = PiecewiseLinearRKHSConfig(
        sigma=1.0,
        count=101,
        bounds=InputQuantileGridBounds(
            lower_quantile=0.0,
            upper_quantile=1.0,
            min_padding=0.5,
        ),
        rebuild=GridRebuildPolicy(
            name="adaptive",
            phi_tolerance=1.0e9,
            min_active_fraction=0.25,
        ),
    )
    phi = PiecewiseLinearRKHS(sigma=1.0, config=cfg)

    phi.prepare_grid(np.array([0.0, 1.0]))
    initial_grid = phi.scalars.copy()

    phi.prepare_grid(np.array([0.45, 0.55]))

    assert not np.allclose(phi.scalars, initial_grid)
    assert (phi.scalars[-1] - phi.scalars[0]) < (initial_grid[-1] - initial_grid[0])


def test_piecewise_linear_grid_config_rejects_old_or_incoherent_fields() -> None:
    from bcd_lab.config import LinkConfig

    with pytest.raises(ValueError, match="unknown piecewise_linear_rkhs parameter.*grid_mode"):
        PiecewiseLinearRKHSConfig.from_link_config(
            LinkConfig(
                kind="piecewise_linear_rkhs",
                params={"grid_mode": "dynamic"},
            )
        )
    with pytest.raises(ValueError, match="fixed grid bounds require"):
        PiecewiseLinearRKHSConfig.from_link_config(
            LinkConfig(
                kind="piecewise_linear_rkhs",
                params={
                    "grid": {
                        "bounds": {"name": "fixed", "params": {}},
                        "rebuild": {"name": "adaptive", "params": {}},
                    }
                },
            )
        )


def test_piecewise_linear_checkpoint_restores_resolved_quantile_grid_exactly() -> None:
    config = PiecewiseLinearRKHSConfig(
        sigma=1.0,
        count=11,
        bounds=InputQuantileGridBounds(),
        rebuild=GridRebuildPolicy(name="every_iteration"),
    )
    original = PiecewiseLinearRKHS(sigma=1.0, config=config, dtype=np.float32)
    original.prepare_grid(np.arange(101, dtype=float))
    original.coefs[:] = np.linspace(-1.0, 1.0, 11, dtype=np.float32)

    restored = PiecewiseLinearRKHS(sigma=1.0, config=config, dtype=np.float32)
    restored.restore_checkpoint_state(
        original.checkpoint_state(),
        version=original.checkpoint_version,
    )

    assert np.array_equal(restored.scalars, original.scalars)
    assert np.array_equal(restored.coefs, original.coefs)
    assert restored._resolved_input_bounds == original._resolved_input_bounds
    with pytest.raises(ValueError, match="unsupported piecewise-linear state version"):
        restored.restore_checkpoint_state(original.checkpoint_state(), version=1)


def test_binned_piecewise_linear_gradient_matches_exact_on_grid_points() -> None:
    grid = np.array([-1.0, 0.0, 1.0])
    values = np.zeros_like(grid)
    residuals = np.array([2.0, -1.0, 0.5])

    exact = PiecewiseLinearRKHS(
        sigma=0.8,
        grid_values=(grid, values),
    )
    binned = BinnedPiecewiseLinearRKHS(
        sigma=0.8,
        grid_values=(grid, values),
    )

    assert binned.kind == "binned_piecewise_linear_rkhs"
    assert np.allclose(
        binned.gradient(inputs=grid, residuals=residuals),
        exact.gradient(inputs=grid, residuals=residuals),
    )


def test_binned_piecewise_linear_regrid_clears_kernel_cache() -> None:
    cfg = PiecewiseLinearRKHSConfig(
        sigma=1.0,
        count=21,
        bounds=InputQuantileGridBounds(
            lower_quantile=0.0,
            upper_quantile=1.0,
            min_padding=0.1,
        ),
        rebuild=GridRebuildPolicy(name="every_iteration"),
    )
    phi = BinnedPiecewiseLinearRKHS(sigma=1.0, config=cfg)

    phi.gradient(inputs=np.array([0.0, 1.0]), residuals=np.array([1.0, 1.0]))
    first_kernel = phi._K_grid

    phi.gradient(inputs=np.array([10.0, 11.0]), residuals=np.array([1.0, 1.0]))

    assert phi._K_grid is not first_kernel
    assert np.isclose(phi.scalars[0], 9.75)
    assert np.isclose(phi.scalars[-1], 11.25)
