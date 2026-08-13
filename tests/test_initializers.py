from __future__ import annotations

import numpy as np

from bcd_lab.config import (
    LinkConfig,
    ModelConfig,
    PhiUpdateConfig,
    RegularizationConfig,
    RuntimeConfig,
    ShapeConfig,
)
from bcd_lab.initializers import (
    InitializationSpec,
    InitializerSpec,
    build_initialization,
    list_joint_initializers,
    list_phi_initializers,
    list_z_initializers,
)
from bcd_lab.optimizers import OptimizerSpec
from bcd_lab.types import DatasetBundle, SampleSet


def _sample_set() -> SampleSet:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    return values, (rows, cols)


def _dataset() -> DatasetBundle:
    return DatasetBundle(source="toy", n_rows=2, n_cols=2, rank=1, train=_sample_set())


def _config() -> ModelConfig:
    return ModelConfig(
        shape=ShapeConfig(n_rows=2, n_cols=2, rank=1, n_observations=4),
        z_optimizer=OptimizerSpec("gradient_descent", {"learning_rate": 0.1}),
        phi_update=PhiUpdateConfig(learning_rate=0.1),
        regularization=RegularizationConfig(lambda_=0.0, mu=1.0, alpha=0.0),
        link=LinkConfig(kind="gaussian_rkhs", sigma=1.0),
        runtime=RuntimeConfig(max_iter=10, tol=1.0e-6),
    )


def test_separate_initialization_builds_z_and_phi() -> None:
    z_init, phi_init = build_initialization(
        InitializationSpec(
            mode="separate",
            z=InitializerSpec(name="svd"),
            phi=InitializerSpec(name="zero"),
        ),
        dataset=_dataset(),
        config=_config(),
        seed=0,
    )

    assert z_init is not None
    assert z_init.shape == (4, 1)
    assert phi_init is not None


def test_joint_initialization_builds_bundle() -> None:
    z_init, phi_init = build_initialization(
        InitializationSpec(
            mode="joint",
            joint=InitializerSpec(name="svd_with_zero_phi"),
        ),
        dataset=_dataset(),
        config=_config(),
        seed=0,
    )

    assert z_init is not None
    assert phi_init is not None


def test_initializers_are_namespaced() -> None:
    assert "svd" in list_z_initializers()
    assert "zero" in list_phi_initializers()
    assert "identity_phi" in list_phi_initializers()
    assert "identity" in list_phi_initializers()
    assert "sigmoid" in list_phi_initializers()
    assert "exponential" in list_phi_initializers()
    assert "svd_with_zero_phi" in list_joint_initializers()


def test_sigmoid_and_exponential_initializers() -> None:
    # Test sigmoid
    config_sig = ModelConfig(
        shape=ShapeConfig(n_rows=2, n_cols=2, rank=1, n_observations=4),
        z_optimizer=OptimizerSpec("gradient_descent", {"learning_rate": 0.1}),
        phi_update=PhiUpdateConfig(learning_rate=0.1),
        regularization=RegularizationConfig(lambda_=0.0, mu=1.0, alpha=0.0),
        link=LinkConfig(kind="sigmoid", sigma=1.0),
        runtime=RuntimeConfig(max_iter=10, tol=1.0e-6),
    )
    z_sig, phi_sig = build_initialization(
        InitializationSpec(
            mode="separate",
            z=InitializerSpec(name="svd"),
            phi=InitializerSpec(name="sigmoid"),
        ),
        dataset=_dataset(),
        config=config_sig,
        seed=0,
    )
    assert phi_sig is not None
    assert phi_sig.kind == "sigmoid"

    # Test exponential
    config_exp = ModelConfig(
        shape=ShapeConfig(n_rows=2, n_cols=2, rank=1, n_observations=4),
        z_optimizer=OptimizerSpec("gradient_descent", {"learning_rate": 0.1}),
        phi_update=PhiUpdateConfig(learning_rate=0.1),
        regularization=RegularizationConfig(lambda_=0.0, mu=1.0, alpha=0.0),
        link=LinkConfig(kind="exponential", sigma=1.0),
        runtime=RuntimeConfig(max_iter=10, tol=1.0e-6),
    )
    z_exp, phi_exp = build_initialization(
        InitializationSpec(
            mode="separate",
            z=InitializerSpec(name="svd"),
            phi=InitializerSpec(name="exponential"),
        ),
        dataset=_dataset(),
        config=config_exp,
        seed=0,
    )
    assert phi_exp is not None
    assert phi_exp.kind == "exponential"


def test_zero_initializer() -> None:
    _, phi_zero = build_initialization(
        InitializationSpec(
            mode="separate",
            z=InitializerSpec(name="svd"),
            phi=InitializerSpec(name="zero"),
        ),
        dataset=_dataset(),
        config=_config(),
        seed=0,
    )
    assert phi_zero is not None


def test_identity_phi_initializes_trainable_piecewise_grid() -> None:
    config = ModelConfig(
        shape=ShapeConfig(n_rows=2, n_cols=2, rank=1, n_observations=4),
        z_optimizer=OptimizerSpec("gradient_descent", {"learning_rate": 0.1}),
        phi_update=PhiUpdateConfig(learning_rate=0.1),
        regularization=RegularizationConfig(lambda_=0.0, mu=1.0, alpha=0.0),
        link=LinkConfig(
            kind="piecewise_linear_rkhs",
            sigma=1.0,
            params={
                "grid": {
                    "count": 8,
                    "bounds": {"name": "input_quantiles", "params": {}},
                    "rebuild": {"name": "never", "params": {}},
                }
            },
        ),
        runtime=RuntimeConfig(max_iter=10, tol=1.0e-6),
    )

    _, phi = build_initialization(
        InitializationSpec(
            mode="separate",
            z=InitializerSpec(name="svd"),
            phi=InitializerSpec(name="identity_phi"),
        ),
        dataset=_dataset(),
        config=config,
        seed=0,
    )

    assert phi is not None
    assert phi.kind == "piecewise_linear_rkhs"
    from bcd_lab.function_models import TrainableFunction

    assert isinstance(phi, TrainableFunction)
    assert np.allclose(phi.coefs, phi.scalars)
    assert np.allclose(phi.eval_array(phi.scalars), phi.scalars)
