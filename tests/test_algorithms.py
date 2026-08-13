"""Unit tests for BCD algorithms."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from bcd_lab.algorithms import AlgorithmContext, get_algorithm, list_algorithms
from bcd_lab.algorithms.bcd_loop import run_bcd_iterations
from bcd_lab.algorithms.numerics import BcdNumericalError
from bcd_lab.algorithms.state import init_state
from bcd_lab.algorithms.updates_z import apply_lambda
from bcd_lab.config import (
    LinkConfig,
    ModelConfig,
    PhiUpdateConfig,
    RegularizationConfig,
    RuntimeConfig,
    ShapeConfig,
)
from bcd_lab.diagnostics import DiagnosticRequest, RunSnapshot
from bcd_lab.function_models import IdentityFunction
from bcd_lab.function_models.rkhs import GaussianRKHS
from bcd_lab.initializers import InitializationSpec, InitializerSpec, build_initialization
from bcd_lab.losses import (
    Objective,
    build_balancing_operator,
    get_objective,
    register_objective,
    z_regularization,
)
from bcd_lab.optimizers import OptimizerSpec
from bcd_lab.types import (
    AlgorithmCheckpoint,
    DatasetBundle,
    DatasetCapabilities,
    FunctionReferences,
    SampleSet,
    SelectionPolicy,
)


def _sample_set() -> SampleSet:
    # Small rank-1 toy dataset
    values = np.array([1.0, 2.0, 3.0, 4.0])
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    return values, (rows, cols)


def _dataset() -> DatasetBundle:
    return DatasetBundle(
        source="toy",
        n_rows=2,
        n_cols=2,
        rank=1,
        train=_sample_set(),
        dense_truth=np.array([[1.0, 2.0], [3.0, 4.0]]),
        latent_truth=np.array([[1.0], [1.0], [1.0], [1.0]]),
    )


def _objective() -> Objective:
    return get_objective("squared_error")


def _config(*, link_kind: str = "identity", params: dict | None = None) -> ModelConfig:
    return ModelConfig(
        shape=ShapeConfig(n_rows=2, n_cols=2, rank=1, n_observations=4),
        z_optimizer=OptimizerSpec("gradient_descent", {"learning_rate": 0.01}),
        phi_update=PhiUpdateConfig(learning_rate=0.01),
        regularization=RegularizationConfig(lambda_=0.0, mu=0.0, alpha=0.01),
        link=LinkConfig(kind=link_kind, sigma=1.0, params=params or {}),
        runtime=RuntimeConfig(
            max_iter=5,
            tol=0.0,
            dtype="float64",
            diagnostics_interval=1,
            checkpoint_iters=(0, 2, 5),
        ),
    )


def test_registry_listing() -> None:
    algos = list_algorithms()
    assert "bcd" in algos


def test_iterative_lifecycle_schedules_observations_and_converges() -> None:
    config = dataclasses.replace(
        _config(),
        runtime=RuntimeConfig(
            max_iter=20,
            tol=0.05,
            diagnostics_interval=7,
            checkpoint_iters=(7,),
        ),
    )
    dataset = _dataset()
    state = init_state(
        dataset.train,
        config,
        z_init=np.ones((4, 1)),
        phi_init=IdentityFunction(),
    )
    context = AlgorithmContext(config=config, dataset=dataset, objective=_objective())
    observed_losses = {0: 1.0, 1: 0.5, 7: 0.12, 10: 0.1}
    updated: list[int] = []

    def publish_snapshot(
        state,
        context,
        iteration,
        elapsed,
        sec_per_iter,
        rem_sec,
        request,
    ):
        _ = state, context, sec_per_iter, rem_sec, request
        return RunSnapshot(
            iteration=iteration,
            elapsed_seconds=elapsed,
            objective_value=observed_losses[iteration],
            objective_name="squared_error",
        )

    result = run_bcd_iterations(
        state=state,
        context=context,
        update=lambda state, context, iteration: updated.append(iteration),
        publish_snapshot=publish_snapshot,
        diagnostics_request=DiagnosticRequest(families={"objective": {"loss"}}),
        capture_checkpoint=lambda state, elapsed: AlgorithmCheckpoint(
            format_version=1,
            algorithm="test",
            state_version=1,
            elapsed_seconds=elapsed,
            payload={"z": state.z.copy()},
        ),
    )

    assert [record["iteration"] for record in result.diagnostics] == [0, 1, 7, 10]
    assert updated == list(range(1, 11))
    assert result.termination.reason == "converged"
    assert result.termination.iteration == 10
    assert set(result.checkpoints) == {7}


def test_iterative_lifecycle_selects_one_coherent_observed_state() -> None:
    config = dataclasses.replace(
        _config(),
        runtime=RuntimeConfig(
            max_iter=10,
            tol=0.0,
            diagnostics_interval=1,
        ),
    )
    dataset = _dataset()
    state = init_state(
        dataset.train,
        config,
        z_init=np.ones((4, 1)),
        phi_init=IdentityFunction(),
    )
    context = AlgorithmContext(config=config, dataset=dataset, objective=_objective())
    losses = {iteration: float((iteration - 4) ** 2) for iteration in range(11)}

    def update(state, context, iteration):
        _ = context
        state.z[0, 0] = iteration

    def publish_snapshot(state, context, iteration, elapsed, sec_per_iter, rem_sec, request):
        _ = state, context, sec_per_iter, rem_sec, request
        return RunSnapshot(
            iteration=iteration,
            elapsed_seconds=elapsed,
            objective_value=losses[iteration],
            objective_name="squared_error",
        )

    result = run_bcd_iterations(
        state=state,
        context=context,
        update=update,
        publish_snapshot=publish_snapshot,
        diagnostics_request=DiagnosticRequest(families={"objective": {"loss"}}),
        selection_policy=SelectionPolicy(metric="loss", mode="min"),
        capture_checkpoint=lambda state, elapsed: AlgorithmCheckpoint(
            format_version=1,
            algorithm="test",
            state_version=1,
            elapsed_seconds=elapsed,
            payload={"z": state.z.copy()},
        ),
    )

    assert result.selection is not None
    assert result.selection.iteration == 4
    assert result.selection.value == 0.0
    assert result.selection.diagnostics == result.diagnostics[4]
    assert result.selection.checkpoint is not None
    assert result.selection.checkpoint.payload["z"][0, 0] == 4
    assert result.z[0, 0] == 10


def test_generic_bcd_solver() -> None:
    # 1. Test generic solver in standard RKHS mode
    dataset = _dataset()
    phi_truth = GaussianRKHS(1.0, np.array([0.0]), np.array([0.0]))
    dataset.capabilities = DatasetCapabilities(
        functions=FunctionReferences(truth=phi_truth),
    )
    config = _config(link_kind="gaussian_rkhs", params={"phi_0": phi_truth})

    z_init, phi_init = build_initialization(
        InitializationSpec(
            mode="joint",
            joint=InitializerSpec(name="svd_with_zero_phi"),
        ),
        dataset=dataset,
        config=config,
        seed=0,
    )

    algo = get_algorithm("bcd")
    result = algo.run(
        config=config,
        dataset=dataset,
        objective=_objective(),
        z_init=z_init,
        phi_init=phi_init,
        rng=np.random.default_rng(0),
    )

    assert result.z.shape == (4, 1)
    assert isinstance(result.phi, GaussianRKHS)
    assert len(result.diagnostics) >= 2

    # 2. Test generic solver in piecewise-linear RKHS mode
    config_piecewise = _config(link_kind="piecewise_linear_rkhs")
    z_piecewise, phi_piecewise = build_initialization(
        InitializationSpec(
            mode="separate",
            z=InitializerSpec(name="svd"),
            phi=InitializerSpec(name="zero"),
        ),
        dataset=dataset,
        config=config_piecewise,
        seed=0,
    )

    result_piecewise = algo.run(
        config=config_piecewise,
        dataset=dataset,
        objective=_objective(),
        z_init=z_piecewise,
        phi_init=phi_piecewise,
        rng=np.random.default_rng(0),
    )

    assert result_piecewise.z.shape == (4, 1)
    assert result_piecewise.phi is not None
    assert result_piecewise.phi.kind == "piecewise_linear_rkhs"
    assert len(result_piecewise.diagnostics) >= 2


def test_bcd_identity_solver() -> None:
    dataset = _dataset()
    config = _config(link_kind="identity")

    # Initialize Z via SVD
    z_init, phi_init = build_initialization(
        InitializationSpec(
            mode="separate",
            z=InitializerSpec(name="svd"),
            phi=InitializerSpec(name="identity"),
        ),
        dataset=dataset,
        config=config,
        seed=0,
    )

    algo = get_algorithm("bcd")
    result = algo.run(
        config=config,
        dataset=dataset,
        objective=_objective(),
        z_init=z_init,
        phi_init=phi_init,
        rng=np.random.default_rng(0),
    )

    # Check results
    assert result.z.shape == (4, 1)
    assert isinstance(result.phi, IdentityFunction)
    assert len(result.diagnostics) >= 2
    assert "loss" in result.diagnostics[-1]

    # Check checkpoints
    assert 0 in result.checkpoints
    assert 2 in result.checkpoints
    assert 5 in result.checkpoints
    assert np.asarray(result.checkpoints[2].payload["z"]).shape == (4, 1)


def test_bcd_validates_truth_capability_for_inferred_ball_radius() -> None:
    phi_0 = GaussianRKHS(1.0, np.array([0.0]), np.array([0.0]))
    config = _config(
        link_kind="gaussian_rkhs",
        params={"ball_enabled": True, "phi_0": phi_0},
    )
    algo = get_algorithm("bcd")

    with pytest.raises(ValueError, match="dataset.capabilities.functions.truth"):
        algo.validate_dataset(config=config, dataset=_dataset())  # type: ignore[attr-defined]

    dataset = _dataset()
    dataset.capabilities = DatasetCapabilities(
        functions=FunctionReferences(truth=phi_0),
    )
    algo.validate_dataset(config=config, dataset=dataset)  # type: ignore[attr-defined]
    explicit_radius_config = _config(
        link_kind="gaussian_rkhs",
        params={"ball_enabled": True, "phi_0": phi_0, "ball_radius": 0.5},
    )
    algo.validate_dataset(  # type: ignore[attr-defined]
        config=explicit_radius_config,
        dataset=_dataset(),
    )


def test_bcd_uses_model_owned_ball_radius_capability() -> None:
    class PrivateBallFunction:
        kind = "private_ball"

        def __init__(self) -> None:
            self._resolved_truth = None
            self._projected_radius = None

        def eval_array(self, values: np.ndarray) -> np.ndarray:
            return np.asarray(values)

        def d_array(self, values: np.ndarray) -> np.ndarray:
            return np.ones_like(values, dtype=float)

        def eval_and_d_array(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return self.eval_array(values), self.d_array(values)

        def copy(self) -> PrivateBallFunction:
            return PrivateBallFunction()

        def fit(self, x: np.ndarray, y: np.ndarray) -> None:
            _ = x, y

        def scale_parameters(self, scale: float) -> None:
            _ = scale

        def gradient(self, *, inputs: np.ndarray, residuals: np.ndarray) -> float:
            _ = inputs, residuals
            return 0.0

        def apply_gradient(self, gradient: float, *, step_size: float) -> None:
            _ = gradient, step_size

        def resolve_projection_ball_radius(self, *, truth: object | None) -> float:
            self._resolved_truth = truth
            return 0.25

        def project(
            self,
            *,
            centers: np.ndarray | None = None,
            rng: np.random.Generator | None = None,
            l2_grid: np.ndarray | None = None,
            phi_ball_radius: float | None = None,
        ) -> None:
            _ = centers, rng, l2_grid
            self._projected_radius = phi_ball_radius

    truth = IdentityFunction()
    dataset = _dataset()
    dataset.capabilities = DatasetCapabilities(
        functions=FunctionReferences(truth=truth),
    )
    config = dataclasses.replace(
        _config(
            link_kind=PrivateBallFunction.kind,
            params={"ball_enabled": True, "phi_0": IdentityFunction()},
        ),
        runtime=RuntimeConfig(max_iter=1, tol=0.0, diagnostics_interval=1),
    )

    result = get_algorithm("bcd").run(
        config=config,
        dataset=dataset,
        objective=_objective(),
        z_init=np.ones((4, 1)),
        phi_init=PrivateBallFunction(),
        rng=np.random.default_rng(0),
    )

    assert isinstance(result.phi, PrivateBallFunction)
    assert result.phi._resolved_truth is truth
    assert result.phi._projected_radius == 0.25
    assert not hasattr(result.phi, "config")
    assert not hasattr(result.phi, "scalars")


def test_bcd_sigmoid_solver_remains_finite() -> None:
    dataset = _dataset()
    config = _config(link_kind="sigmoid")
    z_init, phi_init = build_initialization(
        InitializationSpec(
            mode="separate",
            z=InitializerSpec(name="svd"),
            phi=InitializerSpec(name="sigmoid"),
        ),
        dataset=dataset,
        config=config,
        seed=0,
    )
    result = get_algorithm("bcd").run(
        config=config,
        dataset=dataset,
        objective=_objective(),
        z_init=z_init,
        phi_init=phi_init,
        rng=np.random.default_rng(0),
    )

    assert result.z.shape == (4, 1)
    assert result.phi.kind == "sigmoid"
    assert np.all(np.isfinite(result.z))
    assert all(np.isfinite(record["loss"]) for record in result.diagnostics)


def test_bcd_exponential_fails_with_iteration_context() -> None:
    dataset = _dataset()
    config = _config(link_kind="exponential")
    z_init, phi_init = build_initialization(
        InitializationSpec(
            mode="separate",
            z=InitializerSpec(name="svd"),
            phi=InitializerSpec(name="exponential"),
        ),
        dataset=dataset,
        config=config,
        seed=0,
    )

    with pytest.raises(BcdNumericalError, match=r"BCD.*iteration [1-9]") as error:
        get_algorithm("bcd").run(
            config=config,
            dataset=dataset,
            objective=_objective(),
            z_init=z_init,
            phi_init=phi_init,
            rng=np.random.default_rng(0),
        )

    assert error.value.iteration > 0
    assert error.value.quantity


def test_bcd_without_phi_is_explicit_identity_only() -> None:
    dataset = _dataset()
    algo = get_algorithm("bcd")
    result = algo.run(
        config=_config(link_kind="identity"),
        dataset=dataset,
        objective=_objective(),
        rng=np.random.default_rng(0),
    )

    assert result.z.shape == (4, 1)
    assert result.phi is None
    assert len(result.diagnostics) >= 2

    with pytest.raises(ValueError, match="valid only with link.kind='identity'"):
        algo.run(
            config=_config(link_kind="gaussian_rkhs"),
            dataset=dataset,
            objective=_objective(),
            rng=np.random.default_rng(0),
        )


def test_selected_objective_controls_bcd_updates_and_reported_loss() -> None:
    dataset = _dataset()
    config = dataclasses.replace(
        _config(link_kind="gaussian_rkhs"),
        phi_update=PhiUpdateConfig(learning_rate=0.01),
        regularization=dataclasses.replace(_config().regularization, alpha=0.0),
        runtime=RuntimeConfig(
            max_iter=1,
            tol=0.0,
            diagnostics_interval=1,
        ),
        bcd=dataclasses.replace(_config().bcd, project_z=False),
    )
    z_init = np.array([[0.5], [1.0], [0.75], [1.25]])
    phi_init = GaussianRKHS(1.0, np.array([0.0]), np.array([0.2]))
    algorithm = get_algorithm("bcd")

    squared = algorithm.run(
        config=config,
        dataset=dataset,
        objective=get_objective("squared_error"),
        z_init=z_init,
        phi_init=phi_init,
        diagnostics_request=DiagnosticRequest(families={"objective": {"loss"}}),
        rng=np.random.default_rng(0),
    )
    half_squared = algorithm.run(
        config=config,
        dataset=dataset,
        objective=get_objective("half_squared"),
        z_init=z_init,
        phi_init=phi_init,
        diagnostics_request=DiagnosticRequest(families={"objective": {"loss"}}),
        rng=np.random.default_rng(0),
    )

    assert np.isclose(
        squared.diagnostics[0]["loss"],
        2.0 * half_squared.diagnostics[0]["loss"],
    )
    assert not np.allclose(squared.z, half_squared.z)
    assert squared.phi is not None
    assert half_squared.phi is not None
    assert not np.allclose(squared.phi.coefs, half_squared.phi.coefs)


def test_bcd_consumes_a_registered_objective_gradient() -> None:
    gradient_calls: list[np.ndarray] = []

    def zero_gradient(observed, predictions, params):
        gradient_calls.append(np.asarray(predictions).copy())
        return np.zeros_like(predictions, dtype=float)

    register_objective(
        "a16_constant_objective",
        value=lambda observed, predictions, params: 0.0,
        prediction_gradient=zero_gradient,
    )
    config = dataclasses.replace(
        _config(link_kind="identity"),
        z_optimizer=OptimizerSpec("gradient_descent", {"learning_rate": 0.1}),
        phi_update=PhiUpdateConfig(learning_rate=0.0),
        regularization=RegularizationConfig(lambda_=0.0, mu=0.0, alpha=0.0),
        runtime=RuntimeConfig(max_iter=1, tol=0.0, diagnostics_interval=1),
        bcd=dataclasses.replace(_config().bcd, project_z=False),
    )
    initial_z = np.array([[0.5], [1.0], [0.75], [1.25]])

    result = get_algorithm("bcd").run(
        config=config,
        dataset=_dataset(),
        objective=get_objective("a16_constant_objective"),
        z_init=initial_z,
        phi_init=IdentityFunction(),
        diagnostics_request=DiagnosticRequest(families={"objective": {"loss"}}),
        rng=np.random.default_rng(0),
    )

    assert len(gradient_calls) == 1
    assert np.array_equal(result.z, initial_z)


def test_factor_regularization_update_matches_reported_value_gradient() -> None:
    config = dataclasses.replace(
        _config(link_kind="identity"),
        z_optimizer=OptimizerSpec("gradient_descent", {"learning_rate": 0.05}),
        phi_update=PhiUpdateConfig(learning_rate=0.0),
        regularization=RegularizationConfig(lambda_=0.4, mu=0.0, alpha=0.0),
    )
    initial_z = np.array([[0.5], [1.0], [0.75], [1.25]])
    state = init_state(
        _sample_set(),
        config,
        z_init=initial_z,
        phi_init=IdentityFunction(),
    )
    context = AlgorithmContext(
        config=config,
        dataset=_dataset(),
        objective=get_objective("squared_error"),
    )
    operator = build_balancing_operator(2, 2, initial_z.dtype)
    epsilon = 1.0e-6
    finite_difference = np.zeros_like(initial_z)
    for index in np.ndindex(initial_z.shape):
        plus = initial_z.copy()
        minus = initial_z.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        finite_difference[index] = (
            z_regularization(plus, operator, 0.4) - z_regularization(minus, operator, 0.4)
        ) / (2.0 * epsilon)

    apply_lambda(state, context)

    applied_gradient = (initial_z - state.z) / state.z_optimizer.learning_rate
    assert np.allclose(applied_gradient, finite_difference, rtol=1.0e-6, atol=1.0e-8)


def test_adam_bcd_run_is_deterministic_finite_and_reports_update_norm() -> None:
    config = dataclasses.replace(
        _config(link_kind="identity"),
        z_optimizer=OptimizerSpec("adam", {"learning_rate": 0.01}),
        phi_update=PhiUpdateConfig(learning_rate=0.0),
        regularization=RegularizationConfig(lambda_=0.2, mu=0.05, alpha=0.0),
        runtime=RuntimeConfig(max_iter=2, tol=0.0, diagnostics_interval=1),
    )
    initial_z = np.array([[0.5], [1.0], [0.75], [1.25]])
    request = DiagnosticRequest(families={"gradients": {"z_grad_norm", "z_optimizer_update_norm"}})

    results = [
        get_algorithm("bcd").run(
            config=config,
            dataset=_dataset(),
            objective=_objective(),
            z_init=initial_z,
            phi_init=IdentityFunction(),
            diagnostics_request=request,
            rng=np.random.default_rng(7),
        )
        for _ in range(2)
    ]

    assert np.array_equal(results[0].z, results[1].z)
    assert np.all(np.isfinite(results[0].z))
    assert results[0].diagnostics[-1]["z_optimizer_update_norm"] > 0.0
    assert np.max(np.linalg.norm(results[0].z, axis=1)) < np.max(np.linalg.norm(initial_z, axis=1))
