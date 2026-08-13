from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from bcd_lab.algorithms.bcd_loop import run_bcd_iterations
from bcd_lab.algorithms.bcd_snapshot import publish_bcd_snapshot
from bcd_lab.algorithms.registry import AlgorithmContext
from bcd_lab.algorithms.state import init_state
from bcd_lab.algorithms.updates_z import compute_z_gradient_identity
from bcd_lab.config import (
    LinkConfig,
    ModelConfig,
    PhiUpdateConfig,
    RegularizationConfig,
    RuntimeConfig,
    ShapeConfig,
)
from bcd_lab.diagnostics import (
    DiagnosticComputationError,
    DiagnosticContext,
    DiagnosticRequest,
    ObservedSplit,
    RunSnapshot,
    compute_requested_diagnostics,
    list_diagnostic_families,
)
from bcd_lab.function_models import IdentityFunction
from bcd_lab.function_models.rkhs import GaussianRKHS
from bcd_lab.losses import get_objective, register_objective
from bcd_lab.optimizers import OptimizerSpec
from bcd_lab.types import (
    DatasetBundle,
    DatasetCapabilities,
    FunctionEvaluationDomain,
    FunctionReferences,
    SampleSet,
    SplitStatistics,
)


def _sample_set() -> SampleSet:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    return values, (rows, cols)


def _config(*, link_kind: str = "identity") -> ModelConfig:
    return ModelConfig(
        shape=ShapeConfig(n_rows=2, n_cols=2, rank=1, n_observations=4),
        z_optimizer=OptimizerSpec("gradient_descent", {"learning_rate": 0.1}),
        phi_update=PhiUpdateConfig(learning_rate=0.1),
        regularization=RegularizationConfig(lambda_=0.0, mu=1.0, alpha=0.0),
        link=LinkConfig(kind=link_kind, sigma=1.0),
        runtime=RuntimeConfig(max_iter=10, tol=1.0e-6),
    )


def _algorithm_context(config: ModelConfig, dataset: DatasetBundle) -> AlgorithmContext:
    return AlgorithmContext(
        config=config,
        dataset=dataset,
        objective=get_objective("squared_error"),
    )


def _diagnostic_context(
    state: object,
    config: ModelConfig,
    dataset: DatasetBundle,
    *,
    iteration: int = 0,
    elapsed: float = 0.0,
    objective_name: str | None = None,
    diagnostics_request: dict[str, set[str] | None] | None = None,
) -> DiagnosticContext:
    algorithm_context = _algorithm_context(config, dataset)
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


def _compute(
    ctx: DiagnosticContext,
    families: dict[str, set[str] | None],
    *,
    on_error: str = "raise",
):
    return compute_requested_diagnostics(
        ctx,
        DiagnosticRequest(families=families, on_error=on_error),  # type: ignore[arg-type]
    ).metrics


def test_reconstruction_diagnostics_are_requestable_by_family() -> None:
    z = np.array([[1.0], [2.0], [1.0], [1.0]])
    dataset = DatasetBundle(
        source="toy",
        n_rows=2,
        n_cols=2,
        rank=1,
        train=_sample_set(),
    )
    state = init_state(
        _sample_set(),
        _config(),
        z_init=z,
        phi_init=IdentityFunction(),
    )
    ctx = _diagnostic_context(state, _config(), dataset)

    metrics = _compute(ctx, {"reconstruction": {"rmse", "mae"}})

    assert "reconstruction" in list_diagnostic_families()
    assert np.isclose(metrics["train_rmse"], np.sqrt(1.5))
    assert np.isclose(metrics["train_mae"], 1.0)


def test_objective_truth_and_timing_diagnostics() -> None:
    z = np.array([[1.0], [2.0], [1.0], [1.0]])
    dataset = DatasetBundle(
        source="toy",
        n_rows=2,
        n_cols=2,
        rank=1,
        train=_sample_set(),
        dense_truth=np.array([[1.0, 1.0], [2.0, 2.0]]),
        latent_truth=z.copy(),
        capabilities=DatasetCapabilities(splits=SplitStatistics(train_count=4)),
    )
    state = init_state(
        _sample_set(),
        _config(),
        z_init=z,
        phi_init=IdentityFunction(),
    )
    ctx = _diagnostic_context(state, _config(), dataset, iteration=3, elapsed=1.5)

    metrics = _compute(
        ctx,
        {
            "objective": {"loss"},
            "truth_z": {"goodness", "relative_goodness", "procrustes_det"},
            "truth_matrix": {"x_error", "x_rmse"},
            "timing": {"elapsed"},
            "gradients": {"z_grad_norm", "phi_grad_norm"},
            "data_summary": {"density", "train_rating_count"},
        },
    )

    assert np.isclose(metrics["loss"], 6.0)
    assert np.isclose(metrics["goodness"], 0.0)
    assert np.isclose(metrics["relative_goodness"], 0.0)
    assert np.isclose(metrics["procrustes_det"], 1.0)
    assert np.isclose(metrics["x_error"], 0.0)
    assert np.isclose(metrics["x_rmse"], 0.0)
    assert np.isclose(metrics["elapsed"], 1.5)
    assert np.isclose(metrics["z_grad_norm"], np.sqrt(156.0))
    assert metrics["phi_grad_norm"] is None
    assert np.isclose(metrics["density"], 1.0)
    assert metrics["train_rating_count"] == 4


def test_dynamic_gradients_and_phi_sharp_diagnostics() -> None:
    z = np.array([[1.0], [2.0], [1.0], [1.0]])
    phi_truth = GaussianRKHS(1.0, np.array([0.0]), np.array([0.0]))
    phi = GaussianRKHS(1.0, np.array([0.0]), np.array([0.0]))
    dataset = DatasetBundle(
        source="toy",
        n_rows=2,
        n_cols=2,
        rank=1,
        train=_sample_set(),
        latent_truth=z.copy(),
        capabilities=DatasetCapabilities(
            functions=FunctionReferences(truth=phi_truth),
        ),
    )
    import dataclasses

    config = _config(link_kind="gaussian_rkhs")
    config = dataclasses.replace(
        config,
        regularization=dataclasses.replace(config.regularization, alpha=1.0),
    )

    state = init_state(
        _sample_set(),
        config,
        z_init=z,
        phi_init=phi,
    )
    ctx = _diagnostic_context(state, config, dataset)

    # Compute requested diagnostics including gradients (uncached) and phi_sharp
    metrics = _compute(
        ctx,
        {
            "gradients": {"z_grad_norm", "phi_grad_norm"},
            "phi_sharp": {"phi_sharp_diff", "lyapunov", "regret"},
        },
    )

    # 1. Assert gradients are dynamically computed and correct
    assert metrics["z_grad_norm"] is not None
    assert metrics["phi_grad_norm"] is not None
    assert metrics["z_grad_norm"] >= 0.0
    assert metrics["phi_grad_norm"] >= 0.0

    # 2. Assert phi_sharp metrics are correctly calculated
    assert metrics["phi_sharp_diff"] is not None
    assert metrics["lyapunov"] is not None
    assert metrics["regret"] is not None
    assert metrics["phi_sharp_diff"] >= 0.0
    assert np.isclose(metrics["lyapunov"], metrics["phi_sharp_diff"])


def test_phi_sharp_piecewise_linear_rkhs() -> None:
    from bcd_lab.function_models.piecewise_linear_rkhs import PiecewiseLinearRKHS

    z = np.array([[1.0], [2.0], [1.0], [1.0]])
    phi = PiecewiseLinearRKHS(1.0, (np.array([-2.0, 0.0, 2.0]), np.array([0.0, 0.0, 0.0])))
    dataset = DatasetBundle(
        source="toy",
        n_rows=2,
        n_cols=2,
        rank=1,
        train=_sample_set(),
        latent_truth=z.copy(),
        capabilities=DatasetCapabilities(
            functions=FunctionReferences(truth=phi),
            function_domain=FunctionEvaluationDomain(
                grid=np.array([-0.5, 0.0, 0.5]),
                source="test_domain",
                central_mass=0.95,
            ),
        ),
    )
    import dataclasses

    config = _config(link_kind="piecewise_linear_rkhs")
    config = dataclasses.replace(
        config,
        regularization=dataclasses.replace(config.regularization, alpha=1.0),
    )

    state = init_state(
        _sample_set(),
        config,
        z_init=z,
        phi_init=phi,
    )
    ctx = _diagnostic_context(state, config, dataset)

    # Compute requested diagnostics including gradients (uncached) and phi_sharp
    metrics = _compute(
        ctx,
        {
            "phi_sharp": {"phi_sharp_diff", "lyapunov", "regret"},
        },
    )

    # Assert phi_sharp metrics are correctly calculated for PiecewiseLinearRKHS
    assert metrics["phi_sharp_diff"] is not None
    assert metrics["lyapunov"] is not None
    assert metrics["regret"] is not None
    assert metrics["phi_sharp_diff"] >= 0.0
    phi_sharp = ctx.cache["phi_sharp"]
    grid = dataset.capabilities.function_domain.grid
    residual = phi_sharp.eval_array(grid) - phi.eval_array(grid)
    expected_distance = np.sqrt(np.sum(residual**2) * (grid[1] - grid[0]))
    assert np.isclose(metrics["phi_sharp_diff"], expected_distance)
    assert np.isclose(metrics["lyapunov"], expected_distance)


def test_phi_sharp_respects_squared_error_scale_and_rejects_other_objectives() -> None:
    z = np.array([[1.0], [2.0], [1.0], [1.0]])
    phi = GaussianRKHS(1.0, np.array([0.0]), np.array([0.0]))
    dataset = DatasetBundle(
        source="toy",
        n_rows=2,
        n_cols=2,
        rank=1,
        train=_sample_set(),
    )
    config = dataclasses.replace(
        _config(link_kind="gaussian_rkhs"),
        regularization=dataclasses.replace(_config().regularization, alpha=1.0),
    )

    def _phi_sharp_for(objective_name: str):
        state = init_state(
            _sample_set(),
            config,
            z_init=z,
            phi_init=phi,
        )
        ctx = _diagnostic_context(
            state,
            config,
            dataset,
            objective_name=objective_name,
        )
        return _compute(
            ctx,
            {"phi_sharp": {"phi_sharp_diff", "regret"}},
        ), ctx.cache.get("phi_sharp")

    squared_metrics, squared_phi = _phi_sharp_for("squared_error")
    half_metrics, half_phi = _phi_sharp_for("half_squared")
    assert squared_metrics["regret"] is not None
    assert half_metrics["regret"] is not None
    assert squared_phi is not None
    assert half_phi is not None
    assert not np.allclose(squared_phi.coefs, half_phi.coefs)

    register_objective(
        "diagnostic_non_quadratic",
        value=lambda observed, predictions, params: float(np.sum(np.abs(predictions - observed))),
        prediction_gradient=lambda observed, predictions, params: np.sign(predictions - observed),
    )
    unsupported_metrics, unsupported_phi = _phi_sharp_for("diagnostic_non_quadratic")
    assert unsupported_phi is None
    assert unsupported_metrics == {"phi_sharp_diff": None, "regret": None}


def test_diagnostics_accept_snapshots_from_different_private_state_types() -> None:
    @dataclasses.dataclass
    class MomentumState:
        position: np.ndarray
        target: np.ndarray
        velocity: np.ndarray

    @dataclasses.dataclass
    class TreeState:
        leaves: tuple[float, ...]
        observations: list[float]

    def publish_momentum(state: MomentumState) -> RunSnapshot:
        residual = state.position - state.target
        return RunSnapshot(
            iteration=4,
            elapsed_seconds=0.25,
            observed_splits={
                "train": ObservedSplit(state.target, state.position),
            },
            objective_name="quadratic",
            objective_value=float(np.dot(residual, residual)),
            capabilities={"z_gradient": 2.0 * residual},
        )

    def publish_tree(state: TreeState) -> RunSnapshot:
        predictions = np.asarray(state.leaves)
        observed = np.asarray(state.observations)
        return RunSnapshot(
            iteration=9,
            elapsed_seconds=0.5,
            observed_splits={
                "train": ObservedSplit(observed, predictions),
            },
            objective_name="tree_error",
            objective_value=float(np.sum((predictions - observed) ** 2)),
        )

    momentum_state = MomentumState(
        position=np.array([1.0, 3.0]),
        target=np.array([1.0, 1.0]),
        velocity=np.array([0.2, -0.1]),
    )
    tree_state = TreeState(leaves=(1.0, 3.0), observations=[1.0, 1.0])
    momentum_snapshot = publish_momentum(momentum_state)
    tree_snapshot = publish_tree(tree_state)

    requested = {
        "objective": {"loss"},
        "reconstruction": {"rmse"},
        "timing": {"elapsed"},
    }
    momentum_metrics = _compute(DiagnosticContext(snapshot=momentum_snapshot), requested)
    tree_metrics = _compute(DiagnosticContext(snapshot=tree_snapshot), requested)

    assert momentum_metrics == {
        "loss": 4.0,
        "train_rmse": np.sqrt(2.0),
        "validation_rmse": None,
        "test_rmse": None,
        "elapsed": 0.25,
    }
    assert tree_metrics == {
        "loss": 4.0,
        "train_rmse": np.sqrt(2.0),
        "validation_rmse": None,
        "test_rmse": None,
        "elapsed": 0.5,
    }

    momentum_state.position[:] = 99.0
    assert np.array_equal(
        momentum_snapshot.observed_splits["train"].predictions,
        np.array([1.0, 3.0]),
    )
    assert not momentum_snapshot.observed_splits["train"].predictions.flags.writeable


def test_bcd_snapshot_is_detached_from_mutable_solver_state() -> None:
    config = _config(link_kind="gaussian_rkhs")
    dataset = DatasetBundle(
        source="toy",
        n_rows=2,
        n_cols=2,
        rank=1,
        train=_sample_set(),
    )
    state = init_state(
        dataset.train,
        config,
        z_init=np.array([[1.0], [2.0], [1.0], [1.0]]),
        phi_init=GaussianRKHS(1.0, np.array([0.0]), np.array([0.5])),
    )
    snapshot = _diagnostic_context(state, config, dataset).snapshot
    published_z = snapshot.capability("latent_factors").copy()
    published_phi_coefs = snapshot.capability("link_function").coefs.copy()

    state.z[:] = 99.0
    assert state.phi is not None
    state.phi.coefs[:] = 77.0

    assert np.array_equal(snapshot.capability("latent_factors"), published_z)
    assert np.array_equal(
        snapshot.capability("link_function").coefs,
        published_phi_coefs,
    )
    assert not snapshot.capability("latent_factors").flags.writeable
    with pytest.raises(TypeError):
        snapshot.capabilities["new"] = "not allowed"  # type: ignore[index]


def test_bcd_snapshot_recomputes_stale_cached_gradient() -> None:
    config = _config()
    dataset = DatasetBundle(
        source="toy",
        n_rows=2,
        n_cols=2,
        rank=1,
        train=_sample_set(),
    )
    state = init_state(
        dataset.train,
        config,
        z_init=np.array([[1.0], [2.0], [1.0], [1.0]]),
        phi_init=IdentityFunction(),
    )
    stale = np.full_like(state.z, -999.0)
    state.cache["z_gradient"] = stale
    context = _algorithm_context(config, dataset)

    snapshot = publish_bcd_snapshot(
        state,
        context,
        3,
        0.1,
        0.01,
        0.0,
        DiagnosticRequest(families={"gradients": {"z_grad_norm"}}),
    )
    expected = compute_z_gradient_identity(state, context, iteration=0)
    metrics = _compute(
        DiagnosticContext(snapshot=snapshot),
        {"gradients": {"z_grad_norm"}},
    )

    assert np.array_equal(snapshot.capability("z_gradient"), expected)
    assert not np.array_equal(snapshot.capability("z_gradient"), stale)
    assert np.isclose(metrics["z_grad_norm"], np.linalg.norm(expected))


def test_objective_only_snapshot_skips_expensive_optional_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bcd_lab.algorithms.bcd_snapshot as snapshot_module

    config = _config(link_kind="gaussian_rkhs")
    dataset = DatasetBundle(
        source="toy",
        n_rows=2,
        n_cols=2,
        rank=1,
        train=_sample_set(),
        validation=_sample_set(),
        test=_sample_set(),
    )
    state = init_state(
        dataset.train,
        config,
        z_init=np.array([[1.0], [2.0], [1.0], [1.0]]),
        phi_init=GaussianRKHS(1.0, np.array([0.0]), np.array([0.5])),
    )

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unrequested optional diagnostic work was performed")

    monkeypatch.setattr(snapshot_module, "_predict_split", fail)
    monkeypatch.setattr(snapshot_module, "_dense_predictions", fail)
    monkeypatch.setattr(snapshot_module, "compute_z_gradient", fail)
    monkeypatch.setattr(snapshot_module, "compute_z_gradient_identity", fail)
    monkeypatch.setattr(snapshot_module, "_phi_gradient_norm", fail)
    monkeypatch.setattr(snapshot_module, "_copy_link", fail)

    snapshot = publish_bcd_snapshot(
        state,
        _algorithm_context(config, dataset),
        0,
        0.0,
        0.0,
        0.0,
        DiagnosticRequest(families={"objective": {"loss"}}),
    )

    assert snapshot.objective_value is not None
    assert snapshot.observed_splits == {}
    assert snapshot.capabilities == {}


def test_phi_sharp_is_unavailable_without_published_latent_factors() -> None:
    link = GaussianRKHS(1.0, np.array([0.0]), np.array([0.0]))
    snapshot = RunSnapshot(
        iteration=0,
        elapsed_seconds=0.0,
        objective_name="squared_error",
        objective_value=1.0,
        capabilities={
            "link_function": link,
            "training_centers": np.array([0.0, 1.0]),
            "training_observed": np.array([1.0, 2.0]),
            "link_regularization": 1.0,
            "squared_error_scale": 1.0,
        },
    )

    report = compute_requested_diagnostics(
        DiagnosticContext(snapshot=snapshot),
        DiagnosticRequest(families={"phi_sharp": {"phi_sharp_diff", "regret"}}),
    )

    assert report.metrics == {"phi_sharp_diff": None, "regret": None}
    assert report.unavailable == {
        "phi_sharp_diff": "required inputs or mathematical preconditions are unavailable",
        "regret": "required inputs or mathematical preconditions are unavailable",
    }
    assert report.failures == ()


def test_function_distance_defects_reach_the_diagnostic_failure_policy() -> None:
    class BrokenDistanceFunction:
        kind = "broken"

        def eval_array(self, values):
            return np.zeros_like(values)

        def copy(self):
            return self

        def distance_to(self, other, *, mode="auto", grid=None):
            raise RuntimeError("broken model-native distance")

    link = BrokenDistanceFunction()
    dataset = DatasetBundle(
        source="toy",
        n_rows=2,
        n_cols=2,
        rank=1,
        train=_sample_set(),
        capabilities=DatasetCapabilities(
            functions=FunctionReferences(truth=link),
        ),
    )
    snapshot = RunSnapshot(
        iteration=7,
        elapsed_seconds=0.25,
        capabilities={"link_function": link},
    )
    ctx = DiagnosticContext(snapshot=snapshot, dataset=dataset)
    families = {"phi": {"phi_diff"}}

    with pytest.raises(
        DiagnosticComputationError,
        match="Diagnostic family 'phi' failed: broken model-native distance",
    ) as raised:
        compute_requested_diagnostics(
            ctx,
            DiagnosticRequest(families=families),
        )
    assert isinstance(raised.value.__cause__, RuntimeError)

    report = compute_requested_diagnostics(
        ctx,
        DiagnosticRequest(families=families, on_error="record"),
    )
    assert report.metrics == {}
    assert report.unavailable == {}
    assert [dataclasses.asdict(failure) for failure in report.failures] == [
        {
            "family": "phi",
            "metrics": ("phi_diff",),
            "error_type": "RuntimeError",
            "message": "broken model-native distance",
        }
    ]

    config = dataclasses.replace(
        _config(),
        runtime=dataclasses.replace(_config().runtime, max_iter=0),
    )
    state = init_state(
        dataset.train,
        config,
        z_init=np.ones((4, 1)),
        phi_init=IdentityFunction(),
    )

    def publish_snapshot(state, context, iteration, elapsed, sec_per_iter, rem_sec, request):
        return RunSnapshot(
            iteration=iteration,
            elapsed_seconds=elapsed,
            capabilities={"link_function": link},
        )

    result = run_bcd_iterations(
        state=state,
        context=_algorithm_context(config, dataset),
        update=lambda state, context, iteration: None,
        publish_snapshot=publish_snapshot,
        diagnostics_request=DiagnosticRequest(
            families=families,
            on_error="record",
        ),
    )
    assert result.metadata["diagnostic_events"] == [
        {
            "iteration": 0,
            "unavailable": {},
            "failures": [
                {
                    "family": "phi",
                    "metrics": ("phi_diff",),
                    "error_type": "RuntimeError",
                    "message": "broken model-native distance",
                }
            ],
        }
    ]
