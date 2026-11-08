"""Unified generic BCD algorithm solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from bcd_lab.checkpoints import CHECKPOINT_FORMAT_VERSION
from bcd_lab.config import (
    BcdConfig,
    LinkConfig,
    ModelConfig,
    PhiType,
    RuntimeConfig,
    ShapeConfig,
)
from bcd_lab.diagnostics import DiagnosticRequest
from bcd_lab.function_models import (
    BallRadiusResolver,
    CopyableFunction,
    ProjectableFunction,
    ScalarFunction,
    SerializableFunctionModel,
    TrainableFunction,
    UpdatePreparedFunction,
)
from bcd_lab.function_models.checkpoints import (
    restore_function_model,
    serialize_function_model,
)
from bcd_lab.losses import Objective
from bcd_lab.optimizers import (
    ArrayOptimizer,
    restore_optimizer,
    serialize_optimizer,
)
from bcd_lab.types import (
    AlgorithmCheckpoint,
    AlgorithmResult,
    DatasetBundle,
    SelectionPolicy,
)

from .bcd_config import parse_bcd_parameters
from .bcd_loop import run_bcd_iterations
from .bcd_snapshot import publish_bcd_snapshot
from .numerics import require_finite
from .registry import Algorithm, AlgorithmContext
from .state import SolverState, init_state
from .updates_z import make_z_updater


@dataclass
class _BcdRestoredState:
    """Algorithm-private state reconstructed from a BCD checkpoint."""

    z: np.ndarray
    phi: ScalarFunction | None
    z_optimizer: ArrayOptimizer


def _inner_product_range(z_rows: np.ndarray, z_columns: np.ndarray) -> tuple[float, float]:
    """Compute an exact range without materializing a potentially huge matrix."""
    lower = np.inf
    upper = -np.inf
    for start in range(0, z_columns.shape[0], 4096):
        values = z_rows @ z_columns[start : start + 4096].T
        require_finite(values, "selected inner-product matrix", 0)
        lower = min(lower, float(np.min(values)))
        upper = max(upper, float(np.max(values)))
    return float(lower), float(upper)


def _link_curve_inputs(
    payload: Mapping[str, Any],
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Prefer a model's resolved grid, otherwise sample its observed input range."""
    raw_grid = payload.get("grid")
    if raw_grid is not None:
        grid = np.asarray(raw_grid, dtype=float)
        if (
            grid.ndim == 1
            and grid.size >= 2
            and np.all(np.isfinite(grid))
            and np.all(np.diff(grid) > 0.0)
        ):
            return grid
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    return np.linspace(lower, upper, 200)


def _update_phi(
    state: SolverState,
    context: AlgorithmContext,
    iteration: int,
    *,
    phi_ball_radius: float | None = None,
    rng: np.random.Generator,
) -> None:
    """Modular phi update with regridding, step, and projection."""
    if context.config.bcd.freeze_phi_updates:
        return

    phi = state.phi
    if phi is None or not isinstance(phi, TrainableFunction):
        return
    if not isinstance(phi, ProjectableFunction):
        raise TypeError(
            f"Trainable link {type(phi).__name__} must also implement ProjectableFunction"
        )

    from bcd_lab.math import compute_centers

    centers = state.centers
    if centers is None:
        centers = compute_centers(state.z, state.rows, state.cols, context.config.shape.n_rows)
        state.centers = centers

    if isinstance(phi, UpdatePreparedFunction):
        phi.prepare_update(centers)
    predictions = phi.eval_array(centers)
    require_finite(predictions, "link predictions", iteration)
    descent_direction = -context.objective.prediction_gradient(
        state.values,
        predictions,
    )
    require_finite(descent_direction, "phi descent direction", iteration)

    learning_rate = context.config.phi_update.learning_rate
    decay = 1.0 - learning_rate * context.config.regularization.alpha
    phi.scale_parameters(decay)
    gradient = phi.gradient(inputs=centers, residuals=descent_direction)
    phi.apply_gradient(gradient, step_size=learning_rate)
    require_finite(phi.eval_array(centers), "link state predictions", iteration)

    phi.project(
        centers=centers,
        rng=rng,
        phi_ball_radius=phi_ball_radius,
    )


class BcdAlgorithm(Algorithm):
    """BCD solver supporting the registered fixed and trainable link models."""

    name = "bcd"
    checkpoint_version = 2

    def __init__(self, *, phi_type: PhiType | None = None) -> None:
        self.phi_type = phi_type

    def build_config(
        self,
        *,
        data: Mapping[str, Any],
        link: Mapping[str, Any],
        runtime: Mapping[str, Any],
        params: Mapping[str, Any],
    ) -> ModelConfig:
        """Validate and build BCD-owned configuration from component parameters."""
        bcd_params = parse_bcd_parameters(params)
        shape_data = data.get("shape", {})
        return ModelConfig(
            shape=ShapeConfig(
                n_rows=int(shape_data["n_rows"]),
                n_cols=int(shape_data["n_cols"]),
                rank=int(shape_data["rank"]),
                n_observations=int(shape_data["n_observations"]),
            ),
            z_optimizer=bcd_params.z_optimizer,
            phi_update=bcd_params.phi_update,
            regularization=bcd_params.regularization,
            link=LinkConfig(
                kind=link["kind"],
                sigma=float(link.get("sigma", 1.0)),
                params=dict(link.get("params", {})),
            ),
            runtime=RuntimeConfig(
                max_iter=int(runtime.get("max_iter", 1000)),
                tol=float(runtime.get("tol", 1e-6)),
                diagnostics_interval=runtime.get("diagnostics_interval", 100),
                dtype=runtime.get("dtype", "float64"),
                checkpoint_iters=tuple(runtime.get("checkpoint_iters", ())),
            ),
            bcd=BcdConfig(
                project_z=bcd_params.project_z,
                freeze_phi_updates=bcd_params.freeze_phi_updates,
            ),
        )

    def validate_params(self, *, params: Mapping[str, Any]) -> None:
        """Validate BCD-owned parameters before data loading or output creation."""
        parse_bcd_parameters(params)

    def run(
        self,
        *,
        config: ModelConfig,
        dataset: DatasetBundle,
        objective: Objective,
        z_init: np.ndarray | None = None,
        phi_init: ScalarFunction | None = None,
        diagnostics_request: DiagnosticRequest | None = None,
        selection_policy: SelectionPolicy | None = None,
        rng: np.random.Generator,
    ) -> AlgorithmResult:
        """Run BCD optimization with modular phi and Z updates."""
        self.validate_dataset(config=config, dataset=dataset)
        state = init_state(
            dataset.train,
            config,
            z_init=z_init,
            phi_init=phi_init,
            rng=rng,
        )
        require_finite(state.values, "training observations", 0)
        require_finite(state.z, "initial Z state", 0)
        require_finite(state.centers, "initial training centers", 0)
        # Resolve representation mode (PiecewiseLinearRKHS vs GaussianRKHS/other)
        phi_type: PhiType | str | None = self.phi_type
        if phi_type is None:
            kind = config.link.kind
            phi_type = PhiType.from_str(kind) if isinstance(kind, str) else kind

        if state.phi is None and phi_type != PhiType.IDENTITY:
            raise ValueError(
                "BCD requires a phi initializer for non-identity links; "
                "initialization.phi.name='none' is valid only with link.kind='identity'"
            )
        if state.phi is not None and not isinstance(state.phi, ScalarFunction):
            raise TypeError(
                f"BCD link must implement ScalarFunction, got {type(state.phi).__name__}"
            )
        if state.phi is not None and not isinstance(state.phi, CopyableFunction):
            raise TypeError("BCD links must implement CopyableFunction")
        if isinstance(state.phi, TrainableFunction) and not isinstance(
            state.phi, ProjectableFunction
        ):
            raise TypeError("BCD trainable links must also implement ProjectableFunction")

        phi_ball_radius = None
        if isinstance(state.phi, BallRadiusResolver):
            phi_ball_radius = state.phi.resolve_projection_ball_radius(
                truth=dataset.capabilities.functions.truth,
            )
        elif (
            state.phi is not None
            and config.link.params.get("ball_enabled", False)
            and config.link.params.get("ball_radius") is None
        ):
            raise TypeError(
                f"Link {state.phi.kind!r} must implement BallRadiusResolver "
                "when its projection radius is inferred from dataset truth"
            )

        link_kind = (
            config.link.kind.value if isinstance(config.link.kind, PhiType) else config.link.kind
        )
        objective.validate_link(str(link_kind))
        context = AlgorithmContext(
            config=config,
            dataset=dataset,
            objective=objective,
        )

        update_z_step = make_z_updater(is_identity=(phi_type == PhiType.IDENTITY))

        def update_step(s: SolverState, ctx: AlgorithmContext, iteration: int) -> None:
            update_z_step(s, ctx, iteration)
            _update_phi(
                s,
                ctx,
                iteration,
                phi_ball_radius=phi_ball_radius,
                rng=rng,
            )

        return run_bcd_iterations(
            state=state,
            context=context,
            update=update_step,
            publish_snapshot=publish_bcd_snapshot,
            diagnostics_request=diagnostics_request,
            selection_policy=selection_policy,
            capture_checkpoint=self.capture_checkpoint,
        )

    def capture_checkpoint(
        self,
        state: SolverState,
        elapsed_seconds: float,
    ) -> AlgorithmCheckpoint:
        """Capture BCD-owned factors and component-owned link state."""
        phi_checkpoint = None if state.phi is None else serialize_function_model(state.phi)
        return AlgorithmCheckpoint(
            format_version=CHECKPOINT_FORMAT_VERSION,
            algorithm=self.name,
            state_version=self.checkpoint_version,
            elapsed_seconds=elapsed_seconds,
            payload={
                "z": state.z.copy(),
                "phi": phi_checkpoint,
                "z_optimizer": serialize_optimizer(state.z_optimizer),
            },
        )

    def restore_checkpoint(
        self,
        checkpoint: AlgorithmCheckpoint,
        *,
        config: ModelConfig,
        dataset: DatasetBundle | None = None,
    ) -> _BcdRestoredState:
        """Restore BCD initialization values from a compatible checkpoint."""
        if checkpoint.format_version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"Unsupported checkpoint format_version {checkpoint.format_version}")
        if checkpoint.algorithm != self.name:
            raise ValueError(f"Cannot restore {checkpoint.algorithm!r} checkpoint with BCD")
        if checkpoint.state_version != self.checkpoint_version:
            raise ValueError(f"Unsupported BCD state_version {checkpoint.state_version}")
        z = np.asarray(checkpoint.payload["z"])
        expected_shape = (
            config.shape.n_rows + config.shape.n_cols,
            config.shape.rank,
        )
        if z.shape != expected_shape:
            raise ValueError(f"checkpoint factors must have shape {expected_shape}, got {z.shape}")
        if z.dtype != config.runtime.np_dtype:
            raise ValueError(
                f"checkpoint factors must have dtype {config.runtime.np_dtype}, got {z.dtype}"
            )
        require_finite(z, "checkpoint Z state", 0)
        phi_payload = checkpoint.payload.get("phi")
        if phi_payload is None:
            phi = None
        elif isinstance(phi_payload, Mapping):
            phi = restore_function_model(
                phi_payload,
                config=config,
                dataset=dataset,
                z=z,
            )
        else:
            raise ValueError("checkpoint phi payload must be a mapping or null")
        optimizer_payload = checkpoint.payload.get("z_optimizer")
        if not isinstance(optimizer_payload, Mapping):
            raise ValueError("checkpoint z_optimizer payload must be a mapping")
        optimizer = restore_optimizer(
            optimizer_payload,
            spec=config.z_optimizer,
            parameter=z,
        )
        return _BcdRestoredState(z=z.copy(), phi=phi, z_optimizer=optimizer)

    def export_selected_state(
        self,
        checkpoint: AlgorithmCheckpoint,
        *,
        config: ModelConfig,
        dataset: DatasetBundle,
    ) -> Mapping[str, Any]:
        """Export small scientific views of one restored BCD checkpoint."""
        restored = self.restore_checkpoint(
            checkpoint,
            config=config,
            dataset=dataset,
        )
        z_rows = restored.z[: config.shape.n_rows]
        z_columns = restored.z[config.shape.n_rows :]
        lower, upper = _inner_product_range(z_rows, z_columns)
        exports: dict[str, Any] = {
            "inner_matrix_range": {
                "min": lower,
                "max": upper,
            }
        }
        if isinstance(restored.phi, SerializableFunctionModel):
            model_payload = restored.phi.to_payload()
            inputs = _link_curve_inputs(model_payload, lower=lower, upper=upper)
            outputs = np.asarray(restored.phi.eval_array(inputs), dtype=float)
            require_finite(outputs, "selected link curve", 0)
            exports["link_function"] = model_payload
            exports["link_curve"] = {"inputs": inputs, "outputs": outputs}
        return exports

    def validate_dataset(self, *, config: ModelConfig, dataset: DatasetBundle) -> None:
        """Validate capabilities required by configured BCD constraints."""
        params = config.link.params
        if not bool(params.get("ball_enabled", False)):
            return
        if params.get("phi_0") is None:
            raise ValueError("link.params.ball_enabled requires link.params.phi_0")
        if params.get("ball_radius") is None and dataset.capabilities.functions.truth is None:
            raise ValueError(
                "link-function ball projection without link.params.ball_radius requires "
                "dataset.capabilities.functions.truth"
            )
