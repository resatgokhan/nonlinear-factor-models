"""Lazy diagnostic snapshot publication for the BCD algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from bcd_lab.config import ModelConfig
from bcd_lab.diagnostics import DiagnosticRequest, ObservedSplit, RunSnapshot
from bcd_lab.function_models import (
    CopyableFunction,
    NormedFunction,
    ScalarFunction,
    TrainableFunction,
)
from bcd_lab.losses import build_balancing_operator, z_regularization
from bcd_lab.math import compute_centers
from bcd_lab.types import SampleSet

from .numerics import require_finite
from .registry import AlgorithmContext
from .state import SolverState
from .updates_z import compute_z_gradient, compute_z_gradient_identity

MetricSelection = set[str] | None


def _z_regularization_value(state: SolverState, config: ModelConfig) -> float:
    lambda_ = config.regularization.lambda_
    if lambda_ == 0:
        return 0.0
    operator = state.cache.get("balancing_operator")
    if operator is None:
        operator = build_balancing_operator(
            config.shape.n_rows,
            config.shape.n_cols,
            state.z.dtype,
        )
        state.cache["balancing_operator"] = operator
    return z_regularization(state.z, operator, lambda_)


def _phi_regularization_value(
    phi: ScalarFunction | None,
    config: ModelConfig,
) -> float:
    if phi is None or config.regularization.alpha == 0:
        return 0.0
    norm_value = phi.norm(mode="auto") if isinstance(phi, NormedFunction) else None
    if norm_value is None:
        return 0.0
    return (config.regularization.alpha / 2.0) * float(norm_value) ** 2


def _evaluate_objective(state: SolverState, context: AlgorithmContext, iteration: int) -> float:
    centers = state.centers
    if centers is None:
        centers = compute_centers(
            state.z,
            state.rows,
            state.cols,
            context.config.shape.n_rows,
        )
        state.centers = centers
    predictions = centers if state.phi is None else state.phi.eval_array(centers)
    require_finite(predictions, "diagnostic predictions", iteration)
    objective_value = float(
        context.objective.value(state.values, predictions)
        + _z_regularization_value(state, context.config)
        + _phi_regularization_value(state.phi, context.config)
    )
    require_finite(objective_value, "reported objective", iteration)
    return objective_value


def _copy_link(phi: ScalarFunction | None) -> ScalarFunction | None:
    if phi is None:
        return None
    if not isinstance(phi, CopyableFunction):
        raise TypeError("snapshot links must implement CopyableFunction")
    return phi.copy()


def _predict_split(
    state: SolverState,
    context: AlgorithmContext,
    sample_set: SampleSet,
) -> ObservedSplit:
    observed, (rows, cols) = sample_set
    centers = compute_centers(
        state.z,
        np.asarray(rows),
        np.asarray(cols),
        context.config.shape.n_rows,
    )
    predictions = centers if state.phi is None else state.phi.eval_array(centers)
    return ObservedSplit(observed=np.asarray(observed), predictions=predictions)


def _dense_predictions(
    z: np.ndarray,
    n_rows: int,
    phi: ScalarFunction | None,
    *,
    include_linked: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    inner = z[:n_rows] @ z[n_rows:].T
    if not include_linked:
        return inner, None
    linked = inner if phi is None else phi.eval_array(inner.ravel()).reshape(inner.shape)
    return inner, linked


def _phi_gradient_norm(state: SolverState, context: AlgorithmContext) -> float | None:
    phi = state.phi
    if phi is None or not isinstance(phi, TrainableFunction):
        return None
    centers = state.centers
    if centers is None:
        centers = compute_centers(
            state.z,
            state.rows,
            state.cols,
            context.config.shape.n_rows,
        )
    predictions = phi.eval_array(centers)
    direction = -context.objective.prediction_gradient(state.values, predictions)
    gradient = phi.gradient(inputs=centers, residuals=direction)
    if isinstance(gradient, NormedFunction):
        norm_value = gradient.norm(mode="rkhs")
        if norm_value is not None:
            return float(norm_value)
    if isinstance(gradient, np.ndarray):
        return float(np.linalg.norm(gradient))
    return None


@dataclass
class _BcdSnapshotBuilder:
    """Build only the payloads requested by diagnostic families."""

    state: SolverState
    context: AlgorithmContext
    iteration: int
    elapsed_seconds: float
    sec_per_iter: float
    rem_sec: float
    capabilities: dict[str, Any] = field(default_factory=dict)
    observed_splits: dict[str, ObservedSplit] = field(default_factory=dict)

    def factors(self) -> np.ndarray:
        factors = self.capabilities.get("latent_factors")
        if factors is None:
            factors = self.state.z.copy()
            self.capabilities["latent_factors"] = factors
        return factors

    def link(self) -> ScalarFunction | None:
        if "link_function" not in self.capabilities:
            link = _copy_link(self.state.phi)
            if link is not None:
                self.capabilities["link_function"] = link
        return self.capabilities.get("link_function")

    def training_centers(self) -> np.ndarray:
        centers = self.capabilities.get("training_centers")
        if centers is None:
            state_centers = self.state.centers
            if state_centers is None:
                state_centers = compute_centers(
                    self.state.z,
                    self.state.rows,
                    self.state.cols,
                    self.context.config.shape.n_rows,
                )
            centers = np.asarray(state_centers).copy()
            self.capabilities["training_centers"] = centers
        return centers

    def snapshot(self) -> RunSnapshot:
        return RunSnapshot(
            iteration=self.iteration,
            elapsed_seconds=self.elapsed_seconds,
            observed_splits=self.observed_splits,
            objective_value=_evaluate_objective(self.state, self.context, self.iteration),
            objective_name=self.context.objective.name,
            capabilities=self.capabilities,
        )


CapabilityPublisher = Callable[[_BcdSnapshotBuilder, MetricSelection], None]


def _publish_truth_z(builder: _BcdSnapshotBuilder, metrics: MetricSelection) -> None:
    _ = metrics
    builder.factors()


def _publish_phi(builder: _BcdSnapshotBuilder, metrics: MetricSelection) -> None:
    builder.link()
    if metrics is not None and "phi_d_in_star_bounds" in metrics:
        builder.training_centers()


def _publish_reconstruction(
    builder: _BcdSnapshotBuilder,
    metrics: MetricSelection,
) -> None:
    _ = metrics
    for name in ("train", "validation", "test"):
        sample_set = getattr(builder.context.dataset, name)
        if sample_set is not None:
            builder.observed_splits[name] = _predict_split(
                builder.state,
                builder.context,
                sample_set,
            )


def _publish_truth_matrix(
    builder: _BcdSnapshotBuilder,
    metrics: MetricSelection,
) -> None:
    linked_metrics = {"phi_x_error", "relative_phi_x_fro", "phi_x_rmse"}
    include_linked = bool(metrics is not None and metrics & linked_metrics)
    inner, linked = _dense_predictions(
        builder.state.z,
        builder.context.config.shape.n_rows,
        builder.state.phi,
        include_linked=include_linked,
    )
    builder.capabilities["inner_matrix"] = inner
    if linked is not None:
        builder.capabilities["linked_inner_matrix"] = linked


def _publish_gradients(
    builder: _BcdSnapshotBuilder,
    metrics: MetricSelection,
) -> None:
    if not metrics or "z_grad_norm" in metrics:
        is_identity = builder.context.config.link.kind == "identity"
        builder.capabilities["z_gradient"] = (
            compute_z_gradient_identity(builder.state, builder.context, builder.iteration)
            if is_identity
            else compute_z_gradient(builder.state, builder.context, builder.iteration)
        )
    if not metrics or "phi_grad_norm" in metrics:
        builder.capabilities["phi_grad_norm"] = _phi_gradient_norm(
            builder.state,
            builder.context,
        )
    if not metrics or "z_optimizer_update_norm" in metrics:
        builder.capabilities["z_optimizer_update_norm"] = builder.state.cache.get(
            "z_optimizer_update_norm"
        )


def _publish_timing(builder: _BcdSnapshotBuilder, metrics: MetricSelection) -> None:
    _ = metrics
    builder.capabilities["sec_per_iter"] = builder.sec_per_iter
    builder.capabilities["rem_sec"] = builder.rem_sec


def _publish_phi_sharp(
    builder: _BcdSnapshotBuilder,
    metrics: MetricSelection,
) -> None:
    _ = metrics
    builder.factors()
    builder.link()
    centers = builder.training_centers()
    alpha = builder.context.config.regularization.alpha
    objective = builder.context.objective
    z_regularization_value = _z_regularization_value(
        builder.state,
        builder.context.config,
    )
    observed = np.asarray(builder.state.values).copy()

    def evaluate_link(link: ScalarFunction) -> float:
        predictions = link.eval_array(centers)
        norm_value = link.norm(mode="auto") if isinstance(link, NormedFunction) else None
        link_regularization = 0.0 if norm_value is None else (alpha / 2.0) * float(norm_value) ** 2
        return float(
            objective.value(observed, predictions) + z_regularization_value + link_regularization
        )

    builder.capabilities.update(
        {
            "training_observed": observed,
            "link_regularization": alpha,
            "squared_error_scale": objective.squared_error_scale,
            "link_objective_evaluator": evaluate_link,
        }
    )


_CAPABILITY_PUBLISHERS: Mapping[str, CapabilityPublisher] = {
    "gradients": _publish_gradients,
    "phi": _publish_phi,
    "phi_sharp": _publish_phi_sharp,
    "reconstruction": _publish_reconstruction,
    "timing": _publish_timing,
    "truth_matrix": _publish_truth_matrix,
    "truth_z": _publish_truth_z,
}


def publish_bcd_snapshot(
    state: SolverState,
    context: AlgorithmContext,
    iteration: int,
    elapsed_seconds: float,
    sec_per_iter: float,
    rem_sec: float,
    diagnostics_request: DiagnosticRequest,
) -> RunSnapshot:
    """Publish a detached BCD snapshot using family-owned payload builders."""
    builder = _BcdSnapshotBuilder(
        state=state,
        context=context,
        iteration=iteration,
        elapsed_seconds=elapsed_seconds,
        sec_per_iter=sec_per_iter,
        rem_sec=rem_sec,
    )
    for family, metrics in diagnostics_request.families.items():
        publisher = _CAPABILITY_PUBLISHERS.get(family)
        if publisher is not None:
            publisher(builder, metrics)
    return builder.snapshot()
