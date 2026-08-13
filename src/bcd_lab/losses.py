"""Differentiable data-fit objectives and their registry.

Objectives operate only on observed values and model predictions.  Algorithms
own regularization because its parameters and gradients are algorithm-specific.
The derivative convention is always ``d(value) / d(predictions)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from bcd_lab._registry import ComponentRegistry

ObjectiveValueFn = Callable[[np.ndarray, np.ndarray, Mapping[str, Any]], float]
PredictionGradientFn = Callable[[np.ndarray, np.ndarray, Mapping[str, Any]], np.ndarray]


@dataclass(frozen=True)
class Objective:
    """Configured differentiable data-fit objective.

    ``prediction_gradient`` returns an array with the same shape as
    ``predictions`` containing ``d(value) / d(predictions)``.
    """

    name: str
    value_fn: ObjectiveValueFn
    prediction_gradient_fn: PredictionGradientFn
    params: Mapping[str, Any] = field(default_factory=dict)
    compatible_link_kinds: frozenset[str] | None = None
    squared_error_scale: float | None = None

    def value(self, observed: np.ndarray, predictions: np.ndarray) -> float:
        """Return the unregularized data-fit objective value."""
        observed_array, prediction_array = _aligned_arrays(observed, predictions)
        return float(self.value_fn(observed_array, prediction_array, self.params))

    def prediction_gradient(
        self,
        observed: np.ndarray,
        predictions: np.ndarray,
    ) -> np.ndarray:
        """Return ``d(value) / d(predictions)`` for every observation."""
        observed_array, prediction_array = _aligned_arrays(observed, predictions)
        gradient = np.asarray(
            self.prediction_gradient_fn(
                observed_array,
                prediction_array,
                self.params,
            ),
            dtype=prediction_array.dtype,
        )
        if gradient.shape != prediction_array.shape:
            raise ValueError(
                f"Objective {self.name!r} returned prediction gradient shape "
                f"{gradient.shape}; expected {prediction_array.shape}"
            )
        return gradient

    def validate_link(self, link_kind: str) -> None:
        """Raise when this objective cannot be used with a link representation."""
        if self.compatible_link_kinds is not None and link_kind not in self.compatible_link_kinds:
            supported = ", ".join(sorted(self.compatible_link_kinds))
            raise ValueError(
                f"Objective {self.name!r} does not support link kind "
                f"{link_kind!r}; supported kinds: {supported}"
            )


@dataclass(frozen=True)
class ObjectiveDefinition:
    """Registry entry that creates configured objective instances."""

    name: str
    value: ObjectiveValueFn
    prediction_gradient: PredictionGradientFn
    compatible_link_kinds: frozenset[str] | None = None
    squared_error_scale: float | None = None

    def build(self, params: Mapping[str, Any] | None = None) -> Objective:
        return Objective(
            name=self.name,
            value_fn=self.value,
            prediction_gradient_fn=self.prediction_gradient,
            params=dict(params or {}),
            compatible_link_kinds=self.compatible_link_kinds,
            squared_error_scale=self.squared_error_scale,
        )


_OBJECTIVES = ComponentRegistry[ObjectiveDefinition]("Objective", "objectives")


def register_objective(
    name: str,
    *,
    value: ObjectiveValueFn,
    prediction_gradient: PredictionGradientFn,
    compatible_link_kinds: set[str] | frozenset[str] | None = None,
    squared_error_scale: float | None = None,
) -> ObjectiveDefinition:
    """Register a differentiable objective definition."""
    definition = ObjectiveDefinition(
        name=name,
        value=value,
        prediction_gradient=prediction_gradient,
        compatible_link_kinds=(
            None if compatible_link_kinds is None else frozenset(compatible_link_kinds)
        ),
        squared_error_scale=squared_error_scale,
    )
    return _OBJECTIVES.register(name, definition)


def get_objective(
    name: str,
    params: Mapping[str, Any] | None = None,
) -> Objective:
    """Build a configured objective from a registered definition."""
    definition = _OBJECTIVES.get(name)
    return definition.build(params)


def list_objectives() -> tuple[str, ...]:
    """List registered objective names."""
    return _OBJECTIVES.names()


def build_balancing_operator(
    n_rows: int,
    n_cols: int,
    dtype: np.dtype | str,
) -> np.ndarray:
    """Build the dense row/column balancing operator used by Z regularization."""
    np_dtype = np.dtype(dtype)
    return np.block(
        [
            [
                np.eye(n_rows, dtype=np_dtype) / n_rows,
                np.zeros((n_rows, n_cols), dtype=np_dtype),
            ],
            [
                np.zeros((n_cols, n_rows), dtype=np_dtype),
                -np.eye(n_cols, dtype=np_dtype) / n_cols,
            ],
        ]
    )


def z_regularization(z: np.ndarray, d: np.ndarray, lambda_: float) -> float:
    """Balanced factor regularization term."""
    if lambda_ == 0:
        return 0.0
    zt_d_z = z.T @ d @ z
    return (float(lambda_) / 4.0) * float(np.sum(zt_d_z * zt_d_z))


def _aligned_arrays(
    observed: np.ndarray,
    predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prediction_array = np.asarray(predictions)
    observed_array = np.asarray(observed, dtype=prediction_array.dtype)
    if observed_array.shape != prediction_array.shape:
        raise ValueError(
            "observed values and predictions must have the same shape, got "
            f"{observed_array.shape} and {prediction_array.shape}"
        )
    return observed_array, prediction_array


def _squared_value(
    observed: np.ndarray,
    predictions: np.ndarray,
    params: Mapping[str, Any],
) -> float:
    _reject_params(params)
    residuals = predictions - observed
    return float(np.dot(residuals, residuals))


def _squared_prediction_gradient(
    observed: np.ndarray,
    predictions: np.ndarray,
    params: Mapping[str, Any],
) -> np.ndarray:
    _reject_params(params)
    return 2.0 * (predictions - observed)


def _half_squared_value(
    observed: np.ndarray,
    predictions: np.ndarray,
    params: Mapping[str, Any],
) -> float:
    _reject_params(params)
    residuals = predictions - observed
    return 0.5 * float(np.dot(residuals, residuals))


def _half_squared_prediction_gradient(
    observed: np.ndarray,
    predictions: np.ndarray,
    params: Mapping[str, Any],
) -> np.ndarray:
    _reject_params(params)
    return predictions - observed


def _reject_params(params: Mapping[str, Any]) -> None:
    if params:
        names = ", ".join(sorted(params))
        raise ValueError(f"This objective does not accept parameters: {names}")


register_objective(
    "squared_error",
    value=_squared_value,
    prediction_gradient=_squared_prediction_gradient,
    squared_error_scale=1.0,
)
register_objective(
    "half_squared",
    value=_half_squared_value,
    prediction_gradient=_half_squared_prediction_gradient,
    squared_error_scale=0.5,
)
