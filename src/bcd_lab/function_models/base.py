"""Small, composable capabilities for scalar function models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class FunctionConstraintConfig:
    """Shared projection preferences for trainable function implementations."""

    monotone_enabled: bool = False
    monotone_direction: Literal["increasing", "decreasing"] = "increasing"
    clamp_enabled: bool = False
    clamp_min: float | None = None
    clamp_max: float | None = None
    ball_enabled: bool = False
    ball_center_phi_0: Any | None = None
    ball_radius: float | None = None


@runtime_checkable
class ScalarFunction(Protocol):
    """Evaluate a scalar-to-scalar function over NumPy arrays."""

    kind: str

    def eval_array(self, x: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class CopyableFunction(ScalarFunction, Protocol):
    """A scalar function that can detach its mutable state."""

    def copy(self) -> ScalarFunction: ...


@runtime_checkable
class DifferentiableFunction(ScalarFunction, Protocol):
    """A scalar function exposing its input derivative."""

    def d_array(self, x: np.ndarray) -> np.ndarray: ...

    def eval_and_d_array(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


@runtime_checkable
class TrainableFunction(DifferentiableFunction, Protocol):
    """A mutable function supporting the BCD update convention.

    ``gradient`` returns the parameter-space direction induced by ``residuals``.
    ``apply_gradient`` adds ``step_size * gradient`` to the current parameters.
    Regularization is applied first through ``scale_parameters``.
    """

    def fit(self, x: np.ndarray, y: np.ndarray) -> None: ...

    def scale_parameters(self, scale: float) -> None: ...

    def gradient(self, *, inputs: np.ndarray, residuals: np.ndarray) -> Any: ...

    def apply_gradient(self, gradient: Any, *, step_size: float) -> None: ...


@runtime_checkable
class UpdatePreparedFunction(Protocol):
    """Optional preparation of model coordinates before update residuals are evaluated."""

    def prepare_update(self, inputs: np.ndarray) -> None: ...


@runtime_checkable
class ProjectableFunction(Protocol):
    """A mutable function whose configured constraints can be applied in place."""

    def project(
        self,
        *,
        centers: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        l2_grid: np.ndarray | None = None,
        phi_ball_radius: float | None = None,
    ) -> None: ...


@runtime_checkable
class BallRadiusResolver(Protocol):
    """Optional model-owned resolution of a configured projection-ball radius."""

    def resolve_projection_ball_radius(
        self,
        *,
        truth: ScalarFunction | None,
    ) -> float | None:
        """Return the active radius, inferring it from ``truth`` when configured."""


@runtime_checkable
class SerializableFunctionModel(ScalarFunction, Protocol):
    """A function model with an artifact-ready payload representation."""

    def to_payload(self) -> dict[str, Any]: ...


@runtime_checkable
class CheckpointableFunction(ScalarFunction, Protocol):
    """A function model that owns its versioned mutable checkpoint state."""

    checkpoint_version: int

    def checkpoint_state(self) -> Mapping[str, Any]: ...

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        version: int,
    ) -> None: ...


@runtime_checkable
class NativeDistanceFunction(Protocol):
    """Optional model-native distance capability used by diagnostics."""

    def distance_to(
        self,
        other: object,
        *,
        mode: str = "auto",
        grid: np.ndarray | None = None,
    ) -> float: ...


@runtime_checkable
class NormedFunction(Protocol):
    """Optional model-native norm capability used by objectives and diagnostics."""

    def norm(self, *, mode: str = "auto") -> float | None: ...
