"""BCD-facing configuration dataclasses retained as its direct-call API.

Experiment orchestration no longer constructs or extends these types. Each
registered algorithm owns its configuration through ``Algorithm.build_config``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np

from bcd_lab.optimizers import OptimizerSpec


class PhiType(Enum):
    IDENTITY = "identity"
    SIGMOID = "sigmoid"
    EXPONENTIAL = "exponential"
    GAUSSIAN_RKHS = "gaussian_rkhs"
    PIECEWISE_LINEAR_RKHS = "piecewise_linear_rkhs"
    BINNED_PIECEWISE_LINEAR_RKHS = "binned_piecewise_linear_rkhs"

    @classmethod
    def from_str(cls, value: str) -> PhiType | str:
        """Coerce a string representation into a PhiType Enum member."""
        if value == "piecewise_linear_rkhs":
            return cls.PIECEWISE_LINEAR_RKHS
        elif value == "binned_piecewise_linear_rkhs":
            return cls.BINNED_PIECEWISE_LINEAR_RKHS
        elif value == "gaussian_rkhs":
            return cls.GAUSSIAN_RKHS
        elif value == "identity":
            return cls.IDENTITY
        elif value == "sigmoid":
            return cls.SIGMOID
        elif value == "exponential":
            return cls.EXPONENTIAL
        return value


def coerce_float_dtype(value: Any | None) -> np.dtype:
    """Convert user-provided dtype values into a NumPy floating dtype."""
    if value is None:
        return np.dtype(np.float64)
    dtype = np.dtype(value)
    if dtype.kind != "f":
        raise ValueError(f"dtype must be floating, got {dtype}")
    return dtype


@dataclass(frozen=True)
class ShapeConfig:
    n_rows: int
    n_cols: int
    rank: int
    n_observations: int


@dataclass(frozen=True)
class PhiUpdateConfig:
    """BCD's representation-owned link update settings."""

    learning_rate: float


@dataclass(frozen=True)
class RegularizationConfig:
    lambda_: float
    mu: float
    alpha: float


@dataclass(frozen=True)
class LinkConfig:
    kind: str | PhiType
    sigma: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeConfig:
    max_iter: int
    tol: float
    dtype: np.dtype[Any] | str | type[np.floating[Any]] = np.float64
    diagnostics_interval: int | None = 100
    checkpoint_iters: tuple[int, ...] = ()

    @property
    def np_dtype(self) -> np.dtype:
        return coerce_float_dtype(self.dtype)


class ModelBuildContext(Protocol):
    """Structural contract shared by initializers and model builders."""

    @property
    def shape(self) -> ShapeConfig: ...

    @property
    def link(self) -> LinkConfig: ...

    @property
    def runtime(self) -> RuntimeConfig: ...


@dataclass(frozen=True)
class BasicModelBuildContext:
    """Algorithm-neutral model-building context used by experiments."""

    shape: ShapeConfig
    link: LinkConfig
    runtime: RuntimeConfig


@dataclass(frozen=True)
class BcdConfig:
    project_z: bool = True
    freeze_phi_updates: bool = False


@dataclass(frozen=True)
class ModelConfig:
    shape: ShapeConfig
    z_optimizer: OptimizerSpec
    phi_update: PhiUpdateConfig
    regularization: RegularizationConfig
    link: LinkConfig
    runtime: RuntimeConfig
    bcd: BcdConfig = field(default_factory=BcdConfig)
