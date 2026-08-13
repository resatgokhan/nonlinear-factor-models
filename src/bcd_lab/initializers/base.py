"""Initializer protocols."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from bcd_lab.config import ModelBuildContext
from bcd_lab.function_models import ScalarFunction
from bcd_lab.types import DatasetBundle


class ZInitializer(Protocol):
    """Build an initial latent factor matrix."""

    name: str

    def build(
        self,
        *,
        dataset: DatasetBundle,
        config: ModelBuildContext,
        params: dict[str, Any],
        seed: int,
    ) -> np.ndarray: ...


class PhiInitializer(Protocol):
    """Build an initial link function."""

    name: str

    def build(
        self,
        *,
        dataset: DatasetBundle,
        config: ModelBuildContext,
        params: dict[str, Any],
        seed: int,
        z: np.ndarray | None = None,
    ) -> ScalarFunction | None: ...


class JointInitializer(Protocol):
    """Build a coordinated initialization tuple of (Z, phi)."""

    name: str

    def build(
        self,
        *,
        dataset: DatasetBundle,
        config: ModelBuildContext,
        params: dict[str, Any],
        seed: int,
    ) -> tuple[np.ndarray, ScalarFunction | None]: ...
