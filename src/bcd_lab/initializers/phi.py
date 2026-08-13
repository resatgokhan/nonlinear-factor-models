"""Phi initialization strategies."""

from __future__ import annotations

from typing import Any

import numpy as np

from bcd_lab.config import ModelBuildContext
from bcd_lab.function_models import (
    ScalarFunction,
    TrainableFunction,
    get_function_model_builder,
)
from bcd_lab.types import DatasetBundle


def build_zero_phi(
    config: ModelBuildContext,
    params: dict[str, Any] | None = None,
    dataset: DatasetBundle | None = None,
    z: np.ndarray | None = None,
) -> ScalarFunction:
    """Initialize the correct zero-initialized function model dynamically using the registry."""
    kind = config.link.kind
    kind_str = kind.value if hasattr(kind, "value") else str(kind)

    builder = get_function_model_builder(kind_str)
    return builder(config, params=params, dataset=dataset, z=z)


def identity_phi(
    config: ModelBuildContext,
    params: dict[str, object] | None = None,
    dataset: DatasetBundle | None = None,
    z: np.ndarray | None = None,
) -> ScalarFunction:
    """Initialize the configured trainable grid model as ``phi(x)=x``."""
    phi = build_zero_phi(config, params, dataset=dataset, z=z)
    if phi.kind not in {"piecewise_linear_rkhs", "binned_piecewise_linear_rkhs"}:
        raise ValueError(
            "identity_phi currently supports piecewise_linear_rkhs and binned_piecewise_linear_rkhs"
        )
    scalars = getattr(phi, "scalars", None)
    coefs = getattr(phi, "coefs", None)
    if scalars is None or coefs is None:
        raise ValueError(
            "identity_phi requires a trainable grid phi with initialized scalars and coefs"
        )
    if not isinstance(phi, TrainableFunction):
        raise TypeError("identity_phi requires a trainable function capability")

    identity_values = np.asarray(scalars, dtype=coefs.dtype)
    phi.fit(np.asarray(scalars, dtype=coefs.dtype), identity_values)
    scalars = getattr(phi, "scalars", None)
    coefs = getattr(phi, "coefs", None)
    if scalars is None or coefs is None:
        raise ValueError("identity_phi fit left phi without initialized scalars and coefs")
    identity_values = np.asarray(scalars, dtype=coefs.dtype)
    if not np.allclose(phi.eval_array(identity_values), identity_values):
        fitted_coefs = phi.coefs
        fitted_coefs[:] = identity_values
    return phi


class ZeroPhiInitializer:
    """Registered generic zero phi initializer."""

    name = "zero"

    def build(
        self,
        *,
        dataset: DatasetBundle,
        config: ModelBuildContext,
        params: dict[str, object],
        seed: int,
        z: np.ndarray | None = None,
    ) -> ScalarFunction:
        _ = seed
        return build_zero_phi(config, params, dataset=dataset, z=z)


class IdentityPhiInitializer:
    """Registered trainable identity-shaped phi initializer."""

    name = "identity_phi"

    def build(
        self,
        *,
        dataset: DatasetBundle,
        config: ModelBuildContext,
        params: dict[str, object],
        seed: int,
        z: np.ndarray | None = None,
    ) -> ScalarFunction:
        _ = seed
        return identity_phi(config, params, dataset=dataset, z=z)


class NonePhiInitializer:
    """Select the identity-link path without constructing a function model."""

    name = "none"

    def build(
        self,
        *,
        dataset: DatasetBundle,
        config: ModelBuildContext,
        params: dict[str, object],
        seed: int,
        z: np.ndarray | None = None,
    ) -> None:
        _ = dataset, config, params, seed, z
        return None


class GenericPhiInitializer:
    """Generic initializer wrapper for any registered function model."""

    def __init__(self, name: str):
        self.name = name

    def build(
        self,
        *,
        dataset: DatasetBundle,
        config: ModelBuildContext,
        params: dict[str, object],
        seed: int,
        z: np.ndarray | None = None,
        **kwargs: Any,
    ) -> ScalarFunction | None:
        builder = get_function_model_builder(self.name)
        return builder(config, params=params, dataset=dataset, z=z)
