"""Duplicate-safe registry for fixed-coordinate array optimizers."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from bcd_lab._registry import ComponentRegistry

from .base import ArrayOptimizer, OptimizerBuilder, OptimizerSpec

_OPTIMIZERS = ComponentRegistry[OptimizerBuilder]("Optimizer", "optimizers")


def register_optimizer(name: str, builder: OptimizerBuilder) -> OptimizerBuilder:
    """Register a unique optimizer builder."""
    return _OPTIMIZERS.register(name, builder)


def get_optimizer_builder(name: str) -> OptimizerBuilder:
    """Return the registered builder for ``name``."""
    return _OPTIMIZERS.get(name)


def list_optimizers() -> tuple[str, ...]:
    """Return registered optimizer names deterministically."""
    return _OPTIMIZERS.names()


def build_optimizer(
    spec: OptimizerSpec,
    parameter: np.ndarray,
) -> ArrayOptimizer:
    """Build a fresh optimizer instance for ``parameter``."""
    if not isinstance(spec.params, Mapping):
        raise TypeError("optimizer params must be a mapping")
    optimizer = get_optimizer_builder(spec.name)(spec.params, parameter)
    if not isinstance(optimizer, ArrayOptimizer):
        raise TypeError(f"Registered optimizer {spec.name!r} did not return an ArrayOptimizer")
    if optimizer.name != spec.name:
        raise ValueError(f"Registered optimizer {spec.name!r} returned name {optimizer.name!r}")
    return optimizer


def parse_optimizer_spec(value: object, *, field: str) -> OptimizerSpec:
    """Parse a canonical optimizer mapping without compatibility aliases."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    unknown = sorted(set(value) - {"name", "params"})
    if unknown:
        raise ValueError(f"{field} contains unknown fields: {', '.join(unknown)}")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{field}.name must be a non-empty string")
    params = value.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError(f"{field}.params must be a mapping")
    return OptimizerSpec(name=name, params=dict(params))
