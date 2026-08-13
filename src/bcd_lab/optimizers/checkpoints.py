"""Versioned optimizer checkpoint envelopes."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import ArrayOptimizer, OptimizerSpec
from .registry import build_optimizer

OPTIMIZER_CHECKPOINT_FORMAT_VERSION = 1


def serialize_optimizer(optimizer: ArrayOptimizer) -> dict[str, Any]:
    """Serialize a stateful optimizer without exposing its state to BCD."""
    if optimizer.state_version <= 0:
        raise ValueError("optimizer state_version must be positive")
    return {
        "format_version": OPTIMIZER_CHECKPOINT_FORMAT_VERSION,
        "name": optimizer.name,
        "state_version": optimizer.state_version,
        "state": dict(optimizer.checkpoint_state()),
    }


def restore_optimizer(
    checkpoint: Mapping[str, Any],
    *,
    spec: OptimizerSpec,
    parameter: np.ndarray,
) -> ArrayOptimizer:
    """Build and restore the configured optimizer for ``parameter``."""
    if checkpoint.get("format_version") != OPTIMIZER_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported optimizer checkpoint format_version {checkpoint.get('format_version')!r}"
        )
    name = checkpoint.get("name")
    if name != spec.name:
        raise ValueError(
            f"Optimizer checkpoint name {name!r} does not match configured {spec.name!r}"
        )
    version = checkpoint.get("state_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("optimizer checkpoint state_version must be an integer")
    state = checkpoint.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("optimizer checkpoint state must be a mapping")
    optimizer = build_optimizer(spec, parameter)
    if version != optimizer.state_version:
        raise ValueError(
            f"Unsupported {spec.name!r} optimizer state_version {version}; "
            f"expected {optimizer.state_version}"
        )
    optimizer.restore_checkpoint_state(state, version=version)
    return optimizer
