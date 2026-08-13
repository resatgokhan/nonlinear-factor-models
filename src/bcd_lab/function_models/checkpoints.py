"""Versioned checkpoint envelopes for registered function models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from bcd_lab.config import ModelBuildContext

from .base import CheckpointableFunction, ScalarFunction
from .registry import get_function_model_builder

if TYPE_CHECKING:
    from bcd_lab.types import DatasetBundle

FUNCTION_CHECKPOINT_FORMAT_VERSION = 1


def serialize_function_model(model: ScalarFunction) -> dict[str, Any]:
    """Ask a function model for its own versioned mutable state."""
    if not isinstance(model, CheckpointableFunction):
        raise TypeError(
            f"Function model {type(model).__name__} does not implement CheckpointableFunction"
        )
    if model.checkpoint_version <= 0:
        raise ValueError("function checkpoint_version must be positive")
    return {
        "format_version": FUNCTION_CHECKPOINT_FORMAT_VERSION,
        "kind": model.kind,
        "state_version": model.checkpoint_version,
        "state": dict(model.checkpoint_state()),
    }


def restore_function_model(
    checkpoint: Mapping[str, Any],
    *,
    config: ModelBuildContext,
    dataset: DatasetBundle | None = None,
    z: np.ndarray | None = None,
) -> ScalarFunction:
    """Restore a registered model through its component-owned state method."""
    format_version = checkpoint.get("format_version")
    if format_version != FUNCTION_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported function checkpoint format_version {format_version!r}")
    kind = checkpoint.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValueError("function checkpoint kind must be a non-empty string")
    configured_kind = getattr(config.link.kind, "value", config.link.kind)
    if str(configured_kind) != kind:
        raise ValueError(
            f"Function checkpoint kind {kind!r} does not match config {configured_kind!r}"
        )
    state_version = checkpoint.get("state_version")
    if isinstance(state_version, bool) or not isinstance(state_version, int):
        raise ValueError("function checkpoint state_version must be an integer")
    state = checkpoint.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("function checkpoint state must be a mapping")

    model = get_function_model_builder(kind)(
        config,
        dataset=dataset,
        z=z,
    )
    if not isinstance(model, CheckpointableFunction):
        raise TypeError(
            f"Registered function model {kind!r} does not implement CheckpointableFunction"
        )
    if state_version != model.checkpoint_version:
        raise ValueError(
            f"Unsupported {kind!r} state_version {state_version}; "
            f"expected {model.checkpoint_version}"
        )
    model.restore_checkpoint_state(state, version=state_version)
    return model
