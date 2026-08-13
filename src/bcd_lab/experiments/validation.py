"""Stable invariants shared by every experiment component."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np

from bcd_lab.config import coerce_float_dtype

from .schema import ExperimentSpec


def _integer(value: Any, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    resolved = int(value)
    if resolved < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{field} must be {qualifier}")
    return resolved


def _non_negative_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite non-negative number")
    resolved = float(value)
    if not np.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return resolved


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _validate_shape(data: Mapping[str, Any], *, source: str) -> None:
    shape = _mapping(data.get("shape"), field="data.shape")
    rank = _integer(shape.get("rank"), field="data.shape.rank", minimum=1)
    optional_dimensions = {
        name: (
            _integer(shape[name], field=f"data.shape.{name}", minimum=1) if name in shape else None
        )
        for name in ("n_rows", "n_cols", "n_observations")
    }
    if source == "synthetic":
        for name, value in optional_dimensions.items():
            if value is None:
                raise ValueError(f"data.shape.{name} must be an integer")

    n_rows = optional_dimensions["n_rows"]
    n_cols = optional_dimensions["n_cols"]
    n_observations = optional_dimensions["n_observations"]
    if n_rows is not None and n_cols is not None and rank > min(n_rows, n_cols):
        raise ValueError("data.shape.rank cannot exceed min(n_rows, n_cols)")
    if (
        n_rows is not None
        and n_cols is not None
        and n_observations is not None
        and n_observations > n_rows * n_cols
    ):
        raise ValueError("data.shape.n_observations cannot exceed n_rows * n_cols")


def _validate_runtime(runtime: Mapping[str, Any]) -> None:
    max_iter = _integer(runtime.get("max_iter"), field="runtime.max_iter", minimum=0)
    _non_negative_float(runtime.get("tol"), field="runtime.tol")

    interval = runtime.get("diagnostics_interval")
    if interval is not None:
        _integer(interval, field="runtime.diagnostics_interval", minimum=1)

    dtype_value = runtime.get("dtype", "float64")
    try:
        dtype = coerce_float_dtype(dtype_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"runtime.dtype must be a supported floating dtype: {dtype_value!r}"
        ) from exc
    if dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise ValueError("runtime.dtype must be either 'float32' or 'float64'")

    checkpoint_values = runtime.get("checkpoint_iters", ())
    if isinstance(checkpoint_values, (str, bytes)) or not isinstance(checkpoint_values, Sequence):
        raise ValueError("runtime.checkpoint_iters must be a sequence of integers")
    checkpoints = [
        _integer(value, field="runtime.checkpoint_iters", minimum=0) for value in checkpoint_values
    ]
    if len(checkpoints) != len(set(checkpoints)):
        raise ValueError("runtime.checkpoint_iters must not contain duplicates")
    if any(value > max_iter for value in checkpoints):
        raise ValueError("runtime.checkpoint_iters cannot exceed runtime.max_iter")


def validate_experiment_invariants(spec: ExperimentSpec) -> None:
    """Validate component-independent values on a parsed or expanded case."""
    if not isinstance(spec.experiment, str) or not spec.experiment.strip():
        raise ValueError("experiment must be a non-empty string")
    if not isinstance(spec.algorithm.name, str) or not spec.algorithm.name.strip():
        raise ValueError("algorithm.name must be a non-empty string")
    if not isinstance(spec.objective.name, str) or not spec.objective.name.strip():
        raise ValueError("objective.name must be a non-empty string")
    _integer(spec.trials, field="trials", minimum=1)
    data = _mapping(spec.data, field="data")
    source = data.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("data.source must be a non-empty string")
    link = _mapping(spec.link, field="link")
    link_kind = link.get("kind")
    if not isinstance(link_kind, str) or not link_kind.strip():
        raise ValueError("link.kind must be a non-empty string")
    _mapping(link.get("params", {}), field="link.params")
    runtime = _mapping(spec.runtime, field="runtime")
    _mapping(spec.initialization, field="initialization")
    _mapping(spec.diagnostics, field="diagnostics")
    if spec.selection is not None and not spec.selection.metric.strip():
        raise ValueError("selection.metric must be a non-empty string")
    _mapping(spec.plot, field="plot")
    _validate_shape(data, source=source)
    _validate_runtime(runtime)
