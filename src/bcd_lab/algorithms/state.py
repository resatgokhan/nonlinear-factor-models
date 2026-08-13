"""Solver state assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bcd_lab.config import ModelConfig
from bcd_lab.function_models import CopyableFunction, ScalarFunction
from bcd_lab.math import build_sample_matrix, compute_centers
from bcd_lab.optimizers import ArrayOptimizer, build_optimizer
from bcd_lab.types import SampleSet


@dataclass
class SolverState:
    """Mutable state owned by algorithms."""

    z: np.ndarray
    phi: ScalarFunction | None
    rows: np.ndarray
    cols: np.ndarray
    values: np.ndarray
    z_optimizer: ArrayOptimizer
    centers: np.ndarray | None = None
    cache: dict[str, Any] = field(default_factory=dict)


def init_state(
    sample_set: SampleSet,
    config: ModelConfig,
    *,
    z_init: np.ndarray | None = None,
    phi_init: ScalarFunction | None = None,
    rng: np.random.Generator | None = None,
) -> SolverState:
    """Prepare aligned samples and mutable solver state from supplied initial values."""
    dtype = config.runtime.np_dtype
    shape = config.shape
    sample_matrix, rows, cols = build_sample_matrix(
        shape.n_rows,
        shape.n_cols,
        sample_set,
        dtype=dtype,
    )
    values = np.asarray(sample_matrix.data, dtype=dtype)

    if z_init is not None:
        expected = (shape.n_rows + shape.n_cols, shape.rank)
        z = np.asarray(z_init, dtype=dtype).copy()
        if z.shape != expected:
            raise ValueError(f"z_init must have shape {expected}, got {z.shape}")
    else:
        if rng is None:
            raise ValueError("init_state requires rng when z_init is not provided")
        from bcd_lab.math import svd_z

        z = svd_z(
            sample_set,
            shape.n_rows,
            shape.n_cols,
            shape.rank,
            seed=int(rng.integers(0, 2**32, dtype=np.uint32)),
            dtype=dtype,
        )

    if phi_init is not None:
        # Capability contract requires a detached copy.
        if not isinstance(phi_init, CopyableFunction):
            raise TypeError("phi_init must implement CopyableFunction")
        phi = phi_init.copy()
    else:
        phi = None

    centers = compute_centers(z, rows, cols, shape.n_rows)
    z_optimizer = build_optimizer(config.z_optimizer, z)
    return SolverState(
        z=z,
        phi=phi,
        rows=rows,
        cols=cols,
        values=values,
        z_optimizer=z_optimizer,
        centers=centers,
        cache={"z_optimizer_update_norm": 0.0},
    )
