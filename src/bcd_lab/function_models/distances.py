"""Distance helpers for function models."""

from __future__ import annotations

import numpy as np

from .base import NativeDistanceFunction, ScalarFunction


def l2_grid_distance(
    phi_a: ScalarFunction,
    phi_b: object,
    *,
    grid: np.ndarray | None = None,
    lower: float = -3.0,
    upper: float = 3.0,
    count: int = 1000,
) -> float:
    """Approximate L2 distance between two function models on a grid."""
    if not isinstance(phi_b, ScalarFunction):
        raise TypeError("function distance requires two ScalarFunction values")
    if grid is None:
        grid = np.linspace(lower, upper, count, dtype=float)
    vals_a = np.asarray(phi_a.eval_array(grid), dtype=float)
    vals_b = np.asarray(phi_b.eval_array(grid), dtype=float)
    dx = float(grid[1] - grid[0])
    return float(np.sqrt(np.sum((vals_a - vals_b) ** 2) * dx))


def function_distance(
    phi_a: ScalarFunction,
    phi_b: object,
    *,
    mode: str = "auto",
    grid: np.ndarray | None = None,
) -> float:
    """Use a model-native distance when exposed, otherwise use the grid distance."""
    if mode == "auto" and isinstance(phi_a, NativeDistanceFunction):
        return float(phi_a.distance_to(phi_b, mode=mode, grid=grid))
    if mode == "rkhs":
        if not isinstance(phi_a, NativeDistanceFunction):
            raise ValueError("RKHS distance requires a native distance capability")
        return float(phi_a.distance_to(phi_b, mode=mode, grid=grid))
    if mode in {"auto", "l2_grid"}:
        return l2_grid_distance(phi_a, phi_b, grid=grid)
    raise ValueError(f"Unknown function distance mode {mode!r}")
