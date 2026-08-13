"""Shared dataset-domain policies for scalar-function diagnostics."""

from __future__ import annotations

import numpy as np

from bcd_lab.function_models import ScalarFunction
from bcd_lab.function_models.distances import function_distance
from bcd_lab.types import DatasetBundle


def evaluation_grid(dataset: DatasetBundle | None) -> np.ndarray | None:
    """Return the dataset's explicit function-evaluation grid when available."""
    if dataset is None or dataset.capabilities.function_domain is None:
        return None
    return dataset.capabilities.function_domain.grid


def evaluation_distance(
    phi_a: ScalarFunction,
    phi_b: ScalarFunction,
    dataset: DatasetBundle | None,
) -> float:
    """Measure distance on the dataset domain, otherwise use model-native policy."""
    grid = evaluation_grid(dataset)
    if grid is not None:
        return function_distance(phi_a, phi_b, mode="l2_grid", grid=grid)
    return function_distance(phi_a, phi_b)


def linked_truth_domain_mask(dataset: DatasetBundle) -> np.ndarray | None:
    """Select matrix entries whose latent truth lies in the evaluation domain."""
    grid = evaluation_grid(dataset)
    if grid is None or dataset.dense_truth is None:
        return None
    latent = np.asarray(dataset.dense_truth)
    return (latent >= grid[0]) & (latent <= grid[-1])
