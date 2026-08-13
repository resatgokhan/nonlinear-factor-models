"""Z initialization strategies."""

from __future__ import annotations

import numpy as np

from bcd_lab.config import ModelBuildContext
from bcd_lab.math import svd_z as core_svd_z
from bcd_lab.types import DatasetBundle, SampleSet


def svd_z(
    sample_set: SampleSet,
    config: ModelBuildContext,
    *,
    seed: int,
) -> np.ndarray:
    """Initialize factor matrix from a randomized SVD of the observed matrix."""
    return core_svd_z(
        sample_set,
        n_rows=config.shape.n_rows,
        n_cols=config.shape.n_cols,
        rank=config.shape.rank,
        seed=seed,
        dtype=config.runtime.np_dtype,
    )


class SvdZInitializer:
    """Registered SVD Z initializer."""

    name = "svd"

    def build(
        self,
        *,
        dataset: DatasetBundle,
        config: ModelBuildContext,
        params: dict[str, object],
        seed: int,
    ) -> np.ndarray:
        _ = params
        return svd_z(dataset.train, config, seed=seed)
