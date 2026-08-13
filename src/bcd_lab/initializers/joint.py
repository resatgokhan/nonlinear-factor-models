from typing import Any

import numpy as np

from bcd_lab.config import ModelBuildContext
from bcd_lab.function_models import ScalarFunction
from bcd_lab.types import DatasetBundle

from .phi import build_zero_phi
from .z import svd_z


class SvdWithZeroPhiInitializer:
    """Build SVD ``Z`` and the configured zero function model together."""

    name = "svd_with_zero_phi"

    def build(
        self,
        *,
        dataset: DatasetBundle,
        config: ModelBuildContext,
        params: dict[str, Any],
        seed: int,
    ) -> tuple[np.ndarray, ScalarFunction | None]:
        z = svd_z(dataset.train, config, seed=seed)
        phi = build_zero_phi(config, params, dataset=dataset, z=z)
        return z, phi
