"""Public contracts for data-generating scalar links."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np

from bcd_lab.function_models import ScalarFunction


@dataclass(frozen=True)
class GeneratingLinkSpec:
    """A registered generating-link name and builder-owned parameters."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratingLinkContext:
    """Data-owned inputs available while constructing a generating link."""

    latent_samples: np.ndarray
    dtype: np.dtype
    rng: np.random.Generator


class GeneratingLinkBuilder(Protocol):
    """Validate configuration and construct one truth function per dataset."""

    def validate_params(self, params: Mapping[str, Any]) -> None:
        """Validate builder-owned parameters without generating data."""

    def build(
        self,
        params: Mapping[str, Any],
        context: GeneratingLinkContext,
    ) -> ScalarFunction:
        """Construct a scalar truth function from clean latent samples."""
