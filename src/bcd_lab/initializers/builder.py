"""Build initialization bundles from simple specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bcd_lab.config import ModelBuildContext
from bcd_lab.function_models import ScalarFunction
from bcd_lab.randomness import RunRandomness
from bcd_lab.types import DatasetBundle

from .registry import get_joint_initializer, get_phi_initializer, get_z_initializer


@dataclass(frozen=True)
class InitializerSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InitializationSpec:
    mode: str
    z: InitializerSpec | None = None
    phi: InitializerSpec | None = None
    joint: InitializerSpec | None = None


def build_initialization(
    spec: InitializationSpec,
    *,
    dataset: DatasetBundle,
    config: ModelBuildContext,
    seed: int,
) -> tuple[np.ndarray, ScalarFunction | None]:
    """Build initialization values (Z, phi) from either separate or joint specs."""
    randomness = RunRandomness(seed)
    if spec.mode == "joint":
        if spec.joint is None:
            raise ValueError("joint initialization mode requires spec.joint")
        return get_joint_initializer(spec.joint.name).build(
            dataset=dataset,
            config=config,
            params=spec.joint.params,
            seed=randomness.child_seed("joint"),
        )

    if spec.mode != "separate":
        raise ValueError("initialization mode must be 'separate' or 'joint'")
    if spec.z is None:
        raise ValueError("separate initialization mode requires spec.z")
    if spec.phi is None:
        raise ValueError("separate initialization mode requires spec.phi")

    z = get_z_initializer(spec.z.name).build(
        dataset=dataset,
        config=config,
        params=spec.z.params,
        seed=randomness.child_seed("z"),
    )
    phi = get_phi_initializer(spec.phi.name).build(
        dataset=dataset,
        config=config,
        params=spec.phi.params,
        seed=randomness.child_seed("phi"),
        z=z,
    )
    return z, phi
