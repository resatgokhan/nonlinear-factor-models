"""Parsing and validation for BCD-owned experiment parameters."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping

import numpy as np

from bcd_lab.config import PhiUpdateConfig, RegularizationConfig
from bcd_lab.optimizers import OptimizerSpec, build_optimizer, parse_optimizer_spec


@dataclass(frozen=True)
class BcdParameters:
    """Validated BCD parameters independent of loaded dataset shape."""

    z_optimizer: OptimizerSpec
    phi_update: PhiUpdateConfig
    regularization: RegularizationConfig
    project_z: bool
    freeze_phi_updates: bool


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _finite_float(
    value: object,
    *,
    field: str,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{field} must be finite and at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return result


def parse_bcd_parameters(params: Mapping[str, Any]) -> BcdParameters:
    """Validate canonical BCD parameters without compatibility aliases."""
    allowed = {
        "z_optimizer",
        "phi_update",
        "regularization",
        "project_z",
        "freeze_phi_updates",
    }
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError("bcd params contain unknown fields: " + ", ".join(unknown))

    z_optimizer = parse_optimizer_spec(
        params.get("z_optimizer"),
        field="algorithm.params.z_optimizer",
    )
    # Fixed-coordinate optimizers must accept any floating array shape.
    build_optimizer(z_optimizer, np.zeros(1, dtype=np.float64))

    phi_update_data = _mapping(
        params.get("phi_update"),
        field="algorithm.params.phi_update",
    )
    unknown_phi = sorted(set(phi_update_data) - {"learning_rate"})
    if unknown_phi:
        raise ValueError(
            "algorithm.params.phi_update contains unknown fields: " + ", ".join(unknown_phi)
        )
    if "learning_rate" not in phi_update_data:
        raise ValueError("algorithm.params.phi_update requires learning_rate")
    phi_learning_rate = _finite_float(
        phi_update_data["learning_rate"],
        field="algorithm.params.phi_update.learning_rate",
    )

    regularization_data = _mapping(
        params.get("regularization"),
        field="algorithm.params.regularization",
    )
    unknown_regularization = sorted(set(regularization_data) - {"alpha", "lambda_", "mu"})
    if unknown_regularization:
        raise ValueError(
            "algorithm.params.regularization contains unknown fields: "
            + ", ".join(unknown_regularization)
        )
    lambda_ = _finite_float(
        regularization_data.get("lambda_", 0.01),
        field="algorithm.params.regularization.lambda_",
        maximum=1.0,
    )
    mu = _finite_float(
        regularization_data.get("mu", 1.0),
        field="algorithm.params.regularization.mu",
    )
    alpha = _finite_float(
        regularization_data.get("alpha", 0.01),
        field="algorithm.params.regularization.alpha",
    )

    project_z = params.get("project_z", True)
    freeze_phi_updates = params.get("freeze_phi_updates", False)
    if not isinstance(project_z, bool):
        raise ValueError("algorithm.params.project_z must be a boolean")
    if not isinstance(freeze_phi_updates, bool):
        raise ValueError("algorithm.params.freeze_phi_updates must be a boolean")
    return BcdParameters(
        z_optimizer=z_optimizer,
        phi_update=PhiUpdateConfig(learning_rate=phi_learning_rate),
        regularization=RegularizationConfig(lambda_=lambda_, mu=mu, alpha=alpha),
        project_z=project_z,
        freeze_phi_updates=freeze_phi_updates,
    )
