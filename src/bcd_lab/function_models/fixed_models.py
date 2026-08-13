"""Fixed scalar function models.

These models intentionally expose only evaluation, differentiation, copying,
and serialization. They are not trainable or projectable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from bcd_lab.config import ModelBuildContext
    from bcd_lab.types import DatasetBundle


class IdentityFunction:
    """Identity link ``phi(x)=x``."""

    kind = "identity"
    checkpoint_version = 1

    def eval_array(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x)

    def d_array(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(np.asarray(x), dtype=float)

    def eval_and_d_array(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.eval_array(x), self.d_array(x)

    def __call__(self, x: float) -> float:
        return float(x)

    def d(self, x: float) -> float:
        _ = x
        return 1.0

    def copy(self) -> "IdentityFunction":
        return IdentityFunction()

    def to_payload(self) -> dict[str, Any]:
        return {"kind": self.kind}

    def checkpoint_state(self) -> Mapping[str, Any]:
        return {}

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        version: int,
    ) -> None:
        if version != self.checkpoint_version or state:
            raise ValueError("invalid identity function checkpoint state")


class SigmoidFunction:
    """Sigmoid function model."""

    kind = "sigmoid"
    checkpoint_version = 1

    def __init__(self, *, shifted: bool = True) -> None:
        self.shifted = bool(shifted)

    def eval_array(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        vals = 1.0 / (1.0 + np.exp(-x_arr))
        return vals - 0.5 if self.shifted else vals

    def d_array(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        sig = 1.0 / (1.0 + np.exp(-x_arr))
        return sig * (1.0 - sig)

    def eval_and_d_array(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=float)
        sig = 1.0 / (1.0 + np.exp(-x_arr))
        vals = sig - 0.5 if self.shifted else sig
        return vals, sig * (1.0 - sig)

    def __call__(self, x: float) -> float:
        return float(self.eval_array(np.asarray([x]))[0])

    def d(self, x: float) -> float:
        return float(self.d_array(np.asarray([x]))[0])

    def copy(self) -> "SigmoidFunction":
        return SigmoidFunction(shifted=self.shifted)

    def to_payload(self) -> dict[str, Any]:
        return {"kind": self.kind, "shifted": self.shifted}

    def checkpoint_state(self) -> Mapping[str, Any]:
        return {"shifted": self.shifted}

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        version: int,
    ) -> None:
        shifted = state.get("shifted")
        if version != self.checkpoint_version or not isinstance(shifted, bool):
            raise ValueError("invalid sigmoid function checkpoint state")
        self.shifted = shifted


class ExponentialFunction:
    """Exponential function model."""

    kind = "exponential"
    checkpoint_version = 1

    def __init__(self, *, shifted: bool = False) -> None:
        self.shifted = bool(shifted)

    def eval_array(self, x: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            vals = np.exp(np.asarray(x, dtype=float))
        return vals - 1.0 if self.shifted else vals

    def d_array(self, x: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.exp(np.asarray(x, dtype=float))

    def eval_and_d_array(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        derivs = self.d_array(x)
        vals = derivs - 1.0 if self.shifted else derivs
        return vals, derivs

    def __call__(self, x: float) -> float:
        return float(self.eval_array(np.asarray([x]))[0])

    def d(self, x: float) -> float:
        return float(self.d_array(np.asarray([x]))[0])

    def copy(self) -> "ExponentialFunction":
        return ExponentialFunction(shifted=self.shifted)

    def to_payload(self) -> dict[str, Any]:
        return {"kind": self.kind, "shifted": self.shifted}

    def checkpoint_state(self) -> Mapping[str, Any]:
        return {"shifted": self.shifted}

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        version: int,
    ) -> None:
        shifted = state.get("shifted")
        if version != self.checkpoint_version or not isinstance(shifted, bool):
            raise ValueError("invalid exponential function checkpoint state")
        self.shifted = shifted


class TanhFunction:
    """Hyperbolic-tangent function ``phi(x)=tanh(x)``."""

    kind = "tanh"
    checkpoint_version = 1

    def eval_array(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(np.asarray(x, dtype=float))

    def d_array(self, x: np.ndarray) -> np.ndarray:
        values = self.eval_array(x)
        return 1.0 - values**2

    def eval_and_d_array(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = self.eval_array(x)
        return values, 1.0 - values**2

    def __call__(self, x: float) -> float:
        return float(np.tanh(x))

    def d(self, x: float) -> float:
        value = float(np.tanh(x))
        return 1.0 - value**2

    def copy(self) -> "TanhFunction":
        return TanhFunction()

    def to_payload(self) -> dict[str, Any]:
        return {"kind": self.kind}

    def checkpoint_state(self) -> Mapping[str, Any]:
        return {}

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        version: int,
    ) -> None:
        if version != self.checkpoint_version or state:
            raise ValueError("invalid tanh function checkpoint state")


def build_identity(
    config: ModelBuildContext,
    params: dict[str, Any] | None = None,
    dataset: DatasetBundle | None = None,
    z: np.ndarray | None = None,
) -> IdentityFunction:
    return IdentityFunction()


def build_sigmoid(
    config: ModelBuildContext,
    params: dict[str, Any] | None = None,
    dataset: DatasetBundle | None = None,
    z: np.ndarray | None = None,
) -> SigmoidFunction:
    return SigmoidFunction(shifted=config.link.params.get("shifted", True))


def build_exponential(
    config: ModelBuildContext,
    params: dict[str, Any] | None = None,
    dataset: DatasetBundle | None = None,
    z: np.ndarray | None = None,
) -> ExponentialFunction:
    return ExponentialFunction(shifted=config.link.params.get("shifted", False))
