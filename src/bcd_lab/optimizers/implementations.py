"""Built-in fixed-coordinate array optimizers."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, Mapping

import numpy as np


def _validated_parameter(parameter: np.ndarray) -> np.ndarray:
    array = np.asarray(parameter)
    if array.dtype.kind != "f":
        raise TypeError(f"optimizer parameter must use a floating dtype, got {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError("optimizer parameter must contain only finite values")
    return array


def _positive_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _unit_interval_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1)")
    return result


def _validate_params(
    params: Mapping[str, Any],
    *,
    allowed: set[str],
    optimizer: str,
) -> None:
    unknown = sorted(set(params) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"{optimizer} optimizer received unknown parameters: {joined}")


class GradientDescent:
    """Stateless gradient descent for a fixed-coordinate array."""

    name = "gradient_descent"
    state_version = 1

    def __init__(self, *, learning_rate: float, parameter: np.ndarray) -> None:
        array = _validated_parameter(parameter)
        self.learning_rate = _positive_float(learning_rate, field="learning_rate")
        self._shape = array.shape
        self._dtype = array.dtype

    def compute_update(self, gradient: np.ndarray) -> np.ndarray:
        grad = self._validate_gradient(gradient)
        update = np.asarray(self.learning_rate * grad, dtype=self._dtype)
        if not np.all(np.isfinite(update)):
            raise ValueError("gradient_descent produced a non-finite update")
        return update

    def checkpoint_state(self) -> Mapping[str, Any]:
        return {}

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        version: int,
    ) -> None:
        if version != self.state_version or state:
            raise ValueError("invalid gradient_descent optimizer checkpoint state")

    def _validate_gradient(self, gradient: np.ndarray) -> np.ndarray:
        grad = np.asarray(gradient, dtype=self._dtype)
        if grad.shape != self._shape:
            raise ValueError(f"optimizer gradient must have shape {self._shape}, got {grad.shape}")
        if not np.all(np.isfinite(grad)):
            raise ValueError("optimizer gradient must contain only finite values")
        return grad


class Adam:
    """Adam with bias correction for one fixed-coordinate array."""

    name = "adam"
    state_version = 1

    def __init__(
        self,
        *,
        learning_rate: float,
        beta1: float,
        beta2: float,
        epsilon: float,
        parameter: np.ndarray,
    ) -> None:
        array = _validated_parameter(parameter)
        self.learning_rate = _positive_float(learning_rate, field="learning_rate")
        self.beta1 = _unit_interval_float(beta1, field="beta1")
        self.beta2 = _unit_interval_float(beta2, field="beta2")
        self.epsilon = _positive_float(epsilon, field="epsilon")
        self._shape = array.shape
        self._dtype = array.dtype
        self._first_moment = np.zeros_like(array)
        self._second_moment = np.zeros_like(array)
        self._step = 0

    def compute_update(self, gradient: np.ndarray) -> np.ndarray:
        grad = self._validate_gradient(gradient)
        self._step += 1
        self._first_moment *= self.beta1
        self._first_moment += (1.0 - self.beta1) * grad
        self._second_moment *= self.beta2
        self._second_moment += (1.0 - self.beta2) * np.square(grad)

        first_hat = self._first_moment / (1.0 - self.beta1**self._step)
        second_hat = self._second_moment / (1.0 - self.beta2**self._step)
        update = self.learning_rate * first_hat / (np.sqrt(second_hat) + self.epsilon)
        update = np.asarray(update, dtype=self._dtype)
        if not np.all(np.isfinite(update)):
            raise ValueError("adam produced a non-finite update")
        return update

    def checkpoint_state(self) -> Mapping[str, Any]:
        return {
            "first_moment": self._first_moment.copy(),
            "second_moment": self._second_moment.copy(),
            "step": self._step,
        }

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        version: int,
    ) -> None:
        if version != self.state_version:
            raise ValueError(f"unsupported adam optimizer state_version {version}")
        if set(state) != {"first_moment", "second_moment", "step"}:
            raise ValueError("adam optimizer checkpoint has invalid fields")
        first = self._validate_moment(state["first_moment"], "first_moment")
        second = self._validate_moment(state["second_moment"], "second_moment")
        if np.any(second < 0.0):
            raise ValueError("adam second_moment must be non-negative")
        step = state["step"]
        if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
            raise ValueError("adam step must be a non-negative integer")
        self._first_moment = first.copy()
        self._second_moment = second.copy()
        self._step = int(step)

    def _validate_gradient(self, gradient: np.ndarray) -> np.ndarray:
        grad = np.asarray(gradient, dtype=self._dtype)
        if grad.shape != self._shape:
            raise ValueError(f"optimizer gradient must have shape {self._shape}, got {grad.shape}")
        if not np.all(np.isfinite(grad)):
            raise ValueError("optimizer gradient must contain only finite values")
        return grad

    def _validate_moment(self, value: object, field: str) -> np.ndarray:
        array = np.asarray(value)
        if array.shape != self._shape:
            raise ValueError(f"adam {field} must have shape {self._shape}, got {array.shape}")
        if array.dtype != self._dtype:
            raise ValueError(f"adam {field} must have dtype {self._dtype}, got {array.dtype}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"adam {field} must contain only finite values")
        return array


def build_gradient_descent(
    params: Mapping[str, Any],
    parameter: np.ndarray,
) -> GradientDescent:
    _validate_params(params, allowed={"learning_rate"}, optimizer="gradient_descent")
    if "learning_rate" not in params:
        raise ValueError("gradient_descent optimizer requires learning_rate")
    return GradientDescent(
        learning_rate=params["learning_rate"],
        parameter=parameter,
    )


def build_adam(params: Mapping[str, Any], parameter: np.ndarray) -> Adam:
    _validate_params(
        params,
        allowed={"learning_rate", "beta1", "beta2", "epsilon"},
        optimizer="adam",
    )
    if "learning_rate" not in params:
        raise ValueError("adam optimizer requires learning_rate")
    return Adam(
        learning_rate=params["learning_rate"],
        beta1=params.get("beta1", 0.9),
        beta2=params.get("beta2", 0.999),
        epsilon=params.get("epsilon", 1e-8),
        parameter=parameter,
    )
