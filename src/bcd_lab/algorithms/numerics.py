"""Explicit numerical-failure policy for BCD execution."""

from __future__ import annotations

import numpy as np


class BcdNumericalError(RuntimeError):
    """A non-finite BCD quantity observed at a known iteration boundary."""

    def __init__(self, quantity: str, iteration: int) -> None:
        super().__init__(
            f"BCD produced non-finite {quantity} at iteration {iteration}; "
            "adjust the model, initialization, or step sizes"
        )
        self.quantity = quantity
        self.iteration = iteration


def require_finite(value: object, quantity: str, iteration: int) -> None:
    """Raise a contextual error when a scalar or array is not entirely finite."""
    try:
        finite = np.all(np.isfinite(np.asarray(value)))
    except TypeError as exc:
        raise BcdNumericalError(quantity, iteration) from exc
    if not bool(finite):
        raise BcdNumericalError(quantity, iteration)
