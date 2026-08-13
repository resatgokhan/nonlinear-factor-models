"""Binned approximation for piecewise-linear RKHS function models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from .piecewise_linear_rkhs import PiecewiseLinearRKHS, build_piecewise_linear_model

if TYPE_CHECKING:
    from bcd_lab.config import ModelBuildContext
    from bcd_lab.types import DatasetBundle


class BinnedPiecewiseLinearRKHS(PiecewiseLinearRKHS):
    """Piecewise-linear RKHS with grid-binned approximate gradient products."""

    kind = "binned_piecewise_linear_rkhs"

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        version: int,
    ) -> None:
        super().restore_checkpoint_state(state, version=version)
        self._K_grid = None

    def copy(self) -> "BinnedPiecewiseLinearRKHS":
        copied = BinnedPiecewiseLinearRKHS(
            self.sigma,
            (
                np.array(self.scalars, copy=True),
                np.array(self.coefs, copy=True),
            ),
            config=self._config,
        )
        if hasattr(self, "_resolved_input_bounds"):
            copied._resolved_input_bounds = self._resolved_input_bounds
        return copied

    def prepare_grid(self, inputs: np.ndarray) -> None:
        old_grid = None if self.scalars is None else self.scalars.copy()
        super().prepare_grid(inputs)
        if old_grid is None or not np.array_equal(old_grid, self.scalars):
            self._K_grid = None

    def gradient(
        self,
        *,
        inputs: np.ndarray,
        residuals: np.ndarray,
    ) -> np.ndarray:
        """Return a grid-binned approximation to the Gaussian-kernel update."""
        inputs_arr = np.asarray(inputs).ravel()

        self.prepare_grid(inputs_arr)

        x = np.asarray(inputs, dtype=self.scalars.dtype).ravel()
        residual_arr = np.asarray(residuals, dtype=self.coefs.dtype).ravel()
        if x.size != residual_arr.size:
            raise ValueError("inputs and residuals must have the same length")

        if getattr(self, "_K_grid", None) is None:
            scalars = self.scalars
            k_grid = np.subtract(scalars[:, None], scalars[None, :])
            np.square(k_grid, out=k_grid)
            np.divide(k_grid, -2.0 * (self.sigma**2), out=k_grid)
            np.exp(k_grid, out=k_grid)
            self._K_grid = k_grid
            self._grid_dx = float(scalars[1] - scalars[0])

        dx = self._grid_dx
        idx = np.searchsorted(self.scalars, x, side="right") - 1
        idx = np.clip(idx, 0, len(self.scalars) - 2)

        x_left = self.scalars[idx]
        w_right = (x - x_left) / dx
        w_left = 1.0 - w_right

        weights = np.zeros(len(self.scalars), dtype=self.coefs.dtype)
        np.add.at(weights, idx, residual_arr * w_left)
        np.add.at(weights, idx + 1, residual_arr * w_right)

        return self._K_grid @ weights


def build_binned_piecewise_linear_rkhs(
    config: ModelBuildContext,
    params: dict[str, Any] | None = None,
    dataset: DatasetBundle | None = None,
    z: np.ndarray | None = None,
) -> BinnedPiecewiseLinearRKHS:
    return build_piecewise_linear_model(BinnedPiecewiseLinearRKHS, config, params, dataset, z)
