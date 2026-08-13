"""Diagnostics comparing current phi to the optimal phi w.r.t. current Z."""

from __future__ import annotations

from typing import Any

import numpy as np

from bcd_lab.function_models import ScalarFunction
from bcd_lab.function_models.rkhs import GaussianRKHS, gaussian_kernel_matrix

from .function_evaluation import evaluation_distance
from .registry import DiagnosticContext


def _speedphi_design_stats(
    centers: np.ndarray,
    samples: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate A^T A and A^T y for linear interpolation on a fixed grid."""
    n = grid.size
    dx = float(grid[1] - grid[0])
    ata = np.zeros((n, n), dtype=np.float64)
    aty = np.zeros(n, dtype=np.float64)

    idx = np.asarray(
        np.searchsorted(grid, centers, side="right") - 1,
        dtype=np.intp,
    )
    left_mask = idx < 0
    right_mask = idx >= n - 1
    mid_mask = ~(left_mask | right_mask)

    if np.any(left_mask):
        count = int(np.sum(left_mask))
        ata[0, 0] += count
        aty[0] += float(np.sum(samples[left_mask]))

    if np.any(right_mask):
        count = int(np.sum(right_mask))
        ata[-1, -1] += count
        aty[-1] += float(np.sum(samples[right_mask]))

    if np.any(mid_mask):
        i = idx[mid_mask]
        x = centers[mid_mask]
        y = samples[mid_mask]
        t = (x - grid[i]) / dx
        w0 = 1.0 - t
        w1 = t
        np.add.at(ata, (i, i), w0 * w0)
        np.add.at(ata, (i + 1, i + 1), w1 * w1)
        np.add.at(ata, (i, i + 1), w0 * w1)
        np.add.at(ata, (i + 1, i), w0 * w1)
        np.add.at(aty, i, w0 * y)
        np.add.at(aty, i + 1, w1 * y)

    return ata, aty


def _speedphi_regularizer_matrix(grid: np.ndarray) -> np.ndarray:
    """Return the SpeedPhi quadratic penalty matrix for ||phi||^2."""
    n = grid.size
    dx = float(grid[1] - grid[0])
    diag = np.full(n, dx + (2.0 / dx), dtype=np.float64)
    diag[0] = dx + (1.0 / dx)
    diag[-1] = dx + (1.0 / dx)
    off = np.full(n - 1, -1.0 / dx, dtype=np.float64)
    reg = np.diag(diag)
    reg += np.diag(off, k=1)
    reg += np.diag(off, k=-1)
    return reg


def _compute_phi_sharp_speed(
    ctx: DiagnosticContext,
    *,
    squared_error_scale: float,
    ridge: float = 1e-10,
) -> Any | None:
    """Compute the optimal PiecewiseLinearRKHS (phi^sharp) for current Z."""
    alpha = float(ctx.snapshot.capability("link_regularization", 0.0))
    if alpha <= 0:
        return None

    # Resolve centers
    centers = ctx.snapshot.capability("training_centers")
    link = ctx.snapshot.capability("link_function")
    observed = ctx.snapshot.capability("training_observed")
    if centers is None or link is None or observed is None:
        return None

    grid = np.asarray(link.scalars, dtype=np.float64)
    centers_arr = np.asarray(centers, dtype=np.float64).ravel()
    y = np.asarray(observed, dtype=np.float64).ravel()
    if centers_arr.size == 0:
        return None

    ata, aty = _speedphi_design_stats(centers_arr, y, grid)
    reg = _speedphi_regularizer_matrix(grid)
    regularizer = alpha / (2.0 * squared_error_scale)
    lhs = ata + regularizer * reg + ridge * np.eye(grid.size, dtype=np.float64)

    try:
        values = np.linalg.solve(lhs, aty)
    except np.linalg.LinAlgError:
        values, *_ = np.linalg.lstsq(lhs, aty, rcond=None)
        values = np.asarray(values, dtype=np.float64)

    values = np.asarray(values, dtype=link.coefs.dtype)

    from bcd_lab.function_models.piecewise_linear_rkhs import PiecewiseLinearRKHS

    return PiecewiseLinearRKHS(
        sigma=link.sigma,
        grid_values=(link.scalars.copy(), values),
        config=link.config,
    )


def compute_phi_sharp(
    ctx: DiagnosticContext,
    *,
    ridge: float = 1e-10,
) -> Any | None:
    """Compute phi^sharp(Z) = argmin_phi L(phi, Z) for the current Z."""
    alpha = float(ctx.snapshot.capability("link_regularization", 0.0))
    link = ctx.snapshot.capability("link_function")
    factors = ctx.snapshot.capability("latent_factors")
    if alpha <= 0 or link is None or factors is None:
        return None
    squared_error_scale = ctx.snapshot.capability("squared_error_scale")
    if squared_error_scale is None or squared_error_scale <= 0:
        return None

    from bcd_lab.function_models.piecewise_linear_rkhs import PiecewiseLinearRKHS

    if isinstance(link, PiecewiseLinearRKHS):
        return _compute_phi_sharp_speed(
            ctx,
            squared_error_scale=squared_error_scale,
            ridge=ridge,
        )

    # Fallback to GaussianRKHS solver
    # Resolve centers
    centers = ctx.snapshot.capability("training_centers")
    observed = ctx.snapshot.capability("training_observed")
    if centers is None or observed is None:
        return None

    solve_dtype = np.float64
    y = np.asarray(observed, dtype=solve_dtype).ravel()
    c = np.asarray(centers, dtype=solve_dtype).ravel()

    sigma = float(getattr(link, "sigma", 1.0))
    K = gaussian_kernel_matrix(c, c, sigma).astype(solve_dtype, copy=False)
    regularizer = float(alpha) / (2.0 * squared_error_scale)
    K_reg = K + (regularizer + ridge) * np.eye(K.shape[0], dtype=solve_dtype)

    try:
        beta = np.linalg.solve(K_reg, y)
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(K_reg, y, rcond=None)
        beta = np.asarray(beta, dtype=solve_dtype)

    factor_dtype = np.asarray(factors).dtype
    beta = np.asarray(beta, dtype=factor_dtype)
    c = np.asarray(c, dtype=factor_dtype)

    return GaussianRKHS(sigma, scalars=c, coefs=beta)


def _get_goodness(ctx: DiagnosticContext) -> float | None:
    if "goodness" in ctx.cache:
        return ctx.cache["goodness"]

    z_star = None if ctx.dataset is None else ctx.dataset.latent_truth
    z = ctx.snapshot.capability("latent_factors")
    if z_star is None or z is None:
        return None

    from .truth_z import procrustes_r_fro

    z_star_arr = np.asarray(z_star, dtype=float)
    if "procrustes_r" not in ctx.cache:
        ctx.cache["procrustes_r"] = procrustes_r_fro(np.asarray(z), z_star_arr)
    r = ctx.cache["procrustes_r"]
    aligned_residual = np.asarray(z, dtype=float) - z_star_arr @ r
    goodness = float(np.linalg.norm(aligned_residual, "fro"))
    ctx.cache["goodness"] = goodness
    return goodness


def _phi_sharp_diff(
    phi_sharp: ScalarFunction,
    state_phi: ScalarFunction,
    ctx: DiagnosticContext,
) -> float:
    return evaluation_distance(phi_sharp, state_phi, ctx.dataset)


def _phi_sharp_diff_star(phi_sharp: ScalarFunction, ctx: DiagnosticContext) -> float | None:
    if ctx.dataset is None:
        return None
    phi_truth = ctx.dataset.capabilities.functions.truth
    if phi_truth is None:
        return None
    return evaluation_distance(phi_sharp, phi_truth, ctx.dataset)


def _phi_sharp_regret(
    ctx: DiagnosticContext,
    phi_sharp: Any,
    loss_val: float,
) -> float | None:
    evaluator = ctx.snapshot.capability("link_objective_evaluator")
    if evaluator is None:
        return None
    phi_sharp_loss = float(evaluator(phi_sharp))
    return float(loss_val - phi_sharp_loss)


class PhiSharpDiagnostics:
    """Compute phi_sharp metrics, the Lyapunov value, and optimization regret."""

    name = "phi_sharp"
    default_metrics = {"phi_sharp_diff", "lyapunov", "regret"}

    def compute(
        self,
        ctx: DiagnosticContext,
        *,
        metrics: set[str] | None = None,
    ) -> dict[str, float | None]:
        requested = metrics or self.default_metrics
        out: dict[str, float | None] = {}

        alpha = float(ctx.snapshot.capability("link_regularization", 0.0))
        link = ctx.snapshot.capability("link_function")
        if alpha <= 0 or link is None:
            return {name: None for name in requested}

        # Resolve phi_sharp and cache it
        if "phi_sharp" not in ctx.cache:
            ctx.cache["phi_sharp"] = compute_phi_sharp(ctx)
        phi_sharp = ctx.cache["phi_sharp"]

        if phi_sharp is None:
            return {name: None for name in requested}

        phi_diff = None
        if any(name in requested for name in {"phi_sharp_diff", "lyapunov"}):
            phi_diff = _phi_sharp_diff(phi_sharp, link, ctx)

        if "phi_sharp_diff" in requested:
            out["phi_sharp_diff"] = phi_diff

        if "phi_sharp_diff_star" in requested:
            out["phi_sharp_diff_star"] = _phi_sharp_diff_star(phi_sharp, ctx)

        if "lyapunov" in requested:
            goodness = _get_goodness(ctx)
            if goodness is not None and phi_diff is not None:
                out["lyapunov"] = float(goodness + phi_diff / alpha)
            else:
                out["lyapunov"] = None

        if "regret" in requested:
            if ctx.snapshot.objective_value is None:
                out["regret"] = None
                return out
            loss_val = float(ctx.snapshot.objective_value)
            out["regret"] = _phi_sharp_regret(ctx, phi_sharp, loss_val)

        return out
