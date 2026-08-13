"""Projection routines for RKHS functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gaussianKernelMath import gaussian_dkernel_dx_matrix, gaussian_kernel_matrix


@dataclass(frozen=True)
class ProjectionResult:
    """Projected Gaussian expansion parameters, independent of a model class."""

    scalars: np.ndarray
    coefs: np.ndarray


def project_gaussian_ball(
    *,
    sigma: float,
    source_scalars: np.ndarray,
    source_coefs: np.ndarray,
    centers: np.ndarray,
    center_scalars: np.ndarray,
    center_coefs: np.ndarray,
    distance: float,
    radius: float,
    ridge: float = 1.0e-8,
) -> ProjectionResult:
    """Project expansion parameters into a Gaussian RKHS ball."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    alpha = 1.0 if distance <= radius or distance == 0.0 else radius / distance

    dtype = np.result_type(
        source_scalars,
        source_coefs,
        center_scalars,
        center_coefs,
        centers,
        float,
    )
    c = np.asarray(centers, dtype=dtype).ravel()
    if c.size == 0:
        raise ValueError("centers must be non-empty")

    c0 = np.asarray(center_scalars, dtype=dtype).ravel()
    a0 = np.asarray(center_coefs, dtype=dtype)
    a = np.asarray(source_coefs, dtype=dtype)

    k_cc0 = gaussian_kernel_matrix(c, c0, sigma)
    k_cx = gaussian_kernel_matrix(c, source_scalars, sigma)
    k_cc = gaussian_kernel_matrix(c, c, sigma)
    target_on_c = (1.0 - alpha) * (k_cc0 @ a0) + alpha * (k_cx @ a)

    if ridge:
        k_cc = k_cc + float(ridge) * np.eye(c.size, dtype=dtype)
    k_cc = 0.5 * (k_cc + k_cc.T)
    coefs, *_ = np.linalg.lstsq(k_cc, target_on_c, rcond=None)
    return ProjectionResult(scalars=c, coefs=coefs)


def project_gaussian_unconstrained(
    *,
    sigma: float,
    source_scalars: np.ndarray,
    source_coefs: np.ndarray,
    centers: np.ndarray,
    ridge: float = 0.0,
    max_rhs_bytes: int = 256 * 1024 * 1024,
) -> ProjectionResult:
    """Project expansion parameters onto kernels at ``centers``."""
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if max_rhs_bytes <= 0:
        raise ValueError("max_rhs_bytes must be positive")

    dtype = np.result_type(source_scalars, source_coefs, centers, np.float32)
    x = np.ascontiguousarray(source_scalars, dtype=dtype).ravel()
    beta = np.ascontiguousarray(source_coefs, dtype=dtype).ravel()
    c = np.ascontiguousarray(centers, dtype=dtype).ravel()

    if x.size != beta.size:
        raise ValueError(
            "source_scalars and source_coefs must have the same length; "
            f"got {x.size} vs {beta.size}"
        )
    if c.size == 0:
        raise ValueError("centers must be non-empty")

    k_cc = gaussian_kernel_matrix(c, c, sigma).astype(dtype, copy=False)
    bytes_per_row = c.size * np.dtype(dtype).itemsize
    chunk_rows = max(1, int(max_rhs_bytes // max(bytes_per_row, 1)))
    rhs = np.zeros_like(c, dtype=dtype)

    for start in range(0, x.size, chunk_rows):
        end = min(x.size, start + chunk_rows)
        k_block = gaussian_kernel_matrix(x[start:end], c, sigma).astype(dtype, copy=False)
        rhs += k_block.T @ beta[start:end]

    if ridge:
        k_cc = k_cc + float(ridge) * np.eye(c.size, dtype=dtype)
    k_cc = 0.5 * (k_cc + k_cc.T)
    alpha, *_ = np.linalg.lstsq(k_cc, rhs, rcond=None)
    return ProjectionResult(scalars=c, coefs=alpha)


def project_gaussian_constrained(
    *,
    sigma: float,
    source_scalars: np.ndarray,
    source_coefs: np.ndarray,
    centers: np.ndarray,
    lower: float = -3.0,
    upper: float = 3.0,
    eval_points: np.ndarray | None = None,
    lam: float = 1.0e-8,
    solver: str = "OSQP",
    mode: str = "low",
    max_iter: int | None = None,
    solver_opts: dict | None = None,
    verbose: bool = False,
) -> ProjectionResult:
    """Project expansion parameters with derivative constraints."""
    import cvxpy as cp

    if not (np.isfinite(lower) and np.isfinite(upper)) or lower > upper:
        raise ValueError(f"Invalid bounds: lower={lower}, upper={upper}")

    lam = float(lam)
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")

    target_dtype = np.result_type(source_scalars, source_coefs, centers)
    c = np.asarray(centers, dtype=target_dtype).ravel()
    if c.size == 0:
        raise ValueError("centers must be non-empty")

    x_source = source_scalars if eval_points is None else eval_points
    x = np.asarray(x_source, dtype=target_dtype).ravel()
    if x.size == 0:
        raise ValueError("eval_points must be non-empty")

    source_kernel = gaussian_kernel_matrix(x, source_scalars, sigma)
    y = source_kernel @ np.asarray(source_coefs, dtype=target_dtype)

    K = gaussian_kernel_matrix(x, c, sigma)
    D = gaussian_dkernel_dx_matrix(x, c, sigma)
    Kcc = gaussian_kernel_matrix(c, c, sigma)
    Kcc = 0.5 * (Kcc + Kcc.T)
    Kcc_psd = cp.psd_wrap(Kcc)

    M = c.size
    alpha = cp.Variable(M)

    obj = 0.5 * cp.sum_squares(K @ alpha - y) + 0.5 * lam * cp.quad_form(alpha, Kcc_psd)
    constraints = [D @ alpha >= lower, D @ alpha <= upper]
    prob = cp.Problem(cp.Minimize(obj), constraints)

    if mode not in ("low", "high"):
        raise ValueError(f"mode must be 'low' or 'high', got {mode}")

    tol = 1e-4 if mode == "low" else 1e-7

    defaults = {
        "OSQP": dict(
            eps_abs=tol,
            eps_rel=tol,
            max_iter=(8000 if mode == "low" else 50000),
            polishing=True,
            warm_start=True,
        ),
        "SCS": dict(
            eps=tol,
            max_iters=(5000 if mode == "low" else 50000),
        ),
        "ECOS": dict(),
    }

    def solve_with(name: str) -> None:
        kw = dict(defaults.get(name, {}))
        if max_iter is not None:
            if name == "OSQP":
                kw["max_iter"] = int(max_iter)
            elif name == "SCS":
                kw["max_iters"] = int(max_iter)
        if solver_opts:
            kw.update(solver_opts)
        prob.solve(solver=name, verbose=verbose, **kw)

    if mode == "low":
        solve_with(solver)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Projection failed (low mode): status={prob.status}")
    else:
        for s in (solver, "ECOS", "SCS"):
            try:
                solve_with(s)
            except cp.SolverError:
                continue
            if prob.status in ("optimal", "optimal_inaccurate"):
                break
        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Projection failed (high mode): status={prob.status}")

    alpha_val = np.asarray(alpha.value, dtype=target_dtype).ravel()
    return ProjectionResult(scalars=c, coefs=alpha_val)
