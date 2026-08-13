"""Helpers for constructing synthetic RKHS functions."""

from __future__ import annotations

import numpy as np

from .gaussian_rkhs import GaussianRKHS, gaussian_kernel_matrix


def make_noisy_phi(
    phi: GaussianRKHS,
    noise_scale: float,
    rng: np.random.Generator,
) -> GaussianRKHS:
    """Return a coefficient-perturbed copy of ``phi``."""
    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative")
    noise = rng.uniform(-noise_scale, noise_scale, size=phi.coefs.shape)
    return GaussianRKHS(
        phi.sigma,
        np.asarray(phi.scalars, dtype=float).copy(),
        np.asarray(phi.coefs, dtype=float) + noise,
    )


def random_monotone_poly(
    degree: int,
    *,
    grid: int = 200,
    xmin: float = -2.0,
    xmax: float = 2.0,
    waviness: float = 1.0,
    min_margin: float = 1.0e-2,
    min_range: float = 0.5,
    max_attempts: int = 10_000,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate coefficients for a random strictly increasing polynomial."""
    if degree < 1:
        raise ValueError("degree must be >= 1")
    if grid < 2:
        raise ValueError("grid must be >= 2")
    if xmin >= xmax:
        raise ValueError("xmin must be smaller than xmax")
    if not np.isfinite(waviness) or waviness <= 0.0:
        raise ValueError("waviness must be finite and positive")
    if not np.isfinite(min_margin) or min_margin <= 0.0:
        raise ValueError("min_margin must be finite and positive")
    if not np.isfinite(min_range) or min_range < 0.0:
        raise ValueError("min_range must be finite and non-negative")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    for _ in range(max_attempts):
        derivative_degree = degree - 1
        base = float(rng.uniform(1.0, 2.0))
        rest = []
        base_scale = 0.3 * waviness
        for k in range(1, derivative_degree + 1):
            rest.append(float(rng.normal(0.0, base_scale * (1 + 0.5 * k))))
        derivative = np.concatenate(([base], np.asarray(rest, dtype=float)))

        roots = np.roots(derivative[::-1])
        if any(abs(root.imag) < 1.0e-8 and xmin <= root.real <= xmax for root in roots):
            continue

        xs = np.linspace(xmin, xmax, grid)
        values = np.polynomial.polynomial.polyval(xs, derivative)
        if float(np.min(values)) <= min_margin:
            continue
        if float(np.max(values) - np.min(values)) < min_range:
            continue

        poly = np.zeros(degree + 1, dtype=float)
        for k in range(1, degree + 1):
            poly[k] = derivative[k - 1] / k
        return poly
    raise ValueError(
        "could not draw a monotone polynomial within max_attempts; "
        "relax min_margin/min_range or increase max_attempts"
    )


def polynomial_fit_phi(
    centers: np.ndarray,
    sigma: float,
    poly_coefs: np.ndarray,
    *,
    ridge: float = 1.0e-6,
) -> GaussianRKHS:
    """Fit an RKHS function to polynomial values at the provided centers."""
    centers_arr = np.asarray(centers, dtype=float).ravel()
    if centers_arr.size == 0:
        raise ValueError("centers must be non-empty")
    if not np.all(np.isfinite(centers_arr)):
        raise ValueError("centers must be finite")
    poly = np.asarray(poly_coefs, dtype=float).ravel()
    if poly.size == 0:
        raise ValueError("poly_coefs must be non-empty")
    if not np.all(np.isfinite(poly)):
        raise ValueError("poly_coefs must be finite")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    target = np.vander(centers_arr, N=poly.size, increasing=True) @ poly
    gram = gaussian_kernel_matrix(centers_arr, centers_arr, sigma)
    if ridge:
        gram = gram + ridge * np.eye(centers_arr.size)
    try:
        coefs = np.linalg.solve(gram, target)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Gaussian RKHS fit is singular; increase ridge") from exc
    if not np.all(np.isfinite(coefs)):
        raise ValueError("Gaussian RKHS fit produced non-finite coefficients")
    return GaussianRKHS(sigma=float(sigma), scalars=centers_arr, coefs=coefs)
