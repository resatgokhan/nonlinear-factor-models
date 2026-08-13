"""Latent-factor truth diagnostics."""

from __future__ import annotations

import numpy as np

from .registry import DiagnosticContext


def procrustes_r_fro(z: np.ndarray, z_star: np.ndarray, *, proper: bool = False) -> np.ndarray:
    """Orthogonal Procrustes minimizer for Frobenius alignment."""
    m = z_star.T @ z
    try:
        u, _, vt = np.linalg.svd(m, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.full_like(m, np.nan)
    r = u @ vt
    if proper and np.linalg.det(r) < 0:
        signs = np.ones(r.shape[0])
        signs[-1] = -1.0
        r = u @ np.diag(signs) @ vt
    return r


def goodness_of_z(z: np.ndarray, z_star: np.ndarray, *, proper: bool = False) -> float:
    """Return ``min_R ||z - z_star R||_F``."""
    r = procrustes_r_fro(z, z_star, proper=proper)
    if not np.all(np.isfinite(r)):
        return float("nan")
    return float(np.linalg.norm(z - z_star @ r, "fro"))


class TruthZDiagnostics:
    """Diagnostics comparing latent factors against known truth."""

    name = "truth_z"
    default_metrics = {"goodness", "relative_goodness"}

    def compute(
        self,
        ctx: DiagnosticContext,
        *,
        metrics: set[str] | None = None,
    ) -> dict[str, float | None]:
        requested = metrics or self.default_metrics
        z_star = None if ctx.dataset is None else ctx.dataset.latent_truth
        z = ctx.snapshot.capability("latent_factors")
        if z_star is None or z is None:
            return {name: None for name in requested}

        z_star_arr = np.asarray(z_star, dtype=float)
        if "procrustes_r" not in ctx.cache:
            ctx.cache["procrustes_r"] = procrustes_r_fro(np.asarray(z), z_star_arr)
        r = ctx.cache["procrustes_r"]
        aligned_residual = np.asarray(z, dtype=float) - z_star_arr @ r
        goodness = float(np.linalg.norm(aligned_residual, "fro"))

        out: dict[str, float | None] = {}
        if "goodness" in requested:
            out["goodness"] = goodness
        if "relative_goodness" in requested:
            z_star_norm = float(np.linalg.norm(z_star_arr, 2))
            out["relative_goodness"] = goodness / z_star_norm if z_star_norm > 0 else None
        if "procrustes_det" in requested:
            out["procrustes_det"] = float(np.linalg.det(r))
        return out
