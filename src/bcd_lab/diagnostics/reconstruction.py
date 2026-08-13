"""Observed-entry reconstruction diagnostics."""

from __future__ import annotations

import numpy as np

from .registry import DiagnosticContext


def _metrics_for_sample(
    ctx: DiagnosticContext,
    *,
    prefix: str,
    requested: set[str],
) -> dict[str, float | None]:
    split = ctx.snapshot.observed_splits.get(prefix)
    if split is None:
        return {f"{prefix}_{name}": None for name in requested}

    target = np.asarray(split.observed, dtype=float)
    predictions = np.asarray(split.predictions, dtype=float)
    residual = predictions - target
    out: dict[str, float | None] = {}

    mse = None
    if "mse" in requested or "rmse" in requested:
        mse = float(np.mean(residual**2)) if residual.size else None
    if "mse" in requested:
        out[f"{prefix}_mse"] = mse
    if "rmse" in requested:
        out[f"{prefix}_rmse"] = float(np.sqrt(mse)) if mse is not None else None
    if "mae" in requested:
        out[f"{prefix}_mae"] = float(np.mean(np.abs(residual))) if residual.size else None
    if "relative_mse" in requested or "relative_rmse" in requested:
        target_energy = float(np.dot(target, target))
        residual_energy = float(np.dot(residual, residual))
        relative_mse = residual_energy / target_energy if target_energy > 0 else None
        if "relative_mse" in requested:
            out[f"{prefix}_relative_mse"] = relative_mse
        if "relative_rmse" in requested:
            out[f"{prefix}_relative_rmse"] = (
                float(np.sqrt(relative_mse)) if relative_mse is not None else None
            )
    return out


class ReconstructionDiagnostics:
    """Train/validation/test reconstruction metrics with shared predictions."""

    name = "reconstruction"
    default_metrics = {"rmse", "mae", "relative_rmse"}

    def compute(
        self,
        ctx: DiagnosticContext,
        *,
        metrics: set[str] | None = None,
    ) -> dict[str, float | None]:
        requested = metrics or self.default_metrics
        out: dict[str, float | None] = {}
        out.update(_metrics_for_sample(ctx, prefix="train", requested=requested))
        out.update(_metrics_for_sample(ctx, prefix="validation", requested=requested))
        out.update(_metrics_for_sample(ctx, prefix="test", requested=requested))
        return out
