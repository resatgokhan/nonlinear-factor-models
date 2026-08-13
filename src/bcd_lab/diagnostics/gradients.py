"""Gradient diagnostics published by an algorithm snapshot."""

from __future__ import annotations

import numpy as np

from .registry import DiagnosticContext


class GradientDiagnostics:
    """Report gradient norms for the current solver state."""

    name = "gradients"
    default_metrics = {"z_grad_norm", "phi_grad_norm", "z_optimizer_update_norm"}

    def compute(
        self,
        ctx: DiagnosticContext,
        *,
        metrics: set[str] | None = None,
    ) -> dict[str, float | None]:
        requested = metrics or self.default_metrics
        out: dict[str, float | None] = {}

        if "z_grad_norm" in requested:
            out["z_grad_norm"] = _published_norm(ctx, "z_gradient", "z_grad_norm")

        if "phi_grad_norm" in requested:
            out["phi_grad_norm"] = _published_norm(ctx, "phi_gradient", "phi_grad_norm")

        if "z_optimizer_update_norm" in requested:
            update_norm = ctx.snapshot.capability("z_optimizer_update_norm")
            out["z_optimizer_update_norm"] = None if update_norm is None else float(update_norm)

        return out


def _published_norm(ctx: DiagnosticContext, array_key: str, norm_key: str) -> float | None:
    norm = ctx.snapshot.capability(norm_key)
    if norm is not None:
        return float(norm)
    array = ctx.snapshot.capability(array_key)
    if array is None:
        return None
    return float(np.linalg.norm(array))
