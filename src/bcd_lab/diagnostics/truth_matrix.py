"""Dense matrix truth diagnostics."""

from __future__ import annotations

import numpy as np

from .function_evaluation import linked_truth_domain_mask
from .registry import DiagnosticContext


def _inner_matrix(ctx: DiagnosticContext) -> np.ndarray | None:
    if "inner_matrix" not in ctx.cache:
        inner = ctx.snapshot.capability("inner_matrix")
        if inner is None:
            return None
        ctx.cache["inner_matrix"] = np.asarray(inner)
    return ctx.cache["inner_matrix"]


def _relative_fro(pred: np.ndarray, target: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(target, "fro"))
    if denom == 0:
        return None
    return float(np.linalg.norm(pred - target, "fro") / denom)


def _masked_linked_errors(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[float | None, float]:
    if mask is None:
        relative = _relative_fro(pred, target)
        return relative, float(np.sqrt(np.mean((pred - target) ** 2)))
    selected_pred = pred[mask]
    selected_target = target[mask]
    denominator = float(np.linalg.norm(selected_target))
    relative = (
        None
        if denominator == 0.0
        else float(np.linalg.norm(selected_pred - selected_target) / denominator)
    )
    rmse = float(np.sqrt(np.mean((selected_pred - selected_target) ** 2)))
    return relative, rmse


class TruthMatrixDiagnostics:
    """Diagnostics comparing dense model predictions against known matrices."""

    name = "truth_matrix"
    default_metrics = {"x_error"}

    def compute(
        self,
        ctx: DiagnosticContext,
        *,
        metrics: set[str] | None = None,
    ) -> dict[str, float | None]:
        requested = metrics or self.default_metrics
        if ctx.dataset is None:
            return {name: None for name in requested}

        out: dict[str, float | None] = {}
        inner = _inner_matrix(ctx)
        if inner is None:
            return {name: None for name in requested}

        if any(name in requested for name in {"x_error", "relative_x_fro", "x_rmse"}):
            target = ctx.dataset.dense_truth
            if target is None:
                for name in {"x_error", "relative_x_fro", "x_rmse"} & requested:
                    out[name] = None
            else:
                target_arr = np.asarray(target, dtype=float)
                residual = inner - target_arr
                relative = _relative_fro(inner, target_arr)
                if "x_error" in requested:
                    out["x_error"] = relative
                if "relative_x_fro" in requested:
                    out["relative_x_fro"] = relative
                if "x_rmse" in requested:
                    out["x_rmse"] = float(np.sqrt(np.mean(residual**2)))

        if any(name in requested for name in {"phi_x_error", "relative_phi_x_fro", "phi_x_rmse"}):
            linked_truth = ctx.dataset.capabilities.linked_truth
            linked_inner = ctx.snapshot.capability("linked_inner_matrix")
            if linked_truth is None or linked_inner is None:
                for name in {"phi_x_error", "relative_phi_x_fro", "phi_x_rmse"} & requested:
                    out[name] = None
            else:
                target_arr = np.asarray(linked_truth, dtype=float)
                pred = np.asarray(linked_inner, dtype=float)
                domain_mask = linked_truth_domain_mask(ctx.dataset)
                relative, rmse = _masked_linked_errors(pred, target_arr, domain_mask)
                if "phi_x_error" in requested:
                    out["phi_x_error"] = relative
                if "relative_phi_x_fro" in requested:
                    out["relative_phi_x_fro"] = relative
                if "phi_x_rmse" in requested:
                    out["phi_x_rmse"] = rmse
        return out
