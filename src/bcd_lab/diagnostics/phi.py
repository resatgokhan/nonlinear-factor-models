"""Link-function diagnostics."""

from __future__ import annotations

from statistics import NormalDist
from typing import cast

import numpy as np

from bcd_lab.function_models import (
    DifferentiableFunction,
    PiecewiseLinearRKHS,
    ScalarFunction,
)
from bcd_lab.function_models.rkhs import GaussianRKHS

from .function_evaluation import evaluation_distance, evaluation_grid
from .registry import DiagnosticContext


def _diff_norm(phi_a: object, phi_b: object, ctx: DiagnosticContext) -> float:
    if not isinstance(phi_a, ScalarFunction) or not isinstance(phi_b, ScalarFunction):
        raise TypeError("phi diagnostics require scalar function capabilities")
    return evaluation_distance(phi_a, phi_b, ctx.dataset)


def _gaussian_projection(
    phi: ScalarFunction,
    truth: ScalarFunction | None,
    ctx: DiagnosticContext,
) -> GaussianRKHS | None:
    if not isinstance(phi, PiecewiseLinearRKHS) or not isinstance(truth, GaussianRKHS):
        return None
    if "phi_gaussian_projection" not in ctx.cache:
        ctx.cache["phi_gaussian_projection"] = truth.project_function_onto_span(phi)
    return ctx.cache["phi_gaussian_projection"]


class PhiDiagnostics:
    """Diagnostics comparing learned and reference link functions."""

    name = "phi"
    default_metrics = {"phi_diff"}

    def compute(
        self,
        ctx: DiagnosticContext,
        *,
        metrics: set[str] | None = None,
    ) -> dict[str, float | None]:
        requested = metrics or self.default_metrics
        phi = ctx.snapshot.capability("link_function")
        if phi is None or ctx.dataset is None:
            return {name: None for name in requested}

        references = ctx.dataset.capabilities.functions
        phi_truth = references.truth
        phi_initial = references.initial
        out: dict[str, float | None] = {}

        if "phi_diff" in requested:
            out["phi_diff"] = _diff_norm(phi, phi_truth, ctx) if phi_truth is not None else None
        if "phi_diff_0" in requested:
            out["phi_diff_0"] = (
                _diff_norm(phi, phi_initial, ctx) if phi_initial is not None else None
            )
        if "phi_sample_mse" in requested:
            if phi_truth is None:
                out["phi_sample_mse"] = None
            else:
                samples = evaluation_grid(ctx.dataset)
                if samples is None:
                    probabilities = (
                        np.arange(references.sample_count, dtype=float) + 0.5
                    ) / references.sample_count
                    normal = NormalDist()
                    samples = np.asarray(
                        [normal.inv_cdf(float(probability)) for probability in probabilities]
                    )
                diff = phi.eval_array(samples) - phi_truth.eval_array(samples)
                out["phi_sample_mse"] = float(np.mean(diff**2))
        projection_metrics = {
            "phi_gaussian_rkhs_error",
            "phi_gaussian_rkhs_norm",
        } & requested
        if projection_metrics:
            projection = _gaussian_projection(phi, phi_truth, ctx)
            if "phi_gaussian_rkhs_error" in projection_metrics:
                out["phi_gaussian_rkhs_error"] = (
                    projection.diff_norm(phi_truth, ridge=0.0) if projection is not None else None
                )
            if "phi_gaussian_rkhs_norm" in projection_metrics:
                out["phi_gaussian_rkhs_norm"] = (
                    projection.rkhs_norm(ridge=0.0) if projection is not None else None
                )
        if "phi_d_in_star_bounds" in requested:
            if phi_truth is None or not isinstance(phi_truth, DifferentiableFunction):
                out["phi_d_in_star_bounds"] = None
            else:
                centers = ctx.snapshot.capability("training_centers")
                if centers is None:
                    out["phi_d_in_star_bounds"] = None
                    return out
                if not isinstance(phi, DifferentiableFunction):
                    out["phi_d_in_star_bounds"] = None
                    return out
                truth_d = cast(DifferentiableFunction, phi_truth).d_array(centers)
                phi_d = cast(DifferentiableFunction, phi).d_array(centers)
                out["phi_d_in_star_bounds"] = float(
                    np.mean((phi_d >= float(np.min(truth_d))) & (phi_d <= float(np.max(truth_d))))
                )
        return out
