"""Objective/loss diagnostics."""

from __future__ import annotations

from .registry import DiagnosticContext


class ObjectiveDiagnostics:
    """Compute requested objective values through the objective registry."""

    name = "objective"
    default_metrics = {"loss"}

    def compute(
        self,
        ctx: DiagnosticContext,
        *,
        metrics: set[str] | None = None,
    ) -> dict[str, float | None]:
        requested = metrics or self.default_metrics
        out: dict[str, float | None] = {}

        for metric in requested:
            objective_name = ctx.snapshot.objective_name
            if objective_name is None or ctx.snapshot.objective_value is None:
                raise ValueError(
                    "objective diagnostics require a published objective name and value"
                )
            if metric != "loss" and metric != objective_name:
                raise ValueError(
                    f"Metric {metric!r} does not match the selected objective {objective_name!r}"
                )
            out[metric] = float(ctx.snapshot.objective_value)
        return out
