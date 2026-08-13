"""Timing diagnostics supplied by algorithm loops."""

from __future__ import annotations

from numbers import Real

from .registry import DiagnosticContext


class TimingDiagnostics:
    """Report timing values from the diagnostic context cache."""

    name = "timing"
    default_metrics = {"elapsed", "sec_per_iter", "rem_sec"}

    def compute(
        self,
        ctx: DiagnosticContext,
        *,
        metrics: set[str] | None = None,
    ) -> dict[str, float | None]:
        requested = metrics or self.default_metrics
        values = {
            "elapsed": ctx.snapshot.elapsed_seconds,
            "sec_per_iter": ctx.snapshot.capability("sec_per_iter"),
            "rem_sec": ctx.snapshot.capability("rem_sec"),
        }
        return {name: _maybe_float(values.get(name)) for name in requested}


def _maybe_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, Real):
        raise TypeError(f"timing capability must be numeric, got {type(value).__name__}")
    return float(value)
