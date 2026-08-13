"""Static dataset summary diagnostics."""

from __future__ import annotations

from .registry import DiagnosticContext


class DataSummaryDiagnostics:
    """Expose typed dataset split statistics through the diagnostics registry."""

    name = "data_summary"
    default_metrics = {
        "train_rating_count",
        "validation_rating_count",
        "total_selected_rating_count",
        "density",
    }

    def compute(
        self,
        ctx: DiagnosticContext,
        *,
        metrics: set[str] | None = None,
    ) -> dict[str, float | int | str | None]:
        requested = metrics or self.default_metrics
        if ctx.dataset is None:
            return {name: None for name in requested}

        splits = ctx.dataset.capabilities.splits
        train_count = splits.train_count if splits is not None else ctx.dataset.n_observations
        validation_count = (
            splits.validation_count
            if splits is not None
            else (0 if ctx.dataset.validation is None else int(ctx.dataset.validation[0].size))
        )
        test_count = (
            splits.test_count
            if splits is not None
            else (0 if ctx.dataset.test is None else int(ctx.dataset.test[0].size))
        )
        total_count = (
            splits.resolved_total_count
            if splits is not None
            else train_count + validation_count + test_count
        )
        out: dict[str, float | int | str | None] = {}
        for name in requested:
            if name == "density":
                denom = ctx.dataset.n_rows * ctx.dataset.n_cols
                out[name] = float(total_count / denom) if denom > 0 else None
            elif name == "train_rating_count":
                out[name] = train_count
            elif name == "validation_rating_count":
                out[name] = validation_count
            elif name == "test_rating_count":
                out[name] = test_count
            elif name == "total_selected_rating_count":
                out[name] = total_count
            else:
                out[name] = None
        return out
