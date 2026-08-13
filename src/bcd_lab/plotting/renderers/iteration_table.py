"""Per-iteration metrics table renderer."""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..data import _metric_series_for_record
from ..plot_config import (
    legend_entry_sort_key,
    legend_fallback_label,
    legend_label,
    legend_value_for_record,
)
from .table import _format_table_value


def _metric_value_at_iteration(record: dict, metric: str, iteration: int) -> float | None:
    series = _metric_series_for_record(record, metric)
    if series is None:
        return None
    iters, vals = series
    if iters.size == 0:
        return None
    idx = int(np.searchsorted(iters, iteration, side="right") - 1)
    if idx < 0:
        return None
    value = float(vals[idx])
    if not np.isfinite(value):
        return None
    return value


def _iteration_targets(
    records: Sequence[dict],
    metric: str,
    *,
    iteration_mode: str | None,
    iteration_count: int | None,
    iteration_max: int | None,
    iterations: Sequence[int] | None,
) -> list[int]:
    if iterations:
        return sorted({int(val) for val in iterations if int(val) > 0})

    count = int(iteration_count or 10)
    max_iter = iteration_max or 0
    if max_iter <= 0:
        for record in records:
            series = _metric_series_for_record(record, metric)
            if series is None:
                continue
            iters, _ = series
            if iters.size:
                max_iter = max(max_iter, int(np.max(iters)))
    if max_iter <= 0:
        return []

    mode = iteration_mode or "log"
    if mode == "linear":
        raw = np.linspace(1, max_iter, num=count)
    else:
        raw = np.logspace(0, np.log10(max_iter), num=count)
    targets = sorted({int(round(val)) for val in raw if val >= 1})
    if max_iter not in targets:
        targets.append(max_iter)
    return sorted({min(max_iter, val) for val in targets})


def _plot_iteration_table(
    ax: plt.Axes,
    records: Sequence[dict],
    *,
    metric: str,
    legend_key: str | None,
    title: str | None,
    iteration_mode: str | None,
    iteration_count: int | None,
    iteration_max: int | None,
    iterations: Sequence[int] | None,
) -> None:
    targets = _iteration_targets(
        records,
        metric,
        iteration_mode=iteration_mode,
        iteration_count=iteration_count,
        iteration_max=iteration_max,
        iterations=iterations,
    )
    if not targets:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        return

    grouped: dict[object, list[dict]] = {}
    if legend_key:
        for record in records:
            key = legend_value_for_record(record, legend_key)
            grouped.setdefault(key, []).append(record)
    else:
        grouped[None] = list(records)

    if not grouped:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        return

    entries: list[tuple[str, list[dict]]] = []
    for idx, (key, recs) in enumerate(grouped.items()):
        fallback = legend_fallback_label(recs[0], index=idx) if recs else f"run-{idx + 1}"
        label = legend_label(key, fallback)
        entries.append((label, recs))
    entries.sort(key=lambda item: legend_entry_sort_key(item[0]))

    row_labels = [str(val) for val in targets]
    col_labels = [label for label, _ in entries]
    cell_text: list[list[str]] = []
    for iteration in targets:
        row_vals: list[str] = []
        for _, recs in entries:
            values: list[float] = []
            for record in recs:
                val = _metric_value_at_iteration(record, metric, iteration)
                if val is not None:
                    values.append(val)
            mean_val = float(np.mean(values)) if values else None
            row_vals.append(_format_table_value(mean_val, is_iteration=False))
        cell_text.append(row_vals)

    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.scale(1.0, 1.2)
    ax.set_title(title or metric)


def _iteration_table_handler(ax: plt.Axes, instance: dict, context: dict) -> None:
    plot_entry = instance["plot_entry"]
    full_spec = instance["full_spec"]
    plot_key = plot_entry.get("key")
    metric = plot_entry.get("y")
    if not metric:
        raise ValueError(f"Iteration table plot '{plot_key}' requires y metric.")

    def _spec_opt(name: str, default: Any = None) -> Any:
        val = plot_entry.get(name)
        return val if val is not None else full_spec.get(name, default)

    _plot_iteration_table(
        ax,
        instance["records"],
        metric=metric,
        legend_key=context["legend_key"],
        title=plot_entry.get("title"),
        iteration_mode=_spec_opt("iteration_mode"),
        iteration_count=_spec_opt("iteration_count"),
        iteration_max=_spec_opt("iteration_max"),
        iterations=_spec_opt("iterations"),
    )
