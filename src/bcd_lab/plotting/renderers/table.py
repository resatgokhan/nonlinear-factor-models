"""Sweep-summary table renderer."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..helpers import _summary_metric_value
from ..plot_config import axis_value_for_record


def _format_table_value(val: float | None, *, is_iteration: bool) -> str:
    if val is None or not np.isfinite(val):
        return "--"
    if is_iteration:
        return str(int(round(val)))
    return f"{val:.3g}"


def _plot_table(
    ax: plt.Axes,
    records: Sequence[dict],
    *,
    row_key: str,
    metric: str,
    stats: Sequence[str],
    title: str | None,
) -> None:
    grouped: dict[float, list[dict]] = {}
    for record in records:
        x_val = axis_value_for_record(record, row_key)
        if x_val is None:
            continue
        grouped.setdefault(float(x_val), []).append(record)
    if not grouped:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        return

    stat_labels = {
        "min_iteration": "Iters to Min",
        "final_iteration": "Final Iter",
        "min": "Min",
        "final": "Final",
        "mean": "Mean",
    }
    xs = sorted(grouped.keys())
    row_labels = [f"{x:g}" for x in xs]
    col_labels = [stat_labels.get(stat, stat) for stat in stats]

    cell_text: list[list[str]] = []
    for x in xs:
        row_vals: list[str] = []
        records_at_x = grouped[x]
        for stat in stats:
            values: list[float] = []
            for record in records_at_x:
                val = _summary_metric_value(record, metric, stat=stat)
                if val is not None:
                    values.append(float(val))
            mean_val = float(np.mean(values)) if values else None
            is_iter = stat.endswith("iteration")
            row_vals.append(_format_table_value(mean_val, is_iteration=is_iter))
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
    if title:
        ax.set_title(title)


def _table_handler(ax: plt.Axes, instance: dict, context: dict) -> None:
    plot_entry = instance["plot_entry"]
    full_spec = instance["full_spec"]
    plot_key = plot_entry.get("key")
    x_key = plot_entry.get("x")
    if not x_key:
        raise ValueError(f"Table plot '{plot_key}' requires x key.")
    metric = plot_entry.get("y")
    if not metric:
        raise ValueError(f"Table plot '{plot_key}' requires y metric.")
    stats = plot_entry.get("stats")
    if stats is None:
        stats = full_spec.get("stats") or ["min_iteration", "min", "final"]
    _plot_table(
        ax,
        instance["records"],
        row_key=x_key,
        metric=metric,
        stats=stats,
        title=plot_entry.get("title"),
    )
