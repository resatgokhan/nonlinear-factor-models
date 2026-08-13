"""Layout, sizing, and data extraction helper utilities for plotting."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .plot_config import (
    axis_value_for_record,
    legend_entry_sort_key,
    legend_fallback_label,
    legend_label,
    legend_value_for_record,
)


def _summary_metric_value(
    record: dict,
    metric: str,
    *,
    stat: str | None = None,
) -> float | None:
    summary = record.get("summary") or {}
    metrics = summary.get("metrics") or {}
    entry = metrics.get(metric)
    if isinstance(entry, dict):
        if stat:
            val = entry.get(stat)
            return float(val) if val is not None else None
        for key in ("min", "final", "mean"):
            val = entry.get(key)
            if val is not None:
                return float(val)
    elif entry is not None:
        return float(entry)

    return None


def _phi_circle_points(record: dict) -> tuple[np.ndarray, np.ndarray, float] | None:
    diagnostics = record.get("diagnostics") or []
    entries: list[tuple[int, float, float]] = []
    radius = None
    for entry in diagnostics:
        d0 = entry.get("phi_diff_0")
        dR = entry.get("phi_diff")
        if d0 is None or dR is None:
            continue
        iteration = int(entry["iteration"])
        entries.append((iteration, float(d0), float(dR)))
        if radius is None and iteration == 0:
            radius = float(dR)

    if not entries:
        return None
    entries.sort(key=lambda item: item[0])
    if radius is None:
        radius = float(entries[0][2])

    d0_vals = np.asarray([entry[1] for entry in entries], dtype=float)
    dR_vals = np.asarray([entry[2] for entry in entries], dtype=float)
    if radius <= 0:
        xs = np.zeros_like(d0_vals)
        ys = d0_vals
        return xs, ys, radius

    xs = (d0_vals**2 + radius**2 - dR_vals**2) / (2.0 * radius)
    y_sq = d0_vals**2 - xs**2
    ys = np.sqrt(np.clip(y_sq, 0.0, None))
    return xs, ys, radius


def _build_grid_values(
    records: Sequence[dict],
    *,
    x_key: str,
    y_key: str,
    metric: str,
) -> tuple[list[float], list[float], np.ndarray]:
    filtered: list[dict] = []
    for r in records:
        x_val = axis_value_for_record(r, x_key)
        y_val = axis_value_for_record(r, y_key)
        metric_val = _summary_metric_value(r, metric)
        if x_val is None or y_val is None or metric_val is None:
            continue
        filtered.append({"x": float(x_val), "y": float(y_val), "val": float(metric_val)})

    xs = sorted({r["x"] for r in filtered})
    ys = sorted({r["y"] for r in filtered})
    grid = np.full((len(ys), len(xs)), np.nan)
    if not xs or not ys:
        return xs, ys, grid

    x_idx = {x: j for j, x in enumerate(xs)}
    y_idx = {y: i for i, y in enumerate(ys)}
    for r in filtered:
        grid[y_idx[r["y"]], x_idx[r["x"]]] = r["val"]

    return xs, ys, grid


def _format_plot_title(
    base_title: str | None,
    *,
    metric: str | None,
    legend_token: str,
    label: str,
) -> str:
    title = base_title or metric or ""
    if not title:
        return f"{legend_token}={label}"
    if legend_token == "run":
        return title
    return f"{title} ({legend_token}={label})"


def _legend_labels(records: Sequence[dict], *, legend_key: str | None) -> list[str]:
    labels: list[str] = []
    for idx, record in enumerate(records):
        value = legend_value_for_record(record, legend_key)
        fallback = legend_fallback_label(record, index=idx)
        labels.append(legend_label(value, fallback))
    seen: set[str] = set()
    unique: list[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            unique.append(label)
    unique.sort(key=legend_entry_sort_key)
    return unique
