"""Summary-line renderer and its aggregation services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..helpers import _summary_metric_value
from ..plot_config import (
    axis_value_for_record,
    legend_entry_sort_key,
    legend_fallback_label,
    legend_label,
    legend_value_for_record,
)


def _normalize_by_baseline(
    records: Sequence[dict],
    grouped: dict[float, list[float]],
    xs: np.ndarray,
    ys: np.ndarray,
    x_key: str,
    metric: str,
    summary_stat: str | None,
    baseline_key: str | None,
    baseline_value: float,
) -> tuple[dict[float, list[float]], np.ndarray]:
    key = baseline_key or x_key
    baseline_vals: list[float] = []
    for record in records:
        x_val = axis_value_for_record(record, key)
        if x_val is None:
            continue
        if isinstance(x_val, (int, float, np.integer, np.floating)):
            if not np.isclose(float(x_val), float(baseline_value)):
                continue
        elif str(x_val) != str(baseline_value):
            continue
        metric_val = _summary_metric_value(record, metric, stat=summary_stat)
        if metric_val is not None:
            baseline_vals.append(float(metric_val))
    if baseline_vals:
        baseline = float(np.mean(baseline_vals))
        if baseline != 0.0 and np.isfinite(baseline):
            grouped = {x: [val / baseline for val in vals] for x, vals in grouped.items()}
            ys = ys / baseline
    return grouped, ys


def _plot_summary_error(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    grouped: dict[float, list[float]],
    error: str | None,
    error_style: str | None,
    color: str | None,
) -> None:
    yerr = np.zeros_like(ys)
    for i, x_val in enumerate(xs):
        vals = np.asarray(grouped[float(x_val)], dtype=float)
        if vals.size <= 1:
            yerr[i] = 0.0
            continue
        std = float(np.std(vals, ddof=1))
        sem = std / np.sqrt(float(vals.size))
        if error == "std":
            yerr[i] = std
        elif error == "sem":
            yerr[i] = sem
        else:
            yerr[i] = 1.96 * sem
    style = error_style or "band"
    if style == "bars":
        ax.errorbar(xs, ys, yerr=yerr, fmt="none", capsize=3, alpha=0.8, color=color)
    else:
        ax.fill_between(xs, ys - yerr, ys + yerr, alpha=0.2, color=color)


def _plot_summary_line_fit(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    fit_line: int,
    x_scale: str | None,
    y_scale: str | None,
) -> None:
    power = fit_line
    x_vals = xs.copy()
    y_vals = ys.copy()
    if x_scale == "log":
        mask = x_vals > 0.0
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]
    if y_scale == "log":
        mask = y_vals > 0.0
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]
    if x_vals.size >= 2 and y_vals.size >= 2:
        basis = np.power(x_vals, power)
        design = np.column_stack([basis, np.ones_like(basis)])
        coef, intercept = np.linalg.lstsq(design, y_vals, rcond=None)[0]
        if x_scale == "log":
            x_line = np.logspace(
                np.log10(float(x_vals.min())),
                np.log10(float(x_vals.max())),
                num=100,
            )
        else:
            x_line = np.linspace(float(x_vals.min()), float(x_vals.max()), num=100)
        y_line = coef * np.power(x_line, power) + intercept
        ax.plot(
            x_line,
            y_line,
            linestyle=":",
            linewidth=1.2,
            color="0.35",
            label="_nolegend_",
        )


@dataclass(frozen=True)
class SummaryLineOptions:
    x_key: str
    metric: str
    x_scale: str | None = None
    y_scale: str | None = None
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    summary_stat: str | None = None
    normalize_by_baseline: bool = False
    baseline_key: str | None = None
    baseline_value: float | None = None
    line_style: str | None = None
    show_trials: bool | None = None
    show_mean: bool | None = None
    error: str | None = None
    error_style: str | None = None
    fit_line: int | None = None


def _summary_groups(
    records: Sequence[dict], options: SummaryLineOptions
) -> dict[float, list[float]]:
    grouped: dict[float, list[float]] = {}
    for record in records:
        x_value = axis_value_for_record(record, options.x_key)
        metric_value = _summary_metric_value(
            record,
            options.metric,
            stat=options.summary_stat,
        )
        if x_value is not None and metric_value is not None:
            grouped.setdefault(float(x_value), []).append(float(metric_value))
    if options.x_scale == "log":
        return {x: values for x, values in grouped.items() if x > 0.0}
    return grouped


def _draw_summary_values(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    grouped: dict[float, list[float]],
    options: SummaryLineOptions,
    *,
    label: str | None,
    color: str | None,
) -> None:
    if options.show_trials:
        for x_value, values in grouped.items():
            ax.scatter(
                np.full(len(values), x_value, dtype=float),
                values,
                s=18,
                alpha=0.35,
                color=color,
            )
    if options.show_mean is not False:
        linestyle = "none" if options.line_style == "none" else "solid"
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.6,
            linestyle=linestyle,
            label=label,
            color=color,
        )
    if options.error:
        _plot_summary_error(
            ax,
            xs,
            ys,
            grouped,
            options.error,
            options.error_style,
            color,
        )
    if options.fit_line is not None:
        _plot_summary_line_fit(
            ax,
            xs,
            ys,
            options.fit_line,
            options.x_scale,
            options.y_scale,
        )


def _style_summary_axis(ax: plt.Axes, options: SummaryLineOptions) -> None:
    if options.x_scale in {"log", "linear"}:
        ax.set_xscale(options.x_scale)
    if options.y_scale in {"log", "linear"}:
        ax.set_yscale(options.y_scale)
    ax.set_title(options.title or options.metric)
    ax.set_xlabel(options.xlabel or options.x_key)
    ax.set_ylabel(options.ylabel or options.metric)
    ax.grid(True, linestyle="--", alpha=0.35)


def _plot_summary_line(
    ax: plt.Axes,
    records: Sequence[dict],
    options: SummaryLineOptions,
    *,
    label: str | None = None,
    color: str | None = None,
) -> None:
    grouped = _summary_groups(records, options)
    if not grouped:
        ax.text(
            0.5,
            0.5,
            "no data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
        )
        return
    xs = np.array(sorted(grouped), dtype=float)
    ys = np.array([float(np.mean(grouped[x])) for x in xs], dtype=float)
    if options.normalize_by_baseline and options.baseline_value is not None:
        grouped, ys = _normalize_by_baseline(
            records,
            grouped,
            xs,
            ys,
            options.x_key,
            options.metric,
            options.summary_stat,
            options.baseline_key,
            options.baseline_value,
        )
    _draw_summary_values(
        ax,
        xs,
        ys,
        grouped,
        options,
        label=label,
        color=color,
    )
    _style_summary_axis(ax, options)


def _plot_summary_line_grouped(
    ax: plt.Axes,
    records: Sequence[dict],
    options: SummaryLineOptions,
    *,
    legend_key: str | None,
) -> None:
    grouped: dict[object, list[dict]] = {}
    for record in records:
        key = legend_value_for_record(record, legend_key)
        grouped.setdefault(key, []).append(record)
    entries = [
        (
            legend_label(key, legend_fallback_label(group[0], index=index)),
            group,
        )
        for index, (key, group) in enumerate(grouped.items())
    ]
    entries.sort(key=lambda item: legend_entry_sort_key(item[0]))
    if not entries:
        _plot_summary_line(ax, records, options)
        return
    for label, group in entries:
        _plot_summary_line(ax, group, options, label=label)


def _summary_option(entry: dict, spec: dict, name: str, default: Any = None) -> Any:
    value = entry.get(name)
    return value if value is not None else spec.get(name, default)


def summary_line_options(entry: dict, spec: dict) -> SummaryLineOptions:
    """Build normalized renderer options from one plot entry."""
    x_key = entry.get("x")
    if not x_key:
        raise ValueError(f"Summary line plot '{entry.get('key')}' requires x key.")
    metric = entry.get("y")
    if not metric:
        raise ValueError(f"Summary line plot '{entry.get('key')}' requires y metric.")
    baseline = _summary_option(entry, spec, "baseline_value", 0.0)
    return SummaryLineOptions(
        x_key=x_key,
        metric=metric,
        x_scale=entry.get("x_scale"),
        y_scale=entry.get("y_scale"),
        title=entry.get("title"),
        xlabel=entry.get("xlabel"),
        ylabel=entry.get("ylabel"),
        summary_stat=_summary_option(entry, spec, "summary_stat"),
        normalize_by_baseline=bool(_summary_option(entry, spec, "normalize_by_baseline", False)),
        baseline_key=_summary_option(entry, spec, "baseline_key"),
        baseline_value=float(baseline) if baseline is not None else None,
        line_style=_summary_option(entry, spec, "line_style"),
        show_trials=_summary_option(entry, spec, "show_trials"),
        show_mean=_summary_option(entry, spec, "show_mean"),
        error=_summary_option(entry, spec, "error"),
        error_style=_summary_option(entry, spec, "error_style"),
        fit_line=_summary_option(entry, spec, "fit_line"),
    )


def _summary_line_handler(ax: plt.Axes, instance: dict, context: dict) -> None:
    _plot_summary_line(
        ax,
        instance["records"],
        summary_line_options(instance["plot_entry"], instance["full_spec"]),
    )
