#!/usr/bin/env python3
"""Shared plotting templates/helpers for experiment PDFs."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .data import metric_series
from .plot_config import legend_fallback_label, legend_value_for_record

__all__ = [
    "DEFAULT_PDF_STYLE",
    "PLOT_TEXT_COMBINED",
    "apply_pdf_style",
    "plot_metric_over_iterations",
]

# --- LaTeX-friendly PDF output ---
DEFAULT_PDF_STYLE: dict[str, Any] = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}


def apply_pdf_style(*, scale: float = 1.0) -> None:
    """Apply consistent PDF-friendly matplotlib settings."""
    style = DEFAULT_PDF_STYLE.copy()
    if scale != 1.0:
        for key in (
            "axes.titlesize",
            "axes.labelsize",
            "legend.fontsize",
            "xtick.labelsize",
            "ytick.labelsize",
        ):
            style[key] = float(style[key]) * float(scale)
    plt.rcParams.update(style)


PLOT_TEXT_COMBINED: dict[str, Any] = {
    "x_label_iter": "Iteration",
    "legend_title": r"$\alpha$",
    "metric_titles": {
        "loss": "Loss",
        "goodness": "Goodness",
        "phi_diff": r"$\|\phi^* - \phi^t\|_H$",
    },
    "y_labels": {
        "loss": "loss",
        "goodness": "goodness",
        "phi_diff": r"$\|\phi - \phi^t\|_H$",
    },
}


def _set_goodness_ylim(ax: plt.Axes, series: Sequence[np.ndarray]) -> None:
    tails = []
    for vals in series:
        if vals.size <= 1:
            continue
        v = vals[1:]
        v = v[np.isfinite(v) & (v > 0)]
        if v.size:
            tails.append(v)
    if tails:
        all_tail = np.concatenate(tails)
        if all_tail.size >= 2:
            bounds = np.asarray(np.percentile(all_tail, [1, 99]), dtype=float)
            lo, hi = float(bounds[0]), float(bounds[1])
            if np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi > lo:
                pad = (hi / lo) ** 0.08
                ax.set_ylim(lo / pad, hi * pad)


def _resolve_legend_label(rec: dict, index: int, legend_value_key: str) -> str:
    legend_val = legend_value_for_record(rec, legend_value_key)
    if legend_val is None:
        return rec.get("label") or legend_fallback_label(rec, index=index)
    if isinstance(legend_val, (int, float)):
        return f"{legend_val:.3g}"
    return str(legend_val)


def _adjust_metric_values(
    vals: np.ndarray,
    subtract_min: bool,
    normalize_by_first: bool,
    normalize_eps: float,
) -> np.ndarray:
    if subtract_min:
        finite_vals = vals[np.isfinite(vals)]
        offset = float(np.min(finite_vals)) if finite_vals.size else 0.0
        vals = vals - offset
        shifted_min = vals[np.isfinite(vals)].min() if np.isfinite(vals).any() else 0.0
        if shifted_min <= 0:
            vals = vals + (abs(shifted_min) + 1e-12)
    if normalize_by_first:
        baseline = float(vals[0])
        if not np.isfinite(baseline):
            finite_vals = vals[np.isfinite(vals)]
            baseline = float(finite_vals[0]) if finite_vals.size else 0.0
        if baseline < 0:
            vals = -vals
            baseline = -baseline
        if baseline >= normalize_eps:
            vals = vals / baseline
    return vals


def _set_metric_yscale(
    ax: plt.Axes,
    metric: str,
    series_for_limits: list[np.ndarray],
    goodness_series_for_ylim: list[np.ndarray],
    subtract_min: bool,
    log_metrics: Sequence[str] | str,
) -> None:
    log_all = log_metrics == "all"
    log_set = None if log_all else set(log_metrics)

    if series_for_limits:
        stacked = np.concatenate(series_for_limits)
        finite = stacked[np.isfinite(stacked)]
        min_val = float(np.min(finite)) if finite.size else None
    else:
        min_val = None

    if subtract_min or log_all or (log_set and metric in log_set):
        if min_val is not None and min_val > 0:
            ax.set_yscale("log")
            if metric == "goodness" and goodness_series_for_ylim:
                _set_goodness_ylim(ax, goodness_series_for_ylim)
        else:
            ax.set_yscale("linear")
    else:
        ax.set_yscale("linear")


def plot_metric_over_iterations(
    ax: plt.Axes,
    records: Sequence[Dict],
    metric: str,
    *,
    show_legend: bool = False,
    plot_text: Dict = PLOT_TEXT_COMBINED,
    legend_value_key: str = "alpha",
    log_metrics: Sequence[str] | str = "all",
    subtract_min: bool = False,
    normalize_by_first: bool = False,
    normalize_eps: float = 1.0e-12,
    iteration_cutoff: int | None = None,
) -> None:
    goodness_series_for_ylim: list[np.ndarray] = []
    series_for_limits: list[np.ndarray] = []
    linestyles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]

    for idx, rec in enumerate(records):
        series = metric_series(rec.get("diagnostics") or [], metric)
        if not series:
            continue
        iters, vals = series
        if iteration_cutoff is not None:
            cutoff = int(iteration_cutoff)
            mask = iters <= cutoff
            if not np.any(mask):
                continue
            iters = iters[mask]
            vals = vals[mask]

        label = _resolve_legend_label(rec, idx, legend_value_key)
        vals = _adjust_metric_values(vals, subtract_min, normalize_by_first, normalize_eps)

        ax.plot(
            iters,
            vals,
            label=label,
            linestyle=linestyles[idx % len(linestyles)],
            alpha=0.85 if metric == "goodness" else 1.0,
            linewidth=1.6,
            zorder=2 + idx,
        )

        if metric == "goodness":
            goodness_series_for_ylim.append(vals)
        series_for_limits.append(vals)

    _set_metric_yscale(
        ax,
        metric,
        series_for_limits,
        goodness_series_for_ylim,
        subtract_min,
        log_metrics,
    )

    ax.set_xlabel(plot_text["x_label_iter"])
    ylabel = plot_text["y_labels"].get(metric, metric)
    if subtract_min:
        ylabel = f"{ylabel} - min"
    if normalize_by_first:
        ylabel = f"{ylabel} / initial"
    ax.set_ylabel(ylabel)
    ax.set_title(plot_text["metric_titles"].get(metric, metric))
    ax.grid(True, linestyle="--", alpha=0.35)

    if show_legend:
        ax.legend(title=plot_text["legend_title"])
