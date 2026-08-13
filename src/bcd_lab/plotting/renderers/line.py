"""Iteration-line plot renderer and trial aggregation services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..data import (
    _metric_series_for_record,
    _pick_longest_stable_trials,
    _pick_random_trials,
)
from ..helpers import _format_plot_title
from ..plot_config import (
    legend_entry_sort_key,
    legend_fallback_label,
    legend_label,
    legend_value_for_record,
)
from ..plot_util import plot_metric_over_iterations
from .summary_line import _plot_summary_line_grouped, summary_line_options


def _plot_line(
    ax: plt.Axes,
    records: Sequence[dict],
    *,
    metric: str,
    legend_key: str | None,
    x_scale: str | None,
    y_scale: str | None,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
    normalize_by_first: bool,
    normalize_eps: float,
    iteration_cutoff: int | None,
) -> None:
    log_metrics: Sequence[str] | str
    if y_scale == "log":
        log_metrics = (metric,)
    else:
        log_metrics = ()
    plot_metric_over_iterations(
        ax,
        records,
        metric,
        show_legend=False,
        legend_value_key=legend_key or "label",
        log_metrics=log_metrics,
        normalize_by_first=normalize_by_first,
        normalize_eps=normalize_eps,
        iteration_cutoff=iteration_cutoff,
    )
    if x_scale == "log":
        ax.set_xscale("log")
    elif x_scale == "linear":
        ax.set_xscale("linear")
    if y_scale == "linear":
        ax.set_yscale("linear")
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _trial_iteration_grid(
    records: Sequence[dict],
    metric: str,
    *,
    grid_mode: str | None,
    grid_count: int | None,
    grid_max: int | None,
) -> np.ndarray:
    max_iter = int(grid_max or 0)
    if max_iter <= 0:
        for record in records:
            series = _metric_series_for_record(record, metric)
            if series is None:
                continue
            iters, _ = series
            if iters.size:
                max_iter = max(max_iter, int(np.max(iters)))
    if max_iter <= 0:
        return np.array([], dtype=float)
    count = int(grid_count or 20)
    mode = grid_mode or "log"
    if mode == "linear":
        grid = np.linspace(1.0, float(max_iter), num=count)
    else:
        grid = np.logspace(0.0, np.log10(float(max_iter)), num=count)
    grid = np.unique(np.clip(np.round(grid), 1, max_iter)).astype(float)
    if grid.size == 0:
        grid = np.array([float(max_iter)], dtype=float)
    return grid


@dataclass(frozen=True)
class TrialLineOptions:
    metric: str
    legend_key: str | None
    x_scale: str | None
    y_scale: str | None
    title: str | None
    xlabel: str | None
    ylabel: str | None
    normalize_by_first: bool
    normalize_eps: float
    iteration_cutoff: int | None
    trial_stat: str
    trial_error: str | None
    trial_grid: str | None
    trial_grid_count: int | None
    trial_grid_max: int | None


def _trial_legend_entries(
    records: Sequence[dict], legend_key: str | None
) -> list[tuple[str, list[dict]]]:
    grouped: dict[object, list[dict]] = {}
    for record in records:
        key = legend_value_for_record(record, legend_key)
        grouped.setdefault(key, []).append(record)
    entries = [
        (legend_label(key, legend_fallback_label(group[0], index=index)), group)
        for index, (key, group) in enumerate(grouped.items())
    ]
    return sorted(entries, key=lambda item: legend_entry_sort_key(item[0]))


def _interpolate_trial(
    record: dict, grid: np.ndarray, options: TrialLineOptions
) -> np.ndarray | None:
    series = _metric_series_for_record(record, options.metric)
    if series is None:
        return None
    iterations, values = series
    if options.iteration_cutoff is not None:
        mask = iterations <= options.iteration_cutoff
        iterations = iterations[mask]
        values = values[mask]
    if iterations.size == 0:
        return None
    values = np.asarray(values, dtype=float)
    if options.normalize_by_first:
        baseline = values[0]
        if abs(baseline) <= options.normalize_eps:
            return None
        values = values / baseline
    return np.interp(
        grid,
        iterations.astype(float),
        values,
        left=np.nan,
        right=np.nan,
    )


def _trial_center_and_bounds(
    stacked: np.ndarray, *, statistic: str, error: str | None
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    center = np.nanmedian(stacked, axis=0) if statistic == "median" else np.nanmean(stacked, axis=0)
    if error is None:
        return center, None, None
    if error == "std":
        spread = np.nanstd(stacked, axis=0, ddof=1)
        return center, center - spread, center + spread
    if statistic == "median" and error == "ci95":
        return (
            center,
            np.nanpercentile(stacked, 2.5, axis=0),
            np.nanpercentile(stacked, 97.5, axis=0),
        )
    count = np.sum(np.isfinite(stacked), axis=0)
    spread = np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(np.maximum(count, 1))
    multiplier = 1.96 if error == "ci95" else 1.0
    return center, center - multiplier * spread, center + multiplier * spread


def _style_iteration_axis(ax: plt.Axes, options: TrialLineOptions) -> None:
    if options.x_scale in {"log", "linear"}:
        ax.set_xscale(options.x_scale)
    if options.y_scale in {"log", "linear"}:
        ax.set_yscale(options.y_scale)
    if options.title:
        ax.set_title(options.title)
    if options.xlabel:
        ax.set_xlabel(options.xlabel)
    if options.ylabel:
        ax.set_ylabel(options.ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)


def _plot_line_trials(
    ax: plt.Axes,
    records: Sequence[dict],
    options: TrialLineOptions,
) -> None:
    linestyles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]
    for index, (label, group) in enumerate(_trial_legend_entries(records, options.legend_key)):
        grid = _trial_iteration_grid(
            group,
            options.metric,
            grid_mode=options.trial_grid,
            grid_count=options.trial_grid_count,
            grid_max=options.trial_grid_max,
        )
        trials = [
            values
            for record in group
            if (values := _interpolate_trial(record, grid, options)) is not None
        ]
        if grid.size == 0 or not trials:
            continue
        with np.errstate(invalid="ignore"):
            center, lower, upper = _trial_center_and_bounds(
                np.vstack(trials),
                statistic=options.trial_stat,
                error=options.trial_error,
            )
        ax.plot(
            grid,
            center,
            linewidth=1.6,
            label=label,
            linestyle=linestyles[index % len(linestyles)],
        )
        if lower is not None and upper is not None:
            ax.fill_between(grid, lower, upper, alpha=0.2)
    _style_iteration_axis(ax, options)


def _option(plot_entry: dict, full_spec: dict, name: str, default: Any = None) -> Any:
    value = plot_entry.get(name)
    return value if value is not None else full_spec.get(name, default)


def _plot_summary_from_instance(
    ax: plt.Axes,
    instance: dict,
    *,
    metric: str,
    legend_key: str | None,
) -> None:
    entry = dict(instance["plot_entry"])
    entry["y"] = metric
    _plot_summary_line_grouped(
        ax,
        instance["records"],
        summary_line_options(entry, instance["full_spec"]),
        legend_key=legend_key,
    )


def _global_trial_option(entry: dict, spec: dict, plot_section: dict, name: str) -> Any:
    value = _option(entry, spec, name)
    return value if value is not None else plot_section.get(name)


def _select_line_records(
    records: Sequence[dict],
    *,
    trial_pick: str | None,
    trial_seed: int | None,
    legend_key: str | None,
    metric: str,
) -> Sequence[dict]:
    if trial_pick == "random":
        if trial_seed is None:
            raise ValueError("trial_pick='random' requires trial_seed")
        return _pick_random_trials(records, legend_key=legend_key, seed=trial_seed)
    if trial_pick == "longest_stable":
        return _pick_longest_stable_trials(
            records,
            legend_key=legend_key,
            metric=metric,
        )
    return records


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _trial_line_options(
    instance: dict,
    context: dict,
    *,
    metric: str,
    title: str | None,
) -> TrialLineOptions | None:
    entry = instance["plot_entry"]
    spec = instance["full_spec"]
    section = context["plot_section"]
    trial_stat = _global_trial_option(entry, spec, section, "trial_stat")
    if not trial_stat:
        return None
    return TrialLineOptions(
        metric=metric,
        legend_key=context["legend_key"],
        x_scale=entry.get("x_scale"),
        y_scale=entry.get("y_scale"),
        title=title,
        xlabel=entry.get("xlabel"),
        ylabel=entry.get("ylabel"),
        normalize_by_first=bool(_option(entry, spec, "normalize_by_first", False)),
        normalize_eps=float(_option(entry, spec, "normalize_eps", 1.0e-12)),
        iteration_cutoff=_optional_int(_option(entry, spec, "iteration_cutoff")),
        trial_stat=str(trial_stat),
        trial_error=_optional_str(_global_trial_option(entry, spec, section, "trial_error")),
        trial_grid=_optional_str(_global_trial_option(entry, spec, section, "trial_grid")),
        trial_grid_count=_optional_int(
            _global_trial_option(entry, spec, section, "trial_grid_count")
        ),
        trial_grid_max=_optional_int(_global_trial_option(entry, spec, section, "trial_grid_max")),
    )


def _plot_iteration_from_instance(
    ax: plt.Axes, instance: dict, context: dict, *, metric: str
) -> None:
    entry = instance["plot_entry"]
    spec = instance["full_spec"]
    legend_key = context["legend_key"]
    title = entry.get("title")
    if not instance["overlay"]:
        title = _format_plot_title(
            title,
            metric=metric,
            legend_token=instance["legend_token"],
            label=instance["label"],
        )
    records = _select_line_records(
        instance["records"],
        trial_pick=_option(entry, spec, "trial_pick"),
        trial_seed=_option(entry, spec, "trial_seed"),
        legend_key=legend_key,
        metric=metric,
    )
    trial_options = _trial_line_options(
        instance,
        context,
        metric=metric,
        title=title,
    )
    if trial_options is not None:
        _plot_line_trials(ax, records, trial_options)
        return
    cutoff = _option(entry, spec, "iteration_cutoff")
    _plot_line(
        ax,
        records,
        metric=metric,
        legend_key=legend_key,
        x_scale=entry.get("x_scale"),
        y_scale=entry.get("y_scale"),
        title=title,
        xlabel=entry.get("xlabel"),
        ylabel=entry.get("ylabel"),
        normalize_by_first=bool(_option(entry, spec, "normalize_by_first", False)),
        normalize_eps=float(_option(entry, spec, "normalize_eps", 1.0e-12)),
        iteration_cutoff=int(cutoff) if cutoff is not None else None,
    )


def _line_handler(ax: plt.Axes, instance: dict, context: dict) -> None:
    entry = instance["plot_entry"]
    metric = entry.get("y")
    if not metric:
        raise ValueError(f"Line plot '{entry.get('key')}' requires y metric.")
    if entry.get("x") not in (None, "iteration"):
        _plot_summary_from_instance(
            ax,
            instance,
            metric=metric,
            legend_key=context["legend_key"],
        )
        return
    _plot_iteration_from_instance(ax, instance, context, metric=metric)
