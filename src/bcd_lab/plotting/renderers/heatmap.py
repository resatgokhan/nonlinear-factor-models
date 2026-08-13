"""Sweep heatmap renderer."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from ..helpers import _build_grid_values


def _plot_heatmap(
    ax: plt.Axes,
    records: Sequence[dict],
    *,
    x_key: str,
    y_key: str,
    metric: str,
    y_scale: str | None,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
    x_tick_mode: str | None = None,
    y_tick_mode: str | None = None,
) -> None:
    xs, ys, grid = _build_grid_values(records, x_key=x_key, y_key=y_key, metric=metric)
    cmap = plt.get_cmap().copy()
    cmap.set_bad("white")

    norm = None
    if y_scale == "log":
        positive = grid[np.isfinite(grid) & (grid > 0)]
        if positive.size:
            norm = LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive)))

    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )
    ax.set_title(title or metric)
    ax.set_xlabel(xlabel or x_key)
    ax.set_ylabel(ylabel or y_key)

    if x_tick_mode == "endpoints":
        x_indices = [0, len(xs) - 1] if len(xs) > 1 else [0]
    else:
        x_indices = list(np.arange(len(xs)))
    if y_tick_mode == "endpoints":
        y_indices = [0, len(ys) - 1] if len(ys) > 1 else [0]
    else:
        y_indices = list(np.arange(len(ys)))
    ax.set_xticks(x_indices)
    ax.set_yticks(y_indices)
    ax.set_xticklabels([f"{xs[i]:g}" for i in x_indices])
    ax.set_yticklabels([f"{ys[i]:g}" for i in y_indices])

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(metric)


def _heatmap_handler(ax: plt.Axes, instance: dict, context: dict) -> None:
    plot_entry = instance["plot_entry"]
    full_spec = instance["full_spec"]
    plot_key = plot_entry.get("key")
    x_key = plot_entry.get("x")
    y_key = plot_entry.get("y")
    if not x_key or not y_key:
        raise ValueError(f"Heatmap plot '{plot_key}' requires x and y keys.")
    metric = full_spec.get("metric") or "loss"
    x_tick_mode = plot_entry.get("x_tick_mode")
    if x_tick_mode is None:
        x_tick_mode = full_spec.get("x_tick_mode")
    y_tick_mode = plot_entry.get("y_tick_mode")
    if y_tick_mode is None:
        y_tick_mode = full_spec.get("y_tick_mode")
    _plot_heatmap(
        ax,
        context["records"],
        x_key=x_key,
        y_key=y_key,
        metric=metric,
        y_scale=plot_entry.get("y_scale"),
        title=plot_entry.get("title"),
        xlabel=plot_entry.get("xlabel"),
        ylabel=plot_entry.get("ylabel"),
        x_tick_mode=x_tick_mode,
        y_tick_mode=y_tick_mode,
    )
