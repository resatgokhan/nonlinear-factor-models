"""Embedded phi-trajectory renderer."""

from __future__ import annotations

import itertools
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..helpers import _phi_circle_points
from ..plot_config import legend_fallback_label, legend_label, legend_value_for_record


def _plot_phi_circle(
    ax: plt.Axes,
    records: Sequence[dict],
    *,
    legend_key: str | None,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
) -> None:
    linestyles = itertools.cycle(("-", "--", "-.", ":"))
    markers = itertools.cycle(("o", "s", "D", "^", "v", "P", "X"))
    any_series = False
    for idx, record in enumerate(records):
        series = _phi_circle_points(record)
        if series is None:
            continue
        xs, ys, radius = series
        if ys.size:
            ys = ys if (idx % 2 == 0) else -ys
        label = legend_label(
            legend_value_for_record(record, legend_key),
            legend_fallback_label(record, index=idx),
        )
        linestyle = next(linestyles)
        marker = next(markers)
        markevery = max(int(len(xs) / 12), 1) if xs.size else None
        line = ax.plot(
            xs,
            ys,
            label=label,
            linewidth=1.8,
            alpha=0.75,
            linestyle=linestyle,
            marker=marker,
            markersize=3.5,
            markevery=markevery,
        )[0]
        color = line.get_color()
        if xs.size:
            ax.scatter([radius], [0.0], marker="*", color=color, s=70, label="_nolegend_")
        if xs.size:
            ax.scatter([xs[0]], [ys[0]], marker="o", color=color, s=18, label="_nolegend_")
            ax.scatter([xs[-1]], [ys[-1]], marker="s", color=color, s=22, label="_nolegend_")
        if radius > 0:
            theta = np.linspace(0.0, 2.0 * np.pi, 240)
            ax.plot(
                radius * np.cos(theta),
                radius * np.sin(theta),
                linestyle="--",
                color=color,
                alpha=0.5,
                linewidth=1.0,
                label="_nolegend_",
            )
            ax.scatter([0.0], [0.0], marker="x", color=color, s=30, label="_nolegend_")
            ax.scatter(
                [radius],
                [0.0],
                marker="o",
                facecolors="none",
                edgecolors=color,
                s=36,
                label="_nolegend_",
            )
        any_series = True

    if not any_series:
        ax.text(
            0.5,
            0.5,
            "no phi trajectory data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title or "Phi trajectory (embedded)")
    ax.set_xlabel(xlabel or "embedded x")
    ax.set_ylabel(ylabel or "embedded y")
    ax.grid(True, linestyle="--", alpha=0.35)


def _phi_circle_handler(ax: plt.Axes, instance: dict, context: dict) -> None:
    plot_entry = instance["plot_entry"]
    _plot_phi_circle(
        ax,
        instance["records"],
        legend_key=context["legend_key"],
        title=plot_entry.get("title"),
        xlabel=plot_entry.get("xlabel"),
        ylabel=plot_entry.get("ylabel"),
    )
