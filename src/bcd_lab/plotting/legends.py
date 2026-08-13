"""Legend construction and plot text styling services."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .helpers import _legend_labels
from .output import save_figure
from .plot_config import legend_entry_sort_key, legend_row_major_order


def _legend_kwargs(plot_section: dict) -> dict:
    legend_font_scale = plot_section.get("legend_font_scale")
    legend_box_scale = plot_section.get("legend_box_scale")
    if legend_font_scale is None and legend_box_scale is None:
        return {}

    def _scaled(value: float, scale: float | None) -> float:
        return value * scale if scale is not None else value

    params: dict[str, float] = {}
    if legend_font_scale is not None:
        base = float(plt.rcParams.get("legend.fontsize", 10.0))
        params["fontsize"] = _scaled(base, float(legend_font_scale))
        params["title_fontsize"] = _scaled(base, float(legend_font_scale))
    if legend_box_scale is not None:
        scale = float(legend_box_scale)
        params["borderpad"] = _scaled(float(plt.rcParams.get("legend.borderpad", 0.4)), scale)
        params["labelspacing"] = _scaled(float(plt.rcParams.get("legend.labelspacing", 0.5)), scale)
        params["handlelength"] = _scaled(float(plt.rcParams.get("legend.handlelength", 2.0)), scale)
        params["handletextpad"] = _scaled(
            float(plt.rcParams.get("legend.handletextpad", 0.8)), scale
        )
        params["columnspacing"] = _scaled(
            float(plt.rcParams.get("legend.columnspacing", 2.0)), scale
        )
        params["borderaxespad"] = _scaled(
            float(plt.rcParams.get("legend.borderaxespad", 0.5)), scale
        )
    return params


def _apply_font_scales(axes_flat: list[plt.Axes], plot_section: dict) -> None:
    axis_font_scale = plot_section.get("axis_font_scale")
    tick_font_scale = plot_section.get("tick_font_scale")
    title_font_scale = plot_section.get("title_font_scale")

    if axis_font_scale is None and tick_font_scale is None and title_font_scale is None:
        return

    def _scaled(value: float, scale: float | None) -> float:
        return value * scale if scale is not None else value

    for ax in axes_flat:
        if axis_font_scale is not None:
            scale = float(axis_font_scale)
            ax.xaxis.label.set_size(_scaled(ax.xaxis.label.get_size(), scale))
            ax.yaxis.label.set_size(_scaled(ax.yaxis.label.get_size(), scale))
        if title_font_scale is not None:
            scale = float(title_font_scale)
            ax.title.set_size(_scaled(ax.title.get_size(), scale))
        if tick_font_scale is not None:
            scale = float(tick_font_scale)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontsize(_scaled(tick.get_fontsize(), scale))


def _attach_group_legend(
    fig: plt.Figure,
    legend_axis: plt.Axes,
    legend_title: str | None,
    legend_key: str | None,
    plot_section: dict,
    legend_y: float | None,
    legend_kwargs: dict,
    plot_left: float,
    plot_right: float,
    plot_top: float,
    plot_bottom: float,
) -> None:
    handles, labels = legend_axis.get_legend_handles_labels()
    if handles:
        sorted_entries = sorted(
            zip(handles, labels, strict=True),
            key=lambda item: legend_entry_sort_key(item[1]),
        )
        ncol = min(len(sorted_entries), 6)
        sorted_entries = legend_row_major_order(sorted_entries, ncol=ncol)
        handles, labels = zip(*sorted_entries, strict=True)
        ncol = min(len(labels), 6)
        fig.legend(
            handles,
            labels,
            title=legend_title or legend_key or "",
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=ncol,
            frameon=True,
            **legend_kwargs,
        )
        fig.subplots_adjust(
            left=plot_left,
            right=plot_right,
            top=plot_top,
            bottom=plot_bottom,
            wspace=0.45,
            hspace=0.45,
        )


def _render_standalone_legends(
    legends: list[dict],
    records: Sequence[dict],
    legend_key: str | None,
    legend_title: str | None,
    plot_section: dict,
    layout: dict,
    base_name: str,
    plot_dir: Path,
    legend_kwargs: dict,
) -> list[Path]:
    output_paths: list[Path] = []
    plot_left = float(plt.rcParams.get("figure.subplot.left", 0.125))
    plot_right = float(plt.rcParams.get("figure.subplot.right", 0.9))
    plot_width_frac = plot_right - plot_left
    legend_y = plot_section.get("legend_y", 1.03)
    if legend_y is not None:
        try:
            legend_y = float(legend_y)
        except (TypeError, ValueError) as exc:
            raise ValueError("plot.legend_y must be a float or null") from exc
        if legend_y > 1.0:
            legend_y = 1.0

    labels = _legend_labels(records, legend_key=legend_key)
    if labels:
        ncol = min(len(labels), 6)
        legend_rows = int(np.ceil(len(labels) / ncol))
        legend_height = max(0.6, 0.35 * legend_rows + 0.25)
        colors = plt.rcParams.get("axes.prop_cycle", plt.cycler(color=["C0"]))
        color_list = colors.by_key().get("color", [])
        for legend_entry in legends:
            pdf_name = legend_entry.get("pdf") or f"{base_name}_legend"
            subplot_width = layout.get("width")
            if subplot_width is None:
                subplot_width = 12.0
            size_scale = layout.get("size_scale")
            if size_scale is not None:
                subplot_width = float(subplot_width) * float(size_scale)
            plot_area_width = float(subplot_width)
            fig_width = (
                plot_area_width / plot_width_frac if plot_width_frac > 0 else plot_area_width
            )
            fig = plt.figure(figsize=(fig_width, legend_height), constrained_layout=False)
            handles = [
                Line2D([0], [0], color=color_list[idx % len(color_list)], linewidth=1.6)
                if color_list
                else Line2D([0], [0], linewidth=1.6)
                for idx in range(len(labels))
            ]
            entries = list(zip(handles, labels, strict=True))
            entries = sorted(entries, key=lambda item: legend_entry_sort_key(item[1]))
            entries = legend_row_major_order(entries, ncol=ncol)
            ordered_handles, labels_sorted = zip(*entries, strict=True)
            ncol = min(len(labels_sorted), 6)
            fig.legend(
                ordered_handles,
                labels_sorted,
                title=legend_title or legend_key or "",
                loc="upper center",
                bbox_to_anchor=(0.5, legend_y if legend_y is not None else 1.0),
                ncol=ncol,
                frameon=True,
                **legend_kwargs,
            )
            out_path = plot_dir / f"{pdf_name}_legend.pdf"
            save_figure(fig, out_path)
            print(f"Saved legend to {out_path}")
            output_paths.append(out_path)
            plt.close(fig)
    return output_paths
