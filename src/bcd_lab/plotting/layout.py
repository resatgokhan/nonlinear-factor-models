"""Figure-independent grouping plus matplotlib figure layout construction."""

from __future__ import annotations

from typing import Sequence

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from .plot_config import legend_fallback_label, legend_label, legend_value_for_record
from .registry import get_plot_definition


def _single_size_override(
    plot_instances: Sequence[dict], field: str, group_name: str | None
) -> float | None:
    values = {
        instance["plot_entry"].get(field)
        for instance in plot_instances
        if instance["plot_entry"].get(field) is not None
    }
    if len(values) > 1:
        label = f" '{group_name}'" if group_name else ""
        raise ValueError(f"Conflicting {field} overrides for plot group{label}.")
    return float(next(iter(values))) if values else None


def _subplot_dimensions(
    plot_instances: Sequence[dict],
    layout: dict,
    *,
    max_columns: int,
    group_name: str | None,
) -> tuple[float, float]:
    width = _single_size_override(plot_instances, "width", group_name)
    height = _single_size_override(plot_instances, "height", group_name)
    width = width if width is not None else layout.get("width")
    height = height if height is not None else layout.get("height")
    ratio = layout.get("plot_ratio")
    if ratio is not None and width is not None and height is None:
        height = float(width) * float(ratio)
    if ratio is not None and height is not None and width is None:
        width = float(height) / float(ratio)
    if width is None:
        width = 12.0 / max_columns
    if height is None:
        height = float(width) * float(ratio) if ratio is not None else 3.6
    scale = float(layout.get("size_scale") or 1.0)
    return float(width) * scale, float(height) * scale


def _group_layout(
    plot_instances: Sequence[dict],
    *,
    layout: dict,
    max_per_row: int,
    group_name: str | None = None,
) -> dict:
    full_rows, remainder = divmod(len(plot_instances), max_per_row)
    row_sizes = [max_per_row] * full_rows
    if remainder:
        row_sizes.append(remainder)
    row_count = len(row_sizes)
    max_columns = max(row_sizes) if row_sizes else 1
    width, height = _subplot_dimensions(
        plot_instances,
        layout,
        max_columns=max_columns,
        group_name=group_name,
    )
    return {
        "row_sizes": row_sizes,
        "nrows": row_count,
        "max_cols": max_columns,
        "plot_area_width": width * max_columns,
        "plot_area_height": height * row_count,
    }


def _group_plot_instances(
    plots: Sequence[dict],
    *,
    plot_section: dict,
    records: Sequence[dict],
    legend_key: str | None,
    legend_title: str | None,
    base_name: str,
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for plot_entry in plots:
        plot_key = plot_entry.get("key")
        full_spec = plot_section.get(plot_key, {}) if plot_key else {}
        pdf_name = plot_entry.get("pdf") or full_spec.get("pdf") or base_name
        grouped.setdefault(pdf_name, [])

        plot_type = plot_entry.get("type") or "line"
        definition = get_plot_definition(plot_type)
        overlay = plot_entry.get("overlay_sweep", True)
        if definition.supports_overlay and not overlay:
            for idx, rec in enumerate(records):
                fallback_label = legend_fallback_label(rec, index=idx)
                label = legend_label(
                    legend_value_for_record(rec, legend_key),
                    fallback_label,
                )
                grouped[pdf_name].append(
                    {
                        "plot_entry": plot_entry,
                        "full_spec": full_spec,
                        "records": [rec],
                        "overlay": False,
                        "label": label,
                        "legend_token": legend_title or legend_key or "run",
                    }
                )
        else:
            grouped[pdf_name].append(
                {
                    "plot_entry": plot_entry,
                    "full_spec": full_spec,
                    "records": list(records),
                    "overlay": definition.supports_overlay and overlay,
                    "label": None,
                    "legend_token": legend_title or legend_key or "run",
                }
            )
    return grouped


def _setup_figure_and_axes(group_layout: dict) -> tuple[plt.Figure, list[plt.Axes]]:
    row_sizes = group_layout["row_sizes"]
    nrows = group_layout["nrows"]
    plot_area_width = group_layout["plot_area_width"]
    plot_area_height = group_layout["plot_area_height"]
    fig = plt.figure(figsize=(plot_area_width, plot_area_height), constrained_layout=False)
    outer_grid = gridspec.GridSpec(nrows, 1, figure=fig)
    axes_flat: list[plt.Axes] = []
    for row_idx, row_cols in enumerate(row_sizes):
        row_grid = gridspec.GridSpecFromSubplotSpec(
            1,
            row_cols,
            subplot_spec=outer_grid[row_idx],
        )
        for col_idx in range(row_cols):
            axes_flat.append(fig.add_subplot(row_grid[0, col_idx]))
    return fig, axes_flat
