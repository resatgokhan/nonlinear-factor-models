"""Plot orchestration across loading, registered renderers, layout, and output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from .data import _load_records
from .layout import _group_layout, _group_plot_instances, _setup_figure_and_axes
from .legends import (
    _apply_font_scales,
    _attach_group_legend,
    _legend_kwargs,
    _render_standalone_legends,
)
from .output import save_figure
from .plot_config import infer_legend_key, legend_display_name, legend_sort_key
from .plot_util import apply_pdf_style
from .registry import get_plot_definition


def _render_plots(
    axes_flat: list[plt.Axes], plot_instances: list[dict], context: dict
) -> plt.Axes | None:
    legend_axis = None
    for ax, instance in zip(axes_flat, plot_instances, strict=True):
        plot_entry = instance["plot_entry"]
        plot_type = plot_entry.get("type") or "line"

        definition = get_plot_definition(plot_type)
        definition.render(ax, instance, context)

        # Track legend axis
        is_overlay_candidate = (
            definition.legend_policy
            in {
                "overlay",
                "overlay_or_grouped",
            }
            and instance["overlay"]
        )
        is_grouped_candidate = definition.legend_policy in {
            "grouped",
            "overlay_or_grouped",
        } and plot_entry.get("x") not in (
            None,
            "iteration",
        )
        if (is_overlay_candidate or is_grouped_candidate) and legend_axis is None:
            legend_axis = ax
    return legend_axis


def _load_and_sort_records(
    results_dir: Path, plot_section: dict
) -> tuple[list[dict], str | None, str | None]:
    records = _load_records(results_dir)
    legend_key = infer_legend_key(records, plot_section)
    legend_title = plot_section.get("legend_title")
    if legend_title is None:
        legend_title = legend_display_name(legend_key)
    if legend_key:
        records = sorted(records, key=lambda rec: legend_sort_key(rec, legend_key))
    return records, legend_key, legend_title


def _resolve_plot_dir(plot_config: dict, results_dir: Path, layout: dict) -> tuple[Path, str]:
    plot_dir = plot_config.get("plot_dir")
    if plot_dir:
        plot_dir = Path(plot_dir)
    else:
        plot_dir = results_dir.parent.parent / "plots"
    base_name = layout.get("name") or plot_config.get("yaml_name") or "plot"
    if plot_config.get("plot_dir") is None:
        plot_dir = plot_dir / base_name
    return plot_dir, base_name


@dataclass(frozen=True)
class PlotServiceContext:
    results_dir: Path
    records: list[dict]
    legend_key: str | None
    legend_title: str | None
    plot_section: dict
    layout: dict
    plot_dir: Path
    max_per_row: int
    legend_kwargs: dict
    show: bool


def _legend_position(plot_section: dict, *, plot_top: float, plot_fraction: float) -> float | None:
    value = plot_section.get("legend_y", 1.03)
    if value is None:
        return None
    try:
        position = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("plot.legend_y must be a float or null") from exc
    if position > 1.0:
        return plot_top + (position - 1.0) * plot_fraction
    return position


def _render_plot_group(
    group_name: str,
    plot_instances: list[dict],
    context: PlotServiceContext,
) -> Path:
    group_layout = _group_layout(
        plot_instances,
        layout=context.layout,
        max_per_row=context.max_per_row,
        group_name=group_name,
    )
    figure, axes = _setup_figure_and_axes(group_layout)
    renderer_context = {
        "results_dir": context.results_dir,
        "records": context.records,
        "legend_key": context.legend_key,
        "plot_section": context.plot_section,
    }
    legend_axis = _render_plots(axes, plot_instances, renderer_context)
    show_legend = context.plot_section.get("show_legend", True)
    _apply_font_scales(axes, context.plot_section)

    plot_left = float(plt.rcParams.get("figure.subplot.left", 0.125))
    plot_right = float(plt.rcParams.get("figure.subplot.right", 0.9))
    plot_bottom = 0.14
    plot_top = 0.84 if legend_axis is not None and show_legend else 0.92
    plot_fraction = plot_top - plot_bottom
    width_fraction = plot_right - plot_left
    if plot_fraction <= 0 or width_fraction <= 0:
        raise ValueError("Invalid plot layout fractions; plot area size is non-positive.")
    figure.set_size_inches(
        group_layout["plot_area_width"] / width_fraction,
        group_layout["plot_area_height"] / plot_fraction,
        forward=True,
    )
    legend_y = _legend_position(
        context.plot_section,
        plot_top=plot_top,
        plot_fraction=plot_fraction,
    )
    if legend_axis is not None and show_legend:
        _attach_group_legend(
            figure,
            legend_axis,
            context.legend_title,
            context.legend_key,
            context.plot_section,
            legend_y,
            context.legend_kwargs,
            plot_left,
            plot_right,
            plot_top,
            plot_bottom,
        )
    else:
        figure.subplots_adjust(
            left=plot_left,
            right=plot_right,
            top=plot_top,
            bottom=plot_bottom,
            wspace=0.45,
            hspace=0.45,
        )
    output_path = context.plot_dir / f"{group_name}.pdf"
    raster_dpis = [
        get_plot_definition(instance["plot_entry"].get("type") or "line").raster_dpi
        for instance in plot_instances
    ]
    save_figure(figure, output_path, dpi=max((dpi for dpi in raster_dpis if dpi), default=None))
    print(f"Saved plot to {output_path}")
    if context.show and "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(figure)
    return output_path


def run_central_plots(plot_config: dict, *, show: bool = False) -> list[Path]:
    layout = plot_config.get("layout") or {}
    plot_section = plot_config.get("plot_section") or {}
    results_dir = Path(plot_config.get("results_dir") or "")
    records, legend_key, legend_title = _load_and_sort_records(results_dir, plot_section)
    plots = layout.get("plots") or []
    if not plots:
        raise ValueError("No plot layout entries found.")
    max_per_row = max(1, int(layout.get("max_plots_per_row") or len(plots)))
    apply_pdf_style(scale=1.0 + 0.1 * max(0, min(max_per_row, 4) - 1))
    plot_dir, base_name = _resolve_plot_dir(plot_config, results_dir, layout)
    legend_kwargs = _legend_kwargs(plot_section)
    context = PlotServiceContext(
        results_dir=results_dir,
        records=records,
        legend_key=legend_key,
        legend_title=legend_title,
        plot_section=plot_section,
        layout=layout,
        plot_dir=plot_dir,
        max_per_row=max_per_row,
        legend_kwargs=legend_kwargs,
        show=show,
    )
    groups = _group_plot_instances(
        plots,
        plot_section=plot_section,
        records=records,
        legend_key=legend_key,
        legend_title=legend_title,
        base_name=base_name,
    )
    outputs = [
        _render_plot_group(group_name, instances, context)
        for group_name, instances in groups.items()
    ]
    legends = layout.get("legends") or []
    if legends:
        outputs.extend(
            _render_standalone_legends(
                legends,
                records,
                legend_key,
                legend_title,
                plot_section,
                layout,
                base_name,
                plot_dir,
                legend_kwargs,
            )
        )
    return outputs
