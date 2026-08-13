"""Plotting helper package for BCD experiments."""

from __future__ import annotations

from . import mpl_config as mpl_config
from .options import (
    choice,
    normalize_fields,
    optional_bool,
    optional_float,
    optional_int,
    optional_string,
    positive_iterations,
    string_list,
)
from .plot import run_plots
from .registry import (
    PlotRendererDefinition,
    get_plot_definition,
    list_plot_types,
    register_plot_type,
)
from .renderers.heatmap import _heatmap_handler
from .renderers.image import _image_handler
from .renderers.iteration_table import _iteration_table_handler
from .renderers.line import _line_handler
from .renderers.phi_circle import _phi_circle_handler
from .renderers.selected_link import _selected_link_handler, selected_link_options
from .renderers.summary_line import _summary_line_handler
from .renderers.table import _table_handler

# Built-ins are wired here in one visible, deterministic order. Each definition
# declares its YAML options and the layout capabilities consumed by the service.
register_plot_type(
    PlotRendererDefinition(
        "image",
        _image_handler,
        normalize_fields({"path": optional_string}),
        raster_dpi=300,
    )
)
register_plot_type(
    PlotRendererDefinition(
        "selected_link",
        _selected_link_handler,
        selected_link_options,
        supports_overlay=True,
        legend_policy="overlay",
    )
)
register_plot_type(
    PlotRendererDefinition(
        "heatmap",
        _heatmap_handler,
        normalize_fields(
            {
                "x": optional_string,
                "y": optional_string,
                "y_scale": choice("log", "linear"),
                "metric": optional_string,
                "x_tick_mode": choice("endpoints"),
                "y_tick_mode": choice("endpoints"),
            }
        ),
    )
)
register_plot_type(
    PlotRendererDefinition(
        "summary_line",
        _summary_line_handler,
        normalize_fields(
            {
                "x": optional_string,
                "y": optional_string,
                "x_scale": choice("log", "linear"),
                "y_scale": choice("log", "linear"),
                "line_style": choice("solid", "none"),
                "fit_line": optional_int(positive=True),
                "normalize_by_first": optional_bool,
                "normalize_eps": optional_float(positive=True),
                "summary_stat": choice("min", "final", "mean"),
                "normalize_by_baseline": optional_bool,
                "baseline_key": optional_string,
                "baseline_value": optional_float(),
                "show_trials": optional_bool,
                "show_mean": optional_bool,
                "error": choice("ci95", "sem", "std"),
                "error_style": choice("band", "bars"),
            }
        ),
        supports_overlay=True,
        legend_policy="overlay_or_grouped",
    )
)
register_plot_type(
    PlotRendererDefinition(
        "table",
        _table_handler,
        normalize_fields({"x": optional_string, "y": optional_string, "stats": string_list}),
    )
)
register_plot_type(
    PlotRendererDefinition(
        "iteration_table",
        _iteration_table_handler,
        normalize_fields(
            {
                "y": optional_string,
                "iteration_mode": choice("log", "linear"),
                "iteration_count": optional_int(positive=True),
                "iteration_max": optional_int(positive=True),
                "iterations": positive_iterations,
            }
        ),
    )
)
register_plot_type(
    PlotRendererDefinition(
        "phi_circle",
        _phi_circle_handler,
        supports_overlay=True,
        legend_policy="overlay",
    )
)
register_plot_type(
    PlotRendererDefinition(
        "line",
        _line_handler,
        normalize_fields(
            {
                "x": optional_string,
                "y": optional_string,
                "x_scale": choice("log", "linear"),
                "y_scale": choice("log", "linear"),
                "line_style": choice("solid", "none"),
                "fit_line": optional_int(positive=True),
                "trial_pick": choice("random", "longest_stable"),
                "trial_seed": optional_int(),
                "show_trials": optional_bool,
                "show_mean": optional_bool,
                "error": choice("ci95", "sem", "std"),
                "error_style": choice("band", "bars"),
                "trial_stat": choice("mean", "median"),
                "trial_error": choice("ci95", "sem", "std"),
                "trial_grid": choice("log", "linear"),
                "trial_grid_count": optional_int(positive=True),
                "trial_grid_max": optional_int(positive=True),
                "iteration_cutoff": optional_int(positive=True),
                "normalize_by_first": optional_bool,
                "normalize_eps": optional_float(positive=True),
            }
        ),
        supports_overlay=True,
        legend_policy="overlay_or_grouped",
    )
)

__all__ = [
    "PlotRendererDefinition",
    "get_plot_definition",
    "list_plot_types",
    "register_plot_type",
    "run_plots",
]
