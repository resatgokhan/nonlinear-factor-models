"""Registry and public extension contract for plot renderers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

from bcd_lab._registry import ComponentRegistry

PlotRenderer = Callable[..., None]
PlotOptionNormalizer = Callable[[Mapping[str, Any], str], dict[str, Any]]
LegendPolicy = Literal["none", "overlay", "grouped", "overlay_or_grouped"]


def _no_options(spec: Mapping[str, Any], label: str) -> dict[str, Any]:
    del spec, label
    return {}


@dataclass(frozen=True)
class PlotRendererDefinition:
    """Everything the plotting service needs to use one renderer."""

    name: str
    render: PlotRenderer
    normalize_options: PlotOptionNormalizer = _no_options
    supports_overlay: bool = False
    legend_policy: LegendPolicy = "none"
    raster_dpi: int | None = None


_PLOT_REGISTRY = ComponentRegistry[PlotRendererDefinition]("Plot type", "plot types")


def register_plot_type(definition: PlotRendererDefinition) -> PlotRendererDefinition:
    """Register one complete, duplicate-safe renderer definition."""
    return _PLOT_REGISTRY.register(definition.name, definition)


def get_plot_definition(name: str) -> PlotRendererDefinition:
    """Resolve a plot renderer definition by its public type name."""
    return _PLOT_REGISTRY.get(name)


def list_plot_types() -> tuple[str, ...]:
    """Return all registered plot type names."""
    return _PLOT_REGISTRY.names()
