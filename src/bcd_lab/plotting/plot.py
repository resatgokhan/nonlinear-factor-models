"""Public plotting entrypoint for experiment configurations."""

from __future__ import annotations

from pathlib import Path

from .service import run_central_plots
from .spec import (
    _build_plot_config,
    _load_plot_section,
    _resolve_yaml_path,
)
from .spec import (
    _parse_plot_layout as _parse_plot_layout,
)
from .spec import (
    _resolve_plot_output_dir as _resolve_plot_output_dir,
)


def run_plots(
    project_root: Path | str,
    *,
    yaml_name: str | None = None,
    results_dir: Path | str | None = None,
    plots_dir: Path | str | None = None,
) -> None:
    """Execute plot scripts configured in a sweep YAML.

    Plotting is centralized and driven by the YAML plot section.
    """
    root = Path(project_root).resolve()
    if yaml_name is None:
        print(
            "Plot helper requires a YAML name (e.g., run_plots(..., yaml_name='effect_of_alpha'))."
        )
        return

    yaml_path = _resolve_yaml_path(root, yaml_name)
    if not yaml_path.exists():
        print(f"Plot config not found: {yaml_path}")
        return

    yaml_stem = yaml_path.stem
    if results_dir is None:
        results_dir = root / "results" / yaml_stem
    else:
        results_dir = Path(results_dir)
        if not results_dir.is_absolute():
            results_dir = root / results_dir

    explicit_plots_dir = None
    if plots_dir is not None:
        explicit_plots_dir = Path(plots_dir)
        if not explicit_plots_dir.is_absolute():
            explicit_plots_dir = root / explicit_plots_dir
        explicit_plots_dir = explicit_plots_dir.resolve()

    plot_section = _load_plot_section(yaml_path)
    if plot_section.get("enabled") is False:
        print(f"Plotting disabled in {yaml_path}")
        return

    plot_config = _build_plot_config(
        yaml_name=yaml_stem,
        results_dir=results_dir,
        config_path=yaml_path,
        plot_section=plot_section,
    )

    if explicit_plots_dir is not None:
        plot_config["plot_dir"] = str(explicit_plots_dir)
    elif plot_config["plot_dir"] is None:
        plot_config["plot_dir"] = str(root / "plots" / yaml_stem)

    run_central_plots(plot_config)
