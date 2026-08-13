"""Plot configuration loading, validation, and canonical normalization."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from .options import optional_float, optional_int, optional_string
from .registry import get_plot_definition, list_plot_types


def _resolve_yaml_path(project_root: Path, yaml_name: str) -> Path:
    candidate = Path(yaml_name)
    if candidate.suffix in {".yml", ".yaml"}:
        yaml_path = candidate
        if not yaml_path.is_absolute():
            direct_path = project_root / yaml_path
            if direct_path.exists() or candidate.parent != Path("."):
                return direct_path
            config_dir = project_root / "configs"
            return config_dir / yaml_path.name
        return yaml_path

    if candidate.is_absolute() or candidate.parent != Path("."):
        yaml_path = candidate.with_suffix(".yml")
        if not yaml_path.exists():
            yaml_path = candidate.with_suffix(".yaml")
        return yaml_path

    config_dir = project_root / "configs"
    yaml_path = config_dir / f"{yaml_name}.yml"
    if not yaml_path.exists():
        yaml_path = config_dir / f"{yaml_name}.yaml"
    return yaml_path


def _load_plot_section(yaml_path: Path) -> dict:
    with yaml_path.open("r") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{yaml_path} must contain a mapping at the top level")
    plot_section = config.get("plot") or {}
    if plot_section is None:
        return {}
    if not isinstance(plot_section, dict):
        raise ValueError(f"{yaml_path} plot section must be a mapping")
    return plot_section


def _resolve_plot_output_dir(plot_section: dict, config_path: Path | None) -> str | None:
    output_dir = plot_section.get("dir")
    if output_dir is None:
        return None
    if not isinstance(output_dir, str):
        raise ValueError("plot.dir must be a string or null")
    path = Path(output_dir)
    if not path.is_absolute() and config_path is not None:
        path = config_path.parent / path
    return str(path.resolve())


def _parse_single_plot_entry(
    key: str,
    spec: dict,
    valid_types: tuple[str, ...],
) -> dict:
    plot_type = spec.get("type") or "line"
    if plot_type not in valid_types:
        types_str = ", ".join(f"'{t}'" for t in sorted(valid_types))
        raise ValueError(f"{key}.type must be one of {types_str}")

    title = optional_string(spec.get("title"), f"{key}.title")
    plot_width = optional_float(positive=True)(spec.get("width"), f"{key}.width")
    plot_height = optional_float(positive=True)(spec.get("height"), f"{key}.height")
    plot_font_scale = optional_float(positive=True)(spec.get("font_scale"), f"{key}.font_scale")
    xlabel = optional_string(spec.get("xlabel"), f"{key}.xlabel")
    ylabel = optional_string(spec.get("ylabel"), f"{key}.ylabel")
    pdf_name = optional_string(spec.get("pdf"), f"{key}.pdf")

    overlay_sweep_raw = spec.get("overlay_sweep", True)
    if not isinstance(overlay_sweep_raw, bool):
        raise ValueError(f"{key}.overlay_sweep must be a boolean")
    overlay_sweep = overlay_sweep_raw

    index = int(str(key)[4:])
    normalized = {
        "key": str(key),
        "index": index,
        "type": plot_type,
        "title": title,
        "width": plot_width,
        "height": plot_height,
        "font_scale": plot_font_scale,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "pdf": pdf_name,
        "overlay_sweep": overlay_sweep,
    }
    normalized.update(get_plot_definition(plot_type).normalize_options(spec, key))
    return normalized


def _parse_plot_layout(plot_section: dict) -> dict:
    name = optional_string(plot_section.get("name"), "plot.name")
    plot_ratio = optional_float(positive=True)(plot_section.get("plot_ratio"), "plot.plot_ratio")
    max_plots_per_row = optional_int()(
        plot_section.get("max_plots_per_row"), "plot.max_plots_per_row"
    )
    fig_width = optional_float(positive=True)(plot_section.get("width"), "plot.width")
    fig_height = optional_float(positive=True)(plot_section.get("height"), "plot.height")
    font_scale = optional_float(positive=True)(plot_section.get("font_scale"), "plot.font_scale")
    size_scale = optional_float(positive=True)(plot_section.get("size_scale"), "plot.size_scale")

    plot_keys = [key for key in plot_section.keys() if re.fullmatch(r"plot\d+", str(key))]
    plot_keys.sort(key=lambda key: int(str(key)[4:]))
    legend_keys = [key for key in plot_section.keys() if re.fullmatch(r"legend\d+", str(key))]
    legend_keys.sort(key=lambda key: int(str(key)[6:]))

    plots: list[dict] = []
    global_trial_pick = plot_section.get("trial_pick")
    if global_trial_pick not in {None, "random", "longest_stable"}:
        raise ValueError("plot.trial_pick must be 'random', 'longest_stable', or null")
    global_trial_seed = optional_int()(plot_section.get("trial_seed"), "plot.trial_seed")
    valid_types = list_plot_types()
    for key in plot_keys:
        spec = plot_section.get(key)
        if spec is None:
            continue
        if not isinstance(spec, dict):
            raise ValueError(f"{key} must be a mapping")
        plot = _parse_single_plot_entry(key, spec, valid_types)
        effective_pick = plot.get("trial_pick") or global_trial_pick
        effective_seed = (
            plot.get("trial_seed") if plot.get("trial_seed") is not None else global_trial_seed
        )
        if effective_pick == "random" and effective_seed is None:
            raise ValueError(f"{key} trial_pick='random' requires an explicit trial_seed")
        plots.append(plot)

    legends: list[dict] = []
    for key in legend_keys:
        spec = plot_section.get(key)
        if spec is None:
            continue
        if not isinstance(spec, dict):
            raise ValueError(f"{key} must be a mapping")
        pdf_name = optional_string(spec.get("pdf"), f"{key}.pdf")
        index = int(str(key)[6:])
        legends.append(
            {
                "key": str(key),
                "index": index,
                "pdf": pdf_name,
            }
        )

    return {
        "name": name,
        "max_plots_per_row": max_plots_per_row,
        "width": fig_width,
        "height": fig_height,
        "plot_ratio": plot_ratio,
        "size_scale": size_scale,
        "font_scale": font_scale,
        "plots": plots,
        "legends": legends,
    }


def _build_plot_config(
    *,
    yaml_name: str,
    results_dir: Path,
    config_path: Path | None,
    plot_section: dict,
) -> dict:
    layout = _parse_plot_layout(plot_section)
    plot_dir = _resolve_plot_output_dir(plot_section, config_path)
    return {
        "yaml_name": yaml_name,
        "results_dir": str(results_dir),
        "config_path": str(config_path) if config_path else None,
        "plot_dir": plot_dir,
        "layout": layout,
        "plot_section": plot_section,
    }
