"""Loader for canonical Schema v2 YAML with component-scoped defaults."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from bcd_lab.randomness import validate_seed
from bcd_lab.types import SelectionPolicy

from .schema import (
    ComponentSpec,
    ExperimentSpec,
    ObjectiveSpec,
    SweepAxis,
    SweepSection,
)
from .validation import validate_experiment_invariants

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "experiment",
    "algorithm",
    "objective",
    "trials",
    "seed",
    "data",
    "link",
    "runtime",
    "initialization",
    "diagnostics",
    "selection",
    "sweep",
    "plot",
}


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in overrides.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _merge_selected_defaults(
    defaults: dict[str, Any],
    overrides: dict[str, Any],
    *,
    selector: str,
) -> dict[str, Any]:
    """Merge defaults only while the selected component remains the default."""
    default_selection = defaults.get(selector)
    selected = overrides.get(selector, default_selection)
    if selected != default_selection:
        return dict(overrides)
    return deep_merge(defaults, overrides)


def load_default_block(name: str) -> dict[str, Any]:
    default_resource = files(__package__).joinpath("default_configs").joinpath(f"{name}.yaml")
    if not default_resource.is_file():
        return {}
    with default_resource.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_seed(data: Mapping[str, Any]) -> int:
    if "seed" not in data:
        raise ValueError("seed is required and must be one non-negative integer")
    return validate_seed(data["seed"])


def _section(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _parse_sweep(data: Mapping[str, Any]) -> SweepSection:
    sweep = _section(data, "sweep")
    axes_data = sweep.get("axes", [])
    if not isinstance(axes_data, list):
        raise ValueError("sweep.axes must be a list")
    axes: list[SweepAxis] = []
    for index, axis_value in enumerate(axes_data):
        label = f"sweep.axes[{index}]"
        if not isinstance(axis_value, Mapping):
            raise ValueError(f"{label} must be a mapping")
        path = axis_value.get("path")
        values = axis_value.get("values")
        if not isinstance(path, str) or not path:
            raise ValueError(f"{label}.path must be a non-empty string")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{label}.values must be a non-empty list")
        axes.append(SweepAxis(path=path, values=list(values)))
    return SweepSection(axes=axes)


def _normalize_diagnostics(overrides: dict[str, Any]) -> dict[str, Any]:
    defaults = load_default_block("diagnostics")
    diagnostics = dict(defaults)
    diagnostics.update(overrides)
    if set(diagnostics) != {"on_error", "families"}:
        raise ValueError("diagnostics must contain exactly 'on_error' and 'families'")
    if diagnostics["on_error"] not in {"raise", "record"}:
        raise ValueError("diagnostics.on_error must be 'raise' or 'record'")
    families = diagnostics["families"]
    if not isinstance(families, Mapping):
        raise ValueError("diagnostics.families must be a mapping")
    for family, metrics in families.items():
        if not isinstance(family, str) or not family:
            raise ValueError("diagnostic family names must be non-empty strings")
        valid_metrics = metrics is None or (
            isinstance(metrics, list)
            and all(isinstance(metric, str) and metric for metric in metrics)
        )
        if not valid_metrics:
            raise ValueError(
                f"diagnostics.families.{family} must be null or a list of metric names"
            )
    return diagnostics


def _parse_selection(data: Mapping[str, Any]) -> SelectionPolicy | None:
    value = data.get("selection")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("selection must be a mapping")
    unknown = sorted(set(value) - {"metric", "mode", "retain_checkpoint"})
    if unknown:
        raise ValueError("unknown selection field(s): " + ", ".join(unknown))
    if "metric" not in value:
        raise ValueError("selection.metric is required")
    return SelectionPolicy(
        metric=value["metric"],
        mode=value.get("mode", "min"),
        retain_checkpoint=value.get("retain_checkpoint", True),
    )


def _resolve_relative_path(value: Any, *, label: str, base_dir: Path) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    resolved = Path(value).expanduser()
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    return str(resolved.resolve())


def _resolve_config_paths(
    data_config: dict[str, Any],
    plot_config: dict[str, Any],
    base_dir: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if base_dir is None:
        return data_config, plot_config
    resolved_data = dict(data_config)
    if "path" in resolved_data:
        resolved_data["path"] = _resolve_relative_path(
            resolved_data["path"],
            label="data.path",
            base_dir=base_dir,
        )
    resolved_plot = dict(plot_config)
    if "dir" in resolved_plot and resolved_plot["dir"] is not None:
        resolved_plot["dir"] = _resolve_relative_path(
            resolved_plot["dir"],
            label="plot.dir",
            base_dir=base_dir,
        )
    return resolved_data, resolved_plot


def parse_experiment_spec(
    data: Any,
    *,
    source: str = "experiment mapping",
    base_dir: Path | None = None,
) -> ExperimentSpec:
    """Validate a mapping and construct the canonical experiment specification."""
    if not isinstance(data, Mapping):
        raise ValueError(f"{source} must contain a top-level mapping.")
    unknown_fields = sorted(set(data) - _TOP_LEVEL_FIELDS)
    if unknown_fields:
        names = ", ".join(unknown_fields)
        raise ValueError(f"Unknown top-level experiment fields: {names}")

    schema_version = data.get("schema_version", 0)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("schema_version must be the integer 2")
    if schema_version != 2:
        raise ValueError(f"Unsupported schema_version: {schema_version}. Expected: 2")

    experiment = data.get("experiment")
    if not isinstance(experiment, str) or not experiment.strip():
        raise ValueError("experiment must be a non-empty string")
    seed = _parse_seed(data)
    sweep = _parse_sweep(data)

    # Component defaults apply only to their own selected implementation.
    data_cfg = _merge_selected_defaults(
        load_default_block("data"),
        _section(data, "data"),
        selector="source",
    )
    link_cfg = _merge_selected_defaults(
        load_default_block("link"),
        _section(data, "link"),
        selector="kind",
    )
    runtime_cfg = deep_merge(load_default_block("runtime"), _section(data, "runtime"))
    init_cfg = _merge_selected_defaults(
        load_default_block("initialization"),
        _section(data, "initialization"),
        selector="mode",
    )
    diag_cfg = _normalize_diagnostics(_section(data, "diagnostics"))
    selection = _parse_selection(data)
    plot_data = _section(data, "plot")
    data_cfg, plot_cfg = _resolve_config_paths(
        data_cfg,
        plot_data,
        base_dir,
    )

    objective_data = data.get("objective")
    if not isinstance(objective_data, Mapping):
        raise ValueError("objective must be a mapping with 'name' and optional 'params'")
    if "name" not in objective_data:
        raise ValueError("objective.name is required")
    objective_name = objective_data["name"]
    if not isinstance(objective_name, str) or not objective_name.strip():
        raise ValueError("objective.name must be a non-empty string")
    objective_params = objective_data.get("params", {})
    if not isinstance(objective_params, Mapping):
        raise ValueError("objective.params must be a mapping")
    objective = ObjectiveSpec(
        name=objective_name,
        params=dict(objective_params),
    )

    algorithm_data = data.get("algorithm")
    if not isinstance(algorithm_data, Mapping):
        raise ValueError("algorithm must be a mapping with 'name' and optional 'params'")
    if "name" not in algorithm_data:
        raise ValueError("algorithm.name is required")
    algorithm_name = algorithm_data["name"]
    if not isinstance(algorithm_name, str) or not algorithm_name.strip():
        raise ValueError("algorithm.name must be a non-empty string")
    algorithm_params = algorithm_data.get("params", {})
    if not isinstance(algorithm_params, Mapping):
        raise ValueError("algorithm.params must be a mapping")
    algorithm = ComponentSpec(
        name=algorithm_name,
        params=deep_merge(
            load_default_block(algorithm_name),
            dict(algorithm_params),
        ),
    )

    spec = ExperimentSpec(
        schema_version=schema_version,
        experiment=experiment,
        algorithm=algorithm,
        objective=objective,
        trials=data.get("trials", 1),
        seed=seed,
        data=data_cfg,
        link=link_cfg,
        runtime=runtime_cfg,
        sweep=sweep,
        initialization=init_cfg,
        diagnostics=diag_cfg,
        selection=selection,
        plot=plot_cfg,
    )
    validate_experiment_invariants(spec)
    return spec


def load_sweep_config(path: str | Path) -> ExperimentSpec:
    yaml_path = Path(path).expanduser().resolve()
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return parse_experiment_spec(
        data,
        source=f"YAML file {yaml_path}",
        base_dir=yaml_path.parent,
    )
