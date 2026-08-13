from __future__ import annotations

from typing import Any, Sequence

import numpy as np

PREFERRED_LEGEND_KEYS = (
    "algorithm.name",
    "objective.name",
    "algorithm.params.regularization.alpha",
    "algorithm.params.regularization.lambda_",
    "algorithm.params.z_optimizer.name",
    "algorithm.params.z_optimizer.params.learning_rate",
    "algorithm.params.phi_update.learning_rate",
    "link.kind",
    "link.sigma",
    "data.source",
    "data.noise_std",
    "data.shape.n_observations",
    "data.shape.rank",
)

LEGEND_DISPLAY_NAMES = {
    "m_over_max_n_times_r": "M/(max(n,T)*r)",
    "n_r": "(n, r)",
    "algorithm.params.regularization.alpha": "alpha",
    "algorithm.params.regularization.lambda_": "lambda",
    "algorithm.params.z_optimizer.name": "Z optimizer",
    "algorithm.params.z_optimizer.params.learning_rate": "Z learning rate",
    "algorithm.params.phi_update.learning_rate": "phi learning rate",
    "data.noise_std": "$\\sigma$",
    "phi_diff@0": "R",
    "run.target_goodness": "target_goodness",
    "phi_star_label": "functions",
}


def _experiment_value(record: dict, path: str) -> Any:
    sweep_parameters = record.get("sweep_parameters") or {}
    if path in sweep_parameters:
        return sweep_parameters[path]
    value: Any = record.get("experiment_spec") or {}
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _experiment_leaf_paths(record: dict) -> set[str]:
    excluded = {
        "schema_version",
        "experiment",
        "trials",
        "seed",
        "runtime",
        "sweep",
        "diagnostics",
        "plot",
    }
    paths: set[str] = set()

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                visit(child, path)
        elif isinstance(value, (bool, int, float, str)):
            paths.add(prefix)

    spec = record.get("experiment_spec") or {}
    for key, value in spec.items():
        if key not in excluded:
            visit(value, key)
    paths.update(str(key) for key in (record.get("sweep_parameters") or {}))
    return paths


def _legend_key_varies(records: Sequence[dict], key: str) -> bool:
    values = {
        value
        for record in records
        if isinstance((value := _experiment_value(record, key)), (int, float, str))
    }
    return len(values) > 1


def infer_legend_key(records: Sequence[dict], plot_section: dict) -> str | None:
    explicit = plot_section.get("legend_key")
    if isinstance(explicit, str) and explicit:
        return explicit
    record_keys = {key for record in records for key in _experiment_leaf_paths(record)}
    preferred = [key for key in PREFERRED_LEGEND_KEYS if key in record_keys]
    remaining = sorted(record_keys.difference(preferred))
    return next(
        (key for key in [*preferred, *remaining] if _legend_key_varies(records, key)),
        None,
    )


def legend_display_name(legend_key: str | None) -> str | None:
    if not legend_key:
        return None
    return LEGEND_DISPLAY_NAMES.get(legend_key, legend_key)


def parse_diagnostics_key(legend_key: str) -> tuple[str, int] | None:
    if "@" not in legend_key:
        return None
    metric, _, iter_text = legend_key.rpartition("@")
    if not metric:
        return None
    try:
        iteration = int(iter_text)
    except ValueError:
        return None
    return metric, iteration


def diagnostic_value_at_iter(
    record: dict,
    metric: str,
    iteration: int,
) -> float | None:
    diagnostics = record.get("diagnostics") or []
    for entry in diagnostics:
        if entry["iteration"] == iteration and entry.get(metric) is not None:
            return float(entry[metric])
    return None


def _integer_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _shape_for_record(record: dict) -> dict:
    spec = record.get("experiment_spec") or {}
    return (spec.get("data") or {}).get("shape") or {}


def _phi_star_label(record: dict) -> str | None:
    index = _integer_or_none(record.get("__legend_index"))
    if index is None:
        seed = _integer_or_none(_experiment_value(record, "seed"))
        index = seed + 1 if seed is not None else None
    return rf"$\phi_{{{index}}}^*$" if index is not None else None


def _shape_rank_label(record: dict) -> str | None:
    shape = _shape_for_record(record)
    values = [_integer_or_none(shape.get(key)) for key in ("n_rows", "n_cols", "rank")]
    if any(value is None for value in values):
        return None
    n_rows, n_cols, rank = values
    if n_rows == n_cols:
        return f"n={n_rows}, rank={rank}"
    return f"shape=({n_rows},{n_cols}), rank={rank}"


def _sampling_ratio(record: dict) -> float | None:
    shape = _shape_for_record(record)
    observations = shape.get("n_observations")
    n_rows = shape.get("n_rows")
    n_cols = shape.get("n_cols")
    rank = shape.get("rank")
    if any(value is None for value in (observations, n_rows, n_cols, rank)):
        return None
    assert observations is not None and n_rows is not None
    assert n_cols is not None and rank is not None
    denominator = max(float(n_rows), float(n_cols)) * float(rank)
    return float(observations) / denominator if denominator else None


def derived_legend_value(record: dict, legend_key: str) -> Any:
    if legend_key == "phi_star_label":
        return _phi_star_label(record)
    if legend_key == "n_r":
        return _shape_rank_label(record)
    if legend_key == "m_over_max_n_times_r":
        return _sampling_ratio(record)
    metric_at = parse_diagnostics_key(legend_key)
    if metric_at is None:
        return None
    metric, iteration = metric_at
    return diagnostic_value_at_iter(record, metric, iteration)


def legend_value_for_record(record: dict, legend_key: str | None) -> Any:
    if not legend_key:
        return None
    value = _experiment_value(record, legend_key)
    if value is None:
        value = derived_legend_value(record, legend_key)
    return value


def axis_value_for_record(record: dict, key: str) -> Any:
    value = _experiment_value(record, key)
    if value is None:
        value = derived_legend_value(record, key)
    return value


def legend_sort_key(record: dict, legend_key: str | None) -> tuple[int, object]:
    value = legend_value_for_record(record, legend_key)
    if value is None:
        return (2, "")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (0, float(value))
    return (1, str(value))


def legend_label(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return f"{value:.3g}"
    return str(value)


def legend_entry_sort_key(label: str) -> tuple[int, object]:
    try:
        return (0, float(label))
    except (TypeError, ValueError):
        return (1, label)


def legend_row_major_order(
    entries: list[tuple[Any, str]],
    *,
    ncol: int,
) -> list[tuple[Any, str]]:
    if ncol <= 1:
        return entries
    total = len(entries)
    nrow = int(np.ceil(total / ncol))
    grid: list[list[tuple[Any, str] | None]] = [[None] * ncol for _ in range(nrow)]
    idx = 0
    for row in range(nrow):
        for col in range(ncol):
            if idx >= total:
                break
            grid[row][col] = entries[idx]
            idx += 1
    ordered: list[tuple[Any, str]] = []
    for col in range(ncol):
        for row in range(nrow):
            entry = grid[row][col]
            if entry is not None:
                ordered.append(entry)
    return ordered


def legend_fallback_label(record: dict, *, index: int) -> str:
    return record.get("__record_name") or f"run_{index + 1}"
