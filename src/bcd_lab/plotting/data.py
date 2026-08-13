"""Result-record loading and figure-independent series selection services."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .plot_config import legend_value_for_record


def _load_records(results_dir: Path) -> list[dict]:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    paths = sorted(results_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON results found in {results_dir}")
    records: list[dict] = []
    for path in paths:
        with path.open("r") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            record_name = path.stem
            index = len(records) + 1
            payload["__record_name"] = record_name
            payload["__legend_index"] = index
            records.append(payload)
    if not records:
        raise ValueError(f"No valid JSON payloads found in {results_dir}")
    return records


def _running_mean_series(
    pairs: list[tuple[int, float]],
) -> tuple[np.ndarray, np.ndarray] | None:
    if not pairs:
        return None
    pairs.sort(key=lambda item: item[0])
    iterations, values = zip(*pairs, strict=True)
    values_array = np.asarray(values, dtype=float)
    running = np.cumsum(values_array) / np.arange(1, values_array.size + 1, dtype=float)
    return np.asarray(iterations, dtype=int), running


def _regret_over_loss_series(
    diagnostics: Iterable[dict],
) -> tuple[np.ndarray, np.ndarray] | None:
    pairs: list[tuple[int, float]] = []
    for entry in diagnostics:
        loss = entry.get("loss")
        regret = entry.get("regret")
        if loss is None or regret is None or float(regret) < 0:
            continue
        loss_value = float(loss)
        if loss_value == 0.0:
            continue
        ratio = float(regret) / loss_value
        if np.isfinite(ratio):
            pairs.append((int(entry["iteration"]), ratio))
    return _running_mean_series(pairs)


def _regret_running_mean_series(
    diagnostics: Iterable[dict],
) -> tuple[np.ndarray, np.ndarray] | None:
    pairs = [
        (int(entry["iteration"]), float(entry["regret"]))
        for entry in diagnostics
        if entry.get("regret") is not None and float(entry["regret"]) >= 0
    ]
    return _running_mean_series(pairs)


def _direct_metric_series(
    diagnostics: Iterable[dict], metric: str
) -> tuple[np.ndarray, np.ndarray] | None:
    pairs = [
        (int(entry["iteration"]), float(entry[metric]))
        for entry in diagnostics
        if entry.get(metric) is not None and not (metric == "regret" and float(entry[metric]) < 0)
    ]
    if not pairs:
        return None
    iterations, values = zip(*pairs, strict=True)
    return np.asarray(iterations, dtype=int), np.asarray(values, dtype=float)


def metric_series(diagnostics: Iterable[dict], metric: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Transform diagnostic records into the canonical series for a metric."""
    entries = list(diagnostics)
    if metric == "regret_over_loss":
        return _regret_over_loss_series(entries)
    if metric in {"regret_avg", "regret_running_mean"}:
        return _regret_running_mean_series(entries)
    return _direct_metric_series(entries, metric)


def _metric_series_for_record(record: dict, metric: str) -> tuple[np.ndarray, np.ndarray] | None:
    return metric_series(record.get("diagnostics") or [], metric)


def _pick_random_trials(
    records: Sequence[dict],
    *,
    legend_key: str | None,
    seed: int,
) -> list[dict]:
    if not records:
        return []
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("random trial selection requires an explicit integer seed")
    rng = np.random.default_rng(seed)
    if not legend_key:
        return [records[int(rng.integers(len(records)))]]
    grouped: dict[object, list[dict]] = {}
    for record in records:
        key = legend_value_for_record(record, legend_key)
        grouped.setdefault(key, []).append(record)
    selected: list[dict] = []
    for key in sorted(grouped.keys(), key=lambda k: (k is None, k)):
        group = grouped[key]
        selected.append(group[int(rng.integers(len(group)))])
    return selected


def _pick_longest_stable_trials(
    records: Sequence[dict],
    *,
    legend_key: str | None,
    metric: str,
) -> list[dict]:
    if not records:
        return []
    if not legend_key:
        return [_pick_longest_stable_trials(records, legend_key="__all__", metric=metric)[0]]
    grouped: dict[object, list[dict]] = {}
    for record in records:
        key = legend_value_for_record(record, legend_key)
        grouped.setdefault(key, []).append(record)
    selected: list[dict] = []
    for key in sorted(grouped.keys(), key=lambda k: (k is None, k)):
        best = None
        best_score = (-1, -1)
        for record in grouped[key]:
            series = _metric_series_for_record(record, metric)
            if series is None:
                score = (-1, -1)
            else:
                iters, vals = series
                finite = np.isfinite(vals)
                if not np.any(finite):
                    score = (-1, -1)
                else:
                    last_iter = int(iters[finite][-1])
                    score = (last_iter, int(np.sum(finite)))
            if score > best_score:
                best_score = score
                best = record
        if best is not None:
            selected.append(best)
    return selected
