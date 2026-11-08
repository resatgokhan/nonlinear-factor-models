"""Recompute all summaries from saved daily scores; no model reruns required."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from bcd_lab.atomic_io import atomic_text_writer
from bcd_lab.finance.artifacts import read_arrays, read_json
from bcd_lab.finance.evaluation import performance


def summarize(output: str | Path) -> list[dict[str, Any]]:
    output = Path(output)
    manifest = read_json(output / "run.json")
    if manifest["status"] != "complete":
        raise ValueError("Cannot summarize an incomplete finance run")
    expected = {
        (window["period"], name)
        for window in manifest["windows"]
        for name in manifest["config"]["methods"]
    }
    seen: set[tuple[int, str]] = set()
    grouped: dict[str, list[tuple[dict[str, Any], np.ndarray]]] = {}
    for relative in manifest["records"]:
        path = output / relative
        record = read_json(path)
        key = (record["window"]["period"], record["method"])
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate finance record")
        seen.add(key)
        arrays = read_arrays(path.parent / record["arrays"])
        if len(arrays["daily_excess_returns"]) != len(record["window"]["test_dates"]):
            raise ValueError("Daily scores must align with test dates")
        grouped.setdefault(record["method"], []).append((record, arrays["daily_excess_returns"]))
    if seen != expected:
        raise ValueError("Missing finance records for the configured experiment")
    rows = []
    for name, entries in grouped.items():
        entries.sort(key=lambda item: item[0]["window"]["period"])
        dates = [date for record, _ in entries for date in record["window"]["test_dates"]]
        if len(dates) != len(set(dates)):
            raise ValueError("Duplicate evaluation dates would double-count observations")
        daily = np.concatenate([values for _, values in entries])
        metrics = performance(daily, trading_days=manifest["config"]["evaluation"]["trading_days"])
        rows.append(
            {
                "n_assets": entries[0][0]["n_assets"],
                "estimator": name,
                **metrics,
                "windows": len(entries),
                "fit_seconds": sum(record["fit_seconds"] for record, _ in entries),
                "covariance_seconds": sum(record["covariance_seconds"] for record, _ in entries),
            }
        )
    if not rows:
        raise ValueError("No completed finance records")
    return rows


def write_report(output: str | Path, *, plot: bool = True) -> list[dict[str, Any]]:
    output = Path(output)
    rows = summarize(output)
    with atomic_text_writer(output / "summary.csv") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
        labels = [row["estimator"] for row in rows]
        for ax, metric, unit in zip(axes, ("AV", "SD", "SR"), ("%", "%", ""), strict=True):
            ax.bar(labels, [np.nan if row[metric] is None else row[metric] for row in rows])
            ax.set_title(f"{metric} {unit}".strip())
            ax.tick_params(axis="x", labelrotation=35)
            ax.axhline(0, color="black", linewidth=0.6)
        figure.suptitle("Pooled out-of-sample scores · weighted log returns")
        figure.savefig(output / "metrics.pdf")
        figure.savefig(output / "metrics.png", dpi=150)
        plt.close(figure)
    return rows
