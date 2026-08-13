"""Learned-link renderer for validation-selected model exports."""

from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from ..plot_config import legend_fallback_label, legend_label, legend_value_for_record


def selected_link_options(spec: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Validate renderer-owned selected-link options from public YAML."""
    show_identity = spec.get("show_identity", False)
    if not isinstance(show_identity, bool):
        raise ValueError(f"{label}.show_identity must be a boolean")
    return {"show_identity": show_identity}


def _selected_curve(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    selection = record.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("selected_link requires a result selection payload")
    exports = selection.get("exports")
    if not isinstance(exports, Mapping):
        raise ValueError("selected_link requires selection.exports")
    curve = exports.get("link_curve")
    if not isinstance(curve, Mapping):
        raise ValueError("selected_link requires selection.exports.link_curve")
    inputs = np.asarray(curve.get("inputs"), dtype=float)
    outputs = np.asarray(curve.get("outputs"), dtype=float)
    if inputs.ndim != 1 or outputs.ndim != 1 or inputs.shape != outputs.shape:
        raise ValueError("selected link grid and values must be aligned vectors")
    if inputs.size < 2 or not np.all(np.diff(inputs) > 0.0):
        raise ValueError("selected link grid must be strictly increasing")
    if not np.all(np.isfinite(inputs)) or not np.all(np.isfinite(outputs)):
        raise ValueError("selected link curve must be finite")
    return inputs, outputs


def _selected_link_handler(ax: plt.Axes, instance: dict, context: dict) -> None:
    """Plot the function exported from each run's selected checkpoint."""
    plot_entry = instance["plot_entry"]
    combined_inputs: list[np.ndarray] = []
    for index, record in enumerate(instance["records"]):
        inputs, outputs = _selected_curve(record)
        combined_inputs.append(inputs)
        label = legend_label(
            legend_value_for_record(record, context["legend_key"]),
            legend_fallback_label(record, index=index),
        )
        ax.plot(inputs, outputs, linewidth=1.8, label=label)

    if plot_entry.get("show_identity") and combined_inputs:
        lower = min(float(values[0]) for values in combined_inputs)
        upper = max(float(values[-1]) for values in combined_inputs)
        ax.plot(
            [lower, upper],
            [lower, upper],
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="identity",
        )
    ax.set_title(plot_entry.get("title") or "Selected learned link")
    ax.set_xlabel(plot_entry.get("xlabel") or "Latent inner product")
    ax.set_ylabel(plot_entry.get("ylabel") or "Link output")
    ax.grid(True, linestyle="--", alpha=0.35)
