"""Image plot renderer."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def _resolve_image_path(
    record: dict,
    *,
    path_template: str | None,
    results_dir: Path,
) -> Path | None:
    if not path_template:
        return None
    run_name = record.get("__record_name", "")
    resolved = path_template.replace("{run_name}", str(run_name))
    path = Path(resolved)
    if not path.is_absolute():
        record_dir = record.get("__results_dir")
        if record_dir:
            path = Path(record_dir) / path
        else:
            path = results_dir / path
    return path


def _plot_image(
    ax: plt.Axes,
    record: dict,
    *,
    path_template: str | None,
    results_dir: Path,
    title: str | None,
) -> None:
    path = _resolve_image_path(record, path_template=path_template, results_dir=results_dir)
    if path is None or not path.exists():
        ax.text(
            0.5,
            0.5,
            "image not found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.axis("off")
        if title:
            ax.set_title(title)
        return
    image = plt.imread(str(path))
    ax.imshow(image, interpolation="none")
    ax.axis("off")
    if title:
        ax.set_title(title)


def _image_handler(ax: plt.Axes, instance: dict, context: dict) -> None:
    plot_entry = instance["plot_entry"]
    full_spec = instance["full_spec"]
    path_template = plot_entry.get("path") or full_spec.get("path")
    record = instance["records"][0] if instance["records"] else {}
    _plot_image(
        ax,
        record,
        path_template=path_template,
        results_dir=context["results_dir"],
        title=plot_entry.get("title"),
    )
