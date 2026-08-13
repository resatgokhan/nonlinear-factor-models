"""Unit tests for the experiment plotting helper package."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from bcd_lab.plotting import PlotRendererDefinition, register_plot_type
from bcd_lab.plotting.data import (
    _load_records,
    _pick_longest_stable_trials,
    _pick_random_trials,
    metric_series,
)
from bcd_lab.plotting.layout import _group_layout, _group_plot_instances
from bcd_lab.plotting.plot import (
    _load_plot_section,
    _parse_plot_layout,
    _resolve_plot_output_dir,
    _resolve_yaml_path,
    run_plots,
)
from bcd_lab.plotting.plot_util import _resolve_legend_label


def test_resolve_yaml_path() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        config_dir = root / "configs"
        config_dir.mkdir()

        # 1. Existing YAML file by stem name
        yaml_file = config_dir / "my_experiment.yaml"
        yaml_file.touch()

        resolved = _resolve_yaml_path(root, "my_experiment")
        assert resolved.resolve() == yaml_file.resolve()

        # 2. Existing YML file by stem name
        yml_file = config_dir / "another_exp.yml"
        yml_file.touch()

        resolved = _resolve_yaml_path(root, "another_exp")
        assert resolved.resolve() == yml_file.resolve()

        # 3. Direct path (absolute or relative with subdirectory)
        sub_dir = root / "custom_configs"
        sub_dir.mkdir()
        custom_yaml = sub_dir / "custom.yaml"
        custom_yaml.touch()

        resolved = _resolve_yaml_path(root, "custom_configs/custom.yaml")
        assert resolved.resolve() == custom_yaml.resolve()


def test_load_plot_section() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        yaml_file = root / "exp.yaml"

        # 1. Valid plot section
        yaml_file.write_text("""
experiment: test
plot:
  enabled: true
  dir: ./my_plots
  plot1:
    type: line
    x: iteration
    y: loss
""")
        plot_section = _load_plot_section(yaml_file)
        assert plot_section.get("enabled") is True
        assert plot_section.get("dir") == "./my_plots"
        assert plot_section.get("plot1", {}).get("type") == "line"

        # 2. Missing plot section returns empty dict
        yaml_file.write_text("""
experiment: test
""")
        assert _load_plot_section(yaml_file) == {}
        # 3. Invalid top-level yaml type raises ValueError
        yaml_file.write_text("- item1\n- item2")
        with pytest.raises(ValueError, match="must contain a mapping"):
            _load_plot_section(yaml_file)

        # 4. Invalid plot section type raises ValueError
        yaml_file.write_text("""
experiment: test
plot: [1, 2, 3]
""")
        with pytest.raises(ValueError, match="plot section must be a mapping"):
            _load_plot_section(yaml_file)


@pytest.mark.parametrize("summary_stat", ["min", "final", "mean"])
def test_summary_line_accepts_artifact_summary_statistics(summary_stat: str) -> None:
    layout = _parse_plot_layout(
        {
            "plot1": {
                "type": "summary_line",
                "x": "data.noise_std",
                "y": "goodness",
                "summary_stat": summary_stat,
            }
        }
    )

    assert layout["plots"][0]["summary_stat"] == summary_stat


def test_iteration_legend_uses_nested_sweep_value() -> None:
    record = {
        "__record_name": "long-artifact-id",
        "sweep_parameters": {"data.noise_std": 0.05},
    }

    assert _resolve_legend_label(record, 0, "data.noise_std") == "0.05"


def test_resolve_plot_output_dir() -> None:
    # 1. Absolute dir returned as-is
    assert _resolve_plot_output_dir({"dir": "/abs/path"}, None) == "/abs/path"

    # 2. Relative dir resolved against config_path
    config_path = Path("/my/config/dir/exp.yaml")
    resolved = _resolve_plot_output_dir({"dir": "my_plots"}, config_path)
    assert resolved == "/my/config/dir/my_plots"

    # 3. None or missing dir returns None
    assert _resolve_plot_output_dir({}, None) is None

    # 4. Non-string dir raises ValueError
    with pytest.raises(ValueError, match="plot.dir must be a string or null"):
        _resolve_plot_output_dir({"dir": 123}, None)


def test_run_plots_honors_configured_dir_and_explicit_override(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config_dir = tmp_path / "arbitrary" / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "experiment.yaml"
    config_path.write_text(
        "plot:\n  enabled: true\n  dir: ../configured-plots\n",
        encoding="utf-8",
    )

    with patch("bcd_lab.plotting.plot.run_central_plots") as render:
        run_plots(root, yaml_name=str(config_path))
        configured = render.call_args.args[0]
        assert configured["plot_dir"] == str((config_dir.parent / "configured-plots").resolve())

        run_plots(root, yaml_name=str(config_path), plots_dir="override-plots")
        overridden = render.call_args.args[0]
        assert overridden["plot_dir"] == str((root / "override-plots").resolve())


def test_parse_plot_layout() -> None:
    # 1. Valid parsing
    spec = {
        "name": "My Figure",
        "plot_ratio": 1.5,
        "max_plots_per_row": 3,
        "width": 10.0,
        "height": 8.0,
        "font_scale": 1.2,
        "size_scale": 1.1,
        "plot1": {
            "type": "line",
            "x": "iteration",
            "y": "loss",
            "x_scale": "log",
            "y_scale": "linear",
            "title": "My Line Plot",
        },
        "legend1": {
            "pdf": "my_legend.pdf",
        },
    }
    layout = _parse_plot_layout(spec)
    assert layout["name"] == "My Figure"
    assert layout["plot_ratio"] == 1.5
    assert layout["max_plots_per_row"] == 3
    assert layout["width"] == 10.0
    assert layout["height"] == 8.0
    assert layout["font_scale"] == 1.2
    assert layout["size_scale"] == 1.1

    assert len(layout["plots"]) == 1
    p = layout["plots"][0]
    assert p["key"] == "plot1"
    assert p["type"] == "line"
    assert p["x"] == "iteration"
    assert p["y"] == "loss"
    assert p["x_scale"] == "log"
    assert p["y_scale"] == "linear"

    assert len(layout["legends"]) == 1
    assert layout["legends"][0]["key"] == "legend1"
    assert layout["legends"][0]["pdf"] == "my_legend.pdf"

    # 2. Invalid validation cases
    with pytest.raises(ValueError, match="plot.plot_ratio must be positive"):
        _parse_plot_layout({"plot_ratio": -0.5})

    with pytest.raises(ValueError, match="plot.width must be positive"):
        _parse_plot_layout({"width": 0})

    with pytest.raises(ValueError, match="plot1.type must be"):
        _parse_plot_layout({"plot1": {"type": "invalid_type"}})

    with pytest.raises(ValueError, match="plot1.x_scale must be"):
        _parse_plot_layout({"plot1": {"type": "line", "x_scale": "invalid_scale"}})

    with pytest.raises(ValueError, match="requires an explicit trial_seed"):
        _parse_plot_layout({"plot1": {"type": "line", "trial_pick": "random"}})

    seeded = _parse_plot_layout(
        {"plot1": {"type": "line", "trial_pick": "random", "trial_seed": 7}}
    )
    assert seeded["plots"][0]["trial_seed"] == 7


def test_random_trial_selection_requires_and_uses_explicit_seed() -> None:
    records = [{"id": index} for index in range(8)]
    first = _pick_random_trials(records, legend_key=None, seed=17)
    second = _pick_random_trials(records, legend_key=None, seed=17)
    assert first == second

    with pytest.raises(ValueError, match="explicit integer seed"):
        _pick_random_trials(records, legend_key=None, seed=None)  # type: ignore[arg-type]


def test_record_and_series_services_do_not_require_a_figure(tmp_path: Path) -> None:
    records_dir = tmp_path / "results"
    records_dir.mkdir()
    (records_dir / "short.json").write_text(
        '{"diagnostics": [{"iteration": 1, "loss": 4.0, "regret": 2.0}]}',
        encoding="utf-8",
    )
    (records_dir / "long.json").write_text(
        '{"diagnostics": ['
        '{"iteration": 1, "loss": 2.0, "regret": 1.0},'
        '{"iteration": 3, "loss": 4.0, "regret": 1.0}'
        "]}",
        encoding="utf-8",
    )

    records = _load_records(records_dir)
    assert [record["__record_name"] for record in records] == ["long", "short"]

    series = metric_series(records[0]["diagnostics"], "regret_over_loss")
    assert series is not None
    iterations, values = series
    assert iterations.tolist() == [1, 3]
    assert values.tolist() == pytest.approx([0.5, 0.375])

    selected = _pick_longest_stable_trials(
        records,
        legend_key=None,
        metric="loss",
    )
    assert [record["__record_name"] for record in selected] == ["long"]


def test_grouping_and_layout_services_do_not_require_a_figure() -> None:
    records = [{"__record_name": "one"}, {"__record_name": "two"}]
    plots = [
        {
            "key": "plot1",
            "type": "line",
            "x": "iteration",
            "overlay_sweep": False,
            "width": 2.5,
            "height": 1.5,
        }
    ]
    grouped = _group_plot_instances(
        plots,
        plot_section={"plot1": {}},
        records=records,
        legend_key=None,
        legend_title=None,
        base_name="experiment",
    )
    instances = grouped["experiment"]
    assert len(instances) == 2

    layout = _group_layout(
        instances,
        layout={},
        max_per_row=2,
        group_name="experiment",
    )
    assert layout["row_sizes"] == [2]
    assert layout["plot_area_width"] == pytest.approx(5.0)
    assert layout["plot_area_height"] == pytest.approx(1.5)


def test_external_renderer_runs_from_public_yaml_configuration(tmp_path: Path) -> None:
    rendered: list[tuple[str, str]] = []

    def audit_renderer(ax, instance, context) -> None:
        rendered.append((instance["plot_entry"]["key"], instance["plot_entry"]["message"]))
        ax.text(0.5, 0.5, str(len(context["records"])))

    def normalize_options(spec, label):
        message = spec.get("message")
        if not isinstance(message, str):
            raise ValueError(f"{label}.message must be a string")
        return {"message": message}

    register_plot_type(
        PlotRendererDefinition(
            name="ia05_audit_panel",
            render=audit_renderer,
            normalize_options=normalize_options,
            supports_overlay=True,
            legend_policy="overlay",
            raster_dpi=144,
        )
    )
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "run.json").write_text(
        '{"diagnostics": ['
        '{"iteration": 1, "loss": 2.0},'
        '{"iteration": 2, "loss": 1.0}'
        '], "experiment_spec": {"algorithm": {"name": "bcd"}}}',
        encoding="utf-8",
    )
    plots_dir = tmp_path / "plots"
    config_path = tmp_path / "custom_plot.yml"
    config_path.write_text(
        """plot:
  name: public_extension
  plot1:
    type: ia05_audit_panel
    message: from-yaml
""",
        encoding="utf-8",
    )
    run_plots(
        tmp_path,
        yaml_name=str(config_path),
        results_dir=results_dir,
        plots_dir=plots_dir,
    )
    assert rendered == [("plot1", "from-yaml")]
    assert (plots_dir / "public_extension.pdf").is_file()


def test_every_builtin_renderer_reaches_pdf_from_public_yaml(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for x_value in (1.0, 2.0):
        for y_value in (1.0, 2.0):
            loss = x_value + y_value
            record = {
                "experiment_spec": {"algorithm": {"name": "bcd"}},
                "sweep_parameters": {"x": x_value, "y": y_value},
                "diagnostics": [
                    {"iteration": 1, "loss": loss + 1.0},
                    {"iteration": 2, "loss": loss},
                ],
                "summary": {
                    "metrics": {
                        "loss": {
                            "min": loss,
                            "min_iteration": 2,
                            "final": loss,
                            "final_iteration": 2,
                        }
                    }
                },
                "selection": {
                    "metric": "loss",
                    "mode": "min",
                    "iteration": 2,
                    "value": loss,
                    "diagnostics": {"iteration": 2, "loss": loss},
                    "exports": {
                        "link_curve": {
                            "inputs": [-1.0, 0.0, 1.0],
                            "outputs": [0.0, 0.5, 1.0],
                        }
                    },
                },
            }
            (results_dir / f"run-{x_value}-{y_value}.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
    config_path = tmp_path / "all-renderers.yml"
    config_path.write_text(
        """plot:
  name: all_renderers
  max_plots_per_row: 3
  show_legend: false
  plot1: {type: line, x: iteration, y: loss}
  plot2: {type: summary_line, x: x, y: loss}
  plot3: {type: heatmap, x: x, y: y, metric: loss}
  plot4: {type: table, x: x, y: loss, stats: [min, final]}
  plot5: {type: iteration_table, y: loss, iterations: [1, 2]}
  plot6: {type: image, path: missing.png}
  plot7: {type: phi_circle}
  plot8: {type: selected_link, show_identity: true}
""",
        encoding="utf-8",
    )
    plots_dir = tmp_path / "plots"

    run_plots(
        tmp_path,
        yaml_name=str(config_path),
        results_dir=results_dir,
        plots_dir=plots_dir,
    )

    output = plots_dir / "all_renderers.pdf"
    assert output.is_file()
    assert output.stat().st_size > 0
