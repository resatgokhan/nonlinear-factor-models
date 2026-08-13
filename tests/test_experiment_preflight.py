"""Failure-before-side-effect tests for experiment preflight."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from bcd_lab.experiments.runner import run_sweep

from .test_experiment_composition import EVENTS, _base_config


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("trials", 0, "trials must be positive"),
        ("trials", "1", "trials must be an integer"),
        ("data.shape.n_rows", 0, "data.shape.n_rows must be positive"),
        ("data.shape.rank", 3, "data.shape.rank cannot exceed"),
        ("data.shape.n_observations", 5, "data.shape.n_observations cannot exceed"),
        ("runtime.max_iter", -1, "runtime.max_iter must be non-negative"),
        ("runtime.tol", -0.1, "runtime.tol must be a finite non-negative number"),
        ("runtime.diagnostics_interval", 0, "runtime.diagnostics_interval must be positive"),
        ("runtime.checkpoint_iters", [-1], "runtime.checkpoint_iters must be non-negative"),
        ("runtime.dtype", "int64", "runtime.dtype must be a supported floating dtype"),
        ("runtime", [], "runtime must be a mapping"),
        ("sweep", [], "sweep must be a mapping"),
        ("$swept_runtime", -1, "runtime.max_iter must be non-negative"),
    ],
)
def test_invalid_shared_configuration_fails_at_yaml_boundary(
    tmp_path: Path,
    path: str,
    value: Any,
    message: str,
) -> None:
    if path == "$swept_runtime":
        raw = _base_config({"path": "runtime.max_iter", "values": [value]})
    else:
        raw = _base_config({"path": "link.sigma", "values": [1.0]})
        target = raw
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    config_path = tmp_path / "invalid-shared.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    EVENTS.clear()

    with pytest.raises(ValueError, match=message):
        run_sweep(config_path, output_root=tmp_path / "results")

    assert EVENTS == []
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    "path",
    ["experiment", "schema_version", "sweep.axes", "plot.enabled", "selection.metric"],
)
def test_reserved_orchestration_paths_are_rejected(tmp_path: Path, path: str) -> None:
    config_path = tmp_path / "reserved.yaml"
    config_path.write_text(
        yaml.safe_dump(_base_config({"path": path, "values": ["ignored"]})),
        encoding="utf-8",
    )
    EVENTS.clear()

    with pytest.raises(ValueError, match="reserved orchestration field"):
        run_sweep(config_path, output_root=tmp_path / "results")

    assert EVENTS == []
    assert not (tmp_path / "results").exists()


def test_unrequested_selection_metric_fails_preflight(tmp_path: Path) -> None:
    raw = _base_config({"path": "link.sigma", "values": [1.0]})
    raw["selection"] = {
        "metric": "validation_rmse",
        "mode": "min",
        "retain_checkpoint": True,
    }
    config_path = tmp_path / "selection.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    EVENTS.clear()

    with pytest.raises(ValueError, match="is not requested by diagnostics"):
        run_sweep(config_path, output_root=tmp_path / "results")

    assert EVENTS == []
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    ("path", "unknown_name"),
    [
        ("algorithm.name", "unknown_audit_algorithm"),
        ("data.source", "unknown_audit_data"),
        ("initialization.joint.name", "unknown_audit_initializer"),
    ],
)
def test_unknown_swept_components_fail_preflight(
    tmp_path: Path,
    path: str,
    unknown_name: str,
) -> None:
    config_path = tmp_path / "unknown.yaml"
    config_path.write_text(
        yaml.safe_dump(_base_config({"path": path, "values": [unknown_name]})),
        encoding="utf-8",
    )
    EVENTS.clear()

    with pytest.raises(KeyError, match=unknown_name):
        run_sweep(config_path, output_root=tmp_path / "results")

    assert EVENTS == []
    assert not (tmp_path / "results").exists()


def test_unknown_diagnostic_family_fails_preflight(tmp_path: Path) -> None:
    raw = _base_config({"path": "link.sigma", "values": [2.0]})
    raw["diagnostics"]["families"] = {"unknown_audit_diagnostics": ["metric"]}
    config_path = tmp_path / "unknown-diagnostic.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    EVENTS.clear()

    with pytest.raises(KeyError, match="unknown_audit_diagnostics"):
        run_sweep(config_path, output_root=tmp_path / "results")

    assert EVENTS == []
    assert not (tmp_path / "results").exists()


def test_invalid_initialization_mode_fails_preflight(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-mode.yaml"
    config_path.write_text(
        yaml.safe_dump(_base_config({"path": "initialization.mode", "values": ["unknown_mode"]})),
        encoding="utf-8",
    )
    EVENTS.clear()

    with pytest.raises(ValueError, match="initialization.mode"):
        run_sweep(config_path, output_root=tmp_path / "results")

    assert EVENTS == []
    assert not (tmp_path / "results").exists()
