"""Integration tests for complete-spec sweep composition."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest
import yaml

from bcd_lab.algorithms.registry import register_algorithm
from bcd_lab.checkpoints import load_checkpoint_archive
from bcd_lab.data.registry import register_data_loader
from bcd_lab.diagnostics import DiagnosticContext, register_diagnostic_family
from bcd_lab.experiments.artifacts import deserialize_experiment_spec, write_artifact
from bcd_lab.experiments.composition import (
    expand_experiment_cases,
    preflight_experiment_cases,
)
from bcd_lab.experiments.loader import load_sweep_config
from bcd_lab.experiments.runner import run_sweep
from bcd_lab.function_models import IdentityFunction
from bcd_lab.initializers.registry import (
    register_joint_initializer,
    register_phi_initializer,
    register_z_initializer,
)
from bcd_lab.losses import register_objective
from bcd_lab.randomness import RunRandomness
from bcd_lab.types import AlgorithmResult, DatasetBundle, RunTermination

EVENTS: list[tuple[str, Any]] = []


def _audit_objective_value(observed, predictions, params):
    return float(params["weight"] * np.sum((predictions - observed) ** 2))


def _audit_objective_gradient(observed, predictions, params):
    return 2.0 * float(params["weight"]) * (predictions - observed)


register_objective(
    "audit_weighted_squared",
    value=_audit_objective_value,
    prediction_gradient=_audit_objective_gradient,
)


@dataclass
class _AuditLoader:
    name: str

    def load(self, *, params: dict[str, Any], seed: int) -> DatasetBundle:
        EVENTS.append(("data", (self.name, params.get("variant"), seed)))
        values = np.array([1.0, 1.0])
        indices = (np.array([0, 1]), np.array([0, 1]))
        return DatasetBundle(
            source=self.name,
            n_rows=2,
            n_cols=2,
            rank=1,
            train=(values, indices),
        )


@dataclass
class _AuditInitializer:
    name: str

    def build(self, *, dataset, config, params, seed):
        EVENTS.append(("initialization", (self.name, params.get("variant"), seed)))
        return np.zeros((dataset.n_rows + dataset.n_cols, dataset.rank)), IdentityFunction()


@dataclass
class _AuditZInitializer:
    name: str

    def build(self, *, dataset, config, params, seed):
        EVENTS.append(("initialization", (self.name, params.get("variant"), seed)))
        return np.ones((dataset.n_rows + dataset.n_cols, dataset.rank))


@dataclass
class _AuditPhiInitializer:
    name: str

    def build(self, *, dataset, config, params, seed, z):
        EVENTS.append(("initialization", (self.name, params.get("variant"), seed)))
        return IdentityFunction()


class _AuditDiagnosticFamily:
    name = "audit_extension"

    def compute(
        self,
        ctx: DiagnosticContext,
        *,
        metrics: set[str] | None = None,
    ) -> dict[str, int]:
        requested = metrics or {"audit_iteration"}
        EVENTS.append(("diagnostic", (ctx.iteration, tuple(sorted(requested)))))
        return {metric: ctx.iteration for metric in requested}


@dataclass
class _AuditAlgorithm:
    name: str

    def build_config(
        self,
        *,
        data: Mapping[str, Any],
        link: Mapping[str, Any],
        runtime: Mapping[str, Any],
        params: Mapping[str, Any],
    ):
        EVENTS.append(
            (
                "build",
                (
                    self.name,
                    params.get("knob"),
                    params.get("phi_update", {}).get("learning_rate"),
                    link["kind"],
                    link["sigma"],
                    runtime["max_iter"],
                ),
            )
        )
        return {"name": self.name, "knob": params.get("knob")}

    def run(
        self,
        *,
        config,
        dataset,
        objective,
        z_init=None,
        phi_init=None,
        diagnostics_request=None,
        selection_policy=None,
        rng: np.random.Generator,
    ):
        _ = selection_policy
        EVENTS.append(("run", (self.name, diagnostics_request, int(rng.integers(2**32)))))
        EVENTS.append(("objective", (objective.name, dict(objective.params))))
        EVENTS.append(("inputs", (type(config), np.asarray(z_init).shape, phi_init.kind)))
        return AlgorithmResult(
            z=np.asarray(z_init),
            phi=phi_init,
            diagnostics=[{"iteration": 0, "audit_novel_score": 2.5}],
            termination=RunTermination(reason="completed", iteration=0),
        )


for component in (
    _AuditLoader("audit_data_a"),
    _AuditLoader("audit_data_b"),
):
    register_data_loader(component)
for component in (
    _AuditInitializer("audit_init_a"),
    _AuditInitializer("audit_init_b"),
):
    register_joint_initializer(component)
register_z_initializer(_AuditZInitializer("audit_z"))
register_phi_initializer(_AuditPhiInitializer("audit_phi"))
for component in (
    _AuditAlgorithm("audit_algo_a"),
    _AuditAlgorithm("audit_algo_b"),
):
    register_algorithm(component)
register_diagnostic_family(_AuditDiagnosticFamily())


def _base_config(axis: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "experiment": "composition",
        "algorithm": {
            "name": "audit_algo_a",
            "params": {
                "knob": "base",
                "z_optimizer": {
                    "name": "gradient_descent",
                    "params": {"learning_rate": 0.1},
                },
                "phi_update": {"learning_rate": 0.1},
                "regularization": {"lambda_": 0.0, "mu": 0.0, "alpha": 0.0},
            },
        },
        "objective": {"name": "squared_error", "params": {}},
        "trials": 1,
        "seed": 3,
        "data": {
            "source": "audit_data_a",
            "variant": "base",
            "shape": {
                "n_rows": 2,
                "n_cols": 2,
                "rank": 1,
                "n_observations": 2,
            },
        },
        "link": {"kind": "identity", "sigma": 1.0, "params": {}},
        "initialization": {
            "mode": "joint",
            "joint": {"name": "audit_init_a", "params": {"variant": "base"}},
            "z": {"name": "audit_z", "params": {"variant": "base"}},
            "phi": {"name": "audit_phi", "params": {"variant": "base"}},
        },
        "runtime": {"max_iter": 0, "tol": 0.0, "save_checkpoints": False},
        "diagnostics": {
            "on_error": "raise",
            "families": {"objective": ["loss"]},
        },
        "sweep": {"axes": [axis]},
    }


def _run_axis(
    tmp_path: Path,
    path: str,
    values: list[Any],
    *,
    overwrite: bool = False,
) -> list[tuple[str, Any]]:
    EVENTS.clear()
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(_base_config({"path": path, "values": values})),
        encoding="utf-8",
    )
    run_sweep(config_path, output_root=tmp_path / "results", overwrite=overwrite)
    return list(EVENTS)


@pytest.mark.parametrize(
    ("path", "values", "event_kind", "expected"),
    [
        (
            "algorithm.name",
            ["audit_algo_a", "audit_algo_b"],
            "run",
            {"audit_algo_a", "audit_algo_b"},
        ),
        (
            "data.source",
            ["audit_data_a", "audit_data_b"],
            "data",
            {"audit_data_a", "audit_data_b"},
        ),
        ("data.variant", ["one", "two"], "data", {"one", "two"}),
        (
            "initialization.joint.name",
            ["audit_init_a", "audit_init_b"],
            "initialization",
            {"audit_init_a", "audit_init_b"},
        ),
        (
            "initialization.joint.params.variant",
            ["one", "two"],
            "initialization",
            {"one", "two"},
        ),
        ("algorithm.params.knob", ["one", "two"], "build", {"one", "two"}),
        ("algorithm.params.phi_update.learning_rate", [0.2, 0.3], "build", {0.2, 0.3}),
        ("link.kind", ["identity", "sigmoid"], "build", {"identity", "sigmoid"}),
        ("link.sigma", [2.0, 3.0], "build", {2.0, 3.0}),
        (
            "objective.name",
            ["squared_error", "half_squared"],
            "objective",
            {"squared_error", "half_squared"},
        ),
        ("runtime.max_iter", [2, 3], "build", {2, 3}),
        (
            "diagnostics.families.objective",
            [["loss"], ["other"]],
            "run",
            {"loss", "other"},
        ),
    ],
)
def test_complete_experiment_dimensions_are_sweepable(
    tmp_path: Path,
    path: str,
    values: list[Any],
    event_kind: str,
    expected: set[Any],
) -> None:
    events = [value for kind, value in _run_axis(tmp_path, path, values) if kind == event_kind]
    if path == "algorithm.name":
        observed = {value[0] for value in events}
    elif path == "data.source":
        observed = {value[0] for value in events}
    elif path == "data.variant":
        observed = {value[1] for value in events}
    elif path == "initialization.joint.name":
        observed = {value[0] for value in events}
    elif path == "initialization.joint.params.variant":
        observed = {value[1] for value in events}
    elif path == "algorithm.params.knob":
        observed = {value[1] for value in events}
    elif path == "algorithm.params.phi_update.learning_rate":
        observed = {value[2] for value in events}
    elif path == "link.kind":
        observed = {value[3] for value in events}
    elif path == "link.sigma":
        observed = {value[4] for value in events}
    elif path == "objective.name":
        observed = {value[0] for value in events}
    elif path == "runtime.max_iter":
        observed = {value[5] for value in events}
    else:
        observed = {next(iter(value[1].families["objective"])) for value in events}
    assert observed == expected


def test_invalid_sweep_path_fails_before_output_or_work(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        yaml.safe_dump(_base_config({"path": "data.misspelled", "values": [1]})),
        encoding="utf-8",
    )
    EVENTS.clear()

    with pytest.raises(ValueError, match="data.misspelled"):
        run_sweep(config_path, output_root=tmp_path / "results")

    assert EVENTS == []
    assert not (tmp_path / "results").exists()


def test_custom_algorithm_receives_only_its_declared_parameters(tmp_path: Path) -> None:
    config_path = tmp_path / "custom.yaml"
    raw = _base_config({"path": "algorithm.params.knob", "values": ["value"]})
    raw["algorithm"]["params"] = {"knob": "value"}
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_sweep_config(config_path)

    assert config.algorithm.name == "audit_algo_a"
    assert config.algorithm.params == {"knob": "value"}


def test_typed_sweep_values_receive_distinct_content_ids(tmp_path: Path) -> None:
    config_path = tmp_path / "typed-collision.yaml"
    config_path.write_text(
        yaml.safe_dump(_base_config({"path": "algorithm.params.knob", "values": [1, "1"]})),
        encoding="utf-8",
    )

    run_sweep(config_path, output_root=tmp_path / "results")

    paths = sorted((tmp_path / "results" / "composition").glob("*.json"))
    assert len(paths) == 2
    assert len({path.stem.rsplit("_", 1)[-1] for path in paths}) == 2


def test_existing_run_requires_explicit_overwrite(tmp_path: Path) -> None:
    config_path = tmp_path / "existing.yaml"
    config_path.write_text(
        yaml.safe_dump(_base_config({"path": "algorithm.params.knob", "values": ["one"]})),
        encoding="utf-8",
    )
    output_root = tmp_path / "results"
    run_sweep(config_path, output_root=output_root)
    [result_path] = (output_root / "composition").glob("*.json")
    original = result_path.read_bytes()

    with pytest.raises(FileExistsError, match="overwrite=True"):
        run_sweep(config_path, output_root=output_root)
    assert result_path.read_bytes() == original

    run_sweep(config_path, output_root=output_root, overwrite=True)
    assert result_path.is_file()


def test_artifact_write_is_atomic_on_serialization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.json"
    target.write_text("original\n", encoding="utf-8")

    def fail_dump(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected write failure")

    monkeypatch.setattr("bcd_lab.experiments.artifacts.json.dump", fail_dump)
    with pytest.raises(RuntimeError, match="injected write failure"):
        write_artifact(target, {"valid": 1})

    assert target.read_text(encoding="utf-8") == "original\n"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_result_artifact_round_trips_the_resolved_case(tmp_path: Path) -> None:
    config_path = tmp_path / "result-format.yaml"
    axis = {"path": "algorithm.params.knob", "values": ["canonical"]}
    raw = _base_config(axis)
    raw["plot"] = {"enabled": True, "legend_key": "algorithm.params.knob"}
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    run_sweep(config_path, output_root=tmp_path / "results")

    [result_path] = (tmp_path / "results" / "composition").glob("*.json")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "artifact_version",
        "experiment_spec",
        "run",
        "provenance",
        "diagnostics",
        "metadata",
        "sweep_parameters",
        "summary",
    }
    assert payload["artifact_version"] == 2
    assert payload["sweep_parameters"] == {"algorithm.params.knob": "canonical"}

    reconstructed = deserialize_experiment_spec(payload["experiment_spec"])
    replay_cases = expand_experiment_cases(reconstructed)
    assert len(replay_cases) == 1
    preflight_experiment_cases(replay_cases)
    assert reconstructed.trials == 1
    assert reconstructed.sweep.axes == []
    assert reconstructed.seed == RunRandomness.for_trial(3, 0).seed
    assert payload["run"]["seeds"] == RunRandomness.for_trial(3, 0).manifest(
        "data", "initialization", "algorithm"
    )
    assert reconstructed.plot == raw["plot"]
    assert reconstructed.algorithm.params["knob"] == "canonical"
    assert reconstructed.data["shape"] == {
        "n_rows": 2,
        "n_cols": 2,
        "rank": 1,
        "n_observations": 2,
    }
    assert payload["run"]["termination"] == {
        "reason": "completed",
        "iteration": 0,
    }
    assert payload["summary"]["metrics"]["audit_novel_score"] == {
        "min": 2.5,
        "min_iteration": 0,
        "final": 2.5,
        "final_iteration": 0,
    }
    assert payload["provenance"]["package_version"]
    assert payload["provenance"]["python_version"]
    assert set(payload["provenance"]["dependencies"]) == {
        "numpy",
        "scipy",
        "scikit-learn",
        "PyYAML",
        "matplotlib",
        "cvxpy",
        "pandas",
        "xlrd",
    }


def test_registered_objective_is_selectable_and_sweepable_from_yaml(
    tmp_path: Path,
) -> None:
    raw = _base_config({"path": "objective.params.weight", "values": [0.5, 2.0]})
    raw["objective"] = {
        "name": "audit_weighted_squared",
        "params": {"weight": 1.0},
    }
    config_path = tmp_path / "objective.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    EVENTS.clear()

    run_sweep(config_path, output_root=tmp_path / "results")

    objective_events = [value for kind, value in EVENTS if kind == "objective"]
    assert objective_events == [
        ("audit_weighted_squared", {"weight": 0.5}),
        ("audit_weighted_squared", {"weight": 2.0}),
    ]


def test_private_algorithm_config_works_with_builtin_initializers(tmp_path: Path) -> None:
    raw = _base_config({"path": "algorithm.params.knob", "values": ["value"]})
    raw["algorithm"]["params"] = {"knob": "value"}
    raw["initialization"] = {
        "mode": "separate",
        "z": {"name": "svd", "params": {}},
        "phi": {"name": "identity", "params": {}},
    }
    config_path = tmp_path / "builtin-initializers.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    EVENTS.clear()

    run_sweep(config_path, output_root=tmp_path / "results")

    inputs = [value for kind, value in EVENTS if kind == "inputs"]
    assert inputs == [(dict, (4, 1), "identity")]


def test_none_phi_initializer_remains_none_through_real_bcd_sweep(
    tmp_path: Path,
) -> None:
    raw = _base_config({"path": "link.sigma", "values": [1.0]})
    raw["algorithm"] = {
        "name": "bcd",
        "params": {
            "z_optimizer": {
                "name": "gradient_descent",
                "params": {"learning_rate": 0.1},
            },
            "phi_update": {"learning_rate": 0.0},
            "regularization": {"lambda_": 0.0, "mu": 0.0, "alpha": 0.0},
            "project_z": False,
            "freeze_phi_updates": True,
        },
    }
    raw["initialization"] = {
        "mode": "separate",
        "z": {"name": "svd", "params": {}},
        "phi": {"name": "none", "params": {}},
    }
    raw["runtime"] = {
        "max_iter": 0,
        "tol": 0.0,
        "diagnostics_interval": 1,
        "checkpoint_iters": [0],
        "save_checkpoints": True,
    }
    config_path = tmp_path / "none-phi.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    run_sweep(config_path, output_root=tmp_path / "results")

    checkpoint_path = next((tmp_path / "results" / "composition").glob("*_checkpoints.npz"))
    checkpoint = load_checkpoint_archive(checkpoint_path)[0]
    assert checkpoint.payload["phi"] is None


@pytest.mark.parametrize(
    ("params_update", "message"),
    [
        ({"steps": {"eta": 0.1, "zeta": 0.1, "alpha": 0.0}}, "unknown fields: steps"),
        ({"z_optimizer": {"name": "missing", "params": {}}}, "Unknown optimizer"),
        (
            {
                "z_optimizer": {
                    "name": "adam",
                    "params": {"learning_rate": 0.1, "unknown": 1},
                }
            },
            "unknown parameters",
        ),
    ],
)
def test_invalid_bcd_optimizer_configuration_fails_before_work(
    tmp_path: Path,
    params_update: dict[str, Any],
    message: str,
) -> None:
    raw = _base_config({"path": "link.sigma", "values": [1.0]})
    raw["algorithm"] = {
        "name": "bcd",
        "params": {
            "z_optimizer": {
                "name": "gradient_descent",
                "params": {"learning_rate": 0.1},
            },
            "phi_update": {"learning_rate": 0.0},
            "regularization": {"lambda_": 0.0, "mu": 0.0, "alpha": 0.0},
            "project_z": False,
            **params_update,
        },
    }
    config_path = tmp_path / "invalid-optimizer.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    EVENTS.clear()

    with pytest.raises((KeyError, ValueError), match=message):
        run_sweep(config_path, output_root=tmp_path / "results")

    assert EVENTS == []
    assert not (tmp_path / "results").exists()


def test_separate_initializer_extensions_run_without_orchestration_changes(
    tmp_path: Path,
) -> None:
    events = _run_axis(
        tmp_path,
        "initialization.mode",
        ["joint", "separate"],
    )
    initializer_names = {value[0] for kind, value in events if kind == "initialization"}
    assert initializer_names == {"audit_init_a", "audit_z", "audit_phi"}


def test_registered_diagnostic_runs_through_real_bcd_sweep(tmp_path: Path) -> None:
    raw = _base_config({"path": "link.sigma", "values": [1.0]})
    raw["algorithm"] = {
        "name": "bcd",
        "params": {
            "z_optimizer": {
                "name": "gradient_descent",
                "params": {"learning_rate": 0.1},
            },
            "phi_update": {"learning_rate": 0.0},
            "regularization": {"lambda_": 0.0, "mu": 0.0, "alpha": 0.0},
            "project_z": False,
            "freeze_phi_updates": False,
        },
    }
    raw["runtime"] = {
        "max_iter": 0,
        "tol": 0.0,
        "diagnostics_interval": 1,
        "save_checkpoints": False,
    }
    raw["diagnostics"] = {
        "on_error": "raise",
        "families": {"audit_extension": ["audit_iteration"]},
    }
    config_path = tmp_path / "custom-diagnostic.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    EVENTS.clear()

    run_sweep(config_path, output_root=tmp_path / "results")

    diagnostic_events = [value for kind, value in EVENTS if kind == "diagnostic"]
    assert diagnostic_events == [(0, ("audit_iteration",))]
    [result_path] = (tmp_path / "results" / "composition").glob("*.json")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert len(payload["diagnostics"]) == 1
    assert payload["diagnostics"][0]["iteration"] == 0
    assert payload["diagnostics"][0]["audit_iteration"] == 0


def test_swept_seed_and_trial_count_are_case_local(tmp_path: Path) -> None:
    seed_events = _run_axis(tmp_path, "seed", [10, 20])
    observed_seeds = {value[2] for kind, value in seed_events if kind == "data"}
    assert observed_seeds == {
        RunRandomness.for_trial(seed, 0).child_seed("data") for seed in (10, 20)
    }

    trial_events = _run_axis(tmp_path, "trials", [1, 2])
    assert sum(kind == "run" for kind, _ in trial_events) == 3


def test_one_seed_reproduces_and_changes_all_stochastic_components(tmp_path: Path) -> None:
    first = _run_axis(tmp_path, "seed", [17])
    second = _run_axis(tmp_path, "seed", [17], overwrite=True)
    changed = _run_axis(tmp_path, "seed", [18])

    def stochastic_values(events: list[tuple[str, Any]]) -> tuple[int, int, int]:
        data_seed = next(value[2] for kind, value in events if kind == "data")
        initialization_seed = next(value[2] for kind, value in events if kind == "initialization")
        algorithm_draw = next(value[2] for kind, value in events if kind == "run")
        return data_seed, initialization_seed, algorithm_draw

    assert stochastic_values(first) == stochastic_values(second)
    assert all(
        left != right
        for left, right in zip(stochastic_values(first), stochastic_values(changed), strict=True)
    )
    second_trial = RunRandomness.for_trial(17, 1)
    replay = RunRandomness.for_trial(second_trial.seed, 0)
    assert replay.manifest("data", "initialization", "algorithm") == (
        second_trial.manifest("data", "initialization", "algorithm")
    )


def test_non_integer_configured_and_swept_seeds_fail_before_work(tmp_path: Path) -> None:
    for name, raw in (
        (
            "configured",
            {**_base_config({"path": "link.sigma", "values": [1.0]}), "seed": "3"},
        ),
        (
            "swept",
            _base_config({"path": "seed", "values": [1.5]}),
        ),
    ):
        config_path = tmp_path / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        EVENTS.clear()

        with pytest.raises(ValueError, match="seed must be one non-negative integer"):
            run_sweep(config_path, output_root=tmp_path / "results")

        assert EVENTS == []
        assert not (tmp_path / "results").exists()
