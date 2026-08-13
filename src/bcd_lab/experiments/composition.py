"""Translate validated experiment cases into registry component inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from bcd_lab.algorithms.registry import CheckpointableAlgorithm, get_algorithm
from bcd_lab.config import (
    BasicModelBuildContext,
    LinkConfig,
    ModelBuildContext,
    RuntimeConfig,
    ShapeConfig,
)
from bcd_lab.data.registry import get_data_loader
from bcd_lab.diagnostics.registry import (
    DiagnosticErrorPolicy,
    DiagnosticRequest,
    get_diagnostic_family,
)
from bcd_lab.function_models import validate_function_model_config
from bcd_lab.initializers.builder import InitializationSpec, InitializerSpec
from bcd_lab.initializers.registry import (
    get_joint_initializer,
    get_phi_initializer,
    get_z_initializer,
)
from bcd_lab.losses import Objective, get_objective
from bcd_lab.randomness import validate_seed

from .schema import ExperimentSpec, ObjectiveSpec
from .sweep import expand_axes, replace_nested_strict
from .validation import validate_experiment_invariants


@dataclass(frozen=True)
class ExpandedExperimentCase:
    """One complete experiment specification plus its displayed assignments."""

    spec: ExperimentSpec
    assignments: dict[str, Any]


def expand_experiment_cases(config: ExperimentSpec) -> list[ExpandedExperimentCase]:
    """Apply every sweep assignment to the complete experiment specification."""
    cases: list[ExpandedExperimentCase] = []
    for assignments in expand_axes(config.sweep.axes):
        expanded = config
        for path, value in assignments.items():
            if path.split(".", 1)[0] in {
                "schema_version",
                "experiment",
                "sweep",
                "plot",
                "selection",
            }:
                raise ValueError(f"Sweep path {path!r} targets a reserved orchestration field")
            expanded = replace_nested_strict(expanded, path, value)
        cases.append(ExpandedExperimentCase(spec=expanded, assignments=assignments))
    return cases


def build_initialization_spec(config: dict[str, Any]) -> InitializationSpec:
    """Build a typed initialization request for one expanded case."""
    mode = str(config.get("mode", "joint"))
    if mode == "joint":
        joint = config.get("joint", {})
        return InitializationSpec(
            mode=mode,
            joint=InitializerSpec(
                name=str(joint.get("name", "svd_with_zero_phi")),
                params=dict(joint.get("params", {})),
            ),
        )

    if "z" not in config:
        raise ValueError("separate initialization requires initialization.z")
    if "phi" not in config:
        raise ValueError("separate initialization requires initialization.phi")
    z_data = config["z"]
    phi_data = config["phi"]
    return InitializationSpec(
        mode=mode,
        z=InitializerSpec(
            name=str(z_data.get("name", "svd")),
            params=dict(z_data.get("params", {})),
        ),
        phi=InitializerSpec(
            name=str(phi_data.get("name", "zero")),
            params=dict(phi_data.get("params", {})),
        ),
    )


def build_diagnostics_request(
    config: dict[str, Any],
) -> DiagnosticRequest | None:
    """Normalize diagnostic family configuration for an algorithm run."""
    if not config:
        return None
    families = config.get("families", {})
    on_error = cast(DiagnosticErrorPolicy, config.get("on_error", "raise"))
    return DiagnosticRequest(
        on_error=on_error,
        families={
            family: None if metrics is None else set(metrics)
            for family, metrics in families.items()
        },
    )


def preflight_experiment_cases(cases: list[ExpandedExperimentCase]) -> None:
    """Resolve every selected registry component before work or output begins."""
    for case in cases:
        spec = case.spec
        validate_experiment_invariants(spec)
        validate_seed(spec.seed)
        _validate_selection_request(spec)
        algorithm = get_algorithm(spec.algorithm.name)
        if (
            spec.selection is not None
            and spec.selection.retain_checkpoint
            and not isinstance(algorithm, CheckpointableAlgorithm)
        ):
            raise ValueError(
                f"algorithm {spec.algorithm.name!r} cannot retain a selected checkpoint"
            )
        validate_params = getattr(algorithm, "validate_params", None)
        if validate_params is not None:
            validate_params(params=spec.algorithm.params)
        objective = build_objective(spec.objective)
        link_kind = str(spec.link.get("kind"))
        validate_function_model_config(
            LinkConfig(
                kind=link_kind,
                sigma=float(spec.link.get("sigma", 1.0)),
                params=dict(spec.link.get("params", {})),
            )
        )
        objective.validate_link(link_kind)
        validate_objective = getattr(algorithm, "validate_objective", None)
        if validate_objective is not None:
            validate_objective(objective=objective, link=spec.link)
        data_loader = get_data_loader(str(spec.data.get("source", "synthetic")))
        validate_data_params = getattr(data_loader, "validate_params", None)
        if validate_data_params is not None:
            validate_data_params(spec.data)

        initialization = build_initialization_spec(spec.initialization)
        if initialization.mode == "joint":
            if initialization.joint is None:
                raise ValueError("joint initialization requires initialization.joint")
            get_joint_initializer(initialization.joint.name)
        elif initialization.mode == "separate":
            if initialization.z is None:
                raise ValueError("separate initialization requires initialization.z")
            if initialization.phi is None:
                raise ValueError("separate initialization requires initialization.phi")
            get_z_initializer(initialization.z.name)
            get_phi_initializer(initialization.phi.name)
        else:
            raise ValueError(
                f"initialization.mode must be 'joint' or 'separate', got {initialization.mode!r}"
            )

        for family in spec.diagnostics.get("families", {}):
            get_diagnostic_family(family)


def _validate_selection_request(spec: ExperimentSpec) -> None:
    policy = spec.selection
    if policy is None:
        return
    families = spec.diagnostics.get("families", {})
    requested_names = {
        metric for metrics in families.values() if metrics is not None for metric in metrics
    }
    split_prefixes = ("train_", "validation_", "test_")
    base_metric = next(
        (
            policy.metric[len(prefix) :]
            for prefix in split_prefixes
            if policy.metric.startswith(prefix)
        ),
        policy.metric,
    )
    if not (policy.metric in requested_names or base_metric in requested_names):
        raise ValueError(f"selection.metric {policy.metric!r} is not requested by diagnostics")


def build_objective(spec: ObjectiveSpec) -> Objective:
    """Resolve an explicit objective specification."""
    return get_objective(spec.name, spec.params)


def data_config_for_dataset(
    config: dict[str, Any],
    *,
    n_rows: int,
    n_cols: int,
    rank: int,
    n_observations: int,
) -> dict[str, Any]:
    """Return a case-local data config reflecting the loaded dataset shape."""
    resolved = dict(config)
    resolved["shape"] = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "rank": rank,
        "n_observations": n_observations,
    }
    return resolved


def build_model_context(
    data: dict[str, Any],
    link: dict[str, Any],
    runtime: dict[str, Any],
) -> ModelBuildContext:
    """Build the shared context used by initializers and function builders."""
    shape = data["shape"]
    return BasicModelBuildContext(
        shape=ShapeConfig(
            n_rows=int(shape["n_rows"]),
            n_cols=int(shape["n_cols"]),
            rank=int(shape["rank"]),
            n_observations=int(shape["n_observations"]),
        ),
        link=LinkConfig(
            kind=link["kind"],
            sigma=float(link.get("sigma", 1.0)),
            params=dict(link.get("params", {})),
        ),
        runtime=RuntimeConfig(
            max_iter=int(runtime.get("max_iter", 1000)),
            tol=float(runtime.get("tol", 1e-6)),
            diagnostics_interval=runtime.get("diagnostics_interval", 100),
            dtype=runtime.get("dtype", "float64"),
            checkpoint_iters=tuple(runtime.get("checkpoint_iters", ())),
        ),
    )
