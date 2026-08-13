"""Sweep runner execution loop and serialization utilities."""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
import numbers
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from bcd_lab.algorithms.registry import (
    Algorithm,
    DatasetCapabilityValidator,
    SelectedStateExporter,
    get_algorithm,
)
from bcd_lab.checkpoints import save_checkpoint_archive
from bcd_lab.data.registry import load_dataset
from bcd_lab.function_models import ScalarFunction
from bcd_lab.initializers.builder import build_initialization
from bcd_lab.losses import Objective
from bcd_lab.randomness import RunRandomness
from bcd_lab.types import AlgorithmResult, DatasetBundle

from .artifacts import (
    ARTIFACT_VERSION,
    collect_provenance,
    json_safe,
    serialize_experiment_spec,
    write_artifact,
)
from .composition import (
    ExpandedExperimentCase,
    build_diagnostics_request,
    build_initialization_spec,
    build_model_context,
    build_objective,
    data_config_for_dataset,
    expand_experiment_cases,
    preflight_experiment_cases,
)
from .loader import load_sweep_config
from .schema import SweepSection


def _safe_fragment(value: object, *, maximum: int = 40) -> str:
    serialized = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))
    fragment = re.sub(r"[^A-Za-z0-9_-]+", "-", serialized).strip("-")
    return (fragment or "value")[:maximum]


def build_run_name(
    case: ExpandedExperimentCase,
    trial_idx: int,
    randomness: RunRandomness,
) -> str:
    """Build a readable ID with canonical content identity."""
    resolved_spec = dataclasses.replace(
        case.spec,
        trials=1,
        seed=randomness.seed,
        sweep=SweepSection(),
    )
    identity = {
        "experiment_spec": serialize_experiment_spec(resolved_spec),
        "sweep_parameters": case.assignments,
        "trial": trial_idx + 1,
    }
    canonical = json.dumps(json_safe(identity), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    pieces = [_safe_fragment(case.spec.experiment)]
    for key, value in sorted(case.assignments.items()):
        pieces.append(f"{_safe_fragment(key, maximum=28)}-{_safe_fragment(value)}")
    if case.spec.trials > 1:
        pieces.append(f"trial-{trial_idx + 1}")
    readable = "_".join(pieces)[:180].rstrip("_-")
    return f"{readable}_{digest}"


@dataclasses.dataclass(frozen=True)
class PlannedRun:
    """A unique run identity resolved before output side effects."""

    case: ExpandedExperimentCase
    trial_idx: int
    randomness: RunRandomness
    run_name: str


def _plan_runs(cases: list[ExpandedExperimentCase]) -> list[PlannedRun]:
    planned: list[PlannedRun] = []
    names: set[str] = set()
    for case in cases:
        for trial_idx in range(case.spec.trials):
            randomness = RunRandomness.for_trial(case.spec.seed, trial_idx)
            run_name = build_run_name(case, trial_idx, randomness)
            if run_name in names:
                raise ValueError(
                    f"Experiment expands to duplicate run identity {run_name!r}; "
                    "remove duplicate sweep values or cases"
                )
            names.add(run_name)
            planned.append(PlannedRun(case, trial_idx, randomness, run_name))
    return planned


def _ensure_output_policy(
    results_dir: Path,
    planned: list[PlannedRun],
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    conflicts: list[Path] = []
    for run in planned:
        candidates = [
            results_dir / f"{run.run_name}.json",
            results_dir / f"{run.run_name}_checkpoints.npz",
            results_dir / f"{run.run_name}_selected_checkpoint.npz",
        ]
        conflicts.extend(path for path in candidates if path.exists())
    if conflicts:
        names = ", ".join(path.name for path in conflicts[:3])
        suffix = "..." if len(conflicts) > 3 else ""
        raise FileExistsError(
            f"Refusing to overwrite existing run artifacts: {names}{suffix}. "
            "Pass overwrite=True explicitly to replace planned outputs."
        )


def _build_summary_metrics(diagnostics: list[dict] | None) -> dict[str, Any]:
    """Summarize every finite numeric diagnostic series."""
    summary_metrics: dict[str, Any] = {}
    metric_names = sorted(
        {name for diagnostic in diagnostics or [] for name in diagnostic if name != "iteration"}
    )
    for metric in metric_names:
        values_by_iteration: list[tuple[int, float]] = []
        for diagnostic in diagnostics or []:
            value = diagnostic.get(metric)
            iteration = diagnostic.get("iteration")
            if (
                isinstance(value, numbers.Real)
                and not isinstance(value, bool)
                and isinstance(iteration, numbers.Integral)
                and not isinstance(iteration, bool)
                and math.isfinite(float(value))
            ):
                values_by_iteration.append((int(iteration), float(value)))
        if values_by_iteration:
            iterations, values = zip(*values_by_iteration, strict=True)
            min_idx = int(np.argmin(values))
            summary_metrics[metric] = {
                "min": values[min_idx],
                "min_iteration": iterations[min_idx],
                "final": values[-1],
                "final_iteration": iterations[-1],
            }
    return summary_metrics


def _save_run_results(
    results_dir: Path,
    run_name: str,
    payload: dict,
    result: AlgorithmResult,
    save_checkpoints: bool,
    *,
    overwrite: bool = False,
) -> None:
    """Write the result artifact and optional versioned checkpoint archive."""
    metrics_path = results_dir / f"{run_name}.json"
    npz_path = results_dir / f"{run_name}_checkpoints.npz"
    if save_checkpoints and result.checkpoints:
        save_checkpoint_archive(npz_path, result.checkpoints)
    elif overwrite:
        npz_path.unlink(missing_ok=True)

    selected_path = results_dir / f"{run_name}_selected_checkpoint.npz"
    selection = result.selection
    if selection is not None and selection.checkpoint is not None:
        save_checkpoint_archive(
            selected_path,
            {selection.iteration: selection.checkpoint},
        )
    elif overwrite:
        selected_path.unlink(missing_ok=True)

    write_artifact(metrics_path, payload)


def _selection_payload(run_name: str, result: AlgorithmResult) -> dict[str, Any] | None:
    selected = result.selection
    if selected is None:
        return None
    return {
        "metric": selected.policy.metric,
        "mode": selected.policy.mode,
        "iteration": selected.iteration,
        "value": selected.value,
        "diagnostics": dict(selected.diagnostics),
        "checkpoint_file": (
            f"{run_name}_selected_checkpoint.npz" if selected.checkpoint is not None else None
        ),
        "exports": dict(selected.exports),
    }


@dataclasses.dataclass(frozen=True)
class _PreparedRun:
    """Resolved components and inputs for one planned experiment run."""

    planned: PlannedRun
    dataset: DatasetBundle
    algorithm: Algorithm
    objective: Objective
    algorithm_config: Any
    resolved_data: dict[str, Any]
    seeds: dict[str, int]
    z_init: np.ndarray | None
    phi_init: ScalarFunction | None


def _prepare_run(planned: PlannedRun) -> _PreparedRun:
    case = planned.case.spec
    seeds = planned.randomness.manifest("data", "initialization", "algorithm")
    dataset = load_dataset(str(case.data.get("source", "synthetic")), case.data, seed=seeds["data"])
    algorithm = get_algorithm(case.algorithm.name)
    objective = build_objective(case.objective)
    resolved_data = data_config_for_dataset(
        case.data,
        n_rows=dataset.n_rows,
        n_cols=dataset.n_cols,
        rank=dataset.rank,
        n_observations=dataset.n_observations,
    )
    fingerprint = dataset.capabilities.source_fingerprint
    if fingerprint is not None:
        resolved_data["source_fingerprint"] = dataclasses.asdict(fingerprint)
    algorithm_config = algorithm.build_config(
        data=resolved_data,
        link=case.link,
        runtime=case.runtime,
        params=case.algorithm.params,
    )
    if isinstance(algorithm, DatasetCapabilityValidator):
        algorithm.validate_dataset(config=algorithm_config, dataset=dataset)
    model_context = build_model_context(resolved_data, case.link, case.runtime)
    z_init, phi_init = build_initialization(
        build_initialization_spec(case.initialization),
        dataset=dataset,
        config=model_context,
        seed=seeds["initialization"],
    )
    return _PreparedRun(
        planned=planned,
        dataset=dataset,
        algorithm=algorithm,
        objective=objective,
        algorithm_config=algorithm_config,
        resolved_data=resolved_data,
        seeds=seeds,
        z_init=z_init,
        phi_init=phi_init,
    )


def _execute_prepared_run(prepared: _PreparedRun) -> AlgorithmResult:
    case = prepared.planned.case.spec
    result = prepared.algorithm.run(
        config=prepared.algorithm_config,
        dataset=prepared.dataset,
        objective=prepared.objective,
        z_init=prepared.z_init,
        phi_init=prepared.phi_init,
        diagnostics_request=build_diagnostics_request(case.diagnostics),
        selection_policy=case.selection,
        rng=prepared.planned.randomness.generator("algorithm"),
    )
    if (
        result.selection is not None
        and result.selection.checkpoint is not None
        and isinstance(prepared.algorithm, SelectedStateExporter)
    ):
        exports = prepared.algorithm.export_selected_state(
            result.selection.checkpoint,
            config=prepared.algorithm_config,
            dataset=prepared.dataset,
        )
        result.selection = dataclasses.replace(
            result.selection,
            exports=dict(json_safe(exports)),
        )
    return result


def _build_result_payload(
    prepared: _PreparedRun,
    result: AlgorithmResult,
    *,
    provenance: dict[str, Any],
    started_at: dt.datetime,
    elapsed_seconds: float,
) -> dict[str, Any]:
    planned = prepared.planned
    case = planned.case.spec
    fingerprint = prepared.dataset.capabilities.source_fingerprint
    preprocessing = prepared.dataset.capabilities.preprocessing
    resolved_spec = dataclasses.replace(
        case,
        trials=1,
        seed=planned.randomness.seed,
        data=prepared.resolved_data,
        sweep=SweepSection(),
    )
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "experiment_spec": serialize_experiment_spec(resolved_spec),
        "run": {
            "trial": planned.trial_idx + 1,
            "seeds": prepared.seeds,
            "started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
            "elapsed_seconds": elapsed_seconds,
            "termination": {
                "reason": result.termination.reason,
                "iteration": result.termination.iteration,
            },
        },
        "provenance": {
            **provenance,
            "data_source": None if fingerprint is None else dataclasses.asdict(fingerprint),
            "preprocessing": (None if preprocessing is None else dataclasses.asdict(preprocessing)),
        },
        "diagnostics": result.diagnostics,
        "metadata": result.metadata,
        "sweep_parameters": dict(planned.case.assignments),
        "summary": {"metrics": _build_summary_metrics(result.diagnostics)},
    }
    selection = _selection_payload(planned.run_name, result)
    if selection is not None:
        payload["selection"] = selection
    return payload


def _resolve_results_dir(
    config_path: Path,
    output_root: str | Path | None,
    experiment: str,
) -> Path:
    if output_root is None:
        root = config_path.parent / "results"
    else:
        root = Path(output_root).expanduser()
        if not root.is_absolute():
            root = config_path.parent / root
    return root.resolve() / experiment


def run_sweep(
    config_path: str | Path,
    output_root: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> None:
    config_path = Path(config_path).expanduser().resolve()
    cfg = load_sweep_config(config_path)
    cases = expand_experiment_cases(cfg)
    preflight_experiment_cases(cases)
    planned_runs = _plan_runs(cases)

    # Expansion and registry preflight deliberately finish before output
    # creation so invalid cases cannot leave a partial experiment.
    results_dir = _resolve_results_dir(config_path, output_root, cfg.experiment)
    _ensure_output_policy(results_dir, planned_runs, overwrite=overwrite)
    results_dir.mkdir(parents=True, exist_ok=True)
    provenance = collect_provenance()

    for planned in planned_runs:
        started_at = dt.datetime.now(dt.timezone.utc)
        run_start = time.perf_counter()
        prepared = _prepare_run(planned)
        result = _execute_prepared_run(prepared)
        payload = _build_result_payload(
            prepared,
            result,
            provenance=provenance,
            started_at=started_at,
            elapsed_seconds=time.perf_counter() - run_start,
        )
        _save_run_results(
            results_dir=results_dir,
            run_name=planned.run_name,
            payload=payload,
            result=result,
            save_checkpoints=bool(
                prepared.planned.case.spec.runtime.get("save_checkpoints", False)
            ),
            overwrite=overwrite,
        )
