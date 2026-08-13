"""Iteration lifecycle owned by the bundled BCD algorithm."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from bcd_lab.diagnostics import (
    DiagnosticContext,
    DiagnosticRequest,
    RunSnapshot,
    compute_requested_diagnostics,
)
from bcd_lab.types import (
    AlgorithmCheckpoint,
    AlgorithmResult,
    RunTermination,
    SelectedRunState,
    SelectionPolicy,
)

from .registry import AlgorithmContext
from .state import SolverState

UpdateFn = Callable[[SolverState, AlgorithmContext, int], None]
SnapshotPublisher = Callable[
    [
        SolverState,
        AlgorithmContext,
        int,
        float,
        float,
        float,
        DiagnosticRequest,
    ],
    RunSnapshot,
]
CheckpointCapture = Callable[[SolverState, float], AlgorithmCheckpoint]
CheckpointHook = Callable[[int, AlgorithmCheckpoint], None]


def _is_power_of_ten(value: int) -> bool:
    if value < 1:
        return False
    while value % 10 == 0:
        value //= 10
    return value == 1


def _diagnostics_due(
    iteration: int,
    *,
    max_iter: int,
    interval: int | None,
) -> bool:
    """Return whether the lifecycle should observe this completed iteration."""
    return bool(
        (interval is not None and iteration % interval == 0)
        or _is_power_of_ten(iteration)
        or iteration == max_iter
    )


def _loss_converged(history: list[dict[str, Any]], tolerance: float) -> bool:
    """Decide convergence from the two most recent observed losses."""
    if tolerance <= 0 or len(history) < 2:
        return False
    previous = history[-2].get("loss")
    current = history[-1].get("loss")
    if previous is None or current is None:
        return False
    return bool(abs(float(previous) - float(current)) < tolerance)


def default_diagnostics_request(
    state: SolverState,
    context: AlgorithmContext,
) -> DiagnosticRequest:
    """Build the default diagnostic request from available state and data."""
    families: dict[str, set[str] | None] = {
        "objective": {"loss"},
        "timing": {"elapsed"},
        "gradients": {"z_grad_norm", "phi_grad_norm", "z_optimizer_update_norm"},
        "reconstruction": None,
    }
    dataset = context.dataset
    if dataset.dense_truth is not None:
        families["truth_matrix"] = {"x_rmse", "x_error"}
    if dataset.latent_truth is not None:
        families["truth_z"] = {"goodness", "relative_goodness"}
    if state.phi is not None and dataset.capabilities.functions.truth is not None:
        families["phi"] = {"phi_diff", "phi_diff_0"}
    return DiagnosticRequest(families=families)


def _diagnostics_to_line(diagnostics: dict[str, Any]) -> str:
    priority_order = [
        "iteration",
        "loss",
        "goodness",
        "relative_goodness",
        "lyapunov",
        "phi_diff",
        "phi_diff_0",
        "phi_sample_mse",
        "x_error",
        "phi_x_error",
        "train_rmse",
        "validation_rmse",
        "phi_d_in_star_bounds",
        "phi_sharp_diff",
        "phi_sharp_diff_star",
        "regret",
        "phi_grad_norm",
        "z_grad_norm",
        "z_optimizer_update_norm",
        "sec_per_iter",
        "rem_sec",
    ]
    hidden_metric_keys = {
        "train_mse",
        "train_mae",
        "train_relative_mse",
        "train_relative_rmse",
        "validation_mse",
        "validation_mae",
        "validation_relative_mse",
        "validation_relative_rmse",
    }
    seen: set[str] = set()
    parts: list[str] = []

    def format_value(value: object) -> str:
        return f"{value:.6g}" if isinstance(value, float) else str(value)

    for key in priority_order:
        if key in diagnostics and diagnostics[key] is not None:
            parts.append(f"{key}={format_value(diagnostics[key])}")
            seen.add(key)
    remaining = sorted(
        key for key in diagnostics if key not in seen and key not in hidden_metric_keys
    )
    for key in remaining:
        if diagnostics[key] is not None:
            parts.append(f"{key}={format_value(diagnostics[key])}")
    return " ".join(parts)


@dataclass
class _DiagnosticRecorder:
    """Own diagnostic timing, computation, reporting, and event recording."""

    state: SolverState
    context: AlgorithmContext
    request: DiagnosticRequest
    publish_snapshot: SnapshotPublisher
    max_iter: int
    started_at: float
    history: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    elapsed: float = 0.0
    last_tick: float = field(init=False)
    last_iteration: int = 0

    def __post_init__(self) -> None:
        self.last_tick = self.started_at

    def record(self, iteration: int) -> dict[str, Any]:
        now = time.perf_counter()
        interval_seconds = now - self.last_tick
        self.elapsed += interval_seconds
        iterations = max(iteration - self.last_iteration, 1)
        sec_per_iter = interval_seconds / iterations
        rem_sec = sec_per_iter * max(self.max_iter - iteration, 0)

        snapshot = self.publish_snapshot(
            self.state,
            self.context,
            iteration,
            self.elapsed,
            sec_per_iter,
            rem_sec,
            self.request,
        )
        report = compute_requested_diagnostics(
            DiagnosticContext(snapshot=snapshot, dataset=self.context.dataset),
            self.request,
        )
        self.last_tick = now
        self.last_iteration = iteration

        record = dict(report.metrics)
        record["iteration"] = iteration
        record.setdefault("elapsed", self.elapsed)
        record.setdefault("sec_per_iter", sec_per_iter)
        record.setdefault("rem_sec", rem_sec)
        print(_diagnostics_to_line(record))
        self.history.append(record)

        if report.unavailable or report.failures:
            self.events.append(
                {
                    "iteration": iteration,
                    "unavailable": dict(report.unavailable),
                    "failures": [asdict(failure) for failure in report.failures],
                }
            )
        return record


@dataclass
class _CheckpointRecorder:
    """Capture scheduled in-memory checkpoints for the current algorithm state."""

    state: SolverState
    scheduled: set[int]
    started_at: float
    capture: CheckpointCapture | None = None
    hook: CheckpointHook | None = None
    checkpoints: dict[int, AlgorithmCheckpoint] = field(default_factory=dict)

    def capture_if_due(self, iteration: int) -> None:
        if iteration not in self.scheduled:
            return
        if self.capture is None:
            raise TypeError("scheduled checkpoints require an algorithm-owned capture callback")
        elapsed = time.perf_counter() - self.started_at
        checkpoint = self.capture(self.state, elapsed)
        self.checkpoints[iteration] = checkpoint
        if self.hook is not None:
            self.hook(iteration, checkpoint)


@dataclass
class _SelectionRecorder:
    """Retain the best observed metric and its exact algorithm checkpoint."""

    state: SolverState
    policy: SelectionPolicy
    capture: CheckpointCapture | None
    selected: SelectedRunState | None = None

    def observe(self, record: dict[str, Any]) -> None:
        iteration = int(record["iteration"])
        raw_value = record.get(self.policy.metric)
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float, np.number)):
            raise ValueError(
                f"selection metric {self.policy.metric!r} is unavailable at iteration {iteration}"
            )
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(
                f"selection metric {self.policy.metric!r} is non-finite at iteration {iteration}"
            )
        if self.selected is not None:
            improves = (
                value < self.selected.value
                if self.policy.mode == "min"
                else value > self.selected.value
            )
            if not improves:
                return
        checkpoint = None
        if self.policy.retain_checkpoint:
            if self.capture is None:
                raise TypeError(
                    "selection checkpoint retention requires an algorithm capture callback"
                )
            elapsed = float(record.get("elapsed", 0.0))
            checkpoint = self.capture(self.state, elapsed)
        self.selected = SelectedRunState(
            policy=self.policy,
            iteration=iteration,
            value=value,
            diagnostics=dict(record),
            checkpoint=checkpoint,
        )


def run_bcd_iterations(
    *,
    state: SolverState,
    context: AlgorithmContext,
    update: UpdateFn,
    publish_snapshot: SnapshotPublisher,
    diagnostics_request: DiagnosticRequest | None = None,
    selection_policy: SelectionPolicy | None = None,
    capture_checkpoint: CheckpointCapture | None = None,
    checkpoint_hook: CheckpointHook | None = None,
) -> AlgorithmResult:
    """Run BCD updates, observations, stop decisions, and result assembly."""
    runtime = context.config.runtime
    request = diagnostics_request or default_diagnostics_request(state, context)
    started_at = time.perf_counter()
    diagnostics = _DiagnosticRecorder(
        state=state,
        context=context,
        request=request,
        publish_snapshot=publish_snapshot,
        max_iter=runtime.max_iter,
        started_at=started_at,
    )
    checkpoints = _CheckpointRecorder(
        state=state,
        scheduled=set(runtime.checkpoint_iters or ()),
        started_at=started_at,
        capture=capture_checkpoint,
        hook=checkpoint_hook,
    )
    selection = (
        None
        if selection_policy is None
        else _SelectionRecorder(
            state=state,
            policy=selection_policy,
            capture=capture_checkpoint,
        )
    )

    initial_record = diagnostics.record(0)
    if selection is not None:
        selection.observe(initial_record)
    checkpoints.capture_if_due(0)
    termination = RunTermination(reason="max_iterations", iteration=0)

    for iteration in range(1, runtime.max_iter + 1):
        update(state, context, iteration)
        checkpoints.capture_if_due(iteration)

        observed = _diagnostics_due(
            iteration,
            max_iter=runtime.max_iter,
            interval=runtime.diagnostics_interval,
        )
        if observed:
            record = diagnostics.record(iteration)
            if selection is not None:
                selection.observe(record)

        if _loss_converged(diagnostics.history, runtime.tol):
            if not observed:
                record = diagnostics.record(iteration)
                if selection is not None:
                    selection.observe(record)
            termination = RunTermination(reason="converged", iteration=iteration)
            break
        termination = RunTermination(reason="max_iterations", iteration=iteration)

    return AlgorithmResult(
        z=state.z,
        phi=state.phi,
        diagnostics=diagnostics.history,
        termination=termination,
        checkpoints=checkpoints.checkpoints,
        selection=None if selection is None else selection.selected,
        metadata={"diagnostic_events": diagnostics.events},
    )
