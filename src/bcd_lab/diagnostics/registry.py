"""Diagnostic family registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol

from bcd_lab._registry import ComponentRegistry
from bcd_lab.types import DatasetBundle

from .snapshot import RunSnapshot


@dataclass
class DiagnosticContext:
    """Inputs shared by diagnostic families for one diagnostic pass."""

    snapshot: RunSnapshot
    dataset: DatasetBundle | None = None
    cache: dict[str, Any] = field(default_factory=dict)

    @property
    def iteration(self) -> int:
        """Iteration represented by the published snapshot."""
        return self.snapshot.iteration


class DiagnosticFamily(Protocol):
    """A grouped diagnostic computation with shared intermediate values."""

    name: str

    def compute(
        self,
        ctx: DiagnosticContext,
        *,
        metrics: set[str] | None = None,
    ) -> Mapping[str, float | int | str | None]: ...


DiagnosticErrorPolicy = Literal["raise", "record"]


@dataclass(frozen=True)
class DiagnosticRequest:
    """Families to compute and the policy for unexpected computation defects."""

    families: Mapping[str, set[str] | None]
    on_error: DiagnosticErrorPolicy = "raise"

    def __post_init__(self) -> None:
        if self.on_error not in {"raise", "record"}:
            raise ValueError("diagnostic on_error must be 'raise' or 'record'")


@dataclass(frozen=True)
class DiagnosticFailure:
    """A diagnostic defect recorded by an explicitly non-strict request."""

    family: str
    metrics: tuple[str, ...] | None
    error_type: str
    message: str


class DiagnosticComputationError(RuntimeError):
    """A strict diagnostic failure with the original defect as its cause."""

    def __init__(self, family: str, cause: Exception) -> None:
        super().__init__(f"Diagnostic family {family!r} failed: {cause}")
        self.family = family


@dataclass(frozen=True)
class DiagnosticReport:
    """Computed values plus explicit unavailable and failed diagnostics."""

    metrics: dict[str, float | int | str | None]
    unavailable: dict[str, str]
    failures: tuple[DiagnosticFailure, ...]


_FAMILIES = ComponentRegistry[DiagnosticFamily](
    "Diagnostic family",
    "diagnostic families",
)


def register_diagnostic_family(family: DiagnosticFamily) -> DiagnosticFamily:
    """Register and return a diagnostic family."""
    return _FAMILIES.register(family.name, family)


def get_diagnostic_family(name: str) -> DiagnosticFamily:
    """Return a diagnostic family by name."""
    return _FAMILIES.get(name)


def list_diagnostic_families() -> tuple[str, ...]:
    """List registered diagnostic family names."""
    return _FAMILIES.names()


def compute_requested_diagnostics(
    ctx: DiagnosticContext,
    request: DiagnosticRequest,
) -> DiagnosticReport:
    """Compute diagnostics with an explicit policy for unavailable values and defects."""
    out: dict[str, float | int | str | None] = {}
    unavailable: dict[str, str] = {}
    failures: list[DiagnosticFailure] = []
    for family_name, metric_names in request.families.items():
        try:
            values = get_diagnostic_family(family_name).compute(
                ctx,
                metrics=metric_names,
            )
        except Exception as exc:
            if request.on_error == "raise":
                raise DiagnosticComputationError(family_name, exc) from exc
            failures.append(
                DiagnosticFailure(
                    family=family_name,
                    metrics=None if metric_names is None else tuple(sorted(metric_names)),
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            continue
        for metric_name, value in values.items():
            out[metric_name] = value
            if value is None:
                unavailable[metric_name] = (
                    "required inputs or mathematical preconditions are unavailable"
                )
    return DiagnosticReport(
        metrics=out,
        unavailable=unavailable,
        failures=tuple(failures),
    )
