"""Sweep runner schema configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bcd_lab.types import SelectionPolicy


@dataclass(frozen=True)
class ComponentSpec:
    """A registry name and the parameters owned by that component."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectiveSpec:
    """A registered objective name and objective-owned parameters."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SweepAxis:
    path: str
    values: list[Any]


@dataclass
class SweepSection:
    axes: list[SweepAxis] = field(default_factory=list)


@dataclass
class ExperimentSpec:
    schema_version: int
    experiment: str
    algorithm: ComponentSpec
    objective: ObjectiveSpec
    trials: int
    seed: int
    data: dict[str, Any]
    link: dict[str, Any]
    runtime: dict[str, Any]
    sweep: SweepSection = field(default_factory=SweepSection)
    initialization: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    selection: SelectionPolicy | None = None
    plot: dict[str, Any] = field(default_factory=dict)
