"""Versioned, replay-oriented experiment result artifacts."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from bcd_lab.atomic_io import atomic_text_writer

from .loader import parse_experiment_spec
from .schema import ExperimentSpec

ARTIFACT_VERSION = 2
_DEPENDENCIES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "PyYAML",
    "matplotlib",
    "cvxpy",
    "pandas",
    "xlrd",
)


def json_safe(value: Any) -> Any:
    """Convert supported configuration values to strict JSON primitives."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"Non-finite value {value!r} cannot be saved as JSON")
        return value
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: json_safe(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key must be a string, got {type(key).__name__}")
            converted[key] = json_safe(item)
        return converted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    raise TypeError(f"Unsupported value of type {type(value).__name__} in result artifact")


def serialize_experiment_spec(spec: ExperimentSpec) -> dict[str, Any]:
    """Serialize a complete resolved experiment specification."""
    serialized = json_safe(spec)
    if not isinstance(serialized, dict):  # pragma: no cover - structural invariant
        raise TypeError("ExperimentSpec must serialize to a JSON object")
    return serialized


def deserialize_experiment_spec(data: Mapping[str, Any]) -> ExperimentSpec:
    """Reconstruct and validate an ExperimentSpec saved in an artifact."""
    return parse_experiment_spec(dict(data), source="result artifact experiment_spec")


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _git_provenance() -> tuple[str | None, bool | None]:
    package_path = Path(__file__).resolve().parent
    try:
        revision = subprocess.run(
            ["git", "-C", str(package_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(package_path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return None, None
    return revision or None, bool(status.strip())


def collect_provenance() -> dict[str, Any]:
    """Collect versions and source-control state needed to interpret a run."""
    revision, dirty = _git_provenance()
    return {
        "package_version": _distribution_version("bcd-lab"),
        "git_revision": revision,
        "git_dirty": dirty,
        "python_version": platform.python_version(),
        "dependencies": {name: _distribution_version(name) for name in _DEPENDENCIES},
    }


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Write an artifact with strict JSON validation."""
    serialized = json_safe(artifact)
    with atomic_text_writer(path) as file:
        json.dump(serialized, file, indent=2, allow_nan=False)
        file.write("\n")
