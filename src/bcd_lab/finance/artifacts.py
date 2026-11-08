"""Versioned finance records with pickle-free arrays and detached model checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bcd_lab.atomic_io import atomic_binary_writer, atomic_text_writer
from bcd_lab.experiments.artifacts import json_safe

FINANCE_ARTIFACT_VERSION = 1


def write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(json_safe(payload), allow_nan=False, indent=2)
    with atomic_text_writer(path) as stream:
        stream.write(text + "\n")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("finance_artifact_version") != FINANCE_ARTIFACT_VERSION:
        raise ValueError(f"Unsupported finance artifact version: {path}")
    return payload


def write_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    for name, value in arrays.items():
        if value.dtype.hasobject or not np.isfinite(value).all():
            raise ValueError(f"Artifact array {name} must be finite and pickle-free")
    with atomic_binary_writer(path) as stream:
        np.savez_compressed(stream, **arrays)


def read_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}
