"""Representation-neutral persistence for algorithm-owned checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bcd_lab.atomic_io import atomic_binary_writer
from bcd_lab.types import AlgorithmCheckpoint

CHECKPOINT_FORMAT_VERSION = 1
_MANIFEST_KEY = "manifest"


def _encode(value: Any, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("checkpoint arrays cannot use object dtype")
        key = f"array_{len(arrays)}"
        arrays[key] = value
        return {"type": "array", "key": key}
    if isinstance(value, np.generic):
        return _encode(value.item(), arrays)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("checkpoint mapping keys must be strings")
        return {
            "type": "mapping",
            "items": {key: _encode(item, arrays) for key, item in value.items()},
        }
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_encode(item, arrays) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [_encode(item, arrays) for item in value]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"type": "scalar", "value": value}
    raise TypeError(f"Unsupported checkpoint value {type(value).__name__}")


def _decode(node: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> Any:
    node_type = node.get("type")
    if node_type == "array":
        key = node.get("key")
        if not isinstance(key, str) or key not in arrays:
            raise ValueError(f"Missing checkpoint array {key!r}")
        return np.asarray(arrays[key]).copy()
    if node_type == "mapping":
        items = node.get("items")
        if not isinstance(items, Mapping):
            raise ValueError("checkpoint mapping node requires items")
        decoded: dict[str, Any] = {}
        for key, value in items.items():
            if not isinstance(key, str) or not isinstance(value, Mapping):
                raise ValueError("checkpoint mapping items must be encoded mappings")
            decoded[key] = _decode(value, arrays)
        return decoded
    if node_type in {"tuple", "list"}:
        items = node.get("items")
        if not isinstance(items, list) or any(not isinstance(item, Mapping) for item in items):
            raise ValueError(f"checkpoint {node_type} node requires mapping items")
        decoded_items = [_decode(item, arrays) for item in items]
        return tuple(decoded_items) if node_type == "tuple" else decoded_items
    if node_type == "scalar":
        return node.get("value")
    raise ValueError(f"Unknown checkpoint node type {node_type!r}")


def save_checkpoint_archive(
    path: str | Path,
    checkpoints: Mapping[int, AlgorithmCheckpoint],
) -> Path:
    """Write versioned checkpoint envelopes and opaque component payloads."""
    target = Path(path)
    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for iteration, checkpoint in sorted(checkpoints.items()):
        if iteration < 0:
            raise ValueError("checkpoint iteration must be non-negative")
        if checkpoint.format_version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"Unsupported checkpoint format_version {checkpoint.format_version}")
        records.append(
            {
                "iteration": iteration,
                "format_version": checkpoint.format_version,
                "algorithm": checkpoint.algorithm,
                "state_version": checkpoint.state_version,
                "elapsed_seconds": checkpoint.elapsed_seconds,
                "payload": _encode(checkpoint.payload, arrays),
            }
        )
    manifest = json.dumps(
        {
            "archive_version": CHECKPOINT_FORMAT_VERSION,
            "checkpoints": records,
        },
        allow_nan=False,
        separators=(",", ":"),
    )
    with atomic_binary_writer(target) as file:
        np.savez_compressed(file, **{_MANIFEST_KEY: np.asarray(manifest), **arrays})
    return target


def _read_archive(source: Path) -> tuple[list[Any], dict[str, np.ndarray]]:
    with np.load(source, allow_pickle=False) as archive:
        if _MANIFEST_KEY not in archive:
            raise ValueError(f"Checkpoint archive {source} has no manifest")
        manifest = json.loads(str(np.asarray(archive[_MANIFEST_KEY]).item()))
        if not isinstance(manifest, Mapping):
            raise ValueError("checkpoint manifest must be a mapping")
        if manifest.get("archive_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint archive_version {manifest.get('archive_version')!r}"
            )
        records = manifest.get("checkpoints")
        if not isinstance(records, list):
            raise ValueError("checkpoint manifest requires a checkpoints list")
        arrays = {
            key: np.asarray(archive[key]).copy() for key in archive.files if key != _MANIFEST_KEY
        }
    return records, arrays


def _decode_record(
    record: Any,
    arrays: Mapping[str, np.ndarray],
) -> tuple[int, AlgorithmCheckpoint]:
    if not isinstance(record, Mapping):
        raise ValueError("checkpoint records must be mappings")
    iteration = record.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("checkpoint iteration must be a non-negative integer")
    payload_node = record.get("payload")
    if not isinstance(payload_node, Mapping):
        raise ValueError("checkpoint payload must be an encoded mapping")
    payload = _decode(payload_node, arrays)
    if not isinstance(payload, Mapping):
        raise ValueError("decoded checkpoint payload must be a mapping")
    format_version = record.get("format_version")
    algorithm = record.get("algorithm")
    state_version = record.get("state_version")
    elapsed_seconds = record.get("elapsed_seconds")
    if format_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format_version {format_version!r}")
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError("checkpoint algorithm must be a non-empty string")
    if isinstance(state_version, bool) or not isinstance(state_version, int) or state_version <= 0:
        raise ValueError("checkpoint state_version must be a positive integer")
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or elapsed_seconds < 0
    ):
        raise ValueError("checkpoint elapsed_seconds must be non-negative")
    return iteration, AlgorithmCheckpoint(
        format_version=format_version,
        algorithm=algorithm,
        state_version=state_version,
        elapsed_seconds=float(elapsed_seconds),
        payload=payload,
    )


def load_checkpoint_archive(path: str | Path) -> dict[int, AlgorithmCheckpoint]:
    """Load checkpoint envelopes without importing component implementations."""
    records, arrays = _read_archive(Path(path))
    checkpoints: dict[int, AlgorithmCheckpoint] = {}
    for record in records:
        iteration, checkpoint = _decode_record(record, arrays)
        if iteration in checkpoints:
            raise ValueError(f"duplicate checkpoint iteration {iteration}")
        checkpoints[iteration] = checkpoint
    return checkpoints
