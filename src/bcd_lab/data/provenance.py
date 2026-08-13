"""Versioned provenance for canonical CSVs produced from raw datasets."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bcd_lab.atomic_io import atomic_text_writer
from bcd_lab.types import (
    IdentifierSummary,
    PreprocessingProvenance,
    PreprocessingSource,
    SourceFingerprint,
)


def fingerprint_file(path: Path) -> SourceFingerprint:
    """Return a streaming SHA-256 and byte count for ``path``."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return SourceFingerprint(sha256=digest.hexdigest(), size_bytes=size)


def summarize_identifiers(
    values: Sequence[str | int],
    *,
    source_field: str,
) -> IdentifierSummary:
    """Describe an identifier set without copying the full mapping into metadata."""
    unique = sorted({str(value) for value in values}, key=lambda value: (len(value), value))
    if not unique:
        raise ValueError(f"{source_field} identifiers must be non-empty")
    encoded = "\n".join(unique).encode("utf-8")
    return IdentifierSummary(
        source_field=source_field,
        count=len(unique),
        minimum=unique[0],
        maximum=unique[-1],
        sha256=hashlib.sha256(encoded).hexdigest(),
    )


def provenance_path(csv_path: Path) -> Path:
    """Return the canonical sidecar path for a preprocessed CSV."""
    return csv_path.with_suffix(f"{csv_path.suffix}.provenance.json")


def write_preprocessing_provenance(
    csv_path: Path,
    *,
    preprocessor: str,
    preprocessor_version: int,
    parameters: Mapping[str, Any],
    source_paths: Sequence[tuple[str, Path]],
    counts: Mapping[str, int],
    identifiers: Mapping[str, IdentifierSummary],
) -> PreprocessingProvenance:
    """Fingerprint raw/output files and atomically write the CSV sidecar."""
    provenance = PreprocessingProvenance(
        schema_version=1,
        preprocessor=preprocessor,
        preprocessor_version=preprocessor_version,
        parameters=dict(parameters),
        sources=tuple(
            PreprocessingSource(source_id=source_id, fingerprint=fingerprint_file(path))
            for source_id, path in source_paths
        ),
        output_name=csv_path.name,
        output_fingerprint=fingerprint_file(csv_path),
        counts=dict(counts),
        identifiers=dict(identifiers),
    )
    with atomic_text_writer(provenance_path(csv_path)) as stream:
        json.dump(dataclasses.asdict(provenance), stream, indent=2, sort_keys=True)
        stream.write("\n")
    return provenance


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"preprocessing provenance {field} must be a mapping")
    return value


def _parse_fingerprint(value: object, *, field: str) -> SourceFingerprint:
    mapping = _require_mapping(value, field=field)
    try:
        return SourceFingerprint(
            sha256=str(mapping["sha256"]),
            size_bytes=int(mapping["size_bytes"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid preprocessing provenance {field}") from exc


def load_preprocessing_provenance(csv_path: Path) -> PreprocessingProvenance | None:
    """Load an adjacent sidecar and verify that it describes the current CSV."""
    sidecar = provenance_path(csv_path)
    if not sidecar.exists():
        return None
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid preprocessing provenance file {sidecar}") from exc
    root = _require_mapping(raw, field="root")
    try:
        raw_sources = root["sources"]
        if not isinstance(raw_sources, list):
            raise TypeError("sources must be a list")
        sources = tuple(
            PreprocessingSource(
                source_id=str(_require_mapping(item, field="source")["source_id"]),
                fingerprint=_parse_fingerprint(
                    _require_mapping(item, field="source")["fingerprint"],
                    field="source fingerprint",
                ),
            )
            for item in raw_sources
        )
        raw_identifiers = _require_mapping(root["identifiers"], field="identifiers")
        identifiers = {
            str(name): IdentifierSummary(**dict(_require_mapping(value, field="identifier")))
            for name, value in raw_identifiers.items()
        }
        provenance = PreprocessingProvenance(
            schema_version=int(root["schema_version"]),
            preprocessor=str(root["preprocessor"]),
            preprocessor_version=int(root["preprocessor_version"]),
            parameters=dict(_require_mapping(root["parameters"], field="parameters")),
            sources=sources,
            output_name=str(root["output_name"]),
            output_fingerprint=_parse_fingerprint(root["output_fingerprint"], field="output"),
            counts={
                str(name): int(value)
                for name, value in _require_mapping(root["counts"], field="counts").items()
            },
            identifiers=identifiers,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid preprocessing provenance file {sidecar}") from exc
    if provenance.output_name != csv_path.name:
        raise ValueError("preprocessing provenance output_name does not match the CSV")
    if provenance.output_fingerprint != fingerprint_file(csv_path):
        raise ValueError("preprocessed CSV does not match its provenance fingerprint")
    return provenance
