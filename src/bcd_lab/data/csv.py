"""Sparse observed triples from CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bcd_lab.math import estimate_empirical_mu
from bcd_lab.randomness import RunRandomness
from bcd_lab.types import (
    DatasetBundle,
    DatasetCapabilities,
    IdMappings,
    PreprocessingProvenance,
    SourceFingerprint,
    SplitStatistics,
)

from .base import DatasetLoader
from .provenance import fingerprint_file, load_preprocessing_provenance


def _verify_fingerprint(
    actual: SourceFingerprint,
    expected: object,
) -> None:
    if expected is None:
        return
    if not isinstance(expected, Mapping):
        raise ValueError("data.source_fingerprint must be a mapping")
    sha256 = expected.get("sha256")
    size_bytes = expected.get("size_bytes")
    if not isinstance(sha256, str) or (
        isinstance(size_bytes, bool) or not isinstance(size_bytes, int)
    ):
        raise ValueError("data.source_fingerprint requires string sha256 and integer size_bytes")
    expected_fingerprint = SourceFingerprint(
        sha256=sha256,
        size_bytes=size_bytes,
    )
    if actual != expected_fingerprint:
        raise ValueError(
            "CSV content does not match data.source_fingerprint; the source changed since the run was recorded"
        )


def _summary_stats(counts: np.ndarray) -> dict[str, float]:
    arr = np.asarray(counts, dtype=float)
    if arr.size == 0:
        return {
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p75": 0.0,
            "max": 0.0,
        }
    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def _read_csv_triples(
    path: str,
    row_col: str | int,
    col_col: str | int,
    value_col: str | int,
    has_header: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    csv_path = Path(path)
    rows: list[str] = []
    cols: list[str] = []
    values: list[float] = []
    with csv_path.open("r", newline="") as handle:
        if has_header:
            dict_reader = csv.DictReader(handle)
            for record in dict_reader:
                if record is None:
                    continue
                rows.append(str(record[str(row_col)]))
                cols.append(str(record[str(col_col)]))
                values.append(float(record[str(value_col)]))
        else:
            row_reader = csv.reader(handle)
            row_idx = int(row_col)
            col_idx = int(col_col)
            value_idx = int(value_col)
            for row_record in row_reader:
                rows.append(str(row_record[row_idx]))
                cols.append(str(row_record[col_idx]))
                values.append(float(row_record[value_idx]))
    if not rows:
        raise ValueError(f"No observations found in CSV {csv_path}")
    return (
        np.asarray(rows, dtype=object),
        np.asarray(cols, dtype=object),
        np.asarray(values, dtype=float),
    )


def _top_or_random_ids(
    ids: np.ndarray,
    *,
    limit: int | None,
    strategy: str,
    rng: np.random.Generator,
) -> set[object]:
    unique_ids, counts = np.unique(ids, return_counts=True)
    if limit is None or limit >= unique_ids.size:
        return set(unique_ids.tolist())
    if limit <= 0:
        raise ValueError("CSV size limits must be positive when provided")
    if strategy == "top":
        order = np.lexsort((unique_ids.astype(str), -counts))
        chosen = unique_ids[order[:limit]]
        return set(chosen.tolist())
    if strategy == "random":
        chosen = rng.choice(unique_ids, size=limit, replace=False)
        return set(chosen.tolist())
    raise ValueError(f"Unsupported data.csv.size.strategy {strategy!r}")


def _split_train_validation(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    validation_fraction: float,
    rng: np.random.Generator,
    stratify_by: str,
) -> tuple[np.ndarray, np.ndarray]:
    total = rows.size
    if validation_fraction <= 0.0:
        return np.ones(total, dtype=bool), np.zeros(total, dtype=bool)
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("data.csv.split.validation_fraction must be in [0, 1)")
    if stratify_by != "row":
        raise ValueError("Only data.csv.split.stratify_by='row' is supported in the first pass")

    train_mask = np.ones(total, dtype=bool)
    validation_mask = np.zeros(total, dtype=bool)

    unique_rows = np.unique(rows)
    for row_id in unique_rows:
        row_indices = np.flatnonzero(rows == row_id)
        if row_indices.size <= 1:
            continue
        validation_count = int(round(row_indices.size * validation_fraction))
        validation_count = min(row_indices.size - 1, validation_count)
        if validation_count <= 0:
            continue
        chosen = rng.choice(row_indices, size=validation_count, replace=False)
        validation_mask[chosen] = True
        train_mask[chosen] = False

    train_col_counts = np.bincount(
        cols[train_mask], minlength=int(cols.max()) + 1 if cols.size else 0
    )
    for idx in np.flatnonzero(validation_mask):
        col = int(cols[idx])
        if train_col_counts[col] == 0:
            validation_mask[idx] = False
            train_mask[idx] = True
            train_col_counts[col] += 1

    if not train_mask.any():
        raise ValueError("CSV split left no training observations")
    return train_mask, validation_mask


def _dense_remap(values: np.ndarray) -> tuple[np.ndarray, dict[object, int]]:
    unique_vals = np.unique(values)
    mapping = {val: idx for idx, val in enumerate(unique_vals.tolist())}
    dense = np.asarray([mapping[val] for val in values.tolist()], dtype=int)
    return dense, mapping


def _validate_dense_zero_based_ids(values: np.ndarray, *, label: str) -> np.ndarray:
    dense = values.astype(int)
    if dense.size == 0:
        return dense
    unique_vals = np.unique(dense)
    expected = np.arange(unique_vals.size, dtype=int)
    if unique_vals[0] != 0 or not np.array_equal(unique_vals, expected):
        raise ValueError(
            f"data.csv.remap_ids=false requires {label} ids to already be dense zero-based"
        )
    return dense


@dataclass(frozen=True)
class _CsvSettings:
    """Validated values needed across the CSV loading phases."""

    path: Path
    fingerprint: SourceFingerprint
    preprocessing: PreprocessingProvenance | None
    rank: int
    row_col: str | int
    col_col: str | int
    value_col: str | int
    has_header: bool
    remap_ids: bool
    row_limit: int | None
    col_limit: int | None
    size_strategy: str
    validation_fraction: float
    stratify_by: str


@dataclass(frozen=True)
class _SelectedTriples:
    rows: np.ndarray
    columns: np.ndarray
    values: np.ndarray
    row_mapping: dict[object, int]
    column_mapping: dict[object, int]


def _parse_settings(params: Mapping[str, Any]) -> _CsvSettings:
    csv_path = Path(str(params["path"])).expanduser()
    if not csv_path.is_absolute():
        raise ValueError(
            "data.path must be absolute; experiment-relative paths are resolved "
            "when the configuration is loaded"
        )
    path = csv_path.resolve()
    fingerprint = fingerprint_file(path)
    _verify_fingerprint(fingerprint, params.get("source_fingerprint"))
    preprocessing = load_preprocessing_provenance(path)

    size = params.get("size") or {}
    split = params.get("split") or {}
    if "seed" in split:
        raise ValueError(
            "data.csv.split.seed is not configurable; splits derive from the experiment seed"
        )
    return _CsvSettings(
        path=path,
        fingerprint=fingerprint,
        preprocessing=preprocessing,
        rank=int(params["shape"]["rank"]),
        row_col=params.get("row_col", "row"),
        col_col=params.get("col_col", "col"),
        value_col=params.get("value_col", "value"),
        has_header=bool(params.get("has_header", True)),
        remap_ids=bool(params.get("remap_ids", True)),
        row_limit=size.get("max_rows"),
        col_limit=size.get("max_cols"),
        size_strategy=size.get("strategy", "top"),
        validation_fraction=float(split.get("validation_fraction", 0.0)),
        stratify_by=str(split.get("stratify_by", "row")),
    )


def _select_triples(settings: _CsvSettings, randomness: RunRandomness) -> _SelectedTriples:
    raw_rows, raw_columns, raw_values = _read_csv_triples(
        str(settings.path),
        settings.row_col,
        settings.col_col,
        settings.value_col,
        settings.has_header,
    )
    keep_rows = _top_or_random_ids(
        raw_rows,
        limit=settings.row_limit,
        strategy=settings.size_strategy,
        rng=randomness.generator("row_selection"),
    )
    keep_columns = _top_or_random_ids(
        raw_columns,
        limit=settings.col_limit,
        strategy=settings.size_strategy,
        rng=randomness.generator("column_selection"),
    )
    mask = np.fromiter(
        (
            row in keep_rows and column in keep_columns
            for row, column in zip(raw_rows, raw_columns, strict=True)
        ),
        dtype=bool,
        count=raw_rows.size,
    )
    if not mask.any():
        raise ValueError("CSV size selection retained no observations")

    rows = raw_rows[mask]
    columns = raw_columns[mask]
    values = raw_values[mask]
    if settings.remap_ids:
        dense_rows, row_mapping = _dense_remap(rows)
        dense_columns, column_mapping = _dense_remap(columns)
    else:
        dense_rows = _validate_dense_zero_based_ids(rows, label="row")
        dense_columns = _validate_dense_zero_based_ids(columns, label="column")
        row_mapping = {int(value): int(value) for value in np.unique(dense_rows)}
        column_mapping = {int(value): int(value) for value in np.unique(dense_columns)}
    return _SelectedTriples(
        rows=dense_rows,
        columns=dense_columns,
        values=values,
        row_mapping=row_mapping,
        column_mapping=column_mapping,
    )


def _build_bundle(
    settings: _CsvSettings,
    selected: _SelectedTriples,
    randomness: RunRandomness,
) -> DatasetBundle:
    split_seed = randomness.child_seed("validation_split")
    train_mask, validation_mask = _split_train_validation(
        selected.rows,
        selected.columns,
        validation_fraction=settings.validation_fraction,
        rng=np.random.default_rng(split_seed),
        stratify_by=settings.stratify_by,
    )
    train_rows = selected.rows[train_mask]
    train_columns = selected.columns[train_mask]
    train_values = selected.values[train_mask].astype(np.float64, copy=False)
    validation_rows = selected.rows[validation_mask]
    validation_columns = selected.columns[validation_mask]
    validation_values = selected.values[validation_mask].astype(np.float64, copy=False)
    n_rows = int(np.unique(selected.rows).size)
    n_columns = int(np.unique(selected.columns).size)
    mu, mu_metadata = estimate_empirical_mu(
        n1=n_rows,
        n2=n_columns,
        rank=settings.rank,
        rows=train_rows,
        cols=train_columns,
        values=train_values,
    )
    validation = (
        None
        if validation_values.size == 0
        else (validation_values, (validation_rows, validation_columns))
    )
    return DatasetBundle(
        source="csv",
        n_rows=n_rows,
        n_cols=n_columns,
        rank=settings.rank,
        train=(train_values, (train_rows, train_columns)),
        validation=validation,
        capabilities=DatasetCapabilities(
            splits=SplitStatistics(
                train_count=int(train_values.size),
                validation_count=int(validation_values.size),
                total_count=int(selected.values.size),
                validation_fraction=settings.validation_fraction,
                seed=split_seed,
                stratify_by=settings.stratify_by,
                row_count_summary=_summary_stats(np.bincount(selected.rows, minlength=n_rows)),
                column_count_summary=_summary_stats(
                    np.bincount(selected.columns, minlength=n_columns)
                ),
            ),
            id_mappings=IdMappings(
                row_to_index=selected.row_mapping,
                column_to_index=selected.column_mapping,
            ),
            source_fingerprint=settings.fingerprint,
            preprocessing=settings.preprocessing,
        ),
        metadata={
            "dataset_name": settings.path.stem,
            "csv_path": str(settings.path),
            "size": {
                "max_rows": settings.row_limit,
                "max_cols": settings.col_limit,
                "strategy": settings.size_strategy,
            },
            "mu_estimate": {"value": float(mu), **mu_metadata},
        },
    )


class CsvDatasetLoader(DatasetLoader):
    """Loads sparse observed datasets from CSV files."""

    name = "csv"

    def load(
        self,
        *,
        params: dict[str, Any],
        seed: int,
    ) -> DatasetBundle:
        """Load sparse ratings from a CSV file.

        Args:
            params: Parameters including path, row_col, col_col, value_col,
                has_header, remap_ids, size (max_rows, max_cols, strategy),
                split (validation_fraction, stratify_by), and
                ``shape.rank``.
            seed: Random seed.

        Returns:
            A DatasetBundle containing train/validation sample sets and metadata.
        """
        settings = _parse_settings(params)
        randomness = RunRandomness(seed)
        selected = _select_triples(settings, randomness)
        return _build_bundle(settings, selected, randomness)
