"""Base class for dataset preprocessors."""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bcd_lab.atomic_io import atomic_text_writer
from bcd_lab.data.provenance import write_preprocessing_provenance
from bcd_lab.types import IdentifierSummary


@dataclass(frozen=True)
class ExpectedDatasetCounts:
    """Expected boundary counts for an authoritative raw dataset release."""

    n_rows: int
    n_columns: int
    n_observations: int

    def __post_init__(self) -> None:
        if min(self.n_rows, self.n_columns, self.n_observations) <= 0:
            raise ValueError("expected dataset counts must be positive")


class DataPreprocessor(ABC):
    """Base class for dataset preprocessors.

    A preprocessor reads raw files from ``raw_dir`` and writes a clean CSV
    of (row, col, value) triples to ``out_dir / <name>.csv``.

    Subclasses must implement :meth:`preprocess`.  They can call
    :meth:`write_triples` to handle writing and reporting without duplicating
    boilerplate.
    """

    name: str
    description: str

    @abstractmethod
    def preprocess(
        self,
        raw_dir: Path,
        out_dir: Path,
        **kwargs: object,
    ) -> Path:
        """Transform raw data into a clean triple CSV.

        Args:
            raw_dir: Root directory of raw downloaded files.
            out_dir: Output directory for clean CSV files (typically ``./data/``).
            **kwargs: Preprocessor-specific options.

        Returns:
            Path of the written clean CSV file.
        """
        ...

    def write_triples(
        self,
        out_dir: Path,
        name: str,
        rows: Sequence[str | int],
        cols: Sequence[str | int],
        values: Sequence[float],
    ) -> Path:
        """Write (rows, cols, values) to ``out_dir/<name>.csv`` and print a summary.

        Args:
            out_dir:  Output directory (created if absent).
            name:     Stem of the output file, e.g. ``"ml-100k"``.
            rows:     Source row identifiers.
            cols:     Source column identifiers.
            values:   Observed values.

        Returns:
            Path of the written CSV file.
        """
        if not values or not (len(rows) == len(cols) == len(values)):
            raise ValueError("preprocessed triples must be non-empty and aligned")
        if not np.all(np.isfinite(np.asarray(values, dtype=float))):
            raise ValueError("preprocessed values must be finite")
        out_path = out_dir / f"{name}.csv"
        with atomic_text_writer(out_path) as f:
            writer = csv.writer(f)
            writer.writerow(["row", "col", "value"])
            for r, c, v in zip(rows, cols, values, strict=True):
                writer.writerow([r, c, v])

        n_rows = len(set(rows))
        n_cols = len(set(cols))
        print(
            f"  Written: {out_path}\n"
            f"  Observations : {len(values):,}\n"
            f"  Unique rows  : {n_rows:,}\n"
            f"  Unique cols  : {n_cols:,}\n"
            f"  Value range  : [{min(values):.4g}, {max(values):.4g}]"
        )
        return out_path

    def write_preprocessed_triples(
        self,
        out_dir: Path,
        name: str,
        rows: Sequence[str | int],
        cols: Sequence[str | int],
        values: Sequence[float],
        *,
        parameters: Mapping[str, Any],
        source_paths: Sequence[tuple[str, Path]],
        counts: Mapping[str, int],
        identifiers: Mapping[str, IdentifierSummary],
    ) -> Path:
        """Write canonical triples plus their versioned preprocessing sidecar."""
        out_path = self.write_triples(out_dir, name, rows, cols, values)
        write_preprocessing_provenance(
            out_path,
            preprocessor=self.name,
            preprocessor_version=1,
            parameters=parameters,
            source_paths=source_paths,
            counts=counts,
            identifiers=identifiers,
        )
        print(f"  Provenance  : {out_path.name}.provenance.json")
        return out_path
