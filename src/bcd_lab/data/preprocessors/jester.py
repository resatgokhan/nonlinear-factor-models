"""Exact conversion of the first Jester Dataset 1 spreadsheet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bcd_lab.data.provenance import summarize_identifiers

from .base import DataPreprocessor, ExpectedDatasetCounts

_JESTER_DATASET_1_COUNTS = ExpectedDatasetCounts(24_983, 100, 1_810_455)


class JesterDataset1(DataPreprocessor):
    """Jester Dataset 1 part 1: 24,983 users and 100 jokes."""

    name = "jester-dataset-1"
    description = "Jester Dataset 1 spreadsheet (24,983 users, 100 jokes)"

    def __init__(
        self,
        expected: ExpectedDatasetCounts = _JESTER_DATASET_1_COUNTS,
    ) -> None:
        self.expected = expected

    def preprocess(self, raw_dir: Path, out_dir: Path, **kwargs: Any) -> Path:
        if kwargs:
            raise ValueError(f"{self.name} does not accept preprocessing parameters")
        source_id = "jester/jester-data-1.xls"
        src = raw_dir / source_id
        if not src.exists():
            raise FileNotFoundError(
                f"Expected {src}\nDownload Dataset 1 part 1 from "
                "https://eigentaste.berkeley.edu/dataset/"
            )

        try:
            matrix = pd.read_excel(src, header=None, engine="xlrd").to_numpy(dtype=float)
        except (ImportError, ValueError) as exc:
            raise ValueError(f"could not read Jester spreadsheet {src}: {exc}") from exc
        expected_width = self.expected.n_columns + 1
        if matrix.shape != (self.expected.n_rows, expected_width):
            raise ValueError(
                f"{self.name} expected a {self.expected.n_rows} x {expected_width} matrix; "
                f"found {matrix.shape}"
            )

        declared_counts = matrix[:, 0]
        if not np.all(np.isfinite(declared_counts)) or not np.all(
            declared_counts == np.floor(declared_counts)
        ):
            raise ValueError("Jester's first column must contain integer rated-joke counts")
        ratings = matrix[:, 1:]
        sentinel_mask = ratings == 99.0
        blank_mask = np.isnan(ratings)
        observed_mask = ~(sentinel_mask | blank_mask)
        observed = ratings[observed_mask]
        if not np.all(np.isfinite(observed)) or np.any((observed < -10.0) | (observed > 10.0)):
            raise ValueError("Jester observed ratings must be finite and in [-10, 10]")
        actual_per_row = observed_mask.sum(axis=1)
        if not np.array_equal(declared_counts.astype(int), actual_per_row):
            mismatch = int(np.flatnonzero(declared_counts.astype(int) != actual_per_row)[0])
            raise ValueError(
                f"Jester rated-joke count disagrees with row {mismatch}'s observed ratings"
            )
        if observed.size != self.expected.n_observations:
            raise ValueError(
                f"{self.name} expected {self.expected.n_observations} ratings; "
                f"found {observed.size}"
            )

        row_indices, column_indices = np.nonzero(observed_mask)
        source_rows = row_indices.tolist()
        source_columns = (column_indices + 1).tolist()
        values = ratings[observed_mask].tolist()
        return self.write_preprocessed_triples(
            out_dir,
            self.name,
            source_rows,
            source_columns,
            values,
            parameters={
                "dataset": "Jester Dataset 1 part 1",
                "input_format": "Excel 97-2003 matrix",
                "rated_count_column": 0,
                "joke_id_offset": 1,
                "missing_sentinel": 99.0,
                "rating_range": [-10.0, 10.0],
                "source_user_id": "zero-based spreadsheet row",
            },
            source_paths=[(source_id, src)],
            counts={
                "source_rows": self.expected.n_rows,
                "source_columns": self.expected.n_columns,
                "source_cells": int(ratings.size),
                "output_observations": int(observed.size),
                "missing_sentinel_removed": int(sentinel_mask.sum()),
                "blank_cells_removed": int(blank_mask.sum()),
            },
            identifiers={
                "rows": summarize_identifiers(
                    range(self.expected.n_rows), source_field="spreadsheet_row"
                ),
                "columns": summarize_identifiers(
                    range(1, self.expected.n_columns + 1), source_field="joke_id"
                ),
            },
        )
