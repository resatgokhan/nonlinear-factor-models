"""MovieLens dataset preprocessors.

Supported variants
------------------
ml-100k
    Raw layout: ``data-raw/ml-100k/u.data``
    Tab-separated: user_id \\t item_id \\t rating \\t timestamp
    Writes:  ``data/ml-100k.csv``

ml-1m
    Raw layout: ``data-raw/ml-1m/ratings.dat``
    Double-colon separated: UserID::MovieID::Rating::Timestamp
    Writes:  ``data/ml-1m.csv``

ml-10m
    Raw layout: ``data-raw/ml-10m/ratings.dat``  (same format as ml-1m)
    Writes:  ``data/ml-10m.csv``

ml-20m
    Raw layout: ``data-raw/ml-20m/ratings.csv``
    Comma-separated with header: userId,movieId,rating,timestamp
    Writes:  ``data/ml-20m.csv``

movielens-latest-small
    Raw layout: ``data-raw/ml-latest-small/ratings.csv``
    Validates the pinned September 2018 release: 610 users, 9,724 movies, and
    100,836 ratings. Source user/movie IDs are retained in the clean CSV.

All output files have a header row ``row,col,value``. Historical preprocessors
emit dense IDs; latest-small retains authoritative source IDs and lets the CSV
loader perform its run-local dense mapping.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from bcd_lab.data.provenance import summarize_identifiers

from .base import DataPreprocessor, ExpectedDatasetCounts

_LATEST_SMALL_COUNTS = ExpectedDatasetCounts(610, 9_724, 100_836)

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _dense_remap(ids: list[str]) -> tuple[list[int], dict[str, int]]:
    """Map arbitrary string IDs to consecutive zero-based integers."""
    unique = sorted(set(ids))
    mapping = {uid: idx for idx, uid in enumerate(unique)}
    return [mapping[uid] for uid in ids], mapping


# ---------------------------------------------------------------------------
# ml-100k
# ---------------------------------------------------------------------------


class MovieLens100K(DataPreprocessor):
    """MovieLens 100K — u.data (tab-separated, no header)."""

    name = "ml-100k"
    description = "MovieLens 100K (u.data, tab-separated)"

    def preprocess(self, raw_dir: Path, out_dir: Path, **kwargs: Any) -> Path:
        src = raw_dir / "ml-100k" / "u.data"
        if not src.exists():
            raise FileNotFoundError(
                f"Expected {src}\n"
                "Download: https://files.grouplens.org/datasets/movielens/ml-100k.zip"
            )

        raw_rows, raw_cols, raw_vals = [], [], []
        with src.open("r", newline="") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                raw_rows.append(parts[0])
                raw_cols.append(parts[1])
                raw_vals.append(float(parts[2]))

        rows, _ = _dense_remap(raw_rows)
        cols, _ = _dense_remap(raw_cols)
        return self.write_triples(out_dir, self.name, rows, cols, raw_vals)


# ---------------------------------------------------------------------------
# ml-1m
# ---------------------------------------------------------------------------


class MovieLens1M(DataPreprocessor):
    """MovieLens 1M — ratings.dat (double-colon separated, no header)."""

    name = "ml-1m"
    description = "MovieLens 1M (ratings.dat, double-colon separated)"

    def preprocess(self, raw_dir: Path, out_dir: Path, **kwargs: Any) -> Path:
        src = raw_dir / "ml-1m" / "ratings.dat"
        if not src.exists():
            raise FileNotFoundError(
                f"Expected {src}\n"
                "Download: https://files.grouplens.org/datasets/movielens/ml-1m.zip"
            )

        raw_rows, raw_cols, raw_vals = [], [], []
        with src.open("r", newline="", encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("::")
                if len(parts) < 3:
                    continue
                raw_rows.append(parts[0])
                raw_cols.append(parts[1])
                raw_vals.append(float(parts[2]))

        rows, _ = _dense_remap(raw_rows)
        cols, _ = _dense_remap(raw_cols)
        return self.write_triples(out_dir, self.name, rows, cols, raw_vals)


# ---------------------------------------------------------------------------
# ml-10m
# ---------------------------------------------------------------------------


class MovieLens10M(DataPreprocessor):
    """MovieLens 10M — ratings.dat (double-colon separated, no header)."""

    name = "ml-10m"
    description = "MovieLens 10M (ratings.dat, double-colon separated)"

    def preprocess(self, raw_dir: Path, out_dir: Path, **kwargs: Any) -> Path:
        src = raw_dir / "ml-10m" / "ratings.dat"
        if not src.exists():
            raise FileNotFoundError(
                f"Expected {src}\n"
                "Download: https://files.grouplens.org/datasets/movielens/ml-10m.zip"
            )

        raw_rows, raw_cols, raw_vals = [], [], []
        with src.open("r", newline="", encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("::")
                if len(parts) < 3:
                    continue
                raw_rows.append(parts[0])
                raw_cols.append(parts[1])
                raw_vals.append(float(parts[2]))

        rows, _ = _dense_remap(raw_rows)
        cols, _ = _dense_remap(raw_cols)
        return self.write_triples(out_dir, self.name, rows, cols, raw_vals)


# ---------------------------------------------------------------------------
# ml-20m
# ---------------------------------------------------------------------------


class MovieLens20M(DataPreprocessor):
    """MovieLens 20M — ratings.csv (comma-separated, with header)."""

    name = "ml-20m"
    description = "MovieLens 20M (ratings.csv, comma-separated with header)"

    def preprocess(self, raw_dir: Path, out_dir: Path, **kwargs: Any) -> Path:
        src = raw_dir / "ml-20m" / "ratings.csv"
        if not src.exists():
            raise FileNotFoundError(
                f"Expected {src}\n"
                "Download: https://files.grouplens.org/datasets/movielens/ml-20m.zip"
            )

        raw_rows, raw_cols, raw_vals = [], [], []
        with src.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for record in reader:
                raw_rows.append(record["userId"])
                raw_cols.append(record["movieId"])
                raw_vals.append(float(record["rating"]))

        rows, _ = _dense_remap(raw_rows)
        cols, _ = _dense_remap(raw_cols)
        return self.write_triples(out_dir, self.name, rows, cols, raw_vals)


class MovieLensLatestSmall(DataPreprocessor):
    """Pinned September 2018 MovieLens latest-small release."""

    name = "movielens-latest-small"
    description = "MovieLens latest-small (610 users, 9,724 movies, 100,836 ratings)"

    def __init__(
        self,
        expected: ExpectedDatasetCounts = _LATEST_SMALL_COUNTS,
    ) -> None:
        self.expected = expected

    def preprocess(self, raw_dir: Path, out_dir: Path, **kwargs: Any) -> Path:
        if kwargs:
            raise ValueError(f"{self.name} does not accept preprocessing parameters")
        source_id = "ml-latest-small/ratings.csv"
        src = raw_dir / source_id
        if not src.exists():
            raise FileNotFoundError(
                f"Expected {src}\n"
                "Download: https://files.grouplens.org/datasets/movielens/"
                "ml-latest-small.zip"
            )

        rows: list[str] = []
        columns: list[str] = []
        values: list[float] = []
        pairs: set[tuple[str, str]] = set()
        with src.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            required = {"userId", "movieId", "rating", "timestamp"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"{src} must contain columns {sorted(required)}")
            for line_number, record in enumerate(reader, start=2):
                row_id = record["userId"]
                column_id = record["movieId"]
                try:
                    if int(row_id) <= 0 or int(column_id) <= 0:
                        raise ValueError
                    rating = float(record["rating"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid MovieLens rating at {src}:{line_number}") from exc
                if not 0.5 <= rating <= 5.0:
                    raise ValueError(
                        f"MovieLens rating at {src}:{line_number} must be in [0.5, 5.0]"
                    )
                pair = (row_id, column_id)
                if pair in pairs:
                    raise ValueError(f"duplicate MovieLens user/movie pair at {src}:{line_number}")
                pairs.add(pair)
                rows.append(row_id)
                columns.append(column_id)
                values.append(rating)

        actual = ExpectedDatasetCounts(len(set(rows)), len(set(columns)), len(values))
        if actual != self.expected:
            raise ValueError(
                f"{self.name} expected {self.expected.n_rows} users, "
                f"{self.expected.n_columns} movies, and "
                f"{self.expected.n_observations} ratings; found {actual.n_rows}, "
                f"{actual.n_columns}, and {actual.n_observations}"
            )
        return self.write_preprocessed_triples(
            out_dir,
            self.name,
            rows,
            columns,
            values,
            parameters={
                "release": "September 2018",
                "input_format": "GroupLens ratings.csv",
                "rating_range": [0.5, 5.0],
                "preserve_source_ids": True,
            },
            source_paths=[(source_id, src)],
            counts={
                "source_rows": actual.n_rows,
                "source_columns": actual.n_columns,
                "source_observations": actual.n_observations,
                "output_observations": actual.n_observations,
                "missing_observations_removed": 0,
            },
            identifiers={
                "rows": summarize_identifiers(rows, source_field="userId"),
                "columns": summarize_identifiers(columns, source_field="movieId"),
            },
        )
