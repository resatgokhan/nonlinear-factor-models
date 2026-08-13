"""Unit tests for dataset preprocessors and path resolution."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from bcd_lab.data import load_dataset
from bcd_lab.data.preprocessors import (
    DataPreprocessor,
    get_preprocessor,
    list_preprocessors,
    preprocess_dataset,
    register_preprocessor,
)
from bcd_lab.data.preprocessors.base import ExpectedDatasetCounts
from bcd_lab.data.preprocessors.jester import JesterDataset1
from bcd_lab.data.preprocessors.movielens import MovieLensLatestSmall


def test_preprocessor_registry() -> None:
    # Check default registered preprocessors exist
    preprocessors = list_preprocessors()
    assert "ml-100k" in preprocessors
    assert "ml-1m" in preprocessors
    assert "ml-10m" in preprocessors
    assert "ml-20m" in preprocessors
    assert "movielens-latest-small" in preprocessors
    assert "jester-dataset-1" in preprocessors

    # Retrieve default preprocessor
    ml100k = get_preprocessor("ml-100k")
    assert ml100k.name == "ml-100k"
    assert "u.data" in ml100k.description

    # Test unknown preprocessor lookup
    with pytest.raises(KeyError, match="Unknown preprocessor"):
        get_preprocessor("non_existent_dataset")


def test_custom_preprocessor_registration() -> None:
    class DummyPreprocessor(DataPreprocessor):
        name = "dummy-dataset"
        description = "Dummy description"

        def preprocess(self, raw_dir: Path, out_dir: Path, **kwargs: Any) -> Path:
            return self.write_triples(out_dir, self.name, [0, 1], [1, 0], [4.5, 5.0])

    dummy = DummyPreprocessor()
    register_preprocessor(dummy)

    assert "dummy-dataset" in list_preprocessors()
    assert get_preprocessor("dummy-dataset") is dummy

    # Verify error on empty name
    class EmptyNamePreprocessor(DataPreprocessor):
        name = ""
        description = "Empty"

        def preprocess(self, raw_dir: Path, out_dir: Path, **kwargs: Any) -> Path:
            return out_dir

    with pytest.raises(ValueError, match="Preprocessor name must be non-empty"):
        register_preprocessor(EmptyNamePreprocessor())


def test_preprocess_movielens_100k() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        raw_path = Path(tmp_raw_dir)
        out_path = Path(tmp_out_dir)

        # 1. Missing raw file raises FileNotFoundError
        ml100k = get_preprocessor("ml-100k")
        with pytest.raises(FileNotFoundError, match="u.data"):
            ml100k.preprocess(raw_path, out_path)

        # 2. Setup correct file structure
        raw_ml100k_dir = raw_path / "ml-100k"
        raw_ml100k_dir.mkdir(parents=True)
        u_data = raw_ml100k_dir / "u.data"

        # Tab-separated user_id \t item_id \t rating \t timestamp
        records = [
            "1\t101\t5\t874965758\n",
            "1\t102\t3\t874965758\n",
            "2\t101\t4\t874965758\n",
            "2\t103\t2\t874965758\n",
        ]
        with u_data.open("w") as f:
            f.writelines(records)

        # Run preprocess
        res_path = preprocess_dataset("ml-100k", raw_path, out_path)
        assert res_path == out_path / "ml-100k.csv"
        assert res_path.exists()

        # Read back written triples to verify header and content
        with res_path.open("r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert rows[0] == ["row", "col", "value"]
        # Remapping should assign:
        # row: '1' -> 0, '2' -> 1
        # col: '101' -> 0, '102' -> 1, '103' -> 2
        assert rows[1] == ["0", "0", "5.0"]
        assert rows[2] == ["0", "1", "3.0"]
        assert rows[3] == ["1", "0", "4.0"]
        assert rows[4] == ["1", "2", "2.0"]


def test_preprocess_movielens_1m() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        raw_path = Path(tmp_raw_dir)
        out_path = Path(tmp_out_dir)

        # 1. Missing raw file raises FileNotFoundError
        ml1m = get_preprocessor("ml-1m")
        with pytest.raises(FileNotFoundError, match="ratings.dat"):
            ml1m.preprocess(raw_path, out_path)

        # 2. Setup correct file structure
        raw_ml1m_dir = raw_path / "ml-1m"
        raw_ml1m_dir.mkdir(parents=True)
        ratings_dat = raw_ml1m_dir / "ratings.dat"

        # Double-colon separated UserID::MovieID::Rating::Timestamp
        records = [
            "1::1001::5::978300760\n",
            "1::1002::4::978300760\n",
            "3::1001::2::978300760\n",
        ]
        with ratings_dat.open("w") as f:
            f.writelines(records)

        # Run preprocess
        res_path = preprocess_dataset("ml-1m", raw_path, out_path)
        assert res_path == out_path / "ml-1m.csv"
        assert res_path.exists()

        # Read back and verify remapped triples
        with res_path.open("r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert rows[0] == ["row", "col", "value"]
        # row: '1' -> 0, '3' -> 1
        # col: '1001' -> 0, '1002' -> 1
        assert rows[1] == ["0", "0", "5.0"]
        assert rows[2] == ["0", "1", "4.0"]
        assert rows[3] == ["1", "0", "2.0"]


def test_preprocess_movielens_20m() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        raw_path = Path(tmp_raw_dir)
        out_path = Path(tmp_out_dir)

        # 1. Missing raw file raises FileNotFoundError
        ml20m = get_preprocessor("ml-20m")
        with pytest.raises(FileNotFoundError, match="ratings.csv"):
            ml20m.preprocess(raw_path, out_path)

        # 2. Setup correct file structure
        raw_ml20m_dir = raw_path / "ml-20m"
        raw_ml20m_dir.mkdir(parents=True)
        ratings_csv = raw_ml20m_dir / "ratings.csv"

        # Comma-separated with header userId,movieId,rating,timestamp
        records = [
            ["userId", "movieId", "rating", "timestamp"],
            ["10", "200", "4.5", "1112486027"],
            ["11", "201", "3.0", "1112486027"],
            ["10", "201", "5.0", "1112486027"],
        ]
        with ratings_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(records)

        # Run preprocess
        res_path = preprocess_dataset("ml-20m", raw_path, out_path)
        assert res_path == out_path / "ml-20m.csv"
        assert res_path.exists()

        # Read back and verify remapped triples
        with res_path.open("r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert rows[0] == ["row", "col", "value"]
        # row: '10' -> 0, '11' -> 1
        # col: '200' -> 0, '201' -> 1
        assert rows[1] == ["0", "0", "4.5"]
        assert rows[2] == ["1", "1", "3.0"]
        assert rows[3] == ["0", "1", "5.0"]


def test_movielens_latest_small_preserves_ids_and_records_provenance(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    source = raw_dir / "ml-latest-small" / "ratings.csv"
    source.parent.mkdir(parents=True)
    source.write_text(
        "userId,movieId,rating,timestamp\n1,10,4.5,1\n1,30,3.0,2\n2,10,5.0,3\n2,20,0.5,4\n",
        encoding="utf-8",
    )
    preprocessor = MovieLensLatestSmall(ExpectedDatasetCounts(2, 3, 4))

    output = preprocessor.preprocess(raw_dir, tmp_path / "data")

    with output.open(newline="") as stream:
        records = list(csv.reader(stream))
    assert records[1:] == [
        ["1", "10", "4.5"],
        ["1", "30", "3.0"],
        ["2", "10", "5.0"],
        ["2", "20", "0.5"],
    ]
    provenance = json.loads(output.with_suffix(".csv.provenance.json").read_text(encoding="utf-8"))
    assert provenance["counts"]["output_observations"] == 4
    assert provenance["identifiers"]["columns"]["count"] == 3
    assert provenance["sources"][0]["source_id"] == "ml-latest-small/ratings.csv"
    assert len(provenance["sources"][0]["fingerprint"]["sha256"]) == 64
    assert len(provenance["output_fingerprint"]["sha256"]) == 64
    dataset = load_dataset(
        "csv",
        params={"path": str(output.resolve()), "shape": {"rank": 1}},
        seed=3,
    )
    assert dataset.capabilities.preprocessing is not None
    assert dataset.capabilities.preprocessing.preprocessor == "movielens-latest-small"


def test_jester_matrix_removes_sentinel_and_checks_rating_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "raw" / "jester" / "jester-data-1.xls"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"compact spreadsheet fixture")
    valid = pd.DataFrame(
        [
            [2, -10.0, 99.0, 4.5],
            [2, 10.0, 0.0, 99.0],
        ]
    )
    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: valid)
    preprocessor = JesterDataset1(ExpectedDatasetCounts(2, 3, 4))

    output = preprocessor.preprocess(tmp_path / "raw", tmp_path / "data")

    with output.open(newline="") as stream:
        records = list(csv.reader(stream))
    assert records[1:] == [
        ["0", "1", "-10.0"],
        ["0", "3", "4.5"],
        ["1", "1", "10.0"],
        ["1", "2", "0.0"],
    ]
    provenance = json.loads(output.with_suffix(".csv.provenance.json").read_text(encoding="utf-8"))
    assert provenance["counts"]["missing_sentinel_removed"] == 2
    assert provenance["parameters"]["rating_range"] == [-10.0, 10.0]

    invalid = valid.copy()
    invalid.iloc[0, 1] = -10.01
    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: invalid)
    with pytest.raises(ValueError, match=r"ratings must be finite and in \[-10, 10\]"):
        preprocessor.preprocess(tmp_path / "raw", tmp_path / "invalid")


def test_preprocess_cli_list(capsys) -> None:
    from bcd_lab import cli

    assert cli.main(["preprocess", "--list"]) == 0
    output = capsys.readouterr().out
    assert "ml-100k" in output
    assert "ml-1m" in output
    assert "ml-10m" in output
    assert "ml-20m" in output


def test_preprocess_cli_errors(capsys, tmp_path: Path) -> None:
    from bcd_lab import cli

    assert cli.main(["preprocess", "non_existent_dataset"]) == 1
    assert "Unknown preprocessor" in capsys.readouterr().err

    assert (
        cli.main(
            [
                "preprocess",
                "ml-100k",
                "--raw-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path),
            ]
        )
        == 1
    )
    error = capsys.readouterr().err
    assert "Expected" in error
    assert "u.data" in error
