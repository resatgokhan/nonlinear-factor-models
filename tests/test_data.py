"""Unit tests for dataset loaders."""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from bcd_lab.data import (
    GeneratingLinkContext,
    list_data_loaders,
    list_generating_links,
    load_dataset,
    register_data_loader,
    register_generating_link,
)
from bcd_lab.data.provenance import (
    summarize_identifiers,
    write_preprocessing_provenance,
)
from bcd_lab.experiments.loader import load_sweep_config
from bcd_lab.experiments.runner import run_sweep
from bcd_lab.randomness import RunRandomness
from bcd_lab.types import DatasetBundle


def test_registry_listing() -> None:
    loaders = list_data_loaders()
    assert "synthetic" in loaders
    assert "csv" in loaders


def test_loader_boundary_rejects_malformed_external_bundle() -> None:
    class InvalidLoader:
        name = "invalid_boundary_loader"

        def load(self, *, params, seed):
            del params, seed
            return DatasetBundle(
                source=self.name,
                n_rows=2,
                n_cols=2,
                rank=1,
                train=(
                    np.array([1.0, 2.0]),
                    (np.array([0]), np.array([0, 1])),
                ),
            )

    register_data_loader(InvalidLoader())
    with pytest.raises(
        ValueError,
        match="invalid_boundary_loader.*values and indices must have equal length",
    ):
        load_dataset("invalid_boundary_loader", params={}, seed=1)


def test_synthetic_standard_normal() -> None:
    params = {
        "shape": {
            "n_rows": 20,
            "n_cols": 25,
            "rank": 2,
            "n_observations": 100,
        },
        "type": "standard_normal",
    }
    dataset = load_dataset("synthetic", params=params, seed=42)

    assert dataset.source == "synthetic"
    assert dataset.n_rows == 20
    assert dataset.n_cols == 25
    assert dataset.rank == 2

    # Verify train sample set shape
    values, (rows, cols) = dataset.train
    assert values.shape == (100,)
    assert rows.shape == (100,)
    assert cols.shape == (100,)
    assert np.all(rows >= 0) and np.all(rows < 20)
    assert np.all(cols >= 0) and np.all(cols < 25)
    assert dataset.capabilities.splits is not None
    assert dataset.capabilities.splits.train_count == 100
    assert "m_effective" not in dataset.metadata

    # Verify latent and dense truth matrices
    assert dataset.dense_truth is not None
    assert dataset.dense_truth.shape == (20, 25)
    assert dataset.latent_truth is not None
    assert dataset.latent_truth.shape == (45, 2)  # n1 + n2 = 45

    # Verify SVD reconstruction
    u = dataset.latent_truth[:20]
    v = dataset.latent_truth[20:]
    x_rec = u @ v.T
    assert np.allclose(dataset.dense_truth, x_rec)

    # Check metadata
    assert "mu" in dataset.metadata
    assert "sigma_1_star" in dataset.metadata
    assert "sigma_r_star" in dataset.metadata
    assert dataset.metadata["data_type"] == "standard_normal"

    repeated = load_dataset("synthetic", params=params, seed=42)
    changed = load_dataset("synthetic", params=params, seed=43)
    assert np.array_equal(dataset.train[0], repeated.train[0])
    assert np.array_equal(dataset.train[1][0], repeated.train[1][0])
    assert not np.array_equal(dataset.dense_truth, changed.dense_truth)


def test_synthetic_publishes_central_quantile_evaluation_domain() -> None:
    params = {
        "shape": {
            "n_rows": 20,
            "n_cols": 18,
            "rank": 2,
            "n_observations": 100,
        },
        "evaluation_domain": {"central_mass": 0.95, "grid_count": 17},
    }
    dataset = load_dataset("synthetic", params=params, seed=12)

    domain = dataset.capabilities.function_domain
    assert domain is not None
    assert domain.source == "latent_truth_central_quantile"
    assert domain.central_mass == 0.95
    assert domain.grid.shape == (17,)
    assert dataset.dense_truth is not None
    expected = np.quantile(dataset.dense_truth.ravel(), [0.025, 0.975])
    assert np.allclose(domain.grid[[0, -1]], expected)
    assert not domain.grid.flags.writeable


@pytest.mark.parametrize(
    ("evaluation_domain", "message"),
    [
        ([], "must be a mapping"),
        ({"central_mass": 0.0}, "central_mass"),
        ({"grid_count": 1}, "grid_count"),
        ({"quantile": 0.95}, "unknown fields"),
    ],
)
def test_invalid_synthetic_evaluation_domain_is_rejected(
    evaluation_domain: object,
    message: str,
) -> None:
    params = {
        "shape": {
            "n_rows": 5,
            "n_cols": 5,
            "rank": 1,
            "n_observations": 10,
        },
        "evaluation_domain": evaluation_domain,
    }
    with pytest.raises((TypeError, ValueError), match=message):
        load_dataset("synthetic", params=params, seed=1)


def test_synthetic_discrete_and_noise() -> None:
    params = {
        "shape": {
            "n_rows": 15,
            "n_cols": 15,
            "rank": 3,
            "n_observations": 50,
        },
        "type": "discrete",
        "noise_std": 0.1,
        "noise_const": 1.0,
    }
    dataset = load_dataset("synthetic", params=params, seed=123)

    assert dataset.n_rows == 15
    assert dataset.n_cols == 15
    assert dataset.rank == 3
    assert dataset.train[0].shape == (50,)


@pytest.mark.parametrize(
    ("name", "params", "expected"),
    [
        ("identity", {}, lambda x: x),
        ("sigmoid", {"shifted": True}, lambda x: 1.0 / (1.0 + np.exp(-x)) - 0.5),
        ("sigmoid", {"shifted": False}, lambda x: 1.0 / (1.0 + np.exp(-x))),
        ("exponential", {"shifted": True}, lambda x: np.exp(x) - 1.0),
        ("exponential", {"shifted": False}, np.exp),
        ("tanh", {}, np.tanh),
    ],
)
def test_configured_generating_links_publish_truth(
    name: str,
    params: dict,
    expected,
) -> None:
    config = {
        "shape": {
            "n_rows": 12,
            "n_cols": 10,
            "rank": 2,
            "n_observations": 50,
        },
        "type": "standard_normal",
        "generating_link": {"name": name, "params": params},
    }
    dataset = load_dataset("synthetic", params=config, seed=31)
    truth = dataset.capabilities.functions.truth
    linked_truth = dataset.capabilities.linked_truth
    assert truth is not None
    assert linked_truth is not None
    assert dataset.dense_truth is not None
    assert np.allclose(linked_truth, expected(dataset.dense_truth))
    rows, cols = dataset.train[1]
    assert np.allclose(dataset.train[0], linked_truth[rows, cols])


def test_generating_link_is_independent_from_latent_data_and_reproducible() -> None:
    base = {
        "shape": {
            "n_rows": 20,
            "n_cols": 18,
            "rank": 2,
            "n_observations": 100,
        },
        "type": "standard_normal",
    }
    identity = load_dataset(
        "synthetic",
        params={**base, "generating_link": {"name": "identity", "params": {}}},
        seed=4,
    )
    sigmoid_params = {
        **base,
        "generating_link": {"name": "sigmoid", "params": {"shifted": True}},
    }
    sigmoid = load_dataset("synthetic", params=sigmoid_params, seed=4)
    repeated = load_dataset("synthetic", params=sigmoid_params, seed=4)

    assert np.array_equal(identity.dense_truth, sigmoid.dense_truth)
    assert np.array_equal(identity.train[1][0], sigmoid.train[1][0])
    assert np.array_equal(identity.train[1][1], sigmoid.train[1][1])
    assert not np.array_equal(identity.train[0], sigmoid.train[0])
    assert np.array_equal(sigmoid.train[0], repeated.train[0])


def test_random_monotone_rkhs_truth_is_deterministic_on_its_fit_domain() -> None:
    config = {
        "shape": {
            "n_rows": 20,
            "n_cols": 18,
            "rank": 2,
            "n_observations": 100,
        },
        "type": "standard_normal",
        "generating_link": {
            "name": "random_monotone_rkhs",
            "params": {
                "degree": 4,
                "sigma": 1.0,
                "fit_ridge": 1.0e-6,
                "num_centers": 100,
            },
        },
    }
    dataset = load_dataset("synthetic", params=config, seed=4)
    repeated = load_dataset("synthetic", params=config, seed=4)
    truth = dataset.capabilities.functions.truth
    repeated_truth = repeated.capabilities.functions.truth
    assert truth is not None and repeated_truth is not None
    latent_samples = dataset.dense_truth[dataset.train[1]]
    grid = np.linspace(float(latent_samples.min()), float(latent_samples.max()), 300)
    values = truth.eval_array(grid)
    assert np.all(np.diff(values) > 0.0)
    assert np.array_equal(values, repeated_truth.eval_array(grid))
    assert np.array_equal(dataset.train[0], repeated.train[0])


@pytest.mark.parametrize(
    ("generating_link", "message"),
    [
        ({"name": "missing", "params": {}}, "Unknown generating link"),
        ({"name": "sigmoid", "params": {"shift": True}}, "unknown parameters"),
        ({"name": "random_monotone_rkhs", "params": {"degree": 0}}, "degree"),
    ],
)
def test_invalid_generating_link_fails_contextually(
    generating_link: dict,
    message: str,
) -> None:
    config = {
        "shape": {
            "n_rows": 5,
            "n_cols": 5,
            "rank": 1,
            "n_observations": 10,
        },
        "generating_link": generating_link,
    }
    with pytest.raises((KeyError, ValueError), match=message):
        load_dataset("synthetic", params=config, seed=1)


def test_generating_link_registry_listing() -> None:
    assert list_generating_links() == (
        "exponential",
        "identity",
        "random_monotone_rkhs",
        "sigmoid",
        "tanh",
    )


def test_external_generating_link_runs_without_loader_or_runner_changes() -> None:
    class SquareTruth:
        kind = "test_square_truth"

        def eval_array(self, x: np.ndarray) -> np.ndarray:
            return np.asarray(x) ** 2

    class SquareTruthBuilder:
        def validate_params(self, params) -> None:
            if params:
                raise ValueError("test_square_truth accepts no parameters")

        def build(self, params, context: GeneratingLinkContext) -> SquareTruth:
            del params, context
            return SquareTruth()

    register_generating_link("test_square_truth", SquareTruthBuilder())
    config = {
        "shape": {
            "n_rows": 8,
            "n_cols": 7,
            "rank": 2,
            "n_observations": 20,
        },
        "generating_link": {"name": "test_square_truth", "params": {}},
    }
    dataset = load_dataset("synthetic", params=config, seed=9)

    assert dataset.dense_truth is not None
    assert np.allclose(dataset.capabilities.linked_truth, dataset.dense_truth**2)


def test_generating_link_runs_from_yaml_and_publishes_truth_diagnostics(
    tmp_path: Path,
) -> None:
    public_config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "examples"
        / "synthetic_link_recovery.yaml"
    )
    raw = yaml.safe_load(public_config.read_text(encoding="utf-8"))
    raw["experiment"] = "generating_link_e2e"
    raw["data"]["shape"] = {
        "n_rows": 12,
        "n_cols": 10,
        "rank": 2,
        "n_observations": 60,
    }
    raw["data"]["noise_std"] = 0.0
    raw["data"]["generating_link"] = {
        "name": "sigmoid",
        "params": {"shifted": True},
    }
    raw["data"]["evaluation_domain"] = {
        "central_mass": 0.95,
        "grid_count": 31,
    }
    raw["link"]["params"]["grid"]["count"] = 20
    raw["runtime"]["max_iter"] = 0
    raw["diagnostics"] = {
        "on_error": "raise",
        "families": {
            "phi": ["phi_diff"],
            "phi_sharp": ["phi_sharp_diff", "phi_sharp_diff_star", "lyapunov"],
            "truth_matrix": ["phi_x_error"],
        },
    }
    config_path = tmp_path / "generating-link.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    run_sweep(config_path, output_root=tmp_path / "results")

    artifact_path = next((tmp_path / "results" / "generating_link_e2e").glob("*.json"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["experiment_spec"]["data"]["generating_link"] == {
        "name": "sigmoid",
        "params": {"shifted": True},
    }
    assert artifact["experiment_spec"]["data"]["evaluation_domain"] == {
        "central_mass": 0.95,
        "grid_count": 31,
    }
    for metric in (
        "phi_diff",
        "phi_sharp_diff",
        "phi_sharp_diff_star",
        "lyapunov",
        "phi_x_error",
    ):
        assert artifact["diagnostics"][0][metric] is not None


def test_unknown_generating_link_fails_preflight_before_output(tmp_path: Path) -> None:
    public_config = Path(__file__).resolve().parents[1] / "configs" / "examples" / "quickstart.yaml"
    raw = yaml.safe_load(public_config.read_text(encoding="utf-8"))
    raw["experiment"] = "unknown_generating_link"
    raw["data"]["generating_link"] = {"name": "missing_truth", "params": {}}
    config_path = tmp_path / "unknown-generating-link.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    output_root = tmp_path / "results"

    with pytest.raises(KeyError, match="missing_truth"):
        run_sweep(config_path, output_root=output_root)

    assert not output_root.exists()


def test_synthetic_desirable() -> None:
    params = {
        "shape": {
            "n_rows": 30,
            "n_cols": 30,
            "rank": 4,
            "n_observations": 150,
        },
        "desirable": True,
        "condition_number": 3.0,
        "min_per_row": 2,
        "min_per_col": 2,
    }
    dataset = load_dataset("synthetic", params=params, seed=7)

    assert dataset.n_rows == 30
    assert dataset.n_cols == 30
    assert dataset.rank == 4

    # Verify balanced samples: each row/col should have at least min_per_row/col entries
    _, (rows, cols) = dataset.train
    row_counts = np.bincount(rows, minlength=30)
    col_counts = np.bincount(cols, minlength=30)
    assert np.all(row_counts >= 2)
    assert np.all(col_counts >= 2)

    # Singular values condition check
    u = dataset.latent_truth[:30]
    v = dataset.latent_truth[30:]
    x_rec = u @ v.T
    s = np.linalg.svd(x_rec, compute_uv=False)
    # Ratio of first to fourth singular value should match condition number (3.0)
    assert np.isclose(s[0] / s[3], 3.0)


@pytest.fixture
def toy_csv_file() -> Path:
    # 5 users, 5 items, ratings matrix with 10 records
    records = [
        ("user", "item", "rating"),
        ("u1", "i1", "5.0"),
        ("u1", "i2", "3.0"),
        ("u2", "i1", "4.0"),
        ("u2", "i3", "2.0"),
        ("u3", "i2", "4.0"),
        ("u3", "i4", "1.0"),
        ("u4", "i3", "5.0"),
        ("u4", "i5", "3.0"),
        ("u5", "i4", "2.0"),
        ("u5", "i5", "4.0"),
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as f:
        for r in records:
            f.write(",".join(r) + "\r\n")
        path = Path(f.name)
    yield path
    path.unlink()


def test_csv_loader_basic(toy_csv_file: Path) -> None:
    params = {
        "path": str(toy_csv_file),
        "row_col": "user",
        "col_col": "item",
        "value_col": "rating",
        "has_header": True,
        "remap_ids": True,
        "shape": {"rank": 2},
    }
    dataset = load_dataset("csv", params=params, seed=42)

    assert dataset.source == "csv"
    assert dataset.n_rows == 5
    assert dataset.n_cols == 5
    assert dataset.rank == 2

    # 10 ratings total, no validation split
    assert dataset.train[0].shape == (10,)
    assert dataset.validation is None

    mappings = dataset.capabilities.id_mappings
    assert mappings is not None
    assert len(mappings.row_to_index) == 5
    assert len(mappings.column_to_index) == 5
    assert "row_id_map" not in dataset.metadata

    splits = dataset.capabilities.splits
    assert splits is not None
    assert splits.train_count == 10
    assert splits.validation_count == 0
    assert splits.resolved_total_count == 10

    fingerprint = dataset.capabilities.source_fingerprint
    assert fingerprint is not None
    assert len(fingerprint.sha256) == 64
    assert fingerprint.size_bytes == toy_csv_file.stat().st_size


def test_csv_fingerprint_detects_changed_replay_source(toy_csv_file: Path) -> None:
    params = {
        "path": str(toy_csv_file),
        "row_col": "user",
        "col_col": "item",
        "value_col": "rating",
        "has_header": True,
        "remap_ids": True,
        "shape": {"rank": 1},
    }
    original = load_dataset("csv", params=params, seed=42)
    fingerprint = original.capabilities.source_fingerprint
    assert fingerprint is not None
    replay_params = {**params, "source_fingerprint": dataclasses.asdict(fingerprint)}
    load_dataset("csv", params=replay_params, seed=42)

    toy_csv_file.write_text(
        toy_csv_file.read_text(encoding="utf-8") + "6,6,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source changed"):
        load_dataset("csv", params=replay_params, seed=42)


def test_experiment_relative_paths_do_not_depend_on_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    data_path = experiment_dir / "ratings.csv"
    data_path.write_text("row,col,value\n0,0,1.0\n1,1,2.0\n", encoding="utf-8")
    write_preprocessing_provenance(
        data_path,
        preprocessor="test-preprocessor",
        preprocessor_version=1,
        parameters={"policy": "test"},
        source_paths=[("fixture/ratings.csv", data_path)],
        counts={"output_observations": 2},
        identifiers={
            "rows": summarize_identifiers([0, 1], source_field="row"),
            "columns": summarize_identifiers([0, 1], source_field="column"),
        },
    )
    config_path = experiment_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "experiment": "paths",
                "algorithm": {"name": "bcd", "params": {}},
                "objective": {"name": "squared_error", "params": {}},
                "seed": 7,
                "data": {
                    "source": "csv",
                    "path": "ratings.csv",
                    "shape": {"rank": 1},
                },
                "link": {"kind": "identity"},
                "runtime": {"max_iter": 0},
                "plot": {"dir": "plots"},
            }
        ),
        encoding="utf-8",
    )

    loaded = []
    for working_dir in (tmp_path, tmp_path / "elsewhere"):
        working_dir.mkdir(exist_ok=True)
        monkeypatch.chdir(working_dir)
        spec = load_sweep_config(config_path.resolve())
        dataset = load_dataset("csv", spec.data, seed=11)
        loaded.append((spec, dataset.train[0].copy()))

    assert loaded[0][0].data["path"] == str(data_path.resolve())
    assert loaded[0][0].plot["dir"] == str((experiment_dir / "plots").resolve())
    assert loaded[0][0].data == loaded[1][0].data
    assert np.array_equal(loaded[0][1], loaded[1][1])

    run_sweep(config_path.resolve(), output_root="relative-results")
    result_dir = experiment_dir / "relative-results" / "paths"
    assert result_dir.is_dir()
    [result_path] = result_dir.glob("*.json")
    artifact = json.loads(result_path.read_text(encoding="utf-8"))
    recorded = artifact["experiment_spec"]["data"]["source_fingerprint"]
    assert recorded == artifact["provenance"]["data_source"]
    assert artifact["provenance"]["preprocessing"]["preprocessor"] == "test-preprocessor"
    replay_spec = load_sweep_config(config_path.resolve())
    replay_spec.data["source_fingerprint"] = recorded
    replayed = load_dataset("csv", replay_spec.data, seed=11)
    assert dataclasses.asdict(replayed.capabilities.source_fingerprint) == recorded
    assert not (working_dir / "relative-results").exists()


def test_csv_loader_size_limits(toy_csv_file: Path) -> None:
    params = {
        "path": str(toy_csv_file),
        "row_col": "user",
        "col_col": "item",
        "value_col": "rating",
        "has_header": True,
        "remap_ids": True,
        "shape": {"rank": 1},
        "size": {
            "max_rows": 3,
            "max_cols": 3,
            "strategy": "top",
        },
    }
    dataset = load_dataset("csv", params=params, seed=42)

    assert dataset.n_rows == 3
    assert dataset.n_cols == 3


def test_csv_loader_split(toy_csv_file: Path) -> None:
    params = {
        "path": str(toy_csv_file),
        "row_col": "user",
        "col_col": "item",
        "value_col": "rating",
        "has_header": True,
        "remap_ids": True,
        "shape": {"rank": 2},
        "split": {
            "validation_fraction": 0.5,
            "stratify_by": "row",
        },
    }
    dataset = load_dataset("csv", params=params, seed=42)

    # With validation fraction = 0.5 and stratification, some records go to validation
    assert dataset.train[0].shape[0] < 10
    assert dataset.validation is not None
    assert dataset.validation[0].shape[0] > 0
    assert dataset.train[0].shape[0] + dataset.validation[0].shape[0] == 10
    splits = dataset.capabilities.splits
    assert splits is not None
    assert splits.train_count == dataset.train[0].size
    assert splits.validation_count == dataset.validation[0].size
    assert splits.validation_fraction == 0.5
    assert splits.seed == RunRandomness(42).child_seed("validation_split")
    repeated = load_dataset("csv", params=params, seed=42)
    assert repeated.validation is not None
    assert np.array_equal(dataset.train[1][0], repeated.train[1][0])
    assert np.array_equal(dataset.validation[1][0], repeated.validation[1][0])
