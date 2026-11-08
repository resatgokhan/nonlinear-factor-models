"""Finance numerical contracts and chronological pipeline integration."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bcd_lab.checkpoints import load_checkpoint_archive
from bcd_lab.data.finance import load_return_panel, return_dataset, rolling_windows
from bcd_lab.finance.artifacts import read_arrays, read_json
from bcd_lab.finance.config import load_config, resolve_config
from bcd_lab.finance.covariance import (
    adaptive_residual_covariance,
    fitted_residual_covariance,
    get_estimator,
    positive_definite,
    sample_covariance,
)
from bcd_lab.finance.evaluation import excess_returns, gmv_weights, performance
from bcd_lab.finance.model import prepare_model
from bcd_lab.finance.report import summarize, write_report
from bcd_lab.finance.runner import run_finance


def panel_file(tmp_path: Path) -> Path:
    rng = np.random.default_rng(19)
    frame = pd.DataFrame(rng.normal(0, 0.01, (24, 4)), columns=list("ABCD"))
    frame.insert(0, "Date", pd.date_range("2015-01-01", periods=24).strftime("%Y-%m-%d"))
    path = tmp_path / "returns.csv"
    frame.to_csv(path, index=False)
    return path


def small_spec(path: Path) -> dict:
    base = load_config(Path(__file__).parents[1] / "finance/configs/smoke.yaml")
    base["panel"] = {"path": str(path), "n_assets": 4}
    base["window"].update(lookback=12, horizon=3, rebalance=3, periods=2)
    return resolve_config(base, base_dir=path.parent)


def test_covariance_matches_manual_centered_product() -> None:
    x = np.array([[1, 4], [2, 1], [8, -2]], dtype=float)
    xc = x - x.mean(axis=0)
    np.testing.assert_allclose(sample_covariance(x), xc.T @ xc / 2)
    assert sample_covariance(x[:, :1]).shape == (1, 1)
    with pytest.raises(ValueError, match="finite"):
        sample_covariance(np.array([[np.nan], [1]]))


def test_adaptive_threshold_and_extra_psd_bump() -> None:
    x = np.random.default_rng(8).normal(size=(30, 4)) + 2
    c_star = 0.0003
    sample = np.cov(x, rowvar=False)
    theta = np.mean(np.abs(x[:, :, None] * x[:, None, :] - sample), axis=0)
    omega = c_star * np.sqrt(30 ** (-2 / 5) * np.log(30) ** 8)
    raw = sample * (np.abs(sample) >= theta * omega)
    eig, vectors = np.linalg.eigh(raw)
    expected = raw
    if eig[0] < 1e-8:
        expected = (vectors * np.maximum(eig, 1e-8)) @ vectors.T + 1e-8 * np.eye(4)
    actual = adaptive_residual_covariance(x, c_star=c_star)
    np.testing.assert_allclose(actual.covariance, expected, atol=1e-13)
    constant = adaptive_residual_covariance(np.ones((10, 3)))
    np.testing.assert_allclose(constant.covariance, np.eye(3) * 2e-8)
    assert constant.diagnostics["diagonal_entries_removed"] == 3
    assert constant.diagnostics["repair_applied"]


def test_covariance_composition_cross_terms_and_weights() -> None:
    x = np.random.default_rng(9).normal(size=(30, 3))
    fitted = x * 0.6
    result = fitted_residual_covariance(x, fitted, {})
    residual = adaptive_residual_covariance(x - fitted)
    np.testing.assert_allclose(
        result.covariance, np.cov(fitted, rowvar=False) + residual.covariance
    )
    assert result.diagnostics["fit_residual_cross_frobenius"] > 0
    w = gmv_weights(np.diag([1.0, 4.0]))
    np.testing.assert_allclose(w, [0.8, 0.2])
    assert w.sum() == pytest.approx(1)
    repaired, bump = positive_definite(np.diag([-1.0, 2.0]), 0.01)
    assert bump == pytest.approx(1.01)
    assert np.linalg.eigvalsh(repaired)[0] == pytest.approx(0.01)


def test_one_annualization_basis_and_zero_volatility() -> None:
    returns = np.array([[0.02, -0.01], [-0.01, 0.03], [0.03, 0.01]])
    values = excess_returns(returns, np.array([0.4, 0.6]), risk_free=0.04, trading_days=250)
    expected = returns @ [0.4, 0.6] - (1.04 ** (1 / 250) - 1)
    np.testing.assert_allclose(values, expected)
    metrics = performance(values)
    assert metrics["AV"] == pytest.approx(expected.mean() * 25000)
    assert metrics["SD"] == pytest.approx(expected.std(ddof=1) * np.sqrt(250) * 100)
    assert metrics["SR"] == pytest.approx(metrics["AV"] / metrics["SD"])
    assert performance(np.zeros(3))["SR"] is None


def test_panel_validation_fingerprint_and_window_boundaries(tmp_path: Path) -> None:
    path = panel_file(tmp_path)
    panel = load_return_panel(path)
    with pytest.raises(ValueError, match="fingerprint"):
        load_return_panel(path, sha256="0" * 64)
    spec = small_spec(path)
    windows = rolling_windows(panel, **spec["window"])
    assert [(w.train_start, w.train_end, w.test_end) for w in windows] == [(0, 12, 15), (3, 15, 18)]
    with pytest.raises(ValueError, match="rebalance"):
        rolling_windows(panel, **{**spec["window"], "rebalance": 1})
    with pytest.raises(ValueError, match="Insufficient"):
        rolling_windows(panel, **{**spec["window"], "periods": 100})
    frame = pd.read_csv(path)
    frame.loc[1, "Date"] = frame.loc[0, "Date"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_return_panel(path)
    assert return_dataset(np.zeros((3, 2)), rank=1).n_observations == 6


def test_two_window_artifacts_and_recomputed_report(tmp_path: Path) -> None:
    spec = small_spec(panel_file(tmp_path))
    output = run_finance(spec, tmp_path / "run")
    manifest = read_json(output / "run.json")
    assert manifest["status"] == "complete"
    assert len(manifest["records"]) == 10
    panel = load_return_panel(spec["panel"]["path"])
    for relative in manifest["records"]:
        path = output / relative
        record = read_json(path)
        arrays = read_arrays(path.parent / record["arrays"])
        assert np.linalg.eigvalsh(arrays["covariance"])[0] > 0
        assert arrays["weights"].sum() == pytest.approx(1)
        assert record["window"]["train_end"] < record["window"]["test_start"]
        if "checkpoints" in record:
            checkpoints = load_checkpoint_archive(path.parent / record["checkpoints"])
            assert 5 in checkpoints
            assert record["resolved_model"]["seeds"]["initialization_z"] == 0
            indices = record["window"]["indices"]
            train = panel.values[indices["train_start"] : indices["train_end"]]
            _, dataset, algorithm, config, _ = prepare_model(train, record["resolved_model"])
            restored = algorithm.restore_checkpoint(checkpoints[5], config=config, dataset=dataset)
            predictions = restored.z[: len(train)] @ restored.z[len(train) :].T
            if restored.phi is not None:
                predictions = restored.phi.eval_array(predictions)
            np.testing.assert_allclose(predictions, arrays["fitted_returns"])
    rows = write_report(output, plot=False)
    row = next(row for row in rows if row["estimator"] == "bcd_nonlinear")
    daily = np.concatenate(
        [
            read_arrays(output / f"period_{period:03d}/bcd_nonlinear.npz")["daily_excess_returns"]
            for period in (0, 1)
        ]
    )
    assert row["AV"] == pytest.approx(performance(daily)["AV"])
    assert row["n_obs"] == 6
    assert summarize(output) == rows
    with pytest.raises(FileExistsError):
        run_finance(spec, output)
    manifest["records"].pop()
    (output / "run.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Missing finance records"):
        summarize(output)


def test_future_returns_cannot_change_training_outputs(tmp_path: Path) -> None:
    path = panel_file(tmp_path)
    spec = small_spec(path)
    spec["window"]["periods"] = 1
    spec["methods"] = {"bcd_nonlinear": spec["methods"]["bcd_nonlinear"]}
    first = run_finance(spec, tmp_path / "first")
    frame = pd.read_csv(path)
    frame.iloc[12:, 1:] *= 100
    frame.to_csv(path, index=False)
    second = run_finance(spec, tmp_path / "second")
    a = read_arrays(first / "period_000/bcd_nonlinear.npz")
    b = read_arrays(second / "period_000/bcd_nonlinear.npz")
    for key in ("covariance", "weights", "fitted_returns"):
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.allclose(a["daily_excess_returns"], b["daily_excess_returns"])


def test_preflight_failure_creates_no_output(tmp_path: Path) -> None:
    spec = small_spec(panel_file(tmp_path))
    bad = copy.deepcopy(spec)
    bad["methods"]["bcd_nonlinear"]["model"]["rank"] = 500
    output = tmp_path / "no_output"
    with pytest.raises(ValueError, match="Rank"):
        run_finance(bad, output)
    assert not output.exists()
    with pytest.raises(KeyError):
        get_estimator("unknown")


@pytest.mark.parametrize(
    "order",
    [
        ("foo", "foo_checkpoints"),
        ("foo_checkpoints", "foo"),
        ("foo", "Foo_checkpoints"),
        ("Foo_checkpoints", "foo"),
    ],
)
@pytest.mark.parametrize("existing_output", [False, True])
def test_checkpoint_filename_collision_fails_before_writing(
    tmp_path: Path, order: tuple[str, str], existing_output: bool
) -> None:
    spec = small_spec(panel_file(tmp_path))
    spec["methods"] = {
        name: spec["methods"]["bcd_nonlinear" if name == "foo" else "sample"] for name in order
    }
    output = tmp_path / "run"
    protected = output / "period_000/foo_checkpoints.npz"
    if existing_output:
        protected.parent.mkdir(parents=True)
        protected.write_bytes(b"existing artifact")
    with pytest.raises(ValueError, match=r"(?i)Artifact filename collision.*foo_checkpoints\.npz"):
        run_finance(spec, output, overwrite=True)
    if existing_output:
        assert protected.read_bytes() == b"existing artifact"
        assert sorted(
            p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()
        ) == ["period_000/foo_checkpoints.npz"]
    else:
        assert not output.exists()


@pytest.mark.parametrize("order", [("foo", "FOO"), ("FOO", "foo")])
def test_case_only_method_names_fail_before_writing(tmp_path: Path, order: tuple[str, str]) -> None:
    spec = small_spec(panel_file(tmp_path))
    spec["methods"] = {name: spec["methods"]["sample"] for name in order}
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="Artifact filename collision"):
        run_finance(spec, output)
    assert not output.exists()


@pytest.mark.parametrize("order", [("foo", "foo_checkpoints"), ("foo_checkpoints", "foo")])
def test_checkpoint_suffix_is_valid_without_an_artifact_collision(
    tmp_path: Path, order: tuple[str, str]
) -> None:
    spec = small_spec(panel_file(tmp_path))
    spec["window"]["periods"] = 1
    methods = {
        "foo": spec["methods"]["sample"],
        "foo_checkpoints": spec["methods"]["bcd_nonlinear"],
    }
    spec["methods"] = {name: methods[name] for name in order}
    output = run_finance(spec, tmp_path / "run")
    assert read_json(output / "run.json")["status"] == "complete"
    assert len(summarize(output)) == 2
    for name in order:
        arrays = read_arrays(output / f"period_000/{name}.npz")
        assert arrays["covariance"].shape == (4, 4)
    checkpoints = load_checkpoint_archive(output / "period_000/foo_checkpoints_checkpoints.npz")
    assert 5 in checkpoints


def test_finance_configs_use_standard_model_and_preserve_identity_parameters() -> None:
    root = Path(__file__).parents[1]
    for n, rank, rate in ((50, 15, 3e-5), (100, 20, 2e-5), (200, 20, 1e-5)):
        spec = load_config(root / f"finance/configs/log_panel_N{n}.yaml")
        model = spec["methods"]["bcd_nonlinear"]["model"]
        assert model["link"]["kind"] == "piecewise_linear_rkhs"
        assert model["rank"] == rank
        assert model["algorithm"]["params"]["phi_update"]["learning_rate"] == rate
        assert spec["methods"]["bcd_identity"]["model"]["link"]["params"] == {}
