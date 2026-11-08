"""Rolling finance composition; no portfolio behavior is added to generic sweeps."""

from __future__ import annotations

import dataclasses
import datetime as dt
import time
from pathlib import Path
from typing import Any

import numpy as np

from bcd_lab.checkpoints import save_checkpoint_archive
from bcd_lab.data.finance import ReturnPanel, ReturnWindow, load_return_panel, rolling_windows
from bcd_lab.experiments.artifacts import collect_provenance
from bcd_lab.finance.artifacts import FINANCE_ARTIFACT_VERSION, write_arrays, write_json
from bcd_lab.finance.covariance import get_estimator
from bcd_lab.finance.evaluation import excess_returns, gmv_weights, performance
from bcd_lab.finance.model import fit_model, prepare_model


def _window_dates(panel: ReturnPanel, window: ReturnWindow) -> dict[str, Any]:
    dates = panel.dates.astype(str)
    return {
        "period": window.period,
        "train_start": dates[window.train_start],
        "train_end": dates[window.train_end - 1],
        "test_start": dates[window.train_end],
        "test_end": dates[window.test_end - 1],
        "test_dates": dates[window.train_end : window.test_end].tolist(),
        "indices": dataclasses.asdict(window),
    }


def _validate_artifact_names(methods: dict[str, Any]) -> None:
    owners: dict[str, str] = {}
    for name, method in methods.items():
        filenames = [f"{name}.json", f"{name}.npz"]
        if "model" in method:
            filenames.append(f"{name}_checkpoints.npz")
        for filename in filenames:
            # Keep configurations safe on case-insensitive filesystems as well.
            key = filename.casefold()
            if key in owners:
                raise ValueError(
                    f"Artifact filename collision: {filename!r} is used by methods "
                    f"{owners[key]!r} and {name!r}; rename one of the methods"
                )
            owners[key] = name


def _preflight(spec: dict[str, Any], panel: ReturnPanel, windows: list[ReturnWindow]) -> None:
    _validate_artifact_names(spec["methods"])
    expected_assets = spec["panel"].get("n_assets", len(panel.tickers))
    if len(panel.tickers) != expected_assets:
        raise ValueError(f"Expected {expected_assets} assets, found {len(panel.tickers)}")
    first = windows[0]
    train = panel.values[first.train_start : first.train_end]
    for method in spec["methods"].values():
        if "model" in method:
            prepare_model(train, method["model"])
        # Validate estimator parameters before creating any output or fitting.
        estimator = method["estimator"]
        get_estimator(estimator["name"])(train, np.zeros_like(train), estimator["params"])


def _run_method(
    panel: ReturnPanel,
    window: ReturnWindow,
    name: str,
    method: dict[str, Any],
    evaluation: dict[str, Any],
    output: Path,
) -> str:
    train = panel.values[window.train_start : window.train_end]
    test = panel.values[window.train_end : window.test_end]
    fit_seconds = 0.0
    fitted = None
    if "model" in method:
        started = time.perf_counter()
        fitted = fit_model(train, method["model"])
        fit_seconds = time.perf_counter() - started
    estimator = method["estimator"]
    started = time.perf_counter()
    estimate = get_estimator(estimator["name"])(
        train,
        None if fitted is None else fitted.predictions,
        estimator["params"],
    )
    covariance_seconds = time.perf_counter() - started
    covariance = estimate.covariance
    if covariance.shape != (train.shape[1], train.shape[1]) or not np.isfinite(covariance).all():
        raise ValueError(f"Estimator {estimator['name']} produced an invalid covariance")
    if not np.allclose(covariance, covariance.T) or np.linalg.eigvalsh(covariance)[0] <= 0:
        raise ValueError(
            f"Estimator {estimator['name']} must produce a positive-definite covariance"
        )
    weights = (
        np.ones(train.shape[1]) / train.shape[1]
        if method["portfolio"] == "equal_weight"
        else gmv_weights(covariance)
    )
    daily = excess_returns(
        test,
        weights,
        risk_free=evaluation["risk_free"],
        trading_days=evaluation["trading_days"],
    )
    stem = f"period_{window.period:03d}/{name}"
    arrays = {"covariance": covariance, "weights": weights, "daily_excess_returns": daily}
    record: dict[str, Any] = {
        "finance_artifact_version": FINANCE_ARTIFACT_VERSION,
        "method": name,
        "window": _window_dates(panel, window),
        "settings": method,
        "evaluation": evaluation,
        "n_assets": len(panel.tickers),
        "tickers": panel.tickers,
        "arrays": f"{name}.npz",
        "fit_seconds": fit_seconds,
        "covariance_seconds": covariance_seconds,
        "covariance_trace": float(np.trace(covariance)),
        "covariance_min_eigenvalue": float(np.linalg.eigvalsh(covariance)[0]),
        "weights_sum": float(weights.sum()),
        "gross_exposure": float(np.abs(weights).sum()),
        "metrics": performance(daily, trading_days=evaluation["trading_days"]),
        "diagnostics": estimate.diagnostics,
    }
    if fitted is not None:
        arrays["fitted_returns"] = fitted.predictions
        arrays["residual_returns"] = train - fitted.predictions
        record["resolved_model"] = fitted.settings
        record["termination"] = dataclasses.asdict(fitted.result.termination)
        record["model_diagnostics"] = fitted.result.diagnostics
        if not fitted.result.checkpoints:
            raise ValueError("Fitted algorithm must retain its final model checkpoint")
        save_checkpoint_archive(output / f"{stem}_checkpoints.npz", fitted.result.checkpoints)
        record["checkpoints"] = f"{name}_checkpoints.npz"
    write_arrays(output / f"{stem}.npz", arrays)
    write_json(output / f"{stem}.json", record)
    return f"{stem}.json"


def run_finance(spec: dict[str, Any], output: str | Path, *, overwrite: bool = False) -> Path:
    output = Path(output).expanduser().resolve()
    panel = load_return_panel(spec["panel"]["path"], sha256=spec["panel"].get("sha256"))
    windows = rolling_windows(panel, **spec["window"])
    _preflight(spec, panel, windows)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output}; use --overwrite explicitly"
        )
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "finance_artifact_version": FINANCE_ARTIFACT_VERSION,
        "status": "running",
        "config": spec,
        "provenance": collect_provenance(),
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "panel": {
            "path": str(panel.path),
            "fingerprint": dataclasses.asdict(panel.fingerprint),
            "tickers": panel.tickers,
            "shape": panel.values.shape,
        },
        "windows": [_window_dates(panel, window) for window in windows],
        "records": [],
    }
    write_json(output / "run.json", manifest)
    try:
        for window in windows:
            for name, method in spec["methods"].items():
                record = _run_method(panel, window, name, method, spec["evaluation"], output)
                manifest["records"].append(record)
                write_json(output / "run.json", manifest)
                print(
                    f"{spec['name']}: window {window.period + 1}/{len(windows)} {name}", flush=True
                )
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        write_json(output / "run.json", manifest)
        raise
    manifest["status"] = "complete"
    write_json(output / "run.json", manifest)
    return output
