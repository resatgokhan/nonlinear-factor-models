"""Complete dated stock-return panels; values are never imputed or rescaled."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bcd_lab.data.provenance import fingerprint_file
from bcd_lab.types import DatasetBundle, DatasetCapabilities, SourceFingerprint


@dataclass(frozen=True)
class ReturnPanel:
    values: np.ndarray
    dates: np.ndarray
    tickers: tuple[str, ...]
    path: Path
    fingerprint: SourceFingerprint


@dataclass(frozen=True)
class ReturnWindow:
    period: int
    train_start: int
    train_end: int
    test_end: int


def load_return_panel(path: str | Path, *, sha256: str | None = None) -> ReturnPanel:
    path = Path(path).expanduser().resolve()
    fingerprint = fingerprint_file(path)
    if sha256 is not None and sha256 != fingerprint.sha256:
        raise ValueError(f"Return panel fingerprint mismatch: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        header = next(csv.reader(stream), [])
    if not header or header[0] != "Date" or len(header) < 2:
        raise ValueError("Return panel requires Date followed by stock columns")
    if len(set(header)) != len(header) or any(not name.strip() for name in header):
        raise ValueError("Return panel column names must be unique and nonempty")
    frame = pd.read_csv(path)
    dates = pd.to_datetime(frame.pop("Date"), errors="raise")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Return panel dates must be nonmissing and unique")
    order = np.argsort(dates.to_numpy())
    values = frame.to_numpy(dtype=np.float64)[order]
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Return panel must have at least two complete finite observations")
    date_array = dates.to_numpy()[order].astype("datetime64[D]")
    if np.unique(date_array).size != date_array.size:
        raise ValueError("Return panel must contain only one observation per date")
    values.setflags(write=False)
    date_array.setflags(write=False)
    return ReturnPanel(values, date_array, tuple(frame.columns), path, fingerprint)


def rolling_windows(
    panel: ReturnPanel,
    *,
    start_date: str,
    end_date: str,
    lookback: int,
    rebalance: int,
    horizon: int,
    periods: int,
) -> list[ReturnWindow]:
    if lookback < 2 or horizon < 2 or rebalance < horizon or periods < 1:
        raise ValueError("Require lookback/horizon >= 2, rebalance >= horizon, periods >= 1")
    start, end = np.datetime64(start_date, "D"), np.datetime64(end_date, "D")
    if np.isnat(start) or np.isnat(end) or end <= start:
        raise ValueError("Require finite dates with end_date > start_date")
    base = int(np.searchsorted(panel.dates, start))
    limit = int(np.searchsorted(panel.dates, end))
    windows = []
    for period in range(periods):
        train_end = base + lookback + period * rebalance
        test_end = train_end + horizon
        if test_end > limit:
            raise ValueError(f"Insufficient panel history for requested window {period}")
        windows.append(ReturnWindow(period, train_end - lookback, train_end, test_end))
    return windows


def return_dataset(returns: np.ndarray, *, rank: int) -> DatasetBundle:
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("Returns must be a nonempty finite (time, stocks) matrix")
    if not 1 <= rank <= min(values.shape):
        raise ValueError("Rank must be between 1 and min(time, stocks)")
    rows, cols = np.indices(values.shape)
    dataset = DatasetBundle(
        source="finance_returns",
        n_rows=values.shape[0],
        n_cols=values.shape[1],
        rank=rank,
        train=(values.ravel().copy(), (rows.ravel(), cols.ravel())),
        capabilities=DatasetCapabilities(),
    )
    dataset.validate()
    return dataset


def empirical_mu(returns: np.ndarray, rank: int) -> float:
    """Empirical incoherence bound derived only from the current training window."""
    u, _, vt = np.linalg.svd(returns, full_matrices=False)
    left = returns.shape[0] / rank * np.max(np.sum(u[:, :rank] ** 2, axis=1))
    right = returns.shape[1] / rank * np.max(np.sum(vt[:rank].T ** 2, axis=1))
    return float(max(left, right))
