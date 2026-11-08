"""GMV and weighted-log-return scores, using one annualization basis."""

from __future__ import annotations

import numpy as np


def gmv_weights(covariance: np.ndarray) -> np.ndarray:
    ones = np.ones(covariance.shape[0])
    direction = np.linalg.solve(covariance, ones)
    denominator = float(direction.sum())
    if not np.isfinite(direction).all() or denominator <= 0:
        raise ValueError("GMV requires a positive-definite covariance")
    return direction / denominator


def excess_returns(
    returns: np.ndarray, weights: np.ndarray, *, risk_free: float, trading_days: int
) -> np.ndarray:
    daily_rf = (1 + risk_free) ** (1 / trading_days) - 1
    return np.asarray(returns @ weights - daily_rf, dtype=float)


def performance(values: np.ndarray, *, trading_days: int = 250) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Performance requires at least two finite daily scores")
    mean = float(values.mean())
    variance = float(values.var(ddof=1))
    av = mean * trading_days * 100
    sd = float(np.sqrt(variance * trading_days) * 100)
    return {"n_obs": values.size, "AV": av, "SD": sd, "SR": av / sd if sd > 0 else None}
