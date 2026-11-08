"""Stock-return covariance estimators and configurable residual regularization.

Adaptive thresholding follows the nonlinear-factor formulation of Caner and
Daniele (https://arxiv.org/html/2209.04512v3). Diagonal thresholding, uncentered
cross-products, and numerical PSD repair are explicit estimator choices.
The neural-network theory does not automatically apply to BCD residuals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
from sklearn.covariance import LedoitWolf

from bcd_lab._registry import ComponentRegistry


@dataclass
class CovarianceEstimate:
    covariance: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


Estimator = Callable[[np.ndarray, np.ndarray | None, Mapping[str, Any]], CovarianceEstimate]
_ESTIMATORS = ComponentRegistry[Estimator]("Finance estimator", "finance estimators")


def register_estimator(name: str, estimator: Estimator) -> Estimator:
    return _ESTIMATORS.register(name, estimator)


def get_estimator(name: str) -> Estimator:
    return _ESTIMATORS.get(name)


def matrix(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 2 or result.shape[1] < 1:
        raise ValueError("Expected (time >= 2, stocks >= 1) matrix")
    if not np.isfinite(result).all():
        raise ValueError("Covariance inputs must be finite")
    return result


def sample_covariance(values: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.cov(matrix(values), rowvar=False, ddof=1))


def positive_definite(covariance: np.ndarray, floor: float) -> tuple[np.ndarray, float]:
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or not np.isfinite(cov).all():
        raise ValueError("Covariance must be a finite square matrix")
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("Covariance floor must be positive")
    cov = (cov + cov.T) / 2
    bump = max(0.0, floor - float(np.linalg.eigvalsh(cov)[0]))
    return cov + bump * np.eye(len(cov)), bump


def adaptive_residual_covariance(
    residuals: np.ndarray,
    *,
    c_star: float = 1.0,
    beta: float = 2.0,
    d: float = 3.0,
    eps: float = 1e-8,
) -> CovarianceEstimate:
    residuals = matrix(residuals)
    if not all(np.isfinite(v) for v in (c_star, beta, d, eps)):
        raise ValueError("Residual parameters must be finite")
    if c_star < 0 or beta <= 0 or d <= 0 or eps <= 0:
        raise ValueError("Require c_star >= 0 and beta, d, eps > 0")
    count, stocks = residuals.shape
    sample = sample_covariance(residuals)
    # Column blocks bound memory while retaining every residual cross-product.
    theta = np.empty_like(sample)
    block = max(1, (32 * 1024**2) // (8 * count * stocks))
    for start in range(0, stocks, block):
        stop = min(start + block, stocks)
        products = residuals[:, start:stop, None] * residuals[:, None, :]
        theta[start:stop] = np.mean(np.abs(products - sample[None, start:stop]), axis=0)
    l_n = count ** (-beta / (beta + d)) * np.log(count) ** 8
    omega = c_star * np.sqrt(l_n)
    keep = np.abs(sample) >= theta * omega
    thresholded = sample * keep
    thresholded = (thresholded + thresholded.T) / 2
    eigenvalues, vectors = np.linalg.eigh(thresholded)
    repair = bool(eigenvalues[0] < eps)
    covariance = thresholded
    if repair:
        covariance = (vectors * np.maximum(eigenvalues, eps)) @ vectors.T
        covariance = (covariance + covariance.T) / 2 + eps * np.eye(stocks)
    return CovarianceEstimate(
        covariance,
        {
            "omega_n": float(omega),
            "l_n": float(l_n),
            "sparsity_fraction": float(1 - keep.mean()),
            "diagonal_entries_removed": int(stocks - np.diag(keep).sum()),
            "repair_applied": repair,
            "min_eigenvalue_before": float(eigenvalues[0]),
            "min_eigenvalue_after": float(np.linalg.eigvalsh(covariance)[0]),
            "repair_frobenius": float(np.linalg.norm(covariance - thresholded)),
        },
    )


def fitted_residual_covariance(
    observed: np.ndarray, fitted: np.ndarray | None, params: Mapping[str, Any]
) -> CovarianceEstimate:
    observed = matrix(observed)
    if fitted is None or matrix(fitted).shape != observed.shape:
        raise ValueError("Fitted returns must have the same shape as observed returns")
    unknown = set(params) - {"c_star", "beta", "d", "eps", "final_floor"}
    if unknown:
        raise ValueError(f"Unknown fitted-residual covariance parameters: {sorted(unknown)}")
    residuals = observed - fitted
    residual = adaptive_residual_covariance(
        residuals, **{key: value for key, value in params.items() if key != "final_floor"}
    )
    signal = sample_covariance(fitted)
    covariance, bump = positive_definite(
        signal + residual.covariance, params.get("final_floor", 1e-10)
    )
    centered_fit = fitted - fitted.mean(axis=0)
    centered_error = residuals - residuals.mean(axis=0)
    cross = centered_fit.T @ centered_error / (len(observed) - 1)
    return CovarianceEstimate(
        covariance,
        {
            "signal_trace": float(np.trace(signal)),
            "residual_trace": float(np.trace(residual.covariance)),
            "residual": residual.diagnostics,
            "final_diagonal_bump": bump,
            "fit_residual_cross_frobenius": float(np.linalg.norm(cross)),
            "train_rmse": float(np.sqrt(np.mean(residuals**2))),
        },
    )


def _sample(
    observed: np.ndarray, fitted: np.ndarray | None, params: Mapping[str, Any]
) -> CovarianceEstimate:
    if set(params) - {"ridge", "final_floor"}:
        raise ValueError("Unknown sample covariance parameter")
    ridge = float(params.get("ridge", 1e-6))
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("Sample ridge must be finite and nonnegative")
    cov = sample_covariance(observed) + ridge * np.eye(observed.shape[1])
    cov, bump = positive_definite(cov, params.get("final_floor", 1e-10))
    return CovarianceEstimate(cov, {"ridge": ridge, "final_diagonal_bump": bump})


def _ledoit_wolf(
    observed: np.ndarray, fitted: np.ndarray | None, params: Mapping[str, Any]
) -> CovarianceEstimate:
    if set(params) - {"final_floor"}:
        raise ValueError("Unknown Ledoit-Wolf parameter")
    estimator = LedoitWolf().fit(matrix(observed))
    cov, bump = positive_definite(estimator.covariance_, params.get("final_floor", 1e-10))
    return CovarianceEstimate(
        cov, {"shrinkage": float(estimator.shrinkage_), "final_diagonal_bump": bump}
    )


def regularized_residual_covariance(
    observed: np.ndarray, fitted: np.ndarray | None, params: Mapping[str, Any]
) -> CovarianceEstimate:
    """Construct fitted covariance with a configurable residual treatment.

    paper_diagonal uses 3 sqrt(log(N)/T), preserves variances, and shrinks
    the thresholded correlation toward identity only as needed for PSD.
    This diagonal/PSD modification is explicitly distinct from the paper's rule.
    """
    observed = matrix(observed)
    if fitted is None or matrix(fitted).shape != observed.shape:
        raise ValueError("Fitted returns must have the same shape as observed returns")
    if set(params) - {"method", "final_floor"}:
        raise ValueError("Unknown regularized residual covariance parameter")
    floor = float(params.get("final_floor", 1e-10))
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("Covariance floor must be positive")
    method = params.get("method", "diagonal")
    residuals = observed - fitted
    sample = sample_covariance(residuals)
    diagnostics: dict[str, Any] = {"residual_method": method}
    if method == "diagonal":
        residual = np.diag(np.diag(sample))
    elif method == "sample":
        residual = sample
    elif method == "ledoit_wolf":
        estimator = LedoitWolf().fit(residuals)
        residual = estimator.covariance_
        diagnostics["shrinkage"] = float(estimator.shrinkage_)
    elif method == "paper_diagonal":
        residual, details = _paper_diagonal(residuals, sample, floor)
        diagnostics.update(details)
    else:
        raise ValueError(f"Unknown residual method: {method}")
    signal = sample_covariance(fitted)
    covariance, bump = positive_definite(signal + residual, floor)
    centered_fit = fitted - fitted.mean(axis=0)
    centered_error = residuals - residuals.mean(axis=0)
    cross = centered_fit.T @ centered_error / (len(observed) - 1)
    diagnostics.update(
        residual_sample_trace=float(np.trace(sample)),
        residual_trace=float(np.trace(residual)),
        signal_trace=float(np.trace(signal)),
        fit_residual_cross_frobenius=float(np.linalg.norm(cross)),
        fit_residual_cross_trace=float(2 * np.trace(cross)),
        final_diagonal_bump=bump,
    )
    return CovarianceEstimate(covariance, diagnostics)


def _paper_diagonal(
    residuals: np.ndarray, sample: np.ndarray, floor: float
) -> tuple[np.ndarray, dict[str, Any]]:
    count, stocks = residuals.shape
    omega = 3 * np.sqrt(np.log(stocks) / count)
    thresholded = np.empty_like(sample)
    block = max(1, (32 * 1024**2) // (8 * count * stocks))
    for start in range(0, stocks, block):
        stop = min(start + block, stocks)
        products = residuals[:, start:stop, None] * residuals[:, None, :]
        theta = np.mean(np.abs(products - sample[None, start:stop]), axis=0)
        thresholded[start:stop] = sample[start:stop] * (np.abs(sample[start:stop]) >= theta * omega)
    variances = np.maximum(np.diag(sample), floor)
    np.fill_diagonal(thresholded, variances)
    scale = np.sqrt(variances)
    correlation = thresholded / np.outer(scale, scale)
    correlation = (correlation + correlation.T) / 2
    minimum = float(np.linalg.eigvalsh(correlation)[0])
    target = min(1.0, floor / float(variances.min()))
    shrinkage = (target - minimum) / (1 - minimum) if minimum < target else 0.0
    correlation = (1 - shrinkage) * correlation + shrinkage * np.eye(stocks)
    residual = correlation * np.outer(scale, scale)
    return residual, {
        "omega_n": float(omega),
        "correlation_min_eigenvalue_before": minimum,
        "psd_correlation_shrinkage": float(shrinkage),
        "residual_diagonal_floor_entries": int(np.sum(np.diag(sample) < floor)),
    }


register_estimator("fitted_residual", fitted_residual_covariance)
register_estimator("fitted_residual_regularized", regularized_residual_covariance)
register_estimator("sample", _sample)
register_estimator("linear_shrinkage", _ledoit_wolf)
