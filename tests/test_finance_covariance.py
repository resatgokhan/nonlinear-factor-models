"""Configurable residual covariance estimators retain risk and enforce PSD."""

import numpy as np
import pytest
from sklearn.covariance import LedoitWolf

from bcd_lab.finance.covariance import get_estimator, regularized_residual_covariance


def test_residual_alternatives_retain_the_expected_variances() -> None:
    rng = np.random.default_rng(4)
    fitted = rng.normal(0, 0.01, size=(40, 5))
    residual = rng.normal(0, 0.03, size=(40, 5))
    sample = np.cov(residual, rowvar=False)
    signal = np.cov(fitted, rowvar=False)
    expected = {
        "diagonal": np.diag(np.diag(sample)),
        "sample": sample,
        "ledoit_wolf": LedoitWolf().fit(residual).covariance_,
    }
    estimator = get_estimator("fitted_residual_regularized")
    for method, cov in expected.items():
        result = estimator(fitted + residual, fitted, {"method": method})
        np.testing.assert_allclose(result.covariance, signal + cov, atol=1e-14)
    with pytest.raises(ValueError, match="Unknown residual method"):
        estimator(fitted + residual, fitted, {"method": "bad"})
    with pytest.raises(ValueError, match="same shape"):
        estimator(fitted, None, {})


def test_threshold_psd_repair_preserves_residual_diagonal() -> None:
    covariance = np.array([[1, 0.75, 0.2], [0.75, 1, 0.75], [0.2, 0.75, 1]])
    residual = np.random.default_rng(1).multivariate_normal(np.zeros(3), covariance, size=30)
    result = regularized_residual_covariance(
        residual, np.zeros_like(residual), {"method": "paper_diagonal"}
    )
    assert result.diagnostics["psd_correlation_shrinkage"] > 0
    assert np.linalg.eigvalsh(result.covariance)[0] > 0
    np.testing.assert_allclose(np.diag(result.covariance), residual.var(axis=0, ddof=1))
