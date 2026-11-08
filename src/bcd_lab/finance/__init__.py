"""Stock-return covariance experiments and explicitly scoped portfolio evaluation."""

from .covariance import CovarianceEstimate, fitted_residual_covariance, register_estimator
from .evaluation import gmv_weights

__all__ = ["CovarianceEstimate", "fitted_residual_covariance", "gmv_weights", "register_estimator"]
