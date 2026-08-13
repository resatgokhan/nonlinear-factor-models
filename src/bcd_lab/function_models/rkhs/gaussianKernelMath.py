"""Low-level Gaussian kernel math functions."""

from __future__ import annotations

import numpy as np


def _gaussian_kernel_from_diff(diff: np.ndarray, sigma: float) -> np.ndarray:
    """Compute ``exp(-(diff**2) / (2 * sigma**2))`` from precomputed differences."""
    return np.exp(-(diff**2) / (2.0 * sigma**2))


def gaussian_kernel(x: np.ndarray | float, c: np.ndarray | float, sigma: float) -> np.ndarray:
    """Evaluate the scalar/array Gaussian kernel ``k(x, c)``."""
    dtype = np.result_type(x, c, float)
    x_arr = np.asarray(x, dtype=dtype)
    c_arr = np.asarray(c, dtype=dtype)
    return _gaussian_kernel_from_diff(x_arr - c_arr, dtype.type(sigma))


def gaussian_kernel_matrix(x: np.ndarray, c: np.ndarray, sigma: float) -> np.ndarray:
    """Return ``K[i, j] = k(x_i, c_j)`` for one-dimensional arrays."""
    dtype = np.result_type(x, c, float)
    x_arr = np.asarray(x, dtype=dtype).ravel()
    c_arr = np.asarray(c, dtype=dtype).ravel()
    diff = x_arr[:, None] - c_arr[None, :]
    return _gaussian_kernel_from_diff(diff, dtype.type(sigma))


def gaussian_dkernel_dx(
    x: np.ndarray | float,
    c: np.ndarray | float,
    sigma: float,
) -> np.ndarray:
    """Derivative of the Gaussian kernel with respect to ``x``."""
    dtype = np.result_type(x, c, float)
    x_arr = np.asarray(x, dtype=dtype)
    c_arr = np.asarray(c, dtype=dtype)
    diff = c_arr - x_arr
    sigma_val = dtype.type(sigma)
    return (diff / (sigma_val**2)) * _gaussian_kernel_from_diff(diff, sigma_val)


def gaussian_dkernel_dx_matrix(x: np.ndarray, c: np.ndarray, sigma: float) -> np.ndarray:
    """Return ``d/dx k(x_i, c_j)`` for one-dimensional arrays."""
    dtype = np.result_type(x, c, float)
    x_arr = np.asarray(x, dtype=dtype).ravel()
    c_arr = np.asarray(c, dtype=dtype).ravel()
    diff = c_arr[None, :] - x_arr[:, None]
    sigma_val = dtype.type(sigma)
    kernel = _gaussian_kernel_from_diff(diff, sigma_val)
    return (diff / (sigma_val**2)) * kernel
