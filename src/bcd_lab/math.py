"""Observed-entry sampling and sparse math helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from bcd_lab.types import SampleSet


def build_sample_matrix(
    n_rows: int,
    n_cols: int,
    sample_set: SampleSet,
    *,
    dtype: np.dtype | str | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Build a CSR matrix and aligned row/column arrays from observed triples."""
    values, (rows_raw, cols_raw) = sample_set
    values_arr = np.asarray(values, dtype=dtype)
    rows = np.asarray(rows_raw, dtype=np.int64)
    cols = np.asarray(cols_raw, dtype=np.int64)

    sample_matrix = sparse.coo_matrix((values_arr, (rows, cols)), shape=(n_rows, n_cols)).tocsr()
    sample_matrix.sort_indices()
    aligned_cols = sample_matrix.indices.astype(np.int64, copy=False)
    aligned_rows = np.repeat(
        np.arange(sample_matrix.indptr.size - 1, dtype=np.int64),
        np.diff(sample_matrix.indptr),
    )
    return sample_matrix, aligned_rows, aligned_cols


def compute_centers(
    z: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    """Compute sampled inner products ``<u_i, v_j>``."""
    return np.einsum("ir,ir->i", z[rows], z[n_rows + cols])


def svd_z(
    sample_set: SampleSet,
    n_rows: int,
    n_cols: int,
    rank: int,
    *,
    seed: int,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Initialize factor matrix Z from a randomized SVD of the observed matrix."""
    from sklearn.utils.extmath import randomized_svd

    sample_matrix, _, _ = build_sample_matrix(
        n_rows,
        n_cols,
        sample_set,
        dtype=dtype,
    )
    u, singular_values, vt = randomized_svd(
        sample_matrix,
        n_components=rank,
        n_iter="auto",
        n_oversamples=10,
        random_state=seed,
    )
    u = np.asarray(u, dtype=dtype)
    singular_values = np.asarray(singular_values, dtype=dtype)
    vt = np.asarray(vt, dtype=dtype)

    z = np.empty((n_rows + n_cols, rank), dtype=dtype)
    sqrt_s = np.sqrt(singular_values)
    z[:n_rows] = u * sqrt_s
    z[n_rows:] = vt.T * sqrt_s
    return z


def estimate_empirical_mu(
    *,
    n1: int,
    n2: int,
    rank: int,
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Estimate empirical mu parameter from sparse observations using SVD."""
    matrix = np.zeros((n1, n2), dtype=float)
    matrix[rows, cols] = values
    u, singular_vals, vt = np.linalg.svd(matrix, full_matrices=False)
    effective_rank = min(rank, singular_vals.size)
    if effective_rank <= 0:
        return 0.0, {"method": "truncated_svd_zero_filled_train", "effective_rank": 0}
    u_r = u[:, :effective_rank]
    v_r = vt[:effective_rank, :].T
    mu_u = (n1 / effective_rank) * float(np.max(np.sum(u_r * u_r, axis=1)))
    mu_v = (n2 / effective_rank) * float(np.max(np.sum(v_r * v_r, axis=1)))
    return max(mu_u, mu_v), {
        "method": "truncated_svd_zero_filled_train",
        "effective_rank": int(effective_rank),
    }
