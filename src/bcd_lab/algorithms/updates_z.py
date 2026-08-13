"""Latent factors (Z) update functions for BCD."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

try:
    from scipy import sparse
except ImportError:
    sparse = None

from bcd_lab.math import compute_centers

from .numerics import require_finite
from .registry import AlgorithmContext
from .state import SolverState


def project_z(state: SolverState, context: AlgorithmContext) -> None:
    """Project Z rows to satisfy the row norm bound derived from mu."""
    if not context.config.bcd.project_z:
        return

    beta = state.cache.get("beta")
    if beta is None:
        mu = context.config.regularization.mu
        if mu is not None and mu > 0:
            n_rows = context.config.shape.n_rows
            n_cols = context.config.shape.n_cols
            norm_z0 = np.linalg.norm(state.z, "fro")
            beta = (4.0 / 3.0) * math.sqrt(mu / (n_rows + n_cols)) * norm_z0
            state.cache["beta"] = beta
        else:
            return

    z_norms = np.linalg.norm(state.z, axis=1, keepdims=True)
    scale = np.minimum(1.0, beta / np.maximum(z_norms, 1e-12))
    state.z *= scale
    state.centers = None


def apply_lambda(state: SolverState, context: AlgorithmContext) -> None:
    """Apply Z regularization gradient update based on lambda_."""
    lambda_ = context.config.regularization.lambda_
    if not (lambda_ is not None and 0.0 < lambda_ <= 1.0):
        return

    z = state.z
    n1 = context.config.shape.n_rows
    n2 = context.config.shape.n_cols

    s1 = 1.0 / n1
    s2 = -1.0 / n2

    # Y = D Z via row scaling (avoiding building dense D)
    y = z.copy()
    y[:n1] *= s1
    y[n1:] *= s2

    # A = Z^T (D Z)
    a = z.T @ y

    # reg_grad = (D Z) (Z^T D Z)
    reg_grad = y @ a

    state.z -= state.z_optimizer.learning_rate * lambda_ * reg_grad
    state.centers = None


def _accumulate_z_gradient(
    state: SolverState,
    context: AlgorithmContext,
    grad_z: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    chunk_size: int,
    iteration: int,
) -> None:
    n1 = context.config.shape.n_rows
    z = state.z
    m = len(data)
    phi = state.phi
    if phi is None:
        raise ValueError("compute_z_gradient requires state.phi to be initialized")

    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        batch_rows = rows[start:end]
        batch_cols = cols[start:end]
        z_rows = z[batch_rows]
        z_cols = z[n1 + batch_cols]
        inner_products = np.sum(z_rows * z_cols, axis=1)
        phi_vals, phi_d = phi.eval_and_d_array(inner_products)
        require_finite(phi_vals, "link predictions", iteration)
        require_finite(phi_d, "link derivatives", iteration)
        prediction_gradient = context.objective.prediction_gradient(
            data[start:end],
            phi_vals,
        )
        require_finite(prediction_gradient, "objective prediction gradient", iteration)
        multipliers = prediction_gradient * phi_d
        np.add.at(grad_z, batch_rows, multipliers[:, None] * z_cols)
        np.add.at(grad_z, n1 + batch_cols, multipliers[:, None] * z_rows)


def compute_z_gradient(state: SolverState, context: AlgorithmContext, iteration: int) -> np.ndarray:
    """Compute gradient of the loss with respect to Z for learned links."""
    grad_z = np.zeros_like(state.z)
    _accumulate_z_gradient(
        state,
        context,
        grad_z,
        state.rows,
        state.cols,
        state.values,
        chunk_size=10000,
        iteration=iteration,
    )
    require_finite(grad_z, "Z gradient", iteration)
    state.cache["z_gradient"] = grad_z
    return grad_z


def compute_z_gradient_identity(
    state: SolverState, context: AlgorithmContext, iteration: int
) -> np.ndarray:
    """Compute gradient of the loss with respect to Z assuming identity link."""
    n1 = context.config.shape.n_rows
    n2 = context.config.shape.n_cols
    z = state.z
    rows = state.rows
    cols = state.cols
    data = state.values

    z_u = z[:n1]
    z_v = z[n1:]
    inner_products = np.einsum("ij,ij->i", z_u[rows], z_v[cols])
    prediction_gradient = context.objective.prediction_gradient(data, inner_products)
    require_finite(prediction_gradient, "objective prediction gradient", iteration)

    if sparse is None:
        grad_z = np.zeros_like(state.z)
        m = len(data)
        chunk_size = 4096
        for start in range(0, m, chunk_size):
            end = min(start + chunk_size, m)
            batch_rows = rows[start:end]
            batch_cols = cols[start:end]
            z_rows = z_u[batch_rows]
            z_cols = z_v[batch_cols]
            batch_gradient = prediction_gradient[start:end]
            np.add.at(grad_z, batch_rows, batch_gradient[:, None] * z_cols)
            np.add.at(grad_z, n1 + batch_cols, batch_gradient[:, None] * z_rows)
        require_finite(grad_z, "Z gradient", iteration)
        state.cache["z_gradient"] = grad_z
        return grad_z

    r_mat = state.cache.get("_residual_csr")
    if r_mat is None:
        row_counts = np.bincount(rows, minlength=n1)
        indptr = np.zeros(n1 + 1, dtype=np.int64)
        np.cumsum(row_counts, out=indptr[1:])
        r_mat = sparse.csr_matrix(
            (prediction_gradient, cols, indptr),
            shape=(n1, n2),
        )
        state.cache["_residual_csr"] = r_mat
    else:
        r_mat.data = prediction_gradient

    grad_u = r_mat @ z_v
    grad_v = r_mat.T @ z_u
    grad_z = np.vstack([grad_u, grad_v])
    require_finite(grad_z, "Z gradient", iteration)
    state.cache["z_gradient"] = grad_z
    return grad_z


def make_z_updater(
    *,
    is_identity: bool = False,
) -> Callable[[SolverState, AlgorithmContext, int], None]:
    """Factory to build modular Z updates with regularization and projections."""

    def _update(state: SolverState, context: AlgorithmContext, iteration: int) -> None:
        # 1. Compute gradient and cache it in state.cache["z_gradient"] for diagnostics.
        #    The cache is naturally overwritten next iteration when this runs again.
        if is_identity:
            grad_z = compute_z_gradient_identity(state, context, iteration)
        else:
            grad_z = compute_z_gradient(state, context, iteration)

        # Apply only the optimizer-preconditioned data-gradient update here.
        optimizer_update = state.z_optimizer.compute_update(grad_z)
        require_finite(optimizer_update, "Z optimizer update", iteration)
        state.cache["z_optimizer_update_norm"] = float(np.linalg.norm(optimizer_update))
        state.z -= optimizer_update
        require_finite(state.z, "Z state after gradient update", iteration)

        # 2. Regularization update
        apply_lambda(state, context)
        require_finite(state.z, "Z state after regularization", iteration)

        # 3. Row norm projection
        project_z(state, context)
        require_finite(state.z, "Z state after projection", iteration)

        # Align centers
        state.centers = compute_centers(
            state.z, state.rows, state.cols, context.config.shape.n_rows
        )
        require_finite(state.centers, "training centers", iteration)

    return _update
