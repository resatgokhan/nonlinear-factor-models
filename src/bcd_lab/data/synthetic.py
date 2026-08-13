"""Synthetic low-rank dataset generator."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np

from bcd_lab.function_models import ScalarFunction
from bcd_lab.randomness import RunRandomness
from bcd_lab.types import (
    DatasetBundle,
    DatasetCapabilities,
    FunctionEvaluationDomain,
    FunctionReferences,
    SplitStatistics,
)

from .base import DatasetLoader
from .generating_links import (
    GeneratingLinkContext,
    GeneratingLinkSpec,
    build_generating_link,
    parse_generating_link_spec,
    validate_generating_link_spec,
)

_LINK_EVALUATION_CHUNK_SIZE = 4096


def _evaluation_domain_settings(params: Mapping[str, Any]) -> tuple[float, int] | None:
    value = params.get("evaluation_domain")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("data.evaluation_domain must be a mapping")
    unknown = sorted(set(value) - {"central_mass", "grid_count"})
    if unknown:
        raise ValueError("data.evaluation_domain contains unknown fields: " + ", ".join(unknown))
    central_mass = value.get("central_mass", 0.95)
    if isinstance(central_mass, bool) or not isinstance(central_mass, Real):
        raise TypeError("data.evaluation_domain.central_mass must be a real number")
    central_mass = float(central_mass)
    if not np.isfinite(central_mass) or not 0.0 < central_mass <= 1.0:
        raise ValueError("data.evaluation_domain.central_mass must be in (0, 1]")
    grid_count = value.get("grid_count", 1000)
    if isinstance(grid_count, bool) or not isinstance(grid_count, Integral):
        raise TypeError("data.evaluation_domain.grid_count must be an integer")
    grid_count = int(grid_count)
    if grid_count < 2:
        raise ValueError("data.evaluation_domain.grid_count must be at least 2")
    return central_mass, grid_count


def _build_evaluation_domain(
    x_star: np.ndarray,
    params: Mapping[str, Any],
    dtype: np.dtype,
) -> FunctionEvaluationDomain | None:
    settings = _evaluation_domain_settings(params)
    if settings is None:
        return None
    central_mass, grid_count = settings
    tail = (1.0 - central_mass) / 2.0
    bounds = np.asarray(
        np.quantile(np.asarray(x_star).ravel(), [tail, 1.0 - tail]),
        dtype=float,
    )
    lower = float(bounds[0])
    upper = float(bounds[1])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("data.evaluation_domain requires non-constant finite latent truth")
    return FunctionEvaluationDomain(
        grid=np.linspace(lower, upper, grid_count, dtype=dtype),
        source="latent_truth_central_quantile",
        central_mass=central_mass,
    )


def _generating_link_spec(params: dict[str, Any]) -> GeneratingLinkSpec:
    value = params.get(
        "generating_link",
        {"name": "identity", "params": {}},
    )
    return parse_generating_link_spec(value)


def _generate_linked_values(
    *,
    x_star: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    spec: GeneratingLinkSpec,
    rng: np.random.Generator,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, ScalarFunction]:
    latent_samples = np.asarray(x_star[rows, cols], dtype=dtype)
    truth = build_generating_link(
        spec,
        GeneratingLinkContext(
            latent_samples=latent_samples,
            dtype=dtype,
            rng=rng,
        ),
    )
    latent_flat = np.asarray(x_star, dtype=dtype).ravel()
    linked_flat = np.empty_like(latent_flat)
    for start in range(0, latent_flat.size, _LINK_EVALUATION_CHUNK_SIZE):
        stop = min(start + _LINK_EVALUATION_CHUNK_SIZE, latent_flat.size)
        linked_flat[start:stop] = truth.eval_array(latent_flat[start:stop])
    linked_truth = linked_flat.reshape(x_star.shape)
    if not np.all(np.isfinite(linked_truth)):
        raise ValueError(f"data.generating_link {spec.name!r} produced non-finite linked truth")
    return linked_truth[rows, cols].copy(), linked_truth, truth


def _balanced_sample_indices(
    n1: int,
    n2: int,
    m: int,
    *,
    min_per_row: int = 1,
    min_per_col: int = 1,
    rng: np.random.Generator,
    max_tries: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    if n1 <= 0 or n2 <= 0:
        raise ValueError("n1 and n2 must be positive")
    if m <= 0:
        raise ValueError("m must be positive")
    if min_per_row < 0 or min_per_col < 0:
        raise ValueError("min_per_row and min_per_col must be non-negative")
    if m > n1 * n2:
        raise ValueError("m cannot exceed n1 * n2")
    required_min = max(n1 * min_per_row, n2 * min_per_col)
    if m < required_min:
        raise ValueError("m is too small to satisfy min_per_row/min_per_col requirements")

    total = n1 * n2
    indices = np.arange(total)
    for _ in range(max_tries):
        rng.shuffle(indices)
        row_counts = np.zeros(n1, dtype=int)
        col_counts = np.zeros(n2, dtype=int)
        chosen_mask = np.zeros(total, dtype=bool)
        chosen: list[int] = []

        for idx in indices:
            if len(chosen) >= m:
                break
            r, c = divmod(int(idx), n2)
            if row_counts[r] < min_per_row or col_counts[c] < min_per_col:
                chosen_mask[idx] = True
                chosen.append(idx)
                row_counts[r] += 1
                col_counts[c] += 1

        if (row_counts < min_per_row).any() or (col_counts < min_per_col).any():
            continue

        if len(chosen) < m:
            remaining = np.flatnonzero(~chosen_mask)
            extra = rng.choice(remaining, size=m - len(chosen), replace=False)
            chosen.extend(int(x) for x in extra)

        chosen_array = np.array(chosen, dtype=int)
        if chosen_array.size != m:
            raise RuntimeError("Balanced sampler did not produce the requested sample size.")
        rows, cols = divmod(chosen_array, n2)
        return rows.astype(int, copy=False), cols.astype(int, copy=False)

    raise RuntimeError("Failed to build a balanced sample set; try increasing m.")


class SyntheticDatasetLoader(DatasetLoader):
    """Generates synthetic low-rank datasets."""

    name = "synthetic"

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate the configured generating-link boundary before data work."""
        validate_generating_link_spec(_generating_link_spec(params))
        _evaluation_domain_settings(params)

    def load(
        self,
        *,
        params: dict[str, Any],
        seed: int,
    ) -> DatasetBundle:
        """Generate a synthetic low-rank matrix and sample observed entries.

        Args:
            params: Parameters including a required ``shape`` mapping with
                n_rows, n_cols, rank, and n_observations, plus type, noise_std,
                noise_const, desirable, etc.
            seed: Random seed.

        Returns:
            A DatasetBundle containing train sample set, ground-truth matrices, and metadata.
        """
        shape = params["shape"]
        n1 = int(shape["n_rows"])
        n2 = int(shape["n_cols"])
        rank = int(shape["rank"])
        m = int(shape["n_observations"])

        data_type = str(params.get("type", "standard_normal"))
        noise_std = params.get("noise_std")
        noise_const = params.get("noise_const")
        desirable = bool(params.get("desirable", False))
        dtype_str = params.get("dtype", "float64")
        dtype = np.dtype(dtype_str)

        if desirable:
            return self._load_desirable(
                n1=n1,
                n2=n2,
                rank=rank,
                m=m,
                data_type=data_type,
                noise_std=noise_std,
                noise_const=noise_const,
                seed=seed,
                dtype=dtype,
                params=params,
            )

        randomness = RunRandomness(seed)
        matrix_and_samples_rng = randomness.generator("matrix_and_samples")

        if data_type == "standard_normal":
            matrix = matrix_and_samples_rng.standard_normal((n1, n2), dtype=dtype)
        elif data_type == "discrete":
            matrix = matrix_and_samples_rng.integers(
                low=0, high=11, size=(n1, n2), endpoint=False
            ).astype(dtype, copy=False)
        else:
            raise ValueError(f"Unknown synthetic data type: {data_type!r}")

        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        u_r = u[:, :rank].astype(dtype, copy=False)
        s_r = s[:rank].astype(dtype, copy=False)
        vt_r = vt[:rank, :].astype(dtype, copy=False)

        u_row_sq_norms = np.einsum("ij,ij->i", u_r, u_r)
        v_row_sq_norms = np.einsum("ij,ij->i", vt_r.T, vt_r.T)
        mu_u = (n1 / rank) * float(u_row_sq_norms.max())
        mu_v = (n2 / rank) * float(v_row_sq_norms.max())
        mu = max(mu_u, mu_v)

        sigma_1_star = float(s_r[0])
        sigma_r_star = float(s_r[-1])

        x_star = (u_r * s_r) @ vt_r

        flat_indices = matrix_and_samples_rng.choice(n1 * n2, size=m, replace=False)
        rows, cols = divmod(flat_indices, n2)
        generating_spec = _generating_link_spec(params)
        values, linked_truth, generating_truth = _generate_linked_values(
            x_star=x_star,
            rows=rows,
            cols=cols,
            spec=generating_spec,
            rng=randomness.generator("generating_link"),
            dtype=dtype,
        )
        noise_rng = randomness.generator("observation_noise")
        if noise_std is not None:
            values = values + noise_rng.normal(0.0, float(noise_std), size=values.shape).astype(
                dtype, copy=False
            )
        if noise_const is not None:
            values = values + np.asarray(noise_const, dtype=dtype)
        sample_set = (values, (rows, cols))

        s_sqrt = np.sqrt(s_r)
        z_star = np.zeros(shape=(n1 + n2, rank), dtype=dtype)
        z_star[:n1] = u_r * s_sqrt
        z_star[n1:] = vt_r.T * s_sqrt

        metadata = {
            "mu": mu,
            "sigma_1_star": sigma_1_star,
            "sigma_r_star": sigma_r_star,
            "data_type": data_type,
            "dtype": dtype_str,
            "generating_link": generating_spec.name,
        }

        return DatasetBundle(
            source=self.name,
            n_rows=n1,
            n_cols=n2,
            rank=rank,
            train=sample_set,
            dense_truth=x_star,
            latent_truth=z_star,
            capabilities=DatasetCapabilities(
                functions=FunctionReferences(truth=generating_truth),
                function_domain=_build_evaluation_domain(x_star, params, dtype),
                linked_truth=linked_truth,
                splits=SplitStatistics(train_count=int(values.size)),
            ),
            metadata=metadata,
        )

    def _load_desirable(
        self,
        *,
        n1: int,
        n2: int,
        rank: int,
        m: int,
        data_type: str,
        noise_std: float | None,
        noise_const: float | None,
        seed: int,
        dtype: np.dtype,
        params: dict[str, Any],
    ) -> DatasetBundle:
        condition_number = float(params.get("condition_number", 2.0))
        min_per_row = int(params.get("min_per_row", 1))
        min_per_col = int(params.get("min_per_col", 1))
        if rank > min(n1, n2):
            raise ValueError("rank must be <= min(n1, n2)")
        if condition_number < 1.0:
            raise ValueError("condition_number must be >= 1.0")

        randomness = RunRandomness(seed)
        rng_factors = randomness.generator("factors")
        a = rng_factors.standard_normal((n1, rank))
        b = rng_factors.standard_normal((n2, rank))
        u, _ = np.linalg.qr(a)
        v, _ = np.linalg.qr(b)

        if rank == 1:
            s_r = np.array([1.0], dtype=dtype)
        else:
            exponents = np.linspace(0.0, 1.0, rank)
            s_r = condition_number ** (-exponents)
            s_r = np.asarray(s_r, dtype=dtype)

        u_r = u[:, :rank].astype(dtype, copy=False)
        vt_r = v[:, :rank].T.astype(dtype, copy=False)
        x_star = (u_r * s_r) @ vt_r

        u_row_sq_norms = np.einsum("ij,ij->i", u_r, u_r)
        v_row_sq_norms = np.einsum("ij,ij->i", vt_r.T, vt_r.T)
        mu_u = (n1 / rank) * float(u_row_sq_norms.max())
        mu_v = (n2 / rank) * float(v_row_sq_norms.max())
        mu = max(mu_u, mu_v)

        sigma_1_star = float(s_r[0])
        sigma_r_star = float(s_r[-1])

        rng_samples = randomness.generator("samples_and_noise")
        rows, cols = _balanced_sample_indices(
            n1,
            n2,
            m,
            min_per_row=min_per_row,
            min_per_col=min_per_col,
            rng=rng_samples,
        )
        generating_spec = _generating_link_spec(params)
        values, linked_truth, generating_truth = _generate_linked_values(
            x_star=x_star,
            rows=rows,
            cols=cols,
            spec=generating_spec,
            rng=randomness.generator("generating_link"),
            dtype=dtype,
        )
        noise_rng = randomness.generator("observation_noise")
        if noise_std is not None:
            values = values + noise_rng.normal(0.0, float(noise_std), size=values.shape).astype(
                dtype, copy=False
            )
        if noise_const is not None:
            values = values + np.asarray(noise_const, dtype=dtype)
        sample_set = (values, (rows, cols))

        s_sqrt = np.sqrt(s_r)
        z_star = np.zeros(shape=(n1 + n2, rank), dtype=dtype)
        z_star[:n1] = u_r * s_sqrt
        z_star[n1:] = vt_r.T * s_sqrt

        metadata = {
            "mu": mu,
            "sigma_1_star": sigma_1_star,
            "sigma_r_star": sigma_r_star,
            "data_type": data_type,
            "dtype": str(dtype),
            "desirable": True,
            "condition_number": condition_number,
            "generating_link": generating_spec.name,
        }

        return DatasetBundle(
            source=self.name,
            n_rows=n1,
            n_cols=n2,
            rank=rank,
            train=sample_set,
            dense_truth=x_star,
            latent_truth=z_star,
            capabilities=DatasetCapabilities(
                functions=FunctionReferences(truth=generating_truth),
                function_domain=_build_evaluation_domain(x_star, params, dtype),
                linked_truth=linked_truth,
                splits=SplitStatistics(train_count=int(values.size)),
            ),
            metadata=metadata,
        )
