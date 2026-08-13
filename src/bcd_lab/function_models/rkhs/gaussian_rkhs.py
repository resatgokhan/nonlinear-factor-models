"""Gaussian RKHS representation and kernel operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from bcd_lab.function_models.base import FunctionConstraintConfig, ScalarFunction
from bcd_lab.function_models.distances import function_distance

from .gaussianKernelMath import (
    _gaussian_kernel_from_diff,
    gaussian_kernel_matrix,
)
from .gaussianKernelMath import (
    gaussian_dkernel_dx as gaussian_dkernel_dx,
)
from .gaussianKernelMath import (
    gaussian_dkernel_dx_matrix as gaussian_dkernel_dx_matrix,
)
from .gaussianKernelMath import (
    gaussian_kernel as gaussian_kernel,
)
from .projection import (
    project_gaussian_ball,
    project_gaussian_constrained,
    project_gaussian_unconstrained,
)

if TYPE_CHECKING:
    from bcd_lab.config import ModelBuildContext
    from bcd_lab.types import DatasetBundle


@dataclass(frozen=True)
class GaussianRKHSConfig(FunctionConstraintConfig):
    """Configuration for constructing Gaussian RKHS models."""

    sigma: float = 1.0
    unconstrained_ridge: float = 1e-8
    constrained_percentage: float = 0.0
    constrained_lam: float = 1e-8
    lower: float = -3.0
    upper: float = 3.0

    @classmethod
    def from_link_config(cls, link_config: Any) -> GaussianRKHSConfig:
        link_params = link_config.params
        defaults = cls(sigma=link_config.sigma)

        def float_param(name: str, default: float | None) -> float | None:
            value = link_params.get(name, default)
            return float(value) if value is not None else None

        phi_0 = link_params.get("phi_0", defaults.ball_center_phi_0)
        return cls(
            sigma=link_config.sigma,
            unconstrained_ridge=float(
                link_params.get("unconstrained_ridge", defaults.unconstrained_ridge)
            ),
            constrained_percentage=float(
                link_params.get("constrained_percentage", defaults.constrained_percentage)
            ),
            constrained_lam=float(link_params.get("constrained_lam", defaults.constrained_lam)),
            lower=float(link_params.get("lower", defaults.lower)),
            upper=float(link_params.get("upper", defaults.upper)),
            monotone_enabled=bool(
                link_params.get("monotone_phi_enabled", defaults.monotone_enabled)
            ),
            monotone_direction=link_params.get(
                "monotone_phi_direction", defaults.monotone_direction
            ),
            clamp_enabled=bool(link_params.get("phi_output_clamp_enabled", defaults.clamp_enabled)),
            clamp_min=float_param("phi_output_clamp_min", defaults.clamp_min),
            clamp_max=float_param("phi_output_clamp_max", defaults.clamp_max),
            ball_enabled=bool(link_params.get("ball_enabled", defaults.ball_enabled)),
            ball_center_phi_0=phi_0,
            ball_radius=float_param("ball_radius", defaults.ball_radius),
        )


class GaussianRKHS:
    """Finite Gaussian RKHS expansion ``sum_i coef_i k(scalar_i, .)``."""

    kind = "gaussian_rkhs"
    checkpoint_version = 1

    def __init__(
        self,
        sigma: float,
        scalars: np.ndarray,
        coefs: np.ndarray,
        *,
        config: GaussianRKHSConfig | None = None,
    ) -> None:
        self.sigma = float(sigma)
        scalars_arr = np.asarray(scalars)
        coefs_arr = np.asarray(coefs)
        if scalars_arr.ndim != 1:
            scalars_arr = scalars_arr.ravel()
        if coefs_arr.ndim != 1:
            coefs_arr = coefs_arr.ravel()
        if scalars_arr.size != coefs_arr.size:
            raise ValueError(
                "scalars and coefs must have the same length; "
                f"got {scalars_arr.size} vs {coefs_arr.size}"
            )
        self.scalars = scalars_arr
        self.coefs = coefs_arr
        self._config = (
            config
            if config is not None
            else GaussianRKHSConfig(
                sigma=self.sigma,
            )
        )

    @property
    def config(self) -> GaussianRKHSConfig:
        """Construction and projection policy attached to this model."""
        return self._config

    def copy(self, *, dtype: np.dtype | str | None = None) -> "GaussianRKHS":
        target_dtype = np.dtype(dtype) if dtype is not None else self.coefs.dtype
        return GaussianRKHS(
            self.sigma,
            np.array(self.scalars, copy=True, dtype=target_dtype),
            np.array(self.coefs, copy=True, dtype=target_dtype),
            config=self._config,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit or interpolate the model parameters to approximate the given (x, y) coordinates."""
        self.scalars = np.asarray(x, dtype=self.coefs.dtype).copy()
        K = gaussian_kernel_matrix(self.scalars, self.scalars, self.sigma)
        ridge = getattr(self._config, "unconstrained_ridge", 1e-8)
        self.coefs = np.linalg.solve(
            K + ridge * np.eye(K.shape[0], dtype=K.dtype), y.astype(self.coefs.dtype, copy=False)
        )

    def project_function_onto_span(
        self,
        function: ScalarFunction,
        *,
        ridge: float = 0.0,
    ) -> "GaussianRKHS":
        """Interpolate ``function`` in this model's Gaussian kernel span."""
        if not np.isfinite(ridge) or ridge < 0.0:
            raise ValueError("Gaussian RKHS projection ridge must be finite and non-negative")
        centers = np.asarray(self.scalars)
        if centers.size == 0:
            raise ValueError("Gaussian RKHS projection requires at least one center")
        kernel = gaussian_kernel_matrix(centers, centers, self.sigma)
        kernel = 0.5 * (kernel + kernel.T)
        values = np.asarray(function.eval_array(centers), dtype=kernel.dtype)
        if values.shape != centers.shape or not np.all(np.isfinite(values)):
            raise ValueError("function projection values must be finite and aligned with centers")
        system = kernel + ridge * np.eye(centers.size, dtype=kernel.dtype)
        coefficients, *_ = np.linalg.lstsq(system, values, rcond=None)
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("Gaussian RKHS projection produced non-finite coefficients")
        return GaussianRKHS(
            self.sigma,
            centers.copy(),
            np.asarray(coefficients, dtype=self.coefs.dtype),
        )

    def distance_to(
        self,
        other: object,
        *,
        mode: str = "auto",
        grid: np.ndarray | None = None,
    ) -> float:
        if mode in {"auto", "rkhs"} and isinstance(other, GaussianRKHS):
            return self.diff_norm(other)
        if mode not in {"auto", "l2_grid"}:
            raise ValueError(f"Unknown RKHS distance mode {mode!r}")
        if grid is None:
            grid = np.linspace(-3.0, 3.0, 1000)
        vals_self = self.eval_array(grid)
        vals_other = other.eval_array(grid)  # type: ignore[attr-defined]
        dx = float(grid[1] - grid[0])
        return float(np.sqrt(np.sum((vals_self - vals_other) ** 2) * dx))

    def norm(self, *, mode: str = "auto") -> float | None:
        if mode in {"auto", "rkhs"}:
            return self.rkhs_norm()
        return None

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "sigma": float(self.sigma),
            "scalars": np.asarray(self.scalars, dtype=float),
            "coefs": np.asarray(self.coefs, dtype=float),
        }

    def checkpoint_state(self) -> Mapping[str, Any]:
        return {
            "sigma": self.sigma,
            "scalars": self.scalars.copy(),
            "coefs": self.coefs.copy(),
        }

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        version: int,
    ) -> None:
        if version != self.checkpoint_version:
            raise ValueError(f"unsupported GaussianRKHS state version {version}")
        sigma = float(state["sigma"])
        if not np.isclose(sigma, self.sigma):
            raise ValueError(
                f"GaussianRKHS checkpoint sigma {sigma} does not match config {self.sigma}"
            )
        scalars = np.asarray(state["scalars"])
        coefs = np.asarray(state["coefs"])
        if scalars.ndim != 1 or coefs.ndim != 1 or scalars.shape != coefs.shape:
            raise ValueError("GaussianRKHS checkpoint arrays must be aligned vectors")
        self.scalars = scalars.copy()
        self.coefs = coefs.copy()

    def gradient(
        self,
        *,
        inputs: np.ndarray,
        residuals: np.ndarray,
    ) -> "GaussianRKHS":
        """Return an RKHS expansion representing a residual-weighted update direction."""
        x = np.asarray(inputs, dtype=self.scalars.dtype).ravel()
        residual_arr = np.asarray(residuals, dtype=self.coefs.dtype).ravel()
        if x.size != residual_arr.size:
            raise ValueError("inputs and residuals must have the same length")
        return GaussianRKHS(self.sigma, x, residual_arr)

    def apply_gradient(self, gradient: "GaussianRKHS", *, step_size: float) -> None:
        if not isinstance(gradient, GaussianRKHS):
            raise TypeError("GaussianRKHS.apply_gradient requires a GaussianRKHS gradient")
        if not np.isclose(self.sigma, gradient.sigma):
            raise ValueError("gradient sigma must match function sigma")
        self.scalars = np.concatenate([self.scalars, gradient.scalars])
        self.coefs = np.concatenate([self.coefs, float(step_size) * gradient.coefs])

    def project(
        self,
        *,
        centers: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        l2_grid: np.ndarray | None = None,
        phi_ball_radius: float | None = None,
    ) -> None:
        """Apply in-place RKHS-compatible projection constraints using this model's configuration."""
        if centers is None:
            return
        centers_arr = np.asarray(centers)

        if rng is None:
            raise ValueError("GaussianRKHS.project requires an explicit random generator")

        cfg = self.config

        # 1. Bounds/monotonicity constraint (using cvxpy QP solver)
        constrained_percentage = getattr(cfg, "constrained_percentage", 0.0)
        use_constrained = rng.random() < constrained_percentage

        if use_constrained:
            eval_points = rng.uniform(-2.5, 2.5, 200)

            projected = project_gaussian_constrained(
                sigma=self.sigma,
                source_scalars=self.scalars,
                source_coefs=self.coefs,
                centers=centers_arr,
                lower=cfg.lower,
                upper=cfg.upper,
                lam=cfg.constrained_lam,
                eval_points=eval_points,
                mode="high",
            )
        else:
            # Standard unconstrained projection to cap center size to training centers
            projected = project_gaussian_unconstrained(
                sigma=self.sigma,
                source_scalars=self.scalars,
                source_coefs=self.coefs,
                centers=centers_arr,
                ridge=cfg.unconstrained_ridge,
            )

        self.scalars = projected.scalars
        self.coefs = projected.coefs

        # 2. L2 Ball constraint
        ball_enabled = getattr(cfg, "ball_enabled", False)
        phi_0 = getattr(cfg, "ball_center_phi_0", None)
        radius = (
            phi_ball_radius if phi_ball_radius is not None else getattr(cfg, "ball_radius", None)
        )

        if ball_enabled and phi_0 is not None and radius is not None:
            ridge = 1e-8
            if not isinstance(phi_0, GaussianRKHS):
                raise TypeError("Gaussian RKHS ball center must be a GaussianRKHS")
            distance = self.diff_norm(phi_0)
            projected = project_gaussian_ball(
                sigma=self.sigma,
                source_scalars=self.scalars,
                source_coefs=self.coefs,
                centers=centers_arr,
                center_scalars=phi_0.scalars,
                center_coefs=phi_0.coefs,
                distance=distance,
                radius=float(radius),
                ridge=ridge,
            )
            self.scalars = projected.scalars
            self.coefs = projected.coefs

    def resolve_projection_ball_radius(
        self,
        *,
        truth: object | None,
    ) -> float | None:
        """Resolve this model's configured projection radius from model-owned state."""
        cfg = self.config
        if not cfg.ball_enabled:
            return None
        if cfg.ball_radius is not None:
            return float(cfg.ball_radius)
        center = cfg.ball_center_phi_0
        if center is None:
            raise ValueError("ball projection requires a configured center function")
        if truth is None:
            raise ValueError("inferred ball radius requires a dataset truth function")
        return function_distance(center, truth)

    def _kernel_and_diff(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y_arr = np.asarray(y, dtype=self.scalars.dtype).ravel()
        diff = self.scalars[None, :] - y_arr[:, None]
        kernel_values = _gaussian_kernel_from_diff(diff, self.sigma)
        return diff, kernel_values

    def __call__(self, y: float | np.ndarray) -> float | np.ndarray:
        y_arr = np.asarray(y)
        values = self.eval_array(y_arr)
        if y_arr.shape == () or y_arr.size == 1:
            return float(values.reshape(-1)[0])
        return values

    def d(self, y: float | np.ndarray) -> float | np.ndarray:
        y_arr = np.asarray(y)
        values = self.d_array(y_arr)
        if y_arr.shape == () or y_arr.size == 1:
            return float(values.reshape(-1)[0])
        return values

    def scale_coefs(self, scale: float) -> None:
        self.coefs *= scale

    def scale_parameters(self, scale: float) -> None:
        self.scale_coefs(scale)

    def __iadd__(self, other: "GaussianRKHS") -> "GaussianRKHS":
        if not np.isclose(self.sigma, other.sigma):
            raise ValueError("Cannot add RKHS functions with different sigma values")
        self.scalars = np.concatenate([self.scalars, other.scalars])
        self.coefs = np.concatenate([self.coefs, other.coefs])
        return self

    def eval_array(self, y: np.ndarray) -> np.ndarray:
        """Vectorized function evaluation."""
        _, kernel_values = self._kernel_and_diff(y)
        return kernel_values @ self.coefs

    def d_array(self, y: np.ndarray) -> np.ndarray:
        """Vectorized derivative evaluation."""
        diff, kernel_values = self._kernel_and_diff(y)
        d_values = (diff / (self.sigma**2)) * kernel_values
        return d_values @ self.coefs

    def eval_and_d_array(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate function values and derivatives in one kernel pass."""
        diff, kernel_values = self._kernel_and_diff(y)
        values = kernel_values @ self.coefs
        d_values = (diff / (self.sigma**2)) * kernel_values
        return values, d_values @ self.coefs

    def gram(self) -> np.ndarray:
        """Gram matrix for this expansion's centers."""
        diff = self.scalars[:, None] - self.scalars[None, :]
        return _gaussian_kernel_from_diff(diff, self.sigma)

    def rkhs_norm(self, ridge: float = 1.0e-8) -> float:
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        kernel = self.gram()
        if ridge:
            kernel = kernel + ridge * np.eye(kernel.shape[0], dtype=kernel.dtype)
        norm_sq = float(self.coefs @ (kernel @ self.coefs))
        return float(np.sqrt(max(norm_sq, 0.0)))

    def cond(self, ridge: float = 0.0) -> float:
        """2-norm condition number of the Gram matrix."""
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        kernel = self.gram()
        if ridge:
            kernel = kernel + ridge * np.eye(kernel.shape[0], dtype=kernel.dtype)
        return float(np.linalg.cond(kernel))

    def diff_norm(self, other: "GaussianRKHS", ridge: float = 1.0e-8) -> float:
        """RKHS distance to another Gaussian RKHS function."""
        if not np.isclose(self.sigma, other.sigma):
            raise ValueError("RKHS mismatch: sigma values must match")
        if ridge < 0:
            raise ValueError("ridge must be non-negative")

        dtype = np.result_type(self.coefs, other.coefs, float)
        a = np.asarray(self.coefs, dtype=dtype)
        b = np.asarray(other.coefs, dtype=dtype)

        k_xx = self.gram().astype(dtype, copy=False)
        k_cc = other.gram().astype(dtype, copy=False)
        if ridge:
            k_xx = k_xx + ridge * np.eye(k_xx.shape[0], dtype=dtype)
            k_cc = k_cc + ridge * np.eye(k_cc.shape[0], dtype=dtype)

        diff = self.scalars[:, None] - other.scalars[None, :]
        k_xc = _gaussian_kernel_from_diff(diff, self.sigma).astype(dtype, copy=False)
        norm_sq = a @ k_xx @ a + b @ k_cc @ b - 2.0 * a @ k_xc @ b
        return float(np.sqrt(max(float(norm_sq), 0.0)))


def save_rkhs(phi: GaussianRKHS, path: str | Path) -> Path:
    """Persist a Gaussian RKHS function as a compressed ``.npz`` file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        sigma=np.array([float(phi.sigma)], dtype=float),
        scalars=np.asarray(phi.scalars),
        coefs=np.asarray(phi.coefs),
    )
    return target


def load_rkhs(path: str | Path) -> GaussianRKHS:
    """Load a Gaussian RKHS function saved with :func:`save_rkhs`."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"No RKHS file found at {source}")

    with np.load(source, allow_pickle=False) as data:
        try:
            scalars = np.asarray(data["scalars"])
            coefs = np.asarray(data["coefs"])
            sigma = float(np.asarray(data["sigma"]).squeeze())
        except KeyError as exc:
            raise ValueError(f"Missing key {exc.args[0]} in RKHS file {source}") from exc

    if scalars.shape[0] != coefs.shape[0]:
        raise ValueError(
            f"scalars and coefs must align in {source} (got {scalars.shape} vs {coefs.shape})"
        )
    return GaussianRKHS(sigma=sigma, scalars=scalars, coefs=coefs)


def build_gaussian_rkhs(
    config: ModelBuildContext,
    params: dict[str, Any] | None = None,
    dataset: DatasetBundle | None = None,
    z: np.ndarray | None = None,
) -> GaussianRKHS:
    dtype = config.runtime.np_dtype
    _ = params
    rkhs_config = GaussianRKHSConfig.from_link_config(config.link)

    return GaussianRKHS(
        config.link.sigma,
        np.array([0.0], dtype=dtype),
        np.array([0.0], dtype=dtype),
        config=rkhs_config,
    )
