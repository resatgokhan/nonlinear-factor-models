"""Piecewise-linear RKHS function models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .base import FunctionConstraintConfig
from .distances import function_distance
from .rkhs.gaussianKernelMath import gaussian_kernel_matrix

if TYPE_CHECKING:
    from bcd_lab.config import ModelBuildContext
    from bcd_lab.types import DatasetBundle


@dataclass(frozen=True)
class FixedGridBounds:
    """Explicit interpolation bounds that do not depend on model inputs."""

    lower: float = -3.0
    upper: float = 3.0


@dataclass(frozen=True)
class InputQuantileGridBounds:
    """Interpolation bounds derived from finite model inputs."""

    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    pad_fraction: float = 0.25
    min_padding: float = 0.5


GridBounds = FixedGridBounds | InputQuantileGridBounds


@dataclass(frozen=True)
class GridRebuildPolicy:
    """Control when input-derived interpolation bounds are recomputed."""

    name: Literal["never", "adaptive", "every_iteration"] = "never"
    phi_tolerance: float = 1.0e-3
    outside_margin_cells: float = 1.0
    min_active_fraction: float = 0.05


@dataclass(frozen=True)
class PiecewiseLinearRKHSConfig(FunctionConstraintConfig):
    """Configuration for constructing piecewise-linear RKHS models."""

    sigma: float = 1.0
    count: int = 1000
    bounds: GridBounds = FixedGridBounds()
    rebuild: GridRebuildPolicy = GridRebuildPolicy()

    @classmethod
    def from_link_config(
        cls,
        link_config: Any,
    ) -> PiecewiseLinearRKHSConfig:
        link_params = link_config.params
        defaults = cls(sigma=link_config.sigma)

        def float_param(name: str, default: float | None) -> float | None:
            value = link_params.get(name, default)
            return float(value) if value is not None else None

        allowed_params = {
            "grid",
            "phi_0",
            "monotone_phi_enabled",
            "monotone_phi_direction",
            "phi_output_clamp_enabled",
            "phi_output_clamp_min",
            "phi_output_clamp_max",
            "ball_enabled",
            "ball_radius",
        }
        unknown_params = sorted(set(link_params) - allowed_params)
        if unknown_params:
            raise ValueError(
                "unknown piecewise_linear_rkhs parameter(s): " + ", ".join(unknown_params)
            )

        grid = link_params.get("grid", {})
        if not isinstance(grid, Mapping):
            raise ValueError("link.params.grid must be a mapping")
        unknown_grid = sorted(set(grid) - {"count", "bounds", "rebuild"})
        if unknown_grid:
            raise ValueError("unknown grid field(s): " + ", ".join(unknown_grid))

        bounds_spec = grid.get("bounds", {"name": "fixed", "params": {}})
        bounds = _parse_grid_bounds(bounds_spec)
        rebuild = _parse_grid_rebuild(grid.get("rebuild", {"name": "never", "params": {}}))
        if isinstance(bounds, FixedGridBounds) and rebuild.name != "never":
            raise ValueError("fixed grid bounds require rebuild.name: never")

        phi_0 = link_params.get("phi_0", defaults.ball_center_phi_0)
        grid_count = int(grid.get("count", defaults.count))
        if grid_count < 2:
            raise ValueError("grid.count must be at least 2")
        return cls(
            sigma=link_config.sigma,
            count=grid_count,
            bounds=bounds,
            rebuild=rebuild,
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


def validate_piecewise_linear_config(link_config: Any) -> None:
    """Validate canonical piecewise-linear parameters without building model state."""
    PiecewiseLinearRKHSConfig.from_link_config(link_config)


def _mapping_spec(value: Any, *, field: str) -> tuple[str, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    unknown = sorted(set(value) - {"name", "params"})
    if unknown:
        raise ValueError(f"unknown {field} field(s): " + ", ".join(unknown))
    name = value.get("name")
    params = value.get("params", {})
    if not isinstance(name, str) or not name:
        raise ValueError(f"{field}.name must be a non-empty string")
    if not isinstance(params, Mapping):
        raise ValueError(f"{field}.params must be a mapping")
    return name, params


def _parse_grid_bounds(value: Any) -> GridBounds:
    name, params = _mapping_spec(value, field="link.params.grid.bounds")
    if name == "fixed":
        unknown = sorted(set(params) - {"lower", "upper"})
        if unknown:
            raise ValueError("unknown fixed grid bound parameter(s): " + ", ".join(unknown))
        bounds = FixedGridBounds(
            lower=float(params.get("lower", -3.0)),
            upper=float(params.get("upper", 3.0)),
        )
        if not np.isfinite(bounds.lower) or not np.isfinite(bounds.upper):
            raise ValueError("fixed grid bounds must be finite")
        if bounds.lower >= bounds.upper:
            raise ValueError("fixed grid lower bound must be below its upper bound")
        return bounds
    if name == "input_quantiles":
        allowed = {"lower_quantile", "upper_quantile", "pad_fraction", "min_padding"}
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise ValueError("unknown input-quantile parameter(s): " + ", ".join(unknown))
        bounds = InputQuantileGridBounds(
            lower_quantile=float(params.get("lower_quantile", 0.01)),
            upper_quantile=float(params.get("upper_quantile", 0.99)),
            pad_fraction=float(params.get("pad_fraction", 0.25)),
            min_padding=float(params.get("min_padding", 0.5)),
        )
        values = (
            bounds.lower_quantile,
            bounds.upper_quantile,
            bounds.pad_fraction,
            bounds.min_padding,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("input-quantile grid parameters must be finite")
        if not 0.0 <= bounds.lower_quantile < bounds.upper_quantile <= 1.0:
            raise ValueError("grid quantiles must satisfy 0 <= lower < upper <= 1")
        if bounds.pad_fraction < 0.0 or bounds.min_padding < 0.0:
            raise ValueError("grid padding parameters must be non-negative")
        return bounds
    raise ValueError(f"unknown grid bounds policy {name!r}")


def _parse_grid_rebuild(value: Any) -> GridRebuildPolicy:
    name, params = _mapping_spec(value, field="link.params.grid.rebuild")
    if name not in {"never", "adaptive", "every_iteration"}:
        raise ValueError(f"unknown grid rebuild policy {name!r}")
    allowed = (
        {"phi_tolerance", "outside_margin_cells", "min_active_fraction"}
        if name == "adaptive"
        else set()
    )
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} rebuild parameter(s): " + ", ".join(unknown))
    policy = GridRebuildPolicy(
        name=name,  # type: ignore[arg-type]
        phi_tolerance=float(params.get("phi_tolerance", 1.0e-3)),
        outside_margin_cells=float(params.get("outside_margin_cells", 1.0)),
        min_active_fraction=float(params.get("min_active_fraction", 0.05)),
    )
    if not all(
        np.isfinite(value)
        for value in (
            policy.phi_tolerance,
            policy.outside_margin_cells,
            policy.min_active_fraction,
        )
    ):
        raise ValueError("adaptive rebuild parameters must be finite")
    if policy.phi_tolerance < 0.0 or policy.outside_margin_cells < 0.0:
        raise ValueError("adaptive rebuild tolerances must be non-negative")
    if not 0.0 < policy.min_active_fraction <= 1.0:
        raise ValueError("adaptive min_active_fraction must be in (0, 1]")
    return policy


class PiecewiseLinearRKHS:
    """SpeedPhi-compatible piecewise-linear function on a fixed sorted grid."""

    kind = "piecewise_linear_rkhs"
    checkpoint_version = 2

    def __init__(
        self,
        sigma: float,
        grid_values: tuple[np.ndarray, np.ndarray] | None = None,
        *,
        config: PiecewiseLinearRKHSConfig | None = None,
        dtype: np.dtype | type | str = np.float64,
    ) -> None:
        self.sigma = float(sigma)

        if grid_values is None:
            self._config = (
                config
                if config is not None
                else PiecewiseLinearRKHSConfig(
                    sigma=self.sigma,
                )
            )

            if isinstance(self._config.bounds, InputQuantileGridBounds):
                self.scalars = None
                self.coefs = None
                return

            self.build_fixed_grid(dtype)
            return

        grid, values = grid_values

        grid_arr = np.asarray(grid, dtype=dtype)
        values_arr = np.asarray(values, dtype=dtype)

        if grid_arr.ndim != 1:
            grid_arr = grid_arr.ravel()
        if values_arr.ndim != 1:
            values_arr = values_arr.ravel()
        if grid_arr.size != values_arr.size:
            raise ValueError("grid and values must have the same length")
        if grid_arr.size < 2:
            raise ValueError("grid must contain at least two points")
        if not np.all(np.diff(grid_arr) > 0):
            raise ValueError("grid must be strictly increasing")

        self.scalars = grid_arr
        self.coefs = values_arr

        self._config = (
            config
            if config is not None
            else PiecewiseLinearRKHSConfig(
                sigma=self.sigma,
                count=int(grid_arr.size),
                bounds=FixedGridBounds(float(grid_arr[0]), float(grid_arr[-1])),
            )
        )

    @property
    def grid(self) -> np.ndarray:
        """Alias for the fixed interpolation grid."""
        return self.scalars

    @property
    def values(self) -> np.ndarray:
        """Alias for grid values."""
        return self.coefs

    @property
    def config(self) -> PiecewiseLinearRKHSConfig | None:
        """Construction policy attached by the factory, if available."""
        return self._config

    def _slopes(self) -> tuple[np.ndarray, float]:
        """Return SpeedPhi-style per-interval slopes and uniform grid spacing."""
        dx = float(self.scalars[1] - self.scalars[0])
        slopes = (self.coefs[1:] - self.coefs[:-1]) / dx
        return slopes, dx

    def eval_array(self, x: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(x, dtype=self.coefs.dtype), self.scalars, self.coefs)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to the given input-output data pairs by interpolating onto the grid."""
        self.prepare_grid(x)
        self.coefs[:] = np.interp(self.scalars, x, y).astype(self.coefs.dtype, copy=False)

    def d_array(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=self.coefs.dtype)
        slopes, _ = self._slopes()
        idx = np.searchsorted(self.scalars, x_arr, side="right") - 1
        idx = np.clip(idx, 0, slopes.size - 1)
        return slopes[idx]

    def eval_and_d_array(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.eval_array(x), self.d_array(x)

    def __call__(self, x: float) -> float:
        return float(self.eval_array(np.asarray([x]))[0])

    def d(self, x: float) -> float:
        return float(self.d_array(np.asarray([x]))[0])

    def copy(self) -> "PiecewiseLinearRKHS":
        copied = PiecewiseLinearRKHS(
            self.sigma,
            (
                np.array(self.scalars, copy=True),
                np.array(self.coefs, copy=True),
            ),
            config=self._config,
        )
        if hasattr(self, "_resolved_input_bounds"):
            copied._resolved_input_bounds = self._resolved_input_bounds
        return copied

    def scale_coefs(self, scale: float) -> None:
        """Scale grid values in place."""
        self.coefs *= scale

    def scale_parameters(self, scale: float) -> None:
        self.scale_coefs(scale)

    def prepare_update(self, inputs: np.ndarray) -> None:
        """Regrid before the solver evaluates residuals on the current coordinates."""
        self.prepare_grid(inputs)

    def gradient(
        self,
        *,
        inputs: np.ndarray,
        residuals: np.ndarray,
    ) -> np.ndarray:
        """Return the SpeedPhi Gaussian-kernel grid update direction."""
        inputs_arr = np.asarray(inputs).ravel()

        self.prepare_grid(inputs_arr)

        x = np.asarray(inputs, dtype=self.scalars.dtype).ravel()
        residual_arr = np.asarray(residuals, dtype=self.coefs.dtype).ravel()
        if x.size != residual_arr.size:
            raise ValueError("inputs and residuals must have the same length")
        kernel = gaussian_kernel_matrix(self.scalars, x, self.sigma)
        return kernel @ residual_arr

    def apply_gradient(self, gradient: np.ndarray, *, step_size: float) -> None:
        grad = np.asarray(gradient, dtype=self.coefs.dtype)
        if grad.shape != self.coefs.shape:
            raise ValueError(f"gradient must have shape {self.coefs.shape}, got {grad.shape}")
        self.coefs += float(step_size) * grad

    def project(
        self,
        *,
        centers: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        l2_grid: np.ndarray | None = None,
        phi_ball_radius: float | None = None,
    ) -> None:
        """Apply in-place constraints based on the model's configuration."""
        _ = centers, rng
        cfg = self.config

        # 1. Monotonicity constraints
        monotone_enabled = getattr(cfg, "monotone_enabled", False)
        if monotone_enabled:
            direction = getattr(cfg, "monotone_direction", "increasing")
            if direction == "increasing":
                self.coefs[:] = _isotonic_increasing(self.coefs).astype(
                    self.coefs.dtype, copy=False
                )
            elif direction == "decreasing":
                self.coefs[:] = (-_isotonic_increasing(-self.coefs)).astype(
                    self.coefs.dtype, copy=False
                )

        # 2. Output clamping
        clamp_enabled = getattr(cfg, "clamp_enabled", False)
        if clamp_enabled:
            lower = getattr(cfg, "clamp_min", None)
            upper = getattr(cfg, "clamp_max", None)
            if lower is None or upper is None:
                raise ValueError("clamp constraint requires min and max")
            np.clip(self.coefs, float(lower), float(upper), out=self.coefs)

        # 3. L2 ball projection around phi_0
        ball_enabled = getattr(cfg, "ball_enabled", False)
        center = getattr(cfg, "ball_center_phi_0", None)
        radius = (
            phi_ball_radius if phi_ball_radius is not None else getattr(cfg, "ball_radius", None)
        )

        if ball_enabled and center is not None and radius is not None:
            self.project_l2_ball(center=center, radius=float(radius), grid=l2_grid)

    def resolve_projection_ball_radius(
        self,
        *,
        truth: object | None,
    ) -> float | None:
        """Resolve this model's configured projection radius without exposing its grid."""
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
        return function_distance(center, truth, grid=self.scalars)

    def project_l2_ball(
        self,
        *,
        center: object,
        radius: float,
        grid: np.ndarray | None = None,
    ) -> None:
        """Project onto a SpeedPhi-style L2 ball around another function."""
        if radius < 0:
            raise ValueError("ball radius must be non-negative")
        dist = self.diff_norm(center) if grid is None else l2_grid_diff(self, center, grid)
        if dist <= radius or dist == 0.0:
            return
        center_vals = np.asarray(center.eval_array(self.scalars), dtype=self.coefs.dtype)  # type: ignore[attr-defined]
        self.coefs[:] = center_vals + (self.coefs - center_vals) * (radius / dist)

    def distance_to(
        self,
        other: object,
        *,
        mode: str = "auto",
        grid: np.ndarray | None = None,
    ) -> float:
        if mode in {"auto", "l2_grid"}:
            return self.diff_norm(other, grid=self.scalars if grid is None else grid)
        return function_distance(
            self, other, mode=mode, grid=self.scalars if grid is None else grid
        )

    def norm(self, *, mode: str = "auto") -> float | None:
        if mode in {"auto", "rkhs", "speed"}:
            return self.rkhs_norm()
        if mode not in {"l2_grid"}:
            return None
        dx = np.diff(self.grid)
        mid_vals = 0.5 * (self.coefs[:-1] ** 2 + self.coefs[1:] ** 2)
        return float(np.sqrt(np.sum(mid_vals * dx)))

    def rkhs_norm(self, ridge: float = 0.0) -> float:
        """Compute the SpeedPhi grid proxy norm, not a Gaussian RKHS norm."""
        _ = ridge
        slopes, dx = self._slopes()
        return float(np.sqrt(np.sum(self.coefs**2) * dx + np.sum(slopes**2) * dx))

    def diff_norm(
        self,
        other: object,
        ridge: float = 0.0,
        *,
        grid: np.ndarray | None = None,
    ) -> float:
        """Compute L2 distance to another function on this grid by default."""
        _ = ridge
        eval_grid = self.scalars if grid is None else np.asarray(grid)
        return l2_grid_diff(self, other, eval_grid)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "sigma": float(self.sigma),
            "grid": np.asarray(self.scalars, dtype=float),
            "values": np.asarray(self.coefs, dtype=float),
        }

    def checkpoint_state(self) -> Mapping[str, Any]:
        return {
            "scalars": None if self.scalars is None else self.scalars.copy(),
            "coefs": None if self.coefs is None else self.coefs.copy(),
            "resolved_input_bounds": getattr(self, "_resolved_input_bounds", None),
        }

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        version: int,
    ) -> None:
        if version != self.checkpoint_version:
            raise ValueError(f"unsupported piecewise-linear state version {version}")
        arrays = _checkpoint_arrays(
            state,
            allow_absent=isinstance(self._config.bounds, InputQuantileGridBounds),
        )
        if arrays is None:
            self.scalars = None
            self.coefs = None
        else:
            self.scalars, self.coefs = arrays
        resolved_bounds = _checkpoint_input_bounds(state.get("resolved_input_bounds"))
        if resolved_bounds is None:
            if hasattr(self, "_resolved_input_bounds"):
                del self._resolved_input_bounds
        else:
            self._resolved_input_bounds = resolved_bounds

    def prepare_grid(self, inputs: np.ndarray) -> None:
        """Initialize or rebuild an input-derived grid according to its policy."""
        cfg = self._config
        bounds_policy = cfg.bounds
        if isinstance(bounds_policy, FixedGridBounds):
            return

        dtype = self.coefs.dtype if self.coefs is not None else np.float64
        target_dtype = np.dtype(dtype)

        finite_inputs = np.asarray(inputs, dtype=np.float64).ravel()
        finite_inputs = finite_inputs[np.isfinite(finite_inputs)]

        if finite_inputs.size == 0:
            raise ValueError("input-quantile grid requires at least one finite input")

        lo, hi = np.quantile(
            finite_inputs,
            [bounds_policy.lower_quantile, bounds_policy.upper_quantile],
        )
        lo = float(lo)
        hi = float(hi)

        if self.scalars is not None and self.coefs is not None:
            if cfg.rebuild.name == "never":
                return
            if cfg.rebuild.name == "adaptive" and not self._adaptive_rebuild_needed(lo, hi):
                return

        lower, upper = _padded_grid_bounds(lo, hi, bounds_policy)
        new_grid = np.linspace(lower, upper, cfg.count, dtype=target_dtype)

        # The solver prepares before residual evaluation; gradient() also supports
        # standalone calls. Avoid applying interpolation twice to the same grid.
        if self.scalars is not None and np.array_equal(self.scalars, new_grid):
            self._resolved_input_bounds = (lo, hi)
            return

        if self.scalars is None or self.coefs is None:
            new_values = np.zeros_like(new_grid, dtype=target_dtype)
        else:
            new_values = np.interp(new_grid, self.scalars, self.coefs).astype(
                target_dtype,
                copy=False,
            )

        self.scalars = new_grid
        self.coefs = new_values
        self._resolved_input_bounds = (lo, hi)

    def _adaptive_rebuild_needed(self, lo: float, hi: float) -> bool:
        cfg = self._config
        rebuild = cfg.rebuild
        assert self.scalars is not None
        assert self.coefs is not None

        grid_lo = float(self.scalars[0])
        grid_hi = float(self.scalars[-1])
        dx = float(self.scalars[1] - self.scalars[0])
        outside = max(grid_lo - lo, hi - grid_hi, 0.0)
        if outside > rebuild.outside_margin_cells * dx:
            return True

        left_idx = np.searchsorted(self.scalars, lo, side="left")
        right_idx = np.searchsorted(self.scalars, hi, side="right")
        active_points = int(right_idx - left_idx)
        min_active = max(5, int(rebuild.min_active_fraction * cfg.count))
        if hi > lo and active_points < min(min_active, cfg.count):
            return True

        previous_bounds = getattr(self, "_resolved_input_bounds", None)
        if previous_bounds is None:
            return False
        prev_lo, prev_hi = previous_bounds
        drift = max(abs(lo - prev_lo), abs(hi - prev_hi))
        if drift == 0.0:
            return False

        span_lo = max(grid_lo, min(lo, prev_lo))
        span_hi = min(grid_hi, max(hi, prev_hi))
        slopes, _ = self._slopes()
        if span_lo <= span_hi:
            slope_left = np.searchsorted(self.scalars, span_lo, side="right") - 1
            slope_right = np.searchsorted(self.scalars, span_hi, side="left")
            slope_left = int(np.clip(slope_left, 0, slopes.size - 1))
            slope_right = int(np.clip(slope_right, slope_left, slopes.size - 1))
            max_slope = float(np.max(np.abs(slopes[slope_left : slope_right + 1])))
        else:
            max_slope = 0.0
        return drift * max_slope > rebuild.phi_tolerance

    def build_fixed_grid(
        self,
        dtype: np.dtype | type | str,
        *,
        count: int | None = None,
    ) -> None:
        """Initialize this model on the configured fixed interpolation grid."""
        cfg = self._config
        bounds = cfg.bounds
        if not isinstance(bounds, FixedGridBounds):
            raise RuntimeError("build_fixed_grid requires fixed grid bounds")
        grid_count = cfg.count if count is None else int(count)

        if grid_count < 2:
            raise ValueError("grid count must be at least 2")

        target_dtype = np.dtype(dtype)
        grid = np.linspace(bounds.lower, bounds.upper, grid_count, dtype=target_dtype)

        self.scalars = grid
        self.coefs = np.zeros_like(grid, dtype=target_dtype)


def _padded_grid_bounds(
    lo: float,
    hi: float,
    policy: InputQuantileGridBounds,
) -> tuple[float, float]:
    if lo == hi:
        half_width = max(abs(lo), policy.min_padding)
        lower = lo - half_width
        upper = lo + half_width
    else:
        pad = max(policy.pad_fraction * (hi - lo), policy.min_padding)
        lower = lo - pad
        upper = hi + pad
    if lower >= upper:
        raise ValueError(f"failed to build grid from finite input range [{lo}, {hi}]")
    return lower, upper


def _checkpoint_arrays(
    state: Mapping[str, Any],
    *,
    allow_absent: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    scalars_state = state["scalars"]
    coefs_state = state["coefs"]
    if scalars_state is None or coefs_state is None:
        if scalars_state is not None or coefs_state is not None:
            raise ValueError("piecewise-linear grid and values must both be absent")
        if not allow_absent:
            raise ValueError("only input-derived grid bounds may have absent arrays")
        return None
    scalars = np.asarray(scalars_state)
    coefs = np.asarray(coefs_state)
    if scalars.ndim != 1 or coefs.ndim != 1 or scalars.shape != coefs.shape:
        raise ValueError("piecewise-linear checkpoint arrays must be aligned vectors")
    if scalars.size < 2 or not np.all(np.diff(scalars) > 0):
        raise ValueError("piecewise-linear checkpoint grid must be strictly increasing")
    if not np.all(np.isfinite(scalars)) or not np.all(np.isfinite(coefs)):
        raise ValueError("piecewise-linear checkpoint arrays must be finite")
    return scalars.copy(), coefs.copy()


def _checkpoint_input_bounds(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    bounds = np.asarray(value, dtype=float).ravel()
    if bounds.size != 2 or not np.all(np.isfinite(bounds)):
        raise ValueError("resolved input bounds must contain two finite values")
    if bounds[0] > bounds[1]:
        raise ValueError("resolved input bounds must be ordered")
    return float(bounds[0]), float(bounds[1])


def _isotonic_increasing(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size <= 1:
        return arr.copy()

    x = np.arange(arr.size, dtype=np.float64)
    return np.asarray(
        IsotonicRegression(increasing=True).fit_transform(x, arr),
        dtype=np.float64,
    )


def l2_grid_diff(phi_a: object, phi_b: object, grid: np.ndarray) -> float:
    """Compute the L2 distance between two function models on a fixed grid."""
    grid_arr = np.asarray(grid)

    if grid_arr.ndim != 1:
        grid_arr = grid_arr.ravel()
    if grid_arr.size < 2:
        raise ValueError("grid must contain at least two points")

    vals_a = np.asarray(phi_a.eval_array(grid_arr))  # type: ignore[attr-defined]
    vals_b = np.asarray(phi_b.eval_array(grid_arr), dtype=vals_a.dtype)  # type: ignore[attr-defined]

    diff = vals_a - vals_b
    dx = float(grid_arr[1] - grid_arr[0])

    return float(np.sqrt(np.sum(diff**2) * dx))


def build_piecewise_linear_model(
    model_cls: type[PiecewiseLinearRKHS],
    config: ModelBuildContext,
    params: dict[str, Any] | None = None,
    dataset: DatasetBundle | None = None,
    z: np.ndarray | None = None,
) -> PiecewiseLinearRKHS:
    dtype = config.runtime.np_dtype
    _ = params

    pl_config = PiecewiseLinearRKHSConfig.from_link_config(
        config.link,
    )

    phi = model_cls(
        sigma=config.link.sigma,
        config=pl_config,
        dtype=dtype,
    )

    if isinstance(pl_config.bounds, InputQuantileGridBounds):
        if z is None:
            raise ValueError("input-quantile grid bounds require initialized factors")
        from bcd_lab.math import compute_centers

        if dataset is not None:
            centers = compute_centers(
                z,
                dataset.train[1][0],
                dataset.train[1][1],
                config.shape.n_rows,
            )
        else:
            centers = compute_centers(
                z, config.shape.n_rows, config.shape.n_cols, config.shape.n_rows
            )
        phi.prepare_grid(centers)

    phi_0 = config.link.params.get("phi_0")
    if phi_0 is not None and hasattr(phi_0, "eval_array"):
        phi.coefs[:] = phi_0.eval_array(phi.scalars).astype(dtype, copy=False)

    return phi


def build_piecewise_linear_rkhs(
    config: ModelBuildContext,
    params: dict[str, Any] | None = None,
    dataset: DatasetBundle | None = None,
    z: np.ndarray | None = None,
) -> PiecewiseLinearRKHS:
    return build_piecewise_linear_model(PiecewiseLinearRKHS, config, params, dataset, z)
