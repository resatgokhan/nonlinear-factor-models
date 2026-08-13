"""Built-in generating links used by synthetic research datasets."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, Mapping

import numpy as np

from bcd_lab.function_models import (
    ExponentialFunction,
    IdentityFunction,
    ScalarFunction,
    SigmoidFunction,
    TanhFunction,
)
from bcd_lab.function_models.rkhs import polynomial_fit_phi, random_monotone_poly

from .base import GeneratingLinkContext


def _reject_unknown(params: Mapping[str, Any], *, allowed: set[str], name: str) -> None:
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"{name} received unknown parameters: {', '.join(unknown)}")


def _bool(params: Mapping[str, Any], name: str, default: bool) -> bool:
    value = params.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _real(
    params: Mapping[str, Any],
    name: str,
    default: float,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    value = params.get(name, default)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if non_negative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _integer(
    params: Mapping[str, Any],
    name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    value = params.get(name, default)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


class _IdentityBuilder:
    def validate_params(self, params: Mapping[str, Any]) -> None:
        _reject_unknown(params, allowed=set(), name="identity")

    def build(
        self,
        params: Mapping[str, Any],
        context: GeneratingLinkContext,
    ) -> IdentityFunction:
        del params, context
        return IdentityFunction()


class _SigmoidBuilder:
    def validate_params(self, params: Mapping[str, Any]) -> None:
        _reject_unknown(params, allowed={"shifted"}, name="sigmoid")
        _bool(params, "shifted", True)

    def build(
        self,
        params: Mapping[str, Any],
        context: GeneratingLinkContext,
    ) -> SigmoidFunction:
        del context
        return SigmoidFunction(shifted=_bool(params, "shifted", True))


class _ExponentialBuilder:
    def validate_params(self, params: Mapping[str, Any]) -> None:
        _reject_unknown(params, allowed={"shifted"}, name="exponential")
        _bool(params, "shifted", False)

    def build(
        self,
        params: Mapping[str, Any],
        context: GeneratingLinkContext,
    ) -> ExponentialFunction:
        del context
        return ExponentialFunction(shifted=_bool(params, "shifted", False))


class _TanhBuilder:
    def validate_params(self, params: Mapping[str, Any]) -> None:
        _reject_unknown(params, allowed=set(), name="tanh")

    def build(
        self,
        params: Mapping[str, Any],
        context: GeneratingLinkContext,
    ) -> TanhFunction:
        del params, context
        return TanhFunction()


_RANDOM_MONOTONE_PARAMS = {
    "degree",
    "fit_ridge",
    "max_attempts",
    "min_margin",
    "min_range",
    "num_centers",
    "polynomial_grid_count",
    "sigma",
    "waviness",
    "xmax",
    "xmin",
}


def _random_monotone_settings(params: Mapping[str, Any]) -> dict[str, Any]:
    _reject_unknown(
        params,
        allowed=_RANDOM_MONOTONE_PARAMS,
        name="random_monotone_rkhs",
    )
    settings: dict[str, Any] = {
        "degree": _integer(params, "degree", 4, minimum=1),
        "fit_ridge": _real(params, "fit_ridge", 1.0e-6, non_negative=True),
        "max_attempts": _integer(params, "max_attempts", 10_000, minimum=1),
        "min_margin": _real(params, "min_margin", 1.0e-2, positive=True),
        "min_range": _real(params, "min_range", 0.5, non_negative=True),
        "polynomial_grid_count": _integer(
            params,
            "polynomial_grid_count",
            400,
            minimum=2,
        ),
        "sigma": _real(params, "sigma", 1.0, positive=True),
        "waviness": _real(params, "waviness", 1.0, positive=True),
        "xmin": _real(params, "xmin", -3.0),
        "xmax": _real(params, "xmax", 3.0),
    }
    if settings["xmin"] >= settings["xmax"]:
        raise ValueError("xmin must be smaller than xmax")
    num_centers = params.get("num_centers")
    if num_centers is not None:
        if isinstance(num_centers, bool) or not isinstance(num_centers, Integral):
            raise TypeError("num_centers must be an integer or null")
        if int(num_centers) <= 0:
            raise ValueError("num_centers must be positive")
        settings["num_centers"] = int(num_centers)
    else:
        settings["num_centers"] = None
    return settings


class _RandomMonotoneRkhsBuilder:
    def validate_params(self, params: Mapping[str, Any]) -> None:
        _random_monotone_settings(params)

    def build(
        self,
        params: Mapping[str, Any],
        context: GeneratingLinkContext,
    ) -> ScalarFunction:
        settings = _random_monotone_settings(params)
        available = np.asarray(context.latent_samples, dtype=context.dtype).ravel()
        if available.size == 0 or not np.all(np.isfinite(available)):
            raise ValueError("latent_samples must be non-empty and finite")
        child_seeds = context.rng.integers(0, 2**32, size=2, dtype=np.uint32)
        center_rng = np.random.default_rng(int(child_seeds[0]))
        polynomial_rng = np.random.default_rng(int(child_seeds[1]))
        count = settings["num_centers"]
        if count is None:
            centers = available.copy()
        else:
            if count > available.size:
                raise ValueError("num_centers cannot exceed the number of latent samples")
            indices = center_rng.choice(available.size, size=count, replace=False)
            centers = available[indices]

        polynomial = random_monotone_poly(
            settings["degree"],
            grid=settings["polynomial_grid_count"],
            xmin=settings["xmin"],
            xmax=settings["xmax"],
            waviness=settings["waviness"],
            min_margin=settings["min_margin"],
            min_range=settings["min_range"],
            max_attempts=settings["max_attempts"],
            rng=polynomial_rng,
        )
        link = polynomial_fit_phi(
            centers,
            settings["sigma"],
            polynomial,
            ridge=settings["fit_ridge"],
        ).copy(dtype=context.dtype)
        fit_grid = np.linspace(
            float(np.min(centers)),
            float(np.max(centers)),
            settings["polynomial_grid_count"],
        )
        fit_values = link.eval_array(fit_grid)
        if not np.all(np.isfinite(fit_values)) or not np.all(np.diff(fit_values) > 0.0):
            raise ValueError(
                "Gaussian RKHS approximation is not strictly increasing over its fit domain; "
                "increase num_centers or reduce fit_ridge"
            )
        return link


IDENTITY_BUILDER = _IdentityBuilder()
SIGMOID_BUILDER = _SigmoidBuilder()
EXPONENTIAL_BUILDER = _ExponentialBuilder()
TANH_BUILDER = _TanhBuilder()
RANDOM_MONOTONE_RKHS_BUILDER = _RandomMonotoneRkhsBuilder()
