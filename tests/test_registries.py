"""Boundary tests for the public component registries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from bcd_lab.algorithms import list_algorithms
from bcd_lab.data import (
    GeneratingLinkContext,
    get_data_loader,
    get_generating_link_builder,
    list_data_loaders,
    list_generating_links,
    register_data_loader,
    register_generating_link,
)
from bcd_lab.data.preprocessors import list_preprocessors
from bcd_lab.diagnostics import list_diagnostic_families
from bcd_lab.function_models import (
    ScalarFunction,
    get_function_model_builder,
    list_function_models,
    register_function_model,
)
from bcd_lab.initializers import (
    list_joint_initializers,
    list_phi_initializers,
    list_z_initializers,
)
from bcd_lab.losses import list_objectives
from bcd_lab.optimizers import (
    get_optimizer_builder,
    list_optimizers,
    register_optimizer,
)
from bcd_lab.plotting import list_plot_types
from bcd_lab.types import DatasetBundle


@dataclass
class _DuplicateLoader:
    name: str

    def load(self, *, params: dict[str, Any], seed: int) -> DatasetBundle:
        raise AssertionError("a rejected duplicate must never be called")


class _DuplicateGeneratingLinkBuilder:
    def validate_params(self, params) -> None:
        del params

    def build(self, params, context: GeneratingLinkContext) -> ScalarFunction:
        del params, context
        raise AssertionError("a rejected duplicate must never be called")


def _duplicate_function_builder(
    config: Any,
    params: dict[str, Any] | None = None,
    dataset: DatasetBundle | None = None,
    z: np.ndarray | None = None,
) -> ScalarFunction:
    raise AssertionError("a rejected duplicate must never be called")


def test_builtin_discovery_is_complete_and_deterministic() -> None:
    """Every built-in category is available without hidden lazy imports."""
    discovered = {
        "algorithms": list_algorithms(),
        "data": list_data_loaders(),
        "preprocessors": list_preprocessors(),
        "diagnostics": list_diagnostic_families(),
        "function_models": list_function_models(),
        "generating_links": list_generating_links(),
        "joint_initializers": list_joint_initializers(),
        "phi_initializers": list_phi_initializers(),
        "z_initializers": list_z_initializers(),
        "objectives": list_objectives(),
        "optimizers": list_optimizers(),
        "plots": list_plot_types(),
    }

    assert all(names == tuple(sorted(names)) for names in discovered.values())
    assert {"bcd"}.issubset(discovered["algorithms"])
    assert {"csv", "synthetic"}.issubset(discovered["data"])
    assert {"ml-100k", "ml-1m", "ml-10m", "ml-20m"}.issubset(discovered["preprocessors"])
    assert {
        "data_summary",
        "gradients",
        "objective",
        "phi",
        "phi_sharp",
        "reconstruction",
        "timing",
        "truth_matrix",
        "truth_z",
    }.issubset(discovered["diagnostics"])
    assert {
        "binned_piecewise_linear_rkhs",
        "exponential",
        "gaussian_rkhs",
        "identity",
        "piecewise_linear_rkhs",
        "sigmoid",
    }.issubset(discovered["function_models"])
    assert {
        "exponential",
        "identity",
        "random_monotone_rkhs",
        "sigmoid",
        "tanh",
    }.issubset(discovered["generating_links"])
    assert {"svd_with_zero_phi"}.issubset(discovered["joint_initializers"])
    assert {"identity_phi", "none", "zero"}.issubset(discovered["phi_initializers"])
    assert {"svd"}.issubset(discovered["z_initializers"])
    assert {"half_squared", "squared_error"}.issubset(discovered["objectives"])
    assert {"adam", "gradient_descent"}.issubset(discovered["optimizers"])
    assert {
        "heatmap",
        "image",
        "iteration_table",
        "line",
        "phi_circle",
        "summary_line",
        "table",
    }.issubset(discovered["plots"])


def test_duplicate_registration_is_rejected_without_replacing_components() -> None:
    """Object and decorator-style public APIs share duplicate-safe behavior."""
    original_loader = get_data_loader("synthetic")
    with pytest.raises(ValueError, match="already registered"):
        register_data_loader(_DuplicateLoader("synthetic"))
    assert get_data_loader("synthetic") is original_loader

    original_generating_builder = get_generating_link_builder("identity")
    with pytest.raises(ValueError, match="already registered"):
        register_generating_link("identity", _DuplicateGeneratingLinkBuilder())
    assert get_generating_link_builder("identity") is original_generating_builder

    original_builder = get_function_model_builder("identity")
    with pytest.raises(ValueError, match="already registered"):
        register_function_model("identity")(_duplicate_function_builder)
    assert get_function_model_builder("identity") is original_builder

    original_optimizer = get_optimizer_builder("adam")
    with pytest.raises(ValueError, match="already registered"):
        register_optimizer("adam", original_optimizer)
    assert get_optimizer_builder("adam") is original_optimizer
