"""Finance adapter for registered solvers and explicit component seeds."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from bcd_lab.algorithms import get_algorithm
from bcd_lab.algorithms.registry import DatasetCapabilityValidator
from bcd_lab.data.finance import empirical_mu, return_dataset
from bcd_lab.diagnostics import DiagnosticRequest
from bcd_lab.experiments.composition import build_model_context
from bcd_lab.function_models import validate_function_model_config
from bcd_lab.initializers import get_phi_initializer, get_z_initializer
from bcd_lab.losses import get_objective
from bcd_lab.types import AlgorithmResult


@dataclass
class FittedModel:
    predictions: np.ndarray
    result: AlgorithmResult
    settings: dict[str, Any]


def prepare_model(returns: np.ndarray, settings: dict[str, Any]) -> tuple[Any, ...]:
    settings = copy.deepcopy(settings)
    dataset = return_dataset(returns, rank=settings["rank"])
    data = {
        "shape": {
            "n_rows": dataset.n_rows,
            "n_cols": dataset.n_cols,
            "rank": dataset.rank,
            "n_observations": dataset.n_observations,
        }
    }
    algorithm_spec = settings["algorithm"]
    algorithm = get_algorithm(algorithm_spec["name"])
    params = algorithm_spec["params"]
    regularization = params.get("regularization", {})
    if regularization.get("mu") == "empirical":
        regularization["mu"] = empirical_mu(returns, dataset.rank)
    runtime = settings["runtime"]
    max_iter = runtime["max_iter"]
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 0:
        raise ValueError("model.runtime.max_iter must be a nonnegative integer")
    if runtime.get("tol", 0) != 0:
        raise ValueError("Fixed-budget finance models require runtime.tol: 0")
    if np.dtype(runtime.get("dtype", "float64")) != np.dtype("float64"):
        raise ValueError("Finance models require runtime.dtype: float64")
    runtime["tol"] = 0.0
    runtime["dtype"] = "float64"
    runtime["checkpoint_iters"] = sorted(set(runtime.get("checkpoint_iters", []) + [max_iter]))
    config = algorithm.build_config(
        data=data, link=settings["link"], runtime=runtime, params=params
    )
    context = build_model_context(data, settings["link"], runtime)
    validate_function_model_config(context.link)
    if isinstance(algorithm, DatasetCapabilityValidator):
        algorithm.validate_dataset(config=config, dataset=dataset)
    initialization = settings["initialization"]
    get_z_initializer(initialization["z"]["name"])
    get_phi_initializer(initialization["phi"]["name"])
    return settings, dataset, algorithm, config, context


def fit_model(returns: np.ndarray, settings: dict[str, Any]) -> FittedModel:
    settings, dataset, algorithm, config, context = prepare_model(returns, settings)
    initialization = settings["initialization"]
    seed = settings.get("seed", 0)
    # Pass the recorded initialization seed directly for each training window.
    z_spec, phi_spec = initialization["z"], initialization["phi"]
    z = get_z_initializer(z_spec["name"]).build(
        dataset=dataset,
        config=context,
        params=z_spec.get("params", {}),
        seed=seed,
    )
    phi = get_phi_initializer(phi_spec["name"]).build(
        dataset=dataset,
        config=context,
        params=phi_spec.get("params", {}),
        seed=seed,
        z=z,
    )
    result = algorithm.run(
        config=config,
        dataset=dataset,
        objective=get_objective("squared_error"),
        z_init=z,
        phi_init=phi,
        rng=np.random.default_rng(seed),
        diagnostics_request=DiagnosticRequest(families={"objective": {"loss"}}, on_error="raise"),
    )
    if result.z is None:
        raise ValueError("Finance factor adapter requires algorithm result.z")
    predictions = result.z[: dataset.n_rows] @ result.z[dataset.n_rows :].T
    if result.phi is not None:
        predictions = result.phi.eval_array(predictions)
    if predictions.shape != returns.shape or not np.isfinite(predictions).all():
        raise ValueError("Algorithm produced invalid fitted returns")
    settings["seeds"] = {"initialization_z": seed, "initialization_phi": seed, "algorithm": seed}
    return FittedModel(np.asarray(predictions), result, settings)
