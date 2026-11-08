"""Finance-only YAML configuration; generic sweep semantics are unchanged."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bcd_lab.experiments.loader import deep_merge
from bcd_lab.finance.covariance import get_estimator

WINDOW_DEFAULTS = {
    "start_date": "2015-01-01",
    "end_date": "2025-01-01",
    "lookback": 250,
    "rebalance": 21,
    "horizon": 21,
    "periods": 107,
}
EVALUATION_DEFAULTS = {
    "trading_days": 250,
    "risk_free": 0.04,
    "return_convention": "weighted_log",
}


def mapping(value: Any, label: str, allowed: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    if allowed is not None and set(value) - allowed:
        raise ValueError(f"Unknown {label} fields: {sorted(set(value) - allowed)}")
    return dict(value)


def integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def resolve_config(config: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    spec = mapping(
        copy.deepcopy(config),
        "finance config",
        {
            "schema_version",
            "name",
            "panel",
            "window",
            "evaluation",
            "model_defaults",
            "methods",
        },
    )
    if type(spec.get("schema_version")) is not int or spec["schema_version"] != 1:
        raise ValueError("finance schema_version must be 1")
    if not isinstance(spec.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9_-]+", spec["name"]):
        raise ValueError("Finance name must contain only letters, digits, underscores or hyphens")
    panel = mapping(spec.get("panel"), "panel", {"path", "sha256", "n_assets"})
    if not isinstance(panel.get("path"), str) or not panel["path"]:
        raise ValueError("panel.path is required")
    path = Path(panel["path"]).expanduser()
    panel["path"] = str((base_dir / path).resolve() if not path.is_absolute() else path.resolve())
    if "n_assets" in panel:
        integer(panel["n_assets"], "panel.n_assets", 1)
    spec["panel"] = panel
    spec["window"] = {
        **WINDOW_DEFAULTS,
        **mapping(spec.get("window", {}), "window", set(WINDOW_DEFAULTS)),
    }
    for key in ("lookback", "rebalance", "horizon", "periods"):
        integer(spec["window"][key], f"window.{key}", 1)
    for key in ("start_date", "end_date"):
        spec["window"][key] = str(spec["window"][key])
    evaluation = {
        **EVALUATION_DEFAULTS,
        **mapping(spec.get("evaluation", {}), "evaluation", set(EVALUATION_DEFAULTS)),
    }
    integer(evaluation["trading_days"], "evaluation.trading_days", 1)
    if evaluation["return_convention"] != "weighted_log":
        raise ValueError("Only weighted_log evaluation is supported")
    risk_free = evaluation["risk_free"]
    if (
        isinstance(risk_free, bool)
        or not isinstance(risk_free, (float, int))
        or not np.isfinite(risk_free)
        or risk_free <= -1
    ):
        raise ValueError("evaluation.risk_free must be finite and greater than -1")
    spec["evaluation"] = evaluation
    spec["methods"] = _methods(spec.pop("model_defaults", {}), spec.get("methods"))
    return spec


def _methods(defaults: Any, methods: Any) -> dict[str, Any]:
    defaults = mapping(defaults, "model_defaults")
    methods = mapping(methods, "methods")
    if not methods:
        raise ValueError("At least one method is required")
    result = {}
    for name, value in methods.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(
                "Method names must contain only letters, digits, underscores or hyphens"
            )
        method = mapping(value, f"method {name}", {"model", "estimator", "portfolio"})
        estimator = mapping(method.get("estimator"), "estimator", {"name", "params"})
        get_estimator(estimator.get("name", ""))
        estimator["params"] = mapping(estimator.get("params", {}), "estimator.params")
        method["estimator"] = estimator
        method.setdefault("portfolio", "gmv")
        if method["portfolio"] not in {"gmv", "equal_weight"}:
            raise ValueError("portfolio must be gmv or equal_weight")
        if "model" in method:
            overrides = mapping(method["model"], "model")
            model = deep_merge(defaults, overrides)
            for component, selector in (("link", "kind"), ("algorithm", "name")):
                supplied = overrides.get(component, {})
                if selector in supplied and supplied[selector] != defaults.get(component, {}).get(
                    selector
                ):
                    model[component] = supplied
            mapping(
                model, "model", {"rank", "algorithm", "link", "initialization", "runtime", "seed"}
            )
            integer(model.get("rank"), "model.rank", 1)
            integer(model.setdefault("seed", 0), "model.seed")
            method["model"] = model
        elif estimator["name"] in {"fitted_residual", "fitted_residual_regularized"}:
            raise ValueError(f"{estimator['name']} requires a model")
        result[name] = method
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    return resolve_config(yaml.safe_load(source.read_text()), base_dir=source.parent)
