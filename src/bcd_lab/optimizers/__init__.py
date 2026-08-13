"""Public fixed-coordinate optimizer extension point."""

from __future__ import annotations

from .base import ArrayOptimizer, OptimizerBuilder, OptimizerSpec
from .checkpoints import restore_optimizer, serialize_optimizer
from .implementations import Adam, GradientDescent, build_adam, build_gradient_descent
from .registry import (
    build_optimizer,
    get_optimizer_builder,
    list_optimizers,
    parse_optimizer_spec,
    register_optimizer,
)

# Built-ins are wired explicitly and produce a fresh stateful instance per run.
register_optimizer("adam", build_adam)
register_optimizer("gradient_descent", build_gradient_descent)

__all__ = [
    "Adam",
    "ArrayOptimizer",
    "GradientDescent",
    "OptimizerBuilder",
    "OptimizerSpec",
    "build_optimizer",
    "get_optimizer_builder",
    "list_optimizers",
    "parse_optimizer_spec",
    "register_optimizer",
    "restore_optimizer",
    "serialize_optimizer",
]
