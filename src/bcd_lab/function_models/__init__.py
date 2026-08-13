"""Algorithm-facing function model abstractions."""

from .base import (
    BallRadiusResolver,
    CheckpointableFunction,
    CopyableFunction,
    DifferentiableFunction,
    FunctionConstraintConfig,
    NativeDistanceFunction,
    NormedFunction,
    ProjectableFunction,
    ScalarFunction,
    SerializableFunctionModel,
    TrainableFunction,
)
from .binned_piecewise_linear_rkhs import (
    BinnedPiecewiseLinearRKHS,
    build_binned_piecewise_linear_rkhs,
)
from .checkpoints import restore_function_model, serialize_function_model
from .fixed_models import (
    ExponentialFunction,
    IdentityFunction,
    SigmoidFunction,
    TanhFunction,
    build_exponential,
    build_identity,
    build_sigmoid,
)
from .piecewise_linear_rkhs import (
    FixedGridBounds,
    GridRebuildPolicy,
    InputQuantileGridBounds,
    PiecewiseLinearRKHS,
    PiecewiseLinearRKHSConfig,
    build_piecewise_linear_rkhs,
    l2_grid_diff,
    validate_piecewise_linear_config,
)
from .registry import (
    get_function_model_builder,
    list_function_models,
    register_function_model,
    validate_function_model_config,
)
from .rkhs import GaussianRKHS, GaussianRKHSConfig
from .rkhs.gaussian_rkhs import build_gaussian_rkhs

# Built-ins are wired here in one visible, deterministic order. External
# builders use the public register_function_model() decorator.
register_function_model("identity")(build_identity)
register_function_model("sigmoid")(build_sigmoid)
register_function_model("exponential")(build_exponential)
register_function_model(
    "piecewise_linear_rkhs",
    validate_config=validate_piecewise_linear_config,
)(build_piecewise_linear_rkhs)
register_function_model(
    "binned_piecewise_linear_rkhs",
    validate_config=validate_piecewise_linear_config,
)(build_binned_piecewise_linear_rkhs)
register_function_model("gaussian_rkhs")(build_gaussian_rkhs)

__all__ = [
    "BallRadiusResolver",
    "ExponentialFunction",
    "FixedGridBounds",
    "DifferentiableFunction",
    "CopyableFunction",
    "CheckpointableFunction",
    "FunctionConstraintConfig",
    "GaussianRKHS",
    "GridRebuildPolicy",
    "GaussianRKHSConfig",
    "BinnedPiecewiseLinearRKHS",
    "IdentityFunction",
    "InputQuantileGridBounds",
    "NativeDistanceFunction",
    "NormedFunction",
    "PiecewiseLinearRKHS",
    "PiecewiseLinearRKHSConfig",
    "ProjectableFunction",
    "ScalarFunction",
    "SerializableFunctionModel",
    "SigmoidFunction",
    "TanhFunction",
    "TrainableFunction",
    "get_function_model_builder",
    "list_function_models",
    "l2_grid_diff",
    "register_function_model",
    "restore_function_model",
    "serialize_function_model",
    "validate_function_model_config",
]
