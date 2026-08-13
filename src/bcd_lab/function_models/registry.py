"""Registry for registering and looking up function model builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol

import numpy as np

from bcd_lab._registry import ComponentRegistry
from bcd_lab.config import LinkConfig, ModelBuildContext
from bcd_lab.function_models.base import ScalarFunction

if TYPE_CHECKING:
    from bcd_lab.types import DatasetBundle


class ScalarFunctionBuilder(Protocol):
    """Build a scalar function for a configured link kind."""

    def __call__(
        self,
        config: ModelBuildContext,
        params: dict[str, Any] | None = None,
        dataset: DatasetBundle | None = None,
        z: np.ndarray | None = None,
    ) -> ScalarFunction: ...


FunctionModelConfigValidator = Callable[[LinkConfig], None]


@dataclass(frozen=True)
class FunctionModelDefinition:
    """Builder and optional component-owned configuration validator."""

    builder: ScalarFunctionBuilder
    validate_config: FunctionModelConfigValidator | None = None


_DEFINITIONS = ComponentRegistry[FunctionModelDefinition](
    "Function model",
    "function models",
)


def register_function_model(
    kind: str,
    *,
    validate_config: FunctionModelConfigValidator | None = None,
) -> Callable[[ScalarFunctionBuilder], ScalarFunctionBuilder]:
    """Decorator to register a function model builder under its unique 'kind'."""

    def decorator(builder: ScalarFunctionBuilder) -> ScalarFunctionBuilder:
        _DEFINITIONS.register(kind, FunctionModelDefinition(builder, validate_config))
        return builder

    return decorator


def get_function_model_builder(kind: str) -> ScalarFunctionBuilder:
    """Look up the registered builder for a given model kind."""
    return _DEFINITIONS.get(kind).builder


def validate_function_model_config(config: LinkConfig) -> None:
    """Validate component-owned link parameters without constructing a model."""
    definition = _DEFINITIONS.get(str(getattr(config.kind, "value", config.kind)))
    if definition.validate_config is not None:
        definition.validate_config(config)


def list_function_models() -> tuple[str, ...]:
    """List registered function-model kinds in deterministic order."""
    return _DEFINITIONS.names()
