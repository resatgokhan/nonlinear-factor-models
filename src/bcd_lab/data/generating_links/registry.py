"""Duplicate-safe registry for synthetic data-generating links."""

from __future__ import annotations

from collections.abc import Mapping

from bcd_lab._registry import ComponentRegistry
from bcd_lab.function_models import ScalarFunction

from .base import GeneratingLinkBuilder, GeneratingLinkContext, GeneratingLinkSpec

_GENERATING_LINKS = ComponentRegistry[GeneratingLinkBuilder](
    "Generating link",
    "generating links",
)


def register_generating_link(
    name: str,
    builder: GeneratingLinkBuilder,
) -> GeneratingLinkBuilder:
    """Register one uniquely named generating-link builder."""
    return _GENERATING_LINKS.register(name, builder)


def get_generating_link_builder(name: str) -> GeneratingLinkBuilder:
    """Return the registered builder for ``name``."""
    return _GENERATING_LINKS.get(name)


def list_generating_links() -> tuple[str, ...]:
    """Return generating-link names in deterministic order."""
    return _GENERATING_LINKS.names()


def parse_generating_link_spec(
    value: object,
    *,
    field: str = "data.generating_link",
) -> GeneratingLinkSpec:
    """Parse the canonical ``{name, params}`` generating-link mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    unknown = sorted(set(value) - {"name", "params"})
    if unknown:
        raise ValueError(f"{field} contains unknown fields: {', '.join(unknown)}")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{field}.name must be a non-empty string")
    params = value.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError(f"{field}.params must be a mapping")
    return GeneratingLinkSpec(name=name, params=dict(params))


def validate_generating_link_spec(spec: GeneratingLinkSpec) -> None:
    """Resolve a builder and validate its parameters without random work."""
    builder = get_generating_link_builder(spec.name)
    try:
        builder.validate_params(spec.params)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"data.generating_link {spec.name!r}: {exc}") from exc


def build_generating_link(
    spec: GeneratingLinkSpec,
    context: GeneratingLinkContext,
) -> ScalarFunction:
    """Build and structurally validate a configured generating link."""
    builder = get_generating_link_builder(spec.name)
    try:
        builder.validate_params(spec.params)
        link = builder.build(spec.params, context)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"data.generating_link {spec.name!r}: {exc}") from exc
    if not isinstance(link, ScalarFunction):
        raise TypeError(f"Generating link {spec.name!r} did not return a ScalarFunction")
    return link
