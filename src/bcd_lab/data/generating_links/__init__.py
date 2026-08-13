"""Public generating-link extension point for synthetic datasets."""

from __future__ import annotations

from .base import GeneratingLinkBuilder, GeneratingLinkContext, GeneratingLinkSpec
from .implementations import (
    EXPONENTIAL_BUILDER,
    IDENTITY_BUILDER,
    RANDOM_MONOTONE_RKHS_BUILDER,
    SIGMOID_BUILDER,
    TANH_BUILDER,
)
from .registry import (
    build_generating_link,
    get_generating_link_builder,
    list_generating_links,
    parse_generating_link_spec,
    register_generating_link,
    validate_generating_link_spec,
)

# Built-ins are wired in one visible, deterministic order. External builders
# use the public register_generating_link() API.
register_generating_link("exponential", EXPONENTIAL_BUILDER)
register_generating_link("identity", IDENTITY_BUILDER)
register_generating_link("random_monotone_rkhs", RANDOM_MONOTONE_RKHS_BUILDER)
register_generating_link("sigmoid", SIGMOID_BUILDER)
register_generating_link("tanh", TANH_BUILDER)

__all__ = [
    "GeneratingLinkBuilder",
    "GeneratingLinkContext",
    "GeneratingLinkSpec",
    "build_generating_link",
    "get_generating_link_builder",
    "list_generating_links",
    "parse_generating_link_spec",
    "register_generating_link",
    "validate_generating_link_spec",
]
