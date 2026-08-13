"""Data loading and generation utilities."""

from __future__ import annotations

from .base import DatasetLoader
from .csv import CsvDatasetLoader
from .generating_links import (
    GeneratingLinkBuilder,
    GeneratingLinkContext,
    GeneratingLinkSpec,
    build_generating_link,
    get_generating_link_builder,
    list_generating_links,
    parse_generating_link_spec,
    register_generating_link,
)
from .registry import (
    get_data_loader,
    list_data_loaders,
    load_dataset,
    register_data_loader,
)
from .synthetic import SyntheticDatasetLoader

# Built-ins are wired here in one visible, deterministic order. External
# components use the public register_data_loader() API.
register_data_loader(CsvDatasetLoader())
register_data_loader(SyntheticDatasetLoader())

__all__ = [
    "DatasetLoader",
    "register_data_loader",
    "get_data_loader",
    "list_data_loaders",
    "load_dataset",
    "GeneratingLinkBuilder",
    "GeneratingLinkContext",
    "GeneratingLinkSpec",
    "build_generating_link",
    "get_generating_link_builder",
    "list_generating_links",
    "parse_generating_link_spec",
    "register_generating_link",
]
