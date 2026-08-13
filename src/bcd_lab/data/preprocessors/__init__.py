"""Dataset-specific preprocessors: raw → clean CSV triples in ./data/."""

from .base import DataPreprocessor
from .jester import JesterDataset1
from .movielens import (
    MovieLens1M,
    MovieLens10M,
    MovieLens20M,
    MovieLens100K,
    MovieLensLatestSmall,
)
from .registry import (
    get_preprocessor,
    list_preprocessors,
    preprocess_dataset,
    register_preprocessor,
)

# Built-ins are wired here in one visible, deterministic order. External
# components use the public register_preprocessor() API.
register_preprocessor(MovieLens100K())
register_preprocessor(MovieLens1M())
register_preprocessor(MovieLens10M())
register_preprocessor(MovieLens20M())
register_preprocessor(MovieLensLatestSmall())
register_preprocessor(JesterDataset1())

__all__ = [
    "DataPreprocessor",
    "register_preprocessor",
    "get_preprocessor",
    "list_preprocessors",
    "preprocess_dataset",
]
