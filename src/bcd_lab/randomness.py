"""Deterministic, named random streams for one experiment run."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from numbers import Integral

import numpy as np


def validate_seed(value: object, *, field: str = "seed") -> int:
    """Return an integer seed without accepting coercible strings or floats."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be one non-negative integer")
    seed = int(value)
    if seed < 0:
        raise ValueError(f"{field} must be non-negative")
    return seed


def _stream_id(name: str) -> int:
    if not name:
        raise ValueError("Random stream name must be non-empty")
    digest = hashlib.blake2s(name.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little")


@dataclass(frozen=True)
class RunRandomness:
    """Derive order-independent component seeds from one effective run seed."""

    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", validate_seed(self.seed))

    @classmethod
    def for_trial(cls, configured_seed: int, trial_index: int) -> RunRandomness:
        configured_seed = validate_seed(configured_seed)
        trial_index = validate_seed(trial_index, field="trial_index")
        if trial_index == 0:
            return cls(seed=configured_seed)
        sequence = np.random.SeedSequence([configured_seed, trial_index])
        return cls(seed=int(sequence.generate_state(1, dtype=np.uint32)[0]))

    def child_seed(self, name: str) -> int:
        sequence = np.random.SeedSequence([self.seed, _stream_id(name)])
        return int(sequence.generate_state(1, dtype=np.uint32)[0])

    def generator(self, name: str) -> np.random.Generator:
        return np.random.default_rng(self.child_seed(name))

    def manifest(self, *names: str) -> dict[str, int]:
        """Return the effective root and named component seeds for provenance."""
        return {
            "run": self.seed,
            **{name: self.child_seed(name) for name in names},
        }
