"""Reusable validators for renderer-owned YAML options."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

OptionValidator = Callable[[Any, str], Any]


def optional_float(*, positive: bool = False) -> OptionValidator:
    def validate(value: Any, label: str) -> float | None:
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a float or null") from exc
        if positive and result <= 0:
            raise ValueError(f"{label} must be positive")
        return result

    return validate


def optional_int(*, positive: bool = False) -> OptionValidator:
    def validate(value: Any, label: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"{label} must be an integer or null")
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be an integer or null") from exc
        if positive and result <= 0:
            raise ValueError(f"{label} must be positive")
        return result

    return validate


def optional_bool(value: Any, label: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean or null")
    return value


def optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string or null")
    return value


def choice(*values: str) -> OptionValidator:
    allowed = frozenset(values)

    def validate(value: Any, label: str) -> str | None:
        if value is None:
            return None
        if value not in allowed:
            choices = ", ".join(f"'{item}'" for item in sorted(allowed))
            raise ValueError(f"{label} must be one of {choices}")
        return str(value)

    return validate


def string_list(value: Any, label: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a list of strings or null")
    return list(value)


def positive_iterations(value: Any, label: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of numbers or null")
    normalized: list[int] = []
    for entry in value:
        try:
            iteration = int(round(float(entry)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a list of numbers") from exc
        if iteration <= 0:
            raise ValueError(f"{label} values must be positive")
        normalized.append(iteration)
    return normalized


def normalize_fields(
    fields: Mapping[str, OptionValidator],
) -> Callable[[Mapping[str, Any], str], dict[str, Any]]:
    """Build a renderer normalizer from its explicit option schema."""

    def normalize(spec: Mapping[str, Any], label: str) -> dict[str, Any]:
        return {
            name: validator(spec.get(name), f"{label}.{name}")
            for name, validator in fields.items()
            if name in spec
        }

    return normalize
