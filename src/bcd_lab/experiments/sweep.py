"""Sweep axis expansion and recursive nested replacing utilities."""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any

from .schema import SweepAxis


def expand_axes(axes: list[SweepAxis]) -> list[dict[str, Any]]:
    if not axes:
        return [{}]

    axis_cases = [[{axis.path: val} for val in axis.values] for axis in axes]
    cases = []
    for combo in itertools.product(*axis_cases):
        merged = {}
        for assignment in combo:
            merged.update(assignment)
        cases.append(merged)
    return cases


def replace_nested(obj: Any, path: str, value: Any) -> Any:
    """Recursively replaces nested properties inside dataclasses and dictionaries."""
    parts = path.split(".")
    if len(parts) == 1:
        if isinstance(obj, dict):
            new_obj = dict(obj)
            new_obj[parts[0]] = value
            return new_obj
        else:
            return dataclasses.replace(obj, **{parts[0]: value})

    head = parts[0]
    tail = ".".join(parts[1:])

    if isinstance(obj, dict):
        new_obj = dict(obj)
        new_obj[head] = replace_nested(new_obj.get(head, {}), tail, value)
        return new_obj
    else:
        child = getattr(obj, head)
        new_child = replace_nested(child, tail, value)
        return dataclasses.replace(obj, **{head: new_child})


def replace_nested_strict(obj: Any, path: str, value: Any) -> Any:
    """Replace an existing nested field, raising a readable error for bad paths."""
    parts = path.split(".")
    current = obj
    for index, part in enumerate(parts):
        traversed = ".".join(parts[: index + 1])
        if isinstance(current, dict):
            if part not in current:
                raise ValueError(f"Invalid sweep path {path!r}: {traversed!r} does not exist")
            current = current[part]
        elif dataclasses.is_dataclass(current):
            field_names = {field.name for field in dataclasses.fields(current)}
            if part not in field_names:
                raise ValueError(f"Invalid sweep path {path!r}: {traversed!r} does not exist")
            current = getattr(current, part)
        else:
            raise ValueError(f"Invalid sweep path {path!r}: cannot traverse through {traversed!r}")
    return replace_nested(obj, path, value)
