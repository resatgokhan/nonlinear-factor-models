"""Shared registry behavior for public extension points."""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class ComponentRegistry(Generic[T]):
    """Store named components while rejecting ambiguous registrations."""

    def __init__(self, component_label: str, component_plural: str) -> None:
        self._component_label = component_label
        self._component_plural = component_plural
        self._components: dict[str, T] = {}

    def register(self, name: str, component: T) -> T:
        """Register ``component`` under a unique, non-empty name."""
        if not name or not name.strip():
            raise ValueError(f"{self._component_label} name must be non-empty")
        if name in self._components:
            raise ValueError(f"{self._component_label} {name!r} is already registered")
        self._components[name] = component
        return component

    def get(self, name: str) -> T:
        """Return a component or raise an error that lists valid names."""
        try:
            return self._components[name]
        except KeyError as exc:
            known = ", ".join(self.names()) or "<none>"
            raise KeyError(
                f"Unknown {self._component_label.lower()} {name!r}. "
                f"Known {self._component_plural}: {known}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        """Return component names in deterministic order."""
        return tuple(sorted(self._components))
