"""Namespaced initializer registries."""

from __future__ import annotations

from bcd_lab._registry import ComponentRegistry

from .base import JointInitializer, PhiInitializer, ZInitializer

_Z_INITIALIZERS = ComponentRegistry[ZInitializer]("Z initializer", "Z initializers")
_PHI_INITIALIZERS = ComponentRegistry[PhiInitializer](
    "Phi initializer",
    "phi initializers",
)
_JOINT_INITIALIZERS = ComponentRegistry[JointInitializer](
    "Joint initializer",
    "joint initializers",
)


def register_z_initializer(initializer: ZInitializer) -> ZInitializer:
    return _Z_INITIALIZERS.register(initializer.name, initializer)


def register_phi_initializer(initializer: PhiInitializer) -> PhiInitializer:
    return _PHI_INITIALIZERS.register(initializer.name, initializer)


def register_joint_initializer(initializer: JointInitializer) -> JointInitializer:
    return _JOINT_INITIALIZERS.register(initializer.name, initializer)


def get_z_initializer(name: str) -> ZInitializer:
    return _Z_INITIALIZERS.get(name)


def get_phi_initializer(name: str) -> PhiInitializer:
    return _PHI_INITIALIZERS.get(name)


def get_joint_initializer(name: str) -> JointInitializer:
    return _JOINT_INITIALIZERS.get(name)


def list_z_initializers() -> tuple[str, ...]:
    return _Z_INITIALIZERS.names()


def list_phi_initializers() -> tuple[str, ...]:
    return _PHI_INITIALIZERS.names()


def list_joint_initializers() -> tuple[str, ...]:
    return _JOINT_INITIALIZERS.names()
