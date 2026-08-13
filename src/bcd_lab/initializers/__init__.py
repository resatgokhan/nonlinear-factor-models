"""Configurable initialization strategies."""

from bcd_lab.function_models import list_function_models

from .builder import InitializationSpec, InitializerSpec, build_initialization
from .joint import SvdWithZeroPhiInitializer
from .phi import (
    GenericPhiInitializer,
    IdentityPhiInitializer,
    NonePhiInitializer,
    ZeroPhiInitializer,
    build_zero_phi,
    identity_phi,
)
from .registry import (
    get_joint_initializer,
    get_phi_initializer,
    get_z_initializer,
    list_joint_initializers,
    list_phi_initializers,
    list_z_initializers,
    register_joint_initializer,
    register_phi_initializer,
    register_z_initializer,
)
from .z import SvdZInitializer, svd_z

# Built-ins are wired here in one visible, deterministic order. Each function
# model also receives the generic initializer of the same name.
register_z_initializer(SvdZInitializer())
register_phi_initializer(ZeroPhiInitializer())
register_phi_initializer(IdentityPhiInitializer())
register_phi_initializer(NonePhiInitializer())
for _model_kind in list_function_models():
    register_phi_initializer(GenericPhiInitializer(_model_kind))
register_joint_initializer(SvdWithZeroPhiInitializer())

__all__ = [
    "InitializationSpec",
    "InitializerSpec",
    "build_initialization",
    "get_joint_initializer",
    "get_phi_initializer",
    "get_z_initializer",
    "list_joint_initializers",
    "list_phi_initializers",
    "list_z_initializers",
    "register_joint_initializer",
    "register_phi_initializer",
    "register_z_initializer",
    "svd_z",
    "identity_phi",
    "build_zero_phi",
]
