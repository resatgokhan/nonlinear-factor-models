"""RKHS function representations and utilities."""

from .gaussian_rkhs import (
    GaussianRKHS,
    GaussianRKHSConfig,
    load_rkhs,
    save_rkhs,
)
from .gaussianKernelMath import (
    gaussian_dkernel_dx,
    gaussian_dkernel_dx_matrix,
    gaussian_kernel,
    gaussian_kernel_matrix,
)
from .generators import polynomial_fit_phi, random_monotone_poly

__all__ = [
    "GaussianRKHS",
    "GaussianRKHSConfig",
    "gaussian_dkernel_dx",
    "gaussian_dkernel_dx_matrix",
    "gaussian_kernel",
    "gaussian_kernel_matrix",
    "load_rkhs",
    "polynomial_fit_phi",
    "random_monotone_poly",
    "save_rkhs",
]
