"""Deformation field utilities for image registration."""

from fmfgwreg.deformation.displacement import compute_displacements, compute_displacements_vectorized, compute_displacement_statistics
from fmfgwreg.deformation.interpolation import SpacingAwareRBF, interpolate_dvf_rbf, interpolate_dvf_bspline
from fmfgwreg.deformation.warp import warp_volume, warp_segmentation, compose_dvfs, invert_dvf, compute_displacement_magnitude

__all__ = [
    'compute_displacements',
    'compute_displacements_vectorized',
    'compute_displacement_statistics',
    'SpacingAwareRBF',
    'interpolate_dvf_rbf',
    'interpolate_dvf_bspline',
    'warp_volume',
    'warp_segmentation',
    'compose_dvfs',
    'invert_dvf',
    'compute_displacement_magnitude',
]

