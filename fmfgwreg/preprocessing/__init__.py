"""Preprocessing utilities for medical image registration."""

from fmfgwreg.preprocessing.intensity import normalize_intensity, create_foreground_mask, histogram_matching
from fmfgwreg.preprocessing.resample import resample_to_spacing, align_field_of_view, center_crop_or_pad, resample_volume_to_reference
from fmfgwreg.preprocessing.rigid import rigid_alignment, affine_alignment, apply_transform

__all__ = [
    'normalize_intensity',
    'create_foreground_mask',
    'histogram_matching',
    'resample_to_spacing',
    'align_field_of_view',
    'center_crop_or_pad',
    'resample_volume_to_reference',
    'rigid_alignment',
    'affine_alignment',
    'apply_transform',
]

