"""
Resampling and field-of-view alignment utilities.

Handles spacing normalization and volume alignment using SimpleITK.
"""

import numpy as np
import SimpleITK as sitk
from typing import Tuple, Optional
import warnings


def resample_to_spacing(volume: np.ndarray,
                        spacing_from: Tuple[float, float, float],
                        spacing_to: Tuple[float, float, float],
                        interpolation: str = 'linear',
                        ) -> np.ndarray:
    """
    Resample volume to target spacing using SimpleITK.
    
    Args:
        volume: 3D numpy array (H, W, D)
        spacing_from: Current spacing (sz, sy, sx) in mm
        spacing_to: Target spacing (sz, sy, sx) in mm
        interpolation: 'linear', 'nearest', or 'bspline'
        
    Returns:
        Resampled volume with new spacing
    """
    # Convert to SimpleITK format (D, H, W)
    volume_sitk_order = np.transpose(volume, (2, 0, 1))
    image = sitk.GetImageFromArray(volume_sitk_order)
    
    # Set current spacing (convert to SimpleITK convention: sx, sy, sz)
    image.SetSpacing(spacing_from[::-1])
    
    # Compute new size
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    
    new_spacing = spacing_to[::-1]  # Convert to (sx, sy, sz)
    new_size = [
        int(round(original_size[i] * original_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]
    
    # Set interpolator
    if interpolation == 'linear':
        interpolator = sitk.sitkLinear
    elif interpolation == 'nearest':
        interpolator = sitk.sitkNearestNeighbor
    elif interpolation == 'bspline':
        interpolator = sitk.sitkBSpline
    else:
        raise ValueError(f"Unknown interpolation: {interpolation}")
    
    # Resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(interpolator)
    
    resampled = resampler.Execute(image)
    
    # Convert back to numpy (H, W, D)
    volume_resampled = sitk.GetArrayFromImage(resampled)
    volume_resampled = np.transpose(volume_resampled, (1, 2, 0))
    
    return volume_resampled


def align_field_of_view(fixed: np.ndarray,
                        moving: np.ndarray,
                        fixed_spacing: Tuple[float, float, float],
                        moving_spacing: Tuple[float, float, float],
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align field of view by cropping/padding to common size.
    
    Centers volumes and ensures they have the same physical extent.
    
    Args:
        fixed: Fixed volume (H_f, W_f, D_f)
        moving: Moving volume (H_m, W_m, D_m)
        fixed_spacing: Fixed spacing (sz, sy, sx)
        moving_spacing: Moving spacing (sz, sy, sx)
        
    Returns:
        fixed_aligned: Fixed volume in common FOV
        moving_aligned: Moving volume in common FOV
    """
    # First, resample moving to fixed spacing
    moving_resampled = resample_to_spacing(moving, moving_spacing, fixed_spacing)
    
    # Now both have same spacing, align sizes
    fixed_shape = np.array(fixed.shape)
    moving_shape = np.array(moving_resampled.shape)
    
    # Target size: minimum in each dimension
    target_shape = np.minimum(fixed_shape, moving_shape)
    
    # Center crop both volumes
    fixed_aligned = center_crop_or_pad(fixed, target_shape)
    moving_aligned = center_crop_or_pad(moving_resampled, target_shape)
    
    return fixed_aligned, moving_aligned


def center_crop_or_pad(volume: np.ndarray,
                       target_shape: Tuple[int, int, int],
                       pad_value: float = 0,
                       ) -> np.ndarray:
    """
    Center crop or pad volume to target shape.
    
    Args:
        volume: Input volume
        target_shape: Target (H, W, D)
        pad_value: Value for padding
        
    Returns:
        Volume with target shape
    """
    current_shape = np.array(volume.shape)
    target_shape = np.array(target_shape)
    
    # Compute padding/cropping
    diff = target_shape - current_shape
    
    result = volume.copy()
    
    for axis in range(3):
        if diff[axis] > 0:
            # Pad
            pad_total = diff[axis]
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            
            pad_width = [(0, 0)] * 3
            pad_width[axis] = (pad_before, pad_after)
            result = np.pad(result, pad_width, mode='constant', constant_values=pad_value)
            
        elif diff[axis] < 0:
            # Crop
            crop_total = -diff[axis]
            crop_before = crop_total // 2
            crop_after = crop_total - crop_before
            
            slices = [slice(None)] * 3
            slices[axis] = slice(crop_before, current_shape[axis] - crop_after)
            result = result[tuple(slices)]
    
    return result


def resample_volume_to_reference(volume: np.ndarray,
                                  volume_spacing: Tuple[float, float, float],
                                  volume_origin: Tuple[float, float, float],
                                  reference_shape: Tuple[int, int, int],
                                  reference_spacing: Tuple[float, float, float],
                                  reference_origin: Tuple[float, float, float],
                                  interpolation: str = 'linear',
                                  ) -> np.ndarray:
    """
    Resample volume to match reference geometry.
    
    This ensures both volumes have identical grid positions.
    
    Args:
        volume: Volume to resample
        volume_spacing: Volume spacing
        volume_origin: Volume origin
        reference_shape: Target shape
        reference_spacing: Target spacing
        reference_origin: Target origin
        interpolation: Interpolation method
        
    Returns:
        Resampled volume matching reference geometry
    """
    # Convert to SimpleITK
    volume_sitk_order = np.transpose(volume, (2, 0, 1))
    image = sitk.GetImageFromArray(volume_sitk_order)
    image.SetSpacing(volume_spacing[::-1])
    image.SetOrigin(volume_origin)
    
    # Create reference image
    reference = sitk.Image(
        reference_shape[::-1],  # (D, W, H) in SimpleITK
        sitk.sitkFloat32
    )
    reference.SetSpacing(reference_spacing[::-1])
    reference.SetOrigin(reference_origin)
    
    # Set interpolator
    if interpolation == 'linear':
        interpolator = sitk.sitkLinear
    elif interpolation == 'nearest':
        interpolator = sitk.sitkNearestNeighbor
    elif interpolation == 'bspline':
        interpolator = sitk.sitkBSpline
    else:
        interpolator = sitk.sitkLinear
    
    # Resample
    resampled = sitk.Resample(
        image,
        reference,
        sitk.Transform(),
        interpolator,
        0.0,
        image.GetPixelID()
    )
    
    # Convert back
    volume_resampled = sitk.GetArrayFromImage(resampled)
    volume_resampled = np.transpose(volume_resampled, (1, 2, 0))
    
    return volume_resampled


def compute_common_spacing(spacing1: Tuple[float, float, float],
                           spacing2: Tuple[float, float, float],
                           strategy: str = 'min',
                           ) -> Tuple[float, float, float]:
    """
    Compute a common spacing for two volumes.
    
    Args:
        spacing1, spacing2: Input spacings
        strategy: 'min' (finest), 'max' (coarsest), 'mean', or 'median'
        
    Returns:
        Common spacing
    """
    spacing1 = np.array(spacing1)
    spacing2 = np.array(spacing2)
    
    if strategy == 'min':
        common = np.minimum(spacing1, spacing2)
    elif strategy == 'max':
        common = np.maximum(spacing1, spacing2)
    elif strategy == 'mean':
        common = (spacing1 + spacing2) / 2
    elif strategy == 'median':
        common = np.median([spacing1, spacing2], axis=0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return tuple(common)

