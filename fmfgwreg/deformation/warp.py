"""
Volume warping using deformation vector fields.

Spacing-aware trilinear interpolation.
"""

import numpy as np
from scipy.ndimage import map_coordinates
from typing import Tuple, Optional
import warnings


def warp_volume(volume: np.ndarray,
                dvf: np.ndarray,
                spacing: Optional[Tuple[float, float, float]] = None,
                mode: str = 'constant',
                cval: float = 0.0,
                order: int = 1,
                ) -> np.ndarray:
    """
    Warp volume using deformation vector field.
    
    Args:
        volume: (H, W, D) input volume
        dvf: (H, W, D, 3) deformation field in voxel displacements
        spacing: Physical spacing (not used in voxel-space warping, but kept for API consistency)
        mode: Boundary handling ('constant', 'nearest', 'reflect', 'wrap')
        cval: Value for out-of-bounds regions (if mode='constant')
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
        
    Returns:
        warped: (H, W, D) warped volume
    """
    H, W, D = volume.shape
    
    # Create coordinate grid
    i = np.arange(H)
    j = np.arange(W)
    k = np.arange(D)
    
    ii, jj, kk = np.meshgrid(i, j, k, indexing='ij')
    
    # Apply displacement: new_coords = original_coords + dvf
    # Note: dvf is in (i, j, k) = (H, W, D) ordering
    i_new = ii + dvf[..., 0]
    j_new = jj + dvf[..., 1]
    k_new = kk + dvf[..., 2]
    
    # Stack coordinates for map_coordinates
    coords = np.array([i_new, j_new, k_new])
    
    # Interpolate
    warped = map_coordinates(
        volume,
        coords,
        order=order,
        mode=mode,
        cval=cval,
        prefilter=True if order > 1 else False,
    )
    
    return warped


def warp_segmentation(segmentation: np.ndarray,
                      dvf: np.ndarray,
                      ) -> np.ndarray:
    """
    Warp segmentation mask using nearest-neighbor interpolation.
    
    Args:
        segmentation: (H, W, D) integer segmentation
        dvf: (H, W, D, 3) deformation field
        
    Returns:
        warped_seg: (H, W, D) warped segmentation
    """
    return warp_volume(
        segmentation,
        dvf,
        mode='constant',
        cval=0,
        order=0,  # Nearest neighbor for labels
    )


def compose_dvfs(dvf1: np.ndarray,
                 dvf2: np.ndarray,
                 ) -> np.ndarray:
    """
    Compose two deformation fields.
    
    Result is dvf that applies dvf1 then dvf2.
    
    Args:
        dvf1: (H, W, D, 3) first DVF
        dvf2: (H, W, D, 3) second DVF
        
    Returns:
        composed_dvf: (H, W, D, 3) composition
    """
    H, W, D = dvf1.shape[:3]
    
    # Create coordinate grid
    i = np.arange(H)
    j = np.arange(W)
    k = np.arange(D)
    
    ii, jj, kk = np.meshgrid(i, j, k, indexing='ij')
    
    # Apply first DVF
    i_warped = ii + dvf1[..., 0]
    j_warped = jj + dvf1[..., 1]
    k_warped = kk + dvf1[..., 2]
    
    # Sample second DVF at warped positions
    coords = np.array([i_warped, j_warped, k_warped])
    
    dvf2_at_warped = np.stack([
        map_coordinates(dvf2[..., 0], coords, order=1, mode='nearest'),
        map_coordinates(dvf2[..., 1], coords, order=1, mode='nearest'),
        map_coordinates(dvf2[..., 2], coords, order=1, mode='nearest'),
    ], axis=-1)
    
    # Compose: total displacement is dvf1 + dvf2(warped)
    composed = dvf1 + dvf2_at_warped
    
    return composed


def invert_dvf(dvf: np.ndarray,
               num_iterations: int = 20,
               ) -> np.ndarray:
    """
    Approximate inverse of deformation field using fixed-point iteration.
    
    Args:
        dvf: (H, W, D, 3) forward DVF
        num_iterations: Number of fixed-point iterations
        
    Returns:
        inv_dvf: (H, W, D, 3) approximate inverse DVF
    """
    H, W, D = dvf.shape[:3]
    
    # Initialize inverse as negative of forward
    inv_dvf = -dvf.copy()
    
    # Create coordinate grid
    i = np.arange(H)
    j = np.arange(W)
    k = np.arange(D)
    ii, jj, kk = np.meshgrid(i, j, k, indexing='ij')
    
    # Fixed-point iteration
    for iteration in range(num_iterations):
        # Warp forward DVF by current inverse
        i_warped = ii + inv_dvf[..., 0]
        j_warped = jj + inv_dvf[..., 1]
        k_warped = kk + inv_dvf[..., 2]
        
        coords = np.array([i_warped, j_warped, k_warped])
        
        dvf_warped = np.stack([
            map_coordinates(dvf[..., 0], coords, order=1, mode='nearest'),
            map_coordinates(dvf[..., 1], coords, order=1, mode='nearest'),
            map_coordinates(dvf[..., 2], coords, order=1, mode='nearest'),
        ], axis=-1)
        
        # Update inverse: inv = -(dvf at warped position) - inv
        inv_dvf = -dvf_warped + inv_dvf
    
    return inv_dvf


def compute_displacement_magnitude(dvf: np.ndarray,
                                   spacing: Tuple[float, float, float],
                                   ) -> np.ndarray:
    """
    Compute displacement magnitude in physical space.
    
    Args:
        dvf: (H, W, D, 3) DVF in voxel displacements
        spacing: (sz, sy, sx) physical spacing
        
    Returns:
        magnitude: (H, W, D) displacement magnitude in mm
    """
    spacing_arr = np.array(spacing)
    dvf_physical = dvf * spacing_arr
    magnitude = np.linalg.norm(dvf_physical, axis=-1)
    return magnitude

