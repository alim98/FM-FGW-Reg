"""
Dense DVF interpolation from sparse node displacements.

Spacing-aware RBF interpolation.
"""

import numpy as np
from scipy.interpolate import RBFInterpolator
from typing import Tuple, Optional
import warnings


class SpacingAwareRBF:
    """
    Radial Basis Function interpolation in physical space.
    
    Fits RBF on physical coordinates but returns voxel-space DVF.
    """
    
    def __init__(self,
                 coords_voxel: np.ndarray,
                 displacements_voxel: np.ndarray,
                 spacing: Tuple[float, float, float],
                 smoothing: float = 0.0,
                 kernel: str = 'thin_plate_spline',
                 degree: Optional[int] = None,
                 ):
        """
        Initialize RBF interpolator.
        
        Args:
            coords_voxel: (N, 3) control point coordinates in voxel space
            displacements_voxel: (N, 3) displacements in voxel space
            spacing: (sz, sy, sx) physical spacing in mm
            smoothing: Smoothing parameter (0 = exact interpolation)
            kernel: RBF kernel ('thin_plate_spline', 'multiquadric', 'gaussian', etc.)
            degree: Polynomial degree for augmentation
        """
        self.coords_voxel = coords_voxel
        self.displacements_voxel = displacements_voxel
        self.spacing = np.array(spacing)
        self.smoothing = smoothing
        self.kernel = kernel
        
        # Convert to physical space
        self.coords_physical = coords_voxel * self.spacing
        self.displacements_physical = displacements_voxel * self.spacing
        
        # Fit RBF for each dimension
        self.rbfs = []
        for dim in range(3):
            try:
                rbf = RBFInterpolator(
                    self.coords_physical,
                    self.displacements_physical[:, dim],
                    kernel=kernel,
                    smoothing=smoothing,
                    degree=degree,
                )
                self.rbfs.append(rbf)
            except Exception as e:
                warnings.warn(f"RBF fitting failed for dimension {dim}: {e}")
                # Fallback to constant displacement
                self.rbfs.append(None)
    
    def interpolate(self, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Interpolate to dense DVF.
        
        Args:
            target_shape: (H, W, D) shape of output DVF
            
        Returns:
            dvf: (H, W, D, 3) dense displacement field in voxel space
        """
        H, W, D = target_shape
        
        # Create query grid in physical space
        i = np.arange(H)
        j = np.arange(W)
        k = np.arange(D)
        
        ii, jj, kk = np.meshgrid(i, j, k, indexing='ij')
        query_voxel = np.stack([ii.flatten(), jj.flatten(), kk.flatten()], axis=1)
        query_physical = query_voxel * self.spacing
        
        # Interpolate each dimension
        dvf_components = []
        for dim, rbf in enumerate(self.rbfs):
            if rbf is not None:
                try:
                    disp_physical = rbf(query_physical)
                    # Convert back to voxel space
                    disp_voxel = disp_physical / self.spacing[dim]
                except Exception as e:
                    warnings.warn(f"RBF interpolation failed for dimension {dim}: {e}")
                    disp_voxel = np.zeros(len(query_physical))
            else:
                # Use mean displacement as fallback
                disp_voxel = np.full(len(query_physical), self.displacements_voxel[:, dim].mean())
            
            dvf_components.append(disp_voxel.reshape(H, W, D))
        
        # Stack to form DVF
        dvf = np.stack(dvf_components, axis=-1)
        
        return dvf
    
    def __call__(self, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Alias for interpolate."""
        return self.interpolate(target_shape)


def interpolate_dvf_rbf(coords_voxel: np.ndarray,
                        displacements_voxel: np.ndarray,
                        target_shape: Tuple[int, int, int],
                        spacing: Tuple[float, float, float],
                        smoothing: float = 0.0,
                        kernel: str = 'thin_plate_spline',
                        ) -> np.ndarray:
    """
    Convenience function for RBF interpolation.
    
    Args:
        coords_voxel: (N, 3) control points
        displacements_voxel: (N, 3) displacements
        target_shape: Output shape
        spacing: Physical spacing
        smoothing: Smoothing parameter
        kernel: RBF kernel
        
    Returns:
        dvf: (H, W, D, 3) dense DVF
    """
    rbf = SpacingAwareRBF(
        coords_voxel,
        displacements_voxel,
        spacing,
        smoothing=smoothing,
        kernel=kernel,
    )
    
    return rbf.interpolate(target_shape)


def interpolate_dvf_bspline(coords_voxel: np.ndarray,
                             displacements_voxel: np.ndarray,
                             target_shape: Tuple[int, int, int],
                             spacing: Tuple[float, float, float],
                             control_point_spacing: int = 20,
                             ) -> np.ndarray:
    """
    B-spline interpolation (placeholder for future implementation).
    
    Args:
        coords_voxel: (N, 3) control points
        displacements_voxel: (N, 3) displacements
        target_shape: Output shape
        spacing: Physical spacing
        control_point_spacing: Spacing between control points
        
    Returns:
        dvf: (H, W, D, 3) dense DVF
    """
    # This is a placeholder
    # Proper B-spline FFD would use a regular grid of control points
    # and tensor product B-spline basis functions
    
    warnings.warn("B-spline interpolation not yet implemented, falling back to RBF")
    return interpolate_dvf_rbf(
        coords_voxel,
        displacements_voxel,
        target_shape,
        spacing,
        smoothing=0.1,
    )

