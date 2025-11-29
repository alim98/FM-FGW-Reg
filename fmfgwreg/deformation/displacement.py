"""
Node displacement calculation from coupling matrix.

Computes sparse displacements with outlier filtering.
"""

import numpy as np
from typing import Tuple
import warnings


def compute_displacements(T: np.ndarray,
                          coords_f: np.ndarray,
                          coords_m: np.ndarray,
                          outlier_threshold: float = 0.01,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute displacements with outlier filtering.
    
    For each fixed node, computes weighted average of corresponding moving nodes.
    
    Args:
        T: (Nf, Nm) coupling matrix
        coords_f: (Nf, 3) fixed node coordinates in voxel space
        coords_m: (Nm, 3) moving node coordinates in voxel space
        outlier_threshold: Minimum coupling mass to consider valid
        
    Returns:
        displacements: (Nf, 3) displacement vectors in voxel space
        valid_mask: (Nf,) boolean array (False for outliers)
    """
    Nf = coords_f.shape[0]
    
    # Compute row mass
    row_mass = T.sum(axis=1)
    valid_mask = row_mass >= outlier_threshold
    
    # Initialize displacements
    displacements = np.zeros_like(coords_f, dtype=np.float32)
    
    # Compute for valid nodes
    for i in range(Nf):
        if valid_mask[i]:
            # Weighted average of matched moving nodes
            weights = T[i, :]  # (Nm,)
            matched_pos = (weights @ coords_m) / (row_mass[i] + 1e-10)
            displacements[i] = matched_pos - coords_f[i]
    
    return displacements, valid_mask


def compute_displacements_vectorized(T: np.ndarray,
                                     coords_f: np.ndarray,
                                     coords_m: np.ndarray,
                                     outlier_threshold: float = 0.01,
                                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized version of displacement computation (faster).
    
    Args:
        T: (Nf, Nm) coupling matrix
        coords_f: (Nf, 3) fixed coordinates
        coords_m: (Nm, 3) moving coordinates
        outlier_threshold: Minimum mass threshold
        
    Returns:
        displacements: (Nf, 3) displacements
        valid_mask: (Nf,) validity mask
    """
    # Compute row mass
    row_mass = T.sum(axis=1, keepdims=True)  # (Nf, 1)
    valid_mask = (row_mass >= outlier_threshold).flatten()
    
    # Weighted average: T @ coords_m / row_mass
    matched_positions = T @ coords_m  # (Nf, 3)
    matched_positions = matched_positions / (row_mass + 1e-10)
    
    # Compute displacements
    displacements = matched_positions - coords_f
    
    # Zero out invalid displacements
    displacements[~valid_mask] = 0
    
    return displacements, valid_mask


def compute_displacement_statistics(displacements: np.ndarray,
                                     valid_mask: np.ndarray,
                                     spacing: Tuple[float, float, float],
                                     ) -> dict:
    """
    Compute statistics of displacement field.
    
    Args:
        displacements: (N, 3) displacements in voxel space
        valid_mask: (N,) validity mask
        spacing: Physical spacing for converting to mm
        
    Returns:
        Dictionary of statistics
    """
    valid_disps = displacements[valid_mask]
    
    if len(valid_disps) == 0:
        return {
            'num_valid': 0,
            'num_outliers': len(displacements),
        }
    
    # Convert to physical space
    spacing_arr = np.array(spacing)
    disps_physical = valid_disps * spacing_arr
    
    # Compute magnitudes
    magnitudes = np.linalg.norm(disps_physical, axis=1)
    
    stats = {
        'num_valid': int(valid_mask.sum()),
        'num_outliers': int((~valid_mask).sum()),
        'mean_magnitude_mm': float(magnitudes.mean()),
        'std_magnitude_mm': float(magnitudes.std()),
        'max_magnitude_mm': float(magnitudes.max()),
        'median_magnitude_mm': float(np.median(magnitudes)),
        'mean_displacement': tuple(disps_physical.mean(axis=0)),
        'std_displacement': tuple(disps_physical.std(axis=0)),
    }
    
    return stats


def smooth_displacements(displacements: np.ndarray,
                         valid_mask: np.ndarray,
                         sigma: float = 1.0,
                         ) -> np.ndarray:
    """
    Apply Gaussian smoothing to displacement field (optional preprocessing).
    
    Args:
        displacements: (N, 3) displacements
        valid_mask: (N,) validity mask
        sigma: Smoothing parameter
        
    Returns:
        Smoothed displacements
    """
    # This is a placeholder for future implementation
    # Could use Gaussian kernel on displacement magnitudes
    # or spatial smoothing based on node positions
    
    # For now, just return as-is
    return displacements

