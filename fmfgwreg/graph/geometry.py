"""
Geometric computations for graph nodes.

Computes pairwise distance matrices in physical space.
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple, Optional


def compute_distance_matrix(coords_voxel: np.ndarray,
                            spacing: Tuple[float, float, float],
                            ) -> np.ndarray:
    """
    Compute pairwise distance matrix in physical space.
    
    Args:
        coords_voxel: (N, 3) array of coordinates in voxel indices (i, j, k)
        spacing: (sz, sy, sx) physical spacing in mm
        
    Returns:
        D: (N, N) pairwise distance matrix in mm
    """
    # Convert to physical coordinates
    spacing_arr = np.array(spacing)  # (sz, sy, sx)
    coords_physical = coords_voxel * spacing_arr  # Element-wise multiplication
    
    # Compute pairwise Euclidean distances
    D = cdist(coords_physical, coords_physical, metric='euclidean')
    
    return D


def compute_geodesic_distance_matrix(coords_voxel: np.ndarray,
                                     spacing: Tuple[float, float, float],
                                     intensity: Optional[np.ndarray] = None,
                                     ) -> np.ndarray:
    """
    Compute approximate geodesic distances (for future use).
    
    For now, falls back to Euclidean distance.
    In future, could use fast marching or similar.
    
    Args:
        coords_voxel: (N, 3) coordinates
        spacing: Physical spacing
        intensity: Optional intensity volume for weighted geodesic
        
    Returns:
        D: (N, N) distance matrix
    """
    # TODO: Implement geodesic distance using skfmm or similar
    # For now, use Euclidean
    return compute_distance_matrix(coords_voxel, spacing)


def compute_angle_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute pairwise angles between point vectors from origin.
    
    Can be used as additional geometric feature.
    
    Args:
        coords: (N, 3) coordinates
        
    Returns:
        A: (N, N) angle matrix in radians
    """
    # Normalize vectors
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    coords_normalized = coords / (norms + 1e-8)
    
    # Compute dot products
    dot_products = coords_normalized @ coords_normalized.T
    
    # Clip to valid range for arccos
    dot_products = np.clip(dot_products, -1.0, 1.0)
    
    # Compute angles
    angles = np.arccos(dot_products)
    
    return angles


def compute_local_density(coords_voxel: np.ndarray,
                          spacing: Tuple[float, float, float],
                          radius: float = 10.0,
                          ) -> np.ndarray:
    """
    Compute local point density around each node.
    
    Args:
        coords_voxel: (N, 3) coordinates
        spacing: Physical spacing
        radius: Search radius in mm
        
    Returns:
        density: (N,) array of local densities
    """
    D = compute_distance_matrix(coords_voxel, spacing)
    
    # Count points within radius
    density = (D < radius).sum(axis=1) - 1  # Subtract self
    
    return density


def compute_graph_statistics(coords_voxel: np.ndarray,
                             spacing: Tuple[float, float, float],
                             ) -> dict:
    """
    Compute various statistics of the graph node distribution.
    
    Args:
        coords_voxel: (N, 3) coordinates
        spacing: Physical spacing
        
    Returns:
        Dictionary of statistics
    """
    D = compute_distance_matrix(coords_voxel, spacing)
    
    # Nearest neighbor distances (excluding self)
    np.fill_diagonal(D, np.inf)
    nn_distances = D.min(axis=1)
    
    stats = {
        'num_nodes': len(coords_voxel),
        'mean_nn_distance': nn_distances.mean(),
        'std_nn_distance': nn_distances.std(),
        'min_nn_distance': nn_distances.min(),
        'max_nn_distance': nn_distances.max(),
        'mean_distance': D[D < np.inf].mean(),
        'graph_diameter': D[D < np.inf].max(),
    }
    
    return stats

