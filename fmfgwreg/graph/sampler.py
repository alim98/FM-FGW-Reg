"""
Graph node sampling from feature maps.

Converts dense feature maps into sparse graph representations.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial.distance import pdist, squareform
import warnings


def sample_graph(features: np.ndarray,
                 num_nodes: int = 1000,
                 method: str = 'variance',
                 spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 min_spacing: float = 5.0,
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample graph nodes from feature map.
    
    Args:
        features: 4D feature array (H, W, D, C)
        num_nodes: Target number of nodes
        method: Sampling method ('variance', 'uniform', or 'random')
        spacing: Physical spacing (sz, sy, sx) in mm
        min_spacing: Minimum distance between nodes in mm
        
    Returns:
        coords: (N, 3) array of node coordinates in voxel space
        node_features: (N, C) array of node features
    """
    if method == 'variance':
        return variance_sampling(features, num_nodes, spacing, min_spacing)
    elif method == 'uniform':
        return uniform_sampling(features, num_nodes)
    elif method == 'random':
        return random_sampling(features, num_nodes)
    else:
        raise ValueError(f"Unknown sampling method: {method}")


def variance_sampling(features: np.ndarray,
                      num_nodes: int,
                      spacing: Tuple[float, float, float],
                      min_spacing: float,
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample nodes based on feature variance (high information regions).
    
    Args:
        features: 4D feature array (H, W, D, C)
        num_nodes: Target number of nodes
        spacing: Physical spacing
        min_spacing: Minimum physical distance between nodes
        
    Returns:
        coords: (N, 3) coordinates in voxel space
        node_features: (N, C) features
    """
    H, W, D, C = features.shape
    
    # Compute variance at each voxel
    variance = features.var(axis=-1)  # (H, W, D)
    
    # Flatten and get importance scores
    variance_flat = variance.flatten()
    
    # Create coordinate grid
    i, j, k = np.meshgrid(range(H), range(W), range(D), indexing='ij')
    coords_all = np.stack([i.flatten(), j.flatten(), k.flatten()], axis=1)
    features_flat = features.reshape(-1, C)
    
    # Sort by variance (descending)
    sorted_indices = np.argsort(variance_flat)[::-1]
    
    # Greedily select nodes with minimum spacing constraint
    selected_indices = []
    selected_coords = []
    
    # Convert spacing to numpy array for easier computation
    spacing_arr = np.array(spacing)
    
    for idx in sorted_indices:
        coord = coords_all[idx]
        
        # Check distance to already selected nodes
        if len(selected_coords) == 0:
            selected_indices.append(idx)
            selected_coords.append(coord)
        else:
            # Compute physical distance to all selected nodes
            coord_phys = coord * spacing_arr
            selected_phys = np.array(selected_coords) * spacing_arr
            distances = np.linalg.norm(selected_phys - coord_phys, axis=1)
            
            if distances.min() >= min_spacing:
                selected_indices.append(idx)
                selected_coords.append(coord)
        
        if len(selected_indices) >= num_nodes:
            break
    
    # If we couldn't get enough nodes with spacing constraint, relax it
    if len(selected_indices) < num_nodes:
        warnings.warn(f"Could only sample {len(selected_indices)} nodes with min_spacing={min_spacing}mm")
        # Add more nodes without spacing constraint
        remaining = num_nodes - len(selected_indices)
        for idx in sorted_indices:
            if idx not in selected_indices:
                selected_indices.append(idx)
                if len(selected_indices) >= num_nodes:
                    break
    
    selected_indices = np.array(selected_indices)
    coords = coords_all[selected_indices]
    node_features = features_flat[selected_indices]
    
    return coords, node_features


def uniform_sampling(features: np.ndarray,
                     num_nodes: int,
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniform stride-based sampling.
    
    Args:
        features: 4D feature array (H, W, D, C)
        num_nodes: Target number of nodes (actual may differ slightly)
        
    Returns:
        coords: (N, 3) coordinates
        node_features: (N, C) features
    """
    H, W, D, C = features.shape
    
    # Compute stride to get approximately num_nodes
    total_voxels = H * W * D
    stride = int(np.cbrt(total_voxels / num_nodes))
    stride = max(1, stride)
    
    # Sample with stride
    i = np.arange(0, H, stride)
    j = np.arange(0, W, stride)
    k = np.arange(0, D, stride)
    
    ii, jj, kk = np.meshgrid(i, j, k, indexing='ij')
    
    coords = np.stack([ii.flatten(), jj.flatten(), kk.flatten()], axis=1)
    node_features = features[ii.flatten(), jj.flatten(), kk.flatten(), :]
    
    return coords, node_features


def random_sampling(features: np.ndarray,
                    num_nodes: int,
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random sampling of nodes.
    
    Args:
        features: 4D feature array (H, W, D, C)
        num_nodes: Number of nodes to sample
        
    Returns:
        coords: (N, 3) coordinates
        node_features: (N, C) features
    """
    H, W, D, C = features.shape
    total_voxels = H * W * D
    
    # Random indices
    indices = np.random.choice(total_voxels, size=min(num_nodes, total_voxels), replace=False)
    
    # Convert to coordinates
    coords = np.column_stack(np.unravel_index(indices, (H, W, D)))
    
    # Extract features
    features_flat = features.reshape(-1, C)
    node_features = features_flat[indices]
    
    return coords, node_features


def subsample_nodes(coords: np.ndarray,
                    features: np.ndarray,
                    target_num: int,
                    method: str = 'fps',
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsample nodes if there are too many.
    
    Args:
        coords: (N, 3) coordinates
        features: (N, C) features
        target_num: Target number of nodes
        method: 'fps' (farthest point sampling) or 'random'
        
    Returns:
        subsampled_coords: (M, 3) where M <= target_num
        subsampled_features: (M, C)
    """
    if coords.shape[0] <= target_num:
        return coords, features
    
    if method == 'fps':
        # Farthest point sampling
        selected = farthest_point_sampling(coords, target_num)
        return coords[selected], features[selected]
    else:
        # Random subsampling
        indices = np.random.choice(coords.shape[0], target_num, replace=False)
        return coords[indices], features[indices]


def farthest_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Farthest point sampling algorithm.
    
    Args:
        points: (N, D) array of points
        num_samples: Number of samples to select
        
    Returns:
        indices: Selected indices
    """
    N = points.shape[0]
    num_samples = min(num_samples, N)
    
    # Initialize with random point
    selected = [np.random.randint(N)]
    distances = np.full(N, np.inf)
    
    for _ in range(num_samples - 1):
        # Update distances
        last_point = points[selected[-1]]
        new_distances = np.linalg.norm(points - last_point, axis=1)
        distances = np.minimum(distances, new_distances)
        
        # Select farthest point
        farthest = np.argmax(distances)
        selected.append(farthest)
    
    return np.array(selected)

