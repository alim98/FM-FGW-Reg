"""
Cost matrix computation for optimal transport.

Builds feature cost and structure cost matrices.
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple


def compute_feature_cost(features_fixed: np.ndarray,
                         features_moving: np.ndarray,
                         metric: str = 'euclidean',
                         ) -> np.ndarray:
    """
    Compute pairwise feature cost matrix.
    
    Args:
        features_fixed: (Nf, C) feature vectors for fixed nodes
        features_moving: (Nm, C) feature vectors for moving nodes
        metric: Distance metric ('euclidean', 'cosine', 'sqeuclidean')
        
    Returns:
        C: (Nf, Nm) cost matrix
    """
    if metric == 'euclidean':
        C = cdist(features_fixed, features_moving, metric='euclidean')
    elif metric == 'sqeuclidean':
        C = cdist(features_fixed, features_moving, metric='sqeuclidean')
    elif metric == 'cosine':
        # Cosine distance (1 - cosine similarity)
        C = cdist(features_fixed, features_moving, metric='cosine')
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return C


def compute_structure_distance(D_fixed: np.ndarray,
                                D_moving: np.ndarray,
                                ) -> float:
    """
    Compute structural distance between two graphs (for debugging).
    
    This is related to the Gromov-Wasserstein distance.
    
    Args:
        D_fixed: (Nf, Nf) distance matrix for fixed
        D_moving: (Nm, Nm) distance matrix for moving
        
    Returns:
        Approximate structural distance
    """
    # Compute normalized eigenvalues as structure signature
    # This is a rough approximation
    
    if D_fixed.shape[0] == D_moving.shape[0]:
        # If same size, compute Frobenius norm
        return np.linalg.norm(D_fixed - D_moving, ord='fro')
    else:
        # Different sizes - compute spectral distance
        try:
            eigvals_f = np.linalg.eigvalsh(D_fixed)
            eigvals_m = np.linalg.eigvalsh(D_moving)
            
            # Pad shorter one
            k = min(len(eigvals_f), len(eigvals_m))
            eigvals_f_top = eigvals_f[-k:]
            eigvals_m_top = eigvals_m[-k:]
            
            return np.linalg.norm(eigvals_f_top - eigvals_m_top)
        except:
            return np.inf


class CostMatrixBuilder:
    """
    Helper class to build and normalize cost matrices.
    """
    
    def __init__(self,
                 feature_metric: str = 'euclidean',
                 normalize_features: bool = True,
                 normalize_distances: bool = True,
                 ):
        """
        Initialize cost matrix builder.
        
        Args:
            feature_metric: Metric for feature cost
            normalize_features: Whether to normalize feature cost
            normalize_distances: Whether to normalize distance matrices
        """
        self.feature_metric = feature_metric
        self.normalize_features = normalize_features
        self.normalize_distances = normalize_distances
    
    def build_feature_cost(self,
                          features_fixed: np.ndarray,
                          features_moving: np.ndarray,
                          ) -> np.ndarray:
        """Build and optionally normalize feature cost matrix."""
        C = compute_feature_cost(features_fixed, features_moving, self.feature_metric)
        
        if self.normalize_features:
            C = self._normalize_cost(C)
        
        return C
    
    def build_structure_matrices(self,
                                  coords_fixed: np.ndarray,
                                  coords_moving: np.ndarray,
                                  spacing_fixed: Tuple[float, float, float],
                                  spacing_moving: Tuple[float, float, float],
                                  ) -> Tuple[np.ndarray, np.ndarray]:
        """Build distance matrices for both graphs."""
        from fmfgwreg.graph.geometry import compute_distance_matrix
        
        D_f = compute_distance_matrix(coords_fixed, spacing_fixed)
        D_m = compute_distance_matrix(coords_moving, spacing_moving)
        
        if self.normalize_distances:
            D_f = self._normalize_distance(D_f)
            D_m = self._normalize_distance(D_m)
        
        return D_f, D_m
    
    @staticmethod
    def _normalize_cost(C: np.ndarray) -> np.ndarray:
        """Zero mean, unit variance normalization."""
        mean = C.mean()
        std = C.std()
        if std < 1e-8:
            return C - mean
        return (C - mean) / std
    
    @staticmethod
    def _normalize_distance(D: np.ndarray) -> np.ndarray:
        """Scale to [0, 1] range."""
        max_val = D.max()
        if max_val < 1e-8:
            return D
        return D / max_val

