"""
Feature normalization utilities.

L2 normalization and whitening for feature vectors.
"""

import numpy as np
from typing import Optional
import warnings


def normalize_features(features: np.ndarray,
                       method: str = 'l2',
                       eps: float = 1e-8,
                       ) -> np.ndarray:
    """
    Normalize feature vectors.
    
    Args:
        features: Feature array (..., C) where C is feature dimension
        method: 'l2', 'zscore', or 'minmax'
        eps: Small constant for numerical stability
        
    Returns:
        Normalized features (same shape)
    """
    if method == 'l2':
        # L2 normalize along feature dimension
        norms = np.linalg.norm(features, axis=-1, keepdims=True)
        features_norm = features / (norms + eps)
        
    elif method == 'zscore':
        # Z-score normalize
        mean = features.mean(axis=-1, keepdims=True)
        std = features.std(axis=-1, keepdims=True)
        features_norm = (features - mean) / (std + eps)
        
    elif method == 'minmax':
        # Min-max normalize to [0, 1]
        fmin = features.min(axis=-1, keepdims=True)
        fmax = features.max(axis=-1, keepdims=True)
        features_norm = (features - fmin) / (fmax - fmin + eps)
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return features_norm


def whiten_features(features: np.ndarray,
                    regularization: float = 0.01,
                    ) -> np.ndarray:
    """
    Whiten features using ZCA whitening.
    
    Removes correlations between feature dimensions.
    
    Args:
        features: Feature array (N, C)
        regularization: Regularization parameter
        
    Returns:
        Whitened features (N, C)
    """
    # Compute covariance
    features_centered = features - features.mean(axis=0, keepdims=True)
    cov = features_centered.T @ features_centered / features.shape[0]
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Whitening matrix
    D = np.diag(1.0 / np.sqrt(eigenvalues + regularization))
    W = eigenvectors @ D @ eigenvectors.T
    
    # Apply whitening
    features_whitened = features_centered @ W
    
    return features_whitened


def pca_reduce(features: np.ndarray,
               n_components: int,
               ) -> np.ndarray:
    """
    Reduce feature dimensionality using PCA.
    
    Args:
        features: Feature array (N, C)
        n_components: Target dimension
        
    Returns:
        Reduced features (N, n_components)
    """
    # Center features
    features_centered = features - features.mean(axis=0, keepdims=True)
    
    # SVD
    U, S, Vt = np.linalg.svd(features_centered, full_matrices=False)
    
    # Project onto top components
    features_reduced = U[:, :n_components] * S[:n_components]
    
    return features_reduced

