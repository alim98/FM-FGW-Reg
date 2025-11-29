"""
Matrix normalization utilities for numerical stability in optimal transport.
"""

import numpy as np
import warnings


def normalize_cost_matrix(C: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize cost matrix for numerical stability.
    
    Args:
        C: (N, M) cost matrix
        method: 'zscore', 'minmax', or 'median'
        
    Returns:
        Normalized cost matrix
    """
    if method == 'zscore':
        mean = C.mean()
        std = C.std()
        if std < 1e-8:
            warnings.warn("Cost matrix has near-zero std, using mean normalization")
            return C - mean
        return (C - mean) / std
        
    elif method == 'minmax':
        cmin = C.min()
        cmax = C.max()
        if cmax - cmin < 1e-8:
            warnings.warn("Cost matrix has near-zero range")
            return C - cmin
        return (C - cmin) / (cmax - cmin)
        
    elif method == 'median':
        median = np.median(C)
        mad = np.median(np.abs(C - median))
        if mad < 1e-8:
            return C - median
        return (C - median) / mad
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def normalize_distance_matrix(D: np.ndarray, method: str = 'max') -> np.ndarray:
    """
    Normalize distance matrix for FGW.
    
    Args:
        D: (N, N) symmetric distance matrix
        method: 'max', 'median', or 'std'
        
    Returns:
        Normalized distance matrix
    """
    if method == 'max':
        # Scale to [0, 1] range
        max_val = D.max()
        if max_val < 1e-8:
            warnings.warn("Distance matrix has near-zero max value")
            return D
        return D / max_val
        
    elif method == 'median':
        # Scale by median non-zero distance
        non_zero = D[D > 0]
        if len(non_zero) == 0:
            return D
        median_val = np.median(non_zero)
        return D / (median_val + 1e-8)
        
    elif method == 'std':
        # Scale by standard deviation
        std_val = D.std()
        if std_val < 1e-8:
            return D
        return D / std_val
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def check_matrix_properties(matrix: np.ndarray, name: str = "matrix") -> dict:
    """
    Check properties of a matrix for debugging.
    
    Args:
        matrix: Input matrix
        name: Name for logging
        
    Returns:
        Dictionary of properties
    """
    props = {
        'name': name,
        'shape': matrix.shape,
        'dtype': matrix.dtype,
        'min': float(matrix.min()),
        'max': float(matrix.max()),
        'mean': float(matrix.mean()),
        'std': float(matrix.std()),
        'has_nan': bool(np.isnan(matrix).any()),
        'has_inf': bool(np.isinf(matrix).any()),
    }
    
    # Check for symmetry if square
    if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
        props['is_symmetric'] = bool(np.allclose(matrix, matrix.T))
        props['is_positive_definite'] = False
        try:
            np.linalg.cholesky(matrix)
            props['is_positive_definite'] = True
        except:
            pass
    
    return props

