"""
Coupling matrix analysis and outlier detection.
"""

import numpy as np
from typing import Tuple, Optional
import warnings


def detect_outliers(T: np.ndarray,
                    threshold: float = 0.01,
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect nodes with very low coupling mass (likely outliers/mismatches).
    
    Args:
        T: (Nf, Nm) coupling matrix
        threshold: Minimum row/column sum to consider valid
        
    Returns:
        outlier_mask_fixed: (Nf,) boolean array (True = outlier)
        outlier_mask_moving: (Nm,) boolean array (True = outlier)
    """
    row_mass = T.sum(axis=1)
    col_mass = T.sum(axis=0)
    
    outlier_fixed = row_mass < threshold
    outlier_moving = col_mass < threshold
    
    return outlier_fixed, outlier_moving


def get_top_k_matches(T: np.ndarray,
                      k: int = 1,
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract top-k matches from coupling matrix.
    
    Args:
        T: (Nf, Nm) coupling matrix
        k: Number of top matches per fixed node
        
    Returns:
        fixed_indices: (Nf, k) fixed node indices
        moving_indices: (Nf, k) corresponding moving node indices
        weights: (Nf, k) coupling weights
    """
    Nf, Nm = T.shape
    
    # For each row (fixed node), get top-k columns (moving nodes)
    top_k_indices = np.argsort(T, axis=1)[:, -k:][:, ::-1]  # Descending order
    
    # Get corresponding weights
    fixed_indices = np.arange(Nf)[:, None].repeat(k, axis=1)
    weights = T[fixed_indices, top_k_indices]
    
    return fixed_indices, top_k_indices, weights


def compute_coupling_entropy(T: np.ndarray) -> float:
    """
    Compute entropy of coupling matrix.
    
    Higher entropy = more diffuse matching.
    Lower entropy = more concentrated matching.
    
    Args:
        T: Coupling matrix
        
    Returns:
        Entropy value
    """
    # Normalize to probability distribution
    T_norm = T / (T.sum() + 1e-10)
    
    # Compute entropy
    T_flat = T_norm.flatten()
    T_flat = T_flat[T_flat > 0]  # Remove zeros
    entropy = -np.sum(T_flat * np.log(T_flat))
    
    return entropy


def compute_coupling_sparsity(T: np.ndarray, threshold: float = 1e-6) -> float:
    """
    Compute sparsity of coupling matrix.
    
    Args:
        T: Coupling matrix
        threshold: Values below this are considered zero
        
    Returns:
        Sparsity ratio (0 = dense, 1 = completely sparse)
    """
    num_nonzero = (T > threshold).sum()
    total = T.size
    sparsity = 1.0 - (num_nonzero / total)
    return sparsity


def analyze_coupling(T: np.ndarray) -> dict:
    """
    Comprehensive analysis of coupling matrix.
    
    Args:
        T: Coupling matrix
        
    Returns:
        Dictionary of statistics
    """
    row_mass = T.sum(axis=1)
    col_mass = T.sum(axis=0)
    
    analysis = {
        'shape': T.shape,
        'total_mass': float(T.sum()),
        'mean_coupling': float(T.mean()),
        'max_coupling': float(T.max()),
        'entropy': compute_coupling_entropy(T),
        'sparsity': compute_coupling_sparsity(T),
        'row_mass_mean': float(row_mass.mean()),
        'row_mass_std': float(row_mass.std()),
        'row_mass_min': float(row_mass.min()),
        'col_mass_mean': float(col_mass.mean()),
        'col_mass_std': float(col_mass.std()),
        'col_mass_min': float(col_mass.min()),
        'num_zero_rows': int((row_mass < 1e-6).sum()),
        'num_zero_cols': int((col_mass < 1e-6).sum()),
    }
    
    return analysis


def visualize_coupling_summary(T: np.ndarray, top_k: int = 10) -> str:
    """
    Create text summary of coupling matrix.
    
    Args:
        T: Coupling matrix
        top_k: Number of top entries to show
        
    Returns:
        Summary string
    """
    analysis = analyze_coupling(T)
    
    summary = f"""
Coupling Matrix Summary:
-----------------------
Shape: {analysis['shape']}
Total mass: {analysis['total_mass']:.4f}
Mean coupling: {analysis['mean_coupling']:.6f}
Max coupling: {analysis['max_coupling']:.6f}
Entropy: {analysis['entropy']:.4f}
Sparsity: {analysis['sparsity']:.4f}

Row (fixed) mass:
  Mean ± Std: {analysis['row_mass_mean']:.4f} ± {analysis['row_mass_std']:.4f}
  Min: {analysis['row_mass_min']:.6f}
  Zero rows: {analysis['num_zero_rows']}

Column (moving) mass:
  Mean ± Std: {analysis['col_mass_mean']:.4f} ± {analysis['col_mass_std']:.4f}
  Min: {analysis['col_mass_min']:.6f}
  Zero cols: {analysis['num_zero_cols']}
"""
    
    # Top-k entries
    flat_indices = np.argsort(T.flatten())[-top_k:][::-1]
    rows, cols = np.unravel_index(flat_indices, T.shape)
    
    summary += f"\nTop {top_k} couplings:\n"
    for i, (r, c) in enumerate(zip(rows, cols)):
        summary += f"  {i+1}. ({r}, {c}): {T[r, c]:.6f}\n"
    
    return summary

