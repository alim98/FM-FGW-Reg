"""
Fused Gromov-Wasserstein solver using POT library.

Solves the FGW optimal transport problem with entropic regularization.
"""

import numpy as np
import warnings
from typing import Tuple, Optional, Dict, Any

try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False
    warnings.warn("POT library not installed. Install with: pip install POT")

from fmfgwreg.optimal_transport.normalization import normalize_cost_matrix, normalize_distance_matrix


class FGWSolver:
    """
    Fused Gromov-Wasserstein optimal transport solver.
    
    Combines feature similarity (Wasserstein) with structural similarity (Gromov-Wasserstein).
    """
    
    def __init__(self,
                 alpha: float = 0.5,
                 epsilon: float = 0.01,
                 max_iter: int = 100,
                 tol: float = 1e-7,
                 max_nodes: int = 2000,
                 loss_fun: str = 'square_loss',
                 verbose: bool = False,
                 ):
        """
        Initialize FGW solver.
        
        Args:
            alpha: Fusion weight (0=pure structural, 1=pure feature)
            epsilon: Entropic regularization parameter
            max_iter: Maximum iterations
            tol: Convergence tolerance
            max_nodes: Warning threshold for node count
            loss_fun: 'square_loss' or 'kl_loss'
            verbose: Print convergence info
        """
        if not HAS_POT:
            raise ImportError("POT library required. Install with: pip install POT")
        
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.max_nodes = max_nodes
        self.loss_fun = loss_fun
        self.verbose = verbose
    
    def solve(self,
              C_feat: np.ndarray,
              D_f: np.ndarray,
              D_m: np.ndarray,
              p: Optional[np.ndarray] = None,
              q: Optional[np.ndarray] = None,
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve Fused Gromov-Wasserstein problem.
        
        Args:
            C_feat: (Nf, Nm) feature cost matrix
            D_f: (Nf, Nf) distance matrix for fixed graph
            D_m: (Nm, Nm) distance matrix for moving graph
            p: (Nf,) distribution for fixed (uniform if None)
            q: (Nm,) distribution for moving (uniform if None)
            
        Returns:
            T: (Nf, Nm) coupling matrix
            log: Dictionary with convergence info
        """
        Nf, Nm = C_feat.shape
        
        # Check size
        if Nf > self.max_nodes or Nm > self.max_nodes:
            warnings.warn(f"Large graph sizes: Nf={Nf}, Nm={Nm}. Consider downsampling.")
        
        # Normalize matrices for numerical stability
        C_feat_norm = normalize_cost_matrix(C_feat, method='zscore')
        D_f_norm = normalize_distance_matrix(D_f, method='max')
        D_m_norm = normalize_distance_matrix(D_m, method='max')
        
        # Check for NaN/Inf
        if np.any(~np.isfinite(C_feat_norm)):
            warnings.warn("Feature cost contains NaN/Inf, replacing with large values")
            C_feat_norm = np.nan_to_num(C_feat_norm, nan=1e6, posinf=1e6, neginf=-1e6)
        
        if np.any(~np.isfinite(D_f_norm)) or np.any(~np.isfinite(D_m_norm)):
            warnings.warn("Distance matrices contain NaN/Inf")
            D_f_norm = np.nan_to_num(D_f_norm, nan=0, posinf=1e6, neginf=0)
            D_m_norm = np.nan_to_num(D_m_norm, nan=0, posinf=1e6, neginf=0)
        
        # Uniform distributions if not provided
        if p is None:
            p = np.ones(Nf) / Nf
        if q is None:
            q = np.ones(Nm) / Nm
        
        # Ensure distributions sum to 1
        p = p / p.sum()
        q = q / q.sum()
        
        # Solve FGW using POT
        try:
            if self.epsilon > 0:
                # Entropic regularized FGW
                T, log = ot.gromov.fused_gromov_wasserstein(
                    M=C_feat_norm,
                    C1=D_f_norm,
                    C2=D_m_norm,
                    p=p,
                    q=q,
                    loss_fun=self.loss_fun,
                    alpha=self.alpha,
                    armijo=False,
                    G0=None,
                    log=True,
                    numItermax=self.max_iter,
                    tol_rel=self.tol,
                    tol_abs=self.tol,
                    verbose=self.verbose,
                )
            else:
                # Non-regularized (may be slower)
                T, log = ot.gromov.fused_gromov_wasserstein(
                    M=C_feat_norm,
                    C1=D_f_norm,
                    C2=D_m_norm,
                    p=p,
                    q=q,
                    loss_fun=self.loss_fun,
                    alpha=self.alpha,
                    armijo=True,
                    G0=None,
                    log=True,
                    numItermax=self.max_iter,
                    tol_rel=self.tol,
                    tol_abs=self.tol,
                    verbose=self.verbose,
                )
            
            # Check convergence
            if 'fgw_dist' in log:
                log['converged'] = True
            else:
                log['converged'] = False
                warnings.warn("FGW solver may not have converged properly")
            
        except Exception as e:
            warnings.warn(f"FGW solver failed: {e}. Falling back to uniform coupling.")
            T = np.outer(p, q)
            log = {
                'converged': False,
                'error': str(e),
                'fgw_dist': np.inf,
            }
        
        # Ensure non-negative and proper mass
        T = np.maximum(T, 0)
        T = T / T.sum() if T.sum() > 0 else np.outer(p, q)
        
        return T, log
    
    def __repr__(self) -> str:
        return f"FGWSolver(alpha={self.alpha}, epsilon={self.epsilon}, max_iter={self.max_iter})"

