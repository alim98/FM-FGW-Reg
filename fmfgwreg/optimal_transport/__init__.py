"""Optimal transport utilities for FM-FGW-Reg."""

from fmfgwreg.optimal_transport.fgw_solver import FGWSolver
from fmfgwreg.optimal_transport.coupling import detect_outliers, get_top_k_matches, analyze_coupling
from fmfgwreg.optimal_transport.normalization import normalize_cost_matrix, normalize_distance_matrix

__all__ = [
    'FGWSolver',
    'detect_outliers',
    'get_top_k_matches',
    'analyze_coupling',
    'normalize_cost_matrix',
    'normalize_distance_matrix',
]

