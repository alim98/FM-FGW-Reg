"""Evaluation metrics and benchmarking utilities."""

from fmfgwreg.evaluation.metrics import (
    dice_score,
    hausdorff_distance_95,
    jacobian_determinant,
    jacobian_statistics,
    target_registration_error,
    mean_squared_error,
    normalized_cross_correlation,
)
from fmfgwreg.evaluation.benchmarks import RegistrationBenchmark

__all__ = [
    'dice_score',
    'hausdorff_distance_95',
    'jacobian_determinant',
    'jacobian_statistics',
    'target_registration_error',
    'mean_squared_error',
    'normalized_cross_correlation',
    'RegistrationBenchmark',
]

