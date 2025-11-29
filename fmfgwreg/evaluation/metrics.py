"""
Spacing-aware evaluation metrics for registration.

Includes Dice, Hausdorff Distance, Jacobian determinant, TRE, etc.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Tuple, Optional, List
import warnings


def dice_score(seg_fixed: np.ndarray,
               seg_moving_warped: np.ndarray,
               labels: Optional[List[int]] = None,
               ) -> dict:
    """
    Compute Dice coefficient for segmentation overlap.
    
    Args:
        seg_fixed: Fixed segmentation
        seg_moving_warped: Warped moving segmentation
        labels: List of label IDs to evaluate (None = all non-zero)
        
    Returns:
        Dictionary with per-label and mean Dice scores
    """
    if labels is None:
        # Find all unique labels (excluding background=0)
        labels = list(set(np.unique(seg_fixed).tolist() + np.unique(seg_moving_warped).tolist()))
        labels = [l for l in labels if l != 0]
    
    dice_scores = {}
    for label in labels:
        mask_f = (seg_fixed == label)
        mask_m = (seg_moving_warped == label)
        
        intersection = np.logical_and(mask_f, mask_m).sum()
        union = mask_f.sum() + mask_m.sum()
        
        if union == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = 2.0 * intersection / union
        
        dice_scores[f'label_{label}'] = float(dice)
    
    # Compute mean
    if len(dice_scores) > 0:
        dice_scores['mean'] = np.mean(list(dice_scores.values()))
    else:
        dice_scores['mean'] = 0.0
    
    return dice_scores


def hausdorff_distance_95(seg_fixed: np.ndarray,
                          seg_moving_warped: np.ndarray,
                          spacing: Tuple[float, float, float],
                          label: int = 1,
                          ) -> float:
    """
    Compute 95th percentile Hausdorff Distance in mm.
    
    Args:
        seg_fixed: Fixed segmentation
        seg_moving_warped: Warped moving segmentation
        spacing: Physical spacing (sz, sy, sx)
        label: Label ID to evaluate
        
    Returns:
        HD95 in mm
    """
    mask_f = (seg_fixed == label)
    mask_m = (seg_moving_warped == label)
    
    if not mask_f.any() or not mask_m.any():
        warnings.warn(f"Empty mask for label {label}, returning inf")
        return np.inf
    
    # Compute distance transforms
    dt_f = distance_transform_edt(~mask_f, sampling=spacing)
    dt_m = distance_transform_edt(~mask_m, sampling=spacing)
    
    # Distances from moving surface to fixed
    surface_m = mask_m & (dt_m <= spacing[0])
    if surface_m.any():
        distances_m_to_f = dt_f[surface_m]
    else:
        distances_m_to_f = np.array([np.inf])
    
    # Distances from fixed surface to moving
    surface_f = mask_f & (dt_f <= spacing[0])
    if surface_f.any():
        distances_f_to_m = dt_m[surface_f]
    else:
        distances_f_to_m = np.array([np.inf])
    
    # Compute 95th percentile
    all_distances = np.concatenate([distances_m_to_f, distances_f_to_m])
    hd95 = np.percentile(all_distances, 95)
    
    return float(hd95)


def jacobian_determinant(dvf: np.ndarray,
                         spacing: Tuple[float, float, float],
                         ) -> np.ndarray:
    """
    Compute Jacobian determinant of deformation field with proper spacing.
    
    Args:
        dvf: (H, W, D, 3) deformation field in voxel displacements
        spacing: (sz, sy, sx) physical spacing in mm
        
    Returns:
        jac_det: (H, W, D) Jacobian determinant at each voxel
    """
    H, W, D = dvf.shape[:3]
    
    # Compute gradients in physical space
    # dvf is in voxel space, need to convert to physical
    sz, sy, sx = spacing
    
    # Gradients: ∂u/∂x, ∂u/∂y, ∂u/∂z for each component u_i
    # np.gradient returns gradients in voxel indices, divide by spacing
    
    # Component 0 (i-direction, corresponds to z)
    grad_u0_i = np.gradient(dvf[..., 0] * sz, axis=0) / sz  # ∂u0/∂i in physical units
    grad_u0_j = np.gradient(dvf[..., 0] * sz, axis=1) / sy  # ∂u0/∂j
    grad_u0_k = np.gradient(dvf[..., 0] * sz, axis=2) / sx  # ∂u0/∂k
    
    # Component 1 (j-direction, corresponds to y)
    grad_u1_i = np.gradient(dvf[..., 1] * sy, axis=0) / sz
    grad_u1_j = np.gradient(dvf[..., 1] * sy, axis=1) / sy
    grad_u1_k = np.gradient(dvf[..., 1] * sy, axis=2) / sx
    
    # Component 2 (k-direction, corresponds to x)
    grad_u2_i = np.gradient(dvf[..., 2] * sx, axis=0) / sz
    grad_u2_j = np.gradient(dvf[..., 2] * sx, axis=1) / sy
    grad_u2_k = np.gradient(dvf[..., 2] * sx, axis=2) / sx
    
    # Jacobian matrix of deformation: J = I + ∇u
    # J = [[1 + ∂u0/∂z, ∂u0/∂y, ∂u0/∂x],
    #      [∂u1/∂z, 1 + ∂u1/∂y, ∂u1/∂x],
    #      [∂u2/∂z, ∂u2/∂y, 1 + ∂u2/∂x]]
    
    # Compute determinant
    jac_det = (
        (1 + grad_u0_i) * ((1 + grad_u1_j) * (1 + grad_u2_k) - grad_u1_k * grad_u2_j)
        - grad_u0_j * (grad_u1_i * (1 + grad_u2_k) - grad_u1_k * grad_u2_i)
        + grad_u0_k * (grad_u1_i * grad_u2_j - (1 + grad_u1_j) * grad_u2_i)
    )
    
    return jac_det


def jacobian_statistics(dvf: np.ndarray,
                        spacing: Tuple[float, float, float],
                        ) -> dict:
    """
    Compute statistics of Jacobian determinant.
    
    Args:
        dvf: Deformation field
        spacing: Physical spacing
        
    Returns:
        Dictionary of Jacobian statistics
    """
    jac_det = jacobian_determinant(dvf, spacing)
    
    stats = {
        'mean': float(jac_det.mean()),
        'std': float(jac_det.std()),
        'min': float(jac_det.min()),
        'max': float(jac_det.max()),
        'num_folding': int((jac_det <= 0).sum()),
        'percent_folding': float((jac_det <= 0).mean() * 100),
    }
    
    return stats


def target_registration_error(landmarks_fixed: np.ndarray,
                               landmarks_moving: np.ndarray,
                               dvf: np.ndarray,
                               spacing: Tuple[float, float, float],
                               ) -> dict:
    """
    Compute Target Registration Error (TRE) for landmarks.
    
    Args:
        landmarks_fixed: (N, 3) landmark coordinates in fixed image (voxel space)
        landmarks_moving: (N, 3) landmark coordinates in moving image (voxel space)
        dvf: (H, W, D, 3) deformation field
        spacing: (sz, sy, sx) physical spacing
        
    Returns:
        Dictionary with TRE statistics
    """
    N = len(landmarks_fixed)
    
    if N == 0:
        return {'mean': np.nan, 'std': np.nan, 'max': np.nan}
    
    # Convert landmarks to integers for indexing
    lm_f_int = np.round(landmarks_fixed).astype(int)
    lm_m_int = np.round(landmarks_moving).astype(int)
    
    # Apply DVF to moving landmarks
    H, W, D = dvf.shape[:3]
    errors = []
    
    for i in range(N):
        lm_m = lm_m_int[i]
        lm_f = lm_f_int[i]
        
        # Check bounds
        if (0 <= lm_m[0] < H and 0 <= lm_m[1] < W and 0 <= lm_m[2] < D):
            # Get displacement at moving landmark
            disp = dvf[lm_m[0], lm_m[1], lm_m[2]]
            
            # Warped position
            lm_m_warped = lm_m + disp
            
            # Compute error in physical space
            error_voxel = lm_m_warped - lm_f
            error_physical = error_voxel * np.array(spacing)
            error_magnitude = np.linalg.norm(error_physical)
            
            errors.append(error_magnitude)
    
    if len(errors) == 0:
        return {'mean': np.nan, 'std': np.nan, 'max': np.nan}
    
    errors = np.array(errors)
    
    tre_stats = {
        'mean': float(errors.mean()),
        'std': float(errors.std()),
        'max': float(errors.max()),
        'median': float(np.median(errors)),
        'num_landmarks': N,
    }
    
    return tre_stats


def mean_squared_error(fixed: np.ndarray,
                       warped: np.ndarray,
                       mask: Optional[np.ndarray] = None,
                       ) -> float:
    """
    Compute mean squared error between fixed and warped images.
    
    Args:
        fixed: Fixed image
        warped: Warped moving image
        mask: Optional mask for computing MSE
        
    Returns:
        MSE value
    """
    if mask is not None:
        diff = (fixed[mask > 0] - warped[mask > 0])
    else:
        diff = (fixed - warped)
    
    mse = np.mean(diff ** 2)
    return float(mse)


def normalized_cross_correlation(fixed: np.ndarray,
                                  warped: np.ndarray,
                                  mask: Optional[np.ndarray] = None,
                                  ) -> float:
    """
    Compute normalized cross-correlation.
    
    Args:
        fixed: Fixed image
        warped: Warped moving image
        mask: Optional mask
        
    Returns:
        NCC value (higher is better, 1 is perfect)
    """
    if mask is not None:
        f = fixed[mask > 0]
        w = warped[mask > 0]
    else:
        f = fixed.flatten()
        w = warped.flatten()
    
    f_mean = f.mean()
    w_mean = w.mean()
    
    f_centered = f - f_mean
    w_centered = w - w_mean
    
    numerator = np.sum(f_centered * w_centered)
    denominator = np.sqrt(np.sum(f_centered ** 2) * np.sum(w_centered ** 2))
    
    if denominator < 1e-10:
        return 0.0
    
    ncc = numerator / denominator
    return float(ncc)

