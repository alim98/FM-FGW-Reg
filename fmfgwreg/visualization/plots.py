"""
Static visualization utilities using matplotlib.

Creates publication-quality plots for registration results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Tuple, Optional
import warnings


def plot_registration_overlay(fixed: np.ndarray,
                               moving: np.ndarray,
                               warped: np.ndarray,
                               slice_idx: Optional[int] = None,
                               axis: int = 2,
                               figsize: Tuple[int, int] = (15, 5),
                               save_path: Optional[str] = None,
                               ):
    """
    Plot fixed, moving, and warped images side by side.
    
    Args:
        fixed: Fixed volume
        moving: Moving volume
        warped: Warped moving volume
        slice_idx: Slice index (None = middle slice)
        axis: Slice axis (0, 1, or 2)
        figsize: Figure size
        save_path: Path to save figure
    """
    if slice_idx is None:
        slice_idx = fixed.shape[axis] // 2
    
    # Extract slices
    if axis == 0:
        fixed_slice = fixed[slice_idx, :, :]
        moving_slice = moving[slice_idx, :, :]
        warped_slice = warped[slice_idx, :, :]
    elif axis == 1:
        fixed_slice = fixed[:, slice_idx, :]
        moving_slice = moving[:, slice_idx, :]
        warped_slice = warped[:, slice_idx, :]
    else:
        fixed_slice = fixed[:, :, slice_idx]
        moving_slice = moving[:, :, slice_idx]
        warped_slice = warped[:, :, slice_idx]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    axes[0].imshow(fixed_slice, cmap='gray')
    axes[0].set_title('Fixed')
    axes[0].axis('off')
    
    axes[1].imshow(moving_slice, cmap='gray')
    axes[1].set_title('Moving')
    axes[1].axis('off')
    
    axes[2].imshow(warped_slice, cmap='gray')
    axes[2].set_title('Warped')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_checkerboard(fixed: np.ndarray,
                      warped: np.ndarray,
                      slice_idx: Optional[int] = None,
                      axis: int = 2,
                      grid_size: int = 8,
                      figsize: Tuple[int, int] = (10, 5),
                      save_path: Optional[str] = None,
                      ):
    """
    Create checkerboard visualization for registration assessment.
    
    Args:
        fixed: Fixed volume
        warped: Warped volume
        slice_idx: Slice index
        axis: Slice axis
        grid_size: Checkerboard grid size
        figsize: Figure size
        save_path: Path to save
    """
    if slice_idx is None:
        slice_idx = fixed.shape[axis] // 2
    
    # Extract slices
    if axis == 0:
        fixed_slice = fixed[slice_idx, :, :]
        warped_slice = warped[slice_idx, :, :]
    elif axis == 1:
        fixed_slice = fixed[:, slice_idx, :]
        warped_slice = warped[:, slice_idx, :]
    else:
        fixed_slice = fixed[:, :, slice_idx]
        warped_slice = warped[:, :, slice_idx]
    
    # Create checkerboard mask
    H, W = fixed_slice.shape
    mask = np.zeros((H, W), dtype=bool)
    
    for i in range(0, H, grid_size):
        for j in range(0, W, grid_size):
            if ((i // grid_size) + (j // grid_size)) % 2 == 0:
                mask[i:i+grid_size, j:j+grid_size] = True
    
    # Create checkerboard
    checker = np.where(mask, fixed_slice, warped_slice)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].imshow(fixed_slice, cmap='gray')
    axes[0].set_title('Fixed')
    axes[0].axis('off')
    
    axes[1].imshow(checker, cmap='gray')
    axes[1].set_title('Checkerboard (Fixed / Warped)')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_dvf_quiver(dvf: np.ndarray,
                    slice_idx: Optional[int] = None,
                    axis: int = 2,
                    subsample: int = 10,
                    scale: float = 1.0,
                    figsize: Tuple[int, int] = (10, 10),
                    save_path: Optional[str] = None,
                    ):
    """
    Plot deformation field as quiver plot.
    
    Args:
        dvf: (H, W, D, 3) deformation field
        slice_idx: Slice index
        axis: Slice axis
        subsample: Subsample factor for arrows
        scale: Arrow scale
        figsize: Figure size
        save_path: Save path
    """
    if slice_idx is None:
        slice_idx = dvf.shape[axis] // 2
    
    # Extract slice
    if axis == 0:
        dvf_slice = dvf[slice_idx, ::subsample, ::subsample, :]
        u = dvf_slice[:, :, 1]  # j direction
        v = dvf_slice[:, :, 0]  # i direction
    elif axis == 1:
        dvf_slice = dvf[::subsample, slice_idx, ::subsample, :]
        u = dvf_slice[:, :, 2]  # k direction
        v = dvf_slice[:, :, 0]  # i direction
    else:
        dvf_slice = dvf[::subsample, ::subsample, slice_idx, :]
        u = dvf_slice[:, :, 1]  # j direction
        v = dvf_slice[:, :, 2]  # k direction
    
    # Create grid
    H, W = u.shape
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot magnitude as background
    magnitude = np.sqrt(u**2 + v**2)
    im = ax.imshow(magnitude, cmap='hot', alpha=0.5, origin='upper')
    plt.colorbar(im, ax=ax, label='Displacement magnitude (voxels)')
    
    # Plot vectors
    ax.quiver(X, Y, u, v, scale=scale, color='cyan', alpha=0.7)
    
    ax.set_title('Deformation Field')
    ax.axis('equal')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_jacobian(dvf: np.ndarray,
                  spacing: Tuple[float, float, float],
                  slice_idx: Optional[int] = None,
                  axis: int = 2,
                  figsize: Tuple[int, int] = (10, 8),
                  save_path: Optional[str] = None,
                  ):
    """
    Plot Jacobian determinant.
    
    Args:
        dvf: Deformation field
        spacing: Physical spacing
        slice_idx: Slice index
        axis: Slice axis
        figsize: Figure size
        save_path: Save path
    """
    from fmfgwreg.evaluation import jacobian_determinant
    
    jac_det = jacobian_determinant(dvf, spacing)
    
    if slice_idx is None:
        slice_idx = jac_det.shape[axis] // 2
    
    # Extract slice
    if axis == 0:
        jac_slice = jac_det[slice_idx, :, :]
    elif axis == 1:
        jac_slice = jac_det[:, slice_idx, :]
    else:
        jac_slice = jac_det[:, :, slice_idx]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot with diverging colormap centered at 1
    im = ax.imshow(jac_slice, cmap='RdBu_r', vmin=0, vmax=2, origin='upper')
    plt.colorbar(im, ax=ax, label='Jacobian Determinant')
    
    ax.set_title('Jacobian Determinant (1 = volume preserving)')
    ax.axis('off')
    
    # Overlay folding regions
    folding = jac_slice <= 0
    if folding.any():
        ax.contour(folding, colors='yellow', linewidths=2, levels=[0.5])
        ax.text(0.02, 0.98, f'Folding: {folding.sum()} voxels',
                transform=ax.transAxes, color='yellow',
                verticalalignment='top', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

