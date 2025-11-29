"""
Intensity normalization for medical images.

Prepares images for foundation model feature extraction.
"""

import numpy as np
from typing import Tuple, Optional
import warnings


def normalize_intensity(volume: np.ndarray,
                        method: str = 'zscore',
                        percentile_clip: Tuple[float, float] = (1, 99),
                        target_range: Optional[Tuple[float, float]] = None,
                        mask: Optional[np.ndarray] = None,
                        ) -> np.ndarray:
    """
    Robust intensity normalization for medical images.
    
    Args:
        volume: 3D numpy array
        method: Normalization method ('zscore', 'minmax', or 'percentile')
        percentile_clip: Percentiles for outlier clipping (min, max)
        target_range: Target range for scaling (e.g., (0, 1) or (-1, 1))
        mask: Optional binary mask for computing statistics (foreground only)
        
    Returns:
        Normalized volume (same shape as input)
    """
    volume = volume.astype(np.float32)
    
    # Apply mask if provided
    if mask is not None:
        foreground = volume[mask > 0]
    else:
        foreground = volume.flatten()
    
    # Remove NaN and inf
    foreground = foreground[np.isfinite(foreground)]
    
    if len(foreground) == 0:
        warnings.warn("Empty foreground after filtering, returning zeros")
        return np.zeros_like(volume)
    
    # Clip outliers based on percentiles
    if percentile_clip is not None:
        low, high = np.percentile(foreground, percentile_clip)
        volume = np.clip(volume, low, high)
        foreground = volume[mask > 0] if mask is not None else volume.flatten()
    
    # Normalize based on method
    if method == 'zscore':
        mean = foreground.mean()
        std = foreground.std()
        if std < 1e-8:
            warnings.warn("Standard deviation near zero, using mean normalization")
            volume_norm = volume - mean
        else:
            volume_norm = (volume - mean) / std
            
    elif method == 'minmax':
        vmin = foreground.min()
        vmax = foreground.max()
        if vmax - vmin < 1e-8:
            warnings.warn("Range near zero, returning zeros")
            volume_norm = np.zeros_like(volume)
        else:
            volume_norm = (volume - vmin) / (vmax - vmin)
            
    elif method == 'percentile':
        # Normalize based on percentile range
        low, high = np.percentile(foreground, percentile_clip)
        if high - low < 1e-8:
            warnings.warn("Percentile range near zero, returning zeros")
            volume_norm = np.zeros_like(volume)
        else:
            volume_norm = (volume - low) / (high - low)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Scale to target range if specified
    if target_range is not None:
        tmin, tmax = target_range
        if method == 'minmax' or method == 'percentile':
            # Already in [0, 1], scale to target
            volume_norm = volume_norm * (tmax - tmin) + tmin
        else:
            # zscore: map [-3, 3] approximately to target range
            volume_norm = np.clip(volume_norm, -3, 3)
            volume_norm = (volume_norm + 3) / 6  # Map to [0, 1]
            volume_norm = volume_norm * (tmax - tmin) + tmin
    
    return volume_norm


def create_foreground_mask(volume: np.ndarray,
                           threshold: Optional[float] = None,
                           ) -> np.ndarray:
    """
    Create a simple foreground mask for intensity normalization.
    
    Args:
        volume: 3D numpy array
        threshold: Intensity threshold (if None, uses Otsu-like method)
        
    Returns:
        Binary mask (1 = foreground, 0 = background)
    """
    if threshold is None:
        # Simple heuristic: above mean of non-zero voxels
        nonzero = volume[volume > 0]
        if len(nonzero) > 0:
            threshold = nonzero.mean() * 0.1
        else:
            threshold = 0
    
    mask = (volume > threshold).astype(np.uint8)
    return mask


def histogram_matching(source: np.ndarray,
                       reference: np.ndarray,
                       bins: int = 256,
                       ) -> np.ndarray:
    """
    Match histogram of source to reference image.
    
    Useful for normalizing intensity distributions across different scanners.
    
    Args:
        source: Source volume to transform
        reference: Reference volume
        bins: Number of histogram bins
        
    Returns:
        Transformed source volume
    """
    # Flatten arrays
    source_flat = source.flatten()
    reference_flat = reference.flatten()
    
    # Compute CDFs
    source_hist, source_bins = np.histogram(source_flat, bins=bins)
    reference_hist, reference_bins = np.histogram(reference_flat, bins=bins)
    
    source_cdf = np.cumsum(source_hist).astype(float)
    source_cdf /= source_cdf[-1]
    
    reference_cdf = np.cumsum(reference_hist).astype(float)
    reference_cdf /= reference_cdf[-1]
    
    # Interpolate
    interp_values = np.interp(source_cdf, reference_cdf, reference_bins[:-1])
    
    # Map source values
    matched = np.interp(source_flat, source_bins[:-1], interp_values)
    matched = matched.reshape(source.shape)
    
    return matched


def adaptive_normalization(volume: np.ndarray,
                           window_size: int = 64,
                           overlap: int = 32,
                           ) -> np.ndarray:
    """
    Adaptive local normalization for images with inhomogeneity.
    
    Normalizes each local window independently to handle bias fields.
    
    Args:
        volume: 3D numpy array
        window_size: Size of local window
        overlap: Overlap between windows
        
    Returns:
        Locally normalized volume
    """
    # This is a simplified implementation
    # For production, consider using N4 bias field correction from SimpleITK
    
    from scipy.ndimage import uniform_filter
    
    # Compute local mean and std
    local_mean = uniform_filter(volume.astype(float), size=window_size)
    local_var = uniform_filter(volume.astype(float)**2, size=window_size) - local_mean**2
    local_std = np.sqrt(np.maximum(local_var, 0))
    
    # Normalize
    volume_norm = (volume - local_mean) / (local_std + 1e-8)
    
    return volume_norm

