"""
Save registered images and deformation fields with metadata preservation.
"""

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from typing import Dict, Any, Optional


def save_volume(path: str,
                volume: np.ndarray,
                metadata: Optional[Dict[str, Any]] = None,
                ) -> None:
    """
    Save a medical image volume with metadata.
    
    Format is determined by file extension.
    
    Args:
        path: Output file path
        volume: 3D numpy array (H, W, D)
        metadata: Dictionary with spacing, origin, direction, etc.
    """
    path = Path(path)
    
    if metadata is None:
        metadata = {
            'spacing': (1.0, 1.0, 1.0),
            'origin': (0.0, 0.0, 0.0),
            'direction': np.eye(3),
        }
    
    # Create output directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine format
    if path.suffix.lower() in ['.nii', '.gz']:
        save_nifti(str(path), volume, metadata)
    elif path.suffix.lower() in ['.npy', '.npz']:
        save_numpy(str(path), volume, metadata)
    else:
        # Default to SimpleITK
        save_with_sitk(str(path), volume, metadata)


def save_nifti(path: str,
               volume: np.ndarray,
               metadata: Dict[str, Any],
               ) -> None:
    """Save as NIfTI format."""
    # Get spacing (convert from (sz, sy, sx) to NIfTI (sx, sy, sz))
    spacing = metadata.get('spacing', (1.0, 1.0, 1.0))
    spacing_nifti = spacing[::-1]
    
    # Build affine matrix
    if 'affine' in metadata:
        affine = metadata['affine']
    else:
        origin = metadata.get('origin', (0.0, 0.0, 0.0))
        direction = metadata.get('direction', np.eye(3))
        
        # Construct affine
        affine = np.eye(4)
        affine[:3, :3] = direction * np.array(spacing_nifti)[::-1]
        affine[:3, 3] = origin
    
    # Create NIfTI image
    img = nib.Nifti1Image(volume, affine)
    
    # Save
    nib.save(img, path)


def save_numpy(path: str,
               volume: np.ndarray,
               metadata: Dict[str, Any],
               ) -> None:
    """Save as NumPy format (.npz with metadata)."""
    path = Path(path)
    
    if path.suffix == '.npy':
        # Just save the array
        np.save(path, volume)
    else:
        # Save with metadata as .npz
        spacing = metadata.get('spacing', (1.0, 1.0, 1.0))
        origin = metadata.get('origin', (0.0, 0.0, 0.0))
        direction = metadata.get('direction', np.eye(3))
        
        np.savez(
            path,
            volume=volume,
            spacing=np.array(spacing),
            origin=np.array(origin),
            direction=direction,
        )


def save_with_sitk(path: str,
                   volume: np.ndarray,
                   metadata: Dict[str, Any],
                   ) -> None:
    """Save using SimpleITK."""
    # Convert volume from (H, W, D) to (D, H, W) for SimpleITK
    volume_sitk = np.transpose(volume, (2, 0, 1))
    
    image = sitk.GetImageFromArray(volume_sitk)
    
    # Set spacing (convert from (sz, sy, sx) to (sx, sy, sz))
    spacing = metadata.get('spacing', (1.0, 1.0, 1.0))
    image.SetSpacing(spacing[::-1])
    
    # Set origin
    origin = metadata.get('origin', (0.0, 0.0, 0.0))
    image.SetOrigin(origin)
    
    # Set direction
    direction = metadata.get('direction', np.eye(3))
    image.SetDirection(direction.flatten())
    
    sitk.WriteImage(image, path)


def save_dvf(path: str,
             dvf: np.ndarray,
             metadata: Optional[Dict[str, Any]] = None,
             ) -> None:
    """
    Save deformation vector field.
    
    Args:
        path: Output file path
        dvf: 4D array (H, W, D, 3) representing displacement vectors
        metadata: Metadata from original image
    """
    path = Path(path)
    
    if metadata is None:
        metadata = {
            'spacing': (1.0, 1.0, 1.0),
            'origin': (0.0, 0.0, 0.0),
            'direction': np.eye(3),
        }
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix.lower() in ['.nii', '.gz']:
        # Save as 5D NIfTI (or 4D with vector dimension)
        spacing = metadata.get('spacing', (1.0, 1.0, 1.0))
        spacing_nifti = spacing[::-1]
        
        origin = metadata.get('origin', (0.0, 0.0, 0.0))
        direction = metadata.get('direction', np.eye(3))
        
        # Construct affine
        affine = np.eye(4)
        affine[:3, :3] = direction * np.array(spacing_nifti)[::-1]
        affine[:3, 3] = origin
        
        # Create NIfTI image (H, W, D, 3)
        img = nib.Nifti1Image(dvf, affine)
        nib.save(img, path)
        
    else:
        # Save as NPZ with metadata
        spacing = metadata.get('spacing', (1.0, 1.0, 1.0))
        origin = metadata.get('origin', (0.0, 0.0, 0.0))
        direction = metadata.get('direction', np.eye(3))
        
        np.savez(
            path,
            dvf=dvf,
            spacing=np.array(spacing),
            origin=np.array(origin),
            direction=direction,
        )


def load_dvf(path: str) -> np.ndarray:
    """
    Load a deformation vector field.
    
    Args:
        path: Path to DVF file
        
    Returns:
        dvf: 4D array (H, W, D, 3)
    """
    path = Path(path)
    
    if path.suffix.lower() in ['.nii', '.gz']:
        img = nib.load(path)
        dvf = np.asarray(img.dataobj)
    elif path.suffix == '.npz':
        data = np.load(path)
        dvf = data['dvf']
    else:
        raise ValueError(f"Unsupported DVF format: {path.suffix}")
    
    return dvf

