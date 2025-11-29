"""
Medical image I/O utilities with metadata preservation.

Supports NIfTI, DICOM, and NumPy formats.
"""

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pydicom
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import warnings


def load_volume(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a medical image volume with metadata.
    
    Automatically detects format from file extension and loads appropriately.
    
    Args:
        path: Path to image file or DICOM directory
        
    Returns:
        volume: 3D numpy array (H, W, D)
        metadata: Dictionary containing:
            - spacing: (sz, sy, sx) in mm
            - origin: (oz, oy, ox) in mm
            - direction: 3x3 direction matrix
            - dtype: original data type
            - path: original file path
            
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file does not exist
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Detect format
    if path.is_dir():
        # Assume DICOM series
        return load_dicom_series(str(path))
    elif path.suffix.lower() in ['.nii', '.gz']:
        return load_nifti(str(path))
    elif path.suffix.lower() in ['.npy', '.npz']:
        return load_numpy(str(path))
    else:
        # Try loading with SimpleITK as fallback
        try:
            return load_with_sitk(str(path))
        except Exception as e:
            raise ValueError(f"Unsupported file format: {path.suffix}. Error: {e}")


def load_nifti(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load NIfTI format (.nii or .nii.gz)."""
    img = nib.load(path)
    volume = np.asarray(img.dataobj)
    
    # Get spacing from header
    header = img.header
    spacing = header.get_zooms()[:3]  # (sx, sy, sz) in NIfTI
    
    # Reorder to (sz, sy, sx) convention
    spacing = tuple(spacing[::-1])
    
    # Get affine transformation
    affine = img.affine
    origin = affine[:3, 3]
    direction = affine[:3, :3] / np.array(spacing)[::-1]
    
    metadata = {
        'spacing': spacing,
        'origin': tuple(origin),
        'direction': direction,
        'dtype': volume.dtype,
        'path': str(path),
        'affine': affine,
    }
    
    return volume, metadata


def load_dicom_series(directory: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load DICOM series from directory."""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    
    if not dicom_names:
        raise ValueError(f"No DICOM series found in {directory}")
    
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # Convert to numpy
    volume = sitk.GetArrayFromImage(image)  # (D, H, W) in SimpleITK
    volume = np.transpose(volume, (1, 2, 0))  # Convert to (H, W, D)
    
    # Get metadata
    spacing = image.GetSpacing()  # (sx, sy, sz)
    spacing = tuple(spacing[::-1])  # Convert to (sz, sy, sx)
    
    origin = image.GetOrigin()
    direction = np.array(image.GetDirection()).reshape(3, 3)
    
    metadata = {
        'spacing': spacing,
        'origin': tuple(origin),
        'direction': direction,
        'dtype': volume.dtype,
        'path': str(directory),
    }
    
    return volume, metadata


def load_numpy(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load NumPy array (.npy or .npz).
    
    For .npz files, expects keys: 'volume', optionally 'spacing', 'origin', 'direction'
    """
    path = Path(path)
    
    if path.suffix == '.npy':
        volume = np.load(path)
        metadata = {
            'spacing': (1.0, 1.0, 1.0),  # Default isotropic
            'origin': (0.0, 0.0, 0.0),
            'direction': np.eye(3),
            'dtype': volume.dtype,
            'path': str(path),
        }
    elif path.suffix == '.npz':
        data = np.load(path)
        volume = data['volume']
        metadata = {
            'spacing': tuple(data.get('spacing', np.array([1.0, 1.0, 1.0]))),
            'origin': tuple(data.get('origin', np.array([0.0, 0.0, 0.0]))),
            'direction': data.get('direction', np.eye(3)),
            'dtype': volume.dtype,
            'path': str(path),
        }
    else:
        raise ValueError(f"Unsupported numpy format: {path.suffix}")
    
    return volume, metadata


def load_with_sitk(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load image using SimpleITK as fallback."""
    image = sitk.ReadImage(path)
    volume = sitk.GetArrayFromImage(image)
    volume = np.transpose(volume, (1, 2, 0))  # (D, H, W) -> (H, W, D)
    
    spacing = image.GetSpacing()
    spacing = tuple(spacing[::-1])  # (sx, sy, sz) -> (sz, sy, sx)
    
    origin = image.GetOrigin()
    direction = np.array(image.GetDirection()).reshape(3, 3)
    
    metadata = {
        'spacing': spacing,
        'origin': tuple(origin),
        'direction': direction,
        'dtype': volume.dtype,
        'path': str(path),
    }
    
    return volume, metadata


def validate_metadata(metadata: Dict[str, Any]) -> None:
    """Validate that metadata contains required fields."""
    required = ['spacing', 'origin', 'direction', 'dtype']
    for key in required:
        if key not in metadata:
            raise ValueError(f"Missing required metadata field: {key}")
    
    # Validate spacing
    spacing = metadata['spacing']
    if len(spacing) != 3 or any(s <= 0 for s in spacing):
        raise ValueError(f"Invalid spacing: {spacing}")
    
    # Validate direction
    direction = metadata['direction']
    if direction.shape != (3, 3):
        raise ValueError(f"Invalid direction matrix shape: {direction.shape}")


def get_physical_coords(volume_shape: Tuple[int, int, int],
                        spacing: Tuple[float, float, float],
                        origin: Tuple[float, float, float] = (0., 0., 0.),
                        ) -> np.ndarray:
    """
    Generate physical coordinate grid for a volume.
    
    Args:
        volume_shape: (H, W, D)
        spacing: (sz, sy, sx) in mm
        origin: (oz, oy, ox) in mm
        
    Returns:
        coords: (H, W, D, 3) array of physical coordinates
    """
    H, W, D = volume_shape
    sz, sy, sx = spacing
    oz, oy, ox = origin
    
    # Create voxel indices
    i = np.arange(H)
    j = np.arange(W)
    k = np.arange(D)
    
    ii, jj, kk = np.meshgrid(i, j, k, indexing='ij')
    
    # Convert to physical coordinates
    coords = np.stack([
        oz + kk * sz,  # z coordinate
        oy + jj * sy,  # y coordinate
        ox + ii * sx,  # x coordinate
    ], axis=-1)
    
    return coords

