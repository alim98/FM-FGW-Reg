"""
Rigid and affine registration using SimpleITK.

Pre-alignment step before deformable registration.
"""

import numpy as np
import SimpleITK as sitk
from typing import Tuple, Optional, Dict, Any
import warnings


def rigid_alignment(fixed: np.ndarray,
                    moving: np.ndarray,
                    fixed_spacing: Tuple[float, float, float],
                    moving_spacing: Tuple[float, float, float],
                    fixed_origin: Tuple[float, float, float] = (0., 0., 0.),
                    moving_origin: Tuple[float, float, float] = (0., 0., 0.),
                    metric: str = 'MI',
                    num_iterations: int = 200,
                    ) -> Tuple[np.ndarray, sitk.Transform]:
    """
    Rigid (6-DOF) registration using mutual information.
    
    Args:
        fixed: Fixed volume (H, W, D)
        moving: Moving volume (H, W, D)
        fixed_spacing: Fixed spacing (sz, sy, sx)
        moving_spacing: Moving spacing (sz, sy, sx)
        fixed_origin: Fixed origin
        moving_origin: Moving origin
        metric: 'MI' (Mattes mutual information) or 'MSE' (mean squared error)
        num_iterations: Maximum iterations
        
    Returns:
        aligned_moving: Registered moving volume
        rigid_transform: SimpleITK rigid transform
    """
    # Convert to SimpleITK images
    fixed_img = _numpy_to_sitk(fixed, fixed_spacing, fixed_origin)
    moving_img = _numpy_to_sitk(moving, moving_spacing, moving_origin)
    
    # Initialize registration
    registration = sitk.ImageRegistrationMethod()
    
    # Similarity metric
    if metric == 'MI':
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif metric == 'MSE':
        registration.SetMetricAsMeanSquares()
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.1)
    
    # Optimizer
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=num_iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    
    # Interpolator
    registration.SetInterpolator(sitk.sitkLinear)
    
    # Setup rigid transform
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_img,
        moving_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    
    registration.SetInitialTransform(initial_transform, inPlace=False)
    
    # Multi-resolution
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Execute registration
    try:
        final_transform = registration.Execute(fixed_img, moving_img)
    except RuntimeError as e:
        warnings.warn(f"Registration failed: {e}, returning identity transform")
        final_transform = sitk.Euler3DTransform()
    
    # Apply transform to moving image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)
    
    aligned_img = resampler.Execute(moving_img)
    
    # Convert back to numpy
    aligned_moving = _sitk_to_numpy(aligned_img)
    
    return aligned_moving, final_transform


def affine_alignment(fixed: np.ndarray,
                     moving: np.ndarray,
                     fixed_spacing: Tuple[float, float, float],
                     moving_spacing: Tuple[float, float, float],
                     fixed_origin: Tuple[float, float, float] = (0., 0., 0.),
                     moving_origin: Tuple[float, float, float] = (0., 0., 0.),
                     metric: str = 'MI',
                     num_iterations: int = 200,
                     ) -> Tuple[np.ndarray, sitk.Transform]:
    """
    Affine (12-DOF) registration using mutual information.
    
    Similar to rigid_alignment but allows scaling and shearing.
    
    Args:
        fixed: Fixed volume (H, W, D)
        moving: Moving volume (H, W, D)
        fixed_spacing: Fixed spacing (sz, sy, sx)
        moving_spacing: Moving spacing (sz, sy, sx)
        fixed_origin: Fixed origin
        moving_origin: Moving origin
        metric: 'MI' or 'MSE'
        num_iterations: Maximum iterations
        
    Returns:
        aligned_moving: Registered moving volume
        affine_transform: SimpleITK affine transform
    """
    # Convert to SimpleITK images
    fixed_img = _numpy_to_sitk(fixed, fixed_spacing, fixed_origin)
    moving_img = _numpy_to_sitk(moving, moving_spacing, moving_origin)
    
    # Initialize registration
    registration = sitk.ImageRegistrationMethod()
    
    # Similarity metric
    if metric == 'MI':
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif metric == 'MSE':
        registration.SetMetricAsMeanSquares()
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.1)
    
    # Optimizer
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=num_iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    
    # Interpolator
    registration.SetInterpolator(sitk.sitkLinear)
    
    # Setup affine transform (initialize with rigid first)
    initial_rigid = sitk.CenteredTransformInitializer(
        fixed_img,
        moving_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    
    # Convert to affine
    initial_transform = sitk.AffineTransform(3)
    initial_transform.SetCenter(initial_rigid.GetCenter())
    
    registration.SetInitialTransform(initial_transform, inPlace=False)
    
    # Multi-resolution
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Execute registration
    try:
        final_transform = registration.Execute(fixed_img, moving_img)
    except RuntimeError as e:
        warnings.warn(f"Registration failed: {e}, returning identity transform")
        final_transform = sitk.AffineTransform(3)
    
    # Apply transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)
    
    aligned_img = resampler.Execute(moving_img)
    
    # Convert back to numpy
    aligned_moving = _sitk_to_numpy(aligned_img)
    
    return aligned_moving, final_transform


def apply_transform(moving: np.ndarray,
                    transform: sitk.Transform,
                    reference_shape: Tuple[int, int, int],
                    moving_spacing: Tuple[float, float, float],
                    reference_spacing: Tuple[float, float, float],
                    moving_origin: Tuple[float, float, float] = (0., 0., 0.),
                    reference_origin: Tuple[float, float, float] = (0., 0., 0.),
                    interpolation: str = 'linear',
                    ) -> np.ndarray:
    """
    Apply a transform to a moving image.
    
    Args:
        moving: Moving volume
        transform: SimpleITK transform
        reference_shape: Target shape
        moving_spacing: Moving spacing
        reference_spacing: Reference spacing
        moving_origin: Moving origin
        reference_origin: Reference origin
        interpolation: Interpolation method
        
    Returns:
        Transformed moving volume
    """
    moving_img = _numpy_to_sitk(moving, moving_spacing, moving_origin)
    
    # Create reference image
    reference_img = sitk.Image(
        reference_shape[::-1],  # (D, W, H)
        sitk.sitkFloat32
    )
    reference_img.SetSpacing(reference_spacing[::-1])
    reference_img.SetOrigin(reference_origin)
    
    # Set interpolator
    if interpolation == 'linear':
        interpolator = sitk.sitkLinear
    elif interpolation == 'nearest':
        interpolator = sitk.sitkNearestNeighbor
    elif interpolation == 'bspline':
        interpolator = sitk.sitkBSpline
    else:
        interpolator = sitk.sitkLinear
    
    # Resample
    transformed = sitk.Resample(
        moving_img,
        reference_img,
        transform,
        interpolator,
        0.0,
        moving_img.GetPixelID()
    )
    
    return _sitk_to_numpy(transformed)


def _numpy_to_sitk(volume: np.ndarray,
                   spacing: Tuple[float, float, float],
                   origin: Tuple[float, float, float],
                   ) -> sitk.Image:
    """Convert numpy array to SimpleITK image."""
    # Convert (H, W, D) to (D, H, W)
    volume_sitk = np.transpose(volume, (2, 0, 1))
    image = sitk.GetImageFromArray(volume_sitk.astype(np.float32))
    
    # Set spacing and origin (convert to sx, sy, sz)
    image.SetSpacing(spacing[::-1])
    image.SetOrigin(origin)
    
    return image


def _sitk_to_numpy(image: sitk.Image) -> np.ndarray:
    """Convert SimpleITK image to numpy array."""
    volume = sitk.GetArrayFromImage(image)
    # Convert (D, H, W) to (H, W, D)
    volume = np.transpose(volume, (1, 2, 0))
    return volume


def get_transform_matrix(transform: sitk.Transform) -> np.ndarray:
    """
    Extract 4x4 transformation matrix from SimpleITK transform.
    
    Args:
        transform: SimpleITK transform
        
    Returns:
        4x4 transformation matrix
    """
    # For rigid/affine transforms
    if isinstance(transform, (sitk.Euler3DTransform, sitk.AffineTransform)):
        matrix = np.eye(4)
        
        # Get rotation/scaling part
        params = transform.GetParameters()
        if isinstance(transform, sitk.Euler3DTransform):
            # Euler angles: first 3 params
            # Translation: last 3 params
            matrix[:3, :3] = np.array(transform.GetMatrix()).reshape(3, 3)
            matrix[:3, 3] = params[3:6]
        else:
            # Affine: first 12 params
            matrix[:3, :3] = np.array(params[:9]).reshape(3, 3)
            matrix[:3, 3] = params[9:12]
        
        return matrix
    else:
        raise ValueError(f"Unsupported transform type: {type(transform)}")

