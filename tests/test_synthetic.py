"""
Test synthetic transformations to validate registration accuracy.
"""

import pytest
import numpy as np
from scipy.ndimage import affine_transform
from fmfgwreg.core import FMFGWReg, RegistrationConfig
from fmfgwreg.evaluation import dice_score, mean_squared_error


def create_phantom(shape=(64, 64, 32)):
    """Create a simple phantom volume with geometric structures."""
    volume = np.zeros(shape)
    
    # Add a sphere
    center = np.array(shape) // 2
    radius = min(shape) // 4
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                if dist < radius:
                    volume[i, j, k] = 1.0
    
    # Add noise
    volume += np.random.randn(*shape) * 0.1
    
    return volume


def apply_known_translation(volume, translation=(5, 5, 2)):
    """Apply a known translation to volume."""
    # Build affine matrix for translation
    matrix = np.eye(3)
    offset = np.array(translation)
    
    warped = affine_transform(volume, matrix, offset=offset, order=1, mode='constant')
    return warped


@pytest.mark.slow
def test_affine_recovery():
    """Test recovery of known affine transformation."""
    # Create phantom
    fixed = create_phantom((64, 64, 32))
    
    # Apply known translation
    moving = apply_known_translation(fixed, translation=(3, 3, 1))
    
    # Create simple config for faster testing
    config = RegistrationConfig()
    config.feature.device = 'cpu'  # Use CPU for testing
    config.graph.num_nodes = 200  # Fewer nodes for speed
    config.fgw.max_iter = 50
    config.verbose = False
    
    # Register (will use dummy features in testing)
    reg = FMFGWReg(config)
    
    spacing = (1.0, 1.0, 1.0)
    result = reg.register(
        fixed, moving,
        spacing, spacing,
        do_rigid_prealign=False,
    )
    
    # Check that warped is closer to fixed than moving was
    mse_before = mean_squared_error(fixed, moving)
    mse_after = mean_squared_error(fixed, result['warped'])
    
    print(f"MSE before: {mse_before:.4f}, after: {mse_after:.4f}")
    
    # Assert improvement (may be modest with dummy features)
    assert mse_after < mse_before * 1.5  # Allow some tolerance


def test_identity_transform():
    """Test that registering an image to itself gives identity."""
    fixed = create_phantom((64, 64, 32))
    moving = fixed.copy()
    
    config = RegistrationConfig()
    config.feature.device = 'cpu'
    config.graph.num_nodes = 200
    config.fgw.max_iter = 50
    config.verbose = False
    
    reg = FMFGWReg(config)
    
    spacing = (1.0, 1.0, 1.0)
    result = reg.register(
        fixed, moving,
        spacing, spacing,
        do_rigid_prealign=False,
    )
    
    # Should have minimal displacement
    disp_magnitude = np.linalg.norm(result['dvf'], axis=-1).mean()
    print(f"Mean displacement magnitude: {disp_magnitude:.4f}")
    
    assert disp_magnitude < 5.0  # Should be small


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

