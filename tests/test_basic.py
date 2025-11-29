"""
Basic unit tests for FM-FGW-Reg components.
"""

import pytest
import numpy as np
from fmfgwreg.io import load_numpy, save_numpy
from fmfgwreg.preprocessing import normalize_intensity, resample_to_spacing
from fmfgwreg.graph import sample_graph, compute_distance_matrix
from fmfgwreg.deformation import compute_displacements_vectorized, warp_volume


def test_normalize_intensity():
    """Test intensity normalization."""
    volume = np.random.randn(64, 64, 32)
    
    normalized = normalize_intensity(volume, method='zscore')
    
    assert normalized.shape == volume.shape
    assert abs(normalized.mean()) < 0.1  # Should be close to 0
    assert abs(normalized.std() - 1.0) < 0.1  # Should be close to 1


def test_distance_matrix():
    """Test distance matrix computation."""
    coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    spacing = (1.0, 1.0, 1.0)
    
    D = compute_distance_matrix(coords, spacing)
    
    assert D.shape == (3, 3)
    assert D[0, 0] == 0
    assert abs(D[0, 1] - np.sqrt(3)) < 0.01
    assert abs(D[0, 2] - np.sqrt(12)) < 0.01


def test_sample_graph():
    """Test graph sampling."""
    features = np.random.randn(32, 32, 16, 768)
    spacing = (1.0, 1.0, 1.0)
    
    coords, node_feats = sample_graph(
        features,
        num_nodes=100,
        method='uniform',
        spacing=spacing,
    )
    
    assert coords.shape[0] <= 110  # Allow some tolerance
    assert node_feats.shape[1] == 768


def test_displacements():
    """Test displacement computation."""
    # Create simple coupling matrix
    T = np.zeros((10, 10))
    for i in range(10):
        T[i, i] = 1.0  # Identity matching
    
    coords_f = np.array([[i, i, i] for i in range(10)])
    coords_m = np.array([[i+1, i+1, i+1] for i in range(10)])  # Shifted by 1
    
    displacements, valid_mask = compute_displacements_vectorized(
        T, coords_f, coords_m,
        outlier_threshold=0.5,
    )
    
    assert displacements.shape == (10, 3)
    assert np.allclose(displacements[0], [1, 1, 1])


def test_warp_volume():
    """Test volume warping."""
    volume = np.random.randn(64, 64, 32)
    dvf = np.zeros((64, 64, 32, 3))  # Zero displacement
    
    warped = warp_volume(volume, dvf)
    
    assert warped.shape == volume.shape
    assert np.allclose(warped, volume, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])

