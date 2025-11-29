#!/usr/bin/env python3
"""
تست سریع فقط برای RBF و DINOv2
"""

import numpy as np
import sys

print("="*60)
print("تست RBF + DINOv2")
print("="*60)

# Test 1: RBF Kernel
print("\n[1/2] تست RBF kernel...")
try:
    from scipy.interpolate import RBFInterpolator
    from fmfgwreg.deformation import SpacingAwareRBF
    from fmfgwreg.core.config import InterpolationConfig
    
    # Check default config
    config = InterpolationConfig()
    print(f"   Default kernel: '{config.rbf_kernel}'")
    
    # Test RBF with correct kernel
    coords = np.random.rand(10, 3) * 10
    displacements = np.random.rand(10, 3) * 2
    spacing = (1.0, 1.0, 1.0)
    
    rbf = SpacingAwareRBF(
        coords, 
        displacements, 
        spacing,
        kernel='thin_plate_spline'
    )
    
    dvf = rbf.interpolate((20, 20, 10))
    print(f"   DVF shape: {dvf.shape}")
    print(f"   DVF mean magnitude: {np.mean(np.linalg.norm(dvf, axis=-1)):.3f}")
    
    if np.mean(np.linalg.norm(dvf, axis=-1)) > 0.01:
        print("   ✅ RBF کار میکنه (DVF non-zero)")
    else:
        print("   ❌ RBF هنوز مشکل داره")
        
except Exception as e:
    print(f"   ❌ خطا: {e}")
    import traceback
    traceback.print_exc()

# Test 2: DINOv2 feature extraction
print("\n[2/2] تست DINOv2...")
try:
    import torch
    from fmfgwreg.features import DINOv2Extractor
    
    # Create small volume
    volume = np.random.rand(32, 32, 16).astype(np.float32)
    spacing = (1.0, 1.0, 1.0)
    
    extractor = DINOv2Extractor(
        model_name='dinov2_vitb14',
        device='cpu',
        input_size=224,
    )
    
    print(f"   Feature dim: {extractor.feature_dim}")
    
    # Extract features from 2 slices only for speed
    features = extractor.extract(
        volume[:, :, :2],  # Only 2 slices
        spacing,
    )
    
    print(f"   Features shape: {features.shape}")
    print(f"   Features mean: {np.mean(features):.6f}")
    print(f"   Features std: {np.std(features):.6f}")
    
    if features.shape[-1] == extractor.feature_dim and np.std(features) > 0.001:
        print("   ✅ DINOv2 کار میکنه")
    else:
        print("   ❌ DINOv2 مشکل داره")
        
except Exception as e:
    print(f"   ❌ خطا: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("تست تموم شد")
print("="*60)

