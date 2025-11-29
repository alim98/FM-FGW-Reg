"""
FM-FGW-Reg Quickstart Example

This script demonstrates basic usage of FM-FGW-Reg for medical image registration.
"""

import numpy as np

from fmfgwreg.core import FMFGWReg, RegistrationConfig
from fmfgwreg.io import load_volume, save_volume, save_dvf
from fmfgwreg.evaluation import jacobian_statistics


def main():
    print("="*60)
    print("FM-FGW-Reg Quickstart Example")
    print("="*60)
    
    # 1. Load or create configuration
    print("\n[1] Loading configuration...")
    config = RegistrationConfig()
    config.feature.device = 'cuda'  # or 'cpu'
    config.graph.num_nodes = 1000
    config.fgw.alpha = 0.5
    config.verbose = True
    
    # 2. Initialize registration system
    print("\n[2] Initializing FM-FGW-Reg...")
    reg = FMFGWReg(config)
    
    # 3. Load images
    print("\n[3] Loading images...")
    # Replace these with your actual file paths
    fixed_path = 'path/to/fixed.nii.gz'
    moving_path = 'path/to/moving.nii.gz'
    
    # For demonstration, create synthetic volumes
    print("    (Creating synthetic volumes for demo)")
    fixed = np.random.randn(64, 64, 32).astype(np.float32)
    moving = np.random.randn(64, 64, 32).astype(np.float32)
    spacing = (1.0, 1.0, 1.0)
    
    print(f"    Fixed shape: {fixed.shape}")
    print(f"    Moving shape: {moving.shape}")
    
    # 4. Run registration
    print("\n[4] Running registration...")
    result = reg.register(
        fixed, moving,
        spacing, spacing,
        do_rigid_prealign=True,
    )
    
    warped = result['warped']
    dvf = result['dvf']
    
    # 5. Print statistics
    print("\n[5] Registration statistics:")
    print(f"    Total time: {result['timing']['total']:.2f}s")
    print(f"    Valid nodes: {result['displacement_stats']['num_valid']}")
    print(f"    Mean displacement: {result['displacement_stats']['mean_magnitude_mm']:.2f} mm")
    
    # 6. Compute Jacobian statistics
    print("\n[6] Jacobian statistics:")
    jac_stats = jacobian_statistics(dvf, spacing)
    for key, val in jac_stats.items():
        print(f"    {key}: {val}")
    
    # 7. Save results
    print("\n[7] Saving results...")
    print("    (Skipping save for synthetic demo)")
    # save_volume('warped.nii.gz', warped, {'spacing': spacing})
    # save_dvf('dvf.nii.gz', dvf, {'spacing': spacing})
    
    print("\n" + "="*60)
    print("Registration complete!")
    print("="*60)


if __name__ == '__main__':
    main()

