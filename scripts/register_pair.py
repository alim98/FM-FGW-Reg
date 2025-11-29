#!/usr/bin/env python3
"""
Command-line script for registering a pair of medical images.

Example usage:
    python register_pair.py \\
        --config configs/default.yaml \\
        --fixed fixed.nii.gz \\
        --moving moving.nii.gz \\
        --output-warped warped.nii.gz \\
        --output-dvf dvf.nii.gz \\
        --rigid-prealign \\
        --visualize
"""

import argparse
from pathlib import Path
import sys

from fmfgwreg.core import FMFGWReg, RegistrationConfig
from fmfgwreg.io import load_volume, save_volume, save_dvf
from fmfgwreg.visualization import plot_registration_overlay, plot_jacobian


def main():
    parser = argparse.ArgumentParser(
        description='FM-FGW-Reg: Register a pair of medical images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--fixed', required=True, help='Path to fixed image')
    parser.add_argument('--moving', required=True, help='Path to moving image')
    
    # Output arguments
    parser.add_argument('--output-warped', required=True, help='Path to save warped image')
    parser.add_argument('--output-dvf', help='Path to save deformation field (optional)')
    
    # Configuration
    parser.add_argument('--config', default='configs/default.yaml',
                       help='Path to YAML configuration file')
    
    # Override options
    parser.add_argument('--rigid-prealign', action='store_true',
                       help='Enable rigid pre-alignment')
    parser.add_argument('--no-rigid-prealign', action='store_true',
                       help='Disable rigid pre-alignment')
    parser.add_argument('--alpha', type=float,
                       help='FGW fusion weight (0=structure, 1=feature)')
    parser.add_argument('--num-nodes', type=int,
                       help='Number of graph nodes')
    parser.add_argument('--device', choices=['cuda', 'cpu'],
                       help='Device for feature extraction')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--vis-dir', default='visualizations',
                       help='Directory to save visualizations')
    
    # Verbosity
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print progress information')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        config = RegistrationConfig.from_yaml(args.config)
    else:
        print(f"Warning: Config file {args.config} not found, using defaults")
        config = RegistrationConfig()
    
    # Apply command-line overrides
    if args.rigid_prealign:
        config.rigid_prealign = True
    if args.no_rigid_prealign:
        config.rigid_prealign = False
    if args.alpha is not None:
        config.fgw.alpha = args.alpha
    if args.num_nodes is not None:
        config.graph.num_nodes = args.num_nodes
    if args.device is not None:
        config.feature.device = args.device
    if args.quiet:
        config.verbose = False
    elif args.verbose:
        config.verbose = True
    
    # Initialize registration
    print("Initializing FM-FGW-Reg...")
    reg = FMFGWReg(config)
    
    # Load images
    print(f"Loading fixed image: {args.fixed}")
    fixed, fixed_meta = load_volume(args.fixed)
    
    print(f"Loading moving image: {args.moving}")
    moving, moving_meta = load_volume(args.moving)
    
    print(f"Fixed shape: {fixed.shape}, spacing: {fixed_meta['spacing']}")
    print(f"Moving shape: {moving.shape}, spacing: {moving_meta['spacing']}")
    
    # Register
    print("\nStarting registration...")
    result = reg.register(
        fixed, moving,
        fixed_meta['spacing'],
        moving_meta['spacing'],
        fixed_meta.get('origin', (0., 0., 0.)),
        moving_meta.get('origin', (0., 0., 0.)),
        fixed_id=args.fixed,
        moving_id=args.moving,
    )
    
    # Save warped image
    print(f"\nSaving warped image to: {args.output_warped}")
    save_volume(args.output_warped, result['warped'], fixed_meta)
    
    # Save DVF if requested
    if args.output_dvf:
        print(f"Saving deformation field to: {args.output_dvf}")
        save_dvf(args.output_dvf, result['dvf'], fixed_meta)
    
    # Print statistics
    print("\n" + "="*60)
    print("Registration Statistics:")
    print("="*60)
    print(f"Total time: {result['timing']['total']:.2f}s")
    print(f"  - Feature extraction: {result['timing']['feature_extraction']:.2f}s")
    print(f"  - FGW solving: {result['timing']['fgw_solving']:.2f}s")
    print(f"  - DVF interpolation: {result['timing']['dvf_interpolation']:.2f}s")
    print(f"\nDisplacement statistics:")
    stats = result['displacement_stats']
    print(f"  - Valid nodes: {stats['num_valid']}/{stats['num_valid'] + stats['num_outliers']}")
    print(f"  - Mean displacement: {stats['mean_magnitude_mm']:.2f} mm")
    print(f"  - Max displacement: {stats['max_magnitude_mm']:.2f} mm")
    
    # Generate visualizations if requested
    if args.visualize:
        print(f"\nGenerating visualizations in: {args.vis_dir}")
        vis_dir = Path(args.vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Overlay plot
        plot_registration_overlay(
            fixed, moving, result['warped'],
            save_path=str(vis_dir / 'overlay.png')
        )
        print("  - Saved: overlay.png")
        
        # Jacobian plot
        plot_jacobian(
            result['dvf'],
            fixed_meta['spacing'],
            save_path=str(vis_dir / 'jacobian.png')
        )
        print("  - Saved: jacobian.png")
    
    print("\nRegistration complete!")


if __name__ == '__main__':
    main()

