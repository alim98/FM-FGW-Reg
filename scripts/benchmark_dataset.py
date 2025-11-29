#!/usr/bin/env python3
"""
Command-line script for benchmarking on a dataset.

Example usage:
    python benchmark_dataset.py \\
        --config configs/default.yaml \\
        --dataset-json dataset.json \\
        --output-dir results/
"""

import argparse
import json
from pathlib import Path

from fmfgwreg.core import RegistrationConfig
from fmfgwreg.evaluation import RegistrationBenchmark


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark FM-FGW-Reg on a dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset-json', required=True,
                       help='Path to JSON file with dataset pairs')
    parser.add_argument('--output-dir', required=True,
                       help='Directory to save results')
    parser.add_argument('--save-warped', action='store_true',
                       help='Save warped images and DVFs')
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        config = RegistrationConfig.from_yaml(args.config)
    else:
        print(f"Warning: Config {args.config} not found, using defaults")
        config = RegistrationConfig()
    
    # Load dataset
    with open(args.dataset_json, 'r') as f:
        pairs = json.load(f)
    
    print(f"Loaded {len(pairs)} pairs from {args.dataset_json}")
    
    # Initialize benchmark
    if args.save_warped:
        output_dir = args.output_dir
    else:
        output_dir = None
    
    benchmark = RegistrationBenchmark(config=config, output_dir=output_dir)
    
    # Run benchmark
    print(f"\nRunning benchmark...")
    results_df = benchmark.run_dataset(pairs)
    
    # Print summary
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    print(results_df[['pair_name', 'status', 'dice_mean', 'hd95', 'total']].to_string())
    
    print(f"\nFull results saved to: {args.output_dir}/results.csv")
    print(f"Summary statistics saved to: {args.output_dir}/summary.csv")


if __name__ == '__main__':
    main()

