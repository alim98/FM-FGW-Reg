"""
Benchmarking utilities for dataset-level evaluation.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm
import warnings

from fmfgwreg.core import FMFGWReg, RegistrationConfig
from fmfgwreg.io import load_volume, save_volume, save_dvf
from fmfgwreg.evaluation.metrics import (
    dice_score,
    hausdorff_distance_95,
    jacobian_statistics,
    mean_squared_error,
    normalized_cross_correlation,
)


class RegistrationBenchmark:
    """
    Benchmark registration on a dataset.
    """
    
    def __init__(self,
                 config: Optional[RegistrationConfig] = None,
                 output_dir: Optional[str] = None,
                 ):
        """
        Initialize benchmark.
        
        Args:
            config: Registration configuration
            output_dir: Directory to save results
        """
        self.reg = FMFGWReg(config)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_pair(self,
                 fixed_path: str,
                 moving_path: str,
                 fixed_seg_path: Optional[str] = None,
                 moving_seg_path: Optional[str] = None,
                 pair_name: str = "pair",
                 ) -> Dict[str, Any]:
        """
        Run registration on a single pair.
        
        Args:
            fixed_path: Path to fixed image
            moving_path: Path to moving image
            fixed_seg_path: Optional path to fixed segmentation
            moving_seg_path: Optional path to moving segmentation
            pair_name: Name for this pair
            
        Returns:
            Dictionary of results
        """
        # Load images
        fixed, fixed_meta = load_volume(fixed_path)
        moving, moving_meta = load_volume(moving_path)
        
        # Register
        result = self.reg.register(
            fixed, moving,
            fixed_meta['spacing'],
            moving_meta['spacing'],
            fixed_meta.get('origin', (0., 0., 0.)),
            moving_meta.get('origin', (0., 0., 0.)),
            fixed_id=fixed_path,
            moving_id=moving_path,
        )
        
        # Compute metrics
        metrics = {}
        
        # Intensity metrics
        metrics['mse'] = mean_squared_error(fixed, result['warped'])
        metrics['ncc'] = normalized_cross_correlation(fixed, result['warped'])
        
        # Jacobian statistics
        jac_stats = jacobian_statistics(result['dvf'], fixed_meta['spacing'])
        metrics.update({f'jac_{k}': v for k, v in jac_stats.items()})
        
        # Segmentation metrics if available
        if fixed_seg_path is not None and moving_seg_path is not None:
            from fmfgwreg.deformation import warp_segmentation
            
            fixed_seg, _ = load_volume(fixed_seg_path)
            moving_seg, _ = load_volume(moving_seg_path)
            
            moving_seg_warped = warp_segmentation(moving_seg, result['dvf'])
            
            dice = dice_score(fixed_seg, moving_seg_warped)
            metrics['dice_mean'] = dice['mean']
            
            try:
                hd95 = hausdorff_distance_95(
                    fixed_seg, moving_seg_warped,
                    fixed_meta['spacing'],
                    label=1,
                )
                metrics['hd95'] = hd95
            except:
                metrics['hd95'] = np.nan
        
        # Add timing
        metrics.update(result['timing'])
        
        # Save outputs if requested
        if self.output_dir is not None:
            pair_dir = self.output_dir / pair_name
            pair_dir.mkdir(exist_ok=True)
            
            save_volume(
                str(pair_dir / 'warped.nii.gz'),
                result['warped'],
                fixed_meta,
            )
            
            save_dvf(
                str(pair_dir / 'dvf.nii.gz'),
                result['dvf'],
                fixed_meta,
            )
        
        return metrics
    
    def run_dataset(self,
                    pairs: List[Dict[str, str]],
                    ) -> pd.DataFrame:
        """
        Run on multiple pairs.
        
        Args:
            pairs: List of dictionaries with keys:
                - 'fixed': path to fixed image
                - 'moving': path to moving image
                - 'fixed_seg': (optional) path to fixed segmentation
                - 'moving_seg': (optional) path to moving segmentation
                - 'name': (optional) pair name
                
        Returns:
            DataFrame with results
        """
        results = []
        
        for i, pair in enumerate(tqdm(pairs, desc="Registering pairs")):
            pair_name = pair.get('name', f'pair_{i:03d}')
            
            try:
                metrics = self.run_pair(
                    fixed_path=pair['fixed'],
                    moving_path=pair['moving'],
                    fixed_seg_path=pair.get('fixed_seg'),
                    moving_seg_path=pair.get('moving_seg'),
                    pair_name=pair_name,
                )
                metrics['pair_name'] = pair_name
                metrics['status'] = 'success'
                results.append(metrics)
            except Exception as e:
                warnings.warn(f"Failed on {pair_name}: {e}")
                results.append({
                    'pair_name': pair_name,
                    'status': 'failed',
                    'error': str(e),
                })
        
        df = pd.DataFrame(results)
        
        # Save summary
        if self.output_dir is not None:
            df.to_csv(self.output_dir / 'results.csv', index=False)
            
            # Compute statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            summary = df[numeric_cols].describe()
            summary.to_csv(self.output_dir / 'summary.csv')
        
        return df

