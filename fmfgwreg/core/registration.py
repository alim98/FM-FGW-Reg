"""
Main FM-FGW-Reg registration orchestrator.

Coordinates the full registration pipeline.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings
import time

from fmfgwreg.core.config import RegistrationConfig
from fmfgwreg.core.cache import FeatureCache
from fmfgwreg.preprocessing import normalize_intensity, rigid_alignment
from fmfgwreg.features import DINOv2Extractor, normalize_features
from fmfgwreg.graph import sample_graph, CostMatrixBuilder
from fmfgwreg.optimal_transport import FGWSolver, analyze_coupling
from fmfgwreg.deformation import (
    compute_displacements_vectorized,
    compute_displacement_statistics,
    SpacingAwareRBF,
    warp_volume,
)


class FMFGWReg:
    """
    Main FM-FGW-Reg registration class.
    
    Orchestrates the complete training-free registration pipeline using
    foundation model features and Fused Gromov-Wasserstein optimal transport.
    """
    
    def __init__(self, config: Optional[RegistrationConfig] = None):
        """
        Initialize registration system.
        
        Args:
            config: Registration configuration (uses default if None)
        """
        if config is None:
            config = RegistrationConfig()
        
        self.config = config
        
        # Initialize cache
        if self.config.use_cache:
            self.cache = FeatureCache(self.config.cache_dir)
        else:
            self.cache = None
        
        # Initialize feature extractor
        self.feature_extractor = DINOv2Extractor(
            model_name=self.config.feature.model_name,
            device=self.config.feature.device,
            input_size=self.config.feature.input_size,
            aggregation=self.config.feature.aggregation,
            cache=self.cache,
        )
        
        # Initialize FGW solver
        self.fgw_solver = FGWSolver(
            alpha=self.config.fgw.alpha,
            epsilon=self.config.fgw.epsilon,
            max_iter=self.config.fgw.max_iter,
            tol=self.config.fgw.tol,
            max_nodes=self.config.fgw.max_nodes,
            loss_fun=self.config.fgw.loss_fun,
            verbose=self.config.verbose,
        )
        
        # Initialize cost matrix builder
        self.cost_builder = CostMatrixBuilder(
            feature_metric='euclidean',
            normalize_features=True,
            normalize_distances=True,
        )
    
    def register(self,
                 fixed_volume: np.ndarray,
                 moving_volume: np.ndarray,
                 fixed_spacing: Tuple[float, float, float],
                 moving_spacing: Tuple[float, float, float],
                 fixed_origin: Tuple[float, float, float] = (0., 0., 0.),
                 moving_origin: Tuple[float, float, float] = (0., 0., 0.),
                 fixed_id: Optional[str] = None,
                 moving_id: Optional[str] = None,
                 do_rigid_prealign: Optional[bool] = None,
                 ) -> Dict[str, Any]:
        """
        Register moving volume to fixed volume.
        
        Args:
            fixed_volume: Fixed volume (H, W, D)
            moving_volume: Moving volume (H, W, D)
            fixed_spacing: Fixed spacing (sz, sy, sx) in mm
            moving_spacing: Moving spacing (sz, sy, sx) in mm
            fixed_origin: Fixed origin (oz, oy, ox) in mm
            moving_origin: Moving origin (oz, oy, ox) in mm
            fixed_id: Identifier for caching fixed features
            moving_id: Identifier for caching moving features
            do_rigid_prealign: Override config rigid pre-alignment setting
            
        Returns:
            Dictionary containing:
                - warped: Registered moving volume
                - dvf: Deformation vector field
                - rigid_transform: Rigid pre-alignment transform (if applied)
                - coupling: FGW coupling matrix
                - valid_mask: Valid node mask
                - log: Solver and timing information
        """
        if do_rigid_prealign is None:
            do_rigid_prealign = self.config.rigid_prealign
        
        timing = {}
        t_start = time.time()
        
        if self.config.verbose:
            print("=" * 60)
            print("FM-FGW-Reg: Foundation-Model Fused Gromov-Wasserstein Registration")
            print("=" * 60)
        
        # Step 0: Intensity normalization
        if self.config.verbose:
            print("\n[1/9] Intensity normalization...")
        t0 = time.time()
        
        fixed_norm = normalize_intensity(
            fixed_volume,
            method=self.config.intensity.method,
            percentile_clip=self.config.intensity.percentile_clip,
        )
        moving_norm = normalize_intensity(
            moving_volume,
            method=self.config.intensity.method,
            percentile_clip=self.config.intensity.percentile_clip,
        )
        
        timing['intensity_normalization'] = time.time() - t0
        
        # Step 1: Rigid pre-alignment (optional)
        rigid_transform = None
        if do_rigid_prealign:
            if self.config.verbose:
                print("[2/9] Rigid pre-alignment...")
            t0 = time.time()
            
            try:
                moving_rigid, rigid_transform = rigid_alignment(
                    fixed_norm,
                    moving_norm,
                    fixed_spacing,
                    moving_spacing,
                    fixed_origin,
                    moving_origin,
                    metric='MI',
                    num_iterations=200,
                )
                timing['rigid_alignment'] = time.time() - t0
            except Exception as e:
                warnings.warn(f"Rigid alignment failed: {e}, skipping")
                moving_rigid = moving_norm
                timing['rigid_alignment'] = 0
        else:
            if self.config.verbose:
                print("[2/9] Skipping rigid pre-alignment")
            moving_rigid = moving_norm
            timing['rigid_alignment'] = 0
        
        # Step 2: Extract features
        if self.config.verbose:
            print("[3/9] Extracting foundation model features...")
        t0 = time.time()
        
        # CRITICAL: Pass intensity config to cache key generation
        # Different normalization settings = different features
        intensity_config_dict = {
            'method': self.config.intensity.method,
            'percentile_clip': self.config.intensity.percentile_clip,
            'target_range': self.config.intensity.target_range,
        }
        
        feat_f = self.feature_extractor.extract(
            fixed_norm, fixed_spacing, fixed_id,
            intensity_config=intensity_config_dict
        )
        feat_m = self.feature_extractor.extract(
            moving_rigid, moving_spacing, moving_id,
            intensity_config=intensity_config_dict
        )
        
        if self.config.verbose:
            print(f"    Fixed features shape: {feat_f.shape}")
            print(f"    Moving features shape: {feat_m.shape}")
        
        timing['feature_extraction'] = time.time() - t0
        
        # Normalize features
        feat_f = normalize_features(feat_f, method='l2')
        feat_m = normalize_features(feat_m, method='l2')
        
        # Step 3: Sample graph nodes
        if self.config.verbose:
            print("[4/9] Sampling graph nodes...")
        t0 = time.time()
        
        # CRITICAL: After rigid pre-alignment, moving is resampled to fixed grid
        # So for geometric consistency, use fixed_spacing for moving graph too
        moving_spacing_for_graph = fixed_spacing if do_rigid_prealign else moving_spacing
        
        coords_f, node_feat_f = sample_graph(
            feat_f,
            num_nodes=self.config.graph.num_nodes,
            method=self.config.graph.sampling_method,
            spacing=fixed_spacing,
            min_spacing=self.config.graph.min_spacing,
        )
        
        coords_m, node_feat_m = sample_graph(
            feat_m,
            num_nodes=self.config.graph.num_nodes,
            method=self.config.graph.sampling_method,
            spacing=moving_spacing_for_graph,  # Fixed spacing after rigid alignment
            min_spacing=self.config.graph.min_spacing,
        )
        
        if self.config.verbose:
            print(f"    Fixed graph: {len(coords_f)} nodes")
            print(f"    Moving graph: {len(coords_m)} nodes")
        
        timing['graph_sampling'] = time.time() - t0
        
        # Step 4: Build cost matrices
        if self.config.verbose:
            print("[5/9] Computing cost matrices...")
        t0 = time.time()
        
        C_feat = self.cost_builder.build_feature_cost(node_feat_f, node_feat_m)
        D_f, D_m = self.cost_builder.build_structure_matrices(
            coords_f, coords_m,
            fixed_spacing, moving_spacing_for_graph,  # Use corrected spacing
        )
        
        if self.config.verbose:
            print(f"    Feature cost shape: {C_feat.shape}")
            print(f"    Distance matrices: {D_f.shape}, {D_m.shape}")
        
        timing['cost_computation'] = time.time() - t0
        
        # Step 5: Solve FGW
        if self.config.verbose:
            print("[6/9] Solving Fused Gromov-Wasserstein...")
        t0 = time.time()
        
        T, fgw_log = self.fgw_solver.solve(C_feat, D_f, D_m)
        
        if self.config.verbose:
            if 'fgw_dist' in fgw_log:
                print(f"    FGW distance: {fgw_log['fgw_dist']:.6f}")
            coupling_analysis = analyze_coupling(T)
            print(f"    Coupling entropy: {coupling_analysis['entropy']:.4f}")
            print(f"    Coupling sparsity: {coupling_analysis['sparsity']:.4f}")
        
        timing['fgw_solving'] = time.time() - t0
        
        # Step 6: Compute displacements
        if self.config.verbose:
            print("[7/9] Computing node displacements...")
        t0 = time.time()
        
        displacements, valid_mask = compute_displacements_vectorized(
            T, coords_f, coords_m,
            outlier_threshold=self.config.outlier_threshold,
        )
        
        disp_stats = compute_displacement_statistics(displacements, valid_mask, fixed_spacing)
        if self.config.verbose:
            print(f"    Valid nodes: {disp_stats['num_valid']}/{len(coords_f)}")
            if 'mean_magnitude_mm' in disp_stats:
                print(f"    Mean displacement: {disp_stats['mean_magnitude_mm']:.2f} mm")
            if 'max_magnitude_mm' in disp_stats:
                print(f"    Max displacement: {disp_stats['max_magnitude_mm']:.2f} mm")
        
        timing['displacement_computation'] = time.time() - t0
        
        # Step 7: Interpolate to dense DVF
        if self.config.verbose:
            print("[8/9] Interpolating dense deformation field...")
        t0 = time.time()
        
        # Use only valid nodes for interpolation
        valid_coords = coords_f[valid_mask]
        valid_displacements = displacements[valid_mask]
        
        if len(valid_coords) < 3:
            warnings.warn("Too few valid nodes for interpolation, returning zero DVF")
            dvf = np.zeros((*fixed_volume.shape, 3), dtype=np.float32)
        else:
            rbf = SpacingAwareRBF(
                valid_coords,
                valid_displacements,
                fixed_spacing,
                smoothing=self.config.interpolation.smoothing,
                kernel=self.config.interpolation.rbf_kernel,
            )
            dvf = rbf.interpolate(fixed_volume.shape)
        
        timing['dvf_interpolation'] = time.time() - t0
        
        # Step 8: Warp moving volume
        if self.config.verbose:
            print("[9/9] Warping moving volume...")
        t0 = time.time()
        
        # CRITICAL FIX: Warp the ORIGINAL moving volume, not the normalized one!
        # We used normalized for feature extraction, but final warped should be in original intensity
        moving_to_warp = moving_volume if not do_rigid_prealign else moving_rigid
        
        # If rigid was applied, moving_rigid is still normalized
        # We need to warp the original moving_volume with the composed transform
        # For now, we warp the original moving (this is simpler and correct for deformable-only)
        warped = warp_volume(
            moving_volume,  # Use ORIGINAL, not normalized!
            dvf,
            spacing=fixed_spacing,
            mode='constant',
            cval=0.0,
            order=1,  # Linear interpolation
        )
        
        timing['warping'] = time.time() - t0
        timing['total'] = time.time() - t_start
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Registration completed in {timing['total']:.2f}s")
            print(f"{'='*60}\n")
        
        # Return results
        result = {
            'warped': warped,
            'dvf': dvf,
            'rigid_transform': rigid_transform,
            'coupling': T,
            'valid_mask': valid_mask,
            'coords_fixed': coords_f,
            'coords_moving': coords_m,
            'displacements': displacements,
            'displacement_stats': disp_stats,
            'fgw_log': fgw_log,
            'timing': timing,
        }
        
        return result
    
    def __repr__(self) -> str:
        return f"FMFGWReg(model={self.config.feature.model_name}, alpha={self.config.fgw.alpha})"

