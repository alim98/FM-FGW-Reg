"""
Configuration system for FM-FGW-Reg.

YAML-loadable dataclasses for all pipeline components.
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, Dict, Any
import yaml
from pathlib import Path


@dataclass
class IntensityConfig:
    """Configuration for intensity normalization."""
    method: str = 'zscore'  # 'zscore', 'minmax', or 'percentile'
    percentile_clip: Tuple[float, float] = (1, 99)
    target_range: Optional[Tuple[float, float]] = None
    use_mask: bool = False


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    model_name: str = 'dinov2_vitb14'  # DINOv3 model variant
    device: str = 'cuda'  # 'cuda' or 'cpu'
    input_size: int = 224  # Input size for ViT (224 or 518)
    aggregation: str = 'mean'  # 'mean', 'max', or 'attention'
    num_slices: Optional[int] = None  # None = all slices


@dataclass
class GraphConfig:
    """Configuration for graph sampling."""
    num_nodes: int = 1000
    sampling_method: str = 'variance'  # 'variance', 'uniform', or 'random'
    min_spacing: float = 5.0  # Minimum distance between nodes in mm


@dataclass
class FGWConfig:
    """Configuration for Fused Gromov-Wasserstein solver."""
    alpha: float = 0.5  # Fusion weight: 0=pure structural, 1=pure feature
    epsilon: float = 0.01  # Entropic regularization
    max_iter: int = 100
    tol: float = 1e-7
    max_nodes: int = 2000  # Warning threshold
    loss_fun: str = 'square_loss'  # 'square_loss' or 'kl_loss'


@dataclass
class InterpolationConfig:
    """Configuration for DVF interpolation."""
    method: str = 'rbf'  # 'rbf', 'tps', or 'bspline'
    smoothing: float = 0.0  # RBF smoothing parameter
    rbf_kernel: str = 'thin_plate'  # 'thin_plate', 'multiquadric', etc.


@dataclass
class RegistrationConfig:
    """Main registration configuration."""
    # Sub-configs
    intensity: IntensityConfig = field(default_factory=IntensityConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    fgw: FGWConfig = field(default_factory=FGWConfig)
    interpolation: InterpolationConfig = field(default_factory=InterpolationConfig)
    
    # Global settings
    use_cache: bool = True
    cache_dir: str = '~/.fmfgwreg_cache'
    outlier_threshold: float = 0.01
    rigid_prealign: bool = True
    verbose: bool = True
    
    @classmethod
    def from_yaml(cls, path: str) -> 'RegistrationConfig':
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML config file
            
        Returns:
            RegistrationConfig instance
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse nested configs
        intensity = IntensityConfig(**data.get('intensity', {}))
        feature = FeatureConfig(**data.get('feature', {}))
        graph = GraphConfig(**data.get('graph', {}))
        fgw = FGWConfig(**data.get('fgw', {}))
        interpolation = InterpolationConfig(**data.get('interpolation', {}))
        
        # Get global settings
        global_settings = {
            k: v for k, v in data.items()
            if k not in ['intensity', 'feature', 'graph', 'fgw', 'interpolation']
        }
        
        return cls(
            intensity=intensity,
            feature=feature,
            graph=graph,
            fgw=fgw,
            interpolation=interpolation,
            **global_settings
        )
    
    def to_yaml(self, path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path: Output path
        """
        # Convert to dict
        data = {
            'intensity': asdict(self.intensity),
            'feature': asdict(self.feature),
            'graph': asdict(self.graph),
            'fgw': asdict(self.fgw),
            'interpolation': asdict(self.interpolation),
            'use_cache': self.use_cache,
            'cache_dir': self.cache_dir,
            'outlier_threshold': self.outlier_threshold,
            'rigid_prealign': self.rigid_prealign,
            'verbose': self.verbose,
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'intensity': asdict(self.intensity),
            'feature': asdict(self.feature),
            'graph': asdict(self.graph),
            'fgw': asdict(self.fgw),
            'interpolation': asdict(self.interpolation),
            'use_cache': self.use_cache,
            'cache_dir': self.cache_dir,
            'outlier_threshold': self.outlier_threshold,
            'rigid_prealign': self.rigid_prealign,
            'verbose': self.verbose,
        }


def create_default_config() -> RegistrationConfig:
    """Create default configuration."""
    return RegistrationConfig()

