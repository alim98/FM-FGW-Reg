"""
DINOv3 feature extractor with 2D slice-wise extraction and 3D aggregation.

Extracts features from a frozen DINOv3 ViT model.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import warnings
from pathlib import Path

from fmfgwreg.features.base import FeatureExtractor
from fmfgwreg.core.cache import FeatureCache


class DINOv2Extractor(FeatureExtractor):
    """
    DINOv3 feature extractor for 3D medical images.
    
    Processes volume slice-by-slice and aggregates features along depth.
    """
    
    def __init__(self,
                 model_name: str = 'dinov2_vitb14',
                 device: str = 'cuda',
                 input_size: int = 224,
                 aggregation: str = 'mean',
                 cache: Optional[FeatureCache] = None,
                 ):
        """
        Initialize DINOv3 extractor.
        
        Args:
            model_name: DINOv3 model variant ('dinov2_vits14', 'dinov2_vitb14', etc.)
            device: 'cuda' or 'cpu'
            input_size: Input image size (224 or 518)
            aggregation: 'mean', 'max', or 'sum' for multi-slice aggregation
            cache: Optional feature cache
        """
        self.model_name = model_name
        self.device = device
        self.input_size = input_size
        self.aggregation = aggregation
        self.cache = cache
        
        # Load model
        self._load_model()
        
        # Compute feature dimension and downsample factor
        self._compute_model_properties()
    
    def _load_model(self):
        """Load pretrained DINOv3 model."""
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            warnings.warn(f"Failed to load {self.model_name}: {e}")
            warnings.warn("Falling back to dummy model for testing")
            self.model = None
    
    def _compute_model_properties(self):
        """Compute feature dimension and downsampling factor."""
        if self.model is None:
            self.feature_dim = 768  # Default for vitb14
            self.patch_size = 14
            return
        
        # Get patch size from model
        self.patch_size = self.model.patch_embed.patch_size[0]
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
            try:
                # Try to get features
                features = self.model.forward_features(dummy_input)
                
                # DINOv2 returns a dict with keys like 'x_norm_clstoken', 'x_norm_patchtokens'
                if isinstance(features, dict):
                    # Use patch tokens (without CLS token)
                    features = features['x_norm_patchtokens']
                
                # Features from ViT are usually (B, N, D) where N is num patches
                if len(features.shape) == 3:
                    self.feature_dim = features.shape[-1]
                else:
                    self.feature_dim = features.shape[1]
            except Exception as e:
                warnings.warn(f"Could not determine feature dim: {e}, using default")
                self.feature_dim = 768
    
    def extract(self,
                volume: np.ndarray,
                spacing: Tuple[float, float, float],
                volume_id: Optional[str] = None,
                intensity_config: Optional[dict] = None,
                ) -> np.ndarray:
        """
        Extract DINOv3 features from volume.
        
        Args:
            volume: 3D array (H, W, D), intensity-normalized
            spacing: Physical spacing (sz, sy, sx)
            volume_id: Optional ID for caching
            
        Returns:
            features: 4D array (H', W', D, C) where H', W' are downsampled
        """
        # Check cache
        if self.cache is not None and volume_id is not None:
            cache_key = self.cache.get_cache_key(
                volume_path=volume_id,
                model_name=self.model_name,
                config_dict={
                    'input_size': self.input_size,
                    'aggregation': self.aggregation,
                },
                intensity_config=intensity_config,  # CRITICAL: include preprocessing
            )
            cached = self.cache.load(cache_key)
            if cached is not None:
                return cached
        
        # Extract features
        if self.model is None:
            # Dummy features for testing
            features = self._extract_dummy(volume)
        else:
            features = self._extract_dinov3(volume)
        
        # Cache features
        if self.cache is not None and volume_id is not None:
            self.cache.save(
                cache_key,
                features,
                {'spacing': spacing, 'model': self.model_name}
            )
        
        return features
    
    def _extract_dinov3(self, volume: np.ndarray) -> np.ndarray:
        """Extract features using DINOv3 model."""
        H, W, D = volume.shape
        
        # Compute output size
        H_out = self.input_size // self.patch_size
        W_out = self.input_size // self.patch_size
        
        # Process each slice
        slice_features = []
        
        with torch.no_grad():
            for d in range(D):
                slice_2d = volume[:, :, d]
                
                # Convert to tensor and resize
                slice_tensor = torch.from_numpy(slice_2d).float().unsqueeze(0).unsqueeze(0)
                slice_tensor = slice_tensor.to(self.device)
                
                # Resize to input_size x input_size
                slice_resized = F.interpolate(
                    slice_tensor,
                    size=(self.input_size, self.input_size),
                    mode='bilinear',
                    align_corners=False
                )
                
                # Convert grayscale to RGB (repeat channel 3 times)
                slice_rgb = slice_resized.repeat(1, 3, 1, 1)
                
                # Extract features
                try:
                    features = self.model.forward_features(slice_rgb)
                    
                    # DINOv2 returns a dict with keys like 'x_norm_clstoken', 'x_norm_patchtokens'
                    if isinstance(features, dict):
                        # Use patch tokens (without CLS token)
                        features = features['x_norm_patchtokens']
                    
                    # Handle different output formats
                    if len(features.shape) == 3:
                        # (B, N, D) - patch tokens only
                        # Reshape to (B, H_out, W_out, D)
                        features = features.reshape(1, H_out, W_out, -1)
                    elif len(features.shape) == 4:
                        # Already in (B, C, H, W) format
                        features = features.permute(0, 2, 3, 1)
                    
                    slice_features.append(features.cpu().numpy()[0])
                    
                except Exception as e:
                    warnings.warn(f"Feature extraction failed for slice {d}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Use dummy features
                    slice_features.append(np.zeros((H_out, W_out, self.feature_dim)))
        
        # Stack along depth: (H_out, W_out, D, C)
        features_3d = np.stack(slice_features, axis=2)
        
        return features_3d
    
    def _extract_dummy(self, volume: np.ndarray) -> np.ndarray:
        """Create dummy features for testing."""
        H, W, D = volume.shape
        
        # Downsample factor
        factor = self.patch_size
        H_out = self.input_size // factor
        W_out = self.input_size // factor
        
        # Random features
        features = np.random.randn(H_out, W_out, D, self.feature_dim).astype(np.float32)
        
        return features
    
    def get_feature_dim(self) -> int:
        """Return feature dimension."""
        return self.feature_dim
    
    def get_downsample_factor(self) -> int:
        """Return spatial downsampling factor."""
        return self.patch_size
    
    def __repr__(self) -> str:
        return f"DINOv3Extractor(model={self.model_name}, dim={self.feature_dim})"

