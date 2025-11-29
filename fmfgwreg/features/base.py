"""
Abstract base class for feature extractors.

Defines interface for foundation model feature extraction.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional


class FeatureExtractor(ABC):
    """
    Abstract base class for foundation model feature extractors.
    
    All feature extractors must implement the extract() method.
    """
    
    @abstractmethod
    def extract(self,
                volume: np.ndarray,
                spacing: Tuple[float, float, float],
                volume_id: Optional[str] = None,
                ) -> np.ndarray:
        """
        Extract features from a 3D medical image volume.
        
        Args:
            volume: 3D numpy array (H, W, D), intensity-normalized
            spacing: Physical spacing (sz, sy, sx) in mm
            volume_id: Optional identifier for caching
            
        Returns:
            features: 4D array (H', W', D', C) where H', W', D' are downsampled
                     by the feature extractor's stride, and C is feature dimension
        """
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return the feature dimension."""
        pass
    
    @abstractmethod
    def get_downsample_factor(self) -> int:
        """Return the spatial downsampling factor."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

