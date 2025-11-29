"""
Feature caching system for expensive foundation model extractions.

Caches features to disk to avoid repeated computation.
"""

import hashlib
import pickle
import json
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import warnings


class FeatureCache:
    """
    Disk-based cache for foundation model features.
    
    Features are saved as .npz files with accompanying metadata.
    """
    
    def __init__(self, cache_dir: str = '~/.fmfgwreg_cache'):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory to store cached features
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.features_dir = self.cache_dir / 'features'
        self.metadata_dir = self.cache_dir / 'metadata'
        self.features_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self,
                     volume_path: Optional[str] = None,
                     volume_hash: Optional[str] = None,
                     model_name: str = 'dinov2_vitb14',
                     config_dict: Optional[Dict[str, Any]] = None,
                     intensity_config: Optional[Dict[str, Any]] = None,
                     ) -> str:
        """
        Generate unique cache key.
        
        CRITICAL: Must include ALL settings that affect the extracted features,
        including intensity normalization applied BEFORE feature extraction.
        
        Args:
            volume_path: Path to volume file
            volume_hash: Pre-computed hash of volume data
            model_name: Name of foundation model
            config_dict: Feature extraction configuration
            intensity_config: Intensity normalization configuration (IMPORTANT!)
            
        Returns:
            Cache key string
        """
        # Build key components
        key_parts = [model_name]
        
        if volume_path is not None:
            # Use file path + modification time
            path = Path(volume_path)
            if path.exists():
                mtime = path.stat().st_mtime
                key_parts.append(f"{path.name}_{mtime}")
            else:
                key_parts.append(str(path.name))
        
        if volume_hash is not None:
            key_parts.append(volume_hash)
        
        # CRITICAL: Include intensity config in cache key
        # Different normalization = different input to model = different features
        if intensity_config is not None:
            intensity_str = json.dumps(intensity_config, sort_keys=True)
            intensity_hash = hashlib.md5(intensity_str.encode()).hexdigest()[:8]
            key_parts.append(f"intensity_{intensity_hash}")
        
        if config_dict is not None:
            # Hash config
            config_str = json.dumps(config_dict, sort_keys=True)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
            key_parts.append(config_hash)
        
        # Combine and hash
        key_str = '_'.join(key_parts)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]
        
        return key_hash
    
    def load(self, key: str) -> Optional[np.ndarray]:
        """
        Load cached features.
        
        Args:
            key: Cache key
            
        Returns:
            Features array if cache hit, None if cache miss
        """
        feature_path = self.features_dir / f"{key}.npz"
        
        if not feature_path.exists():
            return None
        
        try:
            data = np.load(feature_path)
            features = data['features']
            return features
        except Exception as e:
            warnings.warn(f"Failed to load cache {key}: {e}")
            return None
    
    def save(self,
             key: str,
             features: np.ndarray,
             metadata: Optional[Dict[str, Any]] = None,
             ) -> None:
        """
        Save features to cache.
        
        Args:
            key: Cache key
            features: Feature array to cache
            metadata: Optional metadata to store
        """
        feature_path = self.features_dir / f"{key}.npz"
        metadata_path = self.metadata_dir / f"{key}.json"
        
        try:
            # Save features
            np.savez_compressed(feature_path, features=features)
            
            # Save metadata
            if metadata is not None:
                with open(metadata_path, 'w') as f:
                    # Convert numpy types to Python types for JSON
                    metadata_serializable = self._make_serializable(metadata)
                    json.dump(metadata_serializable, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save cache {key}: {e}")
    
    def load_with_metadata(self, key: str) -> Optional[tuple]:
        """
        Load features and metadata.
        
        Args:
            key: Cache key
            
        Returns:
            (features, metadata) tuple if cache hit, None if cache miss
        """
        features = self.load(key)
        if features is None:
            return None
        
        metadata_path = self.metadata_dir / f"{key}.json"
        metadata = {}
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                warnings.warn(f"Failed to load metadata {key}: {e}")
        
        return features, metadata
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        feature_path = self.features_dir / f"{key}.npz"
        return feature_path.exists()
    
    def delete(self, key: str) -> None:
        """Delete cached features."""
        feature_path = self.features_dir / f"{key}.npz"
        metadata_path = self.metadata_dir / f"{key}.json"
        
        if feature_path.exists():
            feature_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
    
    def clear(self) -> None:
        """Clear all cached features."""
        for path in self.features_dir.glob("*.npz"):
            path.unlink()
        for path in self.metadata_dir.glob("*.json"):
            path.unlink()
    
    def get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        total_size = 0
        for path in self.features_dir.glob("*.npz"):
            total_size += path.stat().st_size
        return total_size
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        num_entries = len(list(self.features_dir.glob("*.npz")))
        total_size = self.get_cache_size()
        
        return {
            'num_entries': num_entries,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir),
        }
    
    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: FeatureCache._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [FeatureCache._make_serializable(v) for v in obj]
        else:
            return obj


def compute_volume_hash(volume: np.ndarray, sample_ratio: float = 0.01) -> str:
    """
    Compute a hash of volume data for cache key generation.
    
    Samples a subset of voxels for efficiency.
    
    Args:
        volume: 3D numpy array
        sample_ratio: Fraction of voxels to sample
        
    Returns:
        Hash string
    """
    # Sample voxels
    total_voxels = volume.size
    num_samples = max(1000, int(total_voxels * sample_ratio))
    
    flat_volume = volume.flatten()
    indices = np.random.choice(total_voxels, min(num_samples, total_voxels), replace=False)
    samples = flat_volume[indices]
    
    # Hash samples
    hash_obj = hashlib.sha256()
    hash_obj.update(samples.tobytes())
    hash_obj.update(str(volume.shape).encode())
    hash_obj.update(str(volume.dtype).encode())
    
    return hash_obj.hexdigest()[:16]

