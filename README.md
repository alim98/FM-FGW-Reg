# FM-FGW-Reg: Foundation-Model Fused Gromov-Wasserstein Registration

**A Training-Free 3D Medical Image Registration Framework**

## Overview

FM-FGW-Reg is a novel medical image registration system that combines:
- **Frozen foundation models** (DINOv3) for semantic feature extraction
- **Fused Gromov-Wasserstein optimal transport** for structural matching
- **Training-free operation** - no neural network training required

Unlike traditional feature-based registration (e.g., DINO-Reg), FM-FGW-Reg performs **global structural matching** by representing images as graphs and solving an optimal transport problem that considers both feature similarity and geometric structure.

## Key Features

- ✅ **Training-free**: No dataset-level optimization or learning
- ✅ **Spacing-aware**: Proper handling of physical coordinates and anisotropic voxels
- ✅ **Robust**: Rigid pre-alignment and outlier detection
- ✅ **Cacheable**: Feature extraction results cached for efficiency
- ✅ **Numerically stable**: Proper normalization throughout pipeline

## Installation

```bash
# Clone repository
git clone https://github.com/alim98/fmfgwreg.git
cd fmfgwreg

# Install dependencies
pip install -e .
```

## Quick Start

```python
from fmfgwreg.core.registration import FMFGWReg
from fmfgwreg.core.config import RegistrationConfig
from fmfgwreg.io.loaders import load_volume

# Load configuration
config = RegistrationConfig.from_yaml('configs/default.yaml')

# Initialize registration
reg = FMFGWReg(config)

# Load images
fixed, fixed_meta = load_volume('fixed.nii.gz')
moving, moving_meta = load_volume('moving.nii.gz')

# Register
result = reg.register(
    fixed, moving,
    fixed_meta['spacing'], 
    moving_meta['spacing'],
    do_rigid_prealign=True
)

# Access results
warped = result['warped']
dvf = result['dvf']
```

## Command-Line Usage

```bash
python scripts/register_pair.py \
  --config configs/default.yaml \
  --fixed fixed.nii.gz \
  --moving moving.nii.gz \
  --output-warped warped.nii.gz \
  --output-dvf dvf.nii.gz \
  --rigid-prealign
```

## Pipeline Overview

1. **Preprocessing**: Intensity normalization + rigid pre-alignment (SimpleITK)
2. **Feature Extraction**: DINOv3 features extracted slice-wise (with caching)
3. **Graph Construction**: Downsample features to graph nodes (~1000 nodes)
4. **Cost Computation**: Feature similarity + geometric structure matrices
5. **FGW Solving**: Entropic Fused Gromov-Wasserstein optimal transport
6. **Displacement**: Compute node-level displacements with outlier filtering
7. **Interpolation**: RBF interpolation to dense deformation field
8. **Warping**: Apply DVF to moving image

## Configuration

See `configs/default.yaml` for configuration options:
- Foundation model selection
- Graph sampling parameters
- FGW solver settings (alpha, epsilon, max_iter)
- Preprocessing options

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=fmfgwreg tests/
```

## Citation

If you use FM-FGW-Reg in your research, please cite:

```bibtex
@article{fmfgwreg2025,
  title={FM-FGW-Reg: Foundation-Model Fused Gromov-Wasserstein Registration},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- DINOv2 foundation model by Meta AI
- Python Optimal Transport (POT) library
- SimpleITK for medical image processing

# FM-FGW-Reg
