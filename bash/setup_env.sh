#!/bin/bash
# Setup script for FM-FGW-Reg environment

ENV_PATH="/nexus/posix0/MBR-neuralsystems/vx/myenv/Reg"

echo "Activating environment..."
source activate $ENV_PATH

echo "Installing core packages..."
conda install -y numpy scipy pandas matplotlib seaborn pyyaml tqdm pytest pytest-cov -c conda-forge

echo "Installing PyTorch (CUDA 11.8)..."
conda install -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

echo "Installing medical imaging packages..."
conda install -y nibabel simpleitk pydicom -c conda-forge

echo "Installing pip packages..."
pip install --upgrade pip
pip install POT timm plotly scikit-learn scikit-image surface-distance

echo "Installing FM-FGW-Reg in development mode..."
cd /u/almik/REG
pip install -e .

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate this environment, run:"
echo "  conda activate $ENV_PATH"
echo ""
echo "Or add to your ~/.bashrc:"
echo "  alias activate-fmfgwreg='conda activate $ENV_PATH'"

