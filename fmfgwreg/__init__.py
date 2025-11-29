"""
FM-FGW-Reg: Foundation-Model Fused Gromov-Wasserstein Registration

A training-free 3D medical image registration framework using frozen foundation
models and optimal transport for graph-based structural matching.
"""

__version__ = "0.1.0"

from fmfgwreg.core.registration import FMFGWReg
from fmfgwreg.core.config import RegistrationConfig

__all__ = ["FMFGWReg", "RegistrationConfig", "__version__"]

