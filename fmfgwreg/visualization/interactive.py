"""Interactive visualization (placeholder for napari integration)."""

import numpy as np
import warnings


def napari_viewer(fixed, moving, warped, dvf=None):
    """
    Open napari viewer with registration results (if napari is installed).
    
    Args:
        fixed: Fixed volume
        moving: Moving volume
        warped: Warped volume
        dvf: Optional deformation field
    """
    try:
        import napari
        
        viewer = napari.Viewer()
        viewer.add_image(fixed, name='Fixed', colormap='gray')
        viewer.add_image(moving, name='Moving', colormap='magenta', opacity=0.5)
        viewer.add_image(warped, name='Warped', colormap='green', opacity=0.5)
        
        if dvf is not None:
            # Add DVF as vectors (requires preprocessing)
            warnings.warn("DVF visualization in napari not yet fully implemented")
        
        napari.run()
        
    except ImportError:
        warnings.warn("Napari not installed. Install with: pip install napari[all]")

