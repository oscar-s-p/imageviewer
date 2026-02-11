"""
Image Viewer Package

A collection of tools for viewing and analyzing astronomical FITS images.
"""

from .image_viewer_class import image_viewer
from .utils import final_wcs, filter_df, stacking_wcs
from .photometry import photo_analysis
from importlib.metadata import version as _version

__version__ = _version("imageviewer")
__all__ = ["image_viewer", 
           "final_wcs", "filter_df", "stacking_wcs", 
           "photo_analysis"]