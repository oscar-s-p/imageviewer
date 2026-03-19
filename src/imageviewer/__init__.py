"""
Image Viewer Package

A collection of tools for viewing and analyzing astronomical FITS images.
"""

from .image_viewer_class import image_viewer
from .utils import final_wcs, filter_df, stacking_wcs, skycoord_from_df, cross_match_radec
from .photometry import photo_analysis, detect_sources, get_magnitude, get_coordinates, get_catalogue, explore_catalogues
from importlib.metadata import version as _version

__version__ = _version("imageviewer")
__all__ = ["image_viewer", 
           "final_wcs", "filter_df", "stacking_wcs", 
           "skycoord_from_df", "cross_match_radec",
           "photo_analysis", "detect_sources", "get_magnitude",
           "get_coordinates", "get_catalogue", "explore_catalogues"]