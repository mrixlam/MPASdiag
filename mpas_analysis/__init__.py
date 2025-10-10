#!/usr/bin/env python3

"""
MPAS Analysis Package

A comprehensive Python package for analyzing and visualizing MPAS (Model for Prediction Across Scales) 
unstructured mesh model output data. This package provides tools for data processing, visualization, 
and analysis of MPAS atmospheric model simulations.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Version: 1.1.1
"""

__version__ = "1.1.2"
__author__ = "Rubaiat Islam"
__email__ = "mrislam@ucar.edu"
__institution__ = "Mesoscale & Microscale Meteorology Laboratory, NCAR"

from .data_processing import MPASDataProcessor, validate_geographic_extent, normalize_longitude
from .visualization import (
    MPASFileMetadata,
    UnitConverter,
    MPASVisualizer, 
    MPASPrecipitationPlotter,
    MPASSurfacePlotter,
    MPASWindPlotter
)
from .utils import (
    MPASConfig, 
    MPASLogger, 
    FileManager, 
    DataValidator, 
    PerformanceMonitor,
    ArgumentParser,
    setup_warnings,
    print_system_info,
    load_config_file,
    validate_input_files
)

# Backward compatibility: Export functions from the new classes
get_2d_variable_metadata = MPASFileMetadata.get_2d_variable_metadata
get_3d_variable_metadata = MPASFileMetadata.get_3d_variable_metadata
convert_units = UnitConverter.convert_units
convert_data_for_display = UnitConverter.convert_data_for_display
get_display_units = UnitConverter.get_display_units
_normalize_unit_string = UnitConverter._normalize_unit_string
_format_colorbar_label = UnitConverter._format_colorbar_label
validate_plot_parameters = MPASVisualizer.validate_plot_parameters

__all__ = [
    'MPASDataProcessor',
    'validate_geographic_extent',
    'normalize_longitude',
    
    'MPASFileMetadata',
    'UnitConverter',
    'MPASVisualizer',
    'MPASPrecipitationPlotter',
    'MPASSurfacePlotter',
    'MPASWindPlotter',
    
    # Backward compatibility exports
    'get_2d_variable_metadata',
    'get_3d_variable_metadata',
    'convert_units',
    'convert_data_for_display',
    'get_display_units',
    '_normalize_unit_string',
    '_format_colorbar_label',
    'validate_plot_parameters',
    
    'MPASConfig',
    'MPASLogger',
    'FileManager',
    'DataValidator',
    'PerformanceMonitor',
    'ArgumentParser',
    'setup_warnings',
    'print_system_info',
    'load_config_file',
    'validate_input_files',
]

setup_warnings()

PACKAGE_INFO = {
    'name': 'mpas-analysis',
    'version': __version__,
    'author': __author__,
    'email': __email__,
    'institution': __institution__,
    'description': 'Python package for MPAS model output analysis and visualization',
    'keywords': ['MPAS', 'atmospheric modeling', 'unstructured mesh', 'precipitation analysis', 'visualization'],
    'license': 'MIT',
    'url': 'https://github.com/mrixlam/MPASdiag',
}

def get_package_info():
    """Return package information."""
    return PACKAGE_INFO.copy()

def print_package_info():
    """Print package information."""
    info = get_package_info()
    print(f"\n=== {info['name']} v{info['version']} ===")
    print(f"Author: {info['author']} ({info['email']})")
    print(f"Institution: {info['institution']}")
    print(f"Description: {info['description']}")
    print(f"License: {info['license']}")
    print("=" * 50)