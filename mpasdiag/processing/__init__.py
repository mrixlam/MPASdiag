#!/usr/bin/env python3

"""
MPAS Analysis Processing Package

This package contains the core processing modules for MPAS diagnostic analysis, including data handling, remapping, parallel processing management, and utility functions. The modules are designed to facilitate efficient and robust processing of MPAS model output data for a variety of diagnostic analyses and visualizations. Each module provides specialized functionality while adhering to a consistent interface and design philosophy to ensure ease of use and maintainability across the MPASdiag codebase. The package serves as the foundation for all MPAS diagnostic processing workflows, enabling users to easily access, manipulate, and visualize their MPAS data with confidence in the underlying processing steps.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

from .base import MPASBaseProcessor
from .processors_2d import MPAS2DProcessor
from .processors_3d import MPAS3DProcessor
from .utils_metadata import MPASFileMetadata
from .utils_unit import UnitConverter
from .utils_geog import MPASGeographicUtils
from .utils_datetime import MPASDateTimeUtils
from .utils_config import MPASConfig
from .utils_logger import MPASLogger
from .utils_file import FileManager, print_system_info
from .utils_validator import DataValidator
from .utils_monitor import PerformanceMonitor
from .utils_parser import ArgumentParser

try:
    from .remapping import (
        MPASRemapper, 
        remap_mpas_to_latlon, 
        remap_mpas_to_latlon_with_masking,
        build_remapped_valid_mask,
        create_target_grid
    )
    _REMAPPING_AVAILABLE = True
except ImportError:
    _REMAPPING_AVAILABLE = False
    MPASRemapper = None
    remap_mpas_to_latlon = None
    remap_mpas_to_latlon_with_masking = None
    build_remapped_valid_mask = None
    create_target_grid = None

__all__ = [
    'MPASBaseProcessor',
    'MPAS2DProcessor', 
    'MPAS3DProcessor',
    'MPASFileMetadata',
    'UnitConverter',
    'MPASGeographicUtils',
    'MPASDateTimeUtils',
    'MPASConfig',
    'MPASLogger',
    'FileManager',
    'print_system_info',
    'DataValidator',
    'PerformanceMonitor',
    'ArgumentParser',
    'MPASRemapper',
    'remap_mpas_to_latlon',
    'remap_mpas_to_latlon_with_masking',
    'build_remapped_valid_mask',
    'create_target_grid'
]