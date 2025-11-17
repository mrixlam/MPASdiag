#!/usr/bin/env python3

"""
MPAS Visualization Package

This package provides comprehensive visualization tools for MPAS data analysis.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

__version__ = "1.0.0"

from mpasdiag.visualization.styling import MPASVisualizationStyle
from mpasdiag.visualization.base_visualizer import MPASVisualizer, UnitConverter
from mpasdiag.processing.utils_metadata import MPASFileMetadata
from mpasdiag.visualization.surface import MPASSurfacePlotter, create_surface_plot
from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter
from mpasdiag.visualization.wind import MPASWindPlotter

__all__ = [
    'MPASVisualizer',
    'UnitConverter',
    'MPASFileMetadata',
    'MPASVisualizationStyle',
    'MPASSurfacePlotter',
    'create_surface_plot',
    'MPASVerticalCrossSectionPlotter',
    'MPASPrecipitationPlotter',
    'MPASWindPlotter'
]