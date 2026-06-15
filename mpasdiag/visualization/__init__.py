#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

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
from mpasdiag.visualization.base_visualizer import (
    MPASVisualizer,
    WindPlotStyle,
    TransectLineStyle,
)
from mpasdiag.processing.utils_unit import UnitConverter
from mpasdiag.processing.utils_metadata import MPASFileMetadata
from mpasdiag.visualization.surface import (
    MPASSurfacePlotter,
    SurfaceMapStyle,
    create_surface_plot,
)
from mpasdiag.visualization.cross_section import (
    MPASVerticalCrossSectionPlotter,
    CrossSectionStyle,
)
from mpasdiag.visualization.precipitation import (
    MPASPrecipitationPlotter,
    PrecipitationMapStyle,
    PrecipitationRenderStyle,
    OverlayColorSpec,
)
from mpasdiag.visualization.wind import MPASWindPlotter
from mpasdiag.visualization.skewt import MPASSkewTPlotter

__all__ = [
    "MPASVisualizer",
    "UnitConverter",
    "MPASFileMetadata",
    "MPASVisualizationStyle",
    "WindPlotStyle",
    "TransectLineStyle",
    "MPASSurfacePlotter",
    "SurfaceMapStyle",
    "create_surface_plot",
    "MPASVerticalCrossSectionPlotter",
    "CrossSectionStyle",
    "MPASPrecipitationPlotter",
    "PrecipitationMapStyle",
    "PrecipitationRenderStyle",
    "OverlayColorSpec",
    "MPASWindPlotter",
    "MPASSkewTPlotter",
]
