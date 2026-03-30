#!/usr/bin/env python3

"""
MPAS Diagnostics Package

This package provides core diagnostic utilities for validating and analyzing MPAS model output data, with a focus on coordinate arrays and numerical data quality assurance. The diagnostics include comprehensive sanity checking capabilities for geographic coordinates, as well as statistical validation of numerical data arrays. These utilities are designed to serve as pre-processing quality gates, ensuring data integrity before visualization or analysis operations on large unstructured mesh datasets typical of MPAS. The package is optimized for efficiency while providing detailed diagnostic information when problems are detected, helping users identify issues such as invalid coordinates, excessive missing values, uniform data artifacts, or unexpected value ranges in their MPAS model output.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
from mpasdiag.diagnostics.wind import WindDiagnostics
from mpasdiag.diagnostics.sounding import SoundingDiagnostics

__all__ = ['PrecipitationDiagnostics', 'WindDiagnostics', 'SoundingDiagnostics']