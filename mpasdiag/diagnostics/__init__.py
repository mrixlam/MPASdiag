#!/usr/bin/env python3

"""
MPAS Diagnostics Package

This package provides diagnostic calculation functionality for MPAS models,
including precipitation analysis, wind diagnostics, and atmospheric calculations.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
from mpasdiag.diagnostics.wind import WindDiagnostics

__all__ = ['PrecipitationDiagnostics', 'WindDiagnostics']