#!/usr/bin/env python3
"""
MPASdiag Test Suite: Shared helpers for wind visualization tests.

This module provides the ``require_wind_fixtures`` helper function that was previously copy-pasted at the top of every test method in both ``test_wind_data_prep.py`` and ``test_wind_rendering.py``.  This helper encapsulates the common guard block that checks for the availability of the MPAS coordinate and wind data fixtures, and calls ``pytest.skip()`` if any component is unavailable.  By centralizing this check, we reduce code duplication and improve maintainability across the wind visualization test modules. Each test method in those modules should call ``require_wind_fixtures(mpas_coordinates, mpas_wind_data)`` at the beginning of its execution to ensure that it only runs when the necessary data is available.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import pytest


def require_wind_fixtures(mpas_coordinates: tuple, 
                          mpas_wind_data: tuple) -> None:
    """
    This helper checks for the availability of the MPAS coordinate and wind data fixtures, and calls ``pytest.skip()`` if any component is unavailable.  By centralizing this check, we reduce code duplication and improve maintainability across the wind visualization test modules. Each test method in those modules should call this helper at the beginning of its execution to ensure that it only runs when the necessary data is available.

    Parameters:
        mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
        mpas_wind_data: Session fixture providing real MPAS u/v wind data.

    Returns:
        None: Calls pytest.skip() if data is unavailable, otherwise returns normally.
    """
    if (
        mpas_coordinates is None or mpas_wind_data is None or
        mpas_coordinates[0] is None or mpas_coordinates[1] is None or
        mpas_wind_data[0] is None or mpas_wind_data[1] is None
    ):
        pytest.skip("MPAS data not available")
