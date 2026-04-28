#!/usr/bin/env python3
"""
MPASdiag Test Suite: Tests for mpasdiag.processing package __init__.py

This module contains unit tests for the mpasdiag.processing package's __init__.py file, specifically targeting the remapping submodule import fallback logic. The tests verify that when the remapping submodule is not available, the package correctly sets the _REMAPPING_AVAILABLE flag to False and all remapping-related symbols to None. It uses unittest.mock to simulate the ImportError scenario and ensures that the package behaves as expected in both cases (remapping available and not available).

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import sys
import importlib
from unittest.mock import patch


def test_remapping_import_fallback_sets_none() -> None:
    """
    This test verifies that when the remapping submodule is unavailable, the mpasdiag.processing package correctly sets the _REMAPPING_AVAILABLE flag to False and all remapping-related symbols to None. It uses unittest.mock.patch.dict to simulate the ImportError scenario by temporarily removing the remapping module from sys.modules. The test then reloads the processing package and asserts that the expected attributes are set to None, confirming that the fallback logic in __init__.py is functioning as intended. 

    Parameters:
        None

    Returns:
        None
    """
    import mpasdiag.processing as proc_pkg

    with patch.dict(sys.modules, {'mpasdiag.processing.remapping': None}):
        importlib.reload(proc_pkg)

        assert proc_pkg._REMAPPING_AVAILABLE is False
        assert proc_pkg.MPASRemapper is None
        assert proc_pkg.remap_mpas_to_latlon is None
        assert proc_pkg.remap_mpas_to_latlon_with_masking is None
        assert proc_pkg.build_remapped_valid_mask is None
        assert proc_pkg.create_target_grid is None
        assert proc_pkg.dispatch_remap is None

    importlib.reload(proc_pkg)
    assert proc_pkg._REMAPPING_AVAILABLE is True
