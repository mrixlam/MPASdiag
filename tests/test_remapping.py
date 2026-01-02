#!/usr/bin/env python3

"""
Tests for MPAS Remapping Module

Tests the KDTree-based remapping functionality and MPASRemapper class with mock data.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import unittest
import numpy as np
import xarray as xr
from pathlib import Path
import tempfile
import shutil

try:
    from mpasdiag.processing.remapping import (
        MPASRemapper, remap_mpas_to_latlon, create_target_grid
    )
    REMAPPING_AVAILABLE = True
except ImportError:
    REMAPPING_AVAILABLE = False

try:
    import xesmf as xe
    XESMF_AVAILABLE = True
except ImportError:
    XESMF_AVAILABLE = False


@unittest.skipIf(not REMAPPING_AVAILABLE, "Remapping module not available")
class TestRemappingModule(unittest.TestCase):
    """Test basic remapping module functionality."""
    
    def test_import(self) -> None:
        """
        Verify that the remapping module and its core functions can be successfully imported. This test ensures that all required dependencies are available and the module structure is intact. It serves as a basic sanity check before running more complex remapping tests. This is the first test to run to validate the environment setup. The test will fail if critical imports are missing or broken.
        
        Parameters:
            None
        
        Returns:
            None: This test method performs assertions and raises exceptions on failure.
        """
        from mpasdiag.processing.remapping import remap_mpas_to_latlon
        self.assertIsNotNone(remap_mpas_to_latlon)
    
    def test_create_target_grid(self) -> None:
        """
        Verify the creation of a regular latitude-longitude target grid with specified bounds and resolution. This test validates that the create_target_grid function produces an xarray Dataset with correct dimensions and coordinate ranges. The grid spacing is verified to ensure proper resolution handling. This function tests both the grid structure and the accuracy of coordinate arrays. The test uses a regional grid covering North America to ensure realistic parameter ranges.
        
        Parameters:
            None
        
        Returns:
            None: This test method performs assertions and raises exceptions on failure.
        """
        from mpasdiag.processing.remapping import create_target_grid
        grid = create_target_grid(
            lon_min=-130, lon_max=-60,
            lat_min=25, lat_max=50,
            dlon=1.0, dlat=1.0
        )
        
        self.assertIsInstance(grid, xr.Dataset)
        self.assertIn('lon', grid)
        self.assertIn('lat', grid)
        self.assertEqual(len(grid.lon), 71)  # -130 to -60 with 1° spacing
        self.assertEqual(len(grid.lat), 26)  # 25 to 50 with 1° spacing


@unittest.skipIf(not REMAPPING_AVAILABLE, "Remapping module not available")
class TestKDTreeRemapping(unittest.TestCase):
    """Test KDTree-based remapping functionality."""
    
    def setUp(self) -> None:
        """
        Initialize test fixtures including synthetic MPAS data and coordinates for remapping tests. This method creates a mock MPAS mesh with 1000 randomly distributed cells spanning global coverage. A synthetic temperature-like field is generated using sinusoidal patterns to simulate realistic spatial variability. The random seed is fixed to ensure reproducible test results across different runs. These fixtures are used by all test methods in this class to verify remapping accuracy and consistency.
        
        Parameters:
            None
        
        Returns:
            None: This method sets instance variables for use in test methods.
        """
        from mpasdiag.processing.remapping import remap_mpas_to_latlon
        self.remap_func = remap_mpas_to_latlon
        
        np.random.seed(42)
        self.n_cells = 1000
        self.mpas_lon = np.random.uniform(-180, 180, self.n_cells)
        self.mpas_lat = np.random.uniform(-90, 90, self.n_cells)
        lon_rad = np.radians(self.mpas_lon)
        lat_rad = np.radians(self.mpas_lat)
        self.mpas_data = (
            20 + 15 * np.sin(3 * lon_rad) * np.cos(2 * lat_rad) +
            5 * np.random.randn(self.n_cells)
        )
    
    def test_convenience_function(self) -> None:
        """
        Test the high-level remap_mpas_to_latlon convenience function with global coverage. This function validates that the remapping produces a valid xarray DataArray with correct dimensions and finite values. The test uses a coarse 10-degree resolution to ensure fast execution while still validating core functionality. This is the primary interface most users will interact with for remapping operations. The test ensures proper handling of both numpy arrays and coordinate specifications.
        
        Parameters:
            None
        
        Returns:
            None: This test method performs assertions and raises exceptions on failure.
        """
        remapped = self.remap_func(
            data=self.mpas_data,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            lon_min=-180, lon_max=180,
            lat_min=-90, lat_max=90,
            resolution=10.0
        )
        
        self.assertIsInstance(remapped, xr.DataArray)
        self.assertEqual(len(remapped.dims), 2)
        self.assertIn('lon', remapped.dims)
        self.assertIn('lat', remapped.dims)
        self.assertTrue(np.all(np.isfinite(remapped.values)))
    
    def test_remap_with_dataarray(self) -> None:
        """
        Verify that remapping correctly handles xarray DataArray inputs with metadata preservation. This test ensures that variable attributes like units and long_name are carried through the remapping process. The function validates both the data transformation and metadata integrity. This is critical for maintaining traceability in scientific workflows where metadata provides essential context. The test uses a 5-degree resolution grid for efficient computation while maintaining reasonable spatial detail.
        
        Parameters:
            None
        
        Returns:
            None: This test method performs assertions and raises exceptions on failure.
        """
        data_array = xr.DataArray(
            self.mpas_data,
            dims=['nCells'],
            attrs={'units': 'K', 'long_name': 'Temperature'}
        )
        
        remapped = self.remap_func(
            data=data_array,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            lon_min=-180, lon_max=180,
            lat_min=-90, lat_max=90,
            resolution=5.0
        )
        
        self.assertIsInstance(remapped, xr.DataArray)
        self.assertEqual(remapped.attrs['units'], 'K')
        self.assertTrue(np.all(np.isfinite(remapped.values)))
    
    def test_regional_remapping(self) -> None:
        """
        Test remapping functionality for a regional subset covering North America. This validates that the remapping correctly handles non-global domains with specific geographic bounds. The test ensures proper grid size calculation based on the specified resolution and domain extent. Regional remapping is commonly used for focused analysis of specific areas of interest. This test verifies that coordinate arrays match the expected dimensions for the specified 2-degree resolution.
        
        Parameters:
            None
        
        Returns:
            None: This test method performs assertions and raises exceptions on failure.
        """
        remapped = self.remap_func(
            data=self.mpas_data,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            lon_min=-130, lon_max=-60,
            lat_min=25, lat_max=50,
            resolution=2.0
        )
        
        self.assertIsInstance(remapped, xr.DataArray)
        self.assertEqual(len(remapped.lon), 36)  # -130 to -60 with 2° spacing
        self.assertEqual(len(remapped.lat), 13)  # 25 to 50 with 2° spacing
    
    def test_fine_resolution(self) -> None:
        """
        Validate remapping performance and accuracy at high spatial resolution over a small domain. This test uses 0.5-degree resolution over a 20x20 degree domain to verify fine-scale remapping capabilities. The function ensures that high-resolution grids produce finite values without numerical artifacts. Fine resolution remapping is essential for detailed regional studies and model validation. The test confirms proper handling of the increased computational demands of denser target grids.
        
        Parameters:
            None
        
        Returns:
            None: This test method performs assertions and raises exceptions on failure.
        """
        remapped = self.remap_func(
            data=self.mpas_data,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            lon_min=-10, lon_max=10,
            lat_min=-10, lat_max=10,
            resolution=0.5
        )
        
        self.assertIsInstance(remapped, xr.DataArray)
        self.assertTrue(np.all(np.isfinite(remapped.values)))
        self.assertEqual(len(remapped.lon), 41)  # -10 to 10 with 0.5° spacing
    
    def test_data_preservation(self) -> None:
        """
        Verify that the remapping process preserves the physical range of the input data. This test ensures that interpolation does not introduce values outside the original data bounds, which could indicate numerical instability or extrapolation errors. The KDTree nearest-neighbor method should maintain conservative data ranges since it only selects existing values. This is a critical validation for scientific applications where physical constraints must be respected. The test compares minimum and maximum values before and after remapping to detect any range violations.
        
        Parameters:
            None
        
        Returns:
            None: This test method performs assertions and raises exceptions on failure.
        """
        original_min = np.min(self.mpas_data)
        original_max = np.max(self.mpas_data)
        
        remapped = self.remap_func(
            data=self.mpas_data,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            lon_min=-180, lon_max=180,
            lat_min=-90, lat_max=90,
            resolution=10.0
        )
        
        remapped_min = float(remapped.min())
        remapped_max = float(remapped.max())
        
        self.assertGreaterEqual(remapped_min, original_min)
        self.assertLessEqual(remapped_max, original_max)
    
    def test_coordinate_handling(self) -> None:
        """
        Validate that output coordinate arrays exactly match the specified domain boundaries. This test ensures that the longitude and latitude coordinates of the remapped grid align precisely with the requested min/max values. Correct coordinate assignment is fundamental for spatial analysis and data comparison across different datasets. The test uses global bounds (-180 to 180 longitude, -90 to 90 latitude) to verify full-domain coverage. Any coordinate mismatch could lead to misalignment in downstream analyses or visualization.
        
        Parameters:
            None
        
        Returns:
            None: This test method performs assertions and raises exceptions on failure.
        """
        remapped = self.remap_func(
            data=self.mpas_data,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            lon_min=-180, lon_max=180,
            lat_min=-90, lat_max=90,
            resolution=20.0
        )
        
        self.assertEqual(float(remapped.lon.min()), -180.0)
        self.assertEqual(float(remapped.lon.max()), 180.0)
        self.assertEqual(float(remapped.lat.min()), -90.0)
        self.assertEqual(float(remapped.lat.max()), 90.0)


# xESMF tests removed - xESMF is an optional dependency not included in CI/CD
# Users with xESMF installed can still use the MPASRemapper class functionality
# The KDTree-based remapping above provides core functionality without xESMF


class TestRemappingEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_target_grid_with_single_point(self) -> None:
        """
        Test the edge case of creating a degenerate target grid containing only a single point. This validates that the grid creation function handles minimum grid sizes without errors. Single-point grids can occur when users specify identical min/max bounds for a coordinate. The function should produce a valid grid with dimensions of 1x1 rather than failing. This edge case test ensures robustness in the face of unusual but valid parameter combinations.
        
        Parameters:
            None
        
        Returns:
            None: This test method performs assertions and raises exceptions on failure.
        """
        from mpasdiag.processing.remapping import create_target_grid
        grid = create_target_grid(
            lon_min=0, lon_max=0,
            lat_min=0, lat_max=0,
            dlon=1.0, dlat=1.0
        )
        
        self.assertEqual(len(grid.lon), 1)
        self.assertEqual(len(grid.lat), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
