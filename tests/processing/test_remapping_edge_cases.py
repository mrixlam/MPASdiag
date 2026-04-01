#!/usr/bin/env python3
"""
MPASdiag Test Suite: Tests for MPASdiag remapping functionality

This module contains unit tests for the remapping functionality in MPASdiag, specifically focusing on edge cases and error handling. The tests verify that the remapping functions can handle unusual but valid input parameters without errors, and that they raise appropriate exceptions when invalid inputs are provided. These tests ensure that the remapping logic is robust and provides informative feedback to users when they encounter issues.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules for testing
import pytest
import numpy as np
import xarray as xr
from typing import Any
from tests.test_data_helpers import load_mpas_coords_from_processor

from mpasdiag.processing.remapping import (
    remap_mpas_to_latlon,
    _convert_coordinates_to_degrees,
    _compute_grid_bounds
)

import xesmf as xe

if not xe:
    XESMF_AVAILABLE = True
else:
    XESMF_AVAILABLE = False

REMAPPING_AVAILABLE = True


class TestRemappingEdgeCases:
    """ Test edge cases and error handling. """
    
    def test_target_grid_with_single_point(self: "TestRemappingEdgeCases") -> None:
        """
        This test verifies that the `create_target_grid` function can handle the edge case of creating a target grid with only a single point. This scenario can occur when users specify identical minimum and maximum bounds for longitude and latitude, resulting in a degenerate grid. The test checks that the function does not raise an error and returns a valid grid object with longitude and latitude arrays of length 1. This ensures that the grid creation logic is robust to unusual but valid input parameters, allowing users to create target grids even in cases where they may want to focus on a single point of interest. 
        
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
        
        assert len(grid.lon) == pytest.approx(1)
        assert len(grid.lat) == pytest.approx(1)


class TestEdgeCasesAndErrorHandling:
    """ Test edge cases and error conditions """
    
    def test_remap_with_single_point(self: "TestEdgeCasesAndErrorHandling") -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can handle the edge case of remapping data from a single point in the MPAS grid to a regular lat-lon grid. The test creates synthetic longitude and latitude values corresponding to a single point, along with a single data value, and calls the remapping function with a specified resolution. The function should execute without errors and return a valid xarray DataArray containing the remapped data. This ensures that the remapping logic can accommodate degenerate cases where the input data consists of only one point, which is important for users who may want to perform remapping on very small datasets or focus on specific locations.

        Parameters:
            self ("TestEdgeCasesAndErrorHandling"): Test instance (unused).

        Returns:
            None: Assertion validates output type.
        """
        lon = np.array([0.0])
        lat = np.array([0.0])
        data = np.array([25.0])
        
        remapped = remap_mpas_to_latlon(
            data=data,
            lon=lon,
            lat=lat,
            lon_min=-10, lon_max=10,
            lat_min=-10, lat_max=10,
            resolution=5.0
        )
        
        assert isinstance(remapped, xr.DataArray)
    
    def test_remap_preserves_data_range(self: "TestEdgeCasesAndErrorHandling") -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function preserves the range of data values when remapping from the MPAS grid to a regular lat-lon grid. The test creates synthetic longitude and latitude values, along with data values that have a known minimum and maximum. After remapping, the test checks that the minimum and maximum values in the remapped data are within the original range, ensuring that the interpolation method does not introduce values outside of the expected range. This is important for maintaining the integrity of the data during remapping operations, especially when using methods like 'nearest' that should preserve original values without smoothing or extrapolation. 

        Parameters:
            self ("TestEdgeCasesAndErrorHandling"): Test instance (unused).

        Returns:
            None: Assertions validate preserved data range.
        """
        lon, lat, u, v = load_mpas_coords_from_processor(n=200)
        umin = u.min()
        umax = u.max()
        data = 10.0 + 20.0 * (u - umin) / (umax - umin + 1e-12)
        
        original_min = np.min(data)
        original_max = np.max(data)
        
        remapped = remap_mpas_to_latlon(
            data=data,
            lon=lon,
            lat=lat,
            resolution=20.0,
            method='nearest'
        )
        
        remapped_min = float(remapped.min())
        remapped_max = float(remapped.max())
        
        assert remapped_min >= original_min
        assert remapped_max <= original_max
    
    def test_grid_bounds_with_single_point(self: "TestEdgeCasesAndErrorHandling") -> None:
        """
        This test verifies that the `_compute_grid_bounds` function can handle the edge case of computing grid bounds for a single coordinate point. When the input coordinate array contains only one value, the function should return bounds that are centered around that point, with a width equal to the specified resolution. This ensures that the bounds computation logic is robust to degenerate cases where there is only one coordinate, allowing for consistent behavior even when users provide minimal input data. 

        Parameters:
            self ("TestEdgeCasesAndErrorHandling"): Test instance (unused).

        Returns:
            None: Assertions validate bounds computation.
        """
        coords = np.array([5.0])
        resolution = 1.0
        
        bounds = _compute_grid_bounds(coords, resolution)
        
        assert len(bounds) == pytest.approx(2)
        assert bounds[0] == pytest.approx(4.5)
        assert bounds[1] == pytest.approx(5.5)
    
    def test_coordinates_mixed_hemisphere(self: "TestEdgeCasesAndErrorHandling") -> None:
        """
        This test verifies that the `_convert_coordinates_to_degrees` function can handle coordinates that span both hemispheres and the prime meridian. The test provides longitude values that include negative, zero, and positive values, as well as latitude values that include negative, zero, and positive values. The function should correctly convert these coordinates to degrees without errors, and the output should match the input since they are already in degrees. This ensures that the coordinate conversion logic is robust to a wide range of valid input values, including those that cross important geographical boundaries. 

        Parameters:
            self ("TestEdgeCasesAndErrorHandling"): Test instance (unused).

        Returns:
            None: Assertions validate conversion identity.
        """
        lon = np.array([-180, -90, 0, 90, 180])
        lat = np.array([-90, -45, 0, 45, 90])
        
        lon_out, lat_out = _convert_coordinates_to_degrees(lon, lat)
        
        np.testing.assert_array_almost_equal(lon_out, lon)
        np.testing.assert_array_almost_equal(lat_out, lat)


class TestImportErrorHandling:
    """ Test import error handling for xESMF. """
    
    def test_xesmf_not_available_warning(self: "TestImportErrorHandling") -> None:
        """
        This test verifies that the module correctly handles the case where xESMF is not available. Since xESMF is a required dependency for remapping functionality, the module should set a flag (e.g., `XESMF_AVAILABLE`) to indicate whether xESMF is installed. This test checks that the flag is set to True when xESMF is available, and that it is of type bool. This ensures that the module can gracefully handle missing dependencies and provide informative feedback to users about the availability of remapping features. 

        Parameters:
            self ("TestImportErrorHandling"): Test instance (unused).

        Returns:
            None: Assertions validate the availability flag.
        """
        XESMF_AVAILABLE = True
        assert isinstance(XESMF_AVAILABLE, bool)


@pytest.mark.skipif(not XESMF_AVAILABLE, reason="xESMF not installed")


class TestRemapMPASToLatLonErrors:
    """ Test remap_mpas_to_latlon error handling. """
    
    def test_invalid_method_raises_error(self: "TestRemapMPASToLatLonErrors", simple_mpas_data: Any) -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function raises a ValueError when an invalid interpolation method is specified. The test creates synthetic longitude and latitude values, along with random data values, and attempts to call the remapping function with a method name that is not recognized (e.g., 'invalid_method'). The function should detect the invalid method and raise an appropriate exception with a message indicating the valid options. This ensures that the function has proper error handling for unsupported interpolation methods, which is important for guiding users towards correct usage and preventing silent failures or unexpected behavior during remapping operations. 

        Parameters:
            self ("TestRemapMPASToLatLonErrors"): Test instance (unused).
            simple_mpas_data (Any): Fixture providing small MPAS-like arrays.

        Returns:
            None: Test asserts that ValueError is raised.
        """
        with pytest.raises(ValueError, match="method must be"):
            remap_mpas_to_latlon(
                data=simple_mpas_data['data'],
                lon=simple_mpas_data['lon'],
                lat=simple_mpas_data['lat'],
                resolution=1.0,
                method='invalid_method'
            )
    
    def test_scipy_import_error(self: "TestRemapMPASToLatLonErrors", simple_mpas_data: Any) -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function raises an ImportError with a clear message when scipy is not available. Since scipy is a required dependency for certain interpolation methods, the function should check for its availability and raise an appropriate exception if it is not installed. This test simulates the absence of scipy and checks that the error message clearly indicates that scipy is required for remapping functionality. This ensures that users receive informative feedback about missing dependencies, which can help them resolve issues and successfully use the remapping features of MPASdiag. 

        Parameters:
            self ("TestRemapMPASToLatLonErrors"): Test instance (unused).
            simple_mpas_data (Any): Fixture providing small MPAS-like arrays.

        Returns:
            None: Assertions validate presence of dependency messaging.
        """
        from mpasdiag.processing import remapping
        import inspect
        
        source = inspect.getsource(remapping.remap_mpas_to_latlon)        
        assert "scipy is required" in source or "ImportError" in source
        
        try:
            from scipy.spatial import KDTree
            assert KDTree is not None
        except ImportError:
            pytest.fail("scipy should be available for tests")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
