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
from tests.test_data_helpers import load_mpas_coords_from_processor

from mpasdiag.processing.remapping import (
    remap_mpas_to_latlon
)


class TestEdgeCasesAndErrorHandling:
    """ Test edge cases and error conditions """
    
    def test_remap_with_single_point(self: 'TestEdgeCasesAndErrorHandling') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can handle the edge case of remapping data from a single point in the MPAS grid to a regular lat-lon grid. The test creates synthetic longitude and latitude values corresponding to a single point, along with a single data value, and calls the remapping function with a specified resolution. The function should execute without errors and return a valid xarray DataArray containing the remapped data. This ensures that the remapping logic can accommodate degenerate cases where the input data consists of only one point, which is important for users who may want to perform remapping on very small datasets or focus on specific locations.

        Parameters:
            self ('TestEdgeCasesAndErrorHandling'): Test instance (unused).

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
    
    def test_remap_preserves_data_range(self: 'TestEdgeCasesAndErrorHandling') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function preserves the range of data values when remapping from the MPAS grid to a regular lat-lon grid. The test creates synthetic longitude and latitude values, along with data values that have a known minimum and maximum. After remapping, the test checks that the minimum and maximum values in the remapped data are within the original range, ensuring that the interpolation method does not introduce values outside of the expected range. This is important for maintaining the integrity of the data during remapping operations, especially when using methods like 'nearest' that should preserve original values without smoothing or extrapolation. 

        Parameters:
            self ('TestEdgeCasesAndErrorHandling'): Test instance (unused).

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
    
    
class TestRemapToLatlonAllZeroData:
    """ Tests for remap_mpas_to_latlon with all-zero / all-NaN input data. """

    @pytest.fixture
    def dataset(self: 'TestRemapToLatlonAllZeroData') -> xr.Dataset:
        """
        This fixture creates a synthetic xarray Dataset with longitude and latitude coordinates for testing the remap_mpas_to_latlon function. The dataset contains 100 cells with randomly generated longitude values between -105 and -100 degrees, and latitude values between 35 and 40 degrees. The longitude and latitude values are converted to radians, as expected by the remapping function. This dataset is used in multiple tests to verify that the remapping function can handle cases where the input data is all zeros or all NaNs without errors, and that it preserves attributes when using xarray DataArrays.

        Parameters:
            None

        Returns:
            xr.Dataset: A dataset containing 'lonCell' and 'latCell' DataArrays with random coordinates in radians for testing remapping edge cases. 
        """
        n = 100
        rng = np.random.default_rng(11)
        lon = rng.uniform(-105, -100, n)
        lat = rng.uniform(35, 40, n)
        return xr.Dataset({
            'lonCell': xr.DataArray(np.radians(lon), dims=['nCells']),
            'latCell': xr.DataArray(np.radians(lat), dims=['nCells']),
        })

    def test_all_zero_data_returns_result(self: 'TestRemapToLatlonAllZeroData', 
                                          dataset: xr.Dataset) -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can handle the edge case of remapping data that consists entirely of zeros. The test creates a dataset with longitude and latitude coordinates, and a data array filled with zeros. When the remapping function is called with this input, it should execute without errors and return a valid xarray DataArray containing the remapped data. This ensures that the remapping logic can accommodate cases where the input data has no variation, which is important for users who may want to perform remapping on datasets that are initialized to zero or have been processed in a way that results in all-zero values. 

        Parameters:
            dataset (xr.Dataset): Fixture providing longitude and latitude coordinates for testing.

        Returns:
            None: Test validates that remap_mpas_to_latlon returns a DataArray without errors when input data is all zeros. 
        """
        data = np.zeros(dataset.sizes['nCells'])
        lon = np.degrees(dataset['lonCell'].values)
        lat = np.degrees(dataset['latCell'].values)
        result = remap_mpas_to_latlon(data, lon, lat, lon_min=-105, lon_max=-100,
                                      lat_min=35, lat_max=40, resolution=1.0)
        assert isinstance(result, xr.DataArray)

    def test_all_nan_data_returns_result(self: 'TestRemapToLatlonAllZeroData', 
                                         dataset: xr.Dataset) -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can handle the edge case of remapping data that consists entirely of NaN values. The test creates a dataset with longitude and latitude coordinates, and a data array filled with NaNs. When the remapping function is called with this input, it should execute without errors and return a valid xarray DataArray containing the remapped data, which may also contain NaN values. This ensures that the remapping logic can accommodate cases where the input data is invalid or missing, which is important for users who may encounter datasets with NaN values due to processing steps or data quality issues.

        Parameters:
            dataset (xr.Dataset): Fixture providing longitude and latitude coordinates for testing.

        Returns:
            None: Test validates that remap_mpas_to_latlon returns a DataArray without errors when input data is all NaNs. 
        """
        n = dataset.sizes['nCells']
        data = np.full(n, np.nan)
        lon = np.degrees(dataset['lonCell'].values)
        lat = np.degrees(dataset['latCell'].values)
        result = remap_mpas_to_latlon(data, lon, lat, lon_min=-105, lon_max=-100,
                                      lat_min=35, lat_max=40, resolution=1.0)
        assert isinstance(result, xr.DataArray)

    def test_xarray_input_preserves_attrs(self: 'TestRemapToLatlonAllZeroData', 
                                          dataset: xr.Dataset) -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function preserves the attributes of an xarray DataArray when remapping. The test creates a dataset with longitude and latitude coordinates, and a data array filled with random values that has specific attributes (e.g., 'units' and 'long_name'). When the remapping function is called with this input, it should execute without errors and return a valid xarray DataArray containing the remapped data, while also preserving the original attributes. This ensures that the remapping logic maintains important metadata associated with the data, which is crucial for users who rely on attributes for understanding the context and meaning of their data after remapping operations. 

        Parameters:
            dataset (xr.Dataset): Fixture providing longitude and latitude coordinates for testing.

        Returns:
            None: Test validates that remap_mpas_to_latlon preserves DataArray attributes.
        """
        n = dataset.sizes['nCells']
        rng = np.random.default_rng(7)
        attrs = {'units': 'mm', 'long_name': 'precipitation'}
        data = xr.DataArray(rng.uniform(0, 5, n), dims=['nCells'], attrs=attrs)
        lon = np.degrees(dataset['lonCell'].values)
        lat = np.degrees(dataset['latCell'].values)
        result = remap_mpas_to_latlon(data, lon, lat, lon_min=-105, lon_max=-100,
                                      lat_min=35, lat_max=40, resolution=1.0)
        assert result.attrs.get('units') == 'mm'
        assert result.attrs.get('long_name') == 'precipitation'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
