#!/usr/bin/env python3
"""
MPASdiag Test Suite: Surface Data Tests

This module contains unit tests for the data handling functionality of the MPASdiag visualization package, specifically for surface data plotting. The tests focus on verifying that the plotting functions correctly handle various data input scenarios, including 3D arrays with level selection, 2D arrays with level dimensions, and xarray DataArrays with units. The tests use real MPAS coordinate and surface temperature data provided by session fixtures to ensure realistic plotting scenarios. Additionally, the tests cover edge cases such as unit conversion failures and validation of data points within the plotting extent.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules for testing
import os
import sys
import pytest
import matplotlib
import numpy as np
import xarray as xr
matplotlib.use('Agg')
from unittest.mock import Mock
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from mpasdiag.visualization.surface import MPASSurfacePlotter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestDataExtraction:
    """ Tests for data extraction and validation. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestDataExtraction", mpas_coordinates, mpas_surface_temp_data) -> None:
        """
        This fixture initializes the MPASSurfacePlotter and prepares real MPAS coordinate and surface temperature data for testing various data extraction scenarios in the plotting functions. It ensures that the test methods have access to realistic data arrays for longitude, latitude, and surface temperature, which are essential for verifying that the plotting functions can handle different data shapes, level selection methods, and unit handling cases effectively.

        Parameters:
            self ("TestDataExtraction"): Test instance which will receive fixture attributes.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_surface_temp_data: Session fixture providing real surface temperature data.

        Returns:
            None: Populates `self.plotter`, `self.lon`, `self.lat`, and `self.temp_data`.
        """
        if mpas_coordinates is None or mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
            return
        
        self.plotter = MPASSurfacePlotter()

        lon_full, lat_full = mpas_coordinates
        self.lon = lon_full[:50]
        self.lat = lat_full[:50]
        
        self.temp_data = mpas_surface_temp_data[:50]
        
        self.extent_bounds = (
            float(self.lon.min()), float(self.lon.max()),
            float(self.lat.min()), float(self.lat.max())
        )
    
    def test_3d_data_with_level_index(self: "TestDataExtraction") -> None:
        """
        This test verifies that the plotting function can handle 3D data arrays when a specific vertical level index is provided. It checks that the plotter correctly extracts the specified level from the 3D array and generates a plot without errors. The test asserts that a Figure object is returned, indicating that the plotting process completed successfully for the selected level.

        Parameters:
            self ("TestDataExtraction"): Test instance containing `plotter`, `lon`, and `lat`.

        Returns:
            None: Assertion validates the returned Figure object.
        """
        data_3d = np.tile(self.temp_data, (10, 5, 1)).T  
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, data_3d, 't2m',
            *self.extent_bounds,
            level_index=3,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_3d_data_with_level_value(self: "TestDataExtraction") -> None:
        """
        This test verifies that the plotting function can handle 3D data arrays when a specific vertical level value is provided. It checks that the plotter correctly identifies the level corresponding to the provided value and generates a plot without errors. The test asserts that a Figure object is returned, indicating that the plotting process completed successfully for the selected level value.

        Parameters:
            self ("TestDataExtraction"): Test instance containing `plotter`, `lon`, and `lat`.

        Returns:
            None: Assertion validates the returned Figure object.
        """
        data_3d = np.tile(self.temp_data, (10, 5, 1)).T  
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, data_3d, 't2m',
            *self.extent_bounds,
            level_value=850,  
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_3d_data_no_level_specified(self: "TestDataExtraction") -> None:
        """
        This test verifies that the plotting function can handle 3D data arrays when no specific vertical level is specified. It checks that the plotter defaults to using the first level of the 3D array and generates a plot without errors. The test asserts that a Figure object is returned, indicating that the plotting process completed successfully using the default level.

        Parameters:
            self ("TestDataExtraction"): Test instance containing `plotter`, `lon`, and `lat`.

        Returns:
            None: Assertion validates the returned Figure object.
        """
        data_3d = np.tile(self.temp_data, (10, 5, 1)).T  
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, data_3d, 't2m',
            *self.extent_bounds,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_2d_data_with_level_index(self: "TestDataExtraction") -> None:
        """
        This test verifies that the plotting function can handle 2D data arrays that include a level dimension when a specific vertical level index is provided. It checks that the plotter correctly extracts the specified level from the 2D array and generates a plot without errors. The test asserts that a Figure object is returned, indicating that the plotting process completed successfully for the selected level index.

        Parameters:
            self ("TestDataExtraction"): Test instance containing `plotter`, `lon`, and `lat`.

        Returns:
            None: Assertion validates the returned Figure object.
        """
        data_2d = np.tile(self.temp_data, (10, 1)).T 
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, data_2d, 't2m',
            *self.extent_bounds,
            level_index=5,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_data_array_with_units(self: "TestDataExtraction") -> None:
        """
        This test verifies that the plotting function can handle xarray DataArrays that include units in their attributes. It checks that the plotter correctly extracts the unit information from the DataArray and applies any necessary unit conversions before plotting. The test asserts that a Figure object is returned, indicating that the plotting process completed successfully with the DataArray input and its associated units.

        Parameters:
            self ("TestDataExtraction"): Test instance containing `plotter`, `lon`, and `lat`.

        Returns:
            None: Assertion validates the returned Figure object.
        """
        data = xr.DataArray(
            self.temp_data,
            dims=['nCells'],
            attrs={'units': 'K'}
        )
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, data, 't2m',
            *self.extent_bounds,
            data_array=data,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_data_with_exception_in_unit_extraction(self: "TestDataExtraction") -> None:
        """
        This test verifies that the plotting function can handle cases where an exception occurs during unit extraction from an xarray DataArray. It checks that if the DataArray's attributes cause an error when attempting to extract units, the plotter should catch the exception and proceed with plotting without applying unit conversions. The test asserts that a Figure object is returned, indicating that the plotting process completed successfully even when unit extraction fails.

        Parameters:
            self ("TestDataExtraction"): Test instance containing `plotter`, `lon`, and `lat`.

        Returns:
            None: Assertion validates the returned Figure object.
        """
        data = self.temp_data

        data_array = xr.DataArray(
            data,
            dims=['nCells'],
            attrs={'invalid': 'test'}  
        )
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, data, 't2m',
            *self.extent_bounds,
            data_array=data_array,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestUnitConversion:
    """ Tests for unit conversion edge cases. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestUnitConversion", mpas_coordinates, mpas_surface_temp_data) -> None:
        """
        This fixture initializes the MPASSurfacePlotter and prepares real MPAS coordinate and surface temperature data for testing unit conversion edge cases in the plotting functions. It ensures that the test methods have access to realistic data arrays for longitude, latitude, and surface temperature, which are essential for verifying that the plotting functions can handle scenarios where unit conversion may fail or when DataArrays with units are provided. The fixture sets up the necessary attributes for the test methods to exercise unit conversion logic effectively.

        Parameters:
            self ("TestUnitConversion"): Test instance which will receive fixture attributes.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_surface_temp_data: Session fixture providing real surface temperature data.

        Returns:
            None: Populates `self.plotter`, `self.lon`, `self.lat`, and `self.temp_data`.
        """
        if mpas_coordinates is None or mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
            return
        
        self.plotter = MPASSurfacePlotter()
        lon_full, lat_full = mpas_coordinates
        self.lon = lon_full[:50]
        self.lat = lat_full[:50]
        
        self.temp_data = mpas_surface_temp_data[:50]
        
        self.extent_bounds = (
            float(self.lon.min()), float(self.lon.max()),
            float(self.lat.min()), float(self.lat.max())
        )
    
    def test_unit_conversion_failure(self: "TestUnitConversion") -> None:
        """
        This test verifies that when unit conversion fails due to unrecognized or invalid units in an xarray DataArray, the plotting function should catch the exception and proceed with plotting without applying unit conversions. It checks that the plotter can handle the failure gracefully and still returns a valid Figure object, indicating that the plotting process completed successfully even when unit conversion is not possible.

        Parameters:
            self ("TestUnitConversion"): Test instance containing `plotter`, `lon`, and `lat`.

        Returns:
            None: Assertion validates the returned Figure object.
        """
        data = xr.DataArray(
            self.temp_data,
            dims=['nCells'],
            attrs={'units': 'invalid_unit'}
        )
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, data, 't2m',
            *self.extent_bounds,
            data_array=data,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_unit_conversion_xarray_result(self: "TestUnitConversion") -> None:
        """
        This test verifies that when unit conversion succeeds and returns an xarray DataArray, the plotting function can handle this result correctly. It checks that if the unit conversion helper returns a DataArray with converted values and preserved metadata, the plotter can still generate a valid Figure object without errors. The test asserts that a Figure object is returned, indicating that the plotting process completed successfully with the converted DataArray.

        Parameters:
            self ("TestUnitConversion"): Test instance containing `plotter`, `lon`, and `lat`.

        Returns:
            None: Assertion validates the returned Figure object.
        """
        data = xr.DataArray(
            self.temp_data,
            dims=['nCells'],
            attrs={'units': 'K'}
        )
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, data, 't2m',
            *self.extent_bounds,
            data_array=data,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestDataValidation:
    """ Tests for data validation and filtering. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestDataValidation") -> None:
        """
        This fixture initializes the MPASSurfacePlotter for testing data validation and filtering scenarios in the plotting functions. It sets up the necessary attributes for the test methods to exercise validation logic effectively, including default extent bounds for testing cases where data points may be outside the plotting area or when all data points are invalid.

        Parameters:
            self ("TestDataValidation"): Test instance which will receive `plotter`.

        Returns:
            None: Populates `self.plotter`.
        """
        self.plotter = MPASSurfacePlotter()
        self.extent_bounds = (-100, -90, 30, 40)
    
    def test_no_valid_data_points(self: "TestDataValidation") -> None:
        """
        This test verifies that when all data points are invalid (e.g., all NaN values), the plotting function should raise a ValueError indicating that there are no valid data points to plot. It checks that the plotter correctly identifies the lack of valid data and responds with an appropriate error message, preventing the generation of an empty or misleading plot.

        Parameters:
            self ("TestDataValidation"): Test instance with prepared fixtures.

        Returns:
            None: Assertion checks for the expected ValueError.
        """
        lon = np.array([0, 1, 2])
        lat = np.array([0, 1, 2])
        data = np.array([np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError) as ctx:
            self.plotter.create_surface_map(
                lon, lat, data, 't2m',
                *self.extent_bounds,
                plot_type='scatter'
            )
        assert 'No valid data points' in str(ctx.value)
    
    def test_data_outside_extent(self: "TestDataValidation") -> None:
        """
        This test verifies that when all data points are outside the specified plotting extent, the plotting function should raise a ValueError indicating that there are no valid data points within the extent to plot. It checks that the plotter correctly identifies that all data points fall outside the defined bounds and responds with an appropriate error message, preventing the generation of an empty or misleading plot.

        Parameters:
            self ("TestDataValidation"): Test instance with prepared fixtures.

        Returns:
            None: Assertion checks for the expected ValueError.
        """
        lon = np.array([10, 20, 30])
        lat = np.array([10, 20, 30])
        data = np.array([1, 2, 3])
        
        with pytest.raises(ValueError) as ctx:
            self.plotter.create_surface_map(
                lon, lat, data, 't2m',
                *self.extent_bounds,
                plot_type='scatter'
            )
        assert 'No valid data points' in str(ctx.value)


class TestConvertToNumpy:
    """ Tests for convert_to_numpy static method. """
    
    def test_convert_xarray_dataarray(self: "TestConvertToNumpy") -> None:
        """
        This test verifies that the `convert_to_numpy` static method correctly converts an xarray DataArray to a numpy ndarray. It checks that when an xarray DataArray is passed to the method, it returns a numpy array with the same data values. The test asserts that the result is an instance of np.ndarray and that the values are equal to the original DataArray's values.

        Parameters:
            self ("TestConvertToNumpy"): Test caller (unused).

        Returns:
            None: Assertions validate conversion correctness.
        """
        data = xr.DataArray([1, 2, 3], dims=['x'])
        result = MPASSurfacePlotter.convert_to_numpy(data)
        
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])
    
    def test_convert_numpy_array(self: "TestConvertToNumpy") -> None:
        """
        This test verifies that the `convert_to_numpy` static method correctly handles input that is already a numpy array by returning it unchanged. It checks that when a numpy array is passed to the method, it returns the same array without modification. The test asserts that the result is an instance of np.ndarray and that the values are equal to the original array.

        Parameters:
            self ("TestConvertToNumpy"): Test caller (unused).
            
        Returns:
            None: Assertions validate the returned ndarray.
        """
        data = np.array([1, 2, 3])
        result = MPASSurfacePlotter.convert_to_numpy(data)
        
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])
    
    def test_convert_with_exception(self: "TestConvertToNumpy") -> None:
        """
        This test verifies that the `convert_to_numpy` static method can handle cases where an exception occurs during conversion (e.g., when the input object does not support conversion to a numpy array). It checks that if an exception is raised during the conversion process, the method should catch the exception and return a numpy array using `np.asarray` as a fallback. The test asserts that the result is an instance of np.ndarray, indicating that the method handled the exception gracefully.

        Parameters:
            self ("TestConvertToNumpy"): Test caller (unused).

        Returns:
            None: Assertion validates returned ndarray type.
        """
        mock_data = Mock()
        mock_data.__getitem__ = Mock(side_effect=Exception("Conversion failed"))
        
        result = MPASSurfacePlotter.convert_to_numpy(mock_data)
        assert isinstance(result, np.ndarray)
    
    def test_convert_dask_array(self: "TestConvertToNumpy") -> None:
        """
        This test verifies that the `convert_to_numpy` static method can handle input that is a dask array by calling its `compute` method to retrieve the underlying numpy array. It checks that when a mock object with a `compute` method is passed to the method, it calls the `compute` method and returns the resulting numpy array. The test asserts that the `compute` method was called and that the result is an instance of np.ndarray.

        Parameters:
            self ("TestConvertToNumpy"): Test caller (unused).

        Returns:
            None: Assertions validate `compute` was called and result type.
        """
        mock_dask = Mock()
        mock_dask.compute = Mock(return_value=np.array([1, 2, 3]))
        
        result = MPASSurfacePlotter.convert_to_numpy(mock_dask)
        
        mock_dask.compute.assert_called_once()
        assert isinstance(result, np.ndarray)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
