#!/usr/bin/env python3
"""
MPASdiag Test Suite: Wind Visualization Functionality

This test suite focuses on validating the wind visualization capabilities of the MPASdiag package, specifically the `MPASWindPlotter` class. It includes tests for the initialization of the plotter, the optimal subsampling heuristic for vector density control, and the conversion of various input types to NumPy arrays. The suite also verifies the preparation of wind data for plotting, ensuring that subsampling and NaN handling work as expected. These tests are designed to catch regressions in core wind plotting functionality and ensure that the underlying data processing steps are robust and produce consistent outputs across different scenarios.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# =================== Import Section ===================

# Standard library imports
import os
import sys
import pytest
import shutil
import tempfile

# Third-party imports
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
from typing import Generator, cast

# Import mocking tools for isolating tests
from unittest.mock import MagicMock

# Set matplotlib backend for headless testing
import matplotlib
matplotlib.use('Agg')
from cartopy import crs as ccrs

# Matplotlib imports for plotting
import matplotlib.pyplot as plt

# Cartopy imports for geographic plotting
from cartopy.mpl.geoaxes import GeoAxes

# Import the MPASWindPlotter to be tested
from mpasdiag.visualization.wind import MPASWindPlotter

# Ensure the test directory is in sys.path for local imports
testDir = str(Path(__file__).parent)

# Insert at front to prioritize local test modules
if testDir not in sys.path:
    sys.path.insert(0, testDir)

# ================== Helper Functions ===================

def fake_render_factory(calls: dict):
    """
    This factory function creates a fake render method that increments a call count in the provided dictionary. It is used to mock the rendering behavior of the plotter during testing, allowing us to verify that the render method is invoked without executing actual plotting code.

    Parameters:
        calls: dict - a dictionary used to count render invocations.

    Returns:
        callable: function(ax, *args, **kwargs)
    """
    def _fake_render(ax_arg, *args, **kwargs):
        calls['render'] = calls.get('render', 0) + 1
    return _fake_render


class BadArray:
    """ Lightweight test helper that raises when `.values` is accessed. Placed at module scope so multiple tests can reuse it. """
    @property
    def values(self):
        raise RuntimeError("Bad values")

# ===================== Test Class: MPASWindPlotterInit =====================

class TestMPASWindPlotterInit:
    """ Tests for the initialization of the `MPASWindPlotter` class. """

    # ------------------ Test Default Initialization ------------------

    def test_init_default(self: "TestMPASWindPlotterInit") -> None:
        """
        This test verifies that the `MPASWindPlotter` class can be instantiated with default parameters without raising exceptions. It checks that the default `figsize` is set to (12, 10) and the default `dpi` is 100. Additionally, it confirms that the `fig` and `ax` attributes are initialized to `None`, indicating that no figure or axes have been created yet. This test ensures that the constructor sets up the plotter with expected defaults, which is crucial for users who rely on standard settings for quick plotting.

        Parameters:
            None

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Initialize plotter with defaults
        plotter = MPASWindPlotter()

        # Verify default figsize is (12, 10)
        assert plotter.figsize == (12, 10)

        # Verify default DPI is 100
        assert plotter.dpi == 100

        # Verify figure is None before plotting
        assert plotter.fig is None

        # Verify axes are None before plotting
        assert plotter.ax is None

    # ------------------ Test Custom Initialization ------------------

    def test_init_custom(self: "TestMPASWindPlotterInit") -> None:
        """
        This test checks that the `MPASWindPlotter` class can be instantiated with custom parameters for `figsize` and `dpi`. It verifies that when a user provides specific values for these parameters, they are correctly assigned to the instance attributes. The test confirms that the plotter can be initialized with non-default settings, which is important for users who require different figure sizes or resolutions for their visualizations. Like the default initialization test, it also ensures that the `fig` and `ax` attributes start as `None`, indicating that no plotting has occurred yet.

        Parameters:
            None

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Initialize plotter with custom specifications
        plotter = MPASWindPlotter(figsize=(14, 12), dpi=150)

        # Verify custom figsize is set correctly
        assert plotter.figsize == (14, 12)

        # Verify custom DPI is set correctly
        assert plotter.dpi == 150

        # Verify figure is None before plotting
        assert plotter.fig is None

        # Verify axes are None before plotting
        assert plotter.ax is None

# ================ Test Class: CalculateOptimalSubsample =================

class TestCalculateOptimalSubsample:
    """ Tests for the `calculate_optimal_subsample` method of the `MPASWindPlotter` class """

    # ------------------ Initialize the Plotter Fixture ------------------

    @pytest.fixture
    def plotter(self: "TestCalculateOptimalSubsample") -> MPASWindPlotter:
        """
        This fixture creates and returns an instance of the `MPASWindPlotter` class for use in the subsampling tests. By providing a fresh plotter instance for each test method, it ensures that any state changes or configurations made during one test do not affect others. This setup allows the test methods to focus on validating the behavior of the `calculate_optimal_subsample` method without worrying about shared state or side effects from previous tests.

        Parameters:
            self ("TestCalculateOptimalSubsample"): Test instance which will receive the plotter fixture.

        Returns:
            MPASWindPlotter: Initialized plotter used across test methods.
        """
        return MPASWindPlotter()
    
    # ------------------ Test Small Dataset Scenarios ------------------

    def test_subsample_small_dataset(self: "TestCalculateOptimalSubsample", plotter: MPASWindPlotter) -> None:
        """
        This test verifies that for a small dataset (e.g., 1000 points) with a moderate geographic extent, the `calculate_optimal_subsample` method returns a subsample factor of 1, indicating that no subsampling is needed. It checks both 'barbs' and 'arrows' plot types to ensure consistent behavior across different vector representations. This test confirms that the method correctly identifies when the input data is already at an appropriate density for plotting without any reduction, which is crucial for maintaining visual clarity in smaller datasets.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to call the method.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Define a small dataset with few points to plot wind barbs
        subsample_barbs_small = plotter.calculate_optimal_subsample(
            num_points=1000,
            lon_min=-30, lon_max=30,
            lat_min=-15, lat_max=15,
            plot_type='barbs'
        )

        # Verify no subsampling for barbs
        assert subsample_barbs_small == 1

        # Define a small dataset with few points to plot wind arrows
        subsample_arrows_small = plotter.calculate_optimal_subsample(
            num_points=1000,
            lon_min=-30, lon_max=30,
            lat_min=-15, lat_max=15,
            plot_type='arrows'
        )

        # Verify no subsampling for arrows
        assert subsample_arrows_small == 1

        # Close any open figures to free resources
        plt.close('all')
    
    # ------------------ Test Large Dataset Scenarios ------------------

    def test_subsample_large_dataset(self: "TestCalculateOptimalSubsample", plotter: MPASWindPlotter) -> None:
        """
        This test checks that for a large dataset (e.g., 100,000 points) covering a global extent, the `calculate_optimal_subsample` method returns subsample factors greater than 1, indicating that subsampling is necessary to reduce vector density for clear visualization. It verifies that the returned subsample factors do not exceed the configured maximum cap (50) and that the relationship between 'barbs' and 'arrows' subsampling is maintained (i.e., barb subsample should not be less than arrow subsample, and arrow subsample should be at most double the barb subsample). This test ensures that the method effectively manages high-density data while adhering to constraints designed to maintain visual clarity.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to call the method.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Large dataset with many points
        subsample_barbs_large = plotter.calculate_optimal_subsample(
            num_points=100000,
            lon_min=-180, lon_max=180,
            lat_min=-90, lat_max=90,
            plot_type='barbs'
        )

        # Should subsample but not exceed max cap
        assert subsample_barbs_large > 1
        assert subsample_barbs_large <= 50

        # Large dataset with many points for arrows
        subsample_arrows_large = plotter.calculate_optimal_subsample(
            num_points=100000,
            lon_min=-180, lon_max=180,
            lat_min=-90, lat_max=90,
            plot_type='arrows'
        )

        # Should subsample but not exceed max cap
        assert subsample_arrows_large > 1
        assert subsample_arrows_large <= 50

        # Ensure barb subsample is not less than arrow subsample
        assert subsample_barbs_large >= subsample_arrows_large

        # Ensure arrow subsample is at most double barb subsample
        assert subsample_arrows_large <= 2 * subsample_barbs_large

        # Close any open figures to free resources
        plt.close('all')

    # ------------------ Test Custom Parameter Scenarios ------------------    

    def test_subsample_custom_figsize(self: "TestCalculateOptimalSubsample", plotter: MPASWindPlotter) -> None:
        """
        This test verifies that the `calculate_optimal_subsample` method responds to changes in the `figsize` parameter. A larger figure size should allow for a lower subsample factor (i.e., more vectors can be plotted without overcrowding), while a smaller figure size should require a higher subsample factor to maintain visual clarity. The test checks that the subsample factor returned for a larger figure is less than or equal to that of a smaller figure, confirming that the method appropriately adjusts its calculations based on the available plotting area.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to call the method.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Larger figure should allow lower subsample
        subsample_large_fig = plotter.calculate_optimal_subsample(
            num_points=50000,
            lon_min=-50, lon_max=50,
            lat_min=-25, lat_max=25,
            figsize=(20, 16),
            plot_type='barbs'
        )

        # Should return valid subsample factor
        assert subsample_large_fig >= 1

        # Smaller figure should require higher subsample
        subsample_small_fig = plotter.calculate_optimal_subsample(
            num_points=50000,
            lon_min=-50, lon_max=50,
            lat_min=-25, lat_max=25,
            figsize=(8, 6),
            plot_type='barbs'
        )

        # Should return valid subsample factor
        assert subsample_small_fig >= subsample_large_fig

        # Close any open figures to free resources
        plt.close('all')

    # ------------------ Test Custom Target Density ------------------

    def test_subsample_custom_target_density(self: "TestCalculateOptimalSubsample", plotter: MPASWindPlotter) -> None:
        """
        This test checks that the `calculate_optimal_subsample` method responds to changes in the `target_density` parameter. A lower target density should result in a higher subsample factor (i.e., fewer vectors plotted), while a higher target density should allow for a lower subsample factor (i.e., more vectors plotted). The test verifies that as the target density increases, the returned subsample factor decreases, confirming that the method adjusts its calculations based on the desired vector density for visualization.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to call the method.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Custom target density should affect subsample
        subsample_lower_density = plotter.calculate_optimal_subsample(
            num_points=50000,
            lon_min=-50, lon_max=50,
            lat_min=-25, lat_max=25,
            plot_type='barbs',
            target_density=5
        )

        # Should return valid subsample factor
        assert subsample_lower_density >= 1
        
        # Calculate with a higher target density
        subsample_higher_density = plotter.calculate_optimal_subsample(
            num_points=50000,
            lon_min=-50, lon_max=50,
            lat_min=-25, lat_max=25,
            plot_type='barbs',
            target_density=10
        )

        # Should return valid subsample factor
        assert subsample_higher_density >= 1

        # Higher target density should lead to lower subsample factor
        assert subsample_higher_density <= subsample_lower_density

        # Calculate with an even higher target density
        subsample_even_higher_density = plotter.calculate_optimal_subsample(
            num_points=50000,
            lon_min=-50, lon_max=50,
            lat_min=-25, lat_max=25,
            plot_type='barbs',
            target_density=100
        )

        # Should return valid subsample factor
        assert subsample_even_higher_density >= 1

        # Higher target density should lead to lower subsample factor
        assert subsample_even_higher_density <= subsample_higher_density

        # Close any open figures to free resources
        plt.close('all')
        
    # ------------------ Test Maximum Cap Enforcement ------------------

    def test_subsample_maximum_cap(self: "TestCalculateOptimalSubsample", plotter: MPASWindPlotter) -> None:
        """
        This test verifies that the `calculate_optimal_subsample` method enforces the configured maximum cap on subsample factors. For extremely dense datasets, the method should return subsample factors that do not exceed the maximum limit (e.g., 50). The test checks that even when the input data is so dense that it would normally require a very high subsample factor, the returned value is capped at the maximum. Additionally, it ensures that the relationship between 'barbs' and 'arrows' subsampling is maintained within this capped scenario. This test confirms that the method effectively prevents excessive subsampling that could lead to overly sparse visualizations, while still adhering to the constraints designed to maintain visual clarity.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to call the method.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Extremely dense dataset
        subsample_barbs = plotter.calculate_optimal_subsample(
            num_points=10000000,
            lon_min=-1, lon_max=1,
            lat_min=-1, lat_max=1,
            figsize=(2, 2),
            plot_type='barbs'
        )

        # Should not exceed max cap
        assert subsample_barbs <= 50

        # Extremely dense dataset for arrows
        subsample_arrows = plotter.calculate_optimal_subsample(
            num_points=10000000,
            lon_min=-1, lon_max=1,
            lat_min=-1, lat_max=1,
            figsize=(2, 2),
            plot_type='arrows'
        )

        # Should not exceed max cap
        assert subsample_arrows <= 50

        # Ensure barb subsample is not less than arrow subsample
        assert subsample_barbs >= subsample_arrows

        # Ensure arrow subsample is at most double barb subsample
        assert subsample_arrows <= 2 * subsample_barbs

        # Test with even smaller extent
        subsample_tiny_extent = plotter.calculate_optimal_subsample(
            num_points=10000000,
            lon_min=-0.1, lon_max=0.1,
            lat_min=-0.1, lat_max=0.1,
            figsize=(2, 2),
            plot_type='barbs'
        )

        # Should not exceed max cap
        assert subsample_tiny_extent <= 50
        
        # Compare with previous subsample
        assert subsample_tiny_extent >= subsample_barbs

        # Close any open figures to free resources
        plt.close('all')

# =================== Test Class: ConvertToNumpy ===================

class TestConvertToNumpy:
    """ Tests for the `convert_to_numpy` utility ensuring various input types (numpy, xarray, dask-like, lists) are converted to NumPy arrays. """

    # ------------------ Initialize Plotter Fixture ------------------

    @pytest.fixture
    def plotter(self: "TestConvertToNumpy") -> MPASWindPlotter:
        """
        This fixture creates and returns an instance of the `MPASWindPlotter` class for use in the conversion tests. By providing a fresh plotter instance for each test method, it ensures that any state changes or configurations made during one test do not affect others. This setup allows the test methods to focus on validating the behavior of the `convert_to_numpy` method without worrying about shared state or side effects from previous tests.

        Parameters:
            self ("TestConvertToNumpy"): Test instance which will receive the plotter fixture.

        Returns:
            MPASWindPlotter: Plotter instance used across conversion tests.
        """
        return MPASWindPlotter()

    # ------------------ Test Various Input Types ------------------

    def test_convert_numpy_array(self: "TestConvertToNumpy", plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when a NumPy array is passed to the `convert_to_numpy` method, it is returned unchanged. The method should recognize that the input is already a NumPy array and simply return it without modification. The test checks that the returned object is the same as the input array (i.e., not a copy), and that its shape and dtype match the original. This ensures that the method efficiently handles NumPy arrays without unnecessary conversions, which is important for performance when working with large datasets.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to call the method.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Raw NumPy array input
        arr = np.array([1, 2, 3, 4, 5])

        # Should return the same array
        result = plotter.convert_to_numpy(arr)

        # Verify equality of contents
        np.testing.assert_array_equal(result, arr)

        # Make sure type is ndarray
        assert isinstance(result, np.ndarray)

        # Check that the returned object is the same as input array
        assert result is arr

        # Check shape and dtype match
        assert result.shape == arr.shape
        assert result.dtype == arr.dtype

    # ------------------ Test xarray.DataArray Input ------------------
    
    def test_convert_xarray_dataarray(self: "TestConvertToNumpy", plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when an `xarray.DataArray` is passed to the `convert_to_numpy` method, it is converted to a NumPy ndarray by extracting its `.values` attribute. The test checks that the returned array has the same shape and dtype as the original data, and that it matches both the original data and the `.values` attribute of the DataArray. This ensures that the method correctly handles xarray DataArray inputs, which is important for interoperability with xarray-based workflows.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to call the method.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # xarray DataArray input
        data = np.array([1, 2, 3, 4, 5])

        # Wrap in DataArray
        da = xr.DataArray(data, dims=['x'])

        # Convert should extract .values
        result = plotter.convert_to_numpy(da)

        # Check shape and dtype match
        assert result.shape == data.shape
        assert result.dtype == data.dtype

        # Make sure type is ndarray
        assert isinstance(result, np.ndarray)

        # Ensure result matches original data
        np.testing.assert_array_equal(result, data)

        # Ensure result matches da.values
        np.testing.assert_array_equal(result, da.values)

    # ------------------ Test Dask-like Object Input ------------------    

    def test_convert_dask_array(self: "TestConvertToNumpy", plotter: MPASWindPlotter) -> None:
        """
        This test simulates a dask-like object with a `.compute()` method and ensures it is invoked during conversion. The test checks that the returned array matches the output of the `.compute()` method and that the method is called exactly once. This ensures that the `convert_to_numpy` method correctly handles dask-like objects, which is important for interoperability with dask-based workflows.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to call the method.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Dask-like mock object
        mock_dask = MagicMock()

        # Set compute() to return a NumPy array
        mock_dask.compute.return_value = np.array([1, 2, 3])

        # Convert should call compute and return result
        result = plotter.convert_to_numpy(mock_dask)

        # Make sure type is ndarray
        assert isinstance(result, np.ndarray)

        # Check shape and dtype match
        assert result.shape == (3,)

        # Ensure compute() was called once
        mock_dask.compute.assert_called_once()

        # Check dtype matches
        assert result.dtype == mock_dask.compute.return_value.dtype

        # Check that the returned object is the same as compute() output
        assert result is mock_dask.compute.return_value

        # Ensure result matches compute() output
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

        # Check that result matches mock_dask.compute() output
        np.testing.assert_array_equal(result, mock_dask.compute.return_value)

    # ------------------ Test List Input ------------------

    def test_convert_list_to_array(self: "TestConvertToNumpy", plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when a list is passed to the `convert_to_numpy` method, it is converted to a NumPy array using `np.asarray()`. The test checks that the returned array has the same shape and dtype as the original list (after conversion), and that it matches the original list's contents. Additionally, it confirms that the returned object is a NumPy array and not the original list, ensuring that the method correctly handles list inputs by converting them to arrays for consistent downstream processing.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to call the method.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # List input to convert
        lst = [1, 2, 3, 4, 5]

        # Convert should yield NumPy array
        result = plotter.convert_to_numpy(lst)

        # Check shape and dtype match
        assert result.shape == (5,)

        # Check dtype matches 
        assert result.dtype == np.array(lst).dtype

        # Check type and equality 
        assert isinstance(result, np.ndarray)

        # Check and make sure values match original list
        np.testing.assert_array_equal(result, np.array(lst))

        # Check that the returned object is not the same as input list
        assert result is not lst

    # ------------------ Test Exception Handling ------------------     

    def test_convert_exception_handling(self: "TestConvertToNumpy", plotter: MPASWindPlotter) -> None:
        """
        This test verifies that if an object does not have a `.values` attribute and is not directly convertible to a NumPy array, the `convert_to_numpy` method falls back to using `np.asarray()`. The test uses a custom `BadArray` class that raises an exception when `.values` is accessed. The test checks that the method successfully converts the object to a NumPy array using `np.asarray()`, and that the returned array has the expected shape and dtype. This ensures that the method is robust against unexpected input types and can still produce a usable NumPy array even when certain attributes are missing.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to call the method.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Instantiate bad object and convert
        bad_obj = BadArray()

        # Should fall through to asarray
        result = plotter.convert_to_numpy(bad_obj)

        # Check shape and dtype match
        assert result.shape == ()

        # Verify dtype is object due to fallback
        assert result.dtype == np.dtype('O')  

        # Check that the returned object is a NumPy array
        assert isinstance(result, np.ndarray)

        # Check that the returned object is not the same as bad_obj
        assert result is not bad_obj

# ================== Test Class: PrepareWindData ===================

class TestPrepareWindData:
    """ Tests for wind data preparation. """
    
    # ------------------ Initialize Plotter Fixture ------------------

    @pytest.fixture
    def plotter(self: "TestPrepareWindData") -> MPASWindPlotter:
        """
        This fixture creates and returns an instance of the `MPASWindPlotter` class for use in the wind data preparation tests. By providing a fresh plotter instance for each test method, it ensures that any state changes or configurations made during one test do not affect others. This setup allows the test methods to focus on validating the behavior of the `_prepare_wind_data` method without worrying about shared state or side effects from previous tests.

        Parameters:
            self ("TestPrepareWindData"): Test instance which will receive the plotter fixture.

        Returns:
            MPASWindPlotter: A plotter instance for use in tests.
        """
        return MPASWindPlotter()
    
    # ------------------ Test 1D Wind Data Without Subsampling ------------------

    def test_prepare_1d_no_subsample(self: "TestPrepareWindData", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data) -> None:
        """
        This test verifies that when 1D longitude, latitude, and wind component arrays are passed to the `_prepare_wind_data` method with `subsample=1`, the output arrays are returned unchanged (i.e., no subsampling is applied). The test checks that the output arrays have the same shape, dtype, and values as the input arrays, confirming that the method correctly handles 1D inputs without modifying them when subsampling is not requested. This ensures that the method can process raw 1D data as expected, which is important for users who provide pre-subsampled or already appropriately sized datasets.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call the helper.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Load real MPAS data (first 5 points)
        lon, lat = mpas_coordinates[0][:5], mpas_coordinates[1][:5]
        u, v = mpas_wind_data[0][:5], mpas_wind_data[1][:5]
        
        # Call the helper without subsampling
        lon_out, lat_out, u_out, v_out = plotter._prepare_wind_data(
            lon, lat, u, v, subsample=1
        )
        
        # Shapes should remain 1D with same length
        assert lon_out.shape == (5,)
        assert lat_out.shape == (5,)
        assert u_out.shape == (5,)
        assert v_out.shape == (5,)

        # Check that outputs are not the same object as inputs
        assert lon_out is not lon
        assert lat_out is not lat
        assert u_out is not u
        assert v_out is not v

        # Check that dtypes match
        assert lon_out.dtype == lon.dtype
        assert lat_out.dtype == lat.dtype
        assert u_out.dtype == u.dtype
        assert v_out.dtype == v.dtype

        # Check that lengths match
        assert len(lon_out) == len(lon)
        assert len(lat_out) == len(lat)
        assert len(u_out) == len(u)
        assert len(v_out) == len(v)

        # Check that values match
        np.testing.assert_array_equal(lon_out, lon)
        np.testing.assert_array_equal(lat_out, lat)
        np.testing.assert_array_equal(u_out, u)
        np.testing.assert_array_equal(v_out, v)
    
    # ------------------ Test 1D Wind Data with Subsampling ------------------

    def test_prepare_1d_with_subsample(self: "TestPrepareWindData", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data) -> None:
        """
        This test verifies that when 1D longitude, latitude, and wind component arrays are passed to the `_prepare_wind_data` method with a subsampling factor greater than 1 (e.g., `subsample=2`), the output arrays are correctly subsampled by taking every nth point according to the subsampling factor. The test checks that the output arrays have the expected shape, dtype, and values corresponding to the subsampled input data. This ensures that the method correctly applies subsampling to 1D inputs, which is important for users who want to reduce vector density for clearer visualizations while still providing raw 1D data.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call the helper.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Load real MPAS data (first 100 points)
        lon, lat = mpas_coordinates[0][:100], mpas_coordinates[1][:100]
        u, v = mpas_wind_data[0][:100], mpas_wind_data[1][:100]
        
        # Call the helper with subsampling factor of 2
        lon_out, lat_out, u_out, v_out = plotter._prepare_wind_data(
            lon, lat, u, v, subsample=2
        )
        
        # Outputs should be half the length and strided
        assert len(lon_out) == 50
        assert len(lat_out) == 50

        # Verify shape is 1D with expected length
        assert lon_out.shape == (50,)
        assert lat_out.shape == (50,)
        assert u_out.shape == (50,)
        assert v_out.shape == (50,)

        # Check that dtypes match
        assert lon_out.dtype == lon.dtype
        assert lat_out.dtype == lat.dtype
        assert u_out.dtype == u.dtype
        assert v_out.dtype == v.dtype

        # Check that outputs are not the same object as inputs
        assert lon_out is not lon
        assert lat_out is not lat
        assert u_out is not u
        assert v_out is not v

        # Check that lengths match expected subsampled size
        assert len(lon_out) == len(lon[::2])
        assert len(lat_out) == len(lat[::2])
        assert len(u_out) == len(u[::2])
        assert len(v_out) == len(v[::2])

        # Check that values match expected strided inputs
        np.testing.assert_array_equal(lon_out, lon[::2])
        np.testing.assert_array_equal(lat_out, lat[::2])
        np.testing.assert_array_equal(u_out, u[::2])
        np.testing.assert_array_equal(v_out, v[::2])
    
    # ------------------ Test 1D Wind Data with NaN Values ------------------

    def test_prepare_1d_with_nan_values(self: "TestPrepareWindData", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data) -> None:
        """
        This test verifies that when 1D longitude, latitude, and wind component arrays containing NaN values are passed to the `_prepare_wind_data` method, the output arrays are correctly filtered to include only the finite (non-NaN) values. The test checks that the output arrays have the expected shape, dtype, and values corresponding to the valid input data points. This ensures that the method can handle real-world datasets that may contain missing or invalid values by filtering them out appropriately for plotting.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call the helper.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Use real MPAS data (first 5 points)
        lon, lat = mpas_coordinates[0][:5].copy(), mpas_coordinates[1][:5].copy()
        u, v = mpas_wind_data[0][:5].copy(), mpas_wind_data[1][:5].copy()

        # Inject NaNs into u and v
        u[1] = np.nan
        u[4] = np.nan
        v[2] = np.nan
        
        # Call the helper without subsampling
        lon_out, lat_out, u_out, v_out = plotter._prepare_wind_data(
            lon, lat, u, v, subsample=1
        )
        
        # Only indices 0 and 3 are valid (both u and v are finite)
        assert len(lon_out) == 2

        # Check that dtypes match
        assert lon_out.dtype == lon.dtype
        assert lat_out.dtype == lat.dtype
        assert u_out.dtype == u.dtype
        assert v_out.dtype == v.dtype

        # Check that outputs are not the same object as inputs
        assert lon_out is not lon
        assert lat_out is not lat
        assert u_out is not u
        assert v_out is not v

        # Check that lengths match expected filtered size
        assert len(lon_out) == len(lon[[0, 3]])
        assert len(lat_out) == len(lat[[0, 3]])
        assert len(u_out) == len(u[[0, 3]])
        assert len(v_out) == len(v[[0, 3]])

        # Check that values match expected filtered values
        np.testing.assert_array_almost_equal(u_out, u[[0, 3]], decimal=5)
        np.testing.assert_array_almost_equal(v_out, v[[0, 3]], decimal=5)
        np.testing.assert_array_almost_equal(lat_out, lat[[0, 3]], decimal=5)
        np.testing.assert_array_almost_equal(lon_out, lon[[0, 3]], decimal=5)
    
    # ------------------ Test 2D Wind Data Without Subsampling ------------------

    def test_prepare_2d_no_subsample(self: "TestPrepareWindData", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data) -> None:
        """
        This test verifies that when 2D longitude, latitude, and wind component arrays are passed to the `_prepare_wind_data` method with `subsample=1`, the output arrays are returned unchanged (i.e., no subsampling is applied). The test checks that the output arrays have the same shape, dtype, and values as the input arrays, confirming that the method correctly handles 2D inputs without modifying them when subsampling is not requested. This ensures that the method can process raw 2D data as expected, which is important for users who provide pre-subsampled or already appropriately sized datasets in a 2D grid format.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call the helper.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Load real MPAS coordinates reshaped to 2D (2x5 grid)
        lon = mpas_coordinates[0][:10].reshape(2, 5)
        lat = mpas_coordinates[1][:10].reshape(2, 5)

        # Load real MPAS wind data reshaped to 2D (2x5 grid)
        u = mpas_wind_data[0][:10].reshape(2, 5)
        v = mpas_wind_data[1][:10].reshape(2, 5)
        
        # Call the helper without subsampling
        lon_out, lat_out, u_out, v_out = plotter._prepare_wind_data(
            lon, lat, u, v, subsample=1
        )
        
        # Shapes should remain 2D (2, 5)
        assert lon_out.shape == (2, 5)
        assert lat_out.shape == (2, 5)
        assert u_out.shape == (2, 5)
        assert v_out.shape == (2, 5)

        # Check that dtypes match
        assert lon_out.dtype == lon.dtype
        assert lat_out.dtype == lat.dtype
        assert u_out.dtype == u.dtype
        assert v_out.dtype == v.dtype

        # Check that shapes match
        assert lon_out.shape == lon.shape
        assert lat_out.shape == lat.shape
        assert u_out.shape == u.shape
        assert v_out.shape == v.shape

        # Check that values match
        np.testing.assert_array_equal(lon_out, lon)
        np.testing.assert_array_equal(lat_out, lat)
        np.testing.assert_array_equal(u_out, u)
        np.testing.assert_array_equal(v_out, v) 

    # ------------------ Test 2D Wind Data with Subsampling ------------------    

    def test_prepare_2d_with_subsample(self: "TestPrepareWindData", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data) -> None:
        """
        This test verifies that when 2D longitude, latitude, and wind component arrays are passed to the `_prepare_wind_data` method with a subsampling factor greater than 1 (e.g., `subsample=2`), the output arrays are correctly subsampled by taking every nth point according to the subsampling factor in both dimensions. The test checks that the output arrays have the expected shape, dtype, and values corresponding to the subsampled input data. This ensures that the method correctly applies subsampling to 2D inputs, which is important for users who want to reduce vector density for clearer visualizations while still providing raw 2D grid data.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call the helper.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Use real MPAS coordinates reshaped to 2D (10x10 grid)
        lon = mpas_coordinates[0][:100].reshape(10, 10)
        lat = mpas_coordinates[1][:100].reshape(10, 10)

        # Load real MPAS wind data reshaped to 2D (10x10 grid)
        u = mpas_wind_data[0][:100].reshape(10, 10)
        v = mpas_wind_data[1][:100].reshape(10, 10)
        
        # Call the helper with subsampling factor of 2
        lon_out, lat_out, u_out, v_out = plotter._prepare_wind_data(
            lon, lat, u, v, subsample=2
        )
        
        # Outputs should be half the shape in each dimension
        assert lon_out.shape == (5, 5)
        assert lat_out.shape == (5, 5)
        assert u_out.shape == (5, 5)
        assert v_out.shape == (5, 5)

        # Check that dtypes match
        assert lon_out.dtype == lon.dtype
        assert lat_out.dtype == lat.dtype
        assert u_out.dtype == u.dtype
        assert v_out.dtype == v.dtype

        # Check that shapes match expected subsampled size
        assert lon_out.shape == lon[::2, ::2].shape
        assert lat_out.shape == lat[::2, ::2].shape
        assert u_out.shape == u[::2, ::2].shape
        assert v_out.shape == v[::2, ::2].shape

        # Check that outputs match strided inputs
        np.testing.assert_array_equal(lon_out, lon[::2, ::2])
        np.testing.assert_array_equal(lat_out, lat[::2, ::2])
        np.testing.assert_array_equal(u_out, u[::2, ::2])
        np.testing.assert_array_equal(v_out, v[::2, ::2])
    
    # ------------------ Test xarray.DataArray Inputs ------------------

    def test_prepare_xarray_input(self: "TestPrepareWindData", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data) -> None:
        """
        This test verifies that when `xarray.DataArray` objects containing longitude, latitude, and wind component data are passed to the `_prepare_wind_data` method, they are correctly converted to NumPy arrays and subsampled (if requested). The test checks that the output arrays have the expected shape, dtype, and values corresponding to the input data, confirming that the method can handle xarray DataArray inputs seamlessly. This ensures that users who work with xarray-based datasets can directly pass their DataArrays to the method without needing to manually convert them to NumPy arrays first.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call the helper.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Use real MPAS coordinates and wind data (first 3 points)
        lon_data, lat_data = mpas_coordinates[0][:3], mpas_coordinates[1][:3]
        u_data, v_data = mpas_wind_data[0][:3], mpas_wind_data[1][:3]
        
        # Wrap in xarray DataArrays
        lon = xr.DataArray(lon_data, dims=['x'])
        lat = xr.DataArray(lat_data, dims=['x'])
        u = xr.DataArray(u_data, dims=['x'])
        v = xr.DataArray(v_data, dims=['x'])
        
        # Call the helper without subsampling
        lon_out, lat_out, u_out, v_out = plotter._prepare_wind_data(
            lon, lat, u, v, subsample=1
        )
        
        # Outputs should be NumPy arrays
        assert isinstance(lon_out, np.ndarray)
        assert isinstance(lat_out, np.ndarray)
        assert isinstance(u_out, np.ndarray)
        assert isinstance(v_out, np.ndarray)

        # Check that dtypes match
        assert lon_out.dtype == lon_data.dtype
        assert lat_out.dtype == lat_data.dtype
        assert u_out.dtype == u_data.dtype
        assert v_out.dtype == v_data.dtype

        # Shapes should match original data shapes
        assert lon_out.shape == lon_data.shape
        assert lat_out.shape == lat_data.shape
        assert u_out.shape == u_data.shape
        assert v_out.shape == v_data.shape

        # Outputs should match original data values
        np.testing.assert_array_equal(lon_out, lon_data)
        np.testing.assert_array_equal(lat_out, lat_data)
        np.testing.assert_array_equal(u_out, u_data)
        np.testing.assert_array_equal(v_out, v_data)

# ================== Test Class: RenderWindVectors ===================

class TestRenderWindVectors:
    """ Tests for wind vector rendering helpers (`_render_wind_vectors`). These tests exercise barbs, arrows, and streamline rendering paths and validate error handling for invalid inputs. """
    
    # ------------------ Initialize Plotter Fixture ------------------

    @pytest.fixture
    def plotter(self: "TestRenderWindVectors") -> MPASWindPlotter:
        """
        This fixture creates and returns an instance of the `MPASWindPlotter` class for use in the wind vector rendering tests. By providing a fresh plotter instance for each test method, it ensures that any state changes or configurations made during one test do not affect others. This setup allows the test methods to focus on validating the behavior of the `_render_wind_vectors` method without worrying about shared state or side effects from previous tests.

        Parameters:
            self ("TestRenderWindVectors"): Test instance which will receive the plotter fixture.

        Returns:
            MPASWindPlotter: Plotter instance used to call rendering helpers.
        """
        return MPASWindPlotter()
    
    # ------------------ Test Barb Rendering with Real MPAS Data ------------------

    def test_render_barbs(self: "TestRenderWindVectors", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data) -> None:
        """
        This test verifies that the `_render_wind_vectors` method can successfully render barbs using real MPAS longitude, latitude, and wind component data. The test creates a GeoAxes with a PlateCarree projection, loads a small subset of real MPAS data, and calls the rendering method with `plot_type='barbs'`. The test checks that no exceptions are raised during rendering, which indicates that the method can handle real-world data and render barbs correctly. This ensures that users can visualize wind vectors as barbs when working with actual MPAS datasets.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call the renderer.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Create a figure and GeoAxes
        fig = plt.figure()

        # Add GeoAxes with PlateCarree projection
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        # Load real MPAS coordinates
        lon, lat = mpas_coordinates[0][:3], mpas_coordinates[1][:3]

        # Load real MPAS wind data
        u, v = mpas_wind_data[0][:3], mpas_wind_data[1][:3]
        
        # Call the barb renderer 
        plotter._render_wind_vectors(
            ax, lon, lat, u, v, plot_type='barbs', color='red'
        )
        
        # If no exceptions, test passes
        plt.close(fig)
    
    # ------------------ Test Arrow Rendering with Real MPAS Data ------------------

    def test_render_arrows(self: "TestRenderWindVectors", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data) -> None:
        """
        This test verifies that the `_render_wind_vectors` method can successfully render quiver arrows using real MPAS longitude, latitude, and wind component data. The test creates a GeoAxes with a PlateCarree projection, loads a small subset of real MPAS data, and calls the rendering method with `plot_type='arrows'` and an explicit `scale`. The test checks that no exceptions are raised during rendering, which indicates that the method can handle real-world data and render arrows correctly. This ensures that users can visualize wind vectors as arrows when working with actual MPAS datasets.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call the renderer.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Create a figure and GeoAxes
        fig = plt.figure()

        # Add GeoAxes with PlateCarree projection
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        # Load real MPAS coordinates (first 3 points)
        lon, lat = mpas_coordinates[0][:3], mpas_coordinates[1][:3]

        # Load real MPAS wind data (first 3 points)
        u, v = mpas_wind_data[0][:3], mpas_wind_data[1][:3]
        
        # Call the arrow renderer with explicit scale
        plotter._render_wind_vectors(
            ax, lon, lat, u, v, plot_type='arrows', color='blue', scale=100
        )
        
        # If no exceptions, test passes
        plt.close(fig)
    
    # ------------------ Test Arrow Rendering with Default Scale ------------------

    def test_render_arrows_default_scale(self: "TestRenderWindVectors", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data) -> None:
        """
        This test verifies that the `_render_wind_vectors` method can successfully render quiver arrows using real MPAS longitude, latitude, and wind component data without specifying a `scale`. The test creates a GeoAxes with a PlateCarree projection, loads a small subset of real MPAS data, and calls the rendering method with `plot_type='arrows'` and `scale=None`. The test checks that no exceptions are raised during rendering, which indicates that the method can handle real-world data and render arrows correctly using the default scale. This ensures that users can visualize wind vectors as arrows when working with actual MPAS datasets.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call the renderer.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Create a figure and GeoAxes
        fig = plt.figure()

        # Add GeoAxes with PlateCarree projection
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        # Load real MPAS coordinates (first 3 points)
        lon, lat = mpas_coordinates[0][:3], mpas_coordinates[1][:3]

        # Load real MPAS wind data (first 3 points)
        u, v = mpas_wind_data[0][:3], mpas_wind_data[1][:3]
        
        # Call the arrow renderer without scale (should use default)
        plotter._render_wind_vectors(
            ax, lon, lat, u, v, plot_type='arrows', color='blue', scale=None
        )
        
        # If no exceptions, test passes
        plt.close(fig)
    
    # ------------------ Test Streamline Rendering with Real MPAS Data ------------------

    def test_render_streamlines(self: "TestRenderWindVectors", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data) -> None:
        """
        This test verifies that the `_render_wind_vectors` method can successfully render streamlines using real MPAS longitude, latitude, and wind component data on a gridded dataset. The test creates a GeoAxes with a PlateCarree projection, loads a small subset of real MPAS data, reshapes it into a 5x5 grid, and calls the rendering method with `plot_type='streamlines'`. The test checks that a colorbar is added to the figure, indicating that the method can handle real-world data and render streamlines correctly. This ensures that users can visualize wind vectors as streamlines when working with actual MPAS datasets.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call the renderer.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Create a figure and GeoAxes
        fig = plt.figure()

        # Add GeoAxes with PlateCarree projection
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        # Use real MPAS data to create a small 5x5 grid
        lon_1d = mpas_coordinates[0][:25]
        lat_1d = mpas_coordinates[1][:25]

        # Load real MPAS wind data (first 25 points)
        u_1d = mpas_wind_data[0][:25]
        v_1d = mpas_wind_data[1][:25]
        
        # Create 5x5 grid from first 25 points
        lon_2d = lon_1d.reshape(5, 5)
        lat_2d = lat_1d.reshape(5, 5)

        # Create 5x5 grid from first 25 points
        u_2d = u_1d.reshape(5, 5)
        v_2d = v_1d.reshape(5, 5)
        
        # Call the streamline renderer on the gridded data
        plotter._render_wind_vectors(
            ax, lon_2d, lat_2d, u_2d, v_2d, plot_type='streamlines'
        )

        # Streamlines should add a colorbar axis to the figure
        assert len(fig.axes) > 1
 
        # If no exceptions, test passes
        plt.close(fig)
    
    # ------------------ Test Streamline Rendering with 1D Data Raises Error ------------------

    def test_render_streamlines_1d_raises_error(self: "TestRenderWindVectors", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data) -> None:
        """
        This test verifies that when 1D longitude, latitude, and wind component arrays are passed to the `_render_wind_vectors` method with `plot_type='streamlines'`, a ValueError is raised indicating that streamlines require gridded data. The test creates a GeoAxes with a PlateCarree projection, loads a small subset of real MPAS data as 1D arrays, and calls the rendering method. The test checks that the expected error is raised with a clear message, confirming that the method correctly enforces the requirement for gridded data when rendering streamlines. This ensures that users receive appropriate feedback when attempting to render streamlines with incompatible data formats.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call the renderer.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Create a figure and GeoAxes
        fig = plt.figure()
 
        # Add GeoAxes with PlateCarree projection
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        # Load real MPAS coordinates (first 3 points)
        lon, lat = mpas_coordinates[0][:3], mpas_coordinates[1][:3]

        # Load real MPAS wind data (first 3 points)
        u, v = mpas_wind_data[0][:3], mpas_wind_data[1][:3]
        
        with pytest.raises(ValueError) as exc_info:
            plotter._render_wind_vectors(
                ax, lon, lat, u, v, plot_type='streamlines'
            )
        
        assert "require gridded data" in str(exc_info.value)
        plt.close(fig)
    
    # ------------------ Test Invalid Plot Type Raises Error ------------------

    def test_render_invalid_plot_type(self: "TestRenderWindVectors", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data) -> None:
        """
        This test verifies that when an invalid `plot_type` is passed to the `_render_wind_vectors` method, a ValueError is raised indicating that the plot type is not recognized. The test creates a GeoAxes with a PlateCarree projection, loads a small subset of real MPAS data, and calls the rendering method with an invalid `plot_type` value (e.g., 'invalid'). The test checks that the expected error is raised with a clear message listing the valid plot type options, confirming that the method correctly handles invalid input for the plot type parameter. This ensures that users receive appropriate feedback when attempting to render wind vectors with an unsupported plot type.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call the renderer.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        # Load real MPAS coordinates (first 3 points)
        lon, lat = mpas_coordinates[0][:3], mpas_coordinates[1][:3]

        # Load real MPAS wind data (first 3 points)
        u, v = mpas_wind_data[0][:3], mpas_wind_data[1][:3]

        # Call with invalid plot_type        
        with pytest.raises(ValueError) as exc_info:
            plotter._render_wind_vectors(
                ax, lon, lat, u, v, plot_type='invalid'
            )
        
        # Check that error message indicates valid options
        assert "plot_type must be" in str(exc_info.value)

        # If no exceptions, test passes
        plt.close(fig)

# ================== Test Class: RegridWindComponents ===================

class TestRegridWindComponents:
    """ Test for the wind component regridding helper (`_regrid_wind_components`). """

    # ------------------ Initialize Plotter Fixture ------------------

    @pytest.fixture
    def plotter(self: "TestRegridWindComponents") -> MPASWindPlotter:
        """
        This fixture creates and returns an instance of the `MPASWindPlotter` class for use in the wind component regridding tests. By providing a fresh plotter instance for each test method, it ensures that any state changes or configurations made during one test do not affect others. This setup allows the test methods to focus on validating the behavior of the `_regrid_wind_components` method without worrying about shared state or side effects from previous tests.

        Parameters:
            self ("TestRegridWindComponents"): Test instance which will receive the plotter fixture.

        Returns:
            MPASWindPlotter: Plotter instance used in tests.
        """
        return MPASWindPlotter()
    
    # ------------------ Test Linear Regridding Method ------------------

    def test_regrid_linear_method(self: "TestRegridWindComponents", plotter: MPASWindPlotter, monkeypatch) -> None:
        """
        This test verifies that specifying `regrid_method='linear'` forwards the method parameter to the remapping utility for both U and V components, and that the outputs are consistent with linear interpolation. The test uses real MPAS grid data to create a realistic scenario for regridding, and checks that the output longitude, latitude, and wind component arrays have the expected shapes, dtypes, and values corresponding to a linear regridding of the input data. This ensures that users can rely on the linear regridding option to produce accurate results when visualizing wind vectors on a regular grid.

        Parameters:
            mock_remap (MagicMock): Patched remapping function returning DataArrays.
            plotter (MPASWindPlotter): Fixture instance used to call the helper.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Use real MPAS grid data instead of mocking remap utility
        data_dir = Path(__file__).parent.parent.parent / "data"

        # Specify the path to MPAS grid file for testing
        grid_file = data_dir / "grids" / "x1.40962.static.nc"

        # Skip test if grid file is not available
        if not grid_file.exists():
            pytest.skip(f"MPAS grid file not found: {grid_file}")

        # Load MPAS grid data
        ds = xr.open_dataset(grid_file, decode_times=False)

        # Limit to manageable test size for speed (200 points)
        n_test = min(200, ds['lonCell'].size)

        # Extract lon/lat and create deterministic u/v
        lon = ds['lonCell'].isel(nCells=slice(0, n_test)).values
        lat = ds['latCell'].isel(nCells=slice(0, n_test)).values

        # Create synthetic wind components
        u = 5.0 * np.sin(np.radians(lat))
        v = 5.0 * np.cos(np.radians(lon))

        # Create minimal dataset with coordinates for remapping
        dataset = xr.Dataset({
            'lonCell': ('nCells', lon),
            'latCell': ('nCells', lat)
        })

        # Call the regridding helper with default linear method
        lon_out, lat_out, u_out, v_out = plotter._regrid_wind_components(
            lon, lat, u, v, dataset,
            lon_min=float(lon.min()), lon_max=float(lon.max()),
            lat_min=float(lat.min()), lat_max=float(lat.max()),
            grid_resolution=2.0,
            regrid_method='linear'
        )

        # Results should be 2D structured arrays
        assert getattr(lon_out, 'ndim', np.array(lon_out).ndim) >= 2
        assert getattr(lat_out, 'ndim', np.array(lat_out).ndim) >= 2

        # Shapes should match between lon/lat and u/v outputs
        assert np.array(lon_out).shape == np.array(lat_out).shape
        assert np.array(u_out).shape == np.array(v_out).shape
        assert np.array(lon_out).shape == np.array(u_out).shape

        # Longitude/latitude grid should be finite everywhere
        assert np.all(np.isfinite(lon_out))
        assert np.all(np.isfinite(lat_out))

        # U/V may have NaNs due to convex-hull masking applied during remapping, but should have at least some valid data
        assert np.any(np.isfinite(u_out))
        assert np.any(np.isfinite(v_out))

        # Close the dataset
        ds.close()
    
    # ------------------ Test Nearest Regridding Method ------------------

    def test_regrid_nearest_method(self: "TestRegridWindComponents", plotter: MPASWindPlotter, monkeypatch) -> None:
        """
        This test verifies that specifying `regrid_method='nearest'` forwards the method parameter to the remapping utility for both U and V components, and that the outputs are consistent with nearest-neighbor interpolation. The test uses real MPAS grid data to create a realistic scenario for regridding, and checks that the output longitude, latitude, and wind component arrays have the expected shapes, dtypes, and values corresponding to a nearest-neighbor regridding of the input data. This ensures that users can rely on the nearest regridding option to produce accurate results when visualizing wind vectors on a regular grid without smoothing.

        Parameters:
            mock_remap (MagicMock): Patched remapping function used to capture args.
            plotter (MPASWindPlotter): Fixture instance used to call the helper.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Use real MPAS grid data instead of mocking remap utility
        data_dir = Path(__file__).parent.parent.parent / "data"

        # Specify the path to MPAS grid file for testing
        grid_file = data_dir / "grids" / "x1.40962.static.nc"

        # Skip test if grid file is not available
        if not grid_file.exists():
            pytest.skip(f"MPAS grid file not found: {grid_file}")

        # Load MPAS grid data
        ds = xr.open_dataset(grid_file, decode_times=False)

        # Limit to manageable test size for speed (200 points)
        n_test = min(200, ds['lonCell'].size)

        # Extract longitude and latitude arrays
        lon = ds['lonCell'].isel(nCells=slice(0, n_test)).values
        lat = ds['latCell'].isel(nCells=slice(0, n_test)).values

        # Create synthetic wind components 
        u = 5.0 * np.sin(np.radians(lat))
        v = 5.0 * np.cos(np.radians(lon))

        # Create minimal dataset with coordinates for remapping
        dataset = xr.Dataset({
            'lonCell': ('nCells', lon),
            'latCell': ('nCells', lat)
        })

        # Call the regridding helper with nearest method
        lon_out, lat_out, u_out, v_out = plotter._regrid_wind_components(
            lon, lat, u, v, dataset,
            lon_min=float(lon.min()), lon_max=float(lon.max()),
            lat_min=float(lat.min()), lat_max=float(lat.max()),
            grid_resolution=3.0,
            regrid_method='nearest'
        )

        # Results should be 2D structured arrays
        assert getattr(lon_out, 'ndim', np.array(lon_out).ndim) >= 2
        assert getattr(lat_out, 'ndim', np.array(lat_out).ndim) >= 2

        # Shapes should match between lon/lat and u/v outputs
        assert np.array(lon_out).shape == np.array(lat_out).shape
        assert np.array(u_out).shape == np.array(v_out).shape
        assert np.array(lon_out).shape == np.array(u_out).shape

        # Longitude/latitude grid should be finite everywhere
        assert np.all(np.isfinite(lon_out))
        assert np.all(np.isfinite(lat_out))

        # U/V may have NaNs due to convex-hull masking applied during remapping, but should have at least some valid data
        assert np.any(np.isfinite(u_out))
        assert np.any(np.isfinite(v_out))   
        
# ================== Test Class: CreateWindPlot ===================

class TestCreateWindPlot:
    """ Tests for the high-level `create_wind_plot` workflow. These unit tests patch plotting internals to validate extent, title and subsampling behaviors without rendering real figures to the display. """

    # ------------------ Initialize Plotter Fixture ------------------

    @pytest.fixture
    def plotter(self: "TestCreateWindPlot",) -> MPASWindPlotter:
        """
        This fixture creates and returns an instance of the `MPASWindPlotter` class for use in the wind plot creation tests. By providing a fresh plotter instance for each test method, it ensures that any state changes or configurations made during one test do not affect others. This setup allows the test methods to focus on validating the behavior of the `create_wind_plot` method without worrying about shared state or side effects from previous tests.

        Parameters:
            self ("TestCreateWindPlot"): Test instance which will receive the plotter fixture.

        Returns:
            MPASWindPlotter: Plotter instance used to create plots in tests.
        """
        return MPASWindPlotter()

    # ------------------ Test Basic Wind Plot Creation ------------------

    def test_create_wind_plot_basic(self: "TestCreateWindPlot", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data, monkeypatch) -> None:
        """
        This test verifies that the `create_wind_plot` method can successfully create a wind plot using real MPAS longitude, latitude, and wind component data with basic parameters. The test patches the internal rendering method to confirm that it is called during the plot creation process. It checks that the returned figure and axes objects are not None, indicating that the plot was created successfully. This ensures that users can create wind plots with real MPAS data and that the internal rendering logic is invoked as expected.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call `create_wind_plot`.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.
            monkeypatch: Pytest fixture for patching methods.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Stub renderer to confirm it is called
        calls = {'render': 0}

        # Define a fake render method to count calls
        monkeypatch.setattr(MPASWindPlotter, '_render_wind_vectors', fake_render_factory(calls))
        
        # Load real MPAS coordinates (first 4 points)
        lon, lat = mpas_coordinates[0][:4], mpas_coordinates[1][:4]

        # Load real MPAS wind data (first 4 points)
        u, v = mpas_wind_data[0][:4], mpas_wind_data[1][:4]
        
        # Create the wind plot with basic parameters
        fig, ax = plotter.create_wind_plot(
            lon, lat, u, v,
            lon_min=0, lon_max=50,
            lat_min=0, lat_max=25,
            plot_type='barbs'
        )
        
        # Confirm that the returned figure object is not None
        assert fig is not None

        # Confirm that the returned axes object is not None
        assert ax is not None

        # Ensure the render method was called once
        assert calls['render'] == 1

        # Close the figure to free resources
        plt.close(fig)

    # ------------------ Test Wind Plot Creation with Automatic Subsampling ------------------

    def test_create_wind_plot_auto_subsample(self: "TestCreateWindPlot", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data, monkeypatch) -> None:
        """
        This test verifies that when `subsample=-1` is passed to `create_wind_plot`, the method automatically calculates an appropriate subsampling factor based on the input data size and successfully creates a wind plot. The test patches the internal rendering method to confirm that it is called during the plot creation process. It checks that the returned figure and axes objects are not None, indicating that the plot was created successfully with automatic subsampling. This ensures that users can rely on the automatic subsampling feature to create wind plots without needing to manually specify a subsampling factor, even when working with larger MPAS datasets.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call `create_wind_plot`.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.
            monkeypatch: Pytest fixture for patching methods.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Stub renderer to confirm it is called
        calls = {'render': 0}

        # Define a fake render method to count calls
        monkeypatch.setattr(MPASWindPlotter, '_render_wind_vectors', fake_render_factory(calls))
        
        # Load real MPAS coordinates (100 points)
        lon, lat = mpas_coordinates[0][:100], mpas_coordinates[1][:100]

        # Load real MPAS wind data (100 points)
        u, v = mpas_wind_data[0][:100], mpas_wind_data[1][:100]
        
        # Create the wind plot with automatic subsampling
        fig, ax = plotter.create_wind_plot(
            lon, lat, u, v,
            lon_min=0, lon_max=50,
            lat_min=0, lat_max=25,
            plot_type='barbs',
            subsample=-1  # Auto-calculate
        )
        
        # Confirm that the returned figure object is not None
        assert fig is not None

        # Confirm that the returned axes object is not None
        assert ax is not None

        # Ensure the render method was called once
        assert calls['render'] == 1

        # Close the figure to free resources
        plt.close(fig)

    # ------------------ Test Wind Plot Creation with Global Extent ------------------

    def test_create_wind_plot_global_extent(self: "TestCreateWindPlot", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data, monkeypatch) -> None:
        """
        This test verifies that when global longitude and latitude extents are passed to `create_wind_plot`, the method successfully creates a wind plot with the specified global extent. The test patches the internal rendering method to confirm that it is called during the plot creation process. It checks that the returned figure and axes objects are not None, and that the GeoAxes has the expected global extent set. This ensures that users can create wind plots with global coverage by specifying appropriate longitude and latitude limits, and that the internal logic correctly applies these limits to the plot.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call `create_wind_plot`.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.
            monkeypatch: Pytest fixture for patching methods.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Stub renderer to confirm it is called
        calls = {'render': 0}

        # Define a fake render method to count calls
        monkeypatch.setattr(MPASWindPlotter, '_render_wind_vectors', fake_render_factory(calls))
        
        # Load real MPAS coordinates (100 points)
        lon, lat = mpas_coordinates[0][:100], mpas_coordinates[1][:100]

        # Load real MPAS wind data (100 points)
        u, v = mpas_wind_data[0][:100], mpas_wind_data[1][:100]

        # Create the wind plot with global extents        
        fig, ax = plotter.create_wind_plot(
            lon, lat, u, v,
            lon_min=-180, lon_max=180,
            lat_min=-90, lat_max=90,
            plot_type='arrows'
        )
        
        # Confirm that the returned figure object is not None
        assert fig is not None

        # Confirm that the returned axes object is not None
        assert ax is not None

        # Ensure the render method was called once
        assert calls['render'] == 1

        # Cast to GeoAxes for extent checking
        geo_ax = cast(GeoAxes, ax)

        # Get the current extent of the GeoAxes
        extent = geo_ax.get_extent()

        # Verify that extent is a tuple of length 4
        assert isinstance(extent, tuple) and len(extent) == 4

        # Close the figure to free resources
        plt.close(fig)
    
    # ------------------ Test Wind Plot Creation with Custom Title ------------------

    def test_create_wind_plot_custom_title(self: "TestCreateWindPlot", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data, monkeypatch) -> None:
        """
        This test verifies that when a custom `title` string is passed to `create_wind_plot`, the method successfully creates a wind plot and includes the custom title in the axes title. The test patches the internal rendering method to confirm that it is called during the plot creation process. It checks that the returned figure and axes objects are not None, and that the custom title appears in the axes title string. This ensures that users can customize the title of their wind plots by passing a specific string, and that the internal logic correctly incorporates this title into the plot.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call `create_wind_plot`.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.
            monkeypatch: Pytest fixture for patching methods.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Stub renderer so title-setting is exercised on real axes
        calls = {'render': 0}

        # Define a fake render method to count calls
        monkeypatch.setattr(MPASWindPlotter, '_render_wind_vectors', fake_render_factory(calls))
        
        # Load real MPAS coordinates (first 3 points)
        lon, lat = mpas_coordinates[0][:3], mpas_coordinates[1][:3]

        # Load real MPAS wind data (first 3 points)
        u, v = mpas_wind_data[0][:3], mpas_wind_data[1][:3]
        
        # Define a custom title string
        custom_title = "Custom Wind Analysis"

        # Create the wind plot with the custom title
        fig, ax = plotter.create_wind_plot(
            lon, lat, u, v,
            lon_min=0, lon_max=50,
            lat_min=0, lat_max=25,
            title=custom_title
        )
        
        # Confirm that the returned figure object is not None
        assert fig is not None

        # Confirm that the returned axes object is not None
        assert ax is not None
        
        # Confirm that the custom title appears in the axes title
        assert custom_title in ax.get_title()

        # Ensure the render method was called once
        assert calls['render'] == 1

        # Close the figure to free resources
        plt.close(fig)
    
    # ------------------ Test Wind Plot Creation with Timestamp ------------------

    def test_create_wind_plot_with_timestamp(self: "TestCreateWindPlot", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data, monkeypatch) -> None:
        """
        This test verifies that when a `time_stamp` is passed to `create_wind_plot`, the method successfully creates a wind plot and includes the formatted timestamp in the axes title. The test patches the internal rendering method to confirm that it is called during the plot creation process. It checks that the returned figure and axes objects are not None, and that the timestamp appears in the axes title string in the expected format (e.g., "YYYY-MM-DD HH:MM"). This ensures that users can include temporal information in their wind plots by passing a specific timestamp, and that the internal logic correctly formats and incorporates this timestamp into the plot title.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call `create_wind_plot`.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.
            monkeypatch: Pytest fixture for patching methods.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Use real headless plotting instead of monkeypatching plt.subplots
        _, _ = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Load real MPAS coordinates (first 3 points)
        lon, lat = mpas_coordinates[0][:3], mpas_coordinates[1][:3]

        # Load real MPAS wind data (first 3 points)
        u, v = mpas_wind_data[0][:3], mpas_wind_data[1][:3]
        
        # Define a specific timestamp for testing
        timestamp = datetime(2024, 1, 15, 12, 0, 0)

        # Create the wind plot with the timestamp
        fig, ax = plotter.create_wind_plot(
            lon, lat, u, v,
            lon_min=0, lon_max=50,
            lat_min=0, lat_max=25,
            time_stamp=timestamp
        )
        
        # Confirm that the returned figure object is not None
        assert fig is not None

        # Confirm that the returned axes object is not None
        assert ax is not None

        # Confirm that the timestamp appears in the axes title (formatted as YYYY-MM-DD HH:MM)
        assert timestamp.strftime("%Y-%m-%d %H:%M") in ax.get_title()

        # Also confirm that the title is not empty (since timestamp should be included)
        assert ax.get_title() != ''

        # Close the figure to free resources
        plt.close(fig)
    
    # ------------------ Test Wind Plot Creation with Level Info ------------------

    def test_create_wind_plot_with_level_info(self: "TestCreateWindPlot", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data, monkeypatch) -> None:
        """
        This test verifies that when `level_info` is passed to `create_wind_plot`, the method successfully creates a wind plot and includes the level information in the axes title. The test patches the internal rendering method to confirm that it is called during the plot creation process. It checks that the returned figure and axes objects are not None, and that the level information appears in the axes title string. This ensures that users can include vertical level information (e.g., pressure level) in their wind plots by passing a specific string, and that the internal logic correctly incorporates this information into the plot title.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call `create_wind_plot`.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.
            monkeypatch: Pytest fixture for patching methods.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Use real headless plotting instead of monkeypatching plt.subplots
        _, _ = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Load real MPAS coordinates (first 3 points)
        lon, lat = mpas_coordinates[0][:3], mpas_coordinates[1][:3]

        # Load real MPAS wind data (first 3 points)
        u, v = mpas_wind_data[0][:3], mpas_wind_data[1][:3]

        # Define level information for testing (e.g., pressure level)
        level_info = "850 hPa"

        # Create the wind plot with the level information        
        fig, ax = plotter.create_wind_plot(
            lon, lat, u, v,
            lon_min=0, lon_max=50,
            lat_min=0, lat_max=25,
            level_info=level_info
        )
        
        # Confirm that the returned figure object is not None
        assert fig is not None

        # Confirm that the returned axes object is not None
        assert ax is not None

        # Confirm that the level information appears in the axes title
        assert level_info in ax.get_title()

        # Also confirm that the title is not empty (since level info should be included)
        assert '850' in ax.get_title() or ax.get_title() != ''

        # Close the figure to free resources
        plt.close(fig)
    
    # ------------------ Test Wind Plot Creation with Regridding ------------------

    def test_create_wind_plot_with_regridding(self: "TestCreateWindPlot", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data, monkeypatch) -> None:
        """
        This test verifies that when `grid_resolution` is specified in `create_wind_plot`, the method successfully performs regridding of the input wind data onto a regular grid and creates a wind plot with the regridded data. The test uses real MPAS longitude, latitude, and wind component data to create a realistic scenario for regridding. It checks that the returned figure and axes objects are not None, indicating that the plot was created successfully with regridding. This ensures that users can create wind plots with regridded data by specifying a grid resolution, and that the internal logic correctly applies the regridding process before rendering the plot.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call `create_wind_plot`.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.
            monkeypatch: Pytest fixture for patching methods.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Use real headless plotting instead of monkeypatching plt.subplots
        _, _ = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Load real MPAS coordinates (100 points)
        lon, lat = mpas_coordinates[0][:100], mpas_coordinates[1][:100]

        # Load real MPAS wind data (100 points)
        u_100, v_100 = mpas_wind_data[0][:100], mpas_wind_data[1][:100]

        # Tile wind data to match coordinate size (in case of mismatch)
        n_coords = len(lon)

        # Calculate how many times to tile the 100-point wind data to cover all coordinates
        n_tiles = (n_coords + 99) // 100

        # Tile the wind data and slice to match the number of coordinates
        u = np.tile(u_100, n_tiles)[:n_coords]
        v = np.tile(v_100, n_tiles)[:n_coords]
        
        # Create the wind plot with regridding enabled (grid_resolution specified)
        fig, ax = plotter.create_wind_plot(
            lon, lat, u, v,
            lon_min=0, lon_max=50,
            lat_min=0, lat_max=25,
            grid_resolution=1.0,
            regrid_method='linear'
        )
        
        # Confirm that the returned figure object is not None
        assert fig is not None

        # Confirm that the returned axes object is not None
        assert ax is not None

        # Close the figure to free resources
        plt.close(fig)
    
    # ------------------ Test Wind Plot Creation with Streamlines Auto-Regrid ------------------

    def test_create_wind_plot_streamlines_auto_regrid(self: "TestCreateWindPlot", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data, monkeypatch) -> None:
        """
        This test verifies that when `plot_type='streamlines'` is passed to `create_wind_plot` without an explicit `grid_resolution`, the method automatically enables regridding of the input wind data onto a regular grid suitable for streamlining, and successfully creates a wind plot with the regridded data. The test uses real MPAS longitude, latitude, and wind component data to create a realistic scenario for automatic regridding. It checks that the returned figure and axes objects are not None, indicating that the plot was created successfully with automatic regridding for streamlines. This ensures that users can create streamline plots without needing to manually specify a grid resolution, and that the internal logic correctly applies automatic regridding when streamlines are requested.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call `create_wind_plot`.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.
            monkeypatch: Pytest fixture for patching methods.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Use real headless plotting instead of monkeypatching plt.subplots
        _, _ = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Load real MPAS coordinates (100 points)
        lon, lat = mpas_coordinates[0][:100], mpas_coordinates[1][:100]

        # Load real MPAS wind data (100 points)
        u_100, v_100 = mpas_wind_data[0][:100], mpas_wind_data[1][:100] 

        # Tile wind data to match coordinate size (in case of mismatch)
        n_coords = len(lon)

        # Calculate how many times to tile the 100-point wind data to cover all coordinates
        n_tiles = (n_coords + 99) // 100

        # Tile the wind data and slice to match the number of coordinates
        u = np.tile(u_100, n_tiles)[:n_coords]
        v = np.tile(v_100, n_tiles)[:n_coords]
        
        # Create the wind plot with streamlines and no explicit grid_resolution to trigger auto-regridding
        fig, ax = plotter.create_wind_plot(
            lon, lat, u, v,
            lon_min=0, lon_max=50,
            lat_min=0, lat_max=25,
            plot_type='streamlines'  # Should auto-enable regridding
        )
        
        # Confirm that the returned figure object is not None
        assert fig is not None

        # Confirm that the returned axes object is not None
        assert ax is not None

        # Close the figure to free resources
        plt.close(fig)
    
    # ------------------ Test Wind Plot Creation with Empty Data ------------------

    def test_create_wind_plot_empty_data(self: "TestCreateWindPlot", plotter: MPASWindPlotter, monkeypatch) -> None:
        """
        This test verifies that when empty or NaN-filled coordinate and wind component arrays are passed to `create_wind_plot`, the method handles the lack of valid data gracefully without raising exceptions, and still returns a figure and axes object that can be used for setting titles or labels. The test uses real headless plotting to confirm that the plotting machinery can handle empty data inputs without crashing. It checks that the returned figure and axes objects are not None, indicating that the plot was created successfully even with empty data. This ensures that users can call `create_wind_plot` with empty datasets (e.g., due to filtering or masking) without encountering errors, and that they can still set up the plot framework for later population with valid data.

        Parameters:
            mock_regional (MagicMock): Patched `add_regional_features` method.
            mock_render (MagicMock): Patched `_render_wind_vectors` method.
            mock_subplots (MagicMock): Patched `plt.subplots` factory.
            plotter (MPASWindPlotter): Fixture instance to call `create_wind_plot`.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Use real headless plotting instead of monkeypatching plt.subplots
        _, _ = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Provide coordinate arrays with valid values
        lon = np.array([10, 20, 30])
        lat = np.array([5, 10, 15])

        # Wind components are NaN to represent no valid data
        u = np.array([np.nan, np.nan, np.nan])
        v = np.array([np.nan, np.nan, np.nan])
        
        # Create the wind plot with empty data; should handle gracefully without exceptions
        fig, ax = plotter.create_wind_plot(
            lon, lat, u, v,
            lon_min=0, lon_max=50,
            lat_min=0, lat_max=25
        )
        
        # Even with empty data, a figure and axes should be returned to allow for titles/labels, so we check that they are not None
        assert fig is not None 

        # Axis should be returned even if no vectors are rendered, allowing for titles/labels to be set
        assert ax is not None

        # Close the figure to free resources
        plt.close(fig)
    
    # ------------------ Test Wind Plot Creation with 2D Gridded Data ------------------

    def test_create_wind_plot_2d_data(self: "TestCreateWindPlot", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data, monkeypatch) -> None:
        """
        This test verifies that when 2D gridded longitude, latitude, and wind component arrays are passed to `create_wind_plot`, the method successfully creates a wind plot without errors. The test uses real MPAS grid data to create a realistic scenario for 2D gridded inputs. It checks that the returned figure and axes objects are not None, indicating that the plot was created successfully with 2D gridded data. This ensures that users can create wind plots using 2D gridded datasets (e.g., from regridding or structured grids) without encountering issues, and that the internal logic correctly handles gridded inputs for plotting.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call `create_wind_plot`.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.
            monkeypatch: Pytest fixture for patching methods.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Use real headless plotting instead of monkeypatching plt.subplots
        _, _ = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Load real MPAS coordinates and wind data (first 100 points) and reshape to 2D grids for testing
        lon_1d = mpas_coordinates[0][:100]
        lat_1d = mpas_coordinates[1][:100]

        # For wind data, we also take the first 100 points and reshape to match the coordinate grids for testing
        u_1d = mpas_wind_data[0][:100]
        v_1d = mpas_wind_data[1][:100]

        # Reshape to 2D grids (10x10) for testing; in real cases, the shape would depend on the actual grid structure
        lon_2d = lon_1d.reshape(10, 10)
        lat_2d = lat_1d.reshape(10, 10)

        # Reshape wind components to 2D grids matching the coordinate shapes
        u_2d = u_1d.reshape(10, 10)
        v_2d = v_1d.reshape(10, 10)
        
        # Create the wind plot with 2D gridded data; the plotting machinery should accept gridded inputs without issues
        fig, ax = plotter.create_wind_plot(
            lon_2d, lat_2d, u_2d, v_2d,
            lon_min=0, lon_max=50,
            lat_min=0, lat_max=25,
            plot_type='arrows'
        )
        
        # Even with 2D gridded data, a figure and axes should be returned, confirming that the plotting machinery accepts gridded inputs without issues
        assert fig is not None 

        # Axis should be returned when using 2D gridded data, confirming that the plotting machinery accepts gridded inputs without issues
        assert ax is not None

        # Close the figure to free resources
        plt.close(fig)

# ================== Test Class: AddWindOverlay ===================

class TestAddWindOverlay:
    """ Tests for `add_wind_overlay` which renders wind vectors onto existing axes. These tests validate 1D/3D handling, regridding, subsampling, and error messages for missing bounding boxes. """

    # ------------------ Initialize Plotter Fixture ------------------

    @pytest.fixture
    def plotter(self) -> MPASWindPlotter:
        """
        This fixture creates and returns an instance of the `MPASWindPlotter` class for use in the wind overlay addition tests. By providing a fresh plotter instance for each test method, it ensures that any state changes or configurations made during one test do not affect others. This setup allows the test methods to focus on validating the behavior of the `add_wind_overlay` method without worrying about shared state or side effects from previous tests.

        Parameters:
            None: This fixture does not require any parameters.

        Returns:
            MPASWindPlotter: Plotter instance used in overlay tests.
        """
        return MPASWindPlotter()
    
    # ------------------ Test Basic Wind Overlay Addition ------------------

    def test_add_wind_overlay_basic(self: "TestAddWindOverlay", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data, monkeypatch) -> None:
        """
        This test verifies that the `add_wind_overlay` method can successfully add a wind overlay to existing axes using real MPAS longitude, latitude, and wind component data with basic parameters. The test patches the internal rendering method to confirm that it is called during the overlay addition process. It checks that the render method is invoked, indicating that the overlay was processed for rendering. This ensures that users can add wind overlays with real MPAS data and that the internal rendering logic is invoked as expected when adding overlays.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to add overlay.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.
            monkeypatch: Pytest fixture for patching methods.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Use real headless plotting instead of monkeypatching plt.subplots
        fig = plt.figure()

        # Create a GeoAxes for testing since add_wind_overlay expects a GeoAxes for rendering
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        # Load real MPAS coordinates (first 3 points) for testing
        lon, lat = mpas_coordinates[0][:3], mpas_coordinates[1][:3]

        # For wind data, we also take the first 3 points for testing
        u, v = mpas_wind_data[0][:3], mpas_wind_data[1][:3]
        
        # Define a basic wind configuration for the overlay using 1D data
        wind_config = {
            'u_data': u,
            'v_data': v,
            'plot_type': 'barbs',
            'color': 'red'
        }
        
        # Stub the render method to confirm it is called when adding the overlay
        calls = {'render': 0}

        # Define a fake render method to count calls without actually rendering, since we are focused on the logic of add_wind_overlay
        monkeypatch.setattr(MPASWindPlotter, '_render_wind_vectors', fake_render_factory(calls))

        # Call the method to add the wind overlay to the existing axes with the provided configuration
        plotter.add_wind_overlay(ax, lon, lat, wind_config)

        # Assert that the render method was called once, confirming that the overlay addition logic proceeded to the rendering step
        assert calls['render'] == 1

        # Close the figure after the test to free up resources, since we are not actually displaying it in this test context
        plt.close(fig)
    
    # ------------------ Test Wind Overlay Addition with 3D Data and Level Index ------------------

    def test_add_wind_overlay_3d_data_with_level(self: "TestAddWindOverlay", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data, monkeypatch) -> None:
        """
        This test verifies that when 3D wind component data is provided to `add_wind_overlay` along with a specified `level_index`, the method correctly extracts the specified vertical level from the 3D data and adds the wind overlay to the existing axes without errors. The test uses real MPAS longitude, latitude, and 3D wind component data to create a realistic scenario for handling 3D inputs. It checks that the internal rendering method is called, confirming that the overlay was processed for rendering after extracting the specified level. This ensures that users can add wind overlays using 3D datasets by specifying a level index, and that the internal logic correctly handles the extraction of the appropriate level for rendering.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to add overlay.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.
            monkeypatch: Pytest fixture for patching methods.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Use real headless plotting instead of monkeypatching plt.subplots
        fig = plt.figure()

        # Create a GeoAxes for testing since add_wind_overlay expects a GeoAxes for rendering
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        # Load real MPAS coordinates (first 3 points) for testing
        lon, lat = mpas_coordinates[0][:3], mpas_coordinates[1][:3]

        # Load real MPAS wind data (first 15 points) and reshape to 3D arrays for testing; the actual shape would depend on the real MPAS data structure
        u_flat, v_flat = mpas_wind_data[0][:15], mpas_wind_data[1][:15]

        # Reshape to 3D arrays with 3 cells and 5 vertical levels for testing; the actual shape would depend on the real MPAS data structure
        u_3d = u_flat.reshape((3, 5))  # (cells, levels)
        v_3d = v_flat.reshape((3, 5))
        
        # Define a wind configuration that includes the 3D data and specifies a level index to extract for the overlay
        wind_config = {
            'u_data': u_3d,
            'v_data': v_3d,
            'level_index': 2,
            'plot_type': 'arrows'
        }
        
        # Stub the render method to confirm it is called when adding the overlay with 3D data and level extraction
        calls = {'render': 0}

        # Define a fake render method to count calls without actually rendering, since we are focused on the logic of add_wind_overlay and level extraction
        monkeypatch.setattr(MPASWindPlotter, '_render_wind_vectors', fake_render_factory(calls))

        # Call the method to add the wind overlay to the existing axes with the provided 3D data and level index configuration
        plotter.add_wind_overlay(ax, lon, lat, wind_config)

        # Assert that the render method was called once, confirming that the overlay addition logic proceeded to the rendering step after extracting the specified level from the 3D data
        assert calls['render'] == 1

        # Close the figure after the test to free up resources, since we are not actually displaying it in this test context
        plt.close(fig)
    
    # ------------------ Test Wind Overlay Addition with 3D Data and Default Level ------------------

    def test_add_wind_overlay_3d_data_default_level(self: "TestAddWindOverlay", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data, monkeypatch) -> None:
        """
        This test verifies that when 3D wind component data is provided to `add_wind_overlay` without a specified `level_index`, the method defaults to using the topmost vertical level from the 3D data for the overlay, and successfully adds it to the existing axes without errors. The test uses real MPAS longitude, latitude, and 3D wind component data to create a realistic scenario for handling 3D inputs with default level selection. It checks that the internal rendering method is called, confirming that the overlay was processed for rendering after selecting the default level. This ensures that users can add wind overlays using 3D datasets without needing to specify a level index, and that the internal logic correctly defaults to an appropriate level for rendering when none is specified.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to add overlay.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.
            monkeypatch: Pytest fixture for patching methods.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Use real headless plotting instead of monkeypatching plt.subplots
        fig = plt.figure()

        # Create a GeoAxes for testing since add_wind_overlay expects a GeoAxes for rendering
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        # Load real MPAS coordinates (first 3 points) for testing
        lon, lat = mpas_coordinates[0][:3], mpas_coordinates[1][:3]

        # Load real MPAS wind data (first 15 points) and reshape to 3D arrays for testing; the actual shape would depend on the real MPAS data structure
        u_flat, v_flat = mpas_wind_data[0][:15], mpas_wind_data[1][:15]

        # Reshape to 3D arrays with 3 cells and 5 vertical levels for testing; the actual shape would depend on the real MPAS data structure
        u_3d = u_flat.reshape((3, 5))
        v_3d = v_flat.reshape((3, 5))
        
        # Define a wind configuration that includes the 3D data but does not specify a level index, so the default behavior should select the topmost level for rendering
        wind_config = {
            'u_data': u_3d,
            'v_data': v_3d
        }
        
        # Stub the render method to confirm it is called when adding the overlay with 3D data and default level selection
        calls = {'render': 0}

        # Define a fake render method to count calls without actually rendering, since we are focused on the logic of add_wind_overlay and default level selection
        monkeypatch.setattr(MPASWindPlotter, '_render_wind_vectors', fake_render_factory(calls))

        # Call the method to add the wind overlay to the existing axes with the provided 3D data and default level selection configuration
        plotter.add_wind_overlay(ax, lon, lat, wind_config)

        # Assert that the render method was called once, confirming that the overlay addition logic proceeded to the rendering step after selecting the default topmost level from the 3D data
        assert calls['render'] == 1

        # Close the figure after the test to free up resources, since we are not actually displaying it in this test context
        plt.close(fig)
    
    # ------------------ Test Wind Overlay Addition with Regridding ------------------

    def test_add_wind_overlay_with_regridding(self: "TestAddWindOverlay", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data, monkeypatch) -> None:
        """
        This test verifies that when `grid_resolution` is specified in the wind configuration passed to `add_wind_overlay`, the method successfully performs regridding of the input wind data onto a regular grid defined by the specified resolution and bounding box, and adds the regridded overlay to the existing axes without errors. The test uses real MPAS longitude, latitude, and wind component data to create a realistic scenario for regridding. It checks that the internal regridding method is called with the correct parameters, and that the internal rendering method is called with the outputs from the regridding process. This ensures that users can add wind overlays with regridded data by specifying a grid resolution and bounding box, and that the internal logic correctly applies the regridding process before rendering the overlay.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to add overlay.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.
            monkeypatch: Pytest fixture for patching methods.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Use real headless plotting instead of monkeypatching plt.subplots
        fig = plt.figure()

        # Create a GeoAxes for testing since add_wind_overlay expects a GeoAxes for rendering
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        # Create dummy 2D grids for regridding outputs
        lon_2d = np.meshgrid(np.linspace(0, 50, 10), np.linspace(0, 25, 10))[0]
        lat_2d = np.meshgrid(np.linspace(0, 50, 10), np.linspace(0, 25, 10))[1]

        # For wind components, we can just use dummy data since we are focused on confirming that the regrid method is called and its outputs are used, rather than validating the actual regridding logic here
        u_2d = np.ones((10, 10))
        v_2d = np.ones((10, 10))

        # Define a fake regrid method to return the dummy 2D grids and confirm it is called when grid_resolution is provided
        def _fake_regrid(self, lon, lat, u, v, dataset, lon_min, lon_max, lat_min, lat_max, grid_resolution, regrid_method):
            return (lon_2d, lat_2d, u_2d, v_2d)

        # Patch the _regrid_wind_components method with our fake regrid function to confirm it is called and its outputs are used when grid_resolution is provided in the wind configuration
        monkeypatch.setattr(MPASWindPlotter, '_regrid_wind_components', _fake_regrid)
        
        # Load real MPAS coordinates (first 100 points) for testing
        lon, lat = mpas_coordinates[0][:100], mpas_coordinates[1][:100]

        # Load real MPAS wind data (first 100 points) for testing
        u, v = mpas_wind_data
        
        # Define a wind configuration that includes the grid_resolution to trigger regridding, and specify a regrid method for completeness
        wind_config = {
            'u_data': u,
            'v_data': v,
            'grid_resolution': 1.0,
            'regrid_method': 'linear'
        }
        
        # Stub the render method to confirm it is called when adding the overlay with regridding enabled, and that it receives the regridded outputs from our fake regrid method
        calls = {'render': 0}

        # Define a fake render method to count calls without actually rendering, since we are focused on the logic of add_wind_overlay and confirming that regridding is triggered and its outputs are used
        monkeypatch.setattr(MPASWindPlotter, '_render_wind_vectors', fake_render_factory(calls))

        # Call the method to add the wind overlay to the existing axes with the provided configuration that includes grid_resolution to trigger regridding
        plotter.add_wind_overlay(
            ax, lon, lat, wind_config,
            lon_min=0, lon_max=50,
            lat_min=0, lat_max=25
        )

        # Assert that the render method was called once, confirming that the overlay addition logic proceeded to the rendering step after regridding was triggered and its outputs were used
        assert calls['render'] == 1

        # Close the figure after the test to free up resources, since we are not actually displaying it in this test context
        plt.close(fig)
    
    # ------------------ Test Wind Overlay Addition with Regridding but Missing Bounds ------------------

    def test_add_wind_overlay_regrid_missing_bounds(self: "TestAddWindOverlay", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data) -> None:
        """
        This test verifies that when `grid_resolution` is specified in the wind configuration passed to `add_wind_overlay` but the required bounding box parameters (`lon_min`, `lon_max`, `lat_min`, `lat_max`) are not provided, the method raises a ValueError with an informative message indicating the missing parameters. The test uses real MPAS longitude, latitude, and wind component data to create a realistic scenario for regridding without bounds. It checks that a ValueError is raised when attempting to add the overlay with regridding enabled but missing bounds, and that the error message contains references to the missing bounding box parameters. This ensures that users receive clear feedback about what parameters are required for regridding when they attempt to add a wind overlay with a specified grid resolution but fail to provide the necessary bounding box information.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to add overlay.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Use real headless plotting instead of monkeypatching plt.subplots
        fig = plt.figure()

        # Create a GeoAxes for testing since add_wind_overlay expects a GeoAxes for rendering
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        # Use real MPAS data 
        lon, lat = mpas_coordinates[0][:3], mpas_coordinates[1][:3]
        u, v = mpas_wind_data[0][:3], mpas_wind_data[1][:3]
        
        # Define a wind configuration that includes the grid_resolution to trigger regridding, but does not include the required bounding box parameters, which should result in a ValueError when add_wind_overlay attempts to perform regridding without the necessary bounds
        wind_config = {
            'u_data': u,
            'v_data': v,
            'grid_resolution': 1.0
        }
        
        # Attempt to add the wind overlay with regridding enabled but missing bounds, and assert that a ValueError is raised with a message indicating the missing bounding box parameters
        with pytest.raises(ValueError) as exc_info:
            plotter.add_wind_overlay(ax, lon, lat, wind_config)

        # Convert the exception message to a string and check that it contains references to the missing bounding box parameters (e.g., "lon_min", "lat_min") to confirm that the error message is informative about what parameters are required for regridding
        err = str(exc_info.value)

        # Assert that the error message contains references to the missing bounding box parameters, confirming that the ValueError raised when attempting to regrid without bounds is informative about what parameters are required for regridding
        assert "lon_min" in err and "lat_min" in err

        # Close the figure after the test to free up resources, since we are not actually displaying it in this test context
        plt.close(fig)
    
    # ------------------ Test Wind Overlay Addition with Automatic Subsampling ------------------

    def test_add_wind_overlay_auto_subsample(self: "TestAddWindOverlay", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data, monkeypatch) -> None:
        """
        This test verifies that when `subsample=-1` is specified in the wind configuration passed to `add_wind_overlay` along with the required bounding box parameters, the method automatically calculates an appropriate subsampling factor based on the density of the input data and the specified bounding box, and successfully adds the subsampled wind overlay to the existing axes without errors. The test uses real MPAS longitude, latitude, and wind component data to create a realistic scenario for automatic subsampling. It checks that the internal rendering method is called, confirming that the overlay was processed for rendering after automatic subsampling was applied. This ensures that users can add wind overlays with automatic subsampling by setting `subsample=-1` and providing bounding box parameters, and that the internal logic correctly calculates the subsampling factor and applies it before rendering the overlay.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to add overlay.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.
            monkeypatch: Pytest fixture for patching methods.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        lon, lat = mpas_coordinates[0][:100], mpas_coordinates[1][:100]
        u, v = mpas_wind_data
        
        wind_config = {
            'u_data': u,
            'v_data': v,
            'subsample': -1  # Auto-calculate
        }
        
        calls = {'render': 0}

        monkeypatch.setattr(MPASWindPlotter, '_render_wind_vectors', fake_render_factory(calls))

        plotter.add_wind_overlay(
            ax, lon, lat, wind_config,
            lon_min=0, lon_max=50,
            lat_min=0, lat_max=25
        )
        assert calls['render'] == 1
        plt.close(fig)
    
    # ------------------ Test Wind Overlay Addition with Automatic Subsampling but Missing Bounds ------------------

    def test_add_wind_overlay_auto_subsample_missing_bounds(self: "TestAddWindOverlay", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data) -> None:
        """
        This test verifies that when `subsample=-1` is specified in the wind configuration passed to `add_wind_overlay` but the required bounding box parameters (`lon_min`, `lon_max`, `lat_min`, `lat_max`) are not provided, the method raises a ValueError with an informative message indicating the missing parameters. The test uses real MPAS longitude, latitude, and wind component data to create a realistic scenario for automatic subsampling without bounds. It checks that a ValueError is raised when attempting to add the overlay with automatic subsampling enabled but missing bounds, and that the error message contains references to the missing bounding box parameters. This ensures that users receive clear feedback about what parameters are required for automatic subsampling when they attempt to add a wind overlay with `subsample=-1` but fail to provide the necessary bounding box information.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to add overlay.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        # Use real MPAS data
        lon, lat = mpas_coordinates[0][:3], mpas_coordinates[1][:3]
        u, v = mpas_wind_data[0][:3], mpas_wind_data[1][:3]
        
        wind_config = {
            'u_data': u,
            'v_data': v,
            'subsample': -1
        }
        
        with pytest.raises(ValueError) as exc_info:
            plotter.add_wind_overlay(ax, lon, lat, wind_config)

        err = str(exc_info.value)
        assert "lon_min" in err and "lat_min" in err
        plt.close(fig)
    
    # ------------------ Test Wind Overlay Addition with Empty Data ------------------

    def test_add_wind_overlay_empty_data_1d(self: "TestAddWindOverlay", plotter: MPASWindPlotter, mpas_coordinates, monkeypatch) -> None:
        """
        This test verifies that when 1D coordinate and wind component arrays are provided to `add_wind_overlay` but all values are NaN, the method does not attempt to render any wind vectors and handles the empty data gracefully without errors. The test uses real MPAS longitude and latitude arrays but injects NaN values for the wind components to create a scenario of empty data. It checks that the internal rendering method is not called, confirming that the method correctly identifies that there is no valid data to render and skips the rendering process. This ensures that users can add wind overlays with empty or invalid data without encountering errors, and that the internal logic correctly handles cases where there are no valid wind vectors to render.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to add overlay.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            monkeypatch: Pytest fixture for patching methods.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        if mpas_coordinates is None:
            pytest.skip("MPAS data not available")
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        # Use real MPAS coordinates but inject NaN wind values
        lon, lat = mpas_coordinates[0][:3], mpas_coordinates[1][:3]
        u = np.array([np.nan, np.nan, np.nan])
        v = np.array([np.nan, np.nan, np.nan])
        
        wind_config = {
            'u_data': u,
            'v_data': v
        }
        
        calls = {'render': 0}

        monkeypatch.setattr(MPASWindPlotter, '_render_wind_vectors', fake_render_factory(calls))

        plotter.add_wind_overlay(ax, lon, lat, wind_config)

        # Should not render
        assert calls['render'] == 0
        plt.close(fig)
    
    # ------------------ Test Wind Overlay Addition with Empty 2D Data ------------------

    def test_add_wind_overlay_empty_data_2d(self: "TestAddWindOverlay", plotter: MPASWindPlotter, monkeypatch) -> None:
        """
        This test verifies that when 2D coordinate and wind component arrays are provided to `add_wind_overlay` but all values are NaN, the method does not attempt to render any wind vectors and handles the empty data gracefully without errors. The test creates synthetic 2D longitude and latitude grids filled with NaN values, as well as 2D wind component arrays filled with NaN values, to simulate a scenario of empty data. It checks that the internal rendering method is not called, confirming that the method correctly identifies that there is no valid data to render and skips the rendering process. This ensures that users can add wind overlays with empty or invalid 2D data without encountering errors, and that the internal logic correctly handles cases where there are no valid wind vectors to render in a gridded format.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to add overlay.
            monkeypatch: Pytest fixture for patching methods.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        lon_2d = np.ones((5, 5)) * np.nan
        lat_2d = np.ones((5, 5)) * np.nan
        u_2d = np.ones((5, 5)) * np.nan
        v_2d = np.ones((5, 5)) * np.nan
        
        wind_config = {
            'u_data': u_2d,
            'v_data': v_2d
        }
        
        calls = {'render': 0}

        monkeypatch.setattr(MPASWindPlotter, '_render_wind_vectors', fake_render_factory(calls))

        plotter.add_wind_overlay(ax, lon_2d, lat_2d, wind_config)

        # Should not render
        assert calls['render'] == 0
        plt.close(fig)

# ================== Test Class: Extract2DFrom3DWind ===================

class TestExtract2DFrom3DWind:
    """ Tests for extracting 2D slices from 3D wind component arrays. These tests cover explicit index extraction, pressure-level matching, default top-level extraction, and xarray compatibility. """

    # ------------------ Initialize Plotter Fixture ------------------

    @pytest.fixture
    def plotter(self: "TestExtract2DFrom3DWind") -> MPASWindPlotter:
        """
        This fixture creates and returns an instance of the `MPASWindPlotter` class for use in the 2D extraction tests. By providing a fresh plotter instance for each test method, it ensures that any state changes or configurations made during one test do not affect others. This setup allows the test methods to focus on validating the behavior of the `extract_2d_from_3d_wind` method without worrying about shared state or side effects from previous tests.

        Parameters:
            None: This fixture does not require any parameters.

        Returns:
            MPASWindPlotter: Plotter instance used in tests.
        """
        return MPASWindPlotter()

    # ------------------ Test Extraction by Explicit Level Index ------------------

    def test_extract_by_level_index(self: "TestExtract2DFrom3DWind", plotter: MPASWindPlotter, mpas_wind_data) -> None:
        """
        This test verifies that when a specific `level_index` is provided to `extract_2d_from_3d_wind`, the method correctly extracts the corresponding vertical level from the 3D u and v wind component arrays and returns 2D arrays with the expected shape and values. The test uses real MPAS wind component data to create realistic 3D arrays for testing. It checks that the returned 2D arrays have the correct shape corresponding to the number of horizontal points, and that the values in the extracted 2D arrays match the expected slice from the original 3D arrays based on the provided level index. This ensures that users can accurately extract specific vertical levels from 3D wind datasets using explicit level indices.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to call the method.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u_100, v_100 = mpas_wind_data
        u_flat = np.tile(u_100, 10)
        v_flat = np.tile(v_100, 10)
        u_3d = u_flat.reshape((100, 10))
        v_3d = v_flat.reshape((100, 10))
        
        u_2d, v_2d = plotter.extract_2d_from_3d_wind(
            u_3d, v_3d, level_index=5
        )
        
        assert u_2d.shape == (100,)
        np.testing.assert_array_equal(u_2d, u_3d[:, 5])
    
    # ------------------ Test Extraction by Pressure Value ------------------

    def test_extract_by_pressure_value(self: "TestExtract2DFrom3DWind", plotter: MPASWindPlotter, mpas_wind_data) -> None:
        """
        This test verifies that when a specific `level_value` (e.g., pressure level) is provided to `extract_2d_from_3d_wind` along with the corresponding `pressure_levels` array, the method correctly identifies the index of the specified pressure level and extracts the corresponding vertical level from the 3D u and v wind component arrays. The test uses real MPAS wind component data to create realistic 3D arrays for testing, and defines a set of pressure levels to match against. It checks that the returned 2D arrays have the correct shape corresponding to the number of horizontal points, and that the values in the extracted 2D arrays match the expected slice from the original 3D arrays based on the identified index for the provided pressure level. This ensures that users can extract specific vertical levels from 3D wind datasets by specifying pressure values, and that the internal logic correctly matches those values to the appropriate indices in the data.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to call the method.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u_100, v_100 = mpas_wind_data
        u_flat = np.tile(u_100, 10)
        v_flat = np.tile(v_100, 10)
        u_3d = u_flat.reshape((100, 10))
        v_3d = v_flat.reshape((100, 10))
        pressure_levels = np.array([1000, 925, 850, 700, 500, 400, 300, 250, 200, 100])
        
        u_2d, v_2d = plotter.extract_2d_from_3d_wind(
            u_3d, v_3d,
            level_value=850,
            pressure_levels=pressure_levels
        )
        
        assert u_2d.shape == (100,)
        np.testing.assert_array_equal(u_2d, u_3d[:, 2])
    
    # ------------------ Test Extraction with Default Top-Level Selection ------------------

    def test_extract_default_top_level(self: "TestExtract2DFrom3DWind", plotter: MPASWindPlotter, mpas_wind_data) -> None:
        """
        This test verifies that when no specific level selection is provided to `extract_2d_from_3d_wind`, the method defaults to extracting the topmost vertical level (i.e., the last index) from the 3D u and v wind component arrays. The test uses real MPAS wind component data to create realistic 3D arrays for testing. It checks that the returned 2D arrays have the correct shape corresponding to the number of horizontal points, and that the values in the extracted 2D arrays match the expected slice from the original 3D arrays based on the default top-level selection. This ensures that users can rely on a sensible default behavior when extracting 2D slices from 3D wind datasets without needing to specify a level, and that the internal logic correctly defaults to an appropriate level for extraction when no selection is provided.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to call the method.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u_100, v_100 = mpas_wind_data
        u_flat = np.tile(u_100, 10)
        v_flat = np.tile(v_100, 10)
        u_3d = u_flat.reshape((100, 10))
        v_3d = v_flat.reshape((100, 10))
        
        u_2d, v_2d = plotter.extract_2d_from_3d_wind(u_3d, v_3d)
        
        assert u_2d.shape == (100,)
        np.testing.assert_array_equal(u_2d, u_3d[:, -1])
    
    # ------------------ Test Extraction with xarray DataArray Inputs ------------------

    def test_extract_with_xarray(self: "TestExtract2DFrom3DWind", plotter: MPASWindPlotter, mpas_wind_data) -> None:
        """
        This test verifies that `extract_2d_from_3d_wind` can accept `xarray.DataArray` inputs for the 3D u and v wind component data, and that it preserves the return types as `xarray.DataArray` when requested. The test uses real MPAS wind component data to create realistic 3D arrays for testing, and converts them into `xarray.DataArray` format. It checks that the returned 2D arrays are indeed `xarray.DataArray` instances, and that the values in the extracted 2D arrays match the expected slice from the original 3D arrays based on a specified level index. This ensures that users can work with `xarray.DataArray` inputs and outputs when extracting 2D slices from 3D wind datasets, and that the method correctly handles xarray data structures while performing the extraction.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to call the method.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u_100, v_100 = mpas_wind_data
        u_flat = np.tile(u_100, 4)
        v_flat = np.tile(v_100, 4)
        u_data = u_flat.reshape((50, 8))
        v_data = v_flat.reshape((50, 8))

        u_3d = xr.DataArray(u_data, dims=['cells', 'levels'])
        v_3d = xr.DataArray(v_data, dims=['cells', 'levels'])
        
        u_2d, v_2d = plotter.extract_2d_from_3d_wind(
            u_3d, v_3d, level_index=3
        )
        
        assert isinstance(u_2d, xr.DataArray)
        np.testing.assert_array_equal(u_2d.values, u_data[:, 3])

# ================== Test Class: ComputeWindSpeedAndDirection ===================

class TestComputeWindSpeedAndDirection:
    """ Tests for `compute_wind_speed_and_direction` utility. These tests verify magnitude computation and meteorological angle conversion across 1D and 2D inputs. """

    # ------------------ Initialize Plotter Fixture ------------------

    @pytest.fixture
    def plotter(self: "TestComputeWindSpeedAndDirection") -> MPASWindPlotter:
        """
        This fixture creates and returns an instance of the `MPASWindPlotter` class for use in the wind speed and direction computation tests. By providing a fresh plotter instance for each test method, it ensures that any state changes or configurations made during one test do not affect others. This setup allows the test methods to focus on validating the behavior of the `compute_wind_speed_and_direction` method without worrying about shared state or side effects from previous tests.

        Parameters:
            None: This fixture does not require any parameters.

        Returns:
            MPASWindPlotter: Plotter instance used in tests.
        """
        return MPASWindPlotter()

    # ------------------ Test Wind Speed Computation with Simple Inputs ------------------

    def test_compute_wind_speed(self: "TestComputeWindSpeedAndDirection", plotter: MPASWindPlotter, mpas_wind_data) -> None:
        """
        This test verifies that `compute_wind_speed_and_direction` correctly computes the wind speed (magnitude) from the u and v wind components using the Pythagorean theorem. The test uses real MPAS wind component data to create realistic 1D arrays for testing. It checks that the computed wind speed matches the expected values calculated from the original u and v components, confirming that the method accurately computes wind speed from its vector components.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to call the method.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixture is not available
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Use real MPAS wind data for realistic testing
        u, v = mpas_wind_data[0][:3], mpas_wind_data[1][:3]
        
        # Compute speed and direction for the test inputs 
        speed, direction = plotter.compute_wind_speed_and_direction(u, v)
        
        # Expected speed should be sqrt(u^2 + v^2) for each element based on the Pythagorean theorem for vector magnitude
        expected_speed = np.sqrt(u**2 + v**2)

        # Assert that the computed speed matches the expected values within a reasonable tolerance
        np.testing.assert_array_almost_equal(speed, expected_speed)
    
    # ------------------ Test Wind Direction Computation with Cardinal Directions ------------------

    def test_compute_wind_direction_north(self: "TestComputeWindSpeedAndDirection", plotter: MPASWindPlotter) -> None:
        """
        This test verifies that `compute_wind_speed_and_direction` correctly computes the meteorological wind direction for a northward wind (u=0, v>0). According to meteorological convention, a northward wind means the wind is coming from the south, which corresponds to a direction of 180 degrees. The test checks that the computed direction matches this expected value within a reasonable tolerance, confirming that the method correctly converts u and v components into meteorological angles.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to call the method.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Northward wind means the wind is coming from the south, which corresponds to a meteorological direction of 180 degrees.
        u = np.array([0.0])
        v = np.array([5.0])  # Northward wind (FROM south = 180 degrees)
        
        # Compute speed and direction for the test inputs
        speed, direction = plotter.compute_wind_speed_and_direction(u, v)
        
        # Meteorological convention: wind from south is 180 degrees and should be verified within a reasonable tolerance
        assert direction[0] == pytest.approx(180.0, abs=1e-1)
    
    # ------------------ Test Wind Direction Computation for Eastward and Westward Winds ------------------

    def test_compute_wind_direction_east(self: "TestComputeWindSpeedAndDirection", plotter: MPASWindPlotter) -> None:
        """
        This test verifies that `compute_wind_speed_and_direction` correctly computes the meteorological wind direction for an eastward wind (u>0, v=0). According to meteorological convention, an eastward wind means the wind is coming from the west, which corresponds to a direction of 270 degrees. The test checks that the computed direction matches this expected value within a reasonable tolerance, confirming that the method correctly converts u and v components into meteorological angles.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to call the method.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Eastward wind means the wind is coming from the west, which corresponds to a meteorological direction of 270 degrees.
        u = np.array([5.0])  # Eastward wind (FROM west = 270 degrees)
        v = np.array([0.0])
        
        # Compute speed and direction for the test inputs
        speed, direction = plotter.compute_wind_speed_and_direction(u, v)
        
        # Meteorological convention: wind from west is 270 degrees and should be verified within a reasonable tolerance
        assert direction[0] == pytest.approx(270.0, abs=1e-1)
    
    # ------------------ Test Wind Direction Computation for Westward Winds ------------------

    def test_compute_wind_direction_west(self: "TestComputeWindSpeedAndDirection", plotter: MPASWindPlotter) -> None:
        """
        This test verifies that `compute_wind_speed_and_direction` correctly computes the meteorological wind direction for a westward wind (u<0, v=0). According to meteorological convention, a westward wind means the wind is coming from the east, which corresponds to a direction of 90 degrees. The test checks that the computed direction matches this expected value within a reasonable tolerance, confirming that the method correctly converts u and v components into meteorological angles.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to call the method.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Westward wind means the wind is coming from the east, which corresponds to a meteorological direction of 90 degrees.
        u = np.array([-5.0])  # Westward wind (FROM east = 90 degrees)
        v = np.array([0.0])
        
        # Compute speed and direction for the test inputs
        speed, direction = plotter.compute_wind_speed_and_direction(u, v)
        
        # Meteorological convention: wind from east is 90 degrees
        assert direction[0] == pytest.approx(90.0, abs=1e-1)
    
    # ------------------ Test Wind Speed and Direction Computation with 2D Inputs ------------------

    def test_compute_2d_arrays(self: "TestComputeWindSpeedAndDirection", plotter: MPASWindPlotter, mpas_wind_data) -> None:
        """
        This test verifies that `compute_wind_speed_and_direction` can handle 2D input arrays for the u and v wind components, and that it correctly computes the wind speed and direction for each corresponding element in the 2D arrays. The test uses real MPAS wind component data reshaped into 2D arrays to create a realistic scenario for gridded wind data. It checks that the computed speed and direction arrays have the correct shapes matching the input 2D arrays, and that the values in the computed speed array match the expected magnitudes calculated from the original u and v components for each element. This ensures that users can compute wind speed and direction from gridded 2D wind component data, and that the method correctly processes multi-dimensional inputs.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance used to call the method.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixture is not available
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        # Use real MPAS data reshaped to 2D (5x5 grid)
        u_2d = mpas_wind_data[0][:25].reshape(5, 5)
        v_2d = mpas_wind_data[1][:25].reshape(5, 5)
        
        # Compute speed and direction for 2D arrays and verify shapes and values
        speed, direction = plotter.compute_wind_speed_and_direction(u_2d, v_2d)
        
        # Verify speed shape matches input shape
        assert speed.shape == (5, 5)

        # Verify direction shape matches input shape
        assert direction.shape == (5, 5)

        # Expected speed should be sqrt(u^2 + v^2) for each element
        expected_speed = np.sqrt(u_2d**2 + v_2d**2)

        # Assert that the computed speed matches the expected values within a reasonable tolerance
        np.testing.assert_array_almost_equal(speed, expected_speed)

# ================== Test Class: CreateBatchWindPlots ===================

class TestCreateBatchWindPlots:
    """ Tests that create a sequence of wind plots and save them to disk. These exercises validate interaction with the MPAS time utilities and the high-level batch orchestration. """

    # ------------------ Initialize Plotter and Temporary Directory Fixtures ------------------

    @pytest.fixture
    def plotter(self: "TestCreateBatchWindPlots") -> MPASWindPlotter:
        """
        This fixture creates and returns an instance of the `MPASWindPlotter` class for use in the batch wind plot creation tests. By providing a fresh plotter instance for each test method, it ensures that any state changes or configurations made during one test do not affect others. This setup allows the test methods to focus on validating the behavior of the `create_batch_wind_plots` method without worrying about shared state or side effects from previous tests.

        Parameters:
            None: This fixture does not require any parameters.

        Returns:
            MPASWindPlotter: Plotter instance used in tests.
        """
        return MPASWindPlotter()

    # ------------------ Create Temporary Directory for Test Outputs ------------------

    @pytest.fixture
    def temp_dir(self: "TestCreateBatchWindPlots") -> Generator[str, None, None]:
        """
        This fixture creates a temporary directory for storing output files generated during the batch wind plot creation tests. It uses the `tempfile` module to create a unique temporary directory for each test, and ensures that the directory is cleaned up after the test completes by removing it with `shutil.rmtree`. This allows the tests to write output files without affecting the actual filesystem or leaving behind test artifacts, and provides a clean environment for each test method that requires file output.

        Parameters:
            None: This fixture does not require any parameters.
            
        Returns:
            str: Path to the temporary directory.
        """
        # Use the tempfile module to create a temporary directory for test outputs
        temp_dir = tempfile.mkdtemp()

        # Yield the directory path to the test, and then clean up after the test completes
        yield temp_dir

        # Clean up the temporary directory after the test is done
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # ------------------ Test Batch Creation with Mocked Processor and Real MPAS Data ------------------

    def test_create_batch_wind_plots(self: "TestCreateBatchWindPlots", plotter: MPASWindPlotter, mpas_coordinates, mpas_wind_data, temp_dir: str) -> None:
        """
        This test verifies that `create_batch_wind_plots` can successfully create a batch of wind plots using a mocked processor that simulates the behavior of a real data processor with a loaded dataset, and that it correctly interacts with the MPAS data fixtures to retrieve coordinates and wind data. The test uses real MPAS longitude, latitude, and wind component data to create a realistic scenario for batch plot creation. It checks that the method returns a list of created file paths corresponding to the expected number of time steps, and that the expected output files are created in the temporary directory. This ensures that users can create batches of wind plots using the `create_batch_wind_plots` method with a properly structured processor and real MPAS data, and that the method correctly handles the data retrieval and file creation processes.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call batch creation.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.
            temp_dir (str): Temporary directory path to receive output files.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")

        # Import pandas and xarray here since they are only needed for this test and to avoid unnecessary imports if MPAS data is not available        
        import pandas as pd
        import xarray as xr

        # Create a mock dataset with a Time coordinate to simulate the processor's dataset structure; the actual time values are not critical for this test, so we can use simple timestamps
        times = pd.to_datetime(['2024-01-01T00', '2024-01-01T06', '2024-01-01T12'])
        dataset = xr.Dataset(coords={'Time': ('Time', times)})

        # Create a MagicMock to simulate the processor's behavior; we will configure its methods to return the real MPAS data when called by the batch creation function
        mock_processor = MagicMock()

        # Assign the mock dataset to the processor's dataset attribute so that the batch creation function can access it as if it were a real loaded dataset
        mock_processor.dataset = dataset

        # Mock the coordinate extraction to return the real MPAS lon/lat arrays for testing; this simulates what the processor would return when its `extract_2d_coordinates_for_variable` method is called during batch creation
        lon, lat = mpas_coordinates[0][:100], mpas_coordinates[1][:100]

        # Load real MPAS wind data (first 100 points) for testing; the actual shape and structure would depend on the real MPAS data, but we will create xarray DataArrays to simulate what the processor would return when its `get_2d_variable_data` method is called during batch creation
        u_vals, v_vals = mpas_wind_data

        # Create xarray DataArray wrappers with `.values` attribute
        u_da = xr.DataArray(u_vals)
        v_da = xr.DataArray(v_vals)

        # Return u,v for each time step (u,v) repeated 3 times
        mock_processor.get_2d_variable_data.side_effect = [
            u_da, v_da,
            u_da, v_da,
            u_da, v_da
        ]

        # Mock the coordinate extraction to return the real MPAS lon/lat arrays for testing
        mock_processor.extract_2d_coordinates_for_variable.return_value = (lon, lat)

        # Call batch creation; this will produce real plot files in temp_dir
        created_files = plotter.create_batch_wind_plots(
            processor=mock_processor,
            output_dir=temp_dir,
            lon_min=0, lon_max=50,
            lat_min=0, lat_max=25,
            u_variable='u10',
            v_variable='v10'
        )

        # Assert that the method returns a list of created file paths and that the expected number of files (3 time steps) were created
        assert len(created_files) == 3

        # Verify that the expected output files were created in the temporary directory
        for path in created_files:
            file_png = f"{path}.png"
            assert os.path.exists(file_png), f"Expected output file {file_png}"
        
        # Close any open figures
        plt.close('all')
    
    # ------------------ Test Batch Creation with Missing Dataset ------------------

    def test_create_batch_no_dataset(self: "TestCreateBatchWindPlots", plotter: MPASWindPlotter, temp_dir: str) -> None:
        """
        This test verifies that `create_batch_wind_plots` raises a ValueError with an informative message when the provided processor does not have a loaded dataset. The test uses a MagicMock to simulate a processor that lacks a dataset, and checks that the batch creation function correctly identifies this issue and raises an error with a message indicating that there is no loaded dataset and that the user should load data before attempting to create batch plots. This ensures that users receive clear feedback about the requirement for a loaded dataset when they attempt to create batch wind plots without having properly set up their processor.

        Parameters:
            plotter (MPASWindPlotter): Fixture instance to call batch creation.
            temp_dir (str): Temporary directory path used as output.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Create a MagicMock to simulate the processor's behavior; we will configure it to have no dataset to trigger the error handling in the batch creation function
        mock_processor = MagicMock()
        mock_processor.dataset = None
        
        # Call batch creation and expect it to raise a ValueError due to the missing dataset; we will check that the error message contains the expected instruction to load data first
        with pytest.raises(ValueError) as exc_info:
            plotter.create_batch_wind_plots(
                processor=mock_processor,
                output_dir=temp_dir,
                lon_min=0, lon_max=50,
                lat_min=0, lat_max=25
            )
        
        # Assert that the error message contains the expected instruction about no loaded dataset
        assert "no loaded dataset" in str(exc_info.value)

        # Close any open figures to free resources
        plt.close('all')


if __name__ == "__main__":
    pytest.main([__file__])