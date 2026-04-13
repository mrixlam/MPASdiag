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
        assert plotter.dpi == pytest.approx(100)

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
        assert plotter.dpi == pytest.approx(150)

        # Verify figure is None before plotting
        assert plotter.fig is None

        # Verify axes are None before plotting
        assert plotter.ax is None


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
        assert subsample_barbs_small == pytest.approx(1)

        # Define a small dataset with few points to plot wind arrows
        subsample_arrows_small = plotter.calculate_optimal_subsample(
            num_points=1000,
            lon_min=-30, lon_max=30,
            lat_min=-15, lat_max=15,
            plot_type='arrows'
        )

        # Verify no subsampling for arrows
        assert subsample_arrows_small == pytest.approx(1)

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


