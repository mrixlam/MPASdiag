#!/usr/bin/env python3
"""
MPASdiag Test Suite: Wind Visualization Functionality

This module contains unit tests for the wind visualization functionality in the MPASdiag package, specifically focusing on the `MPASWindPlotter` class. The tests validate the behavior of the wind component regridding helper method (`_regrid_wind_components`) and the high-level plot creation workflow (`create_wind_plot`). By using real MPAS grid and wind data, these tests ensure that the visualization functions can handle realistic scenarios and produce expected outputs. The tests also patch internal rendering methods to confirm that they are called during the plotting process, without relying on actual figure rendering in a headless testing environment.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import os
import sys
import pytest
import shutil
import tempfile
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
from typing import Generator, cast
from unittest.mock import MagicMock, Mock, patch
import matplotlib
matplotlib.use("Agg")
from cartopy import crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.geoaxes import GeoAxes
from mpasdiag.visualization.wind import MPASWindPlotter


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

    def test_prepare_1d_no_subsample(self: "TestPrepareWindData", 
                                     plotter: MPASWindPlotter, 
                                     mpas_coordinates: tuple[np.ndarray, np.ndarray],
                                     mpas_wind_data: tuple[np.ndarray, np.ndarray]) -> None:
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
            return
        
        if (
            mpas_coordinates is None or mpas_wind_data is None or
            mpas_coordinates[0] is None or mpas_coordinates[1] is None or
            mpas_wind_data[0] is None or mpas_wind_data[1] is None
        ):
            pytest.skip("MPAS data not available")
            return

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

    def test_prepare_1d_with_subsample(self: "TestPrepareWindData", 
                                       plotter: MPASWindPlotter, 
                                       mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                       mpas_wind_data: tuple[np.ndarray, np.ndarray]) -> None:
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
            return
        
        if (
            mpas_coordinates is None or mpas_wind_data is None or
            mpas_coordinates[0] is None or mpas_coordinates[1] is None or
            mpas_wind_data[0] is None or mpas_wind_data[1] is None
        ):
            pytest.skip("MPAS data not available")
            return

        # Load real MPAS data (first 100 points)
        lon, lat = mpas_coordinates[0][:100], mpas_coordinates[1][:100]
        u, v = mpas_wind_data[0][:100], mpas_wind_data[1][:100]
        
        # Call the helper with subsampling factor of 2
        lon_out, lat_out, u_out, v_out = plotter._prepare_wind_data(
            lon, lat, u, v, subsample=2
        )
        
        # Outputs should be half the length and strided
        assert len(lon_out) == pytest.approx(50)
        assert len(lat_out) == pytest.approx(50)
        assert len(u_out) == pytest.approx(50)
        assert len(v_out) == pytest.approx(50)

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

    def test_prepare_1d_with_nan_values(self: "TestPrepareWindData", 
                                         plotter: MPASWindPlotter, 
                                         mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                         mpas_wind_data: tuple[np.ndarray, np.ndarray]) -> None:
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
            return
        
        if (
            mpas_coordinates is None or mpas_wind_data is None or
            mpas_coordinates[0] is None or mpas_coordinates[1] is None or
            mpas_wind_data[0] is None or mpas_wind_data[1] is None
        ):
            pytest.skip("MPAS data not available")
            return

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
        assert len(lon_out) == pytest.approx(2)
        assert len(lat_out) == pytest.approx(2)
        assert len(u_out) == pytest.approx(2)
        assert len(v_out) == pytest.approx(2)

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

    def test_prepare_2d_no_subsample(self: "TestPrepareWindData", 
                                      plotter: MPASWindPlotter, 
                                      mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                      mpas_wind_data: tuple[np.ndarray, np.ndarray]) -> None:
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
            return
        
        if (
            mpas_coordinates is None or mpas_wind_data is None or
            mpas_coordinates[0] is None or mpas_coordinates[1] is None or
            mpas_wind_data[0] is None or mpas_wind_data[1] is None
        ):
            pytest.skip("MPAS data not available")
            return

        # Load real MPAS coordinates reshaped to 2D (2x5 grid)
        lon = mpas_coordinates[0][:10].reshape(2, 5)
        lat = mpas_coordinates[1][:10].reshape(2, 5)

        if lon.shape != (2, 5) or lat.shape != (2, 5):
            pytest.skip("MPAS data not available or not in expected shape")
            return

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

    def test_prepare_2d_with_subsample(self: "TestPrepareWindData", 
                                       plotter: MPASWindPlotter, 
                                       mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                       mpas_wind_data: tuple[np.ndarray, np.ndarray]) -> None:
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
            return
        
        if (
            mpas_coordinates is None or mpas_wind_data is None or
            mpas_coordinates[0] is None or mpas_coordinates[1] is None or
            mpas_wind_data[0] is None or mpas_wind_data[1] is None
        ):
            pytest.skip("MPAS data not available")
            return

        # Use real MPAS coordinates reshaped to 2D (10x10 grid)
        lon = mpas_coordinates[0][:100].reshape(10, 10)
        lat = mpas_coordinates[1][:100].reshape(10, 10)

        if lon.shape != (10, 10) or lat.shape != (10, 10):
            pytest.skip("MPAS data not available or not in expected shape")
            return

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

    def test_prepare_xarray_input(self: "TestPrepareWindData", 
                                  plotter: MPASWindPlotter, 
                                  mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                  mpas_wind_data: tuple[np.ndarray, np.ndarray]) -> None:
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
            return
        
        if (
            mpas_coordinates is None or mpas_wind_data is None or
            mpas_coordinates[0] is None or mpas_coordinates[1] is None or
            mpas_wind_data[0] is None or mpas_wind_data[1] is None
        ):
            pytest.skip("MPAS data not available")
            return

        # Use real MPAS coordinates and wind data (first 3 points)
        lon_data, lat_data = mpas_coordinates[0][:3], mpas_coordinates[1][:3]
        u_data, v_data = mpas_wind_data[0][:3], mpas_wind_data[1][:3]
        
        if (
            lon_data.shape != (3,) or lat_data.shape != (3,) or
            u_data.shape != (3,) or v_data.shape != (3,)
        ):
            pytest.skip("MPAS data not available or not in expected shape")
            return

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

    def test_render_barbs(self: "TestRenderWindVectors", 
                          plotter: MPASWindPlotter, 
                          mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                          mpas_wind_data: tuple[np.ndarray, np.ndarray]) -> None:
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
            return
        
        # Create a figure and GeoAxes
        fig = plt.figure()

        # Add GeoAxes with PlateCarree projection
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        if ax is None:
            pytest.skip("Failed to create GeoAxes for testing")
            return
        
        if (
            mpas_coordinates is None or mpas_wind_data is None or
            mpas_coordinates[0] is None or mpas_coordinates[1] is None or
            mpas_wind_data[0] is None or mpas_wind_data[1] is None
        ):
            pytest.skip("MPAS data not available")
            return

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

    def test_render_arrows(self: "TestRenderWindVectors", 
                           plotter: MPASWindPlotter, 
                           mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                           mpas_wind_data: tuple[np.ndarray, np.ndarray]) -> None:
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
            return
        
        # Create a figure and GeoAxes
        fig = plt.figure()

        # Add GeoAxes with PlateCarree projection
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

        if (
            mpas_coordinates is None or mpas_wind_data is None or
            mpas_coordinates[0] is None or mpas_coordinates[1] is None or
            mpas_wind_data[0] is None or mpas_wind_data[1] is None
        ):
            pytest.skip("MPAS data not available")
            return

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

    def test_render_arrows_default_scale(self: "TestRenderWindVectors", 
                                         plotter: MPASWindPlotter, 
                                         mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                         mpas_wind_data: tuple[np.ndarray, np.ndarray]) -> None:
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
            return
        
        # Create a figure and GeoAxes
        fig = plt.figure()

        # Add GeoAxes with PlateCarree projection
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

        if (
            mpas_coordinates is None or mpas_wind_data is None or
            mpas_coordinates[0] is None or mpas_coordinates[1] is None or
            mpas_wind_data[0] is None or mpas_wind_data[1] is None
        ):
            pytest.skip("MPAS data not available")
            return

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

    def test_render_streamlines(self: "TestRenderWindVectors", 
                                plotter: MPASWindPlotter, 
                                mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                mpas_wind_data: tuple[np.ndarray, np.ndarray]) -> None:
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
            return
        
        # Create a figure and GeoAxes
        fig = plt.figure()

        # Add GeoAxes with PlateCarree projection
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        if (
            mpas_coordinates is None or mpas_wind_data is None or
            mpas_coordinates[0] is None or mpas_coordinates[1] is None or
            mpas_wind_data[0] is None or mpas_wind_data[1] is None
        ):
            pytest.skip("MPAS data not available")
            return
        
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

    def test_render_streamlines_1d_raises_error(self: "TestRenderWindVectors", 
                                                plotter: MPASWindPlotter, 
                                                mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                                mpas_wind_data: tuple[np.ndarray, np.ndarray]) -> None:
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
            return
        
        # Create a figure and GeoAxes
        fig = plt.figure()
 
        # Add GeoAxes with PlateCarree projection
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        if (
            mpas_coordinates is None or mpas_wind_data is None or
            mpas_coordinates[0] is None or mpas_coordinates[1] is None or
            mpas_wind_data[0] is None or mpas_wind_data[1] is None
        ):
            pytest.skip("MPAS data not available")
            return

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

    def test_render_invalid_plot_type(self: "TestRenderWindVectors", 
                                      plotter: MPASWindPlotter, 
                                      mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                      mpas_wind_data: tuple[np.ndarray, np.ndarray]) -> None:
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
            return
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

        if (
            mpas_coordinates is None or mpas_wind_data is None or
            mpas_coordinates[0] is None or mpas_coordinates[1] is None or
            mpas_wind_data[0] is None or mpas_wind_data[1] is None
        ):
            pytest.skip("MPAS data not available")
            return

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


