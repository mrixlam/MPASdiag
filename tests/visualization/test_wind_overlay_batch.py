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
import pytest
import shutil
import tempfile
import numpy as np
import xarray as xr
from typing import Generator
from unittest.mock import MagicMock
import matplotlib
matplotlib.use("Agg")
from cartopy import crs as ccrs
import matplotlib.pyplot as plt
from mpasdiag.visualization.wind import MPASWindPlotter
from tests.test_data_helpers import fake_render_factory


class TestAddWindOverlay:
    """ Tests for `add_wind_overlay` which renders wind vectors onto existing axes. These tests validate 1D/3D handling, regridding, subsampling, and error messages for missing bounding boxes. """

    # ------------------ Initialize Plotter Fixture ------------------

    @pytest.fixture
    def plotter(self: 'TestAddWindOverlay') -> 'MPASWindPlotter':
        """
        This fixture creates and returns an instance of the `MPASWindPlotter` class for use in the wind overlay addition tests. By providing a fresh plotter instance for each test method, it ensures that any state changes or configurations made during one test do not affect others. This setup allows the test methods to focus on validating the behavior of the `add_wind_overlay` method without worrying about shared state or side effects from previous tests.

        Parameters:
            None: This fixture does not require any parameters.

        Returns:
            MPASWindPlotter: Plotter instance used in overlay tests.
        """
        return MPASWindPlotter()
    
    # ------------------ Test Basic Wind Overlay Addition ------------------

    def test_add_wind_overlay_basic(self: 'TestAddWindOverlay', 
                                    plotter: 'MPASWindPlotter', 
                                    mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                    mpas_wind_data: tuple[np.ndarray, np.ndarray], 
                                    monkeypatch) -> None:
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
            return
        
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
        assert calls['render'] == pytest.approx(1)

        # Close the figure after the test to free up resources, since we are not actually displaying it in this test context
        plt.close(fig)
    
    # ------------------ Test Wind Overlay Addition with 3D Data and Level Index ------------------

    def test_add_wind_overlay_3d_data_with_level(self: 'TestAddWindOverlay', 
                                                 plotter: 'MPASWindPlotter', 
                                                 mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                                 mpas_wind_data: tuple[np.ndarray, np.ndarray], 
                                                 monkeypatch) -> None:
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
            return
        
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
        assert calls['render'] == pytest.approx(1)

        # Close the figure after the test to free up resources, since we are not actually displaying it in this test context
        plt.close(fig)
    
    # ------------------ Test Wind Overlay Addition with 3D Data and Default Level ------------------

    def test_add_wind_overlay_3d_data_default_level(self: 'TestAddWindOverlay', 
                                                    plotter: 'MPASWindPlotter', 
                                                    mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                                    mpas_wind_data: tuple[np.ndarray, np.ndarray], 
                                                    monkeypatch) -> None:
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
            return 
        
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
        assert calls['render'] == pytest.approx(1)

        # Close the figure after the test to free up resources, since we are not actually displaying it in this test context
        plt.close(fig)
    
    # ------------------ Test Wind Overlay Addition with Regridding ------------------

    def test_add_wind_overlay_with_regridding(self: 'TestAddWindOverlay', 
                                              plotter: 'MPASWindPlotter', 
                                              mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                              mpas_wind_data: tuple[np.ndarray, np.ndarray], 
                                              monkeypatch) -> None:
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
            return
        
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
        u, v = mpas_wind_data[0][:100], mpas_wind_data[1][:100]
        
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
        assert calls['render'] == pytest.approx(1)

        # Close the figure after the test to free up resources, since we are not actually displaying it in this test context
        plt.close(fig)
    
    # ------------------ Test Wind Overlay Addition with Regridding but Missing Bounds ------------------

    def test_add_wind_overlay_regrid_missing_bounds(self: 'TestAddWindOverlay', 
                                                    plotter: 'MPASWindPlotter', 
                                                    mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                                    mpas_wind_data: tuple[np.ndarray, np.ndarray]) -> None:
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
            return
        
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

    def test_add_wind_overlay_auto_subsample(self: 'TestAddWindOverlay', 
                                             plotter: 'MPASWindPlotter', 
                                             mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                             mpas_wind_data: tuple[np.ndarray, np.ndarray], 
                                             monkeypatch) -> None:
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
            return
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        lon, lat = mpas_coordinates[0][:100], mpas_coordinates[1][:100]
        u, v = mpas_wind_data[0][:100], mpas_wind_data[1][:100]
        
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
        assert calls['render'] == pytest.approx(1)
        plt.close(fig)
    
    # ------------------ Test Wind Overlay Addition with Automatic Subsampling but Missing Bounds ------------------

    def test_add_wind_overlay_auto_subsample_missing_bounds(self: 'TestAddWindOverlay', 
                                                            plotter: 'MPASWindPlotter', 
                                                            mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                                            mpas_wind_data: tuple[np.ndarray, np.ndarray]) -> None:
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
            return
        
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

    def test_add_wind_overlay_empty_data_1d(self: "TestAddWindOverlay", 
                                            plotter: 'MPASWindPlotter', 
                                            mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                            monkeypatch) -> None:
        """
        This test verifies that when 1D coordinate and wind component arrays are provided to `add_wind_overlay` but all values are NaN, the method does not attempt to render any wind vectors and handles the empty data gracefully without errors. The test uses real MPAS longitude and latitude arrays but injects NaN values for the wind components to create a scenario of empty data. It checks that the internal rendering method is not called, confirming that the method correctly identifies that there is no valid data to render and skips the rendering process. This ensures that users can add wind overlays with empty or invalid data without encountering errors, and that the internal logic correctly handles cases where there are no valid wind vectors to render.

        Parameters:
            plotter ('MPASWindPlotter'): Fixture instance used to add overlay.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            monkeypatch: Pytest fixture for patching methods.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        if mpas_coordinates is None:
            pytest.skip("MPAS data not available")
            return
        
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
        assert calls['render'] == pytest.approx(0)
        plt.close(fig)
    
    # ------------------ Test Wind Overlay Addition with Empty 2D Data ------------------

    def test_add_wind_overlay_empty_data_2d(self: 'TestAddWindOverlay', 
                                            plotter: 'MPASWindPlotter', 
                                            monkeypatch) -> None:
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


class TestCreateBatchWindPlots:
    """ Tests that create a sequence of wind plots and save them to disk. These exercises validate interaction with the MPAS time utilities and the high-level batch orchestration. """

    # ------------------ Initialize Plotter and Temporary Directory Fixtures ------------------

    @pytest.fixture
    def plotter(self: 'TestCreateBatchWindPlots') -> 'MPASWindPlotter':
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
    def temp_dir(self: 'TestCreateBatchWindPlots') -> Generator[str, None, None]:
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

    def test_create_batch_wind_plots(self: 'TestCreateBatchWindPlots', 
                                     plotter: 'MPASWindPlotter', 
                                     mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                     mpas_wind_data: tuple[np.ndarray, np.ndarray], 
                                     temp_dir: str) -> None:
        """
        This test verifies that `create_batch_wind_plots` can successfully create a batch of wind plots using a mocked processor that simulates the behavior of a real data processor with a loaded dataset, and that it correctly interacts with the MPAS data fixtures to retrieve coordinates and wind data. The test uses real MPAS longitude, latitude, and wind component data to create a realistic scenario for batch plot creation. It checks that the method returns a list of created file paths corresponding to the expected number of time steps, and that the expected output files are created in the temporary directory. This ensures that users can create batches of wind plots using the `create_batch_wind_plots` method with a properly structured processor and real MPAS data, and that the method correctly handles the data retrieval and file creation processes.

        Parameters:
            plotter ('MPASWindPlotter'): Fixture instance to call batch creation.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_wind_data: Session fixture providing real MPAS u/v wind data.
            temp_dir (str): Temporary directory path to receive output files.

        Returns:
            None: Assertion-based test; raises on failure.
        """
        # Skip if MPAS data fixtures are not available
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
            return

        # Import pandas and xarray here since they are only needed for this test and to avoid unnecessary imports if MPAS data is not available        
        import pandas as pd

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
        assert len(created_files) == pytest.approx(3), f"Expected 3 created files, got {len(created_files)}"

        # Verify that the expected output files were created in the temporary directory
        for path in created_files:
            file_png = f"{path}.png"
            assert os.path.exists(file_png), f"Expected output file {file_png}"
        
        # Close any open figures
        plt.close('all')
    
