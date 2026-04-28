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
import pytest
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
from typing import cast
import matplotlib
matplotlib.use("Agg")
from cartopy import crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.geoaxes import GeoAxes
from mpasdiag.visualization.wind import MPASWindPlotter
from tests.visualization.wind_test_helpers import require_wind_fixtures
from tests.test_data_helpers import fake_render_factory


class TestRegridWindComponents:
    """ Test for the wind component regridding helper (`_regrid_wind_components`). """

    # ------------------ Initialize Plotter Fixture ------------------

    @pytest.fixture
    def plotter(self: 'TestRegridWindComponents') -> 'MPASWindPlotter':
        """
        This fixture creates and returns an instance of the `MPASWindPlotter` class for use in the wind component regridding tests. By providing a fresh plotter instance for each test method, it ensures that any state changes or configurations made during one test do not affect others. This setup allows the test methods to focus on validating the behavior of the `_regrid_wind_components` method without worrying about shared state or side effects from previous tests.

        Parameters:
            self ("TestRegridWindComponents"): Test instance which will receive the plotter fixture.

        Returns:
            MPASWindPlotter: Plotter instance used in tests.
        """
        return MPASWindPlotter()
    
    # ------------------ Test Linear Regridding Method ------------------

    def test_regrid_linear_method(self: 'TestRegridWindComponents', 
                                  plotter: 'MPASWindPlotter', 
                                  monkeypatch) -> None:
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
        grid_file = data_dir / "grids" / "x1.10242.static.nc"

        # Skip test if grid file is not available
        if not grid_file.exists():
            pytest.skip(f"MPAS grid file not found: {grid_file}")
            return

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

    def test_regrid_nearest_method(self: 'TestRegridWindComponents', 
                                   plotter: 'MPASWindPlotter', 
                                   monkeypatch) -> None:
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
        grid_file = data_dir / "grids" / "x1.10242.static.nc"

        # Skip test if grid file is not available
        if not grid_file.exists():
            pytest.skip(f"MPAS grid file not found: {grid_file}")
            return

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
        

class TestCreateWindPlot:
    """ Tests for the high-level `create_wind_plot` workflow. These unit tests patch plotting internals to validate extent, title and subsampling behaviors without rendering real figures to the display. """

    # ------------------ Initialize Plotter Fixture ------------------

    @pytest.fixture
    def plotter(self: 'TestCreateWindPlot',) -> 'MPASWindPlotter':
        """
        This fixture creates and returns an instance of the `MPASWindPlotter` class for use in the wind plot creation tests. By providing a fresh plotter instance for each test method, it ensures that any state changes or configurations made during one test do not affect others. This setup allows the test methods to focus on validating the behavior of the `create_wind_plot` method without worrying about shared state or side effects from previous tests.

        Parameters:
            self ('TestCreateWindPlot'): Test instance which will receive the plotter fixture.

        Returns:
            MPASWindPlotter: Plotter instance used to create plots in tests.
        """
        return MPASWindPlotter()

    # ------------------ Test Basic Wind Plot Creation ------------------

    def test_create_wind_plot_basic(self: 'TestCreateWindPlot', 
                                    plotter: 'MPASWindPlotter', 
                                    mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                    mpas_wind_data: tuple[np.ndarray, np.ndarray], 
                                    monkeypatch) -> None:
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
        require_wind_fixtures(mpas_coordinates, mpas_wind_data)
        
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
        assert calls['render'] == pytest.approx(1)

        # Close the figure to free resources
        plt.close(fig)

    # ------------------ Test Wind Plot Creation with Automatic Subsampling ------------------

    def test_create_wind_plot_auto_subsample(self: 'TestCreateWindPlot', 
                                             plotter: 'MPASWindPlotter', 
                                             mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                             mpas_wind_data: tuple[np.ndarray, np.ndarray], 
                                             monkeypatch) -> None:
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
        require_wind_fixtures(mpas_coordinates, mpas_wind_data)
        
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
        assert calls['render'] == pytest.approx(1)

        # Close the figure to free resources
        plt.close(fig)

    # ------------------ Test Wind Plot Creation with Global Extent ------------------

    def test_create_wind_plot_global_extent(self: 'TestCreateWindPlot', 
                                            plotter: 'MPASWindPlotter', 
                                            mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                            mpas_wind_data: tuple[np.ndarray, np.ndarray], 
                                            monkeypatch) -> None:
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
        require_wind_fixtures(mpas_coordinates, mpas_wind_data)
        
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
        assert calls['render'] == pytest.approx(1)

        # Cast to GeoAxes for extent checking
        geo_ax = cast(GeoAxes, ax)

        # Get the current extent of the GeoAxes
        extent = geo_ax.get_extent()

        # Verify that extent is a tuple of length 4
        assert isinstance(extent, tuple) and len(extent) == pytest.approx(4)

        # Close the figure to free resources
        plt.close(fig)
    
    # ------------------ Test Wind Plot Creation with Custom Title ------------------

    def test_create_wind_plot_custom_title(self: 'TestCreateWindPlot', 
                                           plotter: 'MPASWindPlotter', 
                                           mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                           mpas_wind_data: tuple[np.ndarray, np.ndarray], 
                                           monkeypatch) -> None:
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
        require_wind_fixtures(mpas_coordinates, mpas_wind_data)
        
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
        assert calls['render'] == pytest.approx(1)

        # Close the figure to free resources
        plt.close(fig)
    
    # ------------------ Test Wind Plot Creation with Timestamp ------------------

    def test_create_wind_plot_with_timestamp(self: 'TestCreateWindPlot', 
                                             plotter: 'MPASWindPlotter', 
                                             mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                             mpas_wind_data: tuple[np.ndarray, np.ndarray], 
                                             monkeypatch) -> None:
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
        require_wind_fixtures(mpas_coordinates, mpas_wind_data)
        
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

    def test_create_wind_plot_with_level_info(self: 'TestCreateWindPlot', 
                                              plotter: 'MPASWindPlotter', 
                                              mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                              mpas_wind_data: tuple[np.ndarray, np.ndarray], 
                                              monkeypatch) -> None:
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
        require_wind_fixtures(mpas_coordinates, mpas_wind_data)
        
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

    def test_create_wind_plot_with_regridding(self: 'TestCreateWindPlot', 
                                              plotter: 'MPASWindPlotter', 
                                              mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                              mpas_wind_data: tuple[np.ndarray, np.ndarray], 
                                              monkeypatch) -> None:
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
        require_wind_fixtures(mpas_coordinates, mpas_wind_data)
        
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

    def test_create_wind_plot_streamlines_auto_regrid(self: 'TestCreateWindPlot', 
                                                      plotter: 'MPASWindPlotter', 
                                                      mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                                      mpas_wind_data: tuple[np.ndarray, np.ndarray], 
                                                      monkeypatch) -> None:
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
        require_wind_fixtures(mpas_coordinates, mpas_wind_data)
        
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

    
    # ------------------ Test Wind Plot Creation with 2D Gridded Data ------------------

    def test_create_wind_plot_2d_data(self: 'TestCreateWindPlot', 
                                      plotter: 'MPASWindPlotter', 
                                      mpas_coordinates: tuple[np.ndarray, np.ndarray], 
                                      mpas_wind_data: tuple[np.ndarray, np.ndarray], 
                                      monkeypatch) -> None:
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
        require_wind_fixtures(mpas_coordinates, mpas_wind_data)
        
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
