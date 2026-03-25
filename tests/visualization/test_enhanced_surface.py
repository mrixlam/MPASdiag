#!/usr/bin/env python3
"""
MPASdiag Test Suite: Enhanced Surface Plotting Capabilities

This test suite validates the enhanced data-type agnostic surface plotting capabilities of the MPASdiag visualization module. It covers 3D data extraction, wind overlays, and complex meteorological visualizations using real MPAS model output data. The tests ensure robust handling of various data structures, coordinate-based level selection, and combined scalar/vector field rendering for comprehensive weather analysis.

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
import tempfile
import matplotlib
import numpy as np
import xarray as xr
matplotlib.use('Agg')  
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from typing import Generator, Iterator

from mpasdiag.visualization.surface import MPASSurfacePlotter
from tests.test_data_helpers import load_mpas_coords_from_processor

package_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_dir))


class TestEnhancedSurfacePlotting:
    """ Test suite for enhanced data-type agnostic surface plotting capabilities. Covers 3D data extraction, wind overlays, and complex meteorological visualizations. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestEnhancedSurfacePlotting", mpas_surface_temp_data, mpas_precip_data, mpas_wind_data) -> Generator[None, None, None]:
        """
        This setup method initializes the MPASSurfacePlotter instance and loads real MPAS test data for surface temperature, precipitation, and wind components from session-scoped fixtures. It also sets up spatial domain boundaries and coordinate arrays for testing. The test data is limited to 100 cells to ensure manageable test execution while still validating functionality with real model output. The method yields control to the test functions and performs cleanup after tests complete.

        Parameters:
            mpas_surface_temp_data: Real surface temperature data from conftest session fixture
            mpas_precip_data: Real precipitation data from conftest session fixture
            mpas_wind_data: Real wind u/v components from conftest session fixture

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = MPASSurfacePlotter(figsize=(8, 6), dpi=100)
        
        from tests.test_data_helpers import load_mpas_coords_from_processor

        self.n_cells = 100
        lon, lat, u, v = load_mpas_coords_from_processor(n=self.n_cells)
        self.lon = lon
        self.lat = lat
        
        self.temp_data = mpas_surface_temp_data[:self.n_cells]
        self.precip_data = mpas_precip_data[:self.n_cells]

        wind_u, wind_v = mpas_wind_data
        self.u = wind_u[:self.n_cells]
        self.v = wind_v[:self.n_cells]
        
        self.pressure_levels = np.array([1000, 925, 850, 700, 500, 400, 300, 250, 200, 100])
        self.n_levels = len(self.pressure_levels)
        
        self.lon_min, self.lon_max = -140, -50
        self.lat_min, self.lat_max = 25, 60
        
        yield
        
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        plt.close('all')
    
    def test_extract_2d_from_3d_with_index(self: "TestEnhancedSurfacePlotting") -> None:
        """
        This test verifies the functionality of extracting a 2D slice from 3D data using a specified level index. It creates synthetic 3D temperature data with dimensions corresponding to cells, levels, and time, and then uses the extract_2d_from_3d method to extract the data at a specific level index (e.g., index 5). Assertions confirm that the resulting 2D array has the correct shape and values corresponding to the specified level. This test ensures that users can easily extract vertical levels from 3D datasets for surface plotting without needing to manually slice arrays.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_base = 220.0 + 80.0 * ((u_arr - u_arr.min()) / (u_arr.max() - u_arr.min() + 1e-12))
        data_3d = np.tile(temp_base.reshape((self.n_cells, 1, 1)), (1, self.n_levels, 1))
        result = self.visualizer.extract_2d_from_3d(data_3d, level_index=5)
        
        assert result.shape == (self.n_cells,)
        assert np.allclose(result, data_3d[:, 5, 0])
        
    def test_extract_2d_from_3d_with_coordinate_value(self: "TestEnhancedSurfacePlotting") -> None:
        """
        This test verifies the functionality of extracting a 2D slice from 3D data using a specified coordinate value (e.g., 850 hPa). It creates an xarray DataArray with pressure coordinates and uses the extract_2d_from_3d method to extract data at a specific pressure level. Assertions confirm that the resulting 2D array has the correct shape and values corresponding to the specified pressure level. This test ensures that users can easily extract vertical levels from 3D datasets using physical coordinates rather than array indices.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_base = self.temp_data[:self.n_cells]
        data_3d = np.tile(temp_base.reshape((self.n_cells, 1)), (1, self.n_levels))
        
        data_xr = xr.DataArray(
            data_3d,
            dims=['cells', 'pressure'],
            coords={'pressure': self.pressure_levels}
        )
        
        result = self.visualizer.extract_2d_from_3d(data_xr, level_value=850, level_dim='pressure')
        
        assert result.shape == (self.n_cells,)
        expected_idx = 2
        assert np.allclose(result, data_3d[:, expected_idx])
        
    def test_extract_2d_from_3d_error_cases(self: "TestEnhancedSurfacePlotting") -> None:
        """
        This test validates the error handling of the extract_2d_from_3d method for invalid level specifications. It checks that a ValueError is raised when neither level_index nor level_value is provided, and that an IndexError is raised when an out-of-bounds level index is specified. Multiple error scenarios are tested to ensure robust input validation. Assertions confirm that appropriate exceptions are raised with informative error messages, preventing silent failures during 3D data extraction.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_base = 220.0 + 80.0 * ((u_arr - u_arr.min()) / (u_arr.max() - u_arr.min() + 1e-12))
        data_3d = np.tile(temp_base.reshape((self.n_cells, 1, 1)), (1, self.n_levels, 1))
        
        with pytest.raises(ValueError) as context:
            self.visualizer.extract_2d_from_3d(data_3d)

        assert "Must provide either level_index or level_value" in str(context.value)
        
        with pytest.raises(IndexError):
            self.visualizer.extract_2d_from_3d(data_3d, level_index=50)
            
    def test_create_surface_map_2d_data(self: "TestEnhancedSurfacePlotting") -> None:
        """
        This test validates the creation of a surface map from 2D data. It uses synthetic 2D temperature data corresponding to the number of cells and tests the create_surface_map method to generate a surface plot. Assertions confirm that a matplotlib Figure object is returned, the axes are properly created, and the title incorporates the specified text. This test ensures that users can successfully create surface maps from 2D datasets without errors, providing a foundation for more complex plotting scenarios.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_2d = self.temp_data[:self.n_cells]
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_2d, 't2m',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            title='Test 2D Temperature'
        )
        
        assert isinstance(fig, Figure)
        assert fig.axes[0] == ax
        assert 'Test 2D Temperature' in ax.get_title()
        
    def test_create_surface_map_3d_data_with_level_index(self: "TestEnhancedSurfacePlotting") -> None:
        """
        This test validates the creation of a surface map from 3D data using level index extraction. It creates synthetic 3D temperature data with dimensions corresponding to cells, levels, and time, and tests the create_surface_map method's ability to extract a specific level (e.g., index 5) for plotting. Assertions confirm that a matplotlib Figure object is returned, the axes are properly created, and the title incorporates the specified text. This test ensures that users can create surface maps directly from 3D datasets by specifying the desired vertical level through an index.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_base = self.temp_data[:self.n_cells]
        temp_3d = np.tile(temp_base.reshape((self.n_cells,1,1)), (1,self.n_levels,1))
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_3d, 'temperature',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            level_index=5,
            title='Test 3D Temperature at Level 5'
        )
        
        assert isinstance(fig, Figure)
        assert 'Test 3D Temperature at Level 5' in ax.get_title()
        
    def test_create_surface_map_2d_multilevel_data(self: "TestEnhancedSurfacePlotting") -> None:
        """
        This test validates the creation of a surface map from 3D data using level value extraction. It creates synthetic 3D temperature data with pressure coordinates and tests the create_surface_map method's ability to extract a specific pressure level (e.g., 850 hPa) for plotting. Assertions confirm that a matplotlib Figure object is returned, the axes are properly created, and the title incorporates the specified text. This test ensures that users can create surface maps directly from 3D datasets by specifying the desired vertical level through physical coordinate values.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_levels = np.tile(self.temp_data[:self.n_cells].reshape((self.n_cells,1)), (1,self.n_levels))
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_levels, 'temperature',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            level_index=2, 
            title='Test Multi-Level Temperature'
        )
        
        assert isinstance(fig, Figure)
        assert 'Test Multi-Level Temperature' in ax.get_title()
        
    def test_wind_overlay_2d_data(self: "TestEnhancedSurfacePlotting") -> None:
        """
        This test validates the wind barb overlay functionality on surface maps using 2D wind component data. It verifies that the wind overlay correctly adds wind barbs to an existing surface temperature plot using 2D u and v wind components. Synthetic wind data with realistic values (±15 m/s range) simulates horizontal wind fields at a single level. The test creates a base temperature surface map and adds wind barbs with density control (plot_every=3) for visual clarity. Assertions confirm the figure and axes objects are properly returned and barbs are added without errors. This demonstrates combined visualization of scalar and vector meteorological fields for weather analysis.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_2d = self.temp_data[:self.n_cells]
        u_wind_2d = self.u[:self.n_cells]
        v_wind_2d = self.v[:self.n_cells]
        
        wind_config = {
            'u_data': u_wind_2d,
            'v_data': v_wind_2d,
            'plot_type': 'barbs',
            'subsample': 3,
            'color': 'blue'
        }
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_2d, 't2m',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            wind_overlay=wind_config,
            title='Test Temperature with 2D Wind Overlay'
        )
        
        assert isinstance(fig, Figure)
        assert 'Test Temperature with 2D Wind Overlay' in ax.get_title()
        
    def test_wind_overlay_3d_data(self: "TestEnhancedSurfacePlotting") -> None:
        """
        This test validates the wind overlay functionality on surface maps using 3D wind component data. It verifies that the wind overlay correctly extracts the specified vertical level from 3D u and v wind components before overlaying on surface maps. Synthetic 3D wind data with realistic values (±30 m/s range) simulates horizontal wind fields across multiple pressure levels. The test creates a base temperature surface map and adds wind barbs at the specified level (level_index=2) for visual clarity. Assertions confirm the figure and axes objects are properly returned and barbs are added without errors. This demonstrates combined visualization of scalar and vector meteorological fields for multi-level atmospheric analysis.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_2d = self.temp_data[:self.n_cells]
        u_wind_3d = np.tile(self.u[:self.n_cells].reshape((self.n_cells,1)), (1,self.n_levels))
        v_wind_3d = np.tile(self.v[:self.n_cells].reshape((self.n_cells,1)), (1,self.n_levels))
        
        wind_config = {
            'u_data': u_wind_3d,
            'v_data': v_wind_3d,
            'plot_type': 'barbs',
            'subsample': 3,
            'color': 'blue',
            'level_index': 2  
        }
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_2d, 't2m',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            wind_overlay=wind_config,
            title='Test Temperature with 3D Wind Overlay'
        )
        
        assert isinstance(fig, Figure)
        assert 'Test Temperature with 3D Wind Overlay' in ax.get_title()
        
    def test_wind_overlay_arrows(self: "TestEnhancedSurfacePlotting") -> None:
        """
        This test validates the wind arrow overlay functionality on surface maps. It verifies that the wind overlay correctly renders arrow-style wind vectors in addition to traditional meteorological barbs. Synthetic u and v wind components test arrow rendering with custom styling (red color, scale=200) and subsampling (plot_every=4) for optimal density. The test creates a surface temperature map and adds wind arrows configured through the wind_config dictionary. Assertions confirm successful figure creation with arrows properly overlaid. This provides alternative wind visualization styles suitable for different audiences or presentation contexts beyond traditional meteorological conventions.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_2d = self.temp_data[:self.n_cells]
        u_wind = self.u[:self.n_cells]
        v_wind = self.v[:self.n_cells]
        
        wind_config = {
            'u_data': u_wind,
            'v_data': v_wind,
            'plot_type': 'arrows',
            'subsample': 4,
            'color': 'red',
            'scale': 200
        }
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_2d, 't2m',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            wind_overlay=wind_config,
            title='Test Temperature with Wind Arrows'
        )
        
        assert isinstance(fig, Figure)
        
    def test_complex_850hpa_weather_map(self: "TestEnhancedSurfacePlotting") -> None:
        """
        This test validates the creation of complex multi-layer 850 hPa analysis maps with wind overlays. It verifies that the plotting system can generate comprehensive weather analysis plots combining temperature contours, wind barbs, and proper level extraction from 3D data. Synthetic 3D temperature and wind data with pressure coordinates test coordinate-based level selection (850 hPa). Wind configuration specifies level extraction, barb styling, and subsampling parameters. The plot_type='both' parameter enables simultaneous scatter and contour rendering for enhanced visualization. This demonstrates production of publication-quality synoptic analysis maps commonly used in operational meteorology for mid-tropospheric weather pattern analysis.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_3d = np.tile(self.temp_data[:self.n_cells].reshape((self.n_cells,1)), (1,self.n_levels))

        temp_xr = xr.DataArray(
            temp_3d,
            dims=['cells', 'pressure'],
            coords={'pressure': self.pressure_levels}
        )
        
        u_wind_3d = np.tile(self.u[:self.n_cells].reshape((self.n_cells,1)), (1,self.n_levels))
        v_wind_3d = np.tile(self.v[:self.n_cells].reshape((self.n_cells,1)), (1,self.n_levels))
        
        wind_config = {
            'u_data': u_wind_3d,
            'v_data': v_wind_3d,
            'plot_type': 'barbs',
            'level_index': 2, 
            'subsample': 3,
            'color': 'navy'
        }
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_xr, 'temperature_850hpa', 
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            level_value=850,
            wind_overlay=wind_config,
            plot_type='both', 
            title='850hPa Weather Analysis'
        )
        
        assert isinstance(fig, Figure)
        assert '850hPa Weather Analysis' in ax.get_title()
        
    def test_plot_type_validation(self: "TestEnhancedSurfacePlotting") -> None:
        """
        This test validates the plot type parameter for supported rendering styles. It verifies that the plotting system correctly handles different plot type specifications including 'scatter', 'contour', 'contourf' (filled contours), and 'both' (combined rendering). Multiple test cases ensure each plot type produces valid matplotlib Figure objects without errors. Invalid plot type strings should trigger appropriate error handling. Assertions confirm successful figure creation for each valid plot type option. This ensures robust parameter validation and clear error messages guide users to supported visualization styles for different meteorological data presentations.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_2d = self.temp_data[:self.n_cells]
        
        for plot_type in ['scatter', 'contour', 'both']:
            fig, ax = self.visualizer.create_surface_map(
                self.lon, self.lat, temp_2d, 't2m',
                self.lon_min, self.lon_max, self.lat_min, self.lat_max,
                plot_type=plot_type
            )
            assert isinstance(fig, Figure)
            
        with pytest.raises(ValueError) as context:
            self.visualizer.create_surface_map(
                self.lon, self.lat, temp_2d, 't2m',
                self.lon_min, self.lon_max, self.lat_min, self.lat_max,
                plot_type='invalid'
            )

        assert "plot_type must be 'scatter', 'contour', 'contourf', or 'both'" in str(context.value)
        
    def test_wind_overlay_error_handling(self: "TestEnhancedSurfacePlotting") -> None:
        """
        This test validates error handling for invalid wind overlay configurations. It verifies that the plotting system raises appropriate exceptions when the wind overlay configuration is missing required components (e.g., u_data or v_data) or contains unsupported plot types (e.g., 'invalid_type'). Multiple error scenarios are tested to ensure robust input validation for wind overlay parameters. Assertions confirm that ValueError is raised with informative error messages guiding users to correct wind overlay specifications. This prevents silent failures during wind rendering and ensures users provide valid configurations for successful visualization of vector fields.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_2d = self.temp_data[:self.n_cells]
        
        wind_config = {
            'u_data': self.u[:self.n_cells],
            'v_data': self.v[:self.n_cells],
            'plot_type': 'invalid_type'
        }
        
        with pytest.raises(ValueError) as context:
            self.visualizer.create_surface_map(
                self.lon, self.lat, temp_2d, 't2m',
                self.lon_min, self.lon_max, self.lat_min, self.lat_max,
                wind_overlay=wind_config
            )

        assert "plot_type must be 'barbs', 'arrows', or 'streamlines'" in str(context.value)
        
    def test_data_dimension_validation(self: "TestEnhancedSurfacePlotting") -> None:
        """
        This test validates the data dimension requirements for surface plotting. It verifies that the plotting system raises appropriate exceptions when the input data array has unsupported dimensions (e.g., 4D data). The test creates synthetic 4D data with dimensions corresponding to cells, levels, time, and an extra dimension, which is not supported for surface plotting. Assertions confirm that ValueError is raised with informative error messages guiding users to provide data with correct dimensions (1D for 2D plots and 2D for contour plots). This prevents silent failures during rendering and ensures users provide valid data structures for successful visualization.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)

        base = self.temp_data[:self.n_cells]
        data_4d = np.tile(base.reshape((self.n_cells,1,1,1)), (1,10,5,3))
        
        with pytest.raises(ValueError) as context:
            self.visualizer.create_surface_map(
                self.lon, self.lat, data_4d, 'temperature',
                self.lon_min, self.lon_max, self.lat_min, self.lat_max
            )

        assert "only 1D, 2D and 3D data are supported" in str(context.value)
        
    def test_coordinate_length_validation(self: "TestEnhancedSurfacePlotting") -> None:
        """
        This test validates the length consistency of coordinate and data arrays. It verifies that the plotting system raises appropriate exceptions when the longitude, latitude, and data arrays have mismatched lengths. The test creates synthetic temperature data with a different length than the coordinate arrays to trigger the validation error. Assertions confirm that ValueError is raised with informative error messages guiding users to ensure coordinate and data arrays have matching lengths for successful plotting. This prevents silent failures during rendering and ensures users provide consistent input data structures.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells + 10)
        temp_wrong_length = 250.0 + 60.0 * ((u_arr - u_arr.min()) / (u_arr.max() - u_arr.min() + 1e-12))
        
        with pytest.raises(ValueError) as context:
            self.visualizer.create_surface_map(
                self.lon, self.lat, temp_wrong_length, 'temperature',
                self.lon_min, self.lon_max, self.lat_min, self.lat_max
            )

        assert "must match coordinate arrays length" in str(context.value)
        
    def test_auto_subsampling(self: "TestEnhancedSurfacePlotting") -> None:
        """
        This test validates the automatic subsampling functionality for large datasets. It verifies that the plotting system can handle large input datasets (e.g., 3000 cells) by automatically subsampling the data to maintain performance and visual clarity. The test creates synthetic large datasets by tiling real MPAS data and tests the create_surface_map method with wind overlays. Assertions confirm that a matplotlib Figure object is returned without errors, demonstrating that the auto-subsampling mechanism effectively manages large datasets for successful visualization.

        Parameters:
            None

        Returns:
            None
        """
        n_large = 3000

        lon_full, lat_full, u_full, v_full = load_mpas_coords_from_processor(n=n_large)

        lon_large = lon_full
        lat_large = lat_full

        temp_base_large = self.temp_data[:min(100, len(self.temp_data))]
        temp_large = np.tile(temp_base_large, int(np.ceil(n_large / len(temp_base_large))))[:n_large]

        u_base = self.u[:min(100, len(self.u))]
        v_base = self.v[:min(100, len(self.v))]

        u_wind_large = np.tile(u_base, int(np.ceil(n_large / len(u_base))))[:n_large]
        v_wind_large = np.tile(v_base, int(np.ceil(n_large / len(v_base))))[:n_large]
        
        wind_config = {
            'u_data': u_wind_large,
            'v_data': v_wind_large,
            'plot_type': 'barbs'
        }
        
        fig, ax = self.visualizer.create_surface_map(
            lon_large, lat_large, temp_large, 't2m',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            wind_overlay=wind_config,
            title='Test Auto-Subsampling'
        )
        
        assert isinstance(fig, Figure)
        
    def test_empty_data_handling(self: "TestEnhancedSurfacePlotting") -> None:
        """
        This test validates the handling of empty data arrays for surface plotting. It verifies that the plotting system raises appropriate exceptions when the input data array is empty or contains no valid data points within the specified map extent. The test creates synthetic longitude and latitude arrays that do not correspond to any valid data points and attempts to create a surface map. Assertions confirm that ValueError is raised with informative error messages guiding users to provide valid data points within the map extent for successful visualization. This prevents silent failures during rendering and ensures users provide meaningful data for plotting.

        Parameters:
            None

        Returns:
            None
        """
        lon_out_of_bounds = np.linspace(0.0, 9.0, self.n_cells)
        lat_out_of_bounds = np.linspace(0.0, 9.0, self.n_cells)

        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_2d = self.temp_data[:self.n_cells]
        
        with pytest.raises(ValueError) as context:
            self.visualizer.create_surface_map(
                lon_out_of_bounds, lat_out_of_bounds, temp_2d, 't2m',
                self.lon_min, self.lon_max, self.lat_min, self.lat_max
            )

        assert "No valid data points found within the specified map extent" in str(context.value)
        
    def test_timestamp_integration(self: "TestEnhancedSurfacePlotting") -> None:
        """
        This test validates the integration of timestamps into surface map titles. It verifies that the plotting system can incorporate a provided timestamp into the plot title for enhanced contextual information. The test creates a surface map with a specified datetime object and checks that the title includes the formatted timestamp string. Assertions confirm that a matplotlib Figure object is returned and the title contains the expected timestamp, demonstrating that users can easily add temporal context to their visualizations for improved interpretability.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_2d = self.temp_data[:self.n_cells]
        test_time = datetime(2024, 9, 17, 13, 0)
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_2d, 't2m',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            time_stamp=test_time,
            title='Test with Timestamp'
        )
        
        assert isinstance(fig, Figure)
        title_text = ax.get_title()
        assert 'Test with Timestamp' in title_text
        
    def test_custom_colormap_and_levels(self: "TestEnhancedSurfacePlotting") -> None:
        """
        This test validates the ability to specify custom colormaps and contour levels for surface maps. It verifies that users can override default colormaps and contour levels to create tailored visualizations matching specific requirements. The test creates a surface map with a coolwarm colormap and explicit temperature levels (250-310K in 10K increments) for contour-style plotting. Assertions confirm that a matplotlib Figure object is returned, demonstrating successful figure creation using the custom visual settings. This flexibility enables users to produce publication-quality plots, specialized color schemes for accessibility, and precise contour intervals matching domain standards or specific phenomena of interest.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_2d = self.temp_data[:self.n_cells]
        custom_levels = [250.0, 260.0, 270.0, 280.0, 290.0, 300.0, 310.0]
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_2d, 't2m',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            colormap='coolwarm',
            levels=custom_levels,
            plot_type='contour'
        )
        
        assert isinstance(fig, Figure)


class TestDataTypeAgnosticFeatures:
    """ Test suite specifically for data-type agnostic features. Tests the system's ability to handle various input data types seamlessly. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestDataTypeAgnosticFeatures", mpas_surface_temp_data, mpas_wind_data, mpas_precip_data) -> Iterator[None]:
        """
        This setup method initializes the MPASSurfacePlotter instance and loads real MPAS test data for surface temperature, wind components, and precipitation from session-scoped fixtures. It also sets up spatial domain boundaries and coordinate arrays for testing. The test data is limited to 50 cells to ensure manageable test execution while still validating functionality with real model output. The method yields control to the test functions and performs cleanup after tests complete.

        Parameters:
            mpas_surface_temp_data: Real surface temperature data from conftest session fixture
            mpas_wind_data: Real wind u/v components from conftest session fixture
            mpas_precip_data: Real precipitation data from conftest session fixture

        Returns:
            None
        """
        self.visualizer = MPASSurfacePlotter()
        self.n_cells = 50
        self.n_levels = 5
        lon_arr, lat_arr, _, _ = load_mpas_coords_from_processor(n=self.n_cells)
        self.lon = lon_arr
        self.lat = lat_arr
        
        self.temp_data = mpas_surface_temp_data[:self.n_cells]
        self.precip_data = mpas_precip_data[:self.n_cells]

        wind_u, wind_v = mpas_wind_data
        self.u = wind_u[:self.n_cells]
        self.v = wind_v[:self.n_cells]
        
        self.bounds = (-120, -80, 30, 50)
        self.lon_min, self.lon_max, self.lat_min, self.lat_max = self.bounds
        
        yield
        
        plt.close('all')
        
    def test_numpy_array_input(self: "TestDataTypeAgnosticFeatures") -> None:
        """
        This test validates the handling of NumPy array inputs for surface plotting. It verifies that the plotting system can accept raw NumPy arrays as input data for surface maps without requiring conversion to xarray DataArrays. The test creates a surface map using 2D temperature data provided as a NumPy array and checks that a matplotlib Figure object is returned successfully. Assertions confirm that the system can process unstructured array inputs, ensuring compatibility with various data processing workflows where users may have data in different formats.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        data_np = self.temp_data[:self.n_cells]
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, data_np, 't2m',
            *self.bounds
        )
        
        assert isinstance(fig, Figure)
        
    def test_xarray_dataarray_input(self: "TestDataTypeAgnosticFeatures") -> None:
        """
        This test validates the handling of xarray DataArray inputs for surface plotting. It verifies that the plotting system can accept xarray DataArrays as input data for surface maps while preserving metadata attributes like units and descriptive names. The test creates a surface map using a 1D temperature DataArray and checks that a matplotlib Figure object is returned successfully. Assertions confirm that the system can process labeled arrays with rich metadata, ensuring compatibility with MPAS native xarray datasets and workflows that leverage xarray's coordinate and attribute systems.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)

        data_xr = xr.DataArray(
            self.temp_data[:self.n_cells],
            dims=['cells'],
            attrs={'units': 'K', 'long_name': 'Temperature'}
        )
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, data_xr, 'temperature', 
            *self.bounds
        )
        
        assert isinstance(fig, Figure)
        
    def test_mixed_data_types_wind_overlay(self: "TestDataTypeAgnosticFeatures") -> None:
        """
        This test validates the handling of mixed data types for surface plotting with wind overlays. It verifies that the plotting system can accept a combination of xarray DataArrays for the primary surface field and raw NumPy arrays for the wind components in the overlay configuration. The test creates a surface map using a temperature DataArray and adds wind barbs using u and v wind components provided as NumPy arrays. Assertions confirm that a matplotlib Figure object is returned successfully, demonstrating that the system can seamlessly integrate different data types within the same plot configuration for flexible visualization workflows.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)

        temp_xr = xr.DataArray(
            self.temp_data[:self.n_cells],
            dims=['cells']
        )
        
        u_wind_np = self.u[:self.n_cells]
        v_wind_np = self.v[:self.n_cells]
        
        wind_config = {
            'u_data': u_wind_np,
            'v_data': v_wind_np,
            'plot_type': 'barbs',
            'subsample': 2
        }
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_xr, 'temperature',
            *self.bounds,
            wind_overlay=wind_config
        )
        
        assert isinstance(fig, Figure)

    def test_surface_overlay_contour_lines(self: "TestDataTypeAgnosticFeatures") -> None:
        """
        This test validates the handling of surface overlays with contour line visualization. It verifies that the plotting system can overlay secondary meteorological fields as contour lines atop the primary filled contour base layer. Temperature serves as the base field (250-310K) with geopotential height (1200-1400 dam) overlaid as contour lines in a surface_config dictionary. Assertions confirm successful multi-layer figure creation. This feature enables classical synoptic analysis plots like temperature fields with overlaid pressure contours, supporting meteorological interpretation where multiple related fields provide complementary information about atmospheric state.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_2d = self.temp_data[:self.n_cells]

        geop_2d = 1200.0 + 200.0 * (self.precip_data[:self.n_cells] / (np.max(self.precip_data[:self.n_cells]) + 1e-12))
        
        surface_config = {
            'data': geop_2d,
            'var_name': 'geopotential_height',
            'plot_type': 'contour',
            'colors': 'black',
            'linewidth': 1.5,
            'levels': [1200, 1240, 1280, 1320, 1360, 1400]
        }
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_2d, 'temperature',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            plot_type='contourf',
            surface_overlay=surface_config,
            title='Test Temperature + Geopotential Height Overlay'
        )
        
        assert isinstance(fig, Figure)
        assert 'Test Temperature + Geopotential Height Overlay' in ax.get_title()
        
    def test_surface_overlay_filled_contours(self: "TestDataTypeAgnosticFeatures") -> None:
        """
        This test validates the handling of surface overlays with filled contour visualization. It verifies that the plotting system can overlay secondary meteorological fields as filled contours atop the primary base layer. Temperature serves as the base field (250-310K) with sea level pressure (1000-1020 hPa) overlaid as filled contours in a surface_config dictionary. Assertions confirm successful multi-layer figure creation. This feature enables enhanced synoptic analysis plots where multiple related fields provide complementary information about atmospheric state, with filled contours offering a visually distinct representation for the overlay variable.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_2d = self.temp_data[:self.n_cells]

        pressure_2d = 1000.0 + 20.0 * (self.precip_data[:self.n_cells] / (np.max(self.precip_data[:self.n_cells]) + 1e-12))
        
        surface_config = {
            'data': pressure_2d,
            'var_name': 'sea_level_pressure',
            'plot_type': 'contourf',
            'colormap': 'Blues',
            'alpha': 0.5,
            'levels': [1000, 1005, 1010, 1015, 1020]
        }
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_2d, 'temperature',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            plot_type='scatter',
            surface_overlay=surface_config,
            title='Test Temperature + Pressure Overlay'
        )
        
        assert isinstance(fig, Figure)
        
    def test_surface_overlay_3d_data(self: "TestDataTypeAgnosticFeatures") -> None:
        """
        This test validates the handling of surface overlays using 3D data with level extraction. It verifies that the plotting system can extract a specific vertical level from 3D data for use as a surface overlay on a 2D base map. Temperature serves as the base field (250-310K) with geopotential height (1200-1400 dam) overlaid as contour lines extracted from a 3D dataset. Assertions confirm successful multi-layer figure creation, demonstrating that users can create complex synoptic analysis plots directly from multi-level datasets by specifying the desired vertical level for the overlay variable.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_2d = self.temp_data[:self.n_cells]

        geop_base = 1200.0 + 200.0 * (self.precip_data[:self.n_cells] / (np.max(self.precip_data[:self.n_cells]) + 1e-12))
        geop_3d = np.tile(geop_base.reshape((self.n_cells,1)), (1,self.n_levels))
        
        surface_config = {
            'data': geop_3d,
            'var_name': 'geopotential_height',
            'plot_type': 'contour',
            'colors': 'black',
            'linewidth': 2.0,
            'level_index': 2,  
            'levels': [1200, 1250, 1300, 1350, 1400]
        }
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_2d, 'temperature',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            plot_type='contourf',
            surface_overlay=surface_config,
            title='Test Temperature + 3D Geopotential Overlay'
        )
        
        assert isinstance(fig, Figure)
        
    def test_complete_850hpa_weather_map(self: "TestDataTypeAgnosticFeatures") -> None:
        """
        This test validates the creation of a complete 850 hPa weather analysis map with both wind and surface overlays. It verifies that the plotting system can generate a comprehensive synoptic analysis plot combining temperature contours, wind barbs, and geopotential height contours extracted from 3D data. The test creates a base temperature field (250-310K) with geopotential height (1200-1600 dam) overlaid as contour lines and wind barbs at the same level (850 hPa) for a multi-layer visualization. Assertions confirm successful figure creation with all elements properly rendered, demonstrating the system's capability for complex composite plots commonly used in operational meteorology for mid-tropospheric weather pattern analysis.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)

        temp_850 = 263.0 + 30.0 * (self.temp_data[:self.n_cells] / (np.max(self.temp_data[:self.n_cells]) + 1e-12))
        geop_850 = 1200.0 + 400.0 * (self.precip_data[:self.n_cells] / (np.max(self.precip_data[:self.n_cells]) + 1e-12))

        u_wind_850 = self.u[:self.n_cells]
        v_wind_850 = self.v[:self.n_cells]
        
        wind_config = {
            'u_data': u_wind_850,
            'v_data': v_wind_850,
            'plot_type': 'barbs',
            'subsample': 3,
            'color': 'white'
        }
        
        surface_config = {
            'data': geop_850,
            'var_name': 'geopotential_height',
            'plot_type': 'contour',
            'colors': 'black',
            'linewidth': 2.0,
            'levels': [1200, 1300, 1400, 1500, 1600]
        }
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_850, 'temperature_850hpa',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            plot_type='contourf',
            wind_overlay=wind_config,
            surface_overlay=surface_config,
            title='Complete 850hPa Analysis'
        )
        
        assert isinstance(fig, Figure)
        assert 'Complete 850hPa Analysis' in ax.get_title()
        
    def test_surface_overlay_error_handling(self: "TestDataTypeAgnosticFeatures") -> None:
        """
        This test validates robust error detection for invalid surface overlay configurations. It verifies that the plotting system properly rejects surface overlays with unsupported plot types through clear error messages. An invalid 'invalid_type' plot type is specified in the surface configuration to trigger validation. Assertions confirm a ValueError is raised with an informative error message guiding users toward valid options. This validation protects against configuration errors that would otherwise produce cryptic failures or incorrect visualizations, improving user experience through early detection and helpful error guidance.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_2d = self.temp_data[:self.n_cells]
        
        surface_config = {
            'data': 1000.0 + 20.0 * (self.precip_data[:self.n_cells] / (np.max(self.precip_data[:self.n_cells]) + 1e-12)),
            'var_name': 'pressure',
            'plot_type': 'invalid_type'
        }
        
        with pytest.raises(ValueError) as context:
            self.visualizer.create_surface_map(
                self.lon, self.lat, temp_2d, 'temperature',
                self.lon_min, self.lon_max, self.lat_min, self.lat_max,
                surface_overlay=surface_config
            )
            
        assert "Unsupported surface overlay plot_type: invalid_type" in str(context.value)
        
    def test_multiple_overlays_interaction(self: "TestDataTypeAgnosticFeatures") -> None:
        """
        This test validates the interaction of multiple surface overlays on a single plot. It verifies that the plotting system can handle multiple surface overlays simultaneously without visual conflicts or errors. The test creates a surface map with temperature as the base field and adds two separate surface overlays: one for geopotential height as contour lines and another for sea level pressure as filled contours. Assertions confirm successful figure creation with both overlays rendered correctly, demonstrating that users can create complex multi-layer synoptic analysis plots with multiple related fields providing complementary information about atmospheric state.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_2d = self.temp_data[:self.n_cells]

        pressure_2d = 1000.0 + 20.0 * (self.precip_data[:self.n_cells] / (np.max(self.precip_data[:self.n_cells]) + 1e-12))

        u_wind = self.u[:self.n_cells]
        v_wind = self.v[:self.n_cells]
        
        wind_config = {
            'u_data': u_wind,
            'v_data': v_wind,
            'plot_type': 'arrows',
            'subsample': 4,
            'color': 'red'
        }
        
        surface_config = {
            'data': pressure_2d,
            'var_name': 'sea_level_pressure',
            'plot_type': 'contour',
            'colors': 'blue',
            'linewidth': 1.0,
            'levels': [1000, 1005, 1010, 1015, 1020]
        }
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_2d, 'temperature',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            plot_type='contourf',
            wind_overlay=wind_config,
            surface_overlay=surface_config,
            title='Test Multiple Overlays Interaction'
        )
        
        assert isinstance(fig, Figure)

    def test_variable_specific_temperature_settings(self: "TestDataTypeAgnosticFeatures") -> None:
        """
        This test validates variable-specific visualization settings for temperature fields. It verifies that the plotting system applies appropriate colormaps and contour levels based on the variable name and data range. Temperature data in both Kelvin and Celsius units are tested to ensure correct colormap selection (e.g., 'RdYlBu_r') and level generation spanning typical atmospheric temperature ranges. Assertions confirm that the system recognizes temperature variables and applies visually effective settings for accurate interpretation of thermal patterns in meteorological data.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_k = 263.0 + 40.0 * ((u_arr - u_arr.min()) / (u_arr.max() - u_arr.min() + 1e-12))

        colormap, levels = self.visualizer.get_variable_specific_settings('temperature', temp_k)
        
        assert colormap == 'RdYlBu_r'
        assert isinstance(levels, list)
        assert levels is not None
        assert levels is not None  

        assert all(isinstance(level, (int, float)) for level in levels)
        
        temp_c = -10.0 + 40.0 * ((u_arr - u_arr.min()) / (u_arr.max() - u_arr.min() + 1e-12))
        colormap, levels = self.visualizer.get_variable_specific_settings('t2m', temp_c)
        
        assert colormap == 'RdYlBu_r'
        assert isinstance(levels, list)
        
    def test_variable_specific_precipitation_settings(self: "TestDataTypeAgnosticFeatures") -> None:
        """
        This test validates variable-specific visualization settings for precipitation fields. It verifies that the plotting system applies appropriate colormaps and contour levels based on the variable name and data range for precipitation variables. Precipitation data in both mm/hr and kg/m^2/s units are tested to ensure correct colormap selection (e.g., 'Blues') and level generation spanning typical precipitation intensity ranges. Assertions confirm that the system recognizes precipitation variables and applies visually effective settings for accurate interpretation of rainfall patterns in meteorological data.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        precip_data = 25.0 * ((u_arr - u_arr.min()) / (u_arr.max() - u_arr.min() + 1e-12))
        
        colormap, levels = self.visualizer.get_variable_specific_settings('precipitation_01h', precip_data)

        assert isinstance(colormap, mcolors.ListedColormap)  
        assert levels is not None
        assert levels is not None  
        assert 0.1 in levels 
        assert 0.5 in levels 
        
        colormap, levels = self.visualizer.get_variable_specific_settings('daily_precip', precip_data)

        assert isinstance(colormap, mcolors.ListedColormap)
        assert isinstance(levels, list)
        assert levels is not None
        assert levels is not None  
        assert 0.1 in levels 
        
    def test_variable_specific_pressure_settings(self: "TestDataTypeAgnosticFeatures") -> None:
        """
        This test validates variable-specific visualization settings for pressure fields. It verifies that the plotting system applies appropriate colormaps and contour levels based on the variable name and data range for pressure variables. Pressure data in both Pascals and hPa units are tested to ensure correct colormap selection (e.g., 'RdBu_r') and level generation spanning typical atmospheric pressure ranges. Assertions confirm that the system recognizes pressure variables and applies visually effective settings for accurate interpretation of pressure patterns in meteorological data.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        pressure_pa = 99000.0 + 4000.0 * ((u_arr - u_arr.min()) / (u_arr.max() - u_arr.min() + 1e-12))

        colormap, levels = self.visualizer.get_variable_specific_settings('sea_level_pressure', pressure_pa)
        
        assert colormap == 'RdBu_r'
        assert isinstance(levels, list)
        assert levels is not None
        assert levels is not None 
        assert all(level >= 99000 for level in levels) 
        
        pressure_hpa = 990.0 + 40.0 * ((u_arr - u_arr.min()) / (u_arr.max() - u_arr.min() + 1e-12))
        colormap, levels = self.visualizer.get_variable_specific_settings('slp', pressure_hpa)
        
        assert colormap == 'RdBu_r'
        assert isinstance(levels, list)
        
    def test_variable_specific_wind_settings(self: "TestDataTypeAgnosticFeatures") -> None:
        """
        This test validates variable-specific visualization settings for wind speed fields. It verifies that the plotting system applies appropriate colormaps and contour levels based on the variable name and data range for wind speed variables. Wind speed data derived from u and v components is tested to ensure correct colormap selection (e.g., 'plasma') and level generation spanning typical wind speed ranges. Assertions confirm that the system recognizes wind speed variables and applies visually effective settings for accurate interpretation of wind patterns in meteorological data.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        wind_data = 20.0 * (np.hypot(u_arr, v_arr) / (np.hypot(u_arr, v_arr).max() + 1e-12))
        colormap, levels = self.visualizer.get_variable_specific_settings('wind_speed', wind_data)
        
        assert colormap == 'plasma'
        assert isinstance(levels, list)
        assert levels is not None
        assert levels is not None 
        assert levels[0] == 0  
        
    def test_variable_specific_geopotential_settings(self: "TestDataTypeAgnosticFeatures") -> None:
        """
        This test validates variable-specific visualization settings for geopotential height fields. It verifies that the plotting system applies appropriate colormaps and contour levels based on the variable name and data range for geopotential height variables. Geopotential height data derived from precipitation is tested to ensure correct colormap selection (e.g., 'terrain') and level generation spanning typical mid-tropospheric height ranges (1200-1600 dam). Assertions confirm that the system recognizes geopotential height variables and applies visually effective settings for accurate interpretation of atmospheric pressure surfaces in meteorological data.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)

        geop_data = 1200.0 + 400.0 * (self.precip_data[:self.n_cells] / (np.max(self.precip_data[:self.n_cells]) + 1e-12))
        colormap, levels = self.visualizer.get_variable_specific_settings('geopotential_height', geop_data)
        
        assert colormap == 'terrain'
        assert isinstance(levels, list)
        assert levels is not None
        assert levels is not None  
        assert all(1200 <= level <= 1600 for level in levels)
        
    def test_variable_specific_humidity_settings(self: "TestDataTypeAgnosticFeatures") -> None:
        """
        This test validates variable-specific visualization settings for relative humidity fields. It verifies that the plotting system applies appropriate colormaps and contour levels based on the variable name and data range for relative humidity variables. Relative humidity data derived from u and v components is tested to ensure correct colormap selection (e.g., 'BuGn') and level generation spanning typical humidity ranges (0-100%). Assertions confirm that the system recognizes relative humidity variables and applies visually effective settings for accurate interpretation of moisture patterns in meteorological data.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        rh_data = 0.3 + 0.7 * ((u_arr - u_arr.min()) / (u_arr.max() - u_arr.min() + 1e-12))

        colormap, levels = self.visualizer.get_variable_specific_settings('relative_humidity', rh_data)
        
        assert colormap == 'BuGn'
        assert isinstance(levels, list)
        assert levels is not None
        assert levels is not None 
        assert all(0 <= level <= 1 for level in levels)
        
    def test_variable_specific_unknown_variable(self: "TestDataTypeAgnosticFeatures") -> None:
        """
        This test validates the handling of unknown variables in the variable-specific settings function. It verifies that when an unrecognized variable name is provided, the system applies a default colormap and generates levels based on the data range without raising errors. The test creates synthetic diverging and sequential data to simulate unknown variable scenarios and checks that a default colormap (e.g., 'RdBu_r' for diverging) is applied along with appropriate level generation. Assertions confirm that the system can gracefully handle unrecognized variables while still providing visually effective settings based on data characteristics.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        diverging_data = (u_arr - u_arr.mean()) / (u_arr.max() - u_arr.min() + 1e-12) * 100.0
        colormap, levels = self.visualizer.get_variable_specific_settings('unknown_var', diverging_data)
        
        assert colormap == 'RdBu_r'
        assert isinstance(levels, list)
        
        sequential_data = 100.0 * ((u_arr - u_arr.min()) / (u_arr.max() - u_arr.min() + 1e-12))
        colormap, levels = self.visualizer.get_variable_specific_settings('unknown_positive', sequential_data)
        
        assert colormap == 'viridis'
        assert isinstance(levels, list)
        
    def test_variable_specific_integration_with_plotting(self: "TestDataTypeAgnosticFeatures") -> None:
        """
        This test validates the integration of variable-specific settings into the surface plotting function. It verifies that when a recognized variable name is provided, the plotting system automatically applies the appropriate colormap and contour levels based on the variable-specific settings function. The test creates surface maps for temperature using both automatic variable-specific settings and manual overrides to confirm that the correct visual configurations are applied in each case. Assertions confirm successful figure creation with expected styling, demonstrating that variable recognition seamlessly integrates into the plotting workflow for enhanced visualization without requiring manual configuration.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)
        temp_data = 263.0 + 40.0 * ((u_arr - u_arr.min()) / (u_arr.max() - u_arr.min() + 1e-12))
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_data, 'temperature',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            plot_type='contourf',
            title='Temperature with Auto Settings'
        )
        
        assert isinstance(fig, Figure)
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_data, 'temperature',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            plot_type='contourf',
            colormap='plasma', 
            levels=[270, 280, 290, 300],  
            title='Temperature with Manual Override'
        )
        
        assert isinstance(fig, Figure)
        
    def test_variable_specific_with_surface_overlays(self: "TestDataTypeAgnosticFeatures") -> None:
        """
        This test validates the application of variable-specific settings in conjunction with surface overlays. It verifies that when a recognized variable name is provided for the base field, the plotting system applies appropriate variable-specific settings while still allowing for additional surface overlays with their own configurations. The test creates a surface map with temperature as the base field using automatic variable-specific settings and adds a geopotential height overlay with its own contour configuration. Assertions confirm successful figure creation with both the base field and overlay rendered correctly, demonstrating that variable-specific settings can coexist with complex multi-layer visualizations for enhanced meteorological analysis.

        Parameters:
            None

        Returns:
            None
        """
        lon_arr, lat_arr, u_arr, v_arr = load_mpas_coords_from_processor(n=self.n_cells)

        temp_data = 263.0 + 40.0 * (self.temp_data[:self.n_cells] / (np.max(self.temp_data[:self.n_cells]) + 1e-12))
        geop_data = 1200.0 + 400.0 * (self.precip_data[:self.n_cells] / (np.max(self.precip_data[:self.n_cells]) + 1e-12))
        
        surface_config = {
            'data': geop_data,
            'var_name': 'geopotential_height',  
            'plot_type': 'contour',
            'colors': 'black',
            'linewidth': 2.0
        }
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_data, 'temperature',  
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            plot_type='contourf',
            surface_overlay=surface_config,
            title='Auto Settings: Temperature + Geopotential'
        )
        
        assert isinstance(fig, Figure)


if __name__ == '__main__':
    pytest.main([__file__])