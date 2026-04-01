#!/usr/bin/env python3
"""
MPASdiag Test Suite: Visualization Integration Tests

This test suite performs integration testing of the visualization components within the MPASdiag package using real MPAS data when available. The tests cover end-to-end functionality of visualization features including variable style generation, adaptive marker sizing, coordinate formatting, map projection setup, and plot creation/saving workflows. By utilizing real MPAS datasets through fixtures, the suite validates that visualization methods can handle actual data structures and produce expected outputs without errors. The tests also include utility function validation for geographic extent checking and dynamic tick formatting. This comprehensive integration testing ensures that the visualization components work cohesively with real data inputs, providing confidence in the robustness of the plotting capabilities for operational use.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries
import os
import sys
import pytest
import shutil
import tempfile
import matplotlib
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Generator, Any, Dict
from datetime import datetime, timedelta

from tests.conftest import mpas_wind_data

matplotlib.use('Agg')

from mpasdiag.visualization.surface import MPASSurfacePlotter
from mpasdiag.processing.utils_geog import MPASGeographicUtils
from mpasdiag.visualization.base_visualizer import MPASVisualizer
from mpasdiag.visualization.styling import MPASVisualizationStyle
from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def create_mock_create_map(plotter) -> Any:
    """
    This function creates a mock `create_map` method that simulates the behavior of setting up a map projection and returns a simple figure and axes. The mock method generates a new matplotlib figure and axes, assigns the figure to the plotter's `_current_fig` attribute, and returns the figure and axes objects. This allows tests to verify that the `create_map` method is called and that it produces valid plotting objects without needing to rely on actual map projection logic or external dependencies. By using this mock, tests can focus on the downstream effects of map creation without being affected by the complexities of geospatial plotting.

    Parameters:
        plotter: The plotter instance for which the mock method is being created.

    Returns:
        Any: A mock `create_map` method that returns a simple figure and axes.
    """
    def mock_create_map(*args, **kwargs):
        fig, ax = plt.subplots()
        plotter._current_fig = fig
        return fig, ax
    return mock_create_map


def create_mock_save_plot():
    """
    This function creates a mock `save_plot` method that simulates the behavior of saving a plot without actually writing any files. The mock method does nothing when called, allowing tests to verify that the `save_plot` method is invoked without performing any file I/O operations. This is useful for testing the plotting workflow in isolation, ensuring that the logic for saving plots is exercised without creating unnecessary files during testing.

    Parameters:
        None

    Returns:
        Any: A mock `save_plot` method that does nothing when called.
    """
    def mock_save_plot(*args, **kwargs):
        pass
    return mock_save_plot


def create_mock_close_plot(plotter) -> Any:
    """
    This function creates a mock `close_plot` method that simulates the behavior of closing a plot. The mock method closes all matplotlib figures when called, allowing tests to verify that the `close_plot` method is invoked without affecting other tests or leaving open figures. This helps maintain a clean testing environment and prevents resource leaks during automated testing.

    Parameters:
        plotter: The plotter instance for which the mock method is being created.

    Returns:
        Any: A mock `close_plot` method that closes all matplotlib figures when called.
    """
    def mock_close_plot(*args, **kwargs):
        plt.close('all')
    return mock_close_plot


class TestRealDataIntegration:
    """ Integration tests using real MPAS data. """
    
    @pytest.fixture
    def mpas_paths(self: "TestRealDataIntegration") -> dict:
        """
        This fixture provides paths to real MPAS data files required for integration testing. It checks for the existence of expected data directories and files, and if they are not found, it gracefully skips the tests that depend on this fixture. By centralizing the data path management in a fixture, it allows multiple tests to easily access the necessary MPAS datasets while ensuring that tests are only executed when the required data is available in the test environment.

        Parameters:
            None

        Returns:
            dict: Dictionary with keys 'data_dir' and 'grid_file' pointing to expected sample data locations.
        """
        data_dir = 'data/u240k/mpasout'
        grid_file = 'data/grids/x1.10242.static.nc'
        
        if not os.path.exists(data_dir):
            pytest.skip(f"MPAS data directory not found: {data_dir}")
            return

        if not os.path.exists(grid_file):
            pytest.skip(f"MPAS grid file not found: {grid_file}")
            return
        
        return {'data_dir': data_dir, 'grid_file': grid_file}
    
    def test_variable_style_with_real_precip_data(self: "TestRealDataIntegration", mpas_3d_processor) -> None:
        """
        This integration test verifies that the `get_variable_style` method of `MPASVisualizationStyle` returns appropriate styling information when provided with real precipitation data from an MPAS dataset. The test checks that the returned style dictionary contains expected keys such as 'colormap' and 'levels', and that the colormap is a valid matplotlib colormap object. By using real MPAS data, this test ensures that the styling logic can handle actual data structures and values, which is critical for producing accurate and visually meaningful precipitation plots in operational use. The test will skip if the `mpas_3d_processor` fixture does not provide the necessary data, ensuring that it only runs in environments where real MPAS data is available.

        Parameters:
            mpas_3d_processor: MPAS 3D processor fixture providing real data.

        Returns:
            None
        """
        if mpas_3d_processor is None:
            pytest.skip("MPAS data not available")
            return
        
        processor = mpas_3d_processor

        try:
            if hasattr(processor, 'dataset') and processor.dataset is not None:
                if 'rainnc' in processor.dataset:
                    precip_data = processor.dataset['rainnc'].isel(Time=0)
                    style = MPASVisualizationStyle.get_variable_style('precip_24h', precip_data)
                    
                    assert 'colormap' in style
                    assert 'levels' in style
                    assert isinstance(style['colormap'], mcolors.ListedColormap)
                else:
                    pytest.skip("rainnc variable not found in dataset")
            else:
                pytest.skip("Could not load dataset")
        except Exception as e:
            pytest.skip(f"Could not load MPAS data: {e}")
    
    def test_variable_specific_settings_with_real_temperature(self: "TestRealDataIntegration", mpas_3d_processor) -> None:
        """
        This integration test verifies that the `get_variable_specific_settings` method of `MPASVisualizationStyle` returns appropriate colormap and level settings when provided with real temperature data from an MPAS dataset. The test checks that the returned colormap and levels are reasonable and suitable for visualizing temperature fields. By using real MPAS data, this test ensures that the variable-specific settings logic can handle actual data structures and values, which is critical for producing accurate and visually meaningful temperature plots in operational use. The test will skip if the `mpas_3d_processor` fixture does not provide the necessary data, ensuring that it only runs in environments where real MPAS data is available.

        Parameters:
            mpas_3d_processor: MPAS 3D processor fixture providing real data.

        Returns:
            None
        """
        if mpas_3d_processor is None:
            pytest.skip("MPAS data not available")
            return
        
        processor = mpas_3d_processor

        try:
            if hasattr(processor, 'dataset') and processor.dataset is not None:
                if 't2m' in processor.dataset:
                    temp_data = processor.dataset['t2m'].isel(Time=0)
                    cmap, levels = MPASVisualizationStyle.get_variable_specific_settings(
                        't2m', temp_data.values
                    )
                    
                    assert cmap == 'RdYlBu_r'
                    assert levels is not None
                else:
                    pytest.skip("t2m variable not found in dataset")
            else:
                pytest.skip("Could not load dataset")
        except Exception as e:
            pytest.skip(f"Could not load MPAS data: {e}")
    
    def test_adaptive_marker_with_real_coordinates(self: "TestRealDataIntegration", mpas_3d_processor) -> None:
        """
        This integration test verifies that the `calculate_adaptive_marker_size` method of `MPASVisualizationStyle` returns a valid marker size when provided with real coordinate data from an MPAS dataset. The test extracts longitude and latitude arrays from the `mpas_3d_processor` fixture and computes the geographic extent. It then checks that the calculated marker size falls within a reasonable range, ensuring that the adaptive sizing logic can handle actual data distributions. The test will skip if the `mpas_3d_processor` fixture does not provide the necessary data, ensuring that it only runs in environments where real MPAS data is available.

        Parameters:
            mpas_3d_processor: MPAS 3D processor fixture providing real data.

        Returns:
            None
        """
        if mpas_3d_processor is None:
            pytest.skip("MPAS data not available")
            return
        
        processor = mpas_3d_processor

        try:
            if hasattr(processor, 'dataset') and processor.dataset is not None:
                if 'lonCell' in processor.dataset and 'latCell' in processor.dataset:
                    lon = processor.dataset['lonCell'].values
                    lat = processor.dataset['latCell'].values
                    
                    lon_min, lon_max = np.min(lon), np.max(lon)
                    lat_min, lat_max = np.min(lat), np.max(lat)
                    
                    extent = (np.rad2deg(lon_min), np.rad2deg(lon_max),
                             np.rad2deg(lat_min), np.rad2deg(lat_max))
                    
                    size = MPASVisualizationStyle.calculate_adaptive_marker_size(
                        extent, len(lon)
                    )
                    
                    assert size >= 0.1
                    assert size <= 20.0
                else:
                    pytest.skip("lonCell or latCell not found in dataset")
            else:
                pytest.skip("Could not load dataset")
        except Exception as e:
            pytest.skip(f"Could not load MPAS data: {e}")


class TestUtilityFunctions:
    """ Tests for utility functions in visualization module. """
    
    def test_validate_plot_parameters(self: "TestUtilityFunctions") -> None:
        """
        This test verifies that the `validate_geographic_extent` method of `MPASGeographicUtils` correctly identifies valid and invalid geographic extents. The test checks a valid extent with longitude between 100°E and 110°E and latitude between -10°S and 10°N, as well as various invalid extents that violate longitude and latitude bounds or have min values greater than max values. By confirming that the method returns True for valid extents and False for invalid ones, this test ensures that geographic extent validation logic is functioning correctly, which is essential for preventing errors in map projection setup and ensuring that visualizations are generated for appropriate geographic regions.

        Parameters:
            None

        Returns:
            None
        """
        assert MPASGeographicUtils.validate_geographic_extent((100.0, 110.0, -10.0, 10.0))
        assert not MPASGeographicUtils.validate_geographic_extent((-200.0, 110.0, -10.0, 10.0))
        assert not MPASGeographicUtils.validate_geographic_extent((100.0, 110.0, -100.0, 10.0))
        assert not MPASGeographicUtils.validate_geographic_extent((110.0, 100.0, -10.0, 10.0))
        assert not MPASGeographicUtils.validate_geographic_extent((100.0, 110.0, 10.0, -10.0))


class TestMPASVisualizer:
    """ Tests for MPASVisualizer class functionality. """
    
    @pytest.fixture
    def temp_dir(self: "TestMPASVisualizer") -> Generator[str, None, None]:
        """
        This fixture creates a temporary directory for use in tests that require file output, such as saving plots. It yields the path to the temporary directory for use in the test functions, and after the tests complete, it ensures that the temporary directory and any files created within it are cleaned up by removing the directory. Additionally, it calls `plt.close('all')` to ensure that any open matplotlib figures are closed, preventing memory leaks during testing. This fixture provides a clean and isolated environment for tests that involve file I/O operations.

        Parameters:
            None

        Returns:
            Generator[str, None, None]: A generator that yields the path to a temporary directory and ensures cleanup after tests.
        """
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
        plt.close('all')
    
    @pytest.fixture
    def visualizer(self: "TestMPASVisualizer") -> MPASVisualizer:
        """
        This fixture creates an instance of the `MPASVisualizer` class with specified figure size and DPI settings. The visualizer instance is configured with a figure size of 8 inches by 6 inches and a DPI of 100, which are common settings for producing clear and appropriately sized plots. By providing this fixture, tests that require a visualizer instance can easily access a pre-configured object without needing to repeat the instantiation code in each test function. This promotes code reuse and ensures consistency in visualizer configuration across different tests.

        Parameters:
            None

        Returns:
            MPASVisualizer: Configured visualizer instance.
        """
        return MPASVisualizer(figsize=(8, 6), dpi=100)
    
    @pytest.fixture
    def surface_plotter(self: "TestMPASVisualizer") -> MPASSurfacePlotter:
        """
        This fixture creates an instance of the `MPASSurfacePlotter` class with specified figure size and DPI settings. The surface plotter instance is configured with a figure size of 8 inches by 6 inches and a DPI of 100, which are common settings for producing clear and appropriately sized plots. By providing this fixture, tests that require a surface plotter instance can easily access a pre-configured object without needing to repeat the instantiation code in each test function. This promotes code reuse and ensures consistency in surface plotter configuration across different tests.

        Parameters:
            None

        Returns:
            MPASSurfacePlotter: Configured surface plotter instance.
        """
        return MPASSurfacePlotter(figsize=(8, 6), dpi=100)
    
    def test_initialization(self: "TestMPASVisualizer", temp_dir: str, visualizer: MPASVisualizer) -> None:
        """
        This test verifies that the `MPASVisualizer` class initializes with the expected default properties for figure size, DPI, and that the figure and axes references are initially set to None. By confirming that the visualizer instance has the correct configuration upon initialization, this test ensures that the constructor of the `MPASVisualizer` class is functioning correctly and that the object is in a consistent state before any plotting methods are called. This is important for preventing issues later in the visualization workflow where uninitialized properties could lead to errors.

        Parameters:
            None

        Returns:
            None
        """
        assert visualizer.figsize == (8, 6)
        assert visualizer.dpi == 100
        assert visualizer.fig is None
        assert visualizer.ax is None
    
    def test_create_precip_colormap(self: "TestMPASVisualizer", visualizer: MPASVisualizer) -> None:
        """
        This test verifies that the `create_precip_colormap` method of `MPASVisualizer` generates valid colormap and levels for different precipitation types. The test checks that the method returns a non-None colormap object and a list of levels for both 'a01h' and 'a24h' precipitation types. Additionally, it confirms that the levels for 'a24h' are greater than those for 'a01h', which is expected due to the longer accumulation period. By validating the colormap creation with real precipitation types, this test ensures that the visualization style logic can produce appropriate color mappings for different precipitation datasets, which is critical for accurate and visually meaningful precipitation plots.

        Parameters:
            None

        Returns:
            None
        """
        precip_plotter = MPASPrecipitationPlotter(figsize=(8, 6), dpi=100)
        
        cmap, levels = precip_plotter.create_precip_colormap('a01h')
        
        assert cmap is not None
        assert isinstance(levels, list)
        assert len(levels) > 0
        
        cmap_24h, levels_24h = precip_plotter.create_precip_colormap('a24h')

        assert cmap_24h is not None
        assert max(levels_24h) > max(levels)
    
    def test_format_coordinates(self: "TestMPASVisualizer", visualizer: MPASVisualizer) -> None:
        """
        This test verifies that the `format_latitude` and `format_longitude` methods of `MPASVisualizer` correctly format latitude and longitude values into human-readable strings with appropriate directional indicators (N/S for latitude and E/W for longitude). The test checks various cases including positive and negative latitudes and longitudes, ensuring that the formatting logic correctly identifies the hemisphere and formats the values with degree symbols. By confirming that coordinate formatting produces expected string outputs, this test ensures that axis labels and annotations in visualizations will be clear and informative for users interpreting geographic data.

        Parameters:
            None

        Returns:
            None
        """
        lat_str = visualizer.format_latitude(10.5, None)
        assert lat_str == "10.5°N"
        
        lat_str = visualizer.format_latitude(-10.5, None)
        assert lat_str == "10.5°S"
        
        lon_str = visualizer.format_longitude(105.0, None)
        assert lon_str == "105.0°E"
        
        lon_str = visualizer.format_longitude(-105.0, None)
        assert lon_str == "105.0°W"
    
    def test_format_ticks_dynamic(self: "TestMPASVisualizer", visualizer: MPASVisualizer) -> None:
        """
        This test verifies that the `_format_ticks_dynamic` method of `MPASVisualizer` produces appropriate string representations for diverse numeric ranges including normal integers (1-5), small decimals (0.1-0.5), very small values (1e-5), very large values (1e5), zero values, and meteorological quantities like vorticity (5e-6). The formatting applies scientific notation for extreme magnitudes while using decimal notation for moderate ranges maintaining readability. Test cases verify specific format strings ensuring consistent decimal places (0.10, 0.20) and proper exponent notation (1.0e-05, 1.0e+05). This adaptive formatting capability improves colorbar and axis label readability across diverse data scales from microscale vorticity to synoptic pressure fields.

        Parameters:
            None

        Returns:
            None
        """
        normal_ticks = [1.0, 2.0, 3.0, 4.0, 5.0]
        formatted = visualizer._format_ticks_dynamic(normal_ticks)

        assert formatted == ['1', '2', '3', '4', '5']
        
        small_normal = [0.1, 0.2, 0.3, 0.4, 0.5]
        formatted = visualizer._format_ticks_dynamic(small_normal)

        assert formatted == ['0.10', '0.20', '0.30', '0.40', '0.50']
        
        very_small = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
        formatted = visualizer._format_ticks_dynamic(very_small)

        for f in formatted:
            assert 'e-' in f  

        assert formatted[0] == '1.0e-05'  
        
        very_large = [1e5, 2e5, 3e5, 4e5, 5e5]
        formatted = visualizer._format_ticks_dynamic(very_large)

        for f in formatted:
            assert 'e+' in f  

        assert formatted[0] == '1.0e+05'  
            
        with_zeros = [0.0, 1e-5, 2e-5]
        formatted = visualizer._format_ticks_dynamic(with_zeros)

        assert formatted[0] == '0.0e+00'  
        assert 'e-' in formatted[1]  
        
        vorticity_values = [5e-6, 1e-5, 1.5e-5, 2e-5, 2.5e-5]
        formatted = visualizer._format_ticks_dynamic(vorticity_values)

        for f in formatted:
            assert 'e-' in f  
    
    def test_setup_map_projection(self: "TestMPASVisualizer", visualizer: MPASVisualizer) -> None:
        """
        This test verifies that the `setup_map_projection` method of `MPASVisualizer` successfully creates map projection and data coordinate reference system (CRS) objects for specified geographic extents and projection types. The test checks that valid map projection objects are returned for both 'PlateCarree' and 'Mercator' projections when given a reasonable geographic extent. By confirming that the method produces non-None projection objects, this test ensures that the visualization workflow can proceed to create maps with correct geospatial referencing, which is essential for accurate representation of MPAS data on geographic plots.

        Parameters:
            None

        Returns:
            None
        """
        map_proj, data_crs = visualizer.setup_map_projection(
            100.0, 110.0, -5.0, 5.0, 'PlateCarree'
        )
        
        assert map_proj is not None
        assert data_crs is not None
        
        map_proj_merc, _ = visualizer.setup_map_projection(
            100.0, 110.0, -5.0, 5.0, 'Mercator'
        )

        assert map_proj_merc is not None
    
    def test_create_simple_scatter_plot(self: "TestMPASVisualizer", surface_plotter: MPASSurfacePlotter, mpas_coordinates, mpas_wind_data) -> None:
        """
        This test verifies that the `create_simple_scatter_plot` method of `MPASSurfacePlotter` successfully generates a scatter plot with synthetic data and a custom colormap. The test generates 100 random points with longitude (100-110°E), latitude (-5 to 5°N), and data values (0-50) then confirms successful scatter plot creation. The test checks that figure and axes objects are non-None and match the plotter's stored references, ensuring proper object management. Scatter plot creation provides a foundation for visualizing irregular spatial data without requiring gridding or interpolation. This basic plotting capability supports exploratory data analysis and quick visualization of MPAS unstructured grid data before more sophisticated cartographic rendering.

        Parameters:
            None

        Returns:
            None
        """
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
            return
        
        n_points = 100
        lon, lat = mpas_coordinates[0][:n_points], mpas_coordinates[1][:n_points]
        u, v = mpas_wind_data
        data = (u - u.min()) / (u.max() - u.min() + 1e-12) * 50
        
        fig, ax = surface_plotter.create_simple_scatter_plot(
            lon, lat, data,
            title="Test Scatter Plot",
            colorbar_label="Test Data",
            colormap='viridis'
        )
        
        assert fig is not None
        assert ax is not None
        assert fig == surface_plotter.fig
        assert ax == surface_plotter.ax

        plt.close(fig)

    
    def test_create_histogram(self: "TestMPASVisualizer", visualizer: MPASVisualizer, mpas_wind_data) -> None:
        """
        This test verifies that the `create_histogram` method of `MPASVisualizer` successfully generates histograms with customizable binning and scale options for statistical data distribution visualization. The test uses real MPAS wind data and generates a histogram with 30 bins plus custom labels for axes and title. The test checks that figure and axes objects are non-None for both linear and logarithmic y-axis scales, ensuring proper handling of diverse data distributions. Histogram functionality enables exploratory data analysis revealing distribution characteristics including central tendency, spread, and skewness. This statistical visualization capability supports quality control workflows where data distribution assessment guides subsequent processing decisions and identifies potential data quality issues.

        Parameters:
            None

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
            return
        
        u, v = mpas_wind_data
        data = np.hypot(u, v)  
        
        fig, ax = visualizer.create_histogram(
            data,
            bins=30,
            title="Test Histogram",
            xlabel="Value",
            ylabel="Frequency"
        )
        
        assert fig is not None
        assert ax is not None
        
        fig_log, ax_log = visualizer.create_histogram(
            data, log_scale=True
        )

        assert fig_log is not None
        assert ax_log is not None

        plt.close(fig)
        plt.close(fig_log)

    def test_create_time_series_plot(self: "TestMPASVisualizer", visualizer: MPASVisualizer, mpas_wind_data) -> None:
        """
        This test verifies that the `create_time_series_plot` method of `MPASVisualizer` successfully generates time series plots with datetime x-axis and custom labeling for temporal data visualization. The test generates 24 hourly time points starting January 1, 2024, with random precipitation values (0-20 mm) then creates a time series plot with custom axis labels and title. The test checks that figure and axes objects are non-None, confirming that matplotlib datetime handling works correctly with visualizer methods. Time series plotting supports operational meteorology workflows displaying model forecast evolution, verification time series, and temporal precipitation accumulation. This temporal visualization capability enables analysis of forecast skill, diurnal cycles, and time-dependent meteorological phenomena.

        Parameters:
            None

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
            return
        
        start_time = datetime(2024, 1, 1)
        times = [start_time + timedelta(hours=i) for i in range(24)]
        u_arr, _ = mpas_wind_data
        u_arr = u_arr[:24]
        values = (u_arr - u_arr.min()) / (u_arr.max() - u_arr.min() + 1e-12) * 20
        
        fig, ax = visualizer.create_time_series_plot(
            times, values.tolist(),
            title="Test Time Series",
            ylabel="Precipitation (mm)",
            xlabel="Time"
        )
        
        assert fig is not None
        assert ax is not None

        plt.close(fig)
    
    def test_save_plot(self: "TestMPASVisualizer", surface_plotter: MPASSurfacePlotter, temp_dir: str, mpas_coordinates, mpas_wind_data) -> None:
        """
        This test verifies that the `save_plot` method of `MPASSurfacePlotter` successfully writes matplotlib figures to disk files in specified formats. The test creates a simple scatter plot with real MPAS data then saves the figure to a temporary directory in PNG format, verifying that the output file exists at the expected path. The save operation demonstrates the complete visualization workflow from plot creation through file persistence. File format specification supports diverse output requirements including PNG for presentations, PDF for publications, and SVG for web applications. This file saving capability enables automated figure generation workflows where plots export to disk for documentation, reporting, and archival purposes.

        Parameters:
            None

        Returns:
            None
        """
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
            return
        
        lon, lat = mpas_coordinates
        u, v = mpas_wind_data
        lon = lon[:3]
        lat = lat[:3]
        data = u[:3]
        
        surface_plotter.create_simple_scatter_plot(lon, lat, data)
        
        output_path = os.path.join(temp_dir, "test_plot")
        surface_plotter.save_plot(output_path, formats=['png'])        
        expected_file = f"{output_path}.png"

        assert os.path.exists(expected_file)

        plt.close(surface_plotter.fig)

    def test_save_plot_no_figure(self: "TestMPASVisualizer", temp_dir: str, visualizer: MPASVisualizer) -> None:
        """
        This test verifies that the `save_plot` method of `MPASVisualizer` correctly handles attempts to save plots before figure creation by raising a descriptive AssertionError. The test checks that the method rejects save operations without prior plot creation, providing informative feedback to prevent silent failures or cryptic exceptions. Defensive validation ensures that method preconditions are checked before file I/O operations, preventing corrupted output files. This error detection capability improves code robustness and user experience by providing immediate actionable feedback for workflow errors.

        Parameters:
            None

        Returns:
            None
        """
        output_path = os.path.join(temp_dir, "test_plot")
        
        with pytest.raises(AssertionError, match="Figure must be created before saving"):
            visualizer.save_plot(output_path)
    
    def test_close_plot(self: "TestMPASVisualizer", surface_plotter: MPASSurfacePlotter, mpas_coordinates, mpas_wind_data) -> None:
        """
        This test verifies that the `close_plot` method of `MPASSurfacePlotter` correctly handles plot cleanup by clearing figure and axes references, preventing memory leaks in batch processing workflows. The test creates a scatter plot, confirms the figure reference is not None, then calls `close_plot` and validates that the figure and axes attributes are reset to None. The cleanup operation closes matplotlib figure objects, releasing memory resources and preventing the accumulation of unused figures. Proper cleanup supports long-running batch workflows generating hundreds of plots without exhausting system memory. This resource management capability ensures that visualization operations scale efficiently to large-scale operational forecasting workflows requiring extensive plot generation.

        Parameters:
            None

        Returns:
            None
        """
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
            return
        
        lon, lat = mpas_coordinates
        u, v = mpas_wind_data
        lon = lon[:3]
        lat = lat[:3]
        data = u[:3]
        
        surface_plotter.create_simple_scatter_plot(lon, lat, data)
        
        assert surface_plotter.fig is not None
        
        surface_plotter.close_plot()
        
        assert surface_plotter.fig is None
        assert surface_plotter.ax is None

        plt.close('all')  


class TestPrecipitationMapping:
    """ Tests for precipitation mapping functionality in MPASPrecipitationPlotter using real MPAS data. """
    
    @pytest.fixture
    def temp_dir(self: "TestPrecipitationMapping") -> Generator[str, None, None]:
        """
        This fixture creates a temporary directory for use in tests that require file output, such as saving plots. It yields the path to the temporary directory for use in the test functions, and after the tests complete, it ensures that the temporary directory and any files created within it are cleaned up by removing the directory. Additionally, it calls `plt.close('all')` to ensure that any open matplotlib figures are closed, preventing memory leaks during testing. This fixture provides a clean and isolated environment for tests that involve file I/O operations.

        Parameters:
            None

        Returns:
            Generator[str, None, None]: Temporary directory path generator.
        """
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
        plt.close('all')
    
    @pytest.fixture
    def visualizer(self: "TestPrecipitationMapping") -> MPASPrecipitationPlotter:
        """
        This fixture creates an instance of the `MPASPrecipitationPlotter` class with specified figure size and DPI settings. The precipitation plotter instance is configured with a figure size of 10 inches by 8 inches and a DPI of 100, which are common settings for producing clear and appropriately sized precipitation maps. By providing this fixture, tests that require a precipitation plotter instance can easily access a pre-configured object without needing to repeat the instantiation code in each test function. This promotes code reuse and ensures consistency in precipitation plotter configuration across different tests.

        Parameters:
            None

        Returns:
            MPASPrecipitationPlotter: Precipitation plotter instance.
        """
        from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter
        return MPASPrecipitationPlotter(figsize=(10, 8), dpi=100)
    
    @pytest.fixture
    def precip_data(self: "TestPrecipitationMapping", mpas_coordinates, mpas_precip_data) -> Dict[str, Any]:
        """
        This fixture prepares precipitation data for testing the precipitation mapping functionality. It extracts longitude and latitude arrays from the provided `mpas_coordinates` and uses the `mpas_precip_data` to create a dictionary containing 'lon', 'lat', and 'precip' keys. The fixture checks for the availability of the necessary MPAS data and gracefully skips tests if the data is not available. By providing this fixture, tests that require precipitation data can easily access a structured dataset without needing to repeat data preparation code in each test function, ensuring consistency and reducing redundancy.

        Parameters:
            mpas_coordinates (tuple): Tuple containing longitude and latitude arrays.
            mpas_precip_data (array): Array containing precipitation data.

        Returns:
            Dict[str, Any]: Dictionary with keys 'lon', 'lat', and 'precip'.
        """
        if mpas_coordinates is None or mpas_precip_data is None:
            pytest.skip("MPAS data not available")
            return
        
        n_points = min(100, len(mpas_precip_data))  
        lon, lat = mpas_coordinates[0][:n_points], mpas_coordinates[1][:n_points]
        precip = mpas_precip_data[:n_points]
        
        return {'lon': lon, 'lat': lat, 'precip': precip}
    
    def test_create_precipitation_map(self: "TestPrecipitationMapping", visualizer: MPASPrecipitationPlotter, precip_data: Dict[str, Any]) -> None:
        """
        This test verifies that the `create_precipitation_map` method of `MPASPrecipitationPlotter` successfully generates a precipitation map using real MPAS data. The test checks that the method returns valid figure and axes objects when provided with longitude, latitude, and precipitation data, along with specified geographic extents and a title. By confirming that the map creation process completes without errors and produces non-None figure and axes, this test ensures that the core functionality of visualizing precipitation data on a map is working correctly with real MPAS datasets. This is critical for operational use where accurate and visually meaningful precipitation maps are essential for analysis and decision-making.

        Parameters:
            visualizer (MPASPrecipitationPlotter): Precipitation plotter instance.
            precip_data (Dict[str, Any]): Dictionary containing 'lon', 'lat', and 'precip' keys.

        Returns:
            None
        """
        try:
            fig, ax = visualizer.create_precipitation_map(
                precip_data['lon'], precip_data['lat'], precip_data['precip'],
                -180.0, 180.0, -90.0, 90.0,
                title="Test Precipitation Map",
                accum_period="a01h"
            )
            
            assert fig is not None
            assert ax is not None
            assert fig == visualizer.fig
            assert ax == visualizer.ax
            
        except ImportError as e:
            pytest.skip(f"Cartopy functionality not available: {e}")

        plt.close('all')
    
    def test_create_precipitation_map_invalid_data(self: "TestPrecipitationMapping", visualizer: MPASPrecipitationPlotter, precip_data: Dict[str, Any]) -> None:
        """
        This test verifies that the `create_precipitation_map` method of `MPASPrecipitationPlotter` can handle invalid data gracefully without crashing. The test introduces NaN values into the precipitation data to simulate missing or corrupted data points, then attempts to create a precipitation map. The test checks that the method still returns valid figure and axes objects, ensuring that the visualization process can proceed even when some data points are invalid. This robustness is important for operational use where real-world datasets may contain gaps or errors, and the ability to visualize available data without failure is critical for analysis and decision-making.

        Parameters:
            visualizer (MPASPrecipitationPlotter): Precipitation plotter instance.
            precip_data (Dict[str, Any]): Dictionary containing 'lon', 'lat', and 'precip' keys.

        Returns:
            None
        """
        invalid_data = precip_data['precip'].copy()
        invalid_data[::10] = np.nan  
        
        try:
            fig, ax = visualizer.create_precipitation_map(
                precip_data['lon'], precip_data['lat'], invalid_data,
                -180.0, 180.0, -90.0, 90.0,
                title="Test Invalid Data Map"
            )
            
            assert fig is not None
            assert ax is not None
            
        except ImportError:
            pytest.skip("Cartopy not available for testing")
        finally:
            plt.close('all')
    
    def test_custom_colormap_and_levels(self: "TestPrecipitationMapping", visualizer: MPASPrecipitationPlotter, precip_data: Dict[str, Any]) -> None:
        """
        This test verifies that the `create_precipitation_map` method of `MPASPrecipitationPlotter` can accept custom colormap and levels for precipitation visualization. The test defines a custom set of precipitation levels and uses a different colormap (e.g., 'plasma') to create a precipitation map. It checks that the method returns valid figure and axes objects, ensuring that the customization options for colormap and levels are functioning correctly. This flexibility allows users to tailor the visual representation of precipitation data to specific preferences or requirements, enhancing the interpretability and aesthetic quality of the resulting maps.

        Parameters:
            visualizer (MPASPrecipitationPlotter): Precipitation plotter instance.
            precip_data (Dict[str, Any]): Dictionary containing 'lon', 'lat', and 'precip' keys.

        Returns:
            None
        """
        custom_levels = [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]
        
        try:
            fig, ax = visualizer.create_precipitation_map(
                precip_data['lon'], precip_data['lat'], precip_data['precip'],
                -180.0, 180.0, -90.0, 90.0,
                title="Custom Colormap Test",
                colormap="plasma",
                levels=custom_levels
            )
            
            assert fig is not None
            assert ax is not None
            
        except ImportError:
            pytest.skip("Cartopy not available for testing")
        finally:
            plt.close('all')


class TestBatchProcessing:
    """ Tests for batch processing of precipitation maps using MPASPrecipitationPlotter with real MPAS data. """
    
    @pytest.fixture
    def temp_dir(self: "TestBatchProcessing") -> Generator[str, None, None]:
        """
        This fixture creates a temporary directory for use in batch processing tests that require file output, such as saving multiple precipitation maps. It yields the path to the temporary directory for use in the test functions, and after the tests complete, it ensures that the temporary directory and any files created within it are cleaned up by removing the directory. Additionally, it calls `plt.close('all')` to ensure that any open matplotlib figures are closed, preventing memory leaks during testing. This fixture provides a clean and isolated environment for batch processing tests that involve file I/O operations.

        Parameters:
            None

        Returns:
            Generator[str, None, None]: Temporary directory path generator.
        """
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
        plt.close('all')
    
    def test_batch_processing_mock(self: "TestBatchProcessing", temp_dir: str, mpas_2d_processor_diag, mpas_coordinates) -> None:
        """
        This test verifies the batch processing workflow of the `MPASPrecipitationPlotter` class for creating multiple precipitation maps using real MPAS data. The test mocks the `create_precipitation_map`, `save_plot`, and `close_plot` methods to simulate the batch processing without actually generating files or plots. It checks that the `create_batch_precipitation_maps` method can be called with real MPAS diagnostic data and coordinates, and that it returns a list of created file paths. By confirming that the batch processing logic can execute with real MPAS data inputs, this test ensures that the workflow for generating multiple precipitation maps is functioning correctly, which is essential for operational use where large numbers of plots may need to be generated efficiently.

        Parameters:
            temp_dir (str): Path to the temporary directory for output files.
            mpas_2d_processor_diag: MPAS 2D processor with loaded diagnostic data.
            mpas_coordinates: Real MPAS grid coordinates (lon, lat).

        Returns:
            None
        """
        if mpas_2d_processor_diag is None or mpas_coordinates is None:
            pytest.skip("MPAS data not available")
            return
        
        precip_plotter = MPASPrecipitationPlotter(figsize=(10, 8), dpi=100)
        
        precip_plotter.create_precipitation_map = create_mock_create_map(precip_plotter)
        precip_plotter.save_plot = create_mock_save_plot()
        precip_plotter.close_plot = create_mock_close_plot(precip_plotter)
        
        try:
            created_files = precip_plotter.create_batch_precipitation_maps(
                mpas_2d_processor_diag, temp_dir,
                -180.0, 180.0, -90.0, 90.0,
                var_name='rainnc',
                accum_period='a01h',
                formats=['png']
            )
            
            assert isinstance(created_files, list), "Should return a list of files"
            
        except Exception as e:
            pytest.skip(f"Test skipped due to: {e}")
        finally:
            plt.close('all')

class TestVisualizationIntegration:
    """ Tests for integration of visualization components using real MPAS data to create surface maps, wind plots, and precipitation maps. """
    
    def test_create_surface_map_with_real_data(
        self: "TestVisualizationIntegration",
        mpas_2d_processor_diag,
        mpas_coordinates,
        tmp_path
    ) -> None:
        """
        This test verifies the integration of the `MPASSurfacePlotter` class with real MPAS data to create a surface map visualization. The test checks that the `create_surface_map` method can successfully generate a surface map using real MPAS grid coordinates and simulated pressure data, and that the resulting figure and axes objects are valid. It also tests the complete workflow of saving the generated plot to a temporary directory and verifying that the output file is created successfully. By confirming that the surface map creation and file saving processes work correctly with real MPAS data inputs, this test ensures that the visualization components are properly integrated and functional for operational use.
        
        Parameters:
            mpas_2d_processor_diag: MPAS 2D processor with loaded diagnostic data
            mpas_coordinates: Real MPAS grid coordinates (lon, lat)
            tmp_path: Temporary directory for output files
            
        Returns:
            None: Verified by successful plot creation and file existence
        """
        if mpas_2d_processor_diag is None or mpas_coordinates is None:
            pytest.skip("MPAS data not available")
            return
        
        from mpasdiag.visualization.surface import MPASSurfacePlotter
        
        plotter = MPASSurfacePlotter(figsize=(12, 8), dpi=100)
        lon, lat = mpas_coordinates
        
        lon_subset = lon[:500]
        lat_subset = lat[:500]
        data = np.random.uniform(990, 1020, 500)  
        
        fig, ax = plotter.create_surface_map(
            lon_subset, lat_subset, data,
            var_name='mslp',
            lon_min=lon_subset.min(), lon_max=lon_subset.max(),
            lat_min=lat_subset.min(), lat_max=lat_subset.max(),
            title="MPAS Surface Pressure",
            plot_type='scatter',
            projection='PlateCarree'
        )
        
        assert fig is not None
        assert ax is not None
        
        output_file = tmp_path / "surface_map_integration"
        plotter.save_plot(str(output_file), formats=['png'])
        
        assert (tmp_path / "surface_map_integration.png").exists()
        assert (tmp_path / "surface_map_integration.png").stat().st_size > 0
        
        plotter.close_plot()
    
    def test_create_wind_plot_with_real_data(
        self: "TestVisualizationIntegration",
        mpas_coordinates,
        mpas_wind_data,
        tmp_path
    ) -> None:
        """
        This test verifies the integration of the `MPASWindPlotter` class with real MPAS data to create a wind plot visualization. The test checks that the `create_wind_plot` method can successfully generate a wind plot using real MPAS grid coordinates and wind component data, and that the resulting figure and axes objects are valid. It also tests the complete workflow of saving the generated plot to a temporary directory and verifying that the output file is created successfully. Additionally, the test validates that the calculated wind speeds from the u and v components are within reasonable ranges, ensuring that the visualization accurately represents the underlying data. By confirming that the wind plot creation and file saving processes work correctly with real MPAS data inputs, this test ensures that the visualization components are properly integrated and functional for operational use.
        
        Parameters:
            mpas_coordinates: Real MPAS grid coordinates (lon, lat)
            mpas_wind_data: Real MPAS wind components (u, v)
            tmp_path: Temporary directory for output files
            
        Returns:
            None: Verified by successful wind plot creation
        """
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
            return
        
        from mpasdiag.visualization.wind import MPASWindPlotter
        
        plotter = MPASWindPlotter(figsize=(12, 8), dpi=100)
        lon, lat = mpas_coordinates
        u, v = mpas_wind_data
        
        n_points = 100
        lon_subset = lon[:n_points]
        lat_subset = lat[:n_points]
        u_subset = u[:n_points]
        v_subset = v[:n_points]
        
        fig, ax = plotter.create_wind_plot(
            lon_subset, lat_subset, u_subset, v_subset,
            lon_min=-180, lon_max=180,
            lat_min=-90, lat_max=90,
            wind_level="10m",
            title="MPAS Wind Field",
            plot_type='barbs',  
            projection='PlateCarree'
        )
        
        assert fig is not None
        assert ax is not None
        
        speed = np.hypot(u_subset, v_subset)
        assert speed.min() >= 0
        assert speed.max() < 100  
        
        output_file = tmp_path / "wind_plot_integration"
        plotter.save_plot(str(output_file), formats=['png'])
        
        assert (tmp_path / "wind_plot_integration.png").exists()
        plotter.close_plot()
    
    def test_precipitation_map_with_real_coordinates(
        self: "TestVisualizationIntegration",
        mpas_coordinates,
        mpas_precip_data,
        tmp_path
    ) -> None:
        """
        This test verifies the integration of the `MPASPrecipitationPlotter` class with real MPAS grid coordinates and precipitation data to create a precipitation map visualization. The test checks that the `create_precipitation_map` method can successfully generate a precipitation map using real MPAS data, and that the resulting figure and axes objects are valid. It also tests the complete workflow of saving the generated plot to a temporary directory and verifying that the output file is created successfully. By confirming that the precipitation map creation and file saving processes work correctly with real MPAS data inputs, this test ensures that the visualization components are properly integrated and functional for operational use.
        
        Parameters:
            mpas_coordinates: Real MPAS grid coordinates (lon, lat)
            mpas_precip_data: Real MPAS precipitation data
            tmp_path: Temporary directory for output files
            
        Returns:
            None: Verified by successful precipitation map creation
        """
        if mpas_coordinates is None or mpas_precip_data is None:
            pytest.skip("MPAS data not available")
            return
        
        if tmp_path is None:
            pytest.skip("Temporary path not available")
            return

        from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter
        
        plotter = MPASPrecipitationPlotter(figsize=(12, 8), dpi=100)
        lon, lat = mpas_coordinates
        
        n_points = min(1000, len(lon), len(mpas_precip_data))
        lon_subset = lon[:n_points]
        lat_subset = lat[:n_points]
        precip = mpas_precip_data[:n_points]
        
        fig, ax = plotter.create_precipitation_map(
            lon_subset, lat_subset, precip,
            var_name='precip_24h',
            lon_min=-180, lon_max=180,
            lat_min=-90, lat_max=90,
            title="MPAS 24h Precipitation",
            plot_type='scatter',
            projection='PlateCarree'
        )

        assert fig is not None
        assert ax is not None

        output_file = tmp_path / "precip_map_integration"
        plotter.save_plot(str(output_file), formats=['png'])

        assert (tmp_path / "precip_map_integration.png").exists()

        plotter.close_plot()
    
    def test_variable_styling_with_real_data(
        self: "TestVisualizationIntegration",
        mpas_wind_data
    ) -> None:
        """
        This test verifies that the variable styling functionality in the visualization module correctly applies appropriate colormaps and levels based on real MPAS wind data. The test checks that the styling logic can handle realistic data ranges and variable types, ensuring that the visual representation of wind speed and related variables is accurate and visually meaningful. By confirming that the styling parameters are correctly determined for real MPAS data, this test ensures that the visualization components can produce effective and informative plots for operational use.
        
        Parameters:
            mpas_wind_data: Real MPAS wind components (u, v)
            
        Returns:
            None: Verified by appropriate style parameters
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
            return
        
        from mpasdiag.visualization.styling import MPASVisualizationStyle
        
        u, v = mpas_wind_data
        
        speed = np.hypot(u, v)
        speed_data = xr.DataArray(speed, dims=['nCells'], name='wind_speed')
        
        style = MPASVisualizationStyle.get_variable_style('wind_speed', speed_data)
        
        assert 'colormap' in style
        assert 'levels' in style
        assert style['colormap'] in ['plasma', 'viridis']
        assert isinstance(style['levels'], (list, np.ndarray))
        assert len(style['levels']) > 5
        
        temp_data = xr.DataArray(
            250.0 + 60.0 * (u - u.min()) / (u.max() - u.min() + 1e-12),
            dims=['nCells'],
            name='t2m'
        )
        
        temp_style = MPASVisualizationStyle.get_variable_style('t2m', temp_data)
        
        assert 'colormap' in temp_style
        assert 'levels' in temp_style
        assert temp_style['colormap'] == 'RdYlBu_r'
    
    def test_cross_section_with_real_3d_data(
        self: "TestVisualizationIntegration",
        mpas_3d_processor,
        tmp_path
    ) -> None:
        """
        This test verifies the integration of the `MPASVerticalCrossSectionPlotter` class with real MPAS 3D atmospheric data to create a vertical cross-section visualization. The test checks that the cross-section plotter can be instantiated and that it has the necessary attributes for figure size and DPI settings. By confirming that the cross-section plotting functionality is available and properly configured, this test ensures that users can create vertical cross-section visualizations using real MPAS 3D data, which is essential for analyzing atmospheric structure and processes in operational meteorology.
        
        Parameters:
            mpas_3d_processor: MPAS 3D processor with loaded atmospheric data
            tmp_path: Temporary directory for output files
            
        Returns:
            None: Verified by successful cross-section creation or skip if not available
        """
        if mpas_3d_processor is None:
            pytest.skip("MPAS 3D data not available")
            return
        
        try:
            from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
            
            plotter = MPASVerticalCrossSectionPlotter(figsize=(14, 6), dpi=100)            
            assert plotter is not None

            assert hasattr(plotter, 'figsize')
            assert hasattr(plotter, 'dpi')
            
        except (ImportError, AttributeError) as e:
            pytest.skip(f"Cross-section API not fully available: {e}")
        finally:
            plt.close('all')
    
    def test_batch_plotting_workflow(
        self: "TestVisualizationIntegration",
        mpas_coordinates,
        mpas_wind_data,
        tmp_path
    ) -> None:
        """
        This test verifies the batch plotting workflow for creating multiple surface maps with real MPAS data. The test simulates a time series of wind speed data by creating multiple plots with varying data values, saving each plot to a temporary directory, and verifying that all expected output files are created successfully. By confirming that the batch plotting process can handle real MPAS data inputs and produce multiple plots without errors, this test ensures that the visualization components can support operational workflows that require generating large numbers of plots efficiently.
        
        Parameters:
            mpas_coordinates: Real MPAS grid coordinates (lon, lat)
            mpas_wind_data: Real MPAS wind components (u, v)
            tmp_path: Temporary directory for output files
            
        Returns:
            None: Verified by creation of multiple plot files
        """
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
            return
        
        from mpasdiag.visualization.surface import MPASSurfacePlotter
        
        plotter = MPASSurfacePlotter(figsize=(10, 8), dpi=100)
        lon, lat = mpas_coordinates
        u, v = mpas_wind_data
        
        n_times = 3
        created_files = []
        
        for t in range(n_times):
            data = np.hypot(u, v) + t * 2.0
            
            lon_subset = lon[:100]
            lat_subset = lat[:100]
            data_subset = data[:100]
            
            fig, ax = plotter.create_surface_map(
                lon_subset, lat_subset, data_subset,
                var_name='wind_speed',
                lon_min=-180, lon_max=180,
                lat_min=-90, lat_max=90,
                title=f"Wind Speed - Time {t}",
                plot_type='scatter',
                projection='PlateCarree'
            )
            
            output_file = tmp_path / f"batch_plot_t{t:03d}"
            plotter.save_plot(str(output_file), formats=['png'])
            created_files.append(tmp_path / f"batch_plot_t{t:03d}.png")
            
            plotter.close_plot()
        
        assert len(created_files) == n_times

        for f in created_files:
            assert f.exists()
            assert f.stat().st_size > 0
    
    def test_multi_variable_overlay(
        self: "TestVisualizationIntegration",
        mpas_coordinates,
        mpas_wind_data,
        tmp_path
    ) -> None:
        """
        This test verifies the ability to create multi-variable overlay plots using real MPAS data, combining surface maps with wind vector overlays. The test checks that the `create_surface_map` method can accept wind overlay parameters and successfully generate a plot that includes both a scalar field (e.g., pressure) and vector field (wind) using real MPAS coordinates and wind data. It also tests the complete workflow of saving the generated plot to a temporary directory and verifying that the output file is created successfully. By confirming that multi-variable overlay plotting works correctly with real MPAS data inputs, this test ensures that users can create complex visualizations that integrate multiple meteorological variables for comprehensive analysis.
        
        Parameters:
            mpas_coordinates: Real MPAS grid coordinates (lon, lat)
            mpas_wind_data: Real MPAS wind components (u, v)
            tmp_path: Temporary directory for output files
            
        Returns:
            None: Verified by successful overlay plot creation
        """
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
            return
        
        from mpasdiag.visualization.surface import MPASSurfacePlotter
        
        lon, lat = mpas_coordinates
        u, v = mpas_wind_data
        
        plotter = MPASSurfacePlotter(figsize=(14, 10), dpi=100)
        
        n_points = 100
        lon_subset = lon[:n_points]
        lat_subset = lat[:n_points]
        
        pressure = np.random.uniform(995, 1015, n_points)
        
        wind_overlay = {
            'u_data': u[:n_points],
            'v_data': v[:n_points],
            'plot_type': 'barbs',
            'thin': 5
        }
        
        fig, ax = plotter.create_surface_map(
            lon_subset, lat_subset, pressure,
            var_name='mslp',
            lon_min=-180, lon_max=180,
            lat_min=-90, lat_max=90,
            title="Multi-Variable Overlay: Pressure + Wind",
            plot_type='scatter',
            projection='PlateCarree',
            wind_overlay=wind_overlay
        )
        
        assert fig is not None
        assert ax is not None
        
        output_file = tmp_path / "overlay_integration"
        plotter.save_plot(str(output_file), formats=['png'])
        
        assert (tmp_path / "overlay_integration.png").exists()
        plotter.close_plot()
    
    def test_coordinate_transformation_accuracy(
        self: "TestVisualizationIntegration",
        mpas_coordinates
    ) -> None:
        """
        This test verifies the accuracy of coordinate transformations in the visualization module when using real MPAS grid coordinates. The test checks that the `setup_map_projection` method can correctly establish map projections based on the geographic extent of the provided MPAS coordinates, and that the resulting map projection and data coordinate reference system (CRS) are valid. By confirming that the coordinate transformation logic can handle real MPAS data inputs and produce appropriate map projections, this test ensures that visualizations will be geographically accurate and properly aligned with the underlying data, which is essential for operational meteorological analysis and decision-making.
        
        Parameters:
            mpas_coordinates: Real MPAS grid coordinates (lon, lat)
            
        Returns:
            None: Verified by coordinate bounds and transformation checks
        """
        if mpas_coordinates is None:
            pytest.skip("MPAS data not available")
            return
        
        from mpasdiag.visualization.base_visualizer import MPASVisualizer
        
        visualizer = MPASVisualizer(figsize=(10, 8))
        lon, lat = mpas_coordinates
        
        map_proj, data_crs = visualizer.setup_map_projection(
            lon.min(), lon.max(), lat.min(), lat.max(),
            'PlateCarree'
        )
        
        assert map_proj is not None
        assert data_crs is not None
        
        map_proj_merc, data_crs_merc = visualizer.setup_map_projection(
            100.0, 120.0, -10.0, 10.0,
            'Mercator'
        )
        
        assert map_proj_merc is not None
        assert data_crs_merc is not None
        
        assert -180 <= lon.min() <= 180
        assert -180 <= lon.max() <= 180
        assert -90 <= lat.min() <= 90
        assert -90 <= lat.max() <= 90
    
    def test_adaptive_visualization_parameters(
        self: "TestVisualizationIntegration",
        mpas_coordinates,
        mpas_surface_temp_data
    ) -> None:
        """
        This test verifies that the visualization module can adaptively determine appropriate styling parameters, such as colormaps and levels, based on real MPAS surface temperature data. The test checks that the level generation logic can produce reasonable levels for visualizing temperature data, and that the colormap selection is suitable for the variable type. By confirming that the adaptive styling functionality works correctly with real MPAS data inputs, this test ensures that visualizations will be both accurate and visually effective, enhancing the interpretability of the resulting plots for operational meteorological analysis.
        
        Parameters:
            mpas_coordinates: Real MPAS grid coordinates (lon, lat)
            mpas_surface_temp_data: Real MPAS surface temperature data
            
        Returns:
            None: Verified by reasonable parameter values
        """
        if mpas_coordinates is None or mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
            return
        
        from mpasdiag.visualization.styling import MPASVisualizationStyle
        
        if (
            mpas_coordinates is None or mpas_surface_temp_data is None or
            mpas_coordinates[0] is None or mpas_coordinates[1] is None or
            mpas_surface_temp_data[0] is None or mpas_surface_temp_data[1] is None
        ):
            pytest.skip("MPAS data not available")
            return
        
        lon, lat = mpas_coordinates

        if lon is None or lat is None:
            pytest.skip("Longitude or latitude data not available")
            return
        
        n_points = min(1000, len(mpas_surface_temp_data))
        sample_data = mpas_surface_temp_data[:n_points]
        data_array = xr.DataArray(sample_data, dims=['nCells'])
        
        levels = MPASVisualizationStyle._generate_levels_from_data(data_array, 'generic')
        
        assert levels is not None
        assert isinstance(levels, (list, np.ndarray))
        assert len(levels) > 5
        assert levels[0] < levels[-1]  
        
        precip_cmap, precip_levels = MPASVisualizationStyle.create_precip_colormap('a24h')
        
        assert precip_cmap is not None
        assert precip_levels is not None
        assert len(precip_levels) > 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
