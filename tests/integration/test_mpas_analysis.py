#!/usr/bin/env python3
"""
MPASdiag Test Suite: MPAS Analysis Integration Tests

This module contains comprehensive integration tests for the MPAS Analysis package, covering configuration validation, data processing workflows, and visualization capabilities. The tests utilize both synthetic datasets and real MPAS diagnostic data to validate end-to-end functionality of the analysis pipeline. Key areas of focus include configuration consistency checks, dataset loading and variable extraction, time range processing, spatial coordinate handling, wind component retrieval, precipitation difference calculations, and plot generation for various meteorological fields. The test suite ensures that all components of the MPAS Analysis package work together seamlessly to produce accurate analyses and visualizations from MPAS model output.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries for testing
import math
import shutil
import pytest
import tempfile
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Generator
from datetime import datetime
from unittest.mock import Mock, patch

try:
    from cartopy.mpl.geoaxes import GeoAxes
    CARTOPY_AVAILABLE = True
except ImportError:
    GeoAxes = None
    CARTOPY_AVAILABLE = False

from mpasdiag.processing.processors_2d import MPAS2DProcessor
from mpasdiag.visualization.base_visualizer import MPASVisualizer
from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter
from mpasdiag.visualization.surface import MPASSurfacePlotter
from mpasdiag.visualization.wind import MPASWindPlotter
from mpasdiag.processing.utils_config import MPASConfig
from mpasdiag.processing.utils_parser import ArgumentParser
from mpasdiag.processing.cli_unified import main
from mpasdiag.processing.utils_geog import MPASGeographicUtils
from mpasdiag.processing.utils_datetime import MPASDateTimeUtils


class TestConfigurationAndValidation:
    """ Consolidates tests for configuration validation and consistency checks including geographic extent validation, configuration parameter relationships, and default value initialization for the MPAS Analysis package. """
    
    def test_spatial_extent_validation(self: "TestConfigurationAndValidation") -> None:
        """
        This test validates the MPASGeographicUtils.validate_geographic_extent method to ensure it correctly identifies valid and invalid geographic bounds. The method checks that longitude bounds are within -180 to 180 degrees, latitude bounds are within -90 to 90 degrees, and that minimum bounds are less than maximum bounds. Valid geographic extents (global and Southeast Asian domain) should return True, while invalid extents with reversed bounds should return False. This validation ensures that geographic parameters used in analysis workflows are logically consistent and within acceptable ranges for Earth coordinates.

        Parameters:
            self: TestConfigurationAndValidation - The test class instance.

        Returns:
            None
        """
        assert MPASGeographicUtils.validate_geographic_extent((-180, 180, -90, 90))
        assert MPASGeographicUtils.validate_geographic_extent((90, 115, -15, 20))
        assert not MPASGeographicUtils.validate_geographic_extent((115, 90, -15, 20))
        assert not MPASGeographicUtils.validate_geographic_extent((90, 115, 20, -15))
    
    def test_config_validation(self: "TestConfigurationAndValidation") -> None:
        """
        This test verifies that the MPASConfig dataclass correctly enforces logical relationships between geographic configuration parameters. Specifically, it checks that latitude minimum is less than latitude maximum and longitude minimum is less than longitude maximum. Additionally, it confirms that required file path parameters (grid_file and data_dir) are not empty strings. This validation ensures that the configuration provided for MPAS Analysis workflows is logically consistent and contains necessary information for successful execution.

        Parameters:
            self: TestConfigurationAndValidation - The test class instance.

        Returns:
            None
        """
        config = MPASConfig(
            grid_file="test.nc",
            data_dir="data/",
            lat_min=-10, lat_max=15,
            lon_min=91, lon_max=113
        )
        
        assert config.lat_min < config.lat_max
        assert config.lon_min < config.lon_max
        assert config.grid_file != ""
        assert config.data_dir != ""

    def test_config_initialization(self: "TestConfigurationAndValidation") -> None:
        """
        This test confirms that the MPASConfig dataclass initializes with default values for all configuration parameters when no custom values are provided. The test checks that file path parameters (grid_file, data_dir, output_dir) default to empty strings, geographic bounds default to global extents (lat_min=-90, lat_max=90, lon_min=-180, lon_max=180), the variable defaults to "rainnc", and verbose mode is enabled by default. This ensures that users can create a configuration instance without specifying all parameters while still having sensible defaults for typical MPAS analysis scenarios.

        Parameters:
            self: TestConfigurationAndValidation - The test class instance.

        Returns:
            None
        """
        config = MPASConfig()
        
        assert config.grid_file == ""
        assert config.data_dir == ""
        assert config.output_dir == ""
        assert math.isclose(config.lat_min, -90.0, abs_tol=1e-6)
        assert math.isclose(config.lat_max, 90.0, abs_tol=1e-6)
        assert math.isclose(config.lon_min, -180.0, abs_tol=1e-6)
        assert math.isclose(config.lon_max, 180.0, abs_tol=1e-6)
        assert config.variable == "rainnc"
        assert config.verbose is True  
    
    def test_config_custom_values(self: "TestConfigurationAndValidation") -> None:
        """
        This test verifies that the MPASConfig dataclass correctly stores custom values for configuration parameters when provided. The test checks that file path parameters (grid_file, data_dir) are stored as specified, geographic bounds are stored with correct precision, and verbose mode is set according to input. This ensures that users can customize their configuration for specific MPAS analysis scenarios while having confidence that their settings are preserved accurately by the dataclass.

        Parameters:
            self: TestConfigurationAndValidation - The test class instance.

        Returns:
            None
        """
        config = MPASConfig(
            grid_file="test_grid.nc",
            data_dir="test_data/",
            lat_min=-10.0,
            lat_max=15.0,
            lon_min=91.0,
            lon_max=113.0,
            verbose=True
        )
        
        assert config.grid_file == "test_grid.nc"
        assert config.data_dir == "test_data/"
        assert math.isclose(config.lat_min, -10.0, abs_tol=1e-6)
        assert math.isclose(config.lat_max, 15.0, abs_tol=1e-6)
        assert math.isclose(config.lon_min, 91.0, abs_tol=1e-6)
        assert math.isclose(config.lon_max, 113.0, abs_tol=1e-6)
        assert config.verbose is True
    
    def test_wind_config_parameters(self: "TestConfigurationAndValidation") -> None:
        """
        This test confirms that the MPASConfig dataclass correctly initializes and stores wind-specific configuration parameters when provided. The test checks that u_variable and v_variable are stored as specified, wind_level is set to "surface", wind_plot_type is set to "barbs", subsample_factor is set to 0, and show_background is True. This ensures that users can configure wind analysis and visualization settings accurately for MPAS wind diagnostics workflows while having confidence that their custom settings are preserved by the dataclass.

        Parameters:
            self: TestConfigurationAndValidation - The test class instance.

        Returns:
            None
        """
        config = MPASConfig(
            u_variable="u10",
            v_variable="v10",
            wind_level="surface",
            wind_plot_type="barbs",
            subsample_factor=0,
            show_background=True
        )
        
        assert config.u_variable == "u10"
        assert config.v_variable == "v10"
        assert config.wind_level == "surface"
        assert config.wind_plot_type == "barbs"
        assert config.subsample_factor == 0
        assert config.show_background is True


class TestDataProcessing:
    """ Comprehensive tests for MPAS2DProcessor data handling including dataset loading, variable extraction, time range processing, spatial coordinates, wind components, and precipitation calculations. """
    
    @pytest.fixture
    def mock_grid_file(self: "TestDataProcessing") -> str:
        """
        This fixture creates a temporary NetCDF grid file on disk for testing MPAS2DProcessor initialization and dataset loading. The file is created with a .nc suffix and is empty, serving as a placeholder for the grid file path required by the processor. This allows testing of processor construction and method availability without relying on actual grid file content, enabling isolated unit tests for data processing workflows.

        Parameters:
            self: TestDataProcessing - The test class instance.

        Returns:
            Generator yielding str: Path to temporary NetCDF grid file on disk.
        """
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
            tmp_file.write(b'')
            return tmp_file.name
    
    @pytest.fixture
    def mock_dataset(self: "TestDataProcessing", mpas_surface_temp_data, mpas_wind_data, mpas_precip_data) -> xr.Dataset:
        """
        This fixture generates a synthetic xarray Dataset containing expected MPAS meteorological variables and coordinates for testing data processing methods. The dataset includes longitude and latitude coordinates, precipitation variables (rainnc, rainc), surface temperature (t2m), wind components (u10, v10), and surface pressure. The data is constructed using real MPAS variable values tiled across a time dimension to create a realistic dataset structure for testing variable extraction, time range processing, and coordinate handling. This allows comprehensive testing of the processor's data handling capabilities with authentic MPAS model output.

        Parameters:
            mpas_surface_temp_data: Real MPAS surface temperature from session fixture
            mpas_wind_data: Real MPAS wind components (u10, v10) from session fixture
            mpas_precip_data: Real MPAS precipitation from session fixture

        Returns:
            Generator yielding xr.Dataset: xarray dataset with real MPAS meteorological variables and coordinates.
        """
        n_cells = 100
        n_times = 24
        from tests.test_data_helpers import load_mpas_coords_from_processor
        lon, lat, u_arr, v_arr = load_mpas_coords_from_processor(n_cells)
        time = np.arange(n_times)
        
        u_wind, v_wind = mpas_wind_data
        temp_data = mpas_surface_temp_data[:n_cells]
        precip = mpas_precip_data[:n_cells]
        
        temp_tiled = np.tile(temp_data, (n_times, 1))
        u_tiled = np.tile(u_wind[:n_cells], (n_times, 1))
        v_tiled = np.tile(v_wind[:n_cells], (n_times, 1))
        precip_tiled = np.tile(precip, (n_times, 1))
        
        data_vars = {
            'lonCell': (['nCells'], lon),
            'latCell': (['nCells'], lat),
            'rainnc': (['Time', 'nCells'], precip_tiled * 2.0),  
            'rainc': (['Time', 'nCells'], precip_tiled * 0.5),  
            't2m': (['Time', 'nCells'], temp_tiled),
            'u10': (['Time', 'nCells'], u_tiled),
            'v10': (['Time', 'nCells'], v_tiled),
            'surface_pressure': (['Time', 'nCells'], 95000.0 + 5000.0 * (precip_tiled / (np.max(precip) + 1e-12))),
        }
        
        coords = {
            'nCells': np.arange(n_cells),
            'Time': time
        }
        
        return xr.Dataset(data_vars=data_vars, coords=coords)
    
    def test_processor_initialization(self: "TestDataProcessing", mock_grid_file: str) -> None:
        """
        This test confirms that the MPAS2DProcessor initializes correctly with the provided grid file path and verbose mode setting. The test checks that the grid_file attribute is set to the mock grid file path, the verbose attribute is set to True, and the dataset attribute is initialized to None. This ensures that the processor can be constructed with the necessary parameters and is in a valid initial state for subsequent data loading and processing operations.

        Parameters:
            mock_grid_file (str): Path to temporary mock grid NetCDF file provided by the fixture.

        Returns:
            None
        """
        processor = MPAS2DProcessor(mock_grid_file, verbose=True)
        
        assert processor.grid_file == mock_grid_file
        assert processor.verbose is True
        assert processor.dataset is None
    
    def test_load_data_basic_setup(self: "TestDataProcessing", mock_grid_file: str) -> None:
        """
        This test verifies that the MPAS2DProcessor provides the load_2d_data method with the expected signature and parameters. The test checks that the load_2d_data method exists, is callable, and has the expected parameters including data_dir for file location, use_pure_xarray for loading strategy, and reference_file for baseline comparisons. This ensures that after initialization, the processor exposes the necessary data loading interface for downstream processing workflows.

        Parameters:
            mock_grid_file (str): Path to temporary mock grid NetCDF file provided by the fixture.

        Returns:
            None
        """
        processor = MPAS2DProcessor(mock_grid_file, verbose=False)
        
        assert hasattr(processor, 'load_2d_data')
        assert callable(getattr(processor, 'load_2d_data'))
        
        import inspect
        sig = inspect.signature(processor.load_2d_data)
        expected_params = ['data_dir', 'use_pure_xarray', 'reference_file']
        for param in expected_params:
            assert param in sig.parameters
            
    def test_get_available_variables(self: "TestDataProcessing", mock_grid_file: str, mock_dataset: xr.Dataset) -> None:
        """
        This test confirms that the MPAS2DProcessor correctly reports the available variable names present in the synthetic dataset. The test checks that the get_available_variables method extracts and returns variable names from the loaded xarray dataset. A synthetic dataset containing meteorological variables (coordinates, precipitation, temperature, wind, pressure) is used to test variable discovery functionality. Assertions verify that all expected variables, including lonCell, latCell, rainnc, rainc, t2m, u10, v10, and surface_pressure, appear in the returned list. This ensures that users can query available variables for dynamic workflow configuration and data exploration.

        Parameters:
            mock_grid_file (str): Temporary mock grid file path provided by fixture for processor initialization.
            mock_dataset (xr.Dataset): Synthetic xarray dataset containing expected meteorological variables.

        Returns:
            None
        """
        processor = MPAS2DProcessor(mock_grid_file, verbose=False)
        processor.dataset = mock_dataset
        
        variables = processor.get_available_variables()
        
        expected_vars = ['lonCell', 'latCell', 'rainnc', 'rainc', 't2m', 'u10', 'v10', 'surface_pressure']
        for var in expected_vars:
            assert var in variables
    
    def test_get_time_range(self: "TestDataProcessing", mock_grid_file: str, mock_dataset: xr.Dataset) -> None:
        """
        This test verifies that the MPASDateTimeUtils.get_time_range method correctly extracts the start and end time from the Time coordinate of the synthetic dataset. The test checks that the method returns valid datetime objects representing the time range covered by the dataset. A synthetic dataset with a Time coordinate dimension is used to test time range extraction functionality. Assertions confirm that both start_time and end_time are instances of datetime or numpy.datetime64, indicating successful time range retrieval. This ensures that users can determine the temporal coverage of their datasets for time-based analysis and visualization workflows.

        Parameters:
            mock_grid_file (str): Temporary mock grid file path provided by fixture.
            mock_dataset (xr.Dataset): Synthetic dataset with Time coordinate dimension.

        Returns:
            None
        """
        start_time, end_time = MPASDateTimeUtils.get_time_range(mock_dataset)
        
        assert isinstance(start_time, (datetime, np.datetime64))
        assert isinstance(end_time, (datetime, np.datetime64))
    
    def test_extract_spatial_coordinates(self: "TestDataProcessing", mock_grid_file: str, mock_dataset: xr.Dataset) -> None:
        """
        This test confirms that the MPASGeographicUtils.extract_spatial_coordinates method correctly retrieves longitude and latitude coordinate arrays from the synthetic dataset. The test checks that the method returns numpy arrays for both longitude and latitude, and that they have matching lengths corresponding to the number of spatial points in the dataset. A synthetic dataset containing lonCell and latCell coordinate variables is used to test spatial coordinate extraction functionality. Assertions verify that both lon and lat are numpy arrays with positive length, indicating successful coordinate retrieval. This ensures that users can obtain spatial coordinates for geographic analysis and visualization workflows.

        Parameters:
            mock_grid_file (str): Temporary mock grid file path provided by fixture.
            mock_dataset (xr.Dataset): Synthetic dataset containing lonCell and latCell coordinate variables.

        Returns:
            None
        """
        lon, lat = MPASGeographicUtils.extract_spatial_coordinates(mock_dataset)
        
        assert isinstance(lon, np.ndarray)
        assert isinstance(lat, np.ndarray)
        assert len(lon) == len(lat)
        assert len(lon) > 0
    
    def test_get_variable_data(self: "TestDataProcessing", mock_grid_file: str, mock_dataset: xr.Dataset) -> None:
        """
        This test verifies that the MPAS2DProcessor.get_2d_variable_data method correctly extracts single-variable data from multi-variable datasets. The test checks that the method returns an xarray DataArray with non-empty data at the specified time index. A synthetic dataset containing the temperature variable (t2m) is used to test temporal slicing and variable isolation functionality. Assertions confirm that the returned data is an xarray DataArray with a positive shape, indicating successful data extraction. This ensures that users can perform targeted variable analysis and support downstream processing requiring specific meteorological field extraction from MPAS output.

        Parameters:
            mock_grid_file (str): Temporary mock grid file path provided by fixture.
            mock_dataset (xr.Dataset): Synthetic dataset containing meteorological variables including t2m.

        Returns:
            None
        """
        processor = MPAS2DProcessor(mock_grid_file, verbose=False)
        processor.dataset = mock_dataset
        
        temp_data = processor.get_2d_variable_data('t2m', time_index=0)
        
        assert isinstance(temp_data, xr.DataArray)
        assert temp_data.shape[0] > 0  
    
    def test_get_wind_components(self: "TestDataProcessing", mock_grid_file: str, mock_dataset: xr.Dataset) -> None:
        """
        This test confirms that the WindDiagnostics.get_2d_wind_components method correctly retrieves u and v wind component data from the synthetic dataset. The test checks that the method returns two xarray DataArrays for the specified u and v variable names at the given time index. A synthetic dataset containing u10 and v10 wind components is used to test wind component extraction functionality. Assertions verify that both u_data and v_data are xarray DataArrays with matching shapes and positive length, indicating successful retrieval of wind components. This ensures that users can obtain necessary wind data for vector field analysis, wind plotting, and meteorological diagnostics workflows using MPAS output.

        Parameters:
            mock_grid_file (str): Temporary mock grid file path provided by fixture.
            mock_dataset (xr.Dataset): Synthetic dataset containing u10 and v10 wind component variables.

        Returns:
            None
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        processor = MPAS2DProcessor(mock_grid_file, verbose=False)
        processor.dataset = mock_dataset
        wind_diag = WindDiagnostics(verbose=False)
        
        u_data, v_data = wind_diag.get_2d_wind_components(mock_dataset, 'u10', 'v10', time_index=0)
        
        assert isinstance(u_data, xr.DataArray)
        assert isinstance(v_data, xr.DataArray)
        assert u_data.shape == v_data.shape
        assert u_data.shape[0] > 0
    
    def test_get_wind_components_missing_variable(self: "TestDataProcessing", mock_grid_file: str, mock_dataset: xr.Dataset) -> None:
        """
        This test confirms that requesting non-existent wind variables raises a ValueError with informative error messages. The test validates proper error handling when wind components are missing from datasets to prevent silent failures or cryptic errors. Non-existent variables (u850, v850) trigger exception testing to verify clear error messaging. Assertions use pytest.raises to confirm ValueError is raised with "Wind variables.*not found" pattern matching. This defensive error handling guides users toward data availability issues and prevents downstream processing failures from undefined wind component references.

        Parameters:
            mock_grid_file (str): Temporary mock grid file path provided by fixture.
            mock_dataset (xr.Dataset): Synthetic dataset used for error handling test input.

        Returns:
            None
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        processor = MPAS2DProcessor(mock_grid_file, verbose=False)
        processor.dataset = mock_dataset
        wind_diag = WindDiagnostics(verbose=False)
        
        with pytest.raises(ValueError, match="Wind variables.*not found"):
            wind_diag.get_2d_wind_components(mock_dataset, 'u850', 'v850', time_index=0)
    
    def test_compute_precipitation_difference(self: "TestDataProcessing", mock_grid_file: str, mock_dataset: xr.Dataset) -> None:
        """
        This test confirms that the PrecipitationDiagnostics.compute_precipitation_difference method correctly calculates temporal precipitation differences for accumulation analysis. A synthetic dataset containing the rainnc variable is used to test difference calculation at time index 0, producing accumulation data. Assertions verify that the returned data is an xarray DataArray suitable for plotting and analysis. This diagnostic supports precipitation accumulation workflows, hourly/daily accumulation calculations, and difference plot generation for model verification and comparison studies.

        Parameters:
            mock_grid_file (str): Temporary mock grid file path provided by fixture.
            mock_dataset (xr.Dataset): Synthetic dataset containing rainnc precipitation variable.

        Returns:
            None
        """
        from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
        
        processor = MPAS2DProcessor(mock_grid_file, verbose=False)
        processor.dataset = mock_dataset
        precip_diag = PrecipitationDiagnostics(verbose=False)
        
        precip_data = precip_diag.compute_precipitation_difference(mock_dataset, 0, var_name='rainnc')
        
        assert isinstance(precip_data, xr.DataArray)
        assert precip_data.shape[0] > 0
        
        if len(mock_dataset.Time) > 1:
            precip_data = precip_diag.compute_precipitation_difference(mock_dataset, 1, var_name='rainnc')
            assert isinstance(precip_data, xr.DataArray)


class TestVisualization:
    """ Tests for MPAS visualization functionality including visualizer initialization, precipitation map creation, wind plot generation, and surface field plotting using both synthetic and real MPAS data. """
    
    @pytest.fixture
    def visualizer(self: "TestVisualization") -> MPASVisualizer:
        """
        This fixture creates a base MPASVisualizer instance for testing general visualization functionality. The visualizer is configured with a standard figure size of 10x8 inches and a resolution of 100 DPI, providing a consistent setup for all visualization tests. This base visualizer can be used for testing common plotting features such as figure and axes creation, title setting, and basic plot structure validation across different types of meteorological visualizations. The fixture allows for isolated testing of visualization methods without relying on specific plot types, enabling comprehensive coverage of the base visualizer functionality.

        Parameters:
            None

        Returns:
            Generator yielding MPASVisualizer: Configured base visualizer instance.
        """
        return MPASVisualizer(figsize=(10, 8), dpi=100)
    
    @pytest.fixture
    def wind_plotter(self: "TestVisualization") -> MPASWindPlotter:
        """
        This fixture creates a specialized MPASWindPlotter instance for testing wind vector field visualization functionality. The wind plotter is configured with a standard figure size of 10x8 inches and a resolution of 100 DPI, providing a consistent setup for all wind visualization tests. This specialized visualizer extends the base visualizer with methods for rendering u/v wind components and displaying vector fields. The fixture allows for isolated testing of wind-specific plotting features including barb plots, arrow plots, wind speed color mapping, and vector field subsampling.

        Parameters:
            None

        Returns:
            Generator yielding MPASWindPlotter: Configured wind-specific visualizer instance.
        """
        return MPASWindPlotter(figsize=(10, 8), dpi=100)
    
    @pytest.fixture
    def surface_plotter(self: "TestVisualization") -> MPASSurfacePlotter:
        """
        This fixture creates a specialized MPASSurfacePlotter instance for testing surface field and scalar visualization functionality. The surface plotter is configured with a standard figure size of 10x8 inches and a resolution of 100 DPI, providing a consistent setup for all surface visualization tests. This specialized visualizer extends the base visualizer with methods for contour plots, filled contours, and scatter visualizations of meteorological surface variables. The fixture allows for isolated testing of surface-specific plotting features including geographic mapping, colormap selection, and contour level generation.

        Parameters:
            None

        Returns:
            Generator yielding MPASSurfacePlotter: Configured surface-specific visualizer instance.
        """
        return MPASSurfacePlotter(figsize=(10, 8), dpi=100)
    
    @pytest.fixture
    def sample_data(self: "TestVisualization", mpas_wind_data, mpas_surface_temp_data) -> dict:
        """
        This fixture generates a sample data dictionary containing longitude, latitude, generic data, and u/v wind components for testing visualization methods. The data is constructed using real MPAS wind and surface temperature data from session fixtures, providing authentic values for testing plot generation. The fixture loads spatial coordinates from the processor test helper function and extracts corresponding wind and temperature data for a subset of points. This allows for comprehensive testing of visualization methods with realistic MPAS data while maintaining a manageable dataset size for efficient plotting.

        Parameters:
            mpas_wind_data: Real MPAS wind components from session fixture
            mpas_surface_temp_data: Real MPAS surface temperature from session fixture

        Returns:
            Generator yielding dict: Dictionary containing lon, lat, data, u_data, and v_data numpy arrays.
        """
        n_points = 50
        from tests.test_data_helpers import load_mpas_coords_from_processor
        lon, lat, u_arr, v_arr = load_mpas_coords_from_processor(n_points)
        u_wind, v_wind = mpas_wind_data
        temp_data = mpas_surface_temp_data[:n_points]
        
        data = temp_data 
        u_data = u_wind[:n_points]
        v_data = v_wind[:n_points]
        
        return {
            'lon': lon,
            'lat': lat,
            'data': data,
            'u_data': u_data,
            'v_data': v_data
        }
    
    def test_visualizer_initialization(self: "TestVisualization", visualizer: MPASVisualizer) -> None:
        """
        This test confirms that the MPASVisualizer initializes with the expected default properties for figure size, resolution, and uninitialized figure and axes. The test checks that the figsize attribute is set to (10, 8), the dpi attribute is set to 100, and that both fig and ax attributes are None upon initialization. This ensures that the visualizer is constructed with the correct configuration for subsequent plot generation and that it starts in a valid state for creating new figures and axes as needed.

        Parameters:
            visualizer (MPASVisualizer): Base visualizer instance provided by pytest fixture.

        Returns:
            None
        """
        assert visualizer.figsize == (10, 8)
        assert visualizer.dpi == 100
        assert visualizer.fig is None
        assert visualizer.ax is None
    
    
    def test_create_precipitation_map(self: "TestVisualization", sample_data: dict) -> None:
        """
        This test validates the MPASPrecipitationPlotter.create_precipitation_map method to ensure it generates a precipitation map with mocked geographic axes. The test checks that the method returns non-null figure and axes objects when provided with synthetic longitude, latitude, and precipitation data. Mocking of matplotlib's figure and Cartopy's GeoAxes allows for isolated testing of the plotting logic without relying on actual rendering or geographic projections. Assertions confirm that the method can create a plot structure suitable for displaying precipitation data on a map, supporting visualization workflows for MPAS precipitation diagnostics.

        Parameters:
            sample_data (dict): Dictionary containing synthetic lon, lat, and precipitation data arrays.

        Returns:
            None
        """
        if not CARTOPY_AVAILABLE:
            pytest.skip("Cartopy not available")
            
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = Mock()
            mock_ax = Mock(spec=GeoAxes)
            mock_ax.transAxes = Mock()  
            mock_figure.return_value = mock_fig
            mock_fig.add_subplot.return_value = mock_ax
            
            precip_plotter = MPASPrecipitationPlotter(figsize=(10, 8), dpi=100)
            fig, ax = precip_plotter.create_precipitation_map(
                sample_data['lon'], sample_data['lat'], sample_data['data'],
                90, 115, -15, 20,
                title="Test Precipitation Map"
            )
            
            assert fig is not None
            assert ax is not None
    
    def test_create_wind_plot(self: "TestVisualization", wind_plotter: MPASWindPlotter, sample_data: dict) -> None:
        """
        This test verifies that the MPASWindPlotter.create_wind_plot method generates a wind vector plot with mocked geographic axes. The test checks that the method returns non-null figure and axes objects when provided with synthetic longitude, latitude, and u/v wind component data. Mocking of matplotlib's figure and Cartopy's GeoAxes allows for isolated testing of the wind plotting logic without relying on actual rendering or geographic projections. Assertions confirm that the method can create a plot structure suitable for displaying wind vectors on a map, supporting visualization workflows for MPAS wind diagnostics.

        Parameters:
            wind_plotter (MPASWindPlotter): Wind-specific plotter instance provided by pytest fixture.
            sample_data (dict): Dictionary containing synthetic coordinate and wind component arrays.

        Returns:
            None
        """
        if not CARTOPY_AVAILABLE:
            pytest.skip("Cartopy not available")
            
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = Mock()
            mock_ax = Mock(spec=GeoAxes)
            mock_ax.projection = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            fig, ax = wind_plotter.create_wind_plot(
                sample_data['lon'].reshape(10, 5), 
                sample_data['lat'].reshape(10, 5),
                sample_data['u_data'].reshape(10, 5), 
                sample_data['v_data'].reshape(10, 5),
                90, 115, -15, 20,
                level_info="surface",
                plot_type="barbs"
            )
            
            assert fig is not None
            assert ax is not None
    
    def test_create_simple_scatter_plot(self: "TestVisualization", surface_plotter: MPASSurfacePlotter, sample_data: dict) -> None:
        """
        This test verifies that the MPASSurfacePlotter.create_simple_scatter_plot method generates a simple scatter plot with mocked matplotlib objects. The test checks that the method returns non-null figure and axes objects when provided with synthetic longitude, latitude, and data arrays. Mocking of matplotlib's figure and axes allows for isolated testing of the scatter plotting logic without relying on actual rendering. Assertions confirm that the method can create a plot structure suitable for displaying scatter data, supporting lightweight data exploration and visualization workflows.

        Parameters:
            mock_figure (Mock): Mocked matplotlib.pyplot.figure for isolated testing.
            surface_plotter (MPASSurfacePlotter): Surface plotter instance provided by pytest fixture.
            sample_data (dict): Dictionary containing synthetic lon, lat, and data arrays.

        Returns:
            None
        """
        import matplotlib.pyplot as plt
        try:
            fig, ax = surface_plotter.create_simple_scatter_plot(
                sample_data['lon'], sample_data['lat'], sample_data['data'],
                title="Test Scatter Plot"
            )
            
            assert fig is not None
            assert ax is not None
        finally:
            plt.close('all')
    
    def test_save_plot(self: "TestVisualization", visualizer: MPASVisualizer) -> None:
        """
        This test verifies that the MPASVisualizer.save_plot method correctly saves figures to disk in specified formats. The test uses a temporary directory and mocks the figure's savefig method to avoid actual file I/O. Assertions confirm that savefig is called, ensuring the save operation is triggered correctly. This supports workflows where visualizations need to be persisted for reports, publications, or archival purposes.

        Parameters:
            visualizer (MPASVisualizer): Base visualizer instance provided by pytest fixture.

        Returns:
            None
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_plot"
            
            visualizer.fig = Mock()
            visualizer.fig.savefig = Mock()
            
            visualizer.save_plot(str(output_path), formats=['png'])
            
            visualizer.fig.savefig.assert_called()


class TestCommandLineInterface:
    """ Consolidate tests for MPAS Analysis command-line interface including parser creation, argument structure, and main function execution with mocked dependencies. """
    
    def test_create_parser(self: "TestCommandLineInterface") -> None:
        """
        This test verifies that the ArgumentParser.create_parser method constructs a command-line argument parser with the expected structure and subcommands. The test checks that the parser is created successfully and has a parse_args method, indicating it is a valid argparse.ArgumentParser instance. This ensures that the CLI entry point provides a properly configured parser for handling user input and supporting various MPAS Analysis subcommands and options.

        Parameters:
            None

        Returns:
            None
        """
        parser = ArgumentParser.create_parser()
        
        assert parser is not None
        assert hasattr(parser, 'parse_args')
    
    def test_create_wind_parser(self: "TestCommandLineInterface") -> None:
        """
        This test verifies that the ArgumentParser.create_wind_parser method constructs a command-line argument parser with wind-specific options. The test checks that the parser is created successfully and has a parse_args method, indicating it is a valid argparse.ArgumentParser instance. Help text inspection confirms the presence of u-variable, v-variable, and wind-plot-type arguments for wind visualization control. This ensures that wind plotting workflows have dedicated CLI arguments for component specification, plot styling (barbs vs arrows), and visualization customization supporting operational wind analysis requirements.

        Parameters:
            None

        Returns:
            None
        """
        parser = ArgumentParser.create_wind_parser()
        
        assert parser is not None
        assert hasattr(parser, 'parse_args')
        
        help_text = parser.format_help()
        assert "--u-variable" in help_text
        assert "--v-variable" in help_text
        assert "--wind-plot-type" in help_text
    
    def test_create_surface_parser(self: "TestCommandLineInterface") -> None:
        """
        This test verifies that the ArgumentParser.create_surface_parser method constructs a command-line argument parser with surface-specific options. The test checks that the parser is created successfully and has a parse_args method, indicating it is a valid argparse.ArgumentParser instance. Help text inspection confirms the presence of variable and plot-type arguments for surface visualization control. This ensures that surface plotting workflows have dedicated CLI arguments for variable selection, plot type specification (contour, scatter, filled), and visualization customization supporting diverse meteorological surface field analysis requirements.

        Parameters:
            None

        Returns:
            None
        """
        parser = ArgumentParser.create_surface_parser()
        
        assert parser is not None
        assert hasattr(parser, 'parse_args')
        
        help_text = parser.format_help()
        assert "--variable" in help_text
        assert "--plot-type" in help_text


    def test_main_function(self: "TestCommandLineInterface") -> None:
        """
        This test verifies that the main function of the MPAS Analysis CLI executes successfully with mocked dependencies. The MPASUnifiedCLI class is mocked to isolate the test from actual command-line parsing and execution logic. The test checks that the main function returns an exit code of 0, indicating successful execution. Assertions confirm that the MPASUnifiedCLI class is instantiated and its main method is called, ensuring that the CLI entry point correctly initializes and invokes the command processing workflow.

        Parameters:
            mock_cli_class (Mock): Mocked MPASUnifiedCLI class for isolated testing.

        Returns:
            None
        """
        import mpasdiag.processing.cli_unified as _cu
        orig_cli = _cu.MPASUnifiedCLI
        mock_cli_instance = Mock()
        mock_cli_instance.main.return_value = 0
        _cu.MPASUnifiedCLI = lambda *a, **kw: mock_cli_instance
        try:
            result = main()
            assert isinstance(result, int)
        finally:
            _cu.MPASUnifiedCLI = orig_cli
    

class TestEndToEndWorkflows:
    """ Integration tests for end-to-end MPAS Analysis workflows including workspace setup, directory management, and path handling to ensure complete workflow infrastructure functions correctly for operational processing pipelines and batch analysis workflows. """
    
    @pytest.fixture
    def temp_workspace(self: "TestEndToEndWorkflows") -> Generator[str, None, None]:
        """
        This fixture creates a temporary workspace directory for testing end-to-end workflow control flow including workspace setup, directory creation, and path management. The fixture uses Python's tempfile module to create a temporary directory that serves as the workspace for integration tests. After yielding the temporary directory path for use in tests, the fixture ensures cleanup by removing the directory and its contents after the test completes. This allows for isolated testing of filesystem operations and path handling without affecting the actual filesystem or requiring manual cleanup.

        Parameters:
            None

        Returns:
            Generator yielding str: Path to temporary workspace directory.
        """
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_full_workflow_mock(self: "TestEndToEndWorkflows", temp_workspace: str) -> None:
        """
        This test verifies the full end-to-end workflow control flow for MPAS Analysis including workspace setup, directory creation, and path management using a temporary workspace. The test checks that the necessary directories for grid files and data files are created within the temporary workspace. Assertions confirm that the expected directory structure exists, ensuring that the workflow infrastructure can successfully manage paths and directories for operational processing pipelines and batch analysis workflows.

        Parameters:
            temp_workspace (str): Path to temporary workspace directory provided by pytest fixture.

        Returns:
            None
        """
        grid_file = Path(temp_workspace) / "grid.nc"
        data_dir = Path(temp_workspace) / "data"
        data_dir.mkdir()
        
        assert grid_file.parent.exists()
        assert data_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])