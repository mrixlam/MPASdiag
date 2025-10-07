#!/usr/bin/env python3
"""
Unit tests for MPAS Analysis package

Scope:
        Integration-style unit tests covering configuration, data processing,
        plotting, CLI wiring, and utility helpers. These tests exercise a
        broad surface of the package and use mocking to keep tests fast and
        deterministic where external libraries (Cartopy, Matplotlib rendering)
        would otherwise be required.

Test data:
        - Synthetic xarray Datasets are created in fixtures for data-processing
            and visualization tests.
        - Temporary files and directories are used for configuration and file
            system tests.
        - Mocks are used to simulate datasets, plotting objects, and CLI
            argument parsing.

Expected results:
        - Core classes (MPASDataProcessor, MPASVisualizer, MPASConfig) construct
            and expose expected interfaces without raising unexpected exceptions.
        - Data extraction and variable handling functions return xarray/
            NumPy-compatible objects with expected shapes.
        - CLI functions return defined exit codes when dependencies are
            mocked.

Per-section expectations (short):
        - TestMPASConfig: basic field defaults and custom initialization.
        - TestMPASDataProcessor: dataset loading/extraction and variable access.
        - TestMPASVisualizer: plot creation functions produce fig/ax objects
            (mocked as necessary).
        - TestArgumentParser: argument parser factories expose expected flags.
        - TestCLIFunctions: CLI wiring functions handle argument/config flows
            under mocked conditions.

Usage:
        pytest tests/
        python -m pytest tests/test_mpas_analysis.py -q

Author: Rubaiat Islam
"""

import pytest
import numpy as np
import xarray as xr
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from mpas_analysis.data_processing import MPASDataProcessor
from mpas_analysis.visualization import MPASVisualizer
from mpas_analysis.utils import MPASConfig, MPASLogger, ArgumentParser
from mpas_analysis.cli import main, wind_plot_main, surface_plot_main


class TestMPASConfig:
    """
    Tests for MPASConfig configuration class.

    Scope:
        Verifies defaults and custom initialization for the configuration
        dataclass used across the CLI and processing code.
    """
    
    def test_config_initialization(self):
        """Test MPASConfig initialization with default values"""
        config = MPASConfig()
        
        assert config.grid_file == ""
        assert config.data_dir == ""
        assert config.output_dir == ""
        assert config.lat_min == -90.0
        assert config.lat_max == 90.0
        assert config.lon_min == -180.0
        assert config.lon_max == 180.0
        assert config.variable == "rainnc"
        assert config.verbose is True  
    
    def test_config_custom_values(self):
        """Test MPASConfig initialization with custom values"""
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
        assert config.lat_min == -10.0
        assert config.lat_max == 15.0
        assert config.lon_min == 91.0
        assert config.lon_max == 113.0
        assert config.verbose is True
    
    def test_wind_config_parameters(self):
        """Test wind-specific configuration parameters"""
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


class TestMPASDataProcessor:
    """
    Tests for MPASDataProcessor data handling.

    Scope:
        Exercises dataset fixtures and processor methods for variable
        extraction, time handling, and spatial coordinate retrieval.
    """
    
    @pytest.fixture
    def mock_grid_file(self):
        """Create a mock grid file for testing"""
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
            import xarray as xr
            dummy_data = xr.Dataset({'dummy': ('x', [1, 2, 3])})
            dummy_data.to_netcdf(tmp_file.name)
            return tmp_file.name
    
    @pytest.fixture
    def mock_dataset(self):
        """Create a mock xarray dataset for testing"""
        n_cells = 100
        n_times = 24
        
        lon = np.random.uniform(-180, 180, n_cells)
        lat = np.random.uniform(-90, 90, n_cells)
        time = np.arange(n_times)
        
        data_vars = {
            'lonCell': (['nCells'], lon),
            'latCell': (['nCells'], lat),
            'rainnc': (['Time', 'nCells'], np.random.uniform(0, 50, (n_times, n_cells))),
            'rainc': (['Time', 'nCells'], np.random.uniform(0, 20, (n_times, n_cells))),
            't2m': (['Time', 'nCells'], np.random.uniform(280, 320, (n_times, n_cells))),
            'u10': (['Time', 'nCells'], np.random.uniform(-20, 20, (n_times, n_cells))),
            'v10': (['Time', 'nCells'], np.random.uniform(-20, 20, (n_times, n_cells))),
            'surface_pressure': (['Time', 'nCells'], np.random.uniform(95000, 105000, (n_times, n_cells))),
        }
        
        coords = {
            'nCells': np.arange(n_cells),
            'Time': time
        }
        
        return xr.Dataset(data_vars=data_vars, coords=coords)
    
    def test_processor_initialization(self, mock_grid_file):
        """Test MPASDataProcessor initialization"""
        processor = MPASDataProcessor(mock_grid_file, verbose=True)
        
        assert processor.grid_file == mock_grid_file
        assert processor.verbose is True
        assert processor.dataset is None
    
    def test_load_data_basic_setup(self, mock_grid_file):
        """Test basic load_data method setup and validation"""
        processor = MPASDataProcessor(mock_grid_file, verbose=False)
        
        assert hasattr(processor, 'load_data')
        assert callable(getattr(processor, 'load_data'))
        
        import inspect
        sig = inspect.signature(processor.load_data)
        expected_params = ['data_dir', 'use_pure_xarray', 'reference_file']
        for param in expected_params:
            assert param in sig.parameters
            
    def test_get_available_variables(self, mock_grid_file, mock_dataset):
        """Test getting available variables from dataset"""
        processor = MPASDataProcessor(mock_grid_file, verbose=False)
        processor.dataset = mock_dataset
        
        variables = processor.get_available_variables()
        
        expected_vars = ['lonCell', 'latCell', 'rainnc', 'rainc', 't2m', 'u10', 'v10', 'surface_pressure']
        for var in expected_vars:
            assert var in variables
    
    def test_get_time_range(self, mock_grid_file, mock_dataset):
        """Test getting time range from dataset"""
        processor = MPASDataProcessor(mock_grid_file, verbose=False)
        processor.dataset = mock_dataset
        
        start_time, end_time = processor.get_time_range()
        
        assert isinstance(start_time, (datetime, np.datetime64))
        assert isinstance(end_time, (datetime, np.datetime64))
    
    def test_extract_spatial_coordinates(self, mock_grid_file, mock_dataset):
        """Test extracting spatial coordinates"""
        processor = MPASDataProcessor(mock_grid_file, verbose=False)
        processor.dataset = mock_dataset
        
        lon, lat = processor.extract_spatial_coordinates()
        
        assert isinstance(lon, np.ndarray)
        assert isinstance(lat, np.ndarray)
        assert len(lon) == len(lat)
        assert len(lon) > 0
    
    def test_get_variable_data(self, mock_grid_file, mock_dataset):
        """Test extracting variable data"""
        processor = MPASDataProcessor(mock_grid_file, verbose=False)
        processor.dataset = mock_dataset
        
        temp_data = processor.get_variable_data('t2m', time_index=0)
        
        assert isinstance(temp_data, xr.DataArray)
        assert temp_data.shape[0] > 0  
    
    def test_get_wind_components(self, mock_grid_file, mock_dataset):
        """Test extracting wind components"""
        processor = MPASDataProcessor(mock_grid_file, verbose=False)
        processor.dataset = mock_dataset
        
        u_data, v_data = processor.get_wind_components('u10', 'v10', time_index=0)
        
        assert isinstance(u_data, xr.DataArray)
        assert isinstance(v_data, xr.DataArray)
        assert u_data.shape == v_data.shape
        assert u_data.shape[0] > 0
    
    def test_get_wind_components_missing_variable(self, mock_grid_file, mock_dataset):
        """Test error handling for missing wind variables"""
        processor = MPASDataProcessor(mock_grid_file, verbose=False)
        processor.dataset = mock_dataset
        
        with pytest.raises(ValueError, match="Wind variables.*not found"):
            processor.get_wind_components('u850', 'v850', time_index=0)
    
    def test_compute_precipitation_difference(self, mock_grid_file, mock_dataset):
        """Test precipitation difference computation"""
        processor = MPASDataProcessor(mock_grid_file, verbose=False)
        processor.dataset = mock_dataset
        
        precip_data = processor.compute_precipitation_difference(0, var_name='rainnc')
        
        assert isinstance(precip_data, xr.DataArray)
        assert precip_data.shape[0] > 0
        
        if len(mock_dataset.Time) > 1:
            precip_data = processor.compute_precipitation_difference(1, var_name='rainnc')
            assert isinstance(precip_data, xr.DataArray)


class TestMPASVisualizer:
    """
    Tests for MPASVisualizer plotting functionality.

    Scope:
        Uses small synthetic datasets and mocking to validate that plotting
        functions create the expected figure/axes objects and expose
        interfaces needed by higher-level workflows.
    """
    
    @pytest.fixture
    def visualizer(self):
        """Create a visualizer instance for testing"""
        return MPASVisualizer(figsize=(10, 8), dpi=100)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for plotting tests"""
        n_points = 50
        lon = np.random.uniform(90, 115, n_points)
        lat = np.random.uniform(-15, 20, n_points)
        data = np.random.uniform(0, 100, n_points)
        u_data = np.random.uniform(-20, 20, n_points)
        v_data = np.random.uniform(-20, 20, n_points)
        
        return {
            'lon': lon,
            'lat': lat,
            'data': data,
            'u_data': u_data,
            'v_data': v_data
        }
    
    def test_visualizer_initialization(self, visualizer):
        """Test MPASVisualizer initialization"""
        assert visualizer.figsize == (10, 8)
        assert visualizer.dpi == 100
        assert visualizer.fig is None
        assert visualizer.ax is None
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.axes')
    def test_create_precipitation_map(self, mock_axes, mock_figure, visualizer, sample_data):
        """Test precipitation map creation"""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_figure.return_value = mock_fig
        mock_axes.return_value = mock_ax
        
        fig, ax = visualizer.create_precipitation_map(
            sample_data['lon'], sample_data['lat'], sample_data['data'],
            90, 115, -15, 20,
            title="Test Precipitation Map"
        )
        
        assert fig is not None
        assert ax is not None
    
    @patch('matplotlib.pyplot.figure')
    def test_create_wind_plot(self, mock_figure, visualizer, sample_data):
        """Test wind plot creation"""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_figure.return_value = mock_fig
        mock_ax.projection = Mock()
        
        with patch('matplotlib.pyplot.subplot') as mock_subplot:
            mock_subplot.return_value = mock_ax
            
            fig, ax = visualizer.create_wind_plot(
                sample_data['lon'].reshape(10, 5), 
                sample_data['lat'].reshape(10, 5),
                sample_data['u_data'].reshape(10, 5), 
                sample_data['v_data'].reshape(10, 5),
                90, 115, -15, 20,
                wind_level="surface",
                plot_type="barbs"
            )
            
            assert fig is not None
            assert ax is not None
    
    @patch('matplotlib.pyplot.figure')
    def test_create_simple_scatter_plot(self, mock_figure, visualizer, sample_data):
        """Test simple scatter plot creation"""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_figure.return_value = mock_fig
        
        with patch('matplotlib.pyplot.axes') as mock_axes:
            mock_axes.return_value = mock_ax
            
            fig, ax = visualizer.create_simple_scatter_plot(
                sample_data['lon'], sample_data['lat'], sample_data['data'],
                title="Test Scatter Plot"
            )
            
            assert fig is not None
            assert ax is not None
    
    def test_save_plot(self, visualizer):
        """Test plot saving functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_plot"
            
            visualizer.fig = Mock()
            visualizer.fig.savefig = Mock()
            
            visualizer.save_plot(output_path, formats=['png'])
            
            visualizer.fig.savefig.assert_called()


class TestArgumentParser:
    """
    Tests for ArgumentParser utility functions.

    Scope:
        Ensures parser factories build parsers with expected flags and
        that parsed arguments map correctly into configuration objects.
    """
    
    def test_create_parser(self):
        """Test creation of main argument parser"""
        parser = ArgumentParser.create_parser()
        
        assert parser is not None
        assert hasattr(parser, 'parse_args')
    
    def test_create_wind_parser(self):
        """Test creation of wind-specific argument parser"""
        parser = ArgumentParser.create_wind_parser()
        
        assert parser is not None
        assert hasattr(parser, 'parse_args')
        
        help_text = parser.format_help()
        assert "--u-variable" in help_text
        assert "--v-variable" in help_text
        assert "--wind-plot-type" in help_text
    
    def test_create_surface_parser(self):
        """Test creation of surface-specific argument parser"""
        parser = ArgumentParser.create_surface_parser()
        
        assert parser is not None
        assert hasattr(parser, 'parse_args')
        
        help_text = parser.format_help()
        assert "--variable" in help_text
        assert "--plot-type" in help_text


class TestCLIFunctions:
    """
    Tests for CLI entry point functions.

    Scope:
        Mocks parser and configuration wiring to validate top-level CLI
        entrypoints behave deterministically when dependencies are
        mocked.
    """
    
    @patch('mpas_analysis.cli.ArgumentParser.create_parser')
    @patch('mpas_analysis.cli.ArgumentParser.parse_args_to_config')
    def test_main_function(self, mock_parse_config, mock_create_parser):
        """Test main CLI function with mocked dependencies"""
        mock_parser = Mock()
        mock_args = Mock()
        mock_config = Mock()
        
        mock_create_parser.return_value = mock_parser
        mock_parser.parse_args.return_value = mock_args
        mock_parse_config.return_value = mock_config
        
        mock_args.config = None
        mock_config.grid_file = "test_grid.nc"
        mock_config.data_dir = "test_data/"
        mock_config.verbose = False
        mock_config.quiet = False
        
        result = main()
        assert result == 1  
    
    @patch('mpas_analysis.cli.ArgumentParser.create_wind_parser')
    @patch('mpas_analysis.cli.ArgumentParser.parse_wind_args_to_config')
    def test_wind_plot_main_function(self, mock_parse_config, mock_create_parser):
        """Test wind plot CLI function with mocked dependencies"""
        mock_parser = Mock()
        mock_args = Mock()
        mock_config = Mock()
        
        mock_create_parser.return_value = mock_parser
        mock_parser.parse_args.return_value = mock_args
        mock_parse_config.return_value = mock_config
        
        mock_config.grid_file = "test_grid.nc"
        mock_config.data_dir = "test_data/"
        mock_config.verbose = False
        
        result = wind_plot_main()
        assert result == 1  


class TestDataValidation:
    """
    Tests for data validation and error handling.

    Scope:
        Validates geographic extent checks and configuration boundaries.
    """
    
    def test_spatial_extent_validation(self):
        """Test spatial extent validation"""
        from mpas_analysis.data_processing import validate_geographic_extent
        
        assert validate_geographic_extent((-180, 180, -90, 90)) == True
        assert validate_geographic_extent((90, 115, -15, 20)) == True
        assert validate_geographic_extent((115, 90, -15, 20)) == False
        assert validate_geographic_extent((90, 115, 20, -15)) == False
    
    def test_config_validation(self):
        """Test configuration validation"""
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


class TestIntegration:
    """
    Integration tests for the complete workflow.

    Scope:
        Lightweight integration checks that create temporary workspace
        directories and verify end-to-end control flow at a high level.
    """
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for integration tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_full_workflow_mock(self, temp_workspace):
        """Test a complete workflow with mocked data"""
        grid_file = Path(temp_workspace) / "grid.nc"
        data_dir = Path(temp_workspace) / "data"
        data_dir.mkdir()
        
        assert grid_file.parent.exists()
        assert data_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])