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
import pytest
import tempfile
import numpy as np
import xarray as xr
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
from mpasdiag.processing.utils_parser import ArgumentParser


class TestDataProcessing:
    """ Comprehensive tests for MPAS2DProcessor data handling including dataset loading, variable extraction, time range processing, spatial coordinates, wind components, and precipitation calculations. """
    
    @pytest.fixture
    def mock_grid_file(self: 'TestDataProcessing') -> str:
        """
        This fixture creates a temporary NetCDF grid file on disk for testing MPAS2DProcessor initialization and dataset loading. The file is created with a .nc suffix and is empty, serving as a placeholder for the grid file path required by the processor. This allows testing of processor construction and method availability without relying on actual grid file content, enabling isolated unit tests for data processing workflows.

        Parameters:
            self: 'TestDataProcessing' - The test class instance.

        Returns:
            Generator yielding str: Path to temporary NetCDF grid file on disk.
        """
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
            tmp_file.write(b'')
            return tmp_file.name
    
    @pytest.fixture
    def mock_dataset(self: 'TestDataProcessing', 
                     mpas_surface_temp_data: np.ndarray, 
                     mpas_wind_data: tuple[np.ndarray, np.ndarray], 
                     mpas_precip_data: np.ndarray) -> xr.Dataset:
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
    
    
    def test_compute_precipitation_difference(self: 'TestDataProcessing', 
                                              mock_grid_file: str, 
                                              mock_dataset: xr.Dataset) -> None:
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
    def visualizer(self: 'TestVisualization') -> MPASVisualizer:
        """
        This fixture creates a base MPASVisualizer instance for testing general visualization functionality. The visualizer is configured with a standard figure size of 10x8 inches and a resolution of 100 DPI, providing a consistent setup for all visualization tests. This base visualizer can be used for testing common plotting features such as figure and axes creation, title setting, and basic plot structure validation across different types of meteorological visualizations. The fixture allows for isolated testing of visualization methods without relying on specific plot types, enabling comprehensive coverage of the base visualizer functionality.

        Parameters:
            None

        Returns:
            Generator yielding MPASVisualizer: Configured base visualizer instance.
        """
        return MPASVisualizer(figsize=(10, 8), dpi=100)
    
    @pytest.fixture
    def wind_plotter(self: 'TestVisualization') -> MPASWindPlotter:
        """
        This fixture creates a specialized MPASWindPlotter instance for testing wind vector field visualization functionality. The wind plotter is configured with a standard figure size of 10x8 inches and a resolution of 100 DPI, providing a consistent setup for all wind visualization tests. This specialized visualizer extends the base visualizer with methods for rendering u/v wind components and displaying vector fields. The fixture allows for isolated testing of wind-specific plotting features including barb plots, arrow plots, wind speed color mapping, and vector field subsampling.

        Parameters:
            None

        Returns:
            Generator yielding MPASWindPlotter: Configured wind-specific visualizer instance.
        """
        return MPASWindPlotter(figsize=(10, 8), dpi=100)
    
    @pytest.fixture
    def surface_plotter(self: 'TestVisualization') -> MPASSurfacePlotter:
        """
        This fixture creates a specialized MPASSurfacePlotter instance for testing surface field and scalar visualization functionality. The surface plotter is configured with a standard figure size of 10x8 inches and a resolution of 100 DPI, providing a consistent setup for all surface visualization tests. This specialized visualizer extends the base visualizer with methods for contour plots, filled contours, and scatter visualizations of meteorological surface variables. The fixture allows for isolated testing of surface-specific plotting features including geographic mapping, colormap selection, and contour level generation.

        Parameters:
            None

        Returns:
            Generator yielding MPASSurfacePlotter: Configured surface-specific visualizer instance.
        """
        return MPASSurfacePlotter(figsize=(10, 8), dpi=100)
    
    @pytest.fixture
    def sample_data(self: 'TestVisualization', 
                    mpas_wind_data: tuple[np.ndarray, np.ndarray], 
                    mpas_surface_temp_data: np.ndarray) -> dict:
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
    
    
    def test_create_precipitation_map(self: 'TestVisualization', 
                                      sample_data: dict) -> None:
        """
        This test validates the MPASPrecipitationPlotter.create_precipitation_map method to ensure it generates a precipitation map with mocked geographic axes. The test checks that the method returns non-null figure and axes objects when provided with synthetic longitude, latitude, and precipitation data. Mocking of matplotlib's figure and Cartopy's GeoAxes allows for isolated testing of the plotting logic without relying on actual rendering or geographic projections. Assertions confirm that the method can create a plot structure suitable for displaying precipitation data on a map, supporting visualization workflows for MPAS precipitation diagnostics.

        Parameters:
            sample_data (dict): Dictionary containing synthetic lon, lat, and precipitation data arrays.

        Returns:
            None
        """
        if not CARTOPY_AVAILABLE:
            pytest.skip("Cartopy not available")
            return
            
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
    
    def test_create_wind_plot(self: 'TestVisualization', 
                              wind_plotter: MPASWindPlotter, 
                              sample_data: dict) -> None:
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
            return
        
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
    
    def test_create_simple_scatter_plot(self: 'TestVisualization', 
                                        surface_plotter: MPASSurfacePlotter, 
                                        sample_data: dict) -> None:
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
    

class TestCommandLineInterface:
    """ Consolidate tests for MPAS Analysis command-line interface including parser creation, argument structure, and main function execution with mocked dependencies. """
    
    def test_create_parser(self: 'TestCommandLineInterface') -> None:
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
    
    def test_create_wind_parser(self: 'TestCommandLineInterface') -> None:
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
    
    def test_create_surface_parser(self: 'TestCommandLineInterface') -> None:
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
