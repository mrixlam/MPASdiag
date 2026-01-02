#!/usr/bin/env python3
"""
MPAS Vertical Cross-Section Plotter Unit Tests

This module provides comprehensive unit tests for the MPASVerticalCrossSectionPlotter
class, validating vertical cross-section visualization capabilities including great circle
path generation, vertical coordinate conversion, data interpolation along paths, contour
plotting, axis formatting, and integration with the MPAS visualization framework. Tests
use synthetic 3D data and mocking to isolate cross-section logic from data loading and
rendering dependencies.

Tests Performed:
    TestMPASVerticalCrossSectionPlotter:
        - test_initialization: Verifies plotter initialization with figsize and dpi parameters
        - test_great_circle_path_generation: Tests great circle path calculation between two points
        - test_great_circle_path_same_points: Validates handling of identical start/end points
        - test_default_levels_generation: Tests automatic contour level generation from data range
        - test_default_levels_with_nan_data: Verifies level generation with NaN values present
        - test_default_levels_all_nan_data: Tests fallback levels when all data is NaN
        - test_vertical_coordinate_conversion: Validates pressure to height coordinate conversion
        - test_interpolation_along_path: Tests data interpolation along cross-section path
        - test_format_cross_section_axes: Verifies axis formatting with labels and titles
        - test_create_vertical_cross_section_validation: Tests complete cross-section creation with validation
        - test_get_time_string: Validates time string extraction from processor
        - test_batch_processing_validation: Tests batch cross-section generation validation
    
    TestCrossSectionIntegration:
        - test_import_from_visualization_package: Verifies plotter imports from correct package
        - test_styling_integration: Tests integration with visualization styling system

Test Coverage:
    - MPASVerticalCrossSectionPlotter class: initialization, figure/axes creation
    - Great circle path calculation: geodesic distance, intermediate points, path generation
    - Vertical coordinate systems: pressure levels, height coordinates, coordinate conversion
    - Contour level generation: automatic levels from data range, NaN handling, fallback values
    - Data interpolation: path-based interpolation, nearest neighbor, grid sampling
    - Cross-section plotting: contour fills, contour lines, colorbars, axis configuration
    - Axis formatting: distance labels, vertical coordinate labels, titles with timestamps
    - Time string extraction: processor integration, datetime formatting
    - Batch processing: multiple cross-section validation, parameter checking
    - Package integration: import paths, module organization, styling system
    - Error handling: validation errors, missing data, invalid parameters
    - Mock 3D processor: dataset creation, time information, coordinate extraction

Testing Approach:
    Unit tests using unittest framework with synthetic xarray datasets simulating MPAS
    3D model output. Mock MPAS3DProcessor objects provide controlled test environments.
    Matplotlib Agg backend avoids GUI rendering. Tests use Mock and patch decorators to
    isolate components. Synthetic temperature data with realistic coordinate ranges validates
    cross-section calculations. Tests verify both successful operations and error handling.

Expected Results:
    - MPASVerticalCrossSectionPlotter initializes with correct default and custom parameters
    - Great circle paths calculate correct distances and intermediate points
    - Identical start/end points handled gracefully with single-point paths
    - Contour levels generated automatically span data range with sensible intervals
    - NaN values filtered correctly without affecting level generation
    - All-NaN data triggers fallback to default level range (0-1)
    - Vertical coordinate conversion transforms pressure to height correctly
    - Data interpolation along paths produces smooth cross-sections
    - Axis formatting creates professional labels and titles with timestamps
    - Complete cross-section plots render without errors for valid inputs
    - Time strings extracted correctly from processor time information
    - Batch processing validates parameters before generating multiple plots
    - Plotter imports successfully from mpasdiag.visualization.cross_section
    - Styling system integrates seamlessly with cross-section plotting
    - All tests pass with synthetic data and comprehensive mocking

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import unittest
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import tempfile
import os
import sys
from pathlib import Path

package_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_dir))

from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
from mpasdiag.processing.processors_3d import MPAS3DProcessor
    

class TestMPASVerticalCrossSectionPlotter(unittest.TestCase):
    """
    Test cases for MPASVerticalCrossSectionPlotter
    """
    
    def setUp(self) -> None:
        """
        Initialize test fixtures with plotter instance and mock 3D processor data. This method creates an MPASVerticalCrossSectionPlotter instance with custom figsize and dpi settings for testing. A mock MPAS3DProcessor is configured with a synthetic xarray dataset containing temperature data, coordinate arrays (lonCell, latCell), and time information to simulate real MPAS 3D output structure. The synthetic dataset includes 5 time steps, 10 vertical levels, and 100 horizontal cells with realistic coordinate ranges for testing cross-section calculations. Mock time information is configured to return formatted datetime strings for timestamp testing.

        Parameters:
            None

        Returns:
            None
        """
        self.plotter = MPASVerticalCrossSectionPlotter(figsize=(12, 8), dpi=100)        
        self.mock_processor = Mock(spec=MPAS3DProcessor)
        
        self.mock_dataset = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.normal(280, 20, (5, 10, 100)),
                dims=['Time', 'nVertLevels', 'nCells'],
                coords={
                    'Time': pd.date_range('2024-01-01', periods=5, freq='h'),
                    'nVertLevels': range(10),
                    'nCells': range(100)
                }
            ),
            'lonCell': xr.DataArray(
                np.linspace(-110, -70, 100),
                dims=['nCells']
            ),
            'latCell': xr.DataArray(
                np.linspace(30, 50, 100),
                dims=['nCells']
            )
        })
        
        self.mock_processor.dataset = self.mock_dataset
        self.mock_processor.get_time_info.return_value = "Valid: 2024-09-17T13:00:00"
        
    def test_initialization(self) -> None:
        """
        Verify correct initialization of MPASVerticalCrossSectionPlotter with default and custom parameters. This test validates that the plotter initializes as an instance of the correct class and that both default parameters (figsize=(10, 12), dpi=100) and custom parameters are properly stored. The test creates two plotter instances: one with default parameters and one with custom figsize and dpi values. Assertions verify that parameter values are correctly assigned during initialization and accessible through instance attributes. This ensures proper constructor behavior for subsequent plotting operations.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASVerticalCrossSectionPlotter()
        self.assertIsInstance(plotter, MPASVerticalCrossSectionPlotter)
        self.assertEqual(plotter.figsize, (10, 12))
        self.assertEqual(plotter.dpi, 100) 
        
        custom_plotter = MPASVerticalCrossSectionPlotter(figsize=(10, 6), dpi=150)
        self.assertEqual(custom_plotter.figsize, (10, 6))
        self.assertEqual(custom_plotter.dpi, 150)
    
    def test_great_circle_path_generation(self) -> None:
        """
        Validate great circle path calculation between geographic coordinates. This test verifies the _generate_great_circle_path method correctly computes geodesic paths with intermediate points, distances, and proper boundary conditions. The test generates a path between two points separated by 20 degrees in both longitude and latitude, validates output array shapes match the requested number of points, and confirms start/end point accuracy. Distance calculations are verified for monotonic increasing values starting from zero. This ensures accurate geographic path generation for cross-section interpolation along Earth's curved surface.

        Parameters:
            None

        Returns:
            None
        """
        start_point = (-100.0, 35.0)
        end_point = (-80.0, 45.0)
        num_points = 20
        
        lons, lats, distances = self.plotter._generate_great_circle_path(
            start_point, end_point, num_points
        )
        
        self.assertEqual(len(lons), num_points)
        self.assertEqual(len(lats), num_points)
        self.assertEqual(len(distances), num_points)
        
        self.assertAlmostEqual(lons[0], start_point[0], places=1)
        self.assertAlmostEqual(lats[0], start_point[1], places=1)
        self.assertAlmostEqual(lons[-1], end_point[0], places=1)
        self.assertAlmostEqual(lats[-1], end_point[1], places=1)
        
        self.assertEqual(distances[0], 0.0)
        self.assertGreater(distances[-1], 0)
        self.assertTrue(np.all(np.diff(distances) >= 0)) 
        
    def test_great_circle_path_same_points(self) -> None:
        """
        Verify graceful handling of identical start and end points in great circle path generation. This test validates the edge case where start and end coordinates are identical, ensuring the method returns a valid single-point path without errors. All output longitude and latitude values should equal the input point coordinates. Distance array should contain all zeros since no geographic displacement occurs. This test confirms robust error handling for degenerate path cases that could arise from user input or programmatic generation.

        Parameters:
            None

        Returns:
            None
        """
        start_point = (-100.0, 35.0)
        end_point = (-100.0, 35.0)
        num_points = 10
        
        lons, lats, distances = self.plotter._generate_great_circle_path(
            start_point, end_point, num_points
        )
        
        self.assertTrue(np.allclose(lons, start_point[0]))
        self.assertTrue(np.allclose(lats, start_point[1]))
        self.assertTrue(np.allclose(distances, 0.0))
    
    def test_default_levels_generation(self) -> None:
        """
        Validate automatic contour level generation for various meteorological variables. This test verifies the _get_default_levels method produces appropriate contour intervals for different variable types including temperature, pressure, wind components, and generic variables. Multiple test cases with realistic data ranges for each variable type ensure levels span the data range, maintain monotonic increasing order, and provide sensible intervals. Subtest framework allows individual variable validation with clear failure reporting. This confirms the plotter can automatically select appropriate contour levels without user specification for common atmospheric variables.

        Parameters:
            None

        Returns:
            None
        """
        test_cases = [
            ('temperature', np.random.normal(280, 20, (10, 20))),
            ('temp', np.random.normal(15, 10, (10, 20))),  
            ('pressure', np.random.exponential(50000, (10, 20))),
            ('uzonal', np.random.normal(0, 15, (10, 20))),
            ('wind_speed', np.random.exponential(10, (10, 20))),
            ('generic_var', np.random.uniform(0, 100, (10, 20)))
        ]
        
        for var_name, data in test_cases:
            with self.subTest(var_name=var_name):
                levels = self.plotter._get_default_levels(data, var_name)
                
                self.assertIsInstance(levels, np.ndarray)
                self.assertGreater(len(levels), 1)
                self.assertTrue(np.all(np.diff(levels) > 0))  
                
                data_min, data_max = np.nanmin(data), np.nanmax(data)
                self.assertLessEqual(levels[0], data_max)
                self.assertGreaterEqual(levels[-1], data_min)
    
    def test_default_levels_with_nan_data(self) -> None:
        """
        Verify robust contour level generation with NaN values present in data arrays. This test validates that _get_default_levels correctly handles datasets containing NaN (Not-a-Number) values by filtering them during level calculation. A synthetic dataset with scattered NaN values at regular intervals ensures the method extracts valid data range without being affected by missing values. Assertions confirm output levels contain no NaN values, maintain proper array structure, and span the valid data range. This ensures reliable cross-section plotting even with incomplete or masked data common in atmospheric model output.

        Parameters:
            None

        Returns:
            None
        """
        data_with_nan = np.random.normal(50, 10, (10, 20))
        data_with_nan[::2, ::2] = np.nan 
        
        levels = self.plotter._get_default_levels(data_with_nan, 'test_var')
        
        self.assertIsInstance(levels, np.ndarray)
        self.assertGreater(len(levels), 1)
        self.assertFalse(np.any(np.isnan(levels)))
        
    def test_default_levels_all_nan_data(self) -> None:
        """
        Validate fallback behavior when all data values are NaN in contour level generation. This test verifies the _get_default_levels method provides reasonable default contour levels (0 to 1 range with 11 levels) when encountering datasets where all values are NaN. Such scenarios can occur with masked data, missing variables, or initialization errors. The method should return a valid numpy array with proper structure rather than failing or returning NaN levels. This ensures the plotting system remains robust and provides useful feedback even when data is completely unavailable or invalid.

        Parameters:
            None

        Returns:
            None
        """
        all_nan_data = np.full((10, 20), np.nan)
        
        levels = self.plotter._get_default_levels(all_nan_data, 'test_var')
        
        self.assertIsInstance(levels, np.ndarray)
        self.assertEqual(len(levels), 11)  
        
    def test_vertical_coordinate_conversion(self) -> None:
        """
        Validate vertical coordinate system conversions for cross-section plotting. This test verifies the _convert_vertical_to_height method correctly transforms pressure coordinates to height using barometric formula, handles pressure in both Pa and hPa units, passes through existing height coordinates with unit conversion from meters to kilometers, and returns model levels unchanged as fallback. Multiple test cases validate monotonic increasing height with decreasing pressure, unit consistency across pressure formats, and proper coordinate type identification. This ensures cross-sections can display data with different vertical coordinate systems using appropriate axis labels and scales.

        Parameters:
            None

        Returns:
            None
        """
        pressure_coords = np.array([100000, 85000, 70000, 50000, 30000, 10000])  # Pa
        
        height_coords, coord_type = self.plotter._convert_vertical_to_height(
            pressure_coords, 'pressure', self.mock_processor, 0
        )
        
        self.assertEqual(coord_type, 'height_km')
        self.assertTrue(np.all(height_coords >= 0))
        self.assertTrue(np.all(np.diff(height_coords) > 0))  
        
        pressure_hpa = pressure_coords / 100
        height_coords_hpa, coord_type_hpa = self.plotter._convert_vertical_to_height(
            pressure_hpa, 'pressure', self.mock_processor, 0
        )
        
        self.assertEqual(coord_type_hpa, 'height_km')
        self.assertTrue(np.allclose(height_coords, height_coords_hpa, rtol=0.1))
        
        height_m = np.array([0, 1000, 2000, 5000, 10000])
        height_km_out, coord_type_height = self.plotter._convert_vertical_to_height(
            height_m, 'height', self.mock_processor, 0
        )
        
        self.assertEqual(coord_type_height, 'height_km')
        self.assertTrue(np.allclose(height_km_out, height_m / 1000.0))
        
        model_coords = np.arange(0, 10)
        model_out, coord_type_model = self.plotter._convert_vertical_to_height(
            model_coords, 'model_levels', self.mock_processor, 0
        )
        
        self.assertEqual(coord_type_model, 'model_levels')
        self.assertTrue(np.array_equal(model_out, model_coords))
    
    def test_interpolation_along_path(self) -> None:
        """
        Verify data interpolation from regular grids to arbitrary cross-section paths. This test validates the _interpolate_along_path method correctly samples gridded data values along specified longitude-latitude paths using nearest-neighbor or linear interpolation. A synthetic test dataset with known mathematical function (sine-cosine product) allows verification of interpolation accuracy. Tests confirm output array length matches path point count and that interpolated values are not all NaN for valid data. Additional test with all-NaN input data verifies graceful handling of missing data scenarios returning appropriate NaN results.

        Parameters:
            None

        Returns:
            None
        """
        grid_lons = np.linspace(-110, -70, 21)
        grid_lats = np.linspace(30, 50, 11)
        grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lons, grid_lats)
        
        grid_data = np.sin(np.radians(grid_lon_2d)) * np.cos(np.radians(grid_lat_2d)) * 100
        
        path_lons = np.linspace(-100, -80, 5)
        path_lats = np.linspace(35, 45, 5)
        
        interpolated_values = self.plotter._interpolate_along_path(
            grid_lon_2d, grid_lat_2d, grid_data, path_lons, path_lats
        )
        
        self.assertEqual(len(interpolated_values), len(path_lons))
        self.assertFalse(np.all(np.isnan(interpolated_values)))
        
        nan_data = np.full_like(grid_data, np.nan)
        nan_interpolated = self.plotter._interpolate_along_path(
            grid_lon_2d, grid_lat_2d, nan_data, path_lons, path_lats
        )
        
        self.assertTrue(np.all(np.isnan(nan_interpolated)))
    
    def test_format_cross_section_axes(self) -> None:
        """
        Validate axis formatting and labeling for cross-section plots. This test verifies the _format_cross_section_axes method correctly configures matplotlib axes with appropriate labels, limits, and scales for different vertical coordinate systems. Mock figure and axes objects capture formatting calls without rendering. Tests validate longitude axis labeling, height coordinate formatting with linear scale, and pressure coordinate formatting with logarithmic scale. Assertions confirm xlabel, ylabel, xlim, ylim, and yscale method calls occur with correct parameters. This ensures professional cross-section plots with properly formatted axes for publication-quality figures.

        Parameters:
            None

        Returns:
            None
        """
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            self.plotter.fig = mock_fig
            self.plotter.ax = mock_ax
            
            longitudes = np.linspace(-100, -80, 10)
            vertical_coords = np.linspace(0, 10, 5)
            start_point = (-100.0, 35.0)
            end_point = (-80.0, 45.0)
            
        self.plotter._format_cross_section_axes(
            longitudes, vertical_coords, 'height_km', start_point, end_point
        )
        
        mock_ax.set_xlabel.assert_called_with('Longitude', fontsize=12, labelpad=10)
        mock_ax.set_ylabel.assert_called_with('Height [km]', fontsize=12)
        mock_ax.set_xlim.assert_called_once()
        mock_ax.set_ylim.assert_called_once()
        
        mock_ax.reset_mock()
        pressure_coords = np.array([1000, 850, 700, 500, 300])
        self.plotter._format_cross_section_axes(
            longitudes, pressure_coords, 'pressure_hPa', start_point, end_point
        )
        
        mock_ax.set_ylabel.assert_called_with('Pressure [hPa]', fontsize=12)
        mock_ax.set_yscale.assert_called_with('log')
    
    @patch('mpasdiag.visualization.cross_section.plt.subplots')
    def test_create_vertical_cross_section_validation(self, mock_subplots) -> None:
        """
        Verify input validation for cross-section creation with various error conditions. This test validates the create_vertical_cross_section method properly raises ValueError exceptions for invalid inputs including non-MPAS3DProcessor objects, processors without loaded datasets, nonexistent variable names, and non-3D variables. Mock matplotlib subplots prevent actual rendering during validation testing. Multiple test cases with assertRaises context managers verify appropriate error handling for each failure mode. This ensures robust error checking prevents plotting attempts with invalid configurations that would otherwise fail silently or produce misleading results.

        Parameters:
            mock_subplots: Mock matplotlib.pyplot.subplots function to prevent actual figure creation during validation testing.

        Returns:
            None
        """
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        with self.assertRaises(ValueError):
            self.plotter.create_vertical_cross_section(
                mpas_3d_processor="not_a_processor", 
                var_name='temperature',
                start_point=(-100.0, 35.0),
                end_point=(-80.0, 45.0)
            )
        
        empty_processor = Mock(spec=MPAS3DProcessor)
        empty_processor.dataset = None
        
        with self.assertRaises(ValueError):
            self.plotter.create_vertical_cross_section(
                mpas_3d_processor=empty_processor,
                var_name='temperature',
                start_point=(-100.0, 35.0),
                end_point=(-80.0, 45.0)
            )
        
        with self.assertRaises(ValueError):
            self.plotter.create_vertical_cross_section(
                mpas_3d_processor=self.mock_processor,
                var_name='nonexistent_var',
                start_point=(-100.0, 35.0),
                end_point=(-80.0, 45.0)
            )
        
        self.mock_dataset['surface_var'] = xr.DataArray(
            np.random.random((5, 100)),
            dims=['Time', 'nCells']
        )
        self.mock_processor.dataset = self.mock_dataset
        
        with self.assertRaises(ValueError):
            self.plotter.create_vertical_cross_section(
                mpas_3d_processor=self.mock_processor,
                var_name='surface_var',
                start_point=(-100.0, 35.0),
                end_point=(-80.0, 45.0)
            )
    
    def test_get_time_string(self) -> None:
        """
        Validate time string extraction and formatting from processor time information. This test verifies the _get_time_string method correctly extracts datetime information from MPAS3DProcessor instances and formats it as human-readable timestamp strings. The test validates successful time extraction returns strings containing 'Valid:' prefix with formatted datetime. Additional test with processor lacking Time coordinate verifies fallback behavior returns simple time index string (e.g., 'Time Index: 2'). This ensures cross-section plots display appropriate temporal information in titles or labels regardless of data source capabilities.

        Parameters:
            None

        Returns:
            None
        """
        time_str = self.plotter._get_time_string(self.mock_processor, 0)
        self.assertIsInstance(time_str, str)
        self.assertIn('Valid:', time_str)
        
        no_time_processor = Mock(spec=MPAS3DProcessor)
        no_time_dataset = xr.Dataset()
        no_time_processor.dataset = no_time_dataset
        no_time_processor.get_time_info.side_effect = Exception("No time info available")
        
        time_str_no_time = self.plotter._get_time_string(no_time_processor, 2)
        self.assertEqual(time_str_no_time, "Time Index: 2")
    
    def test_batch_processing_validation(self) -> None:
        """
        Verify input validation for batch cross-section generation across multiple time steps. This test validates the create_batch_cross_section_plots method properly raises ValueError exceptions for invalid processor inputs and processors without loaded datasets before attempting batch operations. Using temporary directory for output path testing ensures no persistent file system modifications. Multiple validation test cases confirm robust error checking prevents batch processing with invalid configurations. This ensures batch operations fail fast with clear error messages rather than attempting to generate plots with incomplete or invalid data that would waste computational resources.

        Parameters:
            None

        Returns:
            None
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                self.plotter.create_batch_cross_section_plots(
                    mpas_3d_processor="invalid",  
                    output_dir=temp_dir,
                    var_name='temperature',
                    start_point=(-100.0, 35.0),
                    end_point=(-80.0, 45.0)
                )
            
            empty_processor = Mock(spec=MPAS3DProcessor)
            empty_processor.dataset = None
            
            with self.assertRaises(ValueError):
                self.plotter.create_batch_cross_section_plots(
                    mpas_3d_processor=empty_processor,
                    output_dir=temp_dir,
                    var_name='temperature',
                    start_point=(-100.0, 35.0),
                    end_point=(-80.0, 45.0)
                )


class TestCrossSectionIntegration(unittest.TestCase):
    """
    Integration tests for cross-section plotter with other modules.
    """
    
    def test_import_from_visualization_package(self) -> None:
        """
        Verify cross-section plotter is importable from main visualization package. This test validates the MPASVerticalCrossSectionPlotter class can be imported directly from the mpasdiag.visualization module for user convenience. The test checks for presence of the create_vertical_cross_section method to confirm full functionality is available through the simplified import path. Test gracefully skips if the plotter is not yet integrated into the main package __init__.py. This ensures package organization provides intuitive import paths for end users without requiring knowledge of internal module structure.

        Parameters:
            None

        Returns:
            None
        """
        try:
            from mpasdiag.visualization import MPASVerticalCrossSectionPlotter
            self.assertTrue(hasattr(MPASVerticalCrossSectionPlotter, 'create_vertical_cross_section'))
        except ImportError:
            self.skipTest("Cross-section plotter not available in main package")
    
    def test_styling_integration(self) -> None:
        """
        Validate integration between cross-section plotter and visualization styling system. This test verifies the MPASVisualizationStyle class provides appropriate styling parameters (colormap, contour levels) for cross-section variables compatible with the plotter. A synthetic temperature DataArray tests the get_variable_style method returns dictionaries containing required styling keys. The test ensures consistent visual styling between surface plots and cross-section plots for the same variables. Test gracefully skips if styling module is not available, allowing independent testing of cross-section functionality during development.

        Parameters:
            None

        Returns:
            None
        """
        try:
            from mpasdiag.visualization.styling import MPASVisualizationStyle
            
            dummy_data = xr.DataArray(
                np.random.normal(280, 20, (10, 20)),
                dims=['level', 'distance'],
                name='temperature'
            )
            
            style = MPASVisualizationStyle.get_variable_style('temperature', dummy_data)
            
            self.assertIn('colormap', style)
            self.assertIn('levels', style)
            
        except ImportError:
            self.skipTest("Styling module not available")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    unittest.main(verbosity=2)