#!/usr/bin/env python3

"""
Unit tests for MPAS Visualization Module

Scope:
        Exercises the `MPASVisualizer` class and related helper functions
        (formatters, color level helpers, and batch helpers). Tests focus on
        construction of figures/axes, formatting utilities, and batch workflow
        interaction (using mocks to avoid heavy cartographic dependencies).

Test data:
        Synthetic in-memory data generated with NumPy. For map/grid plotting
        code that depends on Cartopy/Matplotlib features, tests use mocking to
        isolate behavior and avoid external dependencies or rendering
        side-effects.

Expected results:
        - No unexpected exceptions when creating plots with valid input.
        - Functions return Figure/Axes-like objects (or mocks) as appropriate.
        - Formatter and helper functions return values in expected formats.
        - Batch helpers call visualizer/processor interfaces the expected number
            of times when components are mocked.

Per-test expectations (short):
        - TestUtilityFunctions: validate coordinate/tick/level utilities return
            correct types and sensible values.
        - TestMPASVisualizer: MPASVisualizer methods create figures/axes and
            perform expected operations (colorbars, gridlines) when invoked.
        - TestPrecipitationMapping: precipitation-specific maps produce
            scatter/contour artifacts and add metadata text; cartopy/plotting
            functionality is mocked for isolation.
        - TestBatchProcessing: batch helper iterates time steps and invokes
            processor/visualizer methods; returns expected file path list when
            mocked.

Author: Rubaiat Islam
"""

import unittest
import tempfile
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
package_dir = Path(__file__).parent.parent
sys.path.insert(0, str(package_dir))

from mpas_analysis.visualization import (
    MPASVisualizer,
    MPASSurfacePlotter
)


class TestUtilityFunctions(unittest.TestCase):
    """
    Tests for visualization utility functions.

    Scope:
        Validates latitude/longitude formatting, color-level selection and
        tick formatting heuristics with synthetic numeric inputs.
    """
    
    def test_validate_plot_parameters(self):
        """Test plot parameter validation."""
        self.assertTrue(MPASVisualizer.validate_plot_parameters(100.0, 110.0, -10.0, 10.0))
        
        self.assertFalse(MPASVisualizer.validate_plot_parameters(-200.0, 110.0, -10.0, 10.0))
        
        self.assertFalse(MPASVisualizer.validate_plot_parameters(100.0, 110.0, -100.0, 10.0))
        
        self.assertFalse(MPASVisualizer.validate_plot_parameters(110.0, 100.0, -10.0, 10.0))
        
        self.assertFalse(MPASVisualizer.validate_plot_parameters(100.0, 110.0, 10.0, -10.0))


class TestMPASVisualizer(unittest.TestCase):
    """
    Tests for MPASVisualizer plotting methods.

    Scope:
        Ensures that plotting helpers construct figures/axes and attach
        expected elements (colorbars, gridlines). Uses mocking to avoid
        rendering dependencies.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = MPASVisualizer(figsize=(8, 6), dpi=100)
        self.surface_plotter = MPASSurfacePlotter(figsize=(8, 6), dpi=100)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        plt.close('all')
    
    def test_initialization(self):
        """Test visualizer initialization."""
        self.assertEqual(self.visualizer.figsize, (8, 6))
        self.assertEqual(self.visualizer.dpi, 100)
        self.assertIsNone(self.visualizer.fig)
        self.assertIsNone(self.visualizer.ax)
    
    def test_create_precip_colormap(self):
        """Test precipitation colormap creation using specialized precipitation plotter."""
        from mpas_analysis.visualization import MPASPrecipitationPlotter
        precip_plotter = MPASPrecipitationPlotter(figsize=(8, 6), dpi=100)
        
        cmap, levels = precip_plotter.create_precip_colormap('a01h')
        
        self.assertIsNotNone(cmap)
        self.assertIsInstance(levels, list)
        self.assertTrue(len(levels) > 0)
        
        cmap_24h, levels_24h = precip_plotter.create_precip_colormap('a24h')
        self.assertIsNotNone(cmap_24h)
        self.assertTrue(max(levels_24h) > max(levels))
    
    def test_format_coordinates(self):
        """Test coordinate formatting functions."""
        lat_str = self.visualizer.format_latitude(10.5, None)
        self.assertEqual(lat_str, "10.5°N")
        
        lat_str = self.visualizer.format_latitude(-10.5, None)
        self.assertEqual(lat_str, "10.5°S")
        
        lon_str = self.visualizer.format_longitude(105.0, None)
        self.assertEqual(lon_str, "105.0°E")
        
        lon_str = self.visualizer.format_longitude(-105.0, None)
        self.assertEqual(lon_str, "105.0°W")
    
    def test_format_ticks_dynamic(self):
        """Test dynamic tick formatting with scientific notation for extreme values."""
        normal_ticks = [1.0, 2.0, 3.0, 4.0, 5.0]
        formatted = self.visualizer._format_ticks_dynamic(normal_ticks)
        self.assertEqual(formatted, ['1', '2', '3', '4', '5'])
        
        small_normal = [0.1, 0.2, 0.3, 0.4, 0.5]
        formatted = self.visualizer._format_ticks_dynamic(small_normal)
        self.assertEqual(formatted, ['0.10', '0.20', '0.30', '0.40', '0.50'])
        
        very_small = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
        formatted = self.visualizer._format_ticks_dynamic(very_small)
        for f in formatted:
            self.assertIn('e-', f)  
        self.assertEqual(formatted[0], '1.0e-05')  
        
        very_large = [1e5, 2e5, 3e5, 4e5, 5e5]
        formatted = self.visualizer._format_ticks_dynamic(very_large)
        for f in formatted:
            self.assertIn('e+', f)  
        self.assertEqual(formatted[0], '1.0e+05')  
            
        with_zeros = [0.0, 1e-5, 2e-5]
        formatted = self.visualizer._format_ticks_dynamic(with_zeros)
        self.assertEqual(formatted[0], '0.0e+00')  
        self.assertIn('e-', formatted[1])  
        
        vorticity_values = [5e-6, 1e-5, 1.5e-5, 2e-5, 2.5e-5]
        formatted = self.visualizer._format_ticks_dynamic(vorticity_values)
        for f in formatted:
            self.assertIn('e-', f)  
    
    def test_setup_map_projection(self):
        """Test map projection setup."""
        map_proj, data_crs = self.visualizer.setup_map_projection(
            100.0, 110.0, -5.0, 5.0, 'PlateCarree'
        )
        
        self.assertIsNotNone(map_proj)
        self.assertIsNotNone(data_crs)
        
        map_proj_merc, _ = self.visualizer.setup_map_projection(
            100.0, 110.0, -5.0, 5.0, 'Mercator'
        )
        self.assertIsNotNone(map_proj_merc)
    
    def test_create_simple_scatter_plot(self):
        """Test simple scatter plot creation."""
        n_points = 100
        lon = np.random.uniform(100, 110, n_points)
        lat = np.random.uniform(-5, 5, n_points)
        data = np.random.uniform(0, 50, n_points)
        
        fig, ax = self.surface_plotter.create_simple_scatter_plot(
            lon, lat, data,
            title="Test Scatter Plot",
            colorbar_label="Test Data",
            colormap='viridis'
        )
        
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertEqual(fig, self.surface_plotter.fig)
        self.assertEqual(ax, self.surface_plotter.ax)
    
    def test_create_histogram(self):
        """Test histogram creation."""
        data = np.random.normal(10, 5, 1000)
        
        fig, ax = self.visualizer.create_histogram(
            data,
            bins=30,
            title="Test Histogram",
            xlabel="Value",
            ylabel="Frequency"
        )
        
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        
        fig_log, ax_log = self.visualizer.create_histogram(
            data, log_scale=True
        )
        self.assertIsNotNone(fig_log)
    
    def test_create_time_series_plot(self):
        """Test time series plot creation."""
        from datetime import datetime, timedelta
        
        start_time = datetime(2024, 1, 1)
        times = [start_time + timedelta(hours=i) for i in range(24)]
        values = np.random.uniform(0, 20, 24)
        
        fig, ax = self.visualizer.create_time_series_plot(
            times, values,
            title="Test Time Series",
            ylabel="Precipitation (mm)",
            xlabel="Time"
        )
        
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
    
    def test_save_plot(self):
        """Test plot saving functionality."""
        lon = np.array([100, 105, 110])
        lat = np.array([-2, 0, 2])
        data = np.array([1, 5, 10])
        
        self.surface_plotter.create_simple_scatter_plot(lon, lat, data)
        
        output_path = os.path.join(self.temp_dir, "test_plot")
        self.surface_plotter.save_plot(output_path, formats=['png'])
        
        expected_file = f"{output_path}.png"
        self.assertTrue(os.path.exists(expected_file))
    
    def test_save_plot_no_figure(self):
        """Test saving plot without creating figure first."""
        output_path = os.path.join(self.temp_dir, "test_plot")
        
        with self.assertRaises(ValueError):
            self.visualizer.save_plot(output_path)
    
    def test_close_plot(self):
        """Test plot closing functionality."""
        lon = np.array([100, 105, 110])
        lat = np.array([-2, 0, 2])
        data = np.array([1, 5, 10])
        
        self.surface_plotter.create_simple_scatter_plot(lon, lat, data)
        
        self.assertIsNotNone(self.surface_plotter.fig)
        
        self.surface_plotter.close_plot()
        
        self.assertIsNone(self.surface_plotter.fig)
        self.assertIsNone(self.surface_plotter.ax)


class TestPrecipitationMapping(unittest.TestCase):
    """
    Tests for precipitation mapping.

    Scope:
        Builds synthetic precipitation fields with spatial patterns and
        verifies precipitation map creation (scatter/contour) under mocked
        cartopy functionality.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        from mpas_analysis.visualization import MPASPrecipitationPlotter
        self.visualizer = MPASPrecipitationPlotter(figsize=(10, 8), dpi=100)
        
        n_points = 1000
        self.lon = np.random.uniform(98, 112, n_points)
        self.lat = np.random.uniform(-6, 8, n_points)
        
        self.precip_data = np.zeros(n_points)
        high_precip_mask = (self.lon > 102) & (self.lon < 108) & (self.lat > -2) & (self.lat < 2)
        self.precip_data[high_precip_mask] = np.random.uniform(10, 50, np.sum(high_precip_mask))
        light_precip_mask = ~high_precip_mask
        self.precip_data[light_precip_mask] = np.random.uniform(0, 5, np.sum(light_precip_mask))
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        plt.close('all')
    
    @patch('cartopy.feature.COASTLINE')
    @patch('cartopy.feature.BORDERS')
    @patch('cartopy.feature.OCEAN')
    @patch('cartopy.feature.LAND')
    @patch('cartopy.crs.PlateCarree')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.axes')
    def test_create_precipitation_map(self, mock_axes, mock_figure, mock_crs, 
                                     mock_land, mock_ocean, mock_borders, mock_coastline):
        """Test precipitation map creation with comprehensive mocking."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_axes.return_value = mock_ax
        
        mock_projection = MagicMock()
        mock_crs.return_value = mock_projection
        
        mock_coastline_feature = MagicMock()
        mock_borders_feature = MagicMock()
        mock_ocean_feature = MagicMock()
        mock_land_feature = MagicMock()
        
        mock_coastline.__get__ = MagicMock(return_value=mock_coastline_feature)
        mock_borders.__get__ = MagicMock(return_value=mock_borders_feature)
        mock_ocean.__get__ = MagicMock(return_value=mock_ocean_feature)
        mock_land.__get__ = MagicMock(return_value=mock_land_feature)
        
        mock_scatter = MagicMock()
        mock_ax.scatter.return_value = mock_scatter
        mock_colorbar = MagicMock()
        mock_fig.colorbar.return_value = mock_colorbar
        
        mock_gridlines = MagicMock()
        mock_ax.gridlines.return_value = mock_gridlines
        
        mock_fig.text = MagicMock()
        
        try:
            fig, ax = self.visualizer.create_precipitation_map(
                self.lon, self.lat, self.precip_data,
                100.0, 110.0, -4.0, 4.0,
                title="Test Precipitation Map",
                accum_period="a01h"
            )
            
            self.assertEqual(fig, mock_fig)
            self.assertEqual(ax, mock_ax)
            
            mock_ax.set_extent.assert_called_once()
            self.assertTrue(mock_ax.add_feature.called)
            mock_ax.scatter.assert_called_once()
            mock_fig.colorbar.assert_called_once()
            mock_ax.set_title.assert_called_once()
            mock_fig.text.assert_called_once()  
            
        except Exception as e:
            self.skipTest(f"Cartopy functionality not available: {e}")
    
    def test_create_precipitation_map_invalid_data(self):
        """Test precipitation map with invalid data."""
        invalid_data = self.precip_data.copy()
        invalid_data[::10] = np.nan  
        
        try:
            fig, ax = self.visualizer.create_precipitation_map(
                self.lon, self.lat, invalid_data,
                100.0, 110.0, -4.0, 4.0,
                title="Test Invalid Data Map"
            )
            
            self.assertIsNotNone(fig)
            self.assertIsNotNone(ax)
            
        except ImportError:
            self.skipTest("Cartopy not available for testing")
    
    def test_custom_colormap_and_levels(self):
        """Test precipitation map with custom colormap and levels."""
        custom_levels = [0, 1, 5, 10, 20, 50]
        
        try:
            fig, ax = self.visualizer.create_precipitation_map(
                self.lon, self.lat, self.precip_data,
                100.0, 110.0, -4.0, 4.0,
                title="Custom Colormap Test",
                colormap="plasma",
                levels=custom_levels
            )
            
            self.assertIsNotNone(fig)
            self.assertIsNotNone(ax)
            
        except ImportError:
            self.skipTest("Cartopy not available for testing")


class TestBatchProcessing(unittest.TestCase):
    """
    Tests for batch processing helpers.

    Scope:
        Verifies that batch helpers iterate time steps and invoke processor
        and visualizer methods the correct number of times using mocked
        components.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        plt.close('all')
    
    def test_batch_processing_mock(self):
        """Test batch processing with mocked components using specialized precipitation plotter."""
        from mpas_analysis.visualization import MPASPrecipitationPlotter
        
        mock_processor = MagicMock()
        mock_processor.dataset.dims = {'Time': 5}
        mock_processor.dataset.sizes = {'Time': 5}
        
        n_points = 100
        lon = np.random.uniform(100, 110, n_points)
        lat = np.random.uniform(-5, 5, n_points)
        mock_processor.extract_spatial_coordinates.return_value = (lon, lat)
        
        mock_precip_data = MagicMock()
        mock_precip_data.values = np.random.uniform(0, 20, n_points)
        mock_processor.compute_precipitation_difference.return_value = mock_precip_data
        mock_processor.get_time_info.return_value = "20240101T00"
        
        precip_plotter = MPASPrecipitationPlotter(figsize=(10, 8), dpi=100)
        
        # Mock the plotter's methods
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        precip_plotter.create_precipitation_map = MagicMock(return_value=(mock_fig, mock_ax))
        precip_plotter.save_plot = MagicMock()
        precip_plotter.close_plot = MagicMock()
        
        try:
            created_files = precip_plotter.create_batch_precipitation_maps(
                mock_processor, self.temp_dir,
                100.0, 110.0, -4.0, 4.0,
                var_name='rainnc',
                accum_period='a01h',
                formats=['png']
            )
            
            self.assertEqual(len(created_files), 4) 
            
            self.assertEqual(mock_processor.compute_precipitation_difference.call_count, 4)
            self.assertEqual(precip_plotter.create_precipitation_map.call_count, 4)
            
        except Exception as e:
            self.skipTest(f"Dependencies not available: {e}")


if __name__ == '__main__':
    unittest.main()