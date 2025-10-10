#!/usr/bin/env python3

"""
Additional tests for recent MPAS Analysis enhancements.

This module tests new functionality added for:
1. MPASFileMetadata class with metadata management functions
2. Conditional valid time display in surface plots  
3. Timestamp-based filename generation
4. Placeholder 3D functions for future development
5. Unit conversion integration with metadata system

These tests complement the existing test suite to ensure new functionality
works correctly and maintains backward compatibility.
"""

import unittest
import tempfile
import os
import warnings
from datetime import datetime
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

from mpas_analysis.visualization import (
    MPASFileMetadata,
    MPASVisualizer,
    MPASSurfacePlotter
)
from mpas_analysis.data_processing import MPASDataProcessor


class TestRefactoredFunctions(unittest.TestCase):
    """Test 2D/3D function refactoring and backward compatibility."""
    
    def test_get_2d_variable_metadata(self):
        """Test the new get_2d_variable_metadata function."""
        metadata = MPASFileMetadata.get_2d_variable_metadata('t2m')
        
        self.assertIsInstance(metadata, dict)
        self.assertIn('units', metadata)
        self.assertIn('long_name', metadata)
        self.assertIn('colormap', metadata)
        self.assertEqual(metadata['units'], '°C')
        self.assertEqual(metadata['original_units'], 'K')
        self.assertIn('°C', metadata['long_name'])  
    
    def test_3d_placeholder_functions(self):
        """Test that 3D placeholder functions raise NotImplementedError."""
        with self.assertRaises(NotImplementedError) as cm:
            MPASFileMetadata.get_3d_variable_metadata('temperature', level=500)
        self.assertIn("3D variable support not yet implemented", str(cm.exception))
        
        with self.assertRaises(NotImplementedError) as cm:
            MPASFileMetadata.get_3d_colormap_and_levels('temperature', level=500)
        self.assertIn("3D variable support not yet implemented", str(cm.exception))
        
        with self.assertRaises(NotImplementedError) as cm:
            import xarray as xr
            dummy_data = xr.DataArray(np.random.rand(10, 10))
            dummy_lon = np.random.rand(10)
            dummy_lat = np.random.rand(10)
            MPASFileMetadata.plot_3d_variable_slice(dummy_data, dummy_lon, dummy_lat, 500, 'temperature')
        self.assertIn("3D variable support not yet implemented", str(cm.exception))


class TestConditionalTimeDisplay(unittest.TestCase):
    """Test conditional valid time display functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.surface_plotter = MPASSurfacePlotter(figsize=(10, 8), dpi=150)
        self.test_time = datetime(2024, 9, 17, 3, 0, 0)
        self.lon = np.random.uniform(91, 113, 100)
        self.lat = np.random.uniform(-10, 12, 100) 
        self.data = np.random.uniform(280, 310, 100)  
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self.surface_plotter, 'fig') and self.surface_plotter.fig:
            plt.close(self.surface_plotter.fig)
    
    @patch('cartopy.crs.PlateCarree')
    @patch('matplotlib.pyplot.figure')
    def test_default_title_with_time(self, mock_figure, mock_crs):
        """Test that default title includes time and no corner text is added."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        text_calls = []
        def capture_text(*args, **kwargs):
            text_calls.append((args, kwargs))
        mock_ax.text = capture_text
        
        fig, ax = self.surface_plotter.create_surface_map(
            lon=self.lon, lat=self.lat, data=self.data,
            var_name='t2m',
            lon_min=91.0, lon_max=113.0, lat_min=-10.0, lat_max=12.0,
            title=None,
            time_stamp=self.test_time
        )
        
        title_calls = [call for call in mock_ax.set_title.call_args_list]
        self.assertTrue(len(title_calls) > 0)
        title_text = title_calls[0][0][0]  
        self.assertIn("Valid Time: 20240917T03", title_text)
        
        corner_text_calls = [call for call in text_calls 
                           if len(call[0]) >= 3 and call[0][1] == 0.98 and call[0][2] == 0.02]
        self.assertEqual(len(corner_text_calls), 0, "Corner text should not be displayed when time is in title")
    
    @patch('cartopy.crs.PlateCarree') 
    @patch('matplotlib.pyplot.figure')
    def test_custom_title_without_time(self, mock_figure, mock_crs):
        """Test that custom title without time shows corner text."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        text_calls = []
        def capture_text(*args, **kwargs):
            text_calls.append((args, kwargs))
        mock_ax.text = capture_text
        
        fig, ax = self.surface_plotter.create_surface_map(
            lon=self.lon, lat=self.lat, data=self.data,
            var_name='t2m',
            lon_min=91.0, lon_max=113.0, lat_min=-10.0, lat_max=12.0,
            title="Custom Temperature Analysis",
            time_stamp=self.test_time
        )
        
        corner_text_calls = [call for call in text_calls 
                           if len(call[0]) >= 3 and 'Valid: 20240917T03' in str(call[0])]
        self.assertTrue(len(corner_text_calls) > 0, "Corner text should be displayed when time is not in title")
    
    @patch('cartopy.crs.PlateCarree')
    @patch('matplotlib.pyplot.figure') 
    def test_custom_title_with_time(self, mock_figure, mock_crs):
        """Test that custom title with time doesn't show corner text."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        text_calls = []
        def capture_text(*args, **kwargs):
            text_calls.append((args, kwargs))
        mock_ax.text = capture_text
        
        fig, ax = self.surface_plotter.create_surface_map(
            lon=self.lon, lat=self.lat, data=self.data,
            var_name='t2m',
            lon_min=91.0, lon_max=113.0, lat_min=-10.0, lat_max=12.0,
            title="Temperature Analysis - Valid: 20240917T03",
            time_stamp=self.test_time
        )
        
        corner_text_calls = [call for call in text_calls 
                           if len(call[0]) >= 3 and 'Valid: 20240917T03' in str(call[0])]
        self.assertEqual(len(corner_text_calls), 0, "Corner text should not be displayed when time is already in title")
    
    @patch('cartopy.crs.PlateCarree')
    @patch('matplotlib.pyplot.figure')
    def test_no_timestamp(self, mock_figure, mock_crs):
        """Test that no time display appears when timestamp is None."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        text_calls = []
        def capture_text(*args, **kwargs):
            text_calls.append((args, kwargs))
        mock_ax.text = capture_text
        
        fig, ax = self.surface_plotter.create_surface_map(
            lon=self.lon, lat=self.lat, data=self.data,
            var_name='t2m',
            lon_min=91.0, lon_max=113.0, lat_min=-10.0, lat_max=12.0,
            title=None,
            time_stamp=None
        )
        
        title_calls = [call for call in mock_ax.set_title.call_args_list]
        self.assertTrue(len(title_calls) > 0)
        title_text = title_calls[0][0][0]
        self.assertNotIn("Valid Time:", title_text)
        
        corner_text_calls = [call for call in text_calls 
                           if len(call[0]) >= 3 and 'Valid:' in str(call[0])]
        self.assertEqual(len(corner_text_calls), 0, "No time display should appear when timestamp is None")


if __name__ == '__main__':
    unittest.main()