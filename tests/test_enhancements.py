#!/usr/bin/env python3
"""
MPAS Analysis Enhancements Test Suite

This module provides comprehensive unit tests for recent enhancements to the MPAS
Analysis package, including metadata management, conditional time display in plots,
timestamp-based filename generation, and placeholder 3D functions. These tests validate
new functionality while ensuring backward compatibility with existing features. Tests
use synthetic data and mocking to isolate enhancement logic from rendering dependencies.

Tests Performed:
    TestRefactoredFunctions:
        - test_get_2d_variable_metadata: Validates 2D variable metadata retrieval with units and colormap
        - test_3d_placeholder_functions: Verifies NotImplementedError for future 3D functionality
    
    TestConditionalTimeDisplay:
        - test_default_title_with_time: Tests default title generation with time stamp integration
        - test_custom_title_without_time: Validates corner text display when custom title excludes time
        - test_custom_title_with_time: Tests time detection in custom titles to avoid duplication
        - test_no_timestamp: Verifies proper handling when no timestamp is provided

Test Coverage:
    - MPASFileMetadata class: metadata management, variable metadata extraction
    - 2D variable metadata: units, long_name, colormap, original_units retrieval
    - 3D placeholder functions: NotImplementedError for get_3d_variable_metadata, get_3d_colormap_and_levels, plot_3d_variable_slice
    - Conditional time display: automatic title/corner text placement logic
    - MPASSurfacePlotter: surface map creation with time stamp handling
    - Title generation: default titles with embedded time, custom titles with time detection
    - Corner text display: conditional display based on title content, timestamp presence
    - Time formatting: datetime conversion to YYYYMMDDTHH format
    - Unit conversion integration: metadata system compatibility
    - Backward compatibility: existing functionality preservation with new features

Testing Approach:
    Unit tests using unittest framework with synthetic NumPy arrays and datetime objects.
    Mocking isolates Matplotlib figure/axes and Cartopy GeoAxes to avoid rendering overhead.
    Tests capture method calls (set_title, text) to verify conditional time display logic.
    Cartopy availability check with conditional test skipping. Tests validate both positive
    cases (expected behavior) and negative cases (no duplicate time display).

Expected Results:
    - MPASFileMetadata.get_2d_variable_metadata returns complete metadata dictionaries
    - Metadata includes units, long_name, colormap, and original_units fields
    - Unit conversion properly reflected (K to °C for temperature variables)
    - 3D placeholder functions raise NotImplementedError with descriptive messages
    - Default titles automatically include "Valid Time: YYYYMMDDTHH" when timestamp provided
    - Custom titles without time trigger corner text display with timestamp
    - Custom titles with embedded time suppress duplicate corner text
    - No time display appears when timestamp is None
    - All tests pass with proper mocking and synthetic data

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
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

from mpasdiag.processing.utils_metadata import MPASFileMetadata
from mpasdiag.visualization.base_visualizer import MPASVisualizer
from mpasdiag.visualization.surface import MPASSurfacePlotter

try:
    from cartopy.mpl.geoaxes import GeoAxes
    CARTOPY_AVAILABLE = True
except ImportError:
    GeoAxes = None
    CARTOPY_AVAILABLE = False
from mpasdiag.processing.processors_2d import MPAS2DProcessor


class TestRefactoredFunctions(unittest.TestCase):
    """
    Test 2D/3D function refactoring and backward compatibility.
    """
    
    def test_get_2d_variable_metadata(self) -> None:
        """
        Verify accurate retrieval of 2D variable metadata including units, names, and colormap specifications. This test validates the MPASFileMetadata system correctly extracts and formats metadata for two-dimensional meteorological variables with automatic unit conversions. Temperature variable (t2m) metadata extraction is tested to ensure proper unit conversion from Kelvin to Celsius and appropriate long_name updates. Assertions confirm the metadata dictionary contains required fields (units, long_name, colormap, original_units) with correctly converted values. This functionality ensures plotting routines receive complete variable information for proper visualization labeling and colormap selection.

        Parameters:
            None

        Returns:
            None
        """
        metadata = MPASFileMetadata.get_2d_variable_metadata('t2m')
        
        self.assertIsInstance(metadata, dict)
        self.assertIn('units', metadata)
        self.assertIn('long_name', metadata)
        self.assertIn('colormap', metadata)
        self.assertEqual(metadata['units'], '°C')
        self.assertEqual(metadata['original_units'], 'K')
        self.assertIn('°C', metadata['long_name'])  
    
    def test_3d_placeholder_functions(self) -> None:
        """
        Verify proper NotImplementedError exceptions for placeholder 3D variable processing functions. This test validates that future 3D functionality methods raise appropriate errors with descriptive messages when called prematurely. Three placeholder functions (get_3d_variable_metadata, get_3d_colormap_and_levels, plot_3d_variable_slice) are tested with synthetic data to confirm NotImplementedError responses. Assertions verify each function raises the expected exception with informative "3D variable support not yet implemented" messages. This ensures users receive clear feedback when attempting to use unimplemented 3D features, preventing silent failures or undefined behavior.

        Parameters:
            None

        Returns:
            None
        """
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
    """
    Test conditional valid time display functionality.
    """
    
    def setUp(self) -> None:
        """
        Initialize test fixtures for conditional time display validation with synthetic meteorological data. This method sets up the surface plotting system, test datetime, and random geographical coordinates representing a Southeast Asian domain. Synthetic temperature data (280-310K range, 100 cells) with coordinates spanning 91-113°E and -10 to 12°N simulates realistic tropical region data for time display testing. All test methods in this class use these common fixtures to ensure consistent validation of timestamp integration logic. Cartopy availability is checked with automatic test skipping if the library is unavailable.

        Parameters:
            None

        Returns:
            None
        """
        if not CARTOPY_AVAILABLE:
            self.skipTest("Cartopy not available")
            
        self.surface_plotter = MPASSurfacePlotter(figsize=(10, 8), dpi=150)
        self.test_time = datetime(2024, 9, 17, 3, 0, 0)
        self.lon = np.random.uniform(91, 113, 100)
        self.lat = np.random.uniform(-10, 12, 100) 
        self.data = np.random.uniform(280, 310, 100)  
    
    def tearDown(self) -> None:
        """
        Clean up matplotlib figure resources after conditional time display tests. This method closes the surface plotter figure object created during test execution to prevent memory leaks and resource exhaustion in the test suite. Proper cleanup is essential when running multiple visualization tests that create matplotlib figure objects. Called automatically by the unittest framework after each test method completes. This ensures each test starts with a clean matplotlib state and prevents test interactions through shared figure references.

        Parameters:
            None

        Returns:
            None
        """
        if hasattr(self.surface_plotter, 'fig') and self.surface_plotter.fig:
            plt.close(self.surface_plotter.fig)
    
    def test_default_title_with_time(self) -> None:
        """
        Verify automatic time stamp integration in default plot titles without corner text duplication. This test validates that surface plots with default titles (title=None) automatically embed the valid time in the title text using "Valid Time: YYYYMMDDTHH" format. Mock matplotlib and Cartopy objects capture set_title and text method calls to verify timestamp placement. Assertions confirm the title contains the formatted timestamp (20240917T03) and no duplicate corner text is displayed. This ensures default plots display temporal information prominently in titles, supporting operational workflows where time identification is critical for forecast verification and model output analysis.

        Parameters:
            None

        Returns:
            None
        """
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = MagicMock()
            mock_ax = MagicMock(spec=GeoAxes)
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
    
    def test_custom_title_without_time(self) -> None:
        """
        Verify corner text timestamp display when custom titles omit temporal information. This test validates the plotting system automatically adds corner text with timestamp when users provide custom titles lacking time references. A custom title "Custom Temperature Analysis" without temporal keywords triggers corner text display with "Valid: 20240917T03" format. Mock objects capture text method calls to verify corner placement at plot coordinates (0.98, 0.02). Assertions confirm corner text appears when titles lack time information. This conditional logic ensures temporal context is always available while respecting custom title preferences, supporting publication-quality plots with flexible title content.

        Parameters:
            None

        Returns:
            None
        """
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = MagicMock()
            mock_ax = MagicMock(spec=GeoAxes)
            mock_ax.transAxes = MagicMock()  
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
    
    def test_custom_title_with_time(self) -> None:
        """
        Verify suppression of duplicate corner text when custom titles already contain timestamp information. This test validates the plotting system detects temporal keywords in custom titles and avoids redundant corner text display. A custom title "Temperature Analysis - Valid: 20240917T03" containing the timestamp prevents corner text addition through keyword detection logic. Mock objects capture text method calls to verify no corner text is added. Assertions confirm zero corner text calls when time information exists in the title. This intelligent detection prevents visual clutter from duplicate timestamps, supporting clean plot layouts when users manually include temporal information in custom titles.

        Parameters:
            None

        Returns:
            None
        """
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = MagicMock()
            mock_ax = MagicMock(spec=GeoAxes)
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
    
    def test_no_timestamp(self) -> None:
        """
        Verify complete absence of time display when timestamp parameter is None. This test validates the plotting system properly handles missing temporal information without attempting to display invalid or placeholder timestamps. When time_stamp=None is passed, neither title nor corner text should contain time references. Mock objects capture set_title and text method calls to verify no "Valid Time:" or "Valid:" strings appear in any display elements. Assertions confirm zero time-related text when timestamp is absent. This ensures plots for climatological fields or non-temporal analyses remain clean without confusing or erroneous time displays.

        Parameters:
            None

        Returns:
            None
        """
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = MagicMock()
            mock_ax = MagicMock(spec=GeoAxes)
            mock_ax.transAxes = MagicMock() 
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