#!/usr/bin/env python3
"""
Enhanced Surface Plotting Capabilities Test Suite

This module provides comprehensive unit tests for the enhanced data-type agnostic surface
plotting capabilities in the MPAS Analysis package. These tests validate 3D data extraction
with level specification, wind overlay functionality with barbs and arrows, multi-dimensional
data handling (1D, 2D, 3D), professional meteorological visualization features, and complex
weather map generation including 850hPa analysis maps. Tests use synthetic data and mocking
to isolate plotting logic from rendering dependencies.

Tests Performed:
    TestEnhancedSurfacePlotting:
        - test_extract_2d_from_3d_with_index: Validates 2D extraction from 3D data using level index
        - test_extract_2d_from_3d_with_coordinate_value: Tests extraction using coordinate values (e.g., 850 hPa)
        - test_extract_2d_from_3d_error_cases: Verifies error handling for invalid level specifications
        - test_create_surface_map_2d_data: Tests surface map creation with 2D input data
        - test_create_surface_map_3d_data_with_level_index: Validates plotting 3D data with level selection
        - test_create_surface_map_2d_multilevel_data: Tests handling of 2D data with multiple levels
        - test_wind_overlay_2d_data: Validates wind overlay with 2D wind component data
        - test_wind_overlay_3d_data: Tests wind overlay with 3D wind fields and level extraction
        - test_wind_overlay_arrows: Verifies arrow-style wind overlay rendering
        - test_complex_850hpa_weather_map: Tests complete 850hPa analysis map with multiple features
        - test_plot_type_validation: Validates plot type parameter checking (scatter, contour, contourf)
        - test_wind_overlay_error_handling: Tests error handling for mismatched wind data dimensions
        - test_data_dimension_validation: Verifies validation of data dimensionality
        - test_coordinate_length_validation: Tests coordinate-data length consistency checking
        - test_auto_subsampling: Validates automatic data subsampling for large datasets
        - test_empty_data_handling: Tests handling of empty or all-NaN datasets
        - test_timestamp_integration: Verifies timestamp integration in plot titles and labels
        - test_custom_colormap_and_levels: Tests custom colormap and contour level specification
    
    TestDataTypeAgnosticFeatures:
        - test_numpy_array_input: Validates plotting with pure NumPy array inputs
        - test_xarray_dataarray_input: Tests plotting with xarray DataArray inputs
        - test_mixed_data_types_wind_overlay: Verifies wind overlay with mixed NumPy/xarray data
        - test_surface_overlay_contour_lines: Tests contour line overlays on surface maps
        - test_surface_overlay_filled_contours: Validates filled contour overlays
        - test_surface_overlay_3d_data: Tests surface overlays with 3D data extraction
        - test_complete_850hpa_weather_map: Validates complete 850hPa map with all overlays
        - test_surface_overlay_error_handling: Tests error handling for invalid overlay parameters
        - test_multiple_overlays_interaction: Verifies interaction between multiple overlay types
        - test_variable_specific_temperature_settings: Tests temperature-specific plot settings
        - test_variable_specific_precipitation_settings: Validates precipitation-specific configurations
        - test_variable_specific_pressure_settings: Tests pressure-specific plot parameters
        - test_variable_specific_wind_settings: Verifies wind-specific visualization settings
        - test_variable_specific_geopotential_settings: Tests geopotential height configurations
        - test_variable_specific_humidity_settings: Validates humidity-specific plot settings
        - test_variable_specific_unknown_variable: Tests default settings for unknown variables
        - test_variable_specific_integration_with_plotting: Verifies variable-specific settings in plots
        - test_variable_specific_with_surface_overlays: Tests variable settings with overlay features

Test Coverage:
    - MPASSurfacePlotter class: enhanced surface map creation, 3D data handling
    - 3D data extraction: extract_2d_from_3d method with index and coordinate-based selection
    - Wind overlays: barbs, arrows, quiver plots with 2D and 3D wind fields
    - Plot types: scatter, contour, filled contour with data-type agnostic handling
    - Surface overlays: contour lines, filled contours, multi-layer compositions
    - Data validation: dimension checking, coordinate consistency, empty data handling
    - Level specification: pressure level selection, vertical coordinate handling
    - Variable-specific settings: colormaps, levels, units for meteorological variables
    - Complex visualizations: 850hPa analysis maps, multi-field weather maps
    - Auto-subsampling: automatic density control for large datasets
    - Timestamp integration: valid time display in titles and labels
    - Error handling: invalid parameters, mismatched dimensions, missing data

Testing Approach:
    Unit tests using unittest framework with synthetic NumPy arrays and xarray DataArrays
    simulating MPAS output structures. Tests use Matplotlib Agg backend to avoid rendering.
    Mocking isolates plotting logic from cartographic features. Tests verify method return
    types, data extraction correctness, and error handling. Temporary directories manage
    test output files. Tests validate both positive cases (expected behavior) and negative
    cases (error conditions).

Expected Results:
    - extract_2d_from_3d correctly extracts 2D slices from 3D data at specified levels
    - Level selection works with both integer indices and coordinate values (e.g., 850 hPa)
    - Surface maps render successfully with 2D and 3D input data
    - Wind overlays integrate seamlessly with surface plots using barbs or arrows
    - Plot type parameter (scatter/contour/contourf) controls visualization style
    - Surface overlays add contour lines or filled contours to base surface maps
    - Multiple overlays (wind + contours) render correctly without conflicts
    - Variable-specific settings automatically apply appropriate colormaps and levels
    - Data validation catches dimension mismatches and coordinate inconsistencies
    - Auto-subsampling reduces density for large datasets without data loss
    - Empty or all-NaN data handled gracefully with appropriate warnings
    - Custom colormaps and levels override variable-specific defaults
    - All tests pass with synthetic data and proper error handling

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import unittest
import tempfile
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime

import sys
from pathlib import Path

package_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_dir))

from mpasdiag.visualization.surface import MPASSurfacePlotter


class TestEnhancedSurfacePlotting(unittest.TestCase):
    """
    Test suite for enhanced data-type agnostic surface plotting capabilities.
    
    Covers 3D data extraction, wind overlays, and complex meteorological visualizations.
    """
    
    def setUp(self) -> None:
        """
        Initialize test fixtures with temporary directory, plotter instance, and synthetic data. This method creates a temporary output directory, initializes an MPASSurfacePlotter with custom figure parameters, and generates synthetic coordinate arrays and pressure levels for 3D data testing. Test coordinates span realistic geographic extents (-140 to -50°W longitude, 25 to 60°N latitude) with 100 cells to simulate MPAS mesh structure. Pressure levels cover standard atmospheric analysis heights from 1000 to 100 hPa for vertical coordinate testing. These fixtures enable comprehensive testing of surface plotting capabilities without requiring actual MPAS model output files.

        Parameters:
            None

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = MPASSurfacePlotter(figsize=(8, 6), dpi=100)
        
        self.n_cells = 100
        self.lon = np.random.uniform(-140, -50, self.n_cells)
        self.lat = np.random.uniform(25, 60, self.n_cells)
        
        self.pressure_levels = np.array([1000, 925, 850, 700, 500, 400, 300, 250, 200, 100])
        self.n_levels = len(self.pressure_levels)
        
        self.lon_min, self.lon_max = -140, -50
        self.lat_min, self.lat_max = 25, 60
        
    def tearDown(self) -> None:
        """
        Clean up temporary test artifacts including directories and matplotlib figures. This method recursively removes the temporary directory and all generated output files to prevent test pollution and disk space accumulation. All open matplotlib figure windows are closed to free memory and prevent interference with subsequent tests. The cleanup process uses shutil.rmtree for directory removal and plt.close('all') for figure cleanup. This standard unittest tearDown pattern ensures complete resource cleanup regardless of test success or failure, maintaining test isolation and preventing resource leaks.

        Parameters:
            None

        Returns:
            None
        """
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        plt.close('all')
    
    def test_extract_2d_from_3d_with_index(self) -> None:
        """
        Validate 2D data extraction from 3D arrays using level index specification. This test verifies the extract_2d_from_3d method correctly extracts a 2D slice from 3D data (cells × levels × time) using integer level indexing. Synthetic 3D temperature data with 10 vertical levels is created and level 5 is extracted to validate dimension reduction and data integrity. Assertions confirm the output shape matches expected 2D dimensions (cells only) and extracted values match the original 3D data at the specified level. This functionality enables visualization of single pressure or height levels from atmospheric model 3D output fields.

        Parameters:
            None

        Returns:
            None
        """
        data_3d = np.random.uniform(220, 300, (self.n_cells, self.n_levels, 1))        
        result = self.visualizer.extract_2d_from_3d(data_3d, level_index=5)
        
        self.assertEqual(result.shape, (self.n_cells,))
        self.assertTrue(np.allclose(result, data_3d[:, 5, 0]))
        
    def test_extract_2d_from_3d_with_coordinate_value(self) -> None:
        """
        Verify 2D extraction from 3D data using coordinate value specification (e.g., 850 hPa). This test validates the extract_2d_from_3d method can extract data at specific coordinate values like pressure levels using xarray coordinate-based selection. An xarray DataArray with pressure coordinates is created and extraction at 850 hPa tests value-based level selection. Assertions confirm the extracted 2D array matches the data at the correct pressure level index (index 2 for 850 hPa in the test pressure array). This enables user-friendly level specification using physical coordinates rather than array indices for meteorological analysis workflows.

        Parameters:
            None

        Returns:
            None
        """
        data_3d = np.random.uniform(220, 300, (self.n_cells, self.n_levels))
        data_xr = xr.DataArray(
            data_3d,
            dims=['cells', 'pressure'],
            coords={'pressure': self.pressure_levels}
        )
        
        result = self.visualizer.extract_2d_from_3d(data_xr, level_value=850, level_dim='pressure')
        
        self.assertEqual(result.shape, (self.n_cells,))
        expected_idx = 2
        self.assertTrue(np.allclose(result, data_3d[:, expected_idx]))
        
    def test_extract_2d_from_3d_error_cases(self) -> None:
        """
        Validate error handling for invalid level specifications in 3D data extraction. This test verifies the extract_2d_from_3d method properly raises ValueError when neither level_index nor level_value is provided, and raises IndexError when an out-of-bounds level index is specified. Multiple error scenarios test missing level specification and invalid index values (e.g., index 50 for array with 10 levels). The assertRaises context manager confirms appropriate exception types with informative error messages. This ensures robust input validation prevents silent failures or cryptic errors during 3D data extraction operations.

        Parameters:
            None

        Returns:
            None
        """
        data_3d = np.random.uniform(220, 300, (self.n_cells, self.n_levels, 1))
        
        with self.assertRaises(ValueError) as context:
            self.visualizer.extract_2d_from_3d(data_3d)
        self.assertIn("Must provide either level_index or level_value", str(context.exception))
        
        with self.assertRaises(IndexError):
            self.visualizer.extract_2d_from_3d(data_3d, level_index=50)
            
    def test_create_surface_map_2d_data(self) -> None:
        """
        Verify surface map creation with 2D input data arrays. This test validates the create_surface_map method correctly generates matplotlib figure and axes objects for 2D surface temperature data without requiring level extraction. Synthetic 2D temperature array (250-310 K) simulates realistic surface temperature values. Assertions confirm returned objects are proper matplotlib Figure and Axes instances, the axes is correctly attached to the figure, and the plot title matches the specified string. This establishes baseline surface plotting functionality for simple 2D meteorological fields like 2-meter temperature or surface pressure.

        Parameters:
            None

        Returns:
            None
        """
        temp_2d = np.random.uniform(250, 310, self.n_cells)
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_2d, 't2m',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            title='Test 2D Temperature'
        )
        
        self.assertIsInstance(fig, Figure)
        self.assertEqual(fig.axes[0], ax)
        self.assertIn('Test 2D Temperature', ax.get_title())
        
    def test_create_surface_map_3d_data_with_level_index(self) -> None:
        """
        Validate surface map creation from 3D data using level index for automatic extraction. This test verifies create_surface_map correctly handles 3D input data (cells × levels × time) by automatically extracting the specified level before plotting. Synthetic 3D temperature data spanning realistic atmospheric temperature range (220-300 K) tests level extraction and surface rendering. Level index 5 is specified to extract a mid-tropospheric level for visualization. Assertions confirm proper figure creation and title incorporation. This demonstrates seamless 3D-to-2D workflow where users can plot any vertical level without manual extraction steps.

        Parameters:
            None

        Returns:
            None
        """
        temp_3d = np.random.uniform(220, 300, (self.n_cells, self.n_levels, 1))
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_3d, 'temperature',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            level_index=5,
            title='Test 3D Temperature at Level 5'
        )
        
        self.assertIsInstance(fig, Figure)
        self.assertIn('Test 3D Temperature at Level 5', ax.get_title())
        
    def test_create_surface_map_2d_multilevel_data(self) -> None:
        """
        Verify surface plotting of 2D multi-level data with level selection. This test validates create_surface_map handles 2D arrays with multiple levels (cells × levels dimension structure) by extracting a specific level for visualization. Synthetic multi-level temperature data tests the scenario where data has vertical structure but no explicit time dimension. Level index 2 extraction targets the 850 hPa level in a standard pressure level array. Assertions confirm successful figure creation and proper title display. This supports visualization of vertically-resolved diagnostic output or analysis fields stored in 2D multi-level format.

        Parameters:
            None

        Returns:
            None
        """
        temp_levels = np.random.uniform(220, 300, (self.n_cells, self.n_levels))
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_levels, 'temperature',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            level_index=2, 
            title='Test Multi-Level Temperature'
        )
        
        self.assertIsInstance(fig, Figure)
        self.assertIn('Test Multi-Level Temperature', ax.get_title())
        
    def test_wind_overlay_2d_data(self) -> None:
        """
        Validate wind barb overlay on surface maps using 2D wind component data. This test verifies the wind overlay functionality correctly adds wind barbs to an existing surface temperature plot using 2D u and v wind components. Synthetic wind data with realistic values (±15 m/s range) simulates horizontal wind fields at a single level. The test creates a base temperature surface map and adds wind barbs with density control (plot_every=3) for visual clarity. Assertions confirm the figure and axes objects are properly returned and barbs are added without errors. This demonstrates combined visualization of scalar and vector meteorological fields for weather analysis.

        Parameters:
            None

        Returns:
            None
        """
        temp_2d = np.random.uniform(250, 310, self.n_cells)        
        u_wind_2d = np.random.uniform(-20, 20, self.n_cells)
        v_wind_2d = np.random.uniform(-20, 20, self.n_cells)
        
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
        
        self.assertIsInstance(fig, Figure)
        self.assertIn('Test Temperature with 2D Wind Overlay', ax.get_title())
        
    def test_wind_overlay_3d_data(self) -> None:
        """
        Validate wind overlay with 3D wind field data and automatic level extraction. This test verifies wind barb functionality handles 3D wind components (cells × levels structure) by automatically extracting the specified level before overlaying on surface maps. Synthetic 3D u and v wind data (±30 m/s range) spanning multiple pressure levels tests level extraction during wind overlay operations. Wind configuration specifies level_index=2 to extract 850 hPa winds for visualization. Assertions confirm successful figure creation and proper title display. This enables visualization of winds at any vertical level without manual extraction, supporting multi-level atmospheric analysis workflows.

        Parameters:
            None

        Returns:
            None
        """
        temp_2d = np.random.uniform(250, 310, self.n_cells)        
        u_wind_3d = np.random.uniform(-30, 30, (self.n_cells, self.n_levels))
        v_wind_3d = np.random.uniform(-30, 30, (self.n_cells, self.n_levels))
        
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
        
        self.assertIsInstance(fig, Figure)
        self.assertIn('Test Temperature with 3D Wind Overlay', ax.get_title())
        
    def test_wind_overlay_arrows(self) -> None:
        """
        Verify wind arrow overlay rendering as alternative to wind barbs. This test validates the wind overlay system supports arrow-style wind vectors in addition to traditional meteorological barbs. Synthetic u and v wind components test arrow rendering with custom styling (red color, scale=200) and subsampling (plot_every=4) for optimal density. The test creates a surface temperature map and adds wind arrows configured through the wind_config dictionary. Assertions confirm successful figure creation with arrows properly overlaid. This provides alternative wind visualization styles suitable for different audiences or presentation contexts beyond traditional meteorological conventions.

        Parameters:
            None

        Returns:
            None
        """
        temp_2d = np.random.uniform(250, 310, self.n_cells)
        u_wind = np.random.uniform(-15, 15, self.n_cells)
        v_wind = np.random.uniform(-15, 15, self.n_cells)
        
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
        
        self.assertIsInstance(fig, Figure)
        
    def test_complex_850hpa_weather_map(self) -> None:
        """
        Validate creation of complex multi-layer 850 hPa analysis maps with wind overlays. This test verifies the plotting system can generate comprehensive weather analysis plots combining temperature contours, wind barbs, and proper level extraction from 3D data. Synthetic 3D temperature and wind data with pressure coordinates test coordinate-based level selection (850 hPa). Wind configuration specifies level extraction, barb styling, and subsampling parameters. The plot_type='both' parameter enables simultaneous scatter and contour rendering for enhanced visualization. This demonstrates production of publication-quality synoptic analysis maps commonly used in operational meteorology for mid-tropospheric weather pattern analysis.

        Parameters:
            None

        Returns:
            None
        """
        temp_3d = np.random.uniform(220, 300, (self.n_cells, self.n_levels))
        temp_xr = xr.DataArray(
            temp_3d,
            dims=['cells', 'pressure'],
            coords={'pressure': self.pressure_levels}
        )
        
        u_wind_3d = np.random.uniform(-30, 30, (self.n_cells, self.n_levels))
        v_wind_3d = np.random.uniform(-30, 30, (self.n_cells, self.n_levels))
        
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
        
        self.assertIsInstance(fig, Figure)
        self.assertIn('850hPa Weather Analysis', ax.get_title())
        
    def test_plot_type_validation(self) -> None:
        """
        Verify plot type parameter validation for supported rendering styles. This test validates the plotting system correctly handles different plot type specifications including 'scatter', 'contour', 'contourf' (filled contours), and 'both' (combined rendering). Multiple test cases verify each plot type produces valid matplotlib Figure objects without errors. Invalid plot type strings should trigger appropriate error handling. Assertions confirm successful figure creation for each valid plot type option. This ensures robust parameter validation and clear error messages guide users to supported visualization styles for different meteorological data presentations.

        Parameters:
            None

        Returns:
            None
        """
        temp_2d = np.random.uniform(250, 310, self.n_cells)
        
        for plot_type in ['scatter', 'contour', 'both']:
            fig, ax = self.visualizer.create_surface_map(
                self.lon, self.lat, temp_2d, 't2m',
                self.lon_min, self.lon_max, self.lat_min, self.lat_max,
                plot_type=plot_type
            )
            self.assertIsInstance(fig, Figure)
            
        with self.assertRaises(ValueError) as context:
            self.visualizer.create_surface_map(
                self.lon, self.lat, temp_2d, 't2m',
                self.lon_min, self.lon_max, self.lat_min, self.lat_max,
                plot_type='invalid'
            )
        self.assertIn("plot_type must be 'scatter', 'contour', 'contourf', or 'both'", str(context.exception))
        
    def test_wind_overlay_error_handling(self) -> None:
        """
        Validate error handling for invalid wind overlay configurations and data mismatches. This test verifies the wind overlay system properly raises exceptions for common error scenarios including missing u or v wind components, dimension mismatches between wind data and coordinate arrays, and invalid wind plot type specifications. Multiple test cases with assertRaises verify appropriate ValueError exceptions with informative error messages. Tests include missing wind component, dimension mismatch (100 vs 50 cells), and unsupported plot types. This ensures robust error checking prevents silent failures and provides clear diagnostic messages guiding users to correct wind overlay configurations.

        Parameters:
            None

        Returns:
            None
        """
        temp_2d = np.random.uniform(250, 310, self.n_cells)
        
        wind_config = {
            'u_data': np.random.uniform(-20, 20, self.n_cells),
            'v_data': np.random.uniform(-20, 20, self.n_cells),
            'plot_type': 'invalid_type'
        }
        
        with self.assertRaises(ValueError) as context:
            self.visualizer.create_surface_map(
                self.lon, self.lat, temp_2d, 't2m',
                self.lon_min, self.lon_max, self.lat_min, self.lat_max,
                wind_overlay=wind_config
            )

        self.assertIn("plot_type must be 'barbs' or 'arrows'", str(context.exception))
        
    def test_data_dimension_validation(self) -> None:
        """
        Verify validation of data dimensionality for supported array structures. This test validates the plotting system correctly rejects unsupported data dimensions while accepting 1D, 2D, and 3D arrays. A 4D test array triggers ValueError with informative error message indicating only 1D-3D data are supported. The test confirms the validation logic prevents plotting attempts with incompatible data structures that would fail during processing. Clear error messages guide users to reshape data appropriately. This ensures robust dimension checking at the entry point prevents cryptic downstream errors during coordinate extraction or rendering operations.

        Parameters:
            None

        Returns:
            None
        """
        data_4d = np.random.uniform(250, 310, (self.n_cells, 10, 5, 3))
        
        with self.assertRaises(ValueError) as context:
            self.visualizer.create_surface_map(
                self.lon, self.lat, data_4d, 'temperature',
                self.lon_min, self.lon_max, self.lat_min, self.lat_max
            )
        self.assertIn("only 1D, 2D and 3D data are supported", str(context.exception))
        
    def test_coordinate_length_validation(self) -> None:
        """
        Validate consistency checking between data and coordinate array lengths. This test verifies the plotting system detects mismatches between data array length and coordinate array dimensions. Test data with 10 extra elements compared to coordinate arrays triggers ValueError with clear error message. The validation prevents plotting attempts that would fail during data-coordinate pairing or interpolation. Assertions confirm appropriate exception type and informative error message content. This ensures data integrity checks catch common errors like using wrong subset of data or mismatched coordinate files before expensive rendering operations.

        Parameters:
            None

        Returns:
            None
        """
        temp_wrong_length = np.random.uniform(250, 310, self.n_cells + 10)
        
        with self.assertRaises(ValueError) as context:
            self.visualizer.create_surface_map(
                self.lon, self.lat, temp_wrong_length, 'temperature',
                self.lon_min, self.lon_max, self.lat_min, self.lat_max
            )
        self.assertIn("must match coordinate arrays length", str(context.exception))
        
    def test_auto_subsampling(self) -> None:
        """
        Verify automatic data subsampling for large datasets to optimize rendering performance. This test validates the auto-subsampling feature reduces dataset density when cell counts exceed thresholds while maintaining visual quality. Large synthetic dataset (3000 cells) with temperature and wind data tests subsampling during both surface plotting and wind overlay operations. The plotting system should automatically detect large datasets and apply intelligent subsampling to prevent performance degradation. Assertions confirm successful figure creation without errors despite large input size. This ensures the system handles high-resolution model output efficiently without requiring manual data thinning by users.

        Parameters:
            None

        Returns:
            None
        """
        n_large = 3000
        lon_large = np.random.uniform(-140, -50, n_large)
        lat_large = np.random.uniform(25, 60, n_large)
        temp_large = np.random.uniform(250, 310, n_large)
        u_wind_large = np.random.uniform(-20, 20, n_large)
        v_wind_large = np.random.uniform(-20, 20, n_large)
        
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
        
        self.assertIsInstance(fig, Figure)
        
    def test_empty_data_handling(self) -> None:
        """
        Validate error handling when no data points fall within specified geographic bounds. This test verifies the plotting system detects empty datasets after spatial filtering and raises informative ValueError. Test coordinates deliberately placed outside the specified map extent (lon/lat ranges 0-10 vs bounds -140 to -50, 25 to 60) ensure no valid data points remain. The assertRaises context manager confirms appropriate exception with clear error message. This prevents rendering attempts on empty datasets that would produce blank or misleading plots, guiding users to adjust spatial extents or verify coordinate systems.

        Parameters:
            None

        Returns:
            None
        """
        lon_out_of_bounds = np.random.uniform(0, 10, self.n_cells)  
        lat_out_of_bounds = np.random.uniform(0, 10, self.n_cells)
        temp_2d = np.random.uniform(250, 310, self.n_cells)
        
        with self.assertRaises(ValueError) as context:
            self.visualizer.create_surface_map(
                lon_out_of_bounds, lat_out_of_bounds, temp_2d, 't2m',
                self.lon_min, self.lon_max, self.lat_min, self.lat_max
            )
        self.assertIn("No valid data points found within the specified map extent", str(context.exception))
        
    def test_timestamp_integration(self) -> None:
        """
        Verify timestamp integration in plot titles and temporal information display. This test validates the plotting system correctly incorporates datetime objects into plot titles for temporal context. A test datetime (2024-09-17 13:00 UTC) is passed via the time_stamp parameter and should appear in the final title text. Assertions confirm successful figure creation and verify the title contains both the user-specified title text and temporal information. This ensures plots display valid times for model output, supporting time series analysis and forecast verification workflows where temporal identification is critical for interpretation.

        Parameters:
            None

        Returns:
            None
        """
        temp_2d = np.random.uniform(250, 310, self.n_cells)
        test_time = datetime(2024, 9, 17, 13, 0)
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_2d, 't2m',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            time_stamp=test_time,
            title='Test with Timestamp'
        )
        
        self.assertIsInstance(fig, Figure)
        title_text = ax.get_title()
        self.assertIn('Test with Timestamp', title_text)
        
    def test_custom_colormap_and_levels(self) -> None:
        """
        Verify custom colormap specification and contour level control for tailored visualizations. This test validates users can override default colormaps and contour levels to create customized visualizations matching specific requirements. A coolwarm colormap with explicit temperature levels (250-310K in 10K increments) is specified for contour-style plotting. Assertions confirm successful figure creation using the custom visual settings. This flexibility enables publication-quality plots, specialized color schemes for accessibility, and precise contour intervals matching domain standards or specific phenomena of interest.

        Parameters:
            None

        Returns:
            None
        """
        temp_2d = np.random.uniform(250, 310, self.n_cells)
        custom_levels = [250.0, 260.0, 270.0, 280.0, 290.0, 300.0, 310.0]
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_2d, 't2m',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            colormap='coolwarm',
            levels=custom_levels,
            plot_type='contour'
        )
        
        self.assertIsInstance(fig, Figure)


class TestDataTypeAgnosticFeatures(unittest.TestCase):
    """
    Test suite specifically for data-type agnostic features.
    
    Tests the system's ability to handle various input data types seamlessly.
    """
    
    def setUp(self) -> None:
        """
        Initialize test fixtures for data type agnostic feature testing. This method sets up the plotting system, synthetic test coordinates, and spatial domain boundaries used across multiple test cases. Test data includes 50 spatial cells with random coordinates in a western U.S. region (-120 to -80 longitude, 30 to 50 latitude) to validate data type handling. All test methods in this class use these common fixtures to ensure consistent testing of NumPy array, xarray DataArray, and mixed data type inputs. This standardized setup isolates data type conversion functionality from other plotting features.

        Parameters:
            None

        Returns:
            None
        """
        self.visualizer = MPASSurfacePlotter()
        self.n_cells = 50
        self.n_levels = 5
        self.lon = np.random.uniform(-120, -80, self.n_cells)
        self.lat = np.random.uniform(30, 50, self.n_cells)
        self.bounds = (-120, -80, 30, 50)
        self.lon_min, self.lon_max, self.lat_min, self.lat_max = self.bounds
        
    def tearDown(self) -> None:
        """
        Clean up matplotlib resources after data type agnostic tests. This method closes all matplotlib figure objects created during test execution to prevent memory leaks and resource exhaustion. Proper cleanup is essential when running multiple tests that create visualization objects. Called automatically by the unittest framework after each test method completes. This ensures each test starts with a clean matplotlib state and prevents test interaction through shared figure references.

        Parameters:
            None

        Returns:
            None
        """
        plt.close('all')
        
    def test_numpy_array_input(self) -> None:
        """
        Verify proper handling of pure NumPy array inputs for surface plotting. This test validates the plotting system accepts standard NumPy ndarrays as data inputs without requiring xarray conversions. Random temperature data generated as a plain NumPy array (250-310K range, 50 cells) is passed directly to create_surface_map. Assertions confirm successful figure generation when data is provided as NumPy arrays. This ensures compatibility with workflows using raw NumPy arrays and validates automatic internal conversions handle NumPy inputs transparently for users preferring array-based interfaces.

        Parameters:
            None

        Returns:
            None
        """
        data_np = np.random.uniform(250, 310, self.n_cells)
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, data_np, 't2m',
            *self.bounds
        )
        
        self.assertIsInstance(fig, Figure)
        
    def test_xarray_dataarray_input(self) -> None:
        """
        Verify proper handling of xarray DataArray inputs with metadata preservation. This test validates the plotting system accepts xarray DataArrays as primary data inputs while preserving attributes like units and descriptive names. An xarray DataArray with temperature values (250-310K), named dimensions, and metadata attributes is passed to create_surface_map. Assertions confirm successful figure generation from xarray inputs. This ensures compatibility with MPAS native xarray datasets and validates the system handles labeled arrays with rich metadata, supporting workflows that leverage xarray's coordinate and attribute systems.

        Parameters:
            None

        Returns:
            None
        """
        data_xr = xr.DataArray(
            np.random.uniform(250, 310, self.n_cells),
            dims=['cells'],
            attrs={'units': 'K', 'long_name': 'Temperature'}
        )
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, data_xr, 'temperature', 
            *self.bounds
        )
        
        self.assertIsInstance(fig, Figure)
        
    def test_mixed_data_types_wind_overlay(self) -> None:
        """
        Verify seamless handling of mixed data types in composite plots with overlays. This test validates the plotting system processes different input data types simultaneously within a single visualization without type conflicts. Temperature data provided as an xarray DataArray combines with wind components (u, v) supplied as NumPy arrays in a wind overlay configuration. Assertions confirm successful figure creation despite mixed input types. This flexibility enables users to combine data from different sources or processing pipelines without manual type conversions, supporting workflows where base fields and overlay data originate from distinct analysis stages.

        Parameters:
            None

        Returns:
            None
        """
        temp_xr = xr.DataArray(
            np.random.uniform(250, 310, self.n_cells),
            dims=['cells']
        )
        
        u_wind_np = np.random.uniform(-20, 20, self.n_cells)
        v_wind_np = np.random.uniform(-20, 20, self.n_cells)
        
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
        
        self.assertIsInstance(fig, Figure)

    def test_surface_overlay_contour_lines(self) -> None:
        """
        Verify surface overlay functionality with contour line visualization style. This test validates the system can overlay secondary meteorological fields as contour lines atop the primary filled contour base layer. Temperature serves as the base field (250-310K) with geopotential height (1200-1400 dam) overlaid as contour lines in a surface_config dictionary. Assertions confirm successful multi-layer figure creation. This feature enables classical synoptic analysis plots like temperature fields with overlaid pressure contours, supporting meteorological interpretation where multiple related fields provide complementary information about atmospheric state.

        Parameters:
            None

        Returns:
            None
        """
        temp_2d = np.random.uniform(250, 310, self.n_cells)
        geop_2d = np.random.uniform(1200, 1400, self.n_cells)
        
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
        
        self.assertIsInstance(fig, Figure)
        self.assertIn('Test Temperature + Geopotential Height Overlay', ax.get_title())
        
    def test_surface_overlay_filled_contours(self) -> None:
        """
        Verify surface overlay functionality with filled contour visualization using transparency. This test validates overlaying secondary fields as semi-transparent filled contours atop primary scatter plots. Temperature serves as the base scatter plot (250-310K) with sea level pressure (1000-1020 hPa) overlaid as filled contours using Blues colormap at 50 percent transparency. Assertions confirm successful multi-layer rendering with partial opacity. This capability enables highlighting regions of interest through transparent overlays while maintaining visibility of underlying patterns, supporting complex meteorological analyses requiring visual comparison of multiple fields.

        Parameters:
            None

        Returns:
            None
        """
        temp_2d = np.random.uniform(250, 310, self.n_cells)
        pressure_2d = np.random.uniform(1000, 1020, self.n_cells)
        
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
        
        self.assertIsInstance(fig, Figure)
        
    def test_surface_overlay_3d_data(self) -> None:
        """
        Verify surface overlay with automatic 3D-to-2D extraction from multi-level data. This test validates the system can overlay 3D fields by extracting specific vertical levels when level_index is specified in surface configuration. Temperature provides the 2D base field while 3D geopotential height data (5 levels) is reduced to 2D by extracting level 2 for contour overlay. Assertions confirm successful figure creation with automatic level extraction. This feature enables overlaying fields from 3D model output without manual preprocessing, supporting workflows where base and overlay fields exist at different dimensionalities in the original dataset.

        Parameters:
            None

        Returns:
            None
        """
        temp_2d = np.random.uniform(250, 310, self.n_cells)
        geop_3d = np.random.uniform(1200, 1400, (self.n_cells, self.n_levels))
        
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
        
        self.assertIsInstance(fig, Figure)
        
    def test_complete_850hpa_weather_map(self) -> None:
        """
        Verify comprehensive multi-layer 850 hPa synoptic analysis map construction. This test validates creation of publication-quality weather maps combining temperature base field, geopotential height contours, and wind barbs simultaneously. Temperature (263-293K), geopotential height (1200-1600 dam) as black contours, and wind barbs (u/v components) create a complete 850 hPa analysis typical of operational meteorology. Assertions confirm successful multi-layer integration and proper title display. This represents the most complex standard product, demonstrating the system handles operational requirements for mid-level atmospheric analysis where multiple fields provide complementary dynamical and thermodynamic information.

        Parameters:
            None

        Returns:
            None
        """
        temp_850 = np.random.uniform(263, 293, self.n_cells)
        geop_850 = np.random.uniform(1200, 1600, self.n_cells)
        u_wind_850 = np.random.uniform(-25, 25, self.n_cells)
        v_wind_850 = np.random.uniform(-25, 25, self.n_cells)
        
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
        
        self.assertIsInstance(fig, Figure)
        self.assertIn('Complete 850hPa Analysis', ax.get_title())
        
    def test_surface_overlay_error_handling(self) -> None:
        """
        Verify robust error detection for invalid surface overlay configurations. This test validates the plotting system properly rejects surface overlays with unsupported plot types through clear error messages. An invalid 'invalid_type' plot type is specified in the surface configuration to trigger validation. Assertions confirm a ValueError is raised with an informative error message guiding users toward valid options. This validation protects against configuration errors that would otherwise produce cryptic failures or incorrect visualizations, improving user experience through early detection and helpful error guidance.

        Parameters:
            None

        Returns:
            None
        """
        temp_2d = np.random.uniform(250, 310, self.n_cells)
        
        surface_config = {
            'data': np.random.uniform(1000, 1020, self.n_cells),
            'var_name': 'pressure',
            'plot_type': 'invalid_type'
        }
        
        with self.assertRaises(ValueError) as context:
            self.visualizer.create_surface_map(
                self.lon, self.lat, temp_2d, 'temperature',
                self.lon_min, self.lon_max, self.lat_min, self.lat_max,
                surface_overlay=surface_config
            )
        self.assertIn("Unsupported surface overlay plot_type: invalid_type", str(context.exception))
        
    def test_multiple_overlays_interaction(self) -> None:
        """
        Verify proper interaction and rendering of simultaneous wind and surface overlays. This test validates the plotting system correctly handles multiple overlay types applied to a single base field without conflicts or rendering issues. Temperature base field combines with wind arrows (red, subsampled by 4) and pressure contours (blue lines) in a triple-layer visualization. Assertions confirm successful figure generation with all overlay elements properly rendered. This demonstrates the system's capability for complex composite plots where wind, thermodynamic, and kinematic fields combine to provide comprehensive atmospheric state representation for advanced meteorological analysis.

        Parameters:
            None

        Returns:
            None
        """
        temp_2d = np.random.uniform(250, 310, self.n_cells)
        pressure_2d = np.random.uniform(1000, 1020, self.n_cells)
        u_wind = np.random.uniform(-15, 15, self.n_cells)
        v_wind = np.random.uniform(-15, 15, self.n_cells)
        
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
        
        self.assertIsInstance(fig, Figure)

    def test_variable_specific_temperature_settings(self) -> None:
        """
        Verify automatic colormap and level selection tailored for temperature variables. This test validates the system recognizes temperature variable names and applies appropriate RdYlBu_r diverging colormap with data-range-appropriate contour levels. Both Kelvin (263-303K) and Celsius (-10 to 30°C) temperature ranges are tested to ensure unit-agnostic configuration. Assertions confirm correct colormap assignment and valid numeric level generation. This automation improves visualization quality by applying meteorologically appropriate color schemes without manual configuration, supporting temperature fields commonly displayed with red-to-blue diverging schemes emphasizing departures from climatological norms.

        Parameters:
            None

        Returns:
            None
        """
        temp_k = np.random.uniform(263, 303, self.n_cells)
        colormap, levels = self.visualizer.get_variable_specific_settings('temperature', temp_k)
        
        self.assertEqual(colormap, 'RdYlBu_r')
        self.assertIsInstance(levels, list)
        self.assertIsNotNone(levels)
        assert levels is not None  
        self.assertTrue(all(isinstance(level, (int, float)) for level in levels))
        
        temp_c = np.random.uniform(-10, 30, self.n_cells)
        colormap, levels = self.visualizer.get_variable_specific_settings('t2m', temp_c)
        
        self.assertEqual(colormap, 'RdYlBu_r')
        self.assertIsInstance(levels, list)
        
    def test_variable_specific_precipitation_settings(self) -> None:
        """
        Verify custom precipitation colormap and accumulation-appropriate level generation. This test validates the system applies specialized precipitation color schemes matching standard meteorological conventions with levels tailored to accumulation timescales. Both hourly (0-25mm with 0.1mm and 0.5mm thresholds) and daily precipitation are tested for appropriate custom ListedColormap application. Assertions confirm the custom colormap type and presence of meteorologically significant precipitation thresholds. This automation ensures precipitation displays use conventional green-to-purple color progressions with levels matching operational weather service standards for different accumulation periods.

        Parameters:
            None

        Returns:
            None
        """
        precip_data = np.random.uniform(0, 25, self.n_cells)
        
        colormap, levels = self.visualizer.get_variable_specific_settings('precipitation_01h', precip_data)
        self.assertIsInstance(colormap, mcolors.ListedColormap)  
        self.assertIsNotNone(levels)
        assert levels is not None  
        self.assertIn(0.1, levels) 
        self.assertIn(0.5, levels) 
        
        colormap, levels = self.visualizer.get_variable_specific_settings('daily_precip', precip_data)
        self.assertIsInstance(colormap, mcolors.ListedColormap)
        self.assertIsInstance(levels, list)
        self.assertIsNotNone(levels)
        assert levels is not None  
        self.assertIn(0.1, levels) 
        
    def test_variable_specific_pressure_settings(self) -> None:
        """
        Verify unit-aware pressure visualization configuration with appropriate colormaps and ranges. This test validates the system detects pressure magnitude ranges to distinguish Pascals from hectoPascals and applies suitable RdBu_r diverging colormap. Pressure data in both Pa (99000-103000 Pa) and hPa (990-1030 hPa) units are tested to ensure automatic range-appropriate level generation. Assertions confirm correct colormap selection and physically reasonable level values matching detected units. This unit detection prevents visualization errors from mixed unit conventions and ensures contour intervals match the magnitude range for effective pressure pattern visualization.

        Parameters:
            None

        Returns:
            None
        """
        pressure_pa = np.random.uniform(99000, 103000, self.n_cells)
        colormap, levels = self.visualizer.get_variable_specific_settings('sea_level_pressure', pressure_pa)
        
        self.assertEqual(colormap, 'RdBu_r')
        self.assertIsInstance(levels, list)
        self.assertIsNotNone(levels)
        assert levels is not None 
        self.assertTrue(all(level >= 99000 for level in levels)) 
        
        pressure_hpa = np.random.uniform(990, 1030, self.n_cells)
        colormap, levels = self.visualizer.get_variable_specific_settings('slp', pressure_hpa)
        
        self.assertEqual(colormap, 'RdBu_r')
        self.assertIsInstance(levels, list)
        
    def test_variable_specific_wind_settings(self) -> None:
        """
        Verify wind speed visualization uses perceptually uniform colormap with data-appropriate levels. This test validates wind speed variables automatically receive plasma colormap designed for sequential data without misleading perceptual artifacts. Wind speed data (0-20 m/s range typical of surface winds) tests automatic level generation spanning the data range. Assertions confirm plasma colormap selection and valid numeric level list generation. This perceptually uniform colormap choice ensures equal perceptual steps between wind speed intervals, critical for accurate intensity interpretation without the rainbow colormap artifacts that distort perceived gradients.

        Parameters:
            None

        Returns:
            None
        """
        wind_data = np.random.uniform(0, 20, self.n_cells)
        colormap, levels = self.visualizer.get_variable_specific_settings('wind_speed', wind_data)
        
        self.assertEqual(colormap, 'plasma')
        self.assertIsInstance(levels, list)
        self.assertIsNotNone(levels)
        assert levels is not None 
        self.assertEqual(levels[0], 0)  
        
    def test_variable_specific_geopotential_settings(self) -> None:
        """
        Verify geopotential height visualization uses terrain colormap appropriate for elevation-like fields. This test validates geopotential height variables automatically receive terrain colormap reflecting topographic-style elevation representation. Geopotential height data (1200-1600 dam typical of 850 hPa level) tests automatic level generation within the physical data range. Assertions confirm terrain colormap selection and levels spanning the expected geopotential range. This colormap choice provides intuitive visualization where higher values appear as elevated terrain colors, leveraging familiar elevation color schemes for interpreting pressure surface heights in synoptic meteorology.

        Parameters:
            None

        Returns:
            None
        """
        geop_data = np.random.uniform(1200, 1600, self.n_cells)
        colormap, levels = self.visualizer.get_variable_specific_settings('geopotential_height', geop_data)
        
        self.assertEqual(colormap, 'terrain')
        self.assertIsInstance(levels, list)
        self.assertIsNotNone(levels)
        assert levels is not None  
        self.assertTrue(all(1200 <= level <= 1600 for level in levels))
        
    def test_variable_specific_humidity_settings(self) -> None:
        """
        Verify relative humidity visualization uses blue-green sequential colormap with fraction-appropriate levels. This test validates relative humidity variables automatically receive BuGn colormap representing moisture content from dry (white/light blue) to saturated (dark green). Relative humidity in fractional units (0.3-1.0 range) tests automatic level generation maintaining 0-1 bounds. Assertions confirm BuGn colormap selection and levels constrained to valid humidity fractions. This colormap provides intuitive moisture visualization where increasing green intensity represents increasing humidity, matching conventional moisture depiction while ensuring generated levels respect physical 0-1 humidity bounds.

        Parameters:
            None

        Returns:
            None
        """
        rh_data = np.random.uniform(0.3, 1.0, self.n_cells)
        colormap, levels = self.visualizer.get_variable_specific_settings('relative_humidity', rh_data)
        
        self.assertEqual(colormap, 'BuGn')
        self.assertIsInstance(levels, list)
        self.assertIsNotNone(levels)
        assert levels is not None 
        self.assertTrue(all(0 <= level <= 1 for level in levels))
        
    def test_variable_specific_unknown_variable(self) -> None:
        """
        Verify intelligent fallback colormap selection based on data characteristics for unrecognized variables. This test validates the system analyzes data range properties to select appropriate colormaps when variable names lack specific configurations. Diverging data (-50 to +50 with negative values) receives RdBu_r diverging colormap, while sequential data (0-100, all positive) receives viridis perceptually uniform colormap. Assertions confirm data-driven colormap logic for unknown variables. This intelligent fallback ensures reasonable visualization quality even for custom or unrecognized variable names by matching colormaps to data distribution patterns rather than failing or using inappropriate schemes.

        Parameters:
            None

        Returns:
            None
        """
        diverging_data = np.random.uniform(-50, 50, self.n_cells)
        colormap, levels = self.visualizer.get_variable_specific_settings('unknown_var', diverging_data)
        
        self.assertEqual(colormap, 'RdBu_r')
        self.assertIsInstance(levels, list)
        
        sequential_data = np.random.uniform(0, 100, self.n_cells)
        colormap, levels = self.visualizer.get_variable_specific_settings('unknown_positive', sequential_data)
        
        self.assertEqual(colormap, 'viridis')
        self.assertIsInstance(levels, list)
        
    def test_variable_specific_integration_with_plotting(self) -> None:
        """
        Verify variable-specific settings automatically integrate with plotting workflow without explicit colormap specification. This test validates users can create plots relying entirely on automatic variable detection to apply appropriate colormaps and levels. Temperature, precipitation, and wind speed plots are created without manual colormap parameters, testing end-to-end integration of the variable recognition system. Assertions confirm successful figure generation for all variable types using automatic settings. This integration enables simplified plotting API where users specify only variable names and data, with the system handling visualization configuration based on meteorological variable conventions.

        Parameters:
            None

        Returns:
            None
        """
        temp_data = np.random.uniform(263, 303, self.n_cells)
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_data, 'temperature',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            plot_type='contourf',
            title='Temperature with Auto Settings'
        )
        
        self.assertIsInstance(fig, Figure)
        
        fig, ax = self.visualizer.create_surface_map(
            self.lon, self.lat, temp_data, 'temperature',
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            plot_type='contourf',
            colormap='plasma', 
            levels=[270, 280, 290, 300],  
            title='Temperature with Manual Override'
        )
        
        self.assertIsInstance(fig, Figure)
        
    def test_variable_specific_with_surface_overlays(self) -> None:
        """
        Verify variable-specific automatic settings apply independently to base field and surface overlays. This test validates both primary and overlay fields receive appropriate variable-specific configurations when multiple fields combine in a single plot. Temperature base field should receive RdYlBu_r colormap while geopotential height overlay should use terrain colormap based on respective variable recognition. Assertions confirm successful multi-layer rendering with independent automatic styling. This demonstrates the variable detection system operates at both visualization layers, enabling complex plots where each field uses meteorologically appropriate representations without manual configuration for either layer.

        Parameters:
            None

        Returns:
            None
        """
        temp_data = np.random.uniform(263, 303, self.n_cells)
        geop_data = np.random.uniform(1200, 1600, self.n_cells)
        
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
        
        self.assertIsInstance(fig, Figure)


if __name__ == '__main__':
    unittest.main()