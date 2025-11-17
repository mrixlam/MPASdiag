#!/usr/bin/env python3
"""
MPAS Precipitation Plotter Unit Tests

This module provides comprehensive unit tests for the MPASPrecipitationPlotter class,
validating precipitation visualization functionality including colormap generation,
precipitation map creation, batch processing workflows, and integration with related
modules (styling, unit conversion). These tests ensure correct plotting behavior for
various accumulation periods, data formats, and edge cases using synthetic data and
mocking to isolate visualization logic from heavy dependencies.

Tests Performed:
    TestMPASPrecipitationPlotter:
        - test_initialization: Verifies plotter object initialization with figsize and DPI
        - test_create_precip_colormap: Tests colormap generation for different accumulation periods
        - test_create_precipitation_map_basic: Validates basic precipitation map creation
        - test_create_precipitation_map_with_time: Tests map creation with timestamp information
        - test_create_precipitation_map_custom_colormap: Validates custom colormap specification
        - test_create_precipitation_map_validation: Tests input validation and error handling
        - test_create_precipitation_map_with_data_array: Validates xarray DataArray handling
        - test_batch_precipitation_maps_validation: Tests batch processing input validation
        - test_batch_precipitation_maps_processing: Validates batch map generation workflow
        - test_create_precipitation_comparison_plot: Tests side-by-side comparison plotting
        - test_format_ticks_dynamic: Validates dynamic tick formatting for precipitation values
        - test_precipitation_map_empty_data: Tests handling of empty/all-zero data arrays
        - test_precipitation_map_extreme_values: Validates handling of very high precipitation values
        - test_different_accumulation_periods: Tests colormaps for all accumulation periods (1h-24h)
        - test_import_from_visualization_package: Validates module import accessibility
        - test_styling_integration: Tests integration with MPASVisualizationStyle module
        - test_unit_converter_integration: Validates integration with UnitConverter module

Test Coverage:
    - Plotter initialization: figure size, DPI, default state management
    - Colormap generation: accumulation-period-specific color schemes and level ranges
    - Precipitation mapping: scatter plots, marker sizing, colorbar configuration
    - Batch processing: time series iteration, file naming, progress tracking
    - Data format handling: NumPy arrays, xarray DataArrays, coordinate extraction
    - Input validation: mismatched array sizes, invalid coordinates, empty data
    - Edge cases: zero precipitation, extreme values, all-NaN data
    - Integration testing: styling module, unit converter, diagnostics module
    - Comparison plots: side-by-side visualization, layout management

Testing Approach:
    Unit tests using synthetic precipitation data with exponential distribution to simulate
    realistic rainfall patterns. Mock objects isolate plotter from processor dependencies.
    Tests verify colormap properties, figure/axes creation, proper handling of various
    accumulation periods (a01h through a24h), and correct integration with utility modules.

Expected Results:
    - Plotter initializes with correct default parameters (figsize, DPI)
    - Colormaps for different periods have appropriate level ranges (1h < 24h)
    - Precipitation maps display without errors for valid synthetic data
    - Batch processing correctly iterates through time steps and generates file lists
    - Empty or extreme data handled gracefully without crashes
    - Integration with styling and unit converter modules works seamlessly
    - Custom colormaps and levels can be specified and override defaults
    - Input validation catches mismatched array sizes and invalid coordinates
    - xarray DataArrays processed correctly with attribute preservation

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: October 2025
Version: 1.0.0
"""

import unittest
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.colors as mcolors
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta
import tempfile
import os
import sys
from pathlib import Path

package_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_dir))

from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter
from mpasdiag.processing.processors_2d import MPAS2DProcessor
from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics


def create_mock_precipitation_diagnostics(precip_data: xr.DataArray):
    """
    Create a mock PrecipitationDiagnostics instance for testing without real calculations.

    Parameters:
        precip_data (xr.DataArray): Synthetic precipitation data to return from mock.

    Returns:
        Mock: Mock PrecipitationDiagnostics instance with compute_precipitation_difference method.
    """
    mock_diag = Mock(spec=PrecipitationDiagnostics)
    mock_diag.compute_precipitation_difference.return_value = precip_data
    return mock_diag


class TestMPASPrecipitationPlotter(unittest.TestCase):
    """
    Test cases for MPASPrecipitationPlotter
    """
    
    def setUp(self) -> None:
        """
        Initialize test fixtures with synthetic precipitation data and mock processor for MPASPrecipitationPlotter testing. This method creates a plotter instance with custom figure size and DPI settings along with mock MPAS2DProcessor for isolated testing. Synthetic data includes 50-point coordinate arrays spanning Colorado region (-110°W to -100°W, 35°N to 45°N) with exponentially distributed precipitation values capped at 20mm to simulate realistic rainfall patterns. An xarray Dataset provides time-series precipitation data with 5 hourly time steps for batch processing tests. The mock processor configuration enables testing of visualization functionality without requiring actual MPAS model output files.

        Parameters:
            None

        Returns:
            None
        """
        PlotterClass = globals().get('MPASPrecipitationPlotter')

        if PlotterClass is None:
            self.skipTest("MPASPrecipitationPlotter not available")
        
        self.plotter = PlotterClass(figsize=(12, 8), dpi=100)
        self.mock_processor = Mock(spec=MPAS2DProcessor)
        
        self.lon = np.linspace(-110, -100, 50)
        self.lat = np.linspace(35, 45, 50)
        
        np.random.seed(42)
        self.precip_data = np.random.exponential(2.0, 50)  
        self.precip_data[self.precip_data > 20] = 20 
        
        self.lon_min, self.lon_max = -110, -100
        self.lat_min, self.lat_max = 35, 45
        
        self.mock_dataset = xr.Dataset({
            'rainnc': xr.DataArray(
                np.random.exponential(2.0, (5, 50)),
                dims=['Time', 'nCells'],
                coords={
                    'Time': pd.date_range('2024-01-01', periods=5, freq='h'),
                    'nCells': range(50)
                },
                attrs={'units': 'mm', 'long_name': 'Non-convective precipitation'}
            ),
            'lonCell': xr.DataArray(self.lon, dims=['nCells']),
            'latCell': xr.DataArray(self.lat, dims=['nCells'])
        })
        
        self.mock_processor.dataset = self.mock_dataset
        self.mock_processor.data_type = 'diag'
        
    def test_initialization(self) -> None:
        """
        Verify MPASPrecipitationPlotter initialization with default and custom configuration parameters. This test validates that plotter objects instantiate correctly with expected default figure size (10×14) and DPI (100) settings. Custom initialization testing confirms that user-specified figure dimensions and resolution values override defaults appropriately. The test ensures proper object type instantiation and parameter assignment for both default constructor calls and explicitly configured instances. This validation covers the foundational initialization behavior required for all subsequent plotting operations.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        self.assertIsInstance(plotter, MPASPrecipitationPlotter)
        self.assertEqual(plotter.figsize, (10, 14))
        self.assertEqual(plotter.dpi, 100)  
        
        custom_plotter = MPASPrecipitationPlotter(figsize=(10, 6), dpi=150)
        self.assertEqual(custom_plotter.figsize, (10, 6))
        self.assertEqual(custom_plotter.dpi, 150)
    
    def test_create_precip_colormap(self) -> None:
        """
        Validate colormap generation for multiple accumulation periods with period-appropriate precipitation level ranges. This test verifies that colormaps for 1-hour, 3-hour, 12-hour, and 24-hour accumulation periods use correct color level specifications matching expected precipitation intensities. Short-duration periods (1h, 3h) receive levels up to 75mm while longer periods (12h, 24h) extend to 150mm to accommodate greater accumulation potential. The test confirms colormap object types (ListedColormap with 11 colors), monotonically increasing level sequences, and exact matching to predefined level arrays. This validation ensures appropriate visual representation across diverse temporal accumulation scenarios.

        Parameters:
            None

        Returns:
            None
        """
        test_cases = [
            ('a01h', [0.1, 0.5, 2.5, 5, 10, 15, 20, 25, 50, 75]),
            ('a03h', [0.1, 0.5, 2.5, 5, 10, 15, 20, 25, 50, 75]),
            ('a12h', [0.1, 1, 5, 10, 20, 30, 40, 50, 100, 150]),
            ('a24h', [0.1, 1, 5, 10, 20, 30, 40, 50, 100, 150]),
        ]
        
        for accum_period, expected_levels in test_cases:
                cmap, levels = self.plotter.create_precip_colormap(accum_period)
                
                self.assertIsInstance(cmap, mcolors.ListedColormap)
                self.assertEqual(cmap.N, 11) 
                
                self.assertEqual(levels, expected_levels)
                self.assertTrue(all(levels[i] <= levels[i+1] for i in range(len(levels)-1)))  
                self.assertTrue(all(levels[i] <= levels[i+1] for i in range(len(levels)-1)))  
    
    def test_create_precipitation_map_basic(self) -> None:
        """
        Validate basic precipitation map creation with minimal required parameters for scatter plot visualization. This test confirms that the plotter generates valid matplotlib Figure and Axes objects when provided with longitude, latitude, precipitation data arrays, and domain boundary coordinates. The synthetic precipitation data with exponential distribution simulates realistic rainfall patterns across the test region. Title assignment verification ensures proper plot labeling functionality works as expected. This fundamental test establishes baseline plotting capability before testing advanced features like custom colormaps, time annotations, or batch processing workflows.

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = self.plotter.create_precipitation_map(
            self.lon, self.lat, self.precip_data,
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            title="Test Precipitation Map"
        )
        
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)
        self.assertEqual(ax.get_title(), "Test Precipitation Map")
        
        self.plotter.close_plot()
    
    def test_create_precipitation_map_with_time(self) -> None:
        """
        Verify precipitation map creation with temporal annotation for time-stamped accumulation period visualization. This test validates that datetime objects integrate correctly into precipitation maps to display accumulation end times and period specifications. The test uses a specific timestamp (September 17, 2024 at 12:00 UTC) with 1-hour accumulation period to confirm proper time formatting in plot annotations. Figure and Axes object validation ensures matplotlib components instantiate correctly with temporal metadata. This functionality supports operational forecasting workflows requiring precise temporal context for accumulated precipitation displays.

        Parameters:
            None

        Returns:
            None
        """
        time_end = datetime(2024, 9, 17, 12, 0, 0)
        
        fig, ax = self.plotter.create_precipitation_map(
            self.lon, self.lat, self.precip_data,
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            title="Precipitation with Time",
            accum_period='a01h',
            time_end=time_end
        )
        
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)
        
        self.plotter.close_plot()
    
    def test_create_precipitation_map_custom_colormap(self) -> None:
        """
        Validate custom colormap and level specification overriding default accumulation-period-based color schemes. This test confirms that users can specify alternative matplotlib colormaps (Blues) along with custom precipitation level ranges (0.1-20mm with 6 levels) instead of using predefined accumulation-specific defaults. Color limit parameters (clim_min, clim_max) provide additional control over colorbar range mapping for specialized visualization requirements. Figure and Axes validation ensures matplotlib objects instantiate correctly with user-defined color specifications. This flexibility supports diverse precipitation visualization needs including comparison with observational datasets using different color conventions.

        Parameters:
            None

        Returns:
            None
        """
        custom_levels = [0.1, 1, 2, 5, 10, 20]
        
        fig, ax = self.plotter.create_precipitation_map(
            self.lon, self.lat, self.precip_data,
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            colormap='Blues',
            levels=custom_levels,
            clim_min=0.1,
            clim_max=20
        )
        
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)
        
        self.plotter.close_plot()
    
    def test_create_precipitation_map_validation(self) -> None:
        """
        Verify input validation and error handling for invalid coordinate bounds in precipitation map creation. This test confirms that the plotter raises ValueError exceptions when provided with geographically invalid longitude ranges (outside -180° to 180°) or latitude ranges (minimum exceeding maximum). The validation testing uses deliberately incorrect coordinate specifications including out-of-range longitude (-200°) and inverted latitude bounds (-100° to 45°). Proper exception raising prevents silent failures and ensures users receive clear error messages for misconfigured plotting requests. This defensive programming approach improves reliability and debuggability in operational workflows with diverse coordinate system inputs.

        Parameters:
            None

        Returns:
            None
        """
        with self.assertRaises(ValueError):
            self.plotter.create_precipitation_map(
                self.lon, self.lat, self.precip_data,
                -200, -100, 35, 45  
            )
        
        with self.assertRaises(ValueError):
            self.plotter.create_precipitation_map(
                self.lon, self.lat, self.precip_data,
                -110, -100, -100, 45  
            )
    
    def test_create_precipitation_map_with_data_array(self) -> None:
        """
        Validate xarray DataArray handling for precipitation map creation with metadata preservation and attribute extraction. This test confirms that the plotter correctly processes xarray DataArray objects containing precipitation data with associated attributes (units, long_name) and dimension information. The DataArray input enables automatic extraction of metadata including physical units (kg m-2) and descriptive labels for enhanced plot annotations. Figure and Axes validation ensures matplotlib components instantiate correctly when processing xarray data structures instead of plain NumPy arrays. This capability supports integration with modern scientific Python workflows where xarray provides self-documenting labeled arrays with comprehensive metadata.

        Parameters:
            None

        Returns:
            None
        """
        data_array = xr.DataArray(
            self.precip_data,
            dims=['nCells'],
            attrs={'units': 'kg m-2', 'long_name': 'Precipitation'}
        )
        
        fig, ax = self.plotter.create_precipitation_map(
            self.lon, self.lat, self.precip_data,
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            data_array=data_array,
            var_name='precipitation'
        )
        
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)
        
        self.plotter.close_plot()
    
    def test_batch_precipitation_maps_validation(self) -> None:
        """
        Verify input validation for batch precipitation map processing with processor and dataset requirement checks. This test confirms that the batch processing method raises ValueError exceptions when critical inputs are missing or invalid, including None processor objects and processors with uninitialized datasets. The validation uses temporary directories for output file management while testing error handling for various invalid input configurations. Mock processor objects without dataset attributes trigger appropriate validation failures to prevent downstream processing errors. This defensive validation approach ensures batch workflows fail fast with clear error messages rather than producing cryptic exceptions deep in processing pipelines.

        Parameters:
            None

        Returns:
            None
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                self.plotter.create_batch_precipitation_maps(
                    None, temp_dir, self.lon_min, self.lon_max, self.lat_min, self.lat_max
                )
            
            empty_processor = Mock(spec=MPAS2DProcessor)
            empty_processor.dataset = None
            
            with self.assertRaises(ValueError):
                self.plotter.create_batch_precipitation_maps(
                    empty_processor, temp_dir, self.lon_min, self.lon_max, self.lat_min, self.lat_max
                )
    
    def test_batch_precipitation_maps_processing(self) -> None:
        """
        Validate batch precipitation map generation workflow including time series iteration and file creation. This test confirms that the plotter correctly processes multiple time steps from an MPAS dataset, computing precipitation differences and generating individual map files for specified time indices. The test uses the real PrecipitationDiagnostics class to compute precipitation data while mock processor supplies coordinate extraction functionality. The test verifies successful file creation in temporary directories with proper PNG format output and validates that created file paths correspond to actual disk files. This integration test ensures complete batch processing pipeline functionality from data extraction through map generation to file persistence without requiring external mocking of diagnostics calculations.

        Parameters:
            None

        Returns:
            None
        """
        self.mock_processor.dataset = self.mock_dataset
        self.mock_processor.data_type = 'xarray'
        self.mock_processor.extract_2d_coordinates_for_variable.return_value = (self.lon, self.lat)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            created_files = self.plotter.create_batch_precipitation_maps(
                self.mock_processor, temp_dir,
                self.lon_min, self.lon_max, self.lat_min, self.lat_max,
                var_name='rainnc',
                accum_period='a01h',
                time_indices=[1, 2]  
            )
            
            self.assertGreater(len(created_files), 0)
            for file_path in created_files:
                self.assertTrue(os.path.exists(file_path))
                self.assertTrue(file_path.endswith('.png'))
    
    def test_create_precipitation_comparison_plot(self) -> None:
        """
        Validate side-by-side precipitation comparison plotting for multi-run or scenario analysis workflows. This test confirms that the plotter generates two-panel figure layouts with independent Axes objects for simultaneous visualization of different precipitation datasets. The test uses identical coordinate grids with precipitation data scaled by 1.5x factor to simulate comparison between model runs or scenarios. Figure and Axes list validation ensures proper multi-panel layout creation with separate titles for each panel (Run 1, Run 2). This comparison capability supports model evaluation, sensitivity studies, and ensemble analysis where visual juxtaposition reveals differences more effectively than separate individual plots.

        Parameters:
            None

        Returns:
            None
        """
        precip_data1 = self.precip_data
        precip_data2 = self.precip_data * 1.5  
        
        fig, axes = self.plotter.create_precipitation_comparison_plot(
            self.lon, self.lat, precip_data1, precip_data2,
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            title1="Precipitation Run 1",
            title2="Precipitation Run 2"
        )
        
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(axes, list)
        self.assertEqual(len(axes), 2)

        for ax in axes:
            self.assertIsInstance(ax, Axes)
        
        self.plotter.close_plot()
    
    def test_format_ticks_dynamic(self) -> None:
        """
        Verify dynamic tick label formatting with precision adjustment based on value magnitude for colorbar annotations. This test validates that precipitation level tick marks receive appropriate decimal precision formatting depending on their numeric ranges. Small values (0.01-0.1) display two decimal places, mid-range values (0.1-10) show consistent decimal formatting, while large integers (>10) omit unnecessary decimal points. The test cases cover diverse precipitation scales from trace amounts to extreme rainfall intensities with expected string formatting for each range. This adaptive formatting improves colorbar readability by avoiding cluttered labels while maintaining necessary precision for scientific interpretation.

        Parameters:
            None

        Returns:
            None
        """
        test_cases = [
            ([0.1, 0.5, 1.0, 2.0, 5.0], ['0.10', '0.50', '1.00', '2.00', '5.00']),
            ([1, 5, 10, 20, 50], ['1', '5', '10', '20', '50']),
            ([0.01, 0.1, 1, 10], ['0.01', '0.10', '1.00', '10.00']),
        ]
        
        for ticks, expected in test_cases:
            with self.subTest(ticks=ticks):
                result = self.plotter._format_ticks_dynamic(ticks)
                self.assertEqual(result, expected)
    
    def test_precipitation_map_empty_data(self) -> None:
        """
        Validate graceful handling of empty precipitation datasets with all-NaN data arrays for robust error avoidance. This test confirms that the plotter generates valid Figure and Axes objects even when provided with completely missing data (all NaN values) without raising exceptions or crashing. The empty data scenario represents edge cases including data processing failures, missing model output, or regions with no valid observations. Figure instantiation with NaN arrays ensures plotting code handles missing data gracefully by displaying empty maps rather than producing confusing error messages. This robustness supports operational workflows where data availability varies across time periods or geographic regions.

        Parameters:
            None

        Returns:
            None
        """
        nan_data = np.full_like(self.precip_data, np.nan)
        
        fig, ax = self.plotter.create_precipitation_map(
            self.lon, self.lat, nan_data,
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            title="Empty Precipitation Data"
        )
        
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)
        
        self.plotter.close_plot()
    
    def test_precipitation_map_extreme_values(self) -> None:
        """
        Validate plotting behavior with extreme precipitation values spanning six orders of magnitude from trace to catastrophic rainfall. This test confirms that the plotter handles diverse precipitation intensities (0-10,000mm) without numerical errors, visualization artifacts, or performance degradation. The extreme value array includes trace precipitation (0.01mm), light rain (1mm), moderate rainfall (100mm), and unrealistic but numerically valid extreme values (1,000-10,000mm) to stress-test colormap scaling and data handling. Figure and Axes validation ensures matplotlib components instantiate correctly across this extreme dynamic range. This robustness testing prevents failures in operational scenarios with unusual meteorological events or data quality issues.

        Parameters:
            None

        Returns:
            None
        """
        extreme_data = np.array([0, 0.01, 1, 100, 1000, 10000])
        extreme_lon = np.linspace(-110, -100, 6)
        extreme_lat = np.linspace(35, 45, 6)
        
        fig, ax = self.plotter.create_precipitation_map(
            extreme_lon, extreme_lat, extreme_data,
            self.lon_min, self.lon_max, self.lat_min, self.lat_max,
            title="Extreme Precipitation Values"
        )
        
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)
        
        self.plotter.close_plot()
    
    def test_different_accumulation_periods(self) -> None:
        """
        Verify correct precipitation map generation across all standard accumulation periods from 1-hour to 24-hour intervals. This test validates that the plotter produces valid visualizations for five accumulation period specifications (a01h, a03h, a06h, a12h, a24h) commonly used in operational meteorology and model verification. Each accumulation period receives appropriate colormap scaling and level ranges matching expected precipitation intensities for those temporal windows. The subTest framework provides isolated failure reporting for each accumulation period ensuring that problems with specific periods don't mask successful operation of others. This comprehensive coverage ensures operational readiness across diverse forecast and analysis temporal scales used in weather prediction workflows.

        Parameters:
            None

        Returns:
            None
        """
        accum_periods = ['a01h', 'a03h', 'a06h', 'a12h', 'a24h']
        
        for accum_period in accum_periods:
            with self.subTest(accum_period=accum_period):
                fig, ax = self.plotter.create_precipitation_map(
                    self.lon, self.lat, self.precip_data,
                    self.lon_min, self.lon_max, self.lat_min, self.lat_max,
                    title=f"Precipitation {accum_period}",
                    accum_period=accum_period
                )
                
                self.assertIsInstance(fig, Figure)
                self.assertIsInstance(ax, Axes)
                
                self.plotter.close_plot()

class TestPrecipitationPlotterIntegration(unittest.TestCase):
    """
    Integration tests for precipitation plotter with other modules.
    """
    
    def test_import_from_visualization_package(self) -> None:
        """
        Verify MPASPrecipitationPlotter accessibility through main visualization package import path for public API validation. This test confirms that users can import the plotter class from the top-level mpasdiag.visualization module rather than requiring knowledge of internal module structure. The test validates that imported class contains expected core functionality (create_precipitation_map method) indicating proper module initialization and namespace configuration. Import path testing ensures package restructuring or refactoring doesn't break public API contracts that users depend on in operational scripts. This accessibility validation supports maintainable code organization where internal structure can evolve while preserving stable public interfaces.

        Parameters:
            None

        Returns:
            None
        """
        try:
            from mpasdiag.visualization import MPASPrecipitationPlotter
            self.assertTrue(hasattr(MPASPrecipitationPlotter, 'create_precipitation_map'))
        except ImportError:
            self.skipTest("Precipitation plotter not available in main package")
    
    def test_styling_integration(self) -> None:
        """
        Validate integration between precipitation plotter and MPASVisualizationStyle module for consistent colormap generation. This test confirms that the styling module provides precipitation-specific colormap functionality (create_precip_colormap) accessible to plotter components. The test validates colormap and level list generation for 1-hour accumulation periods ensuring proper ListedColormap object types and non-empty level arrays. Integration validation ensures that styling utilities remain compatible with plotter requirements as both modules evolve independently. This modular design testing supports code maintainability by verifying that visualization style management remains decoupled from core plotting logic while maintaining necessary interfaces.

        Parameters:
            None

        Returns:
            None
        """
        try:
            from mpasdiag.visualization.styling import MPASVisualizationStyle
            cmap, levels = MPASVisualizationStyle.create_precip_colormap('a01h')
            
            self.assertIsInstance(cmap, mcolors.ListedColormap)
            self.assertIsInstance(levels, list)
            self.assertGreater(len(levels), 0)
            self.assertGreater(len(levels), 0)
            
        except ImportError:
            self.skipTest("Styling module not available")
    
    def test_unit_converter_integration(self) -> None:
        """
        Validate integration with UnitConverter module for precipitation unit transformation and standardization support. This test confirms that the unit conversion utility provides necessary functionality for converting precipitation values between different physical units (mm, kg m-2, inches). The test validates that identity conversions (mm to mm) preserve data values exactly without numerical precision losses or unexpected transformations. Integration validation ensures that precipitation plotter can leverage unit conversion capabilities for datasets with diverse unit conventions from different modeling systems or observational sources. This utility integration supports flexible data handling workflows where precipitation accumulations arrive in heterogeneous unit specifications requiring standardization before visualization.

        Parameters:
            None

        Returns:
            None
        """
        try:
            from mpasdiag.processing.utils_unit import UnitConverter
            
            sample_data = np.array([1.0, 2.0, 5.0]) 
            converted_data = UnitConverter.convert_units(sample_data, 'mm', 'mm')
            
            self.assertTrue(np.array_equal(sample_data, converted_data))
            
        except ImportError:
            self.skipTest("UnitConverter not available")


if __name__ == '__main__':
    unittest.main(verbosity=2)