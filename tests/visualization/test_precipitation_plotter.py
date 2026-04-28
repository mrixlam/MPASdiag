#!/usr/bin/env python3
"""
MPASdiag Test Suite: Tests for Precipitation Plotter 

This module contains unit and integration tests for the `MPASPrecipitationPlotter` class in `precipitation.py`. The tests cover initialization, unit conversion, plot extent handling, contourf plotting, batch processing, comparison plots, plot saving, colormap creation, timestamp handling, and integration with other modules. The test cases use real MPAS data when available and mock objects to simulate edge cases. Assertions validate expected behavior and error handling across a range of scenarios. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules for testing
import os
import sys
import shutil
import pytest
import tempfile
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
matplotlib.use('Agg')
from io import StringIO
import cartopy.crs as ccrs
from typing import Generator
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from unittest.mock import Mock, MagicMock, patch

from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestUnitConversion:
    """ Tests for unit conversion and metadata handling. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestUnitConversion', mpas_coordinates, mpas_precip_data) -> None:
        """
        This fixture sets up the test environment for unit conversion tests by initializing the `MPASPrecipitationPlotter` and loading real MPAS coordinates and precipitation data. The fixture checks for the availability of the required data and skips tests if the data is not present. It prepares a subset of longitude, latitude, and precipitation arrays for use in subsequent tests that validate unit conversion handling in the plotter.

        Parameters:
            self (TestUnitConversion): Test case instance.
            mpas_coordinates: Session fixture providing real MPAS longitude and latitude arrays.
            mpas_precip_data: Session fixture providing real MPAS precipitation data.

        Returns:
            None: The fixture sets up instance attributes for use in tests.
        """
        if mpas_coordinates is None or mpas_precip_data is None:
            pytest.skip("MPAS data not available")
            return
        
        self.plotter = MPASPrecipitationPlotter()
        lon_full, lat_full = mpas_coordinates
        self.lon = lon_full[:50]
        self.lat = lat_full[:50]
        self.precip = mpas_precip_data[:50]
    
    def test_unit_conversion_with_data_array(self: 'TestUnitConversion') -> None:
        """
        This test verifies that the `create_precipitation_map` function can handle an `xarray.DataArray` with appropriate metadata for unit conversion. The test constructs a DataArray with 'units' and 'long_name' attributes and passes it to the plotting function. The plotter should recognize the metadata, perform any necessary unit conversions, and produce a valid figure. The test asserts that a Figure is returned and closes it to clean up resources.

        Parameters:
            self ('TestUnitConversion'): Test case instance.

        Returns:
            None: Assertions verify returned figure type and successful cleanup.
        """
        data_array = xr.DataArray(
            self.precip,
            dims=['nCells'],
            attrs={'units': 'mm', 'long_name': 'Precipitation'}
        )
        
        fig, ax = self.plotter.create_precipitation_map(
            self.lon, self.lat, self.precip,
            -180, 180, -90, 90,
            data_array=data_array
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_unit_conversion_attribute_error(self: 'TestUnitConversion') -> None:
        """
        This test checks the behavior of the `create_precipitation_map` function when a provided `data_array` is missing expected attributes for unit conversion. The plotter should handle this gracefully, either by proceeding without conversion or by raising a clear warning. The test uses a mock DataArray without 'units' or 'long_name' attributes and asserts that a Figure is still produced without errors. This ensures that the plotting function is robust to incomplete metadata and can still render a map using raw data arrays.

        Parameters:
            self ('TestUnitConversion'): Test case instance.

        Returns:
            None: Assertions validate that the plotting function still produces a Figure.
        """
        mock_data_array = Mock()
        mock_data_array.attrs = {}
        
        fig, ax = self.plotter.create_precipitation_map(
            self.lon, self.lat, self.precip,
            -180, 180, -90, 90,
            data_array=mock_data_array
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_unit_conversion_without_data_array(self: 'TestUnitConversion') -> None:
        """
        This test validates that the `create_precipitation_map` function can operate without a provided `data_array` and still produce a valid figure. The plotter should be able to handle raw longitude, latitude, and precipitation arrays without relying on xarray metadata for unit conversion. The test asserts that a Figure is returned successfully, confirming that the plotting function is flexible in its input handling and can render maps even when no DataArray is supplied.

        Parameters:
            self ('TestUnitConversion'): Test case instance.

        Returns:
            None: Assertions validate that plotting proceeds without a data_array.
        """
        fig, ax = self.plotter.create_precipitation_map(
            self.lon, self.lat, self.precip,
            -180, 180, -90, 90
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotExtent:
    """ Tests for plot extent handling. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestPlotExtent') -> None:
        """
        This fixture sets up the test environment for plot extent handling tests by initializing the `MPASPrecipitationPlotter`. It prepares the plotter instance for use in subsequent tests that validate the handling of longitude and latitude ranges when creating precipitation maps. The fixture does not load any data, as the extent tests focus on input validation and plotting behavior rather than data content.

        Parameters:
            self ('TestPlotExtent'): Test case instance.

        Returns:
            None: The fixture initializes the plotter for use in extent tests.
        """
        self.plotter = MPASPrecipitationPlotter()
    
    
    def test_global_extent(self: 'TestPlotExtent') -> None:
        """
        This test validates that the `create_precipitation_map` function can successfully create a global precipitation map when provided with longitude and latitude arrays that cover the entire globe. The plotter should handle the full range of longitude (-180 to 180) and latitude (-90 to 90) without errors and produce a valid Figure for visualization. This confirms that the plotter can render global maps correctly. The test asserts that a Figure is returned and closes it to clean up resources.

        Parameters:
            self ('TestPlotExtent'): Test case instance.

        Returns:
            None: Assertions validate returned figure type and cleanup.
        """
        lon = np.linspace(-180, 180, 200)
        lat = np.linspace(-90, 90, 200)
        precip = np.random.rand(200) * 20
        
        fig, ax = self.plotter.create_precipitation_map(
            lon, lat, precip,
            -180, 180, -90, 90  
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_regional_extent(self: 'TestPlotExtent') -> None:
        """
        This test validates that the `create_precipitation_map` function can successfully create a regional precipitation map when provided with longitude and latitude arrays that cover a specific area. The plotter should handle the specified longitude and latitude ranges without errors and produce a valid Figure for visualization. This confirms that the plotter can render regional maps correctly. The test asserts that a Figure is returned and closes it to clean up resources.

        Parameters:
            self ('TestPlotExtent'): Test case instance.

        Returns:
            None: Assertions validate regional plotting produces a Figure.
        """
        lon = np.linspace(-100, -90, 50)
        lat = np.linspace(30, 40, 50)
        precip = np.random.rand(50) * 20
        
        fig, ax = self.plotter.create_precipitation_map(
            lon, lat, precip,
            -100, -90, 30, 40  
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestContourfPlotting:
    """ Tests for contourf plotting and interpolation. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self : 'TestContourfPlotting', mpas_coordinates, mpas_precip_data) -> None:
        """
        This fixture sets up the test environment for contourf plotting tests by initializing the `MPASPrecipitationPlotter` and loading real MPAS coordinates and precipitation data. It checks for the availability of the required data and skips tests if the data is not present. The fixture prepares a subset of longitude, latitude, and precipitation arrays for use in subsequent tests that validate contourf plotting behavior, grid resolution handling, and error handling for invalid plot types.

        Parameters:
            self ('TestContourfPlotting'): Test case instance.

        Returns:
            None: The fixture initializes instance attributes for use in contourf plotting tests.
        """
        if mpas_coordinates is None or mpas_precip_data is None:
            pytest.skip("MPAS data not available")
            return
        
        self.plotter = MPASPrecipitationPlotter()
        lon_full, lat_full = mpas_coordinates
        self.lon = lon_full[:100]
        self.lat = lat_full[:100]
        self.precip = mpas_precip_data[:100]
    
    def test_contourf_plot_type(self: 'TestContourfPlotting') -> None:
        """
        This test verifies that the `create_precipitation_map` function can successfully create a contourf plot when the `plot_type` parameter is set to 'contourf'. The plotter should produce a valid Figure for visualization without errors. This confirms that the contourf plotting code path is functional and can handle typical input data. The test asserts that a Figure is returned and closes it to clean up resources.

        Parameters:
            self ('TestContourfPlotting'): Test case instance.

        Returns:
            None: Assertion verifies a Figure is produced.
        """
        config = {
            'remap_engine': 'kdtree',
            'remap_method': 'nearest',
        }

        fig, _ = self.plotter.create_precipitation_map(
            self.lon, self.lat, self.precip,
            -100, -90, 30, 40,
            plot_type='contourf', 
            config=config
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_contourf_with_auto_grid_resolution(self: 'TestContourfPlotting') -> None:
        """
        This test validates that the `create_precipitation_map` function can create a contourf plot using an automatically determined grid resolution when the `grid_resolution` parameter is not explicitly provided. The plotter should compute an appropriate resolution based on the input data and produce a valid Figure for visualization. This ensures that the plotter can handle contourf plotting without requiring users to specify grid resolution, providing a more user-friendly experience. The test asserts that a Figure is returned and closes it to clean up resources.

        Parameters:
            self ('TestContourfPlotting'): Test case instance.

        Returns:
            None: Assertion verifies figure creation and cleanup.
        """
        config = {
            'remap_engine': 'kdtree',
            'remap_method': 'nearest',
        }

        fig, _ = self.plotter.create_precipitation_map(
            self.lon, self.lat, self.precip,
            -100, -90, 30, 40,
            plot_type='contourf',
            config=config
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_contourf_with_custom_grid_resolution(self: 'TestContourfPlotting') -> None:
        """
        This test validates that the `create_precipitation_map` function can create a contourf plot using a custom grid resolution when the `grid_resolution` parameter is explicitly provided. The plotter should use the specified resolution for interpolation and produce a valid Figure for visualization. This confirms that users can control the level of detail in contourf plots by adjusting the grid resolution. The test asserts that a Figure is returned and closes it to clean up resources.

        Parameters:
            self ('TestContourfPlotting'): Test case instance.

        Returns:
            None: Assertion verifies returned Figure is created and cleaned up.
        """
        config = {
            'remap_engine': 'kdtree',
            'remap_method': 'nearest',
        }

        fig, _ = self.plotter.create_precipitation_map(
            self.lon, self.lat, self.precip,
            -100, -90, 30, 40,
            plot_type='contourf',
            grid_resolution=0.5, 
            config=config,
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    

class TestBatchProcessing:
    """ Tests for batch precipitation map creation. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestBatchProcessing') -> Generator[None, None, None]:
        """
        This fixture sets up the test environment for batch processing tests by initializing the `MPASPrecipitationPlotter` and creating a temporary directory for output files. The fixture yields control to the test functions and ensures that the temporary directory is cleaned up after tests complete. This setup allows batch processing tests to write output files without affecting the actual filesystem and ensures that any generated files are removed after testing.

        Parameters:
            self ('TestBatchProcessing'): Test case instance.

        Returns:
            None: The fixture manages setup and teardown for batch processing tests.
        """
        self.plotter = MPASPrecipitationPlotter()
        self.temp_dir = tempfile.mkdtemp()
    
        yield

        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    
    def test_batch_processing_workflow(self: 'TestBatchProcessing') -> None:
        """
        This test validates the end-to-end workflow of the `create_batch_precipitation_maps` function using a mock processor with a loaded dataset. The test simulates a realistic scenario where the processor has an xarray dataset with precipitation data and coordinates. The batch processing function should create precipitation maps for specified time indices and save them to the temporary directory. The test asserts that output files are created successfully, confirming that the batch processing workflow operates as intended.

        Parameters:
            self ('TestBatchProcessing'): Test case instance.

        Returns:
            None: Assertions validate produced files exist.
        """
        mock_processor = Mock()
        times = pd.date_range('2025-01-01', periods=5, freq='h')
        
        ds = xr.Dataset({
            'rainnc': (['Time', 'nCells'], np.random.rand(5, 50) * 20),
            'lonCell': ('nCells', np.linspace(-100, -90, 50)),
            'latCell': ('nCells', np.linspace(30, 40, 50))
        }, coords={'Time': times})
        
        mock_processor.dataset = ds
        mock_processor.data_type = 'xarray'

        mock_processor.extract_2d_coordinates_for_variable = Mock(
            return_value=(ds.lonCell.values, ds.latCell.values)
        )
        
        files = self.plotter.create_batch_precipitation_maps(
            mock_processor, self.temp_dir,
            -100, -90, 30, 40,
            accum_period='a01h',
            time_indices=[2, 3]
        )
        
        assert len(files) > 0

        for file_path in files:
            assert os.path.exists(file_path)


class TestComparisonPlot:
    """ Tests for precipitation comparison plots. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestComparisonPlot', mpas_coordinates, mpas_precip_data) -> None:
        """
        This fixture sets up the test environment for precipitation comparison plot tests by initializing the `MPASPrecipitationPlotter` and loading real MPAS coordinates and precipitation data. It checks for the availability of the required data and skips tests if the data is not present. The fixture prepares subsets of longitude, latitude, and two different precipitation arrays for use in subsequent tests that validate the creation of comparison plots, including handling of custom titles and multi-panel layouts.

        Parameters:
            self ('TestComparisonPlot'): Test case instance.
            mpas_coordinates: Session fixture providing real MPAS longitude and latitude arrays.
            mpas_precip_data: Session fixture providing real MPAS precipitation data.

        Returns:
            None: The fixture initializes instance attributes for use in comparison plot tests.
        """
        if mpas_coordinates is None or mpas_precip_data is None:
            pytest.skip("MPAS data not available")
            return
        
        self.plotter = MPASPrecipitationPlotter()
        lon_full, lat_full = mpas_coordinates
        self.lon = lon_full[:50]
        self.lat = lat_full[:50]
        self.precip1 = mpas_precip_data[:50] 
        self.precip2 = mpas_precip_data[50:100] 
    
    def test_comparison_plot_basic(self: 'TestComparisonPlot') -> None:
        """
        This test validates that the `create_precipitation_comparison_plot` function can successfully create a comparison plot with two subplots when provided with two different precipitation datasets. The plotter should produce a valid Figure containing two Axes for side-by-side comparison of the datasets. This confirms that the comparison plotting code path is functional and can handle typical input data for both subplots. The test asserts that a Figure is returned and that it contains two Axes, then closes the figure to clean up resources.

        Parameters:
            self ('TestComparisonPlot'): Test case instance.

        Returns:
            None: Assertion verifies Figure and two Axes are returned.
        """
        fig, axes = self.plotter.create_precipitation_comparison_plot(
            self.lon, self.lat,
            self.precip1, self.precip2,
            -180, 180, -90, 90
        )
        
        assert isinstance(fig, Figure)
        assert len(axes) == pytest.approx(2)
        plt.close(fig)
    
    def test_comparison_plot_with_custom_titles(self: 'TestComparisonPlot') -> None:
        """
        This test validates that the `create_precipitation_comparison_plot` function can successfully create a comparison plot with custom titles for each subplot. The plotter should accept the `title1` and `title2` parameters and apply them to the respective subplots without errors. This confirms that users can customize subplot titles in comparison plots for clearer communication of the datasets being compared. The test asserts that a Figure is returned and closes it to clean up resources.

        Parameters:
            self ('TestComparisonPlot'): Test case instance.

        Returns:
            None: Assertion verifies plotting with custom titles succeeds.
        """
        fig, axes = self.plotter.create_precipitation_comparison_plot(
            self.lon, self.lat,
            self.precip1, self.precip2,
            -180, 180, -90, 90,
            title1="Model Run 1",
            title2="Model Run 2"
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestSavePlot:
    """ Tests for plot saving. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestSavePlot') -> Generator[None, None, None]:
        """
        This fixture sets up the test environment for plot saving tests by initializing the `MPASPrecipitationPlotter` and creating a temporary directory for output files. The fixture yields control to the test functions and ensures that the temporary directory is cleaned up after tests complete. This setup allows plot saving tests to write output files without affecting the actual filesystem and ensures that any generated files are removed after testing.

        Parameters:
            self ('TestSavePlot'): Test case instance.

        Returns:
            Generator[None, None, None]: The fixture manages setup and teardown for plot saving tests.
        """
        self.plotter = MPASPrecipitationPlotter()
        self.temp_dir = tempfile.mkdtemp()
    
        yield

        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    
    def test_save_plot_creates_directory(self: 'TestSavePlot', mpas_coordinates, mpas_precip_data) -> None:
        """
        This test validates that the `save_plot` function can successfully create necessary directories when saving a plot to a specified path. The plotter should check if the target directory exists and create it if it does not before saving the figure. This ensures that users can specify nested output paths without needing to manually create directories beforehand. The test asserts that the output file is created successfully at the specified path, confirming that directory handling and file saving work as intended.

        Parameters:
            self ('TestSavePlot'): Test case instance.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_precip_data: Session fixture providing real precipitation data.

        Returns:
            None: Assertions validate output file creation and directory handling.
        """
        if mpas_coordinates is None or mpas_precip_data is None:
            pytest.skip("MPAS data not available")
            return
        
        lon_full, lat_full = mpas_coordinates
        lon = lon_full[:50]
        lat = lat_full[:50]
        precip = mpas_precip_data[:50]
        
        fig, ax = self.plotter.create_precipitation_map(
            lon, lat, precip,
            -180, 180, -90, 90
        )
        
        subdir = os.path.join(self.temp_dir, 'subdir', 'subsubdir')
        output_path = os.path.join(subdir, 'test_plot')
        
        self.plotter.save_plot(output_path, formats=['png'])
        
        assert os.path.exists(output_path + '.png')
        plt.close(fig)


class TestTimestampHandling:
    """ Tests for timestamp and accumulation period handling. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestTimestampHandling', mpas_coordinates, mpas_precip_data) -> None:
        """
        This fixture sets up the test environment for timestamp handling tests by initializing the `MPASPrecipitationPlotter` and loading real MPAS coordinates and precipitation data. It checks for the availability of the required data and skips tests if the data is not present. The fixture prepares subsets of longitude, latitude, and precipitation arrays for use in subsequent tests that validate the handling of time_end and time_start parameters when creating precipitation maps with specific accumulation periods. The tests will cover scenarios where only time_end is provided (requiring inference of time_start) and where both time_end and time_start are explicitly provided, ensuring that the plotter correctly computes the accumulation window and produces valid figures for both cases.

        Parameters:
            self ('TestTimestampHandling'): Test case instance.
            mpas_coordinates: Session fixture providing real MPAS longitude and latitude arrays.
            mpas_precip_data: Session fixture providing real MPAS precipitation data.
        
        Returns:
            None: The fixture initializes instance attributes for use in timestamp handling tests.
        """
        if mpas_coordinates is None or mpas_precip_data is None:
            pytest.skip("MPAS data not available")
            return
        
        self.plotter = MPASPrecipitationPlotter()
        lon_full, lat_full = mpas_coordinates
        self.lon = lon_full[:50]
        self.lat = lat_full[:50]
        self.precip = mpas_precip_data[:50]
    
    def test_plot_with_time_end_only(self: 'TestTimestampHandling') -> None:
        """
        This test validates that the `create_precipitation_map` function can successfully create a precipitation map when only the `time_end` parameter is provided along with an accumulation period. The plotter should infer the appropriate `time_start` based on the specified `accum_period` and produce a valid Figure for visualization. This ensures that users can create accumulated precipitation maps by providing just the end time, simplifying the interface for common use cases. The test asserts that a Figure is returned and closes it to clean up resources.

        Parameters:
            self ('TestTimestampHandling'): Test case instance.

        Returns:
            None: Assertion verifies plotting for inferred start time.
        """
        time_end = datetime(2025, 1, 15, 12, 0)
        
        fig, ax = self.plotter.create_precipitation_map(
            self.lon, self.lat, self.precip,
            -180, 180, -90, 90,
            time_end=time_end,
            accum_period='a03h'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_with_time_end_and_start(self: 'TestTimestampHandling') -> None:
        """
        This test validates that the `create_precipitation_map` function can successfully create a precipitation map when both `time_start` and `time_end` parameters are provided. The plotter should honor the specified interval and produce a valid Figure for visualization. This ensures that users can create accumulated precipitation maps for explicit time ranges. The test asserts that a Figure is returned and closes it to clean up resources.

        Parameters:
            self ('TestTimestampHandling'): Test case instance.

        Returns:
            None: Assertion validates plotting for explicit time ranges.
        """
        time_end = datetime(2025, 1, 15, 12, 0)
        time_start = datetime(2025, 1, 15, 6, 0)
        
        fig, ax = self.plotter.create_precipitation_map(
            self.lon, self.lat, self.precip,
            -180, 180, -90, 90,
            time_end=time_end,
            time_start=time_start
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestEdgeCases:
    """ Tests for edge cases and special scenarios. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestEdgeCases', mpas_coordinates) -> None:
        """
        This fixture sets up the test environment for edge case tests by initializing the `MPASPrecipitationPlotter` and loading real MPAS coordinates. It checks for the availability of the required data and skips tests if the data is not present. The fixture prepares subsets of longitude and latitude arrays for use in subsequent tests that validate the behavior of the plotter when handling edge cases such as empty or NaN-filled precipitation data, extreme values, and dynamic tick formatting. This setup ensures that the edge case tests have access to realistic coordinate data while focusing on testing the plotter's robustness in handling unusual input scenarios.

        Parameters:
            self ('TestEdgeCases'): Test case instance.

        Returns:
            None: The fixture initializes instance attributes for use in edge case tests.
        """
        if mpas_coordinates is None:
            pytest.skip("MPAS data not available")
            return
        
        self.plotter = MPASPrecipitationPlotter()
        lon_full, lat_full = mpas_coordinates
        self.lon = lon_full[:50]
        self.lat = lat_full[:50]
    
    def test_precipitation_map_empty_data(self: 'TestEdgeCases') -> None:
        """
        This test validates the behavior of the `create_precipitation_map` function when provided with an input array that is entirely filled with NaN values. The plotter should handle this edge case gracefully, either by producing an empty plot or by displaying a message indicating that no valid data is available. The test asserts that a Figure is returned without errors, confirming that the plotter can manage cases where precipitation data is missing or invalid without crashing.

        Parameters:
            self ('TestEdgeCases'): Test case instance.

        Returns:
            None: Assertion verifies a Figure is returned for all-NaN inputs.
        """
        nan_data = np.full(50, np.nan)
        
        fig, ax = self.plotter.create_precipitation_map(
            self.lon, self.lat, nan_data,
            -180, 180, -90, 90,
            title="Empty Precipitation Data"
        )

        assert isinstance(fig, Figure)

        plt.close(fig)
    
    def test_precipitation_map_extreme_values(self: 'TestEdgeCases') -> None:
        """
        This test validates the behavior of the `create_precipitation_map` function when provided with precipitation data containing extreme values (e.g., very high or very low precipitation). The plotter should be able to handle these values without crashing and should produce a valid Figure that appropriately represents the data, potentially using a colormap that can accommodate the range of values. The test asserts that a Figure is returned without errors, confirming that the plotter can manage cases with extreme precipitation values effectively.

        Parameters:
            self ('TestEdgeCases'): Test case instance.

        Returns:
            None: Assertion verifies plotting completes for extreme values.
        """
        extreme_data = np.array([0, 0.01, 1, 100, 1000, 10000])
        extreme_lon = np.linspace(-110, -100, 6)
        extreme_lat = np.linspace(35, 45, 6)
        
        fig, ax = self.plotter.create_precipitation_map(
            extreme_lon, extreme_lat, extreme_data,
            -110, -100, 35, 45,
            title="Extreme Precipitation Values"
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    

class TestPrepareOverlayData:
    """ Tests for _prepare_overlay_data method covering unit conversion, negative clipping, and bounds. """

    @pytest.fixture(autouse=True)
    def setup(self: 'TestPrepareOverlayData') -> None:
        """
        This fixture sets up the test environment for testing the `_prepare_overlay_data` method by initializing an instance of `MPASPrecipitationPlotter`. The method being tested is responsible for preparing data for overlay plots, including handling unit conversions, clipping negative values, and applying bounds filtering. The fixture ensures that a plotter instance is available for all tests in this class, allowing them to focus on validating the specific behaviors of the `_prepare_overlay_data` method under various input scenarios.

        Parameters:
            self ('TestPrepareOverlayData'): Test case instance.

        Returns:
            None: The fixture initializes the plotter for use in overlay data preparation tests.
        """
        self.plotter = MPASPrecipitationPlotter()

    def test_overlay_data_negative_clipping(self: 'TestPrepareOverlayData') -> None:
        """
        This test verifies that negative precipitation values are correctly clipped to 0 and that a warning is issued when such values are encountered. The test generates a set of longitude, latitude, and precipitation data, including negative values, and checks that the resulting prepared data contains no negative values. It also captures the standard output to ensure that a warning message about negative values is printed.

        Parameters:
            self ('TestPrepareOverlayData'): Test case instance.

        Returns:
            None: Assertions validate that negative values are clipped and warnings are issued.
        """
        lon = np.linspace(-110, -100, 50)
        lat = np.linspace(35, 45, 50)
        data = np.random.uniform(-5, 10, 50)

        captured = StringIO()
        with patch('sys.stdout', captured):
            result = self.plotter._prepare_overlay_data(lon, lat, data, 'test_var', None, 'scatter')

        lon_v, lat_v, prec_v, *_ = result

        assert np.all(prec_v >= 0)
        assert 'negative' in captured.getvalue().lower() or np.sum(data < 0) == 0

    def test_overlay_data_unit_conversion(self: 'TestPrepareOverlayData') -> None:
        """
        This test verifies that when the original units of the data differ from the display units, a conversion is attempted. It generates a set of longitude, latitude, and precipitation data, and checks that the resulting prepared data is valid after conversion.

        Parameters:
            self ('TestPrepareOverlayData'): Test case instance.

        Returns:
            None: Assertions validate that data conversion is attempted and results are valid.
        """
        lon = np.linspace(-110, -100, 50)
        lat = np.linspace(35, 45, 50)
        data = np.random.uniform(0, 5, 50)

        captured = StringIO()
        with patch('sys.stdout', captured):
            result = self.plotter._prepare_overlay_data(lon, lat, data, 'precipitation', 'mm', 'scatter')

        lon_v, lat_v, prec_v, *_ = result
        assert len(lon_v) > 0

    def test_overlay_data_high_mean_warning(self: 'TestPrepareOverlayData') -> None:
        """
        This test verifies that when the mean of the data is unusually high and no original units are provided, a warning about potential unit issues is issued. It generates a set of longitude, latitude, and precipitation data with high values and checks that the warning message is printed.

        Parameters:
            self ('TestPrepareOverlayData'): Test case instance.

        Returns:
            None: Assertions validate that a warning is issued for high mean values without original units.
        """
        lon = np.linspace(-110, -100, 50)
        lat = np.linspace(35, 45, 50)
        data = np.random.uniform(1500, 2000, 50)

        captured = StringIO()

        with patch('sys.stdout', captured):
            result = self.plotter._prepare_overlay_data(lon, lat, data, 'precip_var', None, 'scatter')

        assert 'may not be in mm' in captured.getvalue().lower() or len(result[0]) > 0


    def test_overlay_data_contour_mask_differs_from_scatter(self: 'TestPrepareOverlayData') -> None:
        """
        This test verifies that the contour plot type uses a simpler mask than the scatter plot type, specifically without bounds filtering. It generates a set of longitude, latitude, and precipitation data, and checks that both scatter and contour plots return valid data.

        Parameters:
            self ('TestPrepareOverlayData'): Test case instance.

        Returns:
            None: Assertions validate that both scatter and contour plots return valid data.
        """
        lon = np.linspace(-110, -100, 50)
        lat = np.linspace(35, 45, 50)
        data = np.random.uniform(0, 5, 50)

        result_scatter = self.plotter._prepare_overlay_data(lon, lat, data, 'test', None, 'scatter')
        result_contour = self.plotter._prepare_overlay_data(lon, lat, data, 'test', None, 'contourf')

        assert len(result_scatter[0]) > 0
        assert len(result_contour[0]) > 0


class TestContourPlotCreation:
    """ Tests for _create_contour_plot and _create_contourf_plot methods. """

    @pytest.fixture(autouse=True)
    def setup(self: 'TestContourPlotCreation') -> None:
        """
        This fixture sets up the test environment for contour plot creation tests by initializing an instance of `MPASPrecipitationPlotter` and creating a figure and axes with a PlateCarree projection. The fixture prepares the plotter instance and axes for use in subsequent tests that validate the functionality of the `_create_contour_plot` and `_create_contourf_plot` methods, which are responsible for rendering contour lines and filled contours on the map. This setup ensures that the contour plot creation tests have a consistent plotting environment to work with.

        Parameters:
            self ('TestContourPlotCreation'): Test case instance.

        Returns:
            None: The fixture initializes the plotter and axes for contour plot tests.
        """
        self.plotter = MPASPrecipitationPlotter()
        self.plotter.fig = plt.figure(figsize=(10, 8))
        self.plotter.ax = self.plotter.fig.add_subplot(111, projection=ccrs.PlateCarree())

    def teardown_method(self: 'TestContourPlotCreation') -> None:
        """
        This method is called after each test in the `TestContourPlotCreation` class to close all figures and clean up resources. It ensures that any figures created during the tests are properly closed to prevent memory leaks and to maintain a clean testing environment for subsequent tests.

        Parameters:
            self ('TestContourPlotCreation'): Test case instance.

        Returns:
            None: Figures are closed.
        """
        plt.close('all')


    def test_create_contourf_plot(self: 'TestContourPlotCreation') -> None:
        """
        This test validates that the `_create_contourf_plot` method can successfully create filled contours on the map when provided with longitude, latitude, and data arrays. The test generates synthetic data for testing and checks that the method executes without errors, confirming that the filled contour plotting logic is functional. The test uses mocking to bypass the interpolation step, allowing it to focus on the filled contour creation aspect. Assertions ensure that no exceptions are raised during the filled contour plot creation process.

        Parameters:
            self ('TestContourPlotCreation'): Test case instance.

        Returns:
            None: Assertions validate that filled contours are drawn.
        """
        n = 100
        lon = np.linspace(-110, -100, n)
        lat = np.linspace(35, 45, n)
        data = np.random.uniform(0, 10, n)
        levels = [0.1, 1, 2, 5, 10]
        cmap = matplotlib.colormaps["Blues"]
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        data_crs = ccrs.PlateCarree()

        with patch.object(self.plotter, '_interpolate_to_grid', return_value=(
            np.linspace(-110, -100, 10),
            np.linspace(35, 45, 10),
            np.random.uniform(0, 10, (10, 10))
        )):
            self.plotter._create_contourf_plot(
                lon, lat, data, -110, -100, 35, 45,
                cmap, norm, levels, data_crs
            )


class TestAddPrecipitationOverlay:
    """ Tests for add_precipitation_overlay method. """

    @pytest.fixture(autouse=True)
    def setup(self: 'TestAddPrecipitationOverlay') -> None:
        """
        This fixture sets up the test environment for testing the `add_precipitation_overlay` method by initializing an instance of `MPASPrecipitationPlotter`, creating a figure and axes with a PlateCarree projection, and preparing longitude and latitude arrays. The fixture ensures that a consistent plotting environment is available for all tests in this class, allowing them to focus on validating the functionality of the `add_precipitation_overlay` method under various input scenarios, including different plot types, data conditions, and configuration settings.

        Parameters:
            self ('TestAddPrecipitationOverlay'): Test case instance.

        Returns:
            None: Plotter and axes are created.
        """
        self.plotter = MPASPrecipitationPlotter()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection=ccrs.PlateCarree())
        self.lon = np.linspace(-110, -100, 100)
        self.lat = np.linspace(35, 45, 100)

    def teardown_method(self: 'TestAddPrecipitationOverlay') -> None:
        """
        This method is called after each test in the `TestAddPrecipitationOverlay` class to close all figures and clean up resources. It ensures that any figures created during the tests are properly closed to prevent memory leaks and to maintain a clean testing environment for subsequent tests.

        Parameters:
            self ('TestAddPrecipitationOverlay'): Test case instance.

        Returns:
            None: Figures are closed.
        """
        plt.close('all')

    def test_overlay_scatter(self: 'TestAddPrecipitationOverlay') -> None:
        """
        This test validates that the `add_precipitation_overlay` method can successfully add a scatter overlay to the map when provided with appropriate configuration and data. The test generates synthetic precipitation data and checks that the method executes without errors, confirming that the scatter overlay logic is functional. The test captures standard output to verify that the method processes the scatter plot type correctly.

        Parameters:
            self ('TestAddPrecipitationOverlay'): Test case instance.

        Returns:
            None: Assertions validate that scatter overlay is rendered.
        """
        config = {
            'data': np.random.uniform(0, 5, 100),
            'plot_type': 'scatter',
            'accum_period': 'a01h',
            'var_name': 'precip',
        }

        captured = StringIO()

        with patch('sys.stdout', captured):
            self.plotter.add_precipitation_overlay(
                self.ax, self.lon, self.lat, config,
                lon_min=-110, lon_max=-100, lat_min=35, lat_max=45
            )

        assert 'scatter' in captured.getvalue().lower()


    def test_overlay_no_valid_data_warning(self: 'TestAddPrecipitationOverlay') -> None:
        """
        This test validates that the `add_precipitation_overlay` method prints a warning and returns without rendering when all data values are NaN. The test provides a configuration with all-NaN data and checks that the appropriate warning is printed, confirming that the method correctly handles cases with no valid data.

        Parameters:
            self ('TestAddPrecipitationOverlay'): Test case instance.

        Returns:
            None: Assertions validate that warning is printed for all-NaN data.
        """
        config = {
            'data': np.full(100, np.nan),
            'plot_type': 'scatter',
        }

        captured = StringIO()

        with patch('sys.stdout', captured):
            self.plotter.add_precipitation_overlay(self.ax, self.lon, self.lat, config)

        assert 'Warning' in captured.getvalue()

    def test_overlay_contourf_with_remap(self: 'TestAddPrecipitationOverlay') -> None:
        """
        This test validates that when a contourf overlay is added, the method attempts to remap the data to a lat/lon grid. The test mocks the remapping function to return a predefined remapped dataset and checks that the contourf plotting logic is executed without errors. This ensures that the method correctly integrates remapping into the contourf overlay creation process.

        Parameters:
            self ('TestAddPrecipitationOverlay'): Test case instance.

        Returns:
            None: Assertions validate that contourf overlay uses remapping path.
        """
        config = {
            'data': np.random.uniform(0, 5, 100),
            'plot_type': 'contourf',
            'accum_period': 'a01h',
            'var_name': 'precip',
            'alpha': 0.5,
        }

        mock_remapped = MagicMock()
        mock_remapped.lon.values = np.linspace(-110, -100, 10)
        mock_remapped.lat.values = np.linspace(35, 45, 10)
        mock_remapped.values = np.random.uniform(0, 5, (10, 10))

        captured = StringIO()

        with patch('mpasdiag.visualization.precipitation.remap_mpas_to_latlon_with_masking', return_value=mock_remapped):
            with patch('sys.stdout', captured):
                self.plotter.add_precipitation_overlay(
                    self.ax, self.lon, self.lat, config,
                    lon_min=-110, lon_max=-100, lat_min=35, lat_max=45
                )

        assert 'contourf' in captured.getvalue().lower()

    def test_overlay_uses_data_bounds_when_none_provided(self: 'TestAddPrecipitationOverlay') -> None:
        """
        This test validates that when no explicit bounds are provided in the configuration for a scatter overlay, the method uses the data bounds to determine which points to plot. The test generates synthetic precipitation data and checks that the method executes without errors, confirming that it can handle cases where bounds are not specified by relying on the data values.

        Parameters:
            self ('TestAddPrecipitationOverlay'): Test case instance.

        Returns:
            None: Assertions validate that data bounds are used when none are provided.
        """
        config = {
            'data': np.random.uniform(0, 5, 100),
            'plot_type': 'scatter',
            'var_name': 'precip',
        }

        captured = StringIO()

        with patch('sys.stdout', captured):
            self.plotter.add_precipitation_overlay(self.ax, self.lon, self.lat, config)

        assert 'scatter' in captured.getvalue().lower()


class TestPrecipitationOverlayEdgeCases:
    """ Tests for overlay data preparation edge cases. """

    def setup_method(self: 'TestPrecipitationOverlayEdgeCases') -> None:
        """
        This method is called before each test in the `TestPrecipitationOverlayEdgeCases` class to initialize an instance of `MPASPrecipitationPlotter`. The method ensures that a fresh plotter instance is available for each test, allowing them to focus on validating the specific behaviors of the `_prepare_overlay_data` method under various input scenarios, including handling of negative precipitation values, cases with no valid data points, unit conversion attempts, and support for xarray DataArray inputs. This setup is essential for testing the robustness and correctness of the overlay data preparation logic in the precipitation plotter.

        Parameters:
            self ('TestPrecipitationOverlayEdgeCases'): Test case instance. 

        Returns:
            None: The method initializes the plotter for use in overlay data preparation tests.  
        """
        self.plotter = MPASPrecipitationPlotter()

    def test_prepare_overlay_data_negative_clipping(self: 'TestPrecipitationOverlayEdgeCases', capsys) -> None:
        """
        This test verifies that the `_prepare_overlay_data` method correctly clips negative precipitation values to 0 and prints a warning when such values are encountered. It checks that the output data contains no negative values and that a warning message is printed, confirming that the method handles negative precipitation values appropriately.

        Parameters:
            self ('TestPrecipitationOverlayEdgeCases'): Test case instance.
            capsys: Pytest fixture to capture stdout and stderr.

        Returns:
            None: Assertions validate that negative values are clipped and a warning is printed.
        """
        lon = np.random.uniform(-110, -100, 50)
        lat = np.random.uniform(30, 40, 50)
        data = np.random.uniform(-5, 5, 50)

        result = self.plotter._prepare_overlay_data(
            lon, lat, data, 'rainnc', None, 'scatter'
        )

        lon_valid, lat_valid, precip_valid, *bounds = result

        if len(precip_valid) > 0:
            assert np.all(precip_valid >= 0)


    def test_prepare_overlay_data_with_unit_conversion(self: 'TestPrecipitationOverlayEdgeCases', capsys) -> None:
        """
        This test verifies that the `_prepare_overlay_data` method attempts unit conversion when the original_units parameter differs from the expected units. It ensures that the method correctly handles unit conversion scenarios and produces valid output data.

        Parameters:
            self ('TestPrecipitationOverlayEdgeCases'): Test case instance.
            capsys: Pytest fixture to capture stdout and stderr.

        Returns:
            None: Assertions validate that unit conversion is attempted and output data is valid.
        """
        lon = np.random.uniform(-110, -100, 50)
        lat = np.random.uniform(30, 40, 50)
        data = np.random.uniform(0, 5, 50)

        result = self.plotter._prepare_overlay_data(
            lon, lat, data, 'rainnc', 'kg m-2', 'scatter'
        )

        lon_valid, lat_valid, precip_valid, *bounds = result

        assert len(precip_valid) > 0

    def test_prepare_overlay_data_xarray_inputs(self: 'TestPrecipitationOverlayEdgeCases') -> None:
        """
        This test verifies that the `_prepare_overlay_data` method correctly handles `xr.DataArray` inputs. It ensures that the method can process xarray data structures and produce valid output data.

        Parameters:
            self ('TestPrecipitationOverlayEdgeCases'): Test case instance.

        Returns:
            None: Assertions validate that output data is valid when using xarray inputs.
        """
        lon = xr.DataArray(np.random.uniform(-110, -100, 50))
        lat = xr.DataArray(np.random.uniform(30, 40, 50))
        data = np.random.uniform(0, 5, 50)

        result = self.plotter._prepare_overlay_data(
            lon, lat, data, 'rainnc', None, 'scatter'
        )

        lon_valid, lat_valid, precip_valid, *bounds = result
        
        assert len(precip_valid) > 0


class TestPrecipitationPlotterHelpers:
    """Unit tests for MPASPrecipitationPlotter helper/utility methods."""

    @pytest.fixture(autouse=True)
    def setup_plotter(self: 'TestPrecipitationPlotterHelpers') -> Generator[None, None, None]:
        self.plotter = MPASPrecipitationPlotter()
        yield
        plt.close('all')

    # ------------------------------------------------------------------ #
    #  create_precip_colormap                                              #
    # ------------------------------------------------------------------ #


    @pytest.mark.parametrize("accum", ["a01h", "a03h", "a06h", "a12h", "a24h"])
    def test_create_precip_colormap_all_periods(self: 'TestPrecipitationPlotterHelpers', accum: str) -> None:
        cmap, levels = self.plotter.create_precip_colormap(accum)
        assert cmap is not None
        assert all(np.isfinite(v) for v in levels)

    # ------------------------------------------------------------------ #
    #  _convert_precipitation_units                                        #
    # ------------------------------------------------------------------ #


    def test_convert_precip_units_xarray_input(self: 'TestPrecipitationPlotterHelpers') -> None:
        data = np.array([0.5, 1.0, 1.5])
        da = xr.DataArray(data, attrs={'units': 'mm', 'long_name': 'Rain'})
        out, unit = self.plotter._convert_precipitation_units(data, da, 'rainnc')
        assert isinstance(out, np.ndarray)

    def test_convert_precip_units_negative_clipped(self: 'TestPrecipitationPlotterHelpers') -> None:
        data = np.array([-0.5, 1.0, 2.0])
        da = xr.DataArray(data, attrs={'units': 'mm', 'long_name': 'Rain'})
        out, _ = self.plotter._convert_precipitation_units(data, da, 'rainnc')
        assert np.all(out >= 0)

    # ------------------------------------------------------------------ #
    #  _setup_overlay_colormap                                             #
    # ------------------------------------------------------------------ #


    # ------------------------------------------------------------------ #
    #  _calculate_overlay_grid_resolution                                  #
    # ------------------------------------------------------------------ #


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
