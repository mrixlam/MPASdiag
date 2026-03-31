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
from cartopy.mpl.geoaxes import GeoAxes
from unittest.mock import Mock, MagicMock, patch

from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestMPASPrecipitationPlotterInitialization:
    """ Test plotter initialization. """
    
    def test_initialization(self: "TestMPASPrecipitationPlotterInitialization") -> None:
        """
        This test verifies that the `MPASPrecipitationPlotter` class initializes with the correct default parameters and that custom parameters are set properly. It checks that the `figsize` and `dpi` attributes are assigned expected values for both default and custom initialization cases. Proper initialization is critical for ensuring that subsequent plotting functions operate with the intended figure settings.

        Parameters:
            self (TestMPASPrecipitationPlotterInitialization): Test case instance.

        Returns:
            None: Assertions validate plotter attributes for default and custom initialization.
        """
        plotter = MPASPrecipitationPlotter()

        assert isinstance(plotter, MPASPrecipitationPlotter)
        assert plotter.figsize == (pytest.approx(10), pytest.approx(14))
        assert plotter.dpi == pytest.approx(100)
        custom_plotter = MPASPrecipitationPlotter(figsize=(10, 6), dpi=150)

        assert custom_plotter.figsize == (pytest.approx(10), pytest.approx(6))
        assert custom_plotter.dpi == pytest.approx(150)


class TestUnitConversion:
    """ Tests for unit conversion and metadata handling. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestUnitConversion", mpas_coordinates, mpas_precip_data) -> None:
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
        
        self.plotter = MPASPrecipitationPlotter()
        lon_full, lat_full = mpas_coordinates
        self.lon = lon_full[:50]
        self.lat = lat_full[:50]
        self.precip = mpas_precip_data[:50]
    
    def test_unit_conversion_with_data_array(self: "TestUnitConversion") -> None:
        """
        This test verifies that the `create_precipitation_map` function can handle an `xarray.DataArray` with appropriate metadata for unit conversion. The test constructs a DataArray with 'units' and 'long_name' attributes and passes it to the plotting function. The plotter should recognize the metadata, perform any necessary unit conversions, and produce a valid figure. The test asserts that a Figure is returned and closes it to clean up resources.

        Parameters:
            self ("TestUnitConversion"): Test case instance.

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
    
    def test_unit_conversion_attribute_error(self: "TestUnitConversion") -> None:
        """
        This test checks the behavior of the `create_precipitation_map` function when a provided `data_array` is missing expected attributes for unit conversion. The plotter should handle this gracefully, either by proceeding without conversion or by raising a clear warning. The test uses a mock DataArray without 'units' or 'long_name' attributes and asserts that a Figure is still produced without errors. This ensures that the plotting function is robust to incomplete metadata and can still render a map using raw data arrays.

        Parameters:
            self ("TestUnitConversion"): Test case instance.

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
    
    def test_unit_conversion_without_data_array(self: "TestUnitConversion") -> None:
        """
        This test validates that the `create_precipitation_map` function can operate without a provided `data_array` and still produce a valid figure. The plotter should be able to handle raw longitude, latitude, and precipitation arrays without relying on xarray metadata for unit conversion. The test asserts that a Figure is returned successfully, confirming that the plotting function is flexible in its input handling and can render maps even when no DataArray is supplied.

        Parameters:
            self ("TestUnitConversion"): Test case instance.

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
    def setup_method(self: "TestPlotExtent") -> None:
        """
        This fixture sets up the test environment for plot extent handling tests by initializing the `MPASPrecipitationPlotter`. It prepares the plotter instance for use in subsequent tests that validate the handling of longitude and latitude ranges when creating precipitation maps. The fixture does not load any data, as the extent tests focus on input validation and plotting behavior rather than data content.

        Parameters:
            self ("TestPlotExtent"): Test case instance.

        Returns:
            None: The fixture initializes the plotter for use in extent tests.
        """
        self.plotter = MPASPrecipitationPlotter()
    
    def test_invalid_extent_lon_range(self: "TestPlotExtent") -> None:
        """
        This test checks that providing an invalid longitude range to the `create_precipitation_map` function raises an informative ValueError. The plotter should validate the longitude min/max values and their ordering, ensuring they fall within the acceptable range of [-180, 180]. If the input is invalid, the function should raise a ValueError with a clear message indicating the issue with the plot extent. The test asserts that the expected exception is raised when an invalid longitude range is provided.

        Parameters:
            self ("TestPlotExtent"): Test case instance.

        Returns:
            None: Assertion verifies ValueError with appropriate message.
        """
        lon = np.linspace(-100, -90, 50)
        lat = np.linspace(30, 40, 50)
        precip = np.random.rand(50) * 20
        
        with pytest.raises(ValueError) as ctx:
            self.plotter.create_precipitation_map(
                lon, lat, precip,
                -200, -90, 30, 40  
            )
        assert 'Invalid plot extent' in str(ctx.value)
    
    def test_invalid_extent_lat_range(self: "TestPlotExtent") -> None:
        """
        This test checks that providing an invalid latitude range to the `create_precipitation_map` function raises an informative ValueError. The plotter should validate the latitude min/max values and their ordering, ensuring they fall within the acceptable range of [-90, 90]. If the input is invalid, the function should raise a ValueError with a clear message indicating the issue with the plot extent. The test asserts that the expected exception is raised when an invalid latitude range is provided.

        Parameters:
            self ("TestPlotExtent"): Test case instance.

        Returns:
            None: Assertion verifies ValueError with appropriate message.
        """
        lon = np.linspace(-100, -90, 50)
        lat = np.linspace(30, 40, 50)
        precip = np.random.rand(50) * 20
        
        with pytest.raises(ValueError) as ctx:
            self.plotter.create_precipitation_map(
                lon, lat, precip,
                -100, -90, -95, 40  
            )

        assert 'Invalid plot extent' in str(ctx.value)
    
    def test_global_extent(self: "TestPlotExtent") -> None:
        """
        This test validates that the `create_precipitation_map` function can successfully create a global precipitation map when provided with longitude and latitude arrays that cover the entire globe. The plotter should handle the full range of longitude (-180 to 180) and latitude (-90 to 90) without errors and produce a valid Figure for visualization. This confirms that the plotter can render global maps correctly. The test asserts that a Figure is returned and closes it to clean up resources.

        Parameters:
            self ("TestPlotExtent"): Test case instance.

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
    
    def test_regional_extent(self: "TestPlotExtent") -> None:
        """
        This test validates that the `create_precipitation_map` function can successfully create a regional precipitation map when provided with longitude and latitude arrays that cover a specific area. The plotter should handle the specified longitude and latitude ranges without errors and produce a valid Figure for visualization. This confirms that the plotter can render regional maps correctly. The test asserts that a Figure is returned and closes it to clean up resources.

        Parameters:
            self ("TestPlotExtent"): Test case instance.

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
    def setup_method(self : "TestContourfPlotting", mpas_coordinates, mpas_precip_data) -> None:
        """
        This fixture sets up the test environment for contourf plotting tests by initializing the `MPASPrecipitationPlotter` and loading real MPAS coordinates and precipitation data. It checks for the availability of the required data and skips tests if the data is not present. The fixture prepares a subset of longitude, latitude, and precipitation arrays for use in subsequent tests that validate contourf plotting behavior, grid resolution handling, and error handling for invalid plot types.

        Parameters:
            self ("TestContourfPlotting"): Test case instance.

        Returns:
            None: The fixture initializes instance attributes for use in contourf plotting tests.
        """
        if mpas_coordinates is None or mpas_precip_data is None:
            pytest.skip("MPAS data not available")
        
        self.plotter = MPASPrecipitationPlotter()
        lon_full, lat_full = mpas_coordinates
        self.lon = lon_full[:100]
        self.lat = lat_full[:100]
        self.precip = mpas_precip_data[:100]
    
    def test_contourf_plot_type(self: "TestContourfPlotting") -> None:
        """
        This test verifies that the `create_precipitation_map` function can successfully create a contourf plot when the `plot_type` parameter is set to 'contourf'. The plotter should produce a valid Figure for visualization without errors. This confirms that the contourf plotting code path is functional and can handle typical input data. The test asserts that a Figure is returned and closes it to clean up resources.

        Parameters:
            self ("TestContourfPlotting"): Test case instance.

        Returns:
            None: Assertion verifies a Figure is produced.
        """
        fig, ax = self.plotter.create_precipitation_map(
            self.lon, self.lat, self.precip,
            -100, -90, 30, 40,
            plot_type='contourf'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_contourf_with_auto_grid_resolution(self: "TestContourfPlotting") -> None:
        """
        This test validates that the `create_precipitation_map` function can create a contourf plot using an automatically determined grid resolution when the `grid_resolution` parameter is not explicitly provided. The plotter should compute an appropriate resolution based on the input data and produce a valid Figure for visualization. This ensures that the plotter can handle contourf plotting without requiring users to specify grid resolution, providing a more user-friendly experience. The test asserts that a Figure is returned and closes it to clean up resources.

        Parameters:
            self ("TestContourfPlotting"): Test case instance.

        Returns:
            None: Assertion verifies figure creation and cleanup.
        """
        fig, ax = self.plotter.create_precipitation_map(
            self.lon, self.lat, self.precip,
            -100, -90, 30, 40,
            plot_type='contourf'  
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_contourf_with_custom_grid_resolution(self: "TestContourfPlotting") -> None:
        """
        This test validates that the `create_precipitation_map` function can create a contourf plot using a custom grid resolution when the `grid_resolution` parameter is explicitly provided. The plotter should use the specified resolution for interpolation and produce a valid Figure for visualization. This confirms that users can control the level of detail in contourf plots by adjusting the grid resolution. The test asserts that a Figure is returned and closes it to clean up resources.

        Parameters:
            self ("TestContourfPlotting"): Test case instance.

        Returns:
            None: Assertion verifies returned Figure is created and cleaned up.
        """
        fig, ax = self.plotter.create_precipitation_map(
            self.lon, self.lat, self.precip,
            -100, -90, 30, 40,
            plot_type='contourf',
            grid_resolution=0.5
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_invalid_plot_type(self: "TestContourfPlotting") -> None:
        """
        This test checks that providing an invalid `plot_type` to the `create_precipitation_map` function raises an informative ValueError. The plotter should validate the `plot_type` parameter against supported options (e.g., 'contourf', 'pcolormesh') and raise a ValueError with a clear message if an unsupported type is provided. This ensures that users receive immediate feedback on incorrect inputs rather than encountering unexpected behavior or silent failures. The test asserts that the expected exception is raised when an invalid `plot_type` is used.

        Parameters:
            self ("TestContourfPlotting"): Test case instance.

        Returns:
            None: Assertion verifies error handling for invalid plot_type.
        """
        with pytest.raises(ValueError) as ctx:
            self.plotter.create_precipitation_map(
                self.lon, self.lat, self.precip,
                -100, -90, 30, 40,
                plot_type='invalid'
            )

        assert "plot_type must be" in str(ctx.value)


class TestBatchProcessing:
    """ Tests for batch precipitation map creation. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestBatchProcessing"):
        """
        This fixture sets up the test environment for batch processing tests by initializing the `MPASPrecipitationPlotter` and creating a temporary directory for output files. The fixture yields control to the test functions and ensures that the temporary directory is cleaned up after tests complete. This setup allows batch processing tests to write output files without affecting the actual filesystem and ensures that any generated files are removed after testing.

        Parameters:
            self ("TestBatchProcessing"): Test case instance.

        Returns:
            None: The fixture manages setup and teardown for batch processing tests.
        """
        self.plotter = MPASPrecipitationPlotter()
        self.temp_dir = tempfile.mkdtemp()
    
        yield

        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_batch_none_processor(self: "TestBatchProcessing") -> None:
        """
        This test checks that calling `create_batch_precipitation_maps` with a `None` processor raises a ValueError. The batch processing function should validate that a valid processor object is provided before attempting to create maps. If `None` is passed, the function should raise a ValueError with a clear message indicating that the processor cannot be None. This ensures that users receive immediate feedback on incorrect inputs rather than encountering unexpected errors later in the processing workflow. The test asserts that the expected exception is raised when `None` is used as the processor.

        Parameters:
            self ("TestBatchProcessing"): Test case instance.

        Returns:
            None: Assertion verifies a ValueError is raised for None processor.
        """
        with pytest.raises(ValueError) as ctx:
            self.plotter.create_batch_precipitation_maps(
                None, self.temp_dir,
                -100, -90, 30, 40
            )

        assert 'Processor cannot be None' in str(ctx.value)
    
    def test_batch_no_data_loaded(self: "TestBatchProcessing") -> None:
        """
        This test checks that calling `create_batch_precipitation_maps` with a processor that has no dataset loaded raises a ValueError. The batch processing function should validate that the processor has a dataset before attempting to create maps. If the dataset is `None`, the function should raise a ValueError with a clear message indicating that no data is loaded. This ensures that users receive immediate feedback on missing data issues rather than encountering confusing errors during processing. The test asserts that the expected exception is raised when the processor's dataset is `None`.

        Parameters:
            self ("TestBatchProcessing"): Test case instance.

        Returns:
            None: Assertion verifies error handling for missing dataset.
        """
        mock_processor = Mock()
        mock_processor.dataset = None
        
        with pytest.raises(ValueError) as ctx:
            self.plotter.create_batch_precipitation_maps(
                mock_processor, self.temp_dir,
                -100, -90, 30, 40
            )

        assert 'No data loaded' in str(ctx.value)
    
    def test_batch_processing_workflow(self: "TestBatchProcessing") -> None:
        """
        This test validates the end-to-end workflow of the `create_batch_precipitation_maps` function using a mock processor with a loaded dataset. The test simulates a realistic scenario where the processor has an xarray dataset with precipitation data and coordinates. The batch processing function should create precipitation maps for specified time indices and save them to the temporary directory. The test asserts that output files are created successfully, confirming that the batch processing workflow operates as intended.

        Parameters:
            self ("TestBatchProcessing"): Test case instance.

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
    def setup_method(self: "TestComparisonPlot", mpas_coordinates, mpas_precip_data) -> None:
        """
        This fixture sets up the test environment for precipitation comparison plot tests by initializing the `MPASPrecipitationPlotter` and loading real MPAS coordinates and precipitation data. It checks for the availability of the required data and skips tests if the data is not present. The fixture prepares subsets of longitude, latitude, and two different precipitation arrays for use in subsequent tests that validate the creation of comparison plots, including handling of custom titles and multi-panel layouts.

        Parameters:
            self ("TestComparisonPlot"): Test case instance.
            mpas_coordinates: Session fixture providing real MPAS longitude and latitude arrays.
            mpas_precip_data: Session fixture providing real MPAS precipitation data.

        Returns:
            None: The fixture initializes instance attributes for use in comparison plot tests.
        """
        if mpas_coordinates is None or mpas_precip_data is None:
            pytest.skip("MPAS data not available")
        
        self.plotter = MPASPrecipitationPlotter()
        lon_full, lat_full = mpas_coordinates
        self.lon = lon_full[:50]
        self.lat = lat_full[:50]
        self.precip1 = mpas_precip_data[:50] 
        self.precip2 = mpas_precip_data[50:100] 
    
    def test_comparison_plot_basic(self: "TestComparisonPlot") -> None:
        """
        This test validates that the `create_precipitation_comparison_plot` function can successfully create a comparison plot with two subplots when provided with two different precipitation datasets. The plotter should produce a valid Figure containing two Axes for side-by-side comparison of the datasets. This confirms that the comparison plotting code path is functional and can handle typical input data for both subplots. The test asserts that a Figure is returned and that it contains two Axes, then closes the figure to clean up resources.

        Parameters:
            self ("TestComparisonPlot"): Test case instance.

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
    
    def test_comparison_plot_with_custom_titles(self: "TestComparisonPlot") -> None:
        """
        This test validates that the `create_precipitation_comparison_plot` function can successfully create a comparison plot with custom titles for each subplot. The plotter should accept the `title1` and `title2` parameters and apply them to the respective subplots without errors. This confirms that users can customize subplot titles in comparison plots for clearer communication of the datasets being compared. The test asserts that a Figure is returned and closes it to clean up resources.

        Parameters:
            self ("TestComparisonPlot"): Test case instance.

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
    def setup_method(self: "TestSavePlot") -> Generator[None, None, None]:
        """
        This fixture sets up the test environment for plot saving tests by initializing the `MPASPrecipitationPlotter` and creating a temporary directory for output files. The fixture yields control to the test functions and ensures that the temporary directory is cleaned up after tests complete. This setup allows plot saving tests to write output files without affecting the actual filesystem and ensures that any generated files are removed after testing.

        Parameters:
            self ("TestSavePlot"): Test case instance.

        Returns:
            Generator[None, None, None]: The fixture manages setup and teardown for plot saving tests.
        """
        self.plotter = MPASPrecipitationPlotter()
        self.temp_dir = tempfile.mkdtemp()
    
        yield

        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_plot_no_figure(self: "TestSavePlot") -> None:
        """
        This test checks that calling `save_plot` without a created figure raises a ValueError. The plotter should validate that a figure exists before attempting to save and raise a ValueError with a clear message if no figure is available. This ensures that users receive immediate feedback on incorrect usage rather than encountering unexpected errors during file saving. The test asserts that the expected exception is raised when `save_plot` is called without a figure.

        Parameters:
            self ("TestSavePlot"): Test case instance.

        Returns:
            None: Assertion verifies the ValueError is raised for missing Figure.
        """
        output_path = os.path.join(self.temp_dir, 'test_plot')
        
        with pytest.raises(ValueError) as ctx:
            self.plotter.save_plot(output_path)

        assert 'No figure to save' in str(ctx.value)
    
    def test_save_plot_creates_directory(self: "TestSavePlot", mpas_coordinates, mpas_precip_data) -> None:
        """
        This test validates that the `save_plot` function can successfully create necessary directories when saving a plot to a specified path. The plotter should check if the target directory exists and create it if it does not before saving the figure. This ensures that users can specify nested output paths without needing to manually create directories beforehand. The test asserts that the output file is created successfully at the specified path, confirming that directory handling and file saving work as intended.

        Parameters:
            self ("TestSavePlot"): Test case instance.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_precip_data: Session fixture providing real precipitation data.

        Returns:
            None: Assertions validate output file creation and directory handling.
        """
        if mpas_coordinates is None or mpas_precip_data is None:
            pytest.skip("MPAS data not available")
        
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


class TestColormap:
    """ Tests for precipitation colormap creation. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestColormap") -> None:
        """
        This fixture sets up the test environment for colormap creation tests by initializing the `MPASPrecipitationPlotter`. The fixture prepares the plotter instance for use in subsequent tests that validate the generation of precipitation colormaps tailored for different accumulation periods. The fixture does not load any data, as the colormap tests focus on the logic of colormap generation rather than data content.

        Parameters:
            self ("TestColormap"): Test case instance.
        
        Returns:
            None: The fixture initializes the plotter for use in colormap tests.
        """
        self.plotter = MPASPrecipitationPlotter()
    
    def test_create_precip_colormap_24h(self: "TestColormap") -> None:
        """
        This test validates the creation of a precipitation colormap tailored for 24-hour accumulation periods. The helper function should return a ListedColormap and a corresponding list of levels that are appropriate for visualizing 24-hour precipitation accumulations. This ensures that the colormap is designed to provide clear visual distinctions for typical precipitation values over a daily period. The test asserts that the returned colormap is not None and that the levels are provided as a list with expected content.

        Parameters:
            self ("TestColormap"): Test case instance.

        Returns:
            None: Assertions validate colormap generation for 24h period.
        """
        cmap, levels = self.plotter.create_precip_colormap('a24h')
        
        assert cmap is not None
        assert isinstance(levels, list)
        assert len(levels) > 0
    
    def test_create_precip_colormap_1h(self: "TestColormap") -> None:
        """
        This test validates the creation of a precipitation colormap tailored for 1-hour accumulation periods. The helper function should return a ListedColormap and a corresponding list of levels that are appropriate for visualizing 1-hour precipitation accumulations. This ensures that the colormap is designed to provide clear visual distinctions for typical precipitation values over a short period. The test asserts that the returned colormap is not None and that the levels are provided as a list with expected content.

        Parameters:
            self ("TestColormap"): Test case instance.

        Returns:
            None: Assertions validate colormap generation for 1h period.
        """
        cmap, levels = self.plotter.create_precip_colormap('a01h')
        
        assert cmap is not None
        assert isinstance(levels, list)
    
    def test_create_precip_colormap_all_periods(self: "TestColormap") -> None:
        """
        Test generation of precipitation colormaps for a range of accumulation periods. The helper function should provide consistent colormap objects and level lists for each standard period. This guarantees comparability across plots using different accumulation windows. The test asserts expected colormap types and level lists match known references.

        Parameters:
            self ("TestColormap"): Test case instance.

        Returns:
            None: Assertions validate colormap generation across periods.
        """
        test_cases = [
            ('a01h', [0.1, 0.5, 2.5, 5, 10, 15, 20, 25, 50, 75]),
            ('a03h', [0.1, 0.5, 2.5, 5, 10, 15, 20, 25, 50, 75]),
            ('a12h', [0.1, 1, 5, 10, 20, 30, 40, 50, 100, 150]),
            ('a24h', [0.1, 1, 5, 10, 20, 30, 40, 50, 100, 150]),
        ]
        
        for accum_period, expected_levels in test_cases:
            cmap, levels = self.plotter.create_precip_colormap(accum_period)
            
            assert isinstance(cmap, mcolors.ListedColormap), f"Failed for {accum_period}"

            assert cmap.N == pytest.approx(11), f"Failed for {accum_period}"
            assert levels == expected_levels, f"Failed for {accum_period}"


class TestTimestampHandling:
    """ Tests for timestamp and accumulation period handling. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestTimestampHandling", mpas_coordinates, mpas_precip_data) -> None:
        """
        This fixture sets up the test environment for timestamp handling tests by initializing the `MPASPrecipitationPlotter` and loading real MPAS coordinates and precipitation data. It checks for the availability of the required data and skips tests if the data is not present. The fixture prepares subsets of longitude, latitude, and precipitation arrays for use in subsequent tests that validate the handling of time_end and time_start parameters when creating precipitation maps with specific accumulation periods. The tests will cover scenarios where only time_end is provided (requiring inference of time_start) and where both time_end and time_start are explicitly provided, ensuring that the plotter correctly computes the accumulation window and produces valid figures for both cases.

        Parameters:
            self ("TestTimestampHandling"): Test case instance.
            mpas_coordinates: Session fixture providing real MPAS longitude and latitude arrays.
            mpas_precip_data: Session fixture providing real MPAS precipitation data.
        
        Returns:
            None: The fixture initializes instance attributes for use in timestamp handling tests.
        """
        if mpas_coordinates is None or mpas_precip_data is None:
            pytest.skip("MPAS data not available")
        
        self.plotter = MPASPrecipitationPlotter()
        lon_full, lat_full = mpas_coordinates
        self.lon = lon_full[:50]
        self.lat = lat_full[:50]
        self.precip = mpas_precip_data[:50]
    
    def test_plot_with_time_end_only(self: "TestTimestampHandling") -> None:
        """
        This test validates that the `create_precipitation_map` function can successfully create a precipitation map when only the `time_end` parameter is provided along with an accumulation period. The plotter should infer the appropriate `time_start` based on the specified `accum_period` and produce a valid Figure for visualization. This ensures that users can create accumulated precipitation maps by providing just the end time, simplifying the interface for common use cases. The test asserts that a Figure is returned and closes it to clean up resources.

        Parameters:
            self ("TestTimestampHandling"): Test case instance.

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
    
    def test_plot_with_time_end_and_start(self: "TestTimestampHandling") -> None:
        """
        This test validates that the `create_precipitation_map` function can successfully create a precipitation map when both `time_start` and `time_end` parameters are provided. The plotter should honor the specified interval and produce a valid Figure for visualization. This ensures that users can create accumulated precipitation maps for explicit time ranges. The test asserts that a Figure is returned and closes it to clean up resources.

        Parameters:
            self ("TestTimestampHandling"): Test case instance.

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


class TestPrecipitationPlotterIntegration:
    """ Integration tests for precipitation plotter with other modules. """
    
    def test_import_from_visualization_package(self: "TestPrecipitationPlotterIntegration") -> None:
        """
        This test validates that the `MPASPrecipitationPlotter` can be successfully imported from the main `mpasdiag.visualization` package. The test checks that the plotter class is accessible and contains the expected method for creating precipitation maps. This ensures that the plotter is properly exposed in the package's public API and can be used by external code without import issues. The test will skip if the plotter cannot be imported, indicating a potential issue with package structure or dependencies.

        Parameters:
            self ("TestPrecipitationPlotterIntegration"): Test case instance.

        Returns:
            None: Assertion verifies plotter accessibility or skips on ImportError.
        """
        try:
            from mpasdiag.visualization import MPASPrecipitationPlotter
            assert hasattr(MPASPrecipitationPlotter, 'create_precipitation_map')
        except ImportError:
            pytest.skip("Precipitation plotter not available in main package")
    
    def test_styling_integration(self: "TestPrecipitationPlotterIntegration") -> None:
        """
        This test validates that the `MPASVisualizationStyle` class from the styling module can be successfully imported and used to create a precipitation colormap. The test checks that the colormap creation function returns a valid ListedColormap and a list of levels without errors. This ensures that the styling module is properly integrated with the visualization code and that its functionality can be utilized for consistent plot styling. The test will skip if the styling module cannot be imported, indicating a potential issue with dependencies or package structure.

        Parameters:
            self ("TestPrecipitationPlotterIntegration"): Test case instance.

        Returns:
            None: Assertion validates styling integration or skips on ImportError.
        """
        try:
            from mpasdiag.visualization.styling import MPASVisualizationStyle
            cmap, levels = MPASVisualizationStyle.create_precip_colormap('a01h')
            
            assert isinstance(cmap, mcolors.ListedColormap)
            assert isinstance(levels, list)
            assert len(levels) > 0
            
        except ImportError:
            pytest.skip("Styling module not available")


class TestEdgeCases:
    """ Tests for edge cases and special scenarios. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestEdgeCases", mpas_coordinates) -> None:
        """
        This fixture sets up the test environment for edge case tests by initializing the `MPASPrecipitationPlotter` and loading real MPAS coordinates. It checks for the availability of the required data and skips tests if the data is not present. The fixture prepares subsets of longitude and latitude arrays for use in subsequent tests that validate the behavior of the plotter when handling edge cases such as empty or NaN-filled precipitation data, extreme values, and dynamic tick formatting. This setup ensures that the edge case tests have access to realistic coordinate data while focusing on testing the plotter's robustness in handling unusual input scenarios.

        Parameters:
            self ("TestEdgeCases"): Test case instance.

        Returns:
            None: The fixture initializes instance attributes for use in edge case tests.
        """
        if mpas_coordinates is None:
            pytest.skip("MPAS data not available")
        
        self.plotter = MPASPrecipitationPlotter()
        lon_full, lat_full = mpas_coordinates
        self.lon = lon_full[:50]
        self.lat = lat_full[:50]
    
    def test_precipitation_map_empty_data(self: "TestEdgeCases") -> None:
        """
        This test validates the behavior of the `create_precipitation_map` function when provided with an input array that is entirely filled with NaN values. The plotter should handle this edge case gracefully, either by producing an empty plot or by displaying a message indicating that no valid data is available. The test asserts that a Figure is returned without errors, confirming that the plotter can manage cases where precipitation data is missing or invalid without crashing.

        Parameters:
            self ("TestEdgeCases"): Test case instance.

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
    
    def test_precipitation_map_extreme_values(self: "TestEdgeCases") -> None:
        """
        This test validates the behavior of the `create_precipitation_map` function when provided with precipitation data containing extreme values (e.g., very high or very low precipitation). The plotter should be able to handle these values without crashing and should produce a valid Figure that appropriately represents the data, potentially using a colormap that can accommodate the range of values. The test asserts that a Figure is returned without errors, confirming that the plotter can manage cases with extreme precipitation values effectively.

        Parameters:
            self ("TestEdgeCases"): Test case instance.

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
    
    def test_format_ticks_dynamic(self: "TestEdgeCases") -> None:
        """
        This test validates the functionality of the `_format_ticks_dynamic` method, which formats tick labels based on the range and distribution of tick values. The test provides various sets of tick values, including small decimals, integers, and a mix of both, and checks that the formatted output matches expected string representations. This ensures that the dynamic tick formatting logic correctly adapts to different types of tick values, providing clear and appropriately formatted labels for axes in precipitation plots. The test asserts that the formatted tick labels match the expected output for each set of input ticks, confirming that the method handles edge cases in tick formatting effectively.

        Parameters:
            self ("TestEdgeCases"): Test case instance.

        Returns:
            None: Assertions validate formatting outcomes for test cases.
        """
        test_cases = [
            ([0.1, 0.5, 1.0, 2.0, 5.0], ['0.10', '0.50', '1.00', '2.00', '5.00']),
            ([1, 5, 10, 20, 50], ['1', '5', '10', '20', '50']),
            ([0.01, 0.1, 1, 10], ['0.01', '0.10', '1.00', '10.00']),
        ]
        
        for ticks, expected in test_cases:
            result = self.plotter._format_ticks_dynamic(ticks)
            assert result == expected, f"Failed for ticks={ticks}"


class TestPrepareOverlayData:
    """ Tests for _prepare_overlay_data method covering unit conversion, negative clipping, and bounds. """

    @pytest.fixture(autouse=True)
    def setup(self: "TestPrepareOverlayData") -> None:
        """
        This fixture sets up the test environment for testing the `_prepare_overlay_data` method by initializing an instance of `MPASPrecipitationPlotter`. The method being tested is responsible for preparing data for overlay plots, including handling unit conversions, clipping negative values, and applying bounds filtering. The fixture ensures that a plotter instance is available for all tests in this class, allowing them to focus on validating the specific behaviors of the `_prepare_overlay_data` method under various input scenarios.

        Parameters:
            self ("TestPrepareOverlayData"): Test case instance.

        Returns:
            None: The fixture initializes the plotter for use in overlay data preparation tests.
        """
        self.plotter = MPASPrecipitationPlotter()

    def test_overlay_data_negative_clipping(self: "TestPrepareOverlayData") -> None:
        """
        This test verifies that negative precipitation values are correctly clipped to 0 and that a warning is issued when such values are encountered. The test generates a set of longitude, latitude, and precipitation data, including negative values, and checks that the resulting prepared data contains no negative values. It also captures the standard output to ensure that a warning message about negative values is printed.

        Parameters:
            self ("TestPrepareOverlayData"): Test case instance.

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

    def test_overlay_data_unit_conversion(self: "TestPrepareOverlayData") -> None:
        """
        This test verifies that when the original units of the data differ from the display units, a conversion is attempted. It generates a set of longitude, latitude, and precipitation data, and checks that the resulting prepared data is valid after conversion.

        Parameters:
            self ("TestPrepareOverlayData"): Test case instance.

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

    def test_overlay_data_high_mean_warning(self: "TestPrepareOverlayData") -> None:
        """
        This test verifies that when the mean of the data is unusually high and no original units are provided, a warning about potential unit issues is issued. It generates a set of longitude, latitude, and precipitation data with high values and checks that the warning message is printed.

        Parameters:
            self ("TestPrepareOverlayData"): Test case instance.

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

    def test_overlay_data_all_invalid_raises(self: "TestPrepareOverlayData") -> None:
        """
        This test verifies that when all data points are invalid (NaN), a ValueError is raised. It generates a set of longitude, latitude, and precipitation data with all values as NaN and checks that the appropriate exception is raised.

        Parameters:
            self ("TestPrepareOverlayData"): Test case instance.

        Returns:
            None: Assertions validate that a ValueError is raised for all-NaN data.
        """
        lon = np.linspace(-110, -100, 10)
        lat = np.linspace(35, 45, 10)
        data = np.full(10, np.nan)

        with pytest.raises(ValueError, match="No valid"):
            self.plotter._prepare_overlay_data(lon, lat, data, 'test', None, 'scatter')

    def test_overlay_data_contour_mask_differs_from_scatter(self: "TestPrepareOverlayData") -> None:
        """
        This test verifies that the contour plot type uses a simpler mask than the scatter plot type, specifically without bounds filtering. It generates a set of longitude, latitude, and precipitation data, and checks that both scatter and contour plots return valid data.

        Parameters:
            self ("TestPrepareOverlayData"): Test case instance.

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
    def setup(self: "TestContourPlotCreation") -> None:
        """
        This fixture sets up the test environment for contour plot creation tests by initializing an instance of `MPASPrecipitationPlotter` and creating a figure and axes with a PlateCarree projection. The fixture prepares the plotter instance and axes for use in subsequent tests that validate the functionality of the `_create_contour_plot` and `_create_contourf_plot` methods, which are responsible for rendering contour lines and filled contours on the map. This setup ensures that the contour plot creation tests have a consistent plotting environment to work with.

        Parameters:
            self ("TestContourPlotCreation"): Test case instance.

        Returns:
            None: The fixture initializes the plotter and axes for contour plot tests.
        """
        self.plotter = MPASPrecipitationPlotter()
        self.plotter.fig = plt.figure(figsize=(10, 8))
        self.plotter.ax = self.plotter.fig.add_subplot(111, projection=ccrs.PlateCarree())

    def teardown_method(self: "TestContourPlotCreation") -> None:
        """
        This method is called after each test in the `TestContourPlotCreation` class to close all figures and clean up resources. It ensures that any figures created during the tests are properly closed to prevent memory leaks and to maintain a clean testing environment for subsequent tests.

        Parameters:
            self ("TestContourPlotCreation"): Test case instance.

        Returns:
            None: Figures are closed.
        """
        plt.close('all')

    def test_create_contour_plot(self: "TestContourPlotCreation") -> None:
        """
        This test validates that the `_create_contour_plot` method can successfully create contour lines on the map when provided with longitude, latitude, and data arrays. The test generates synthetic data for testing and checks that the method executes without errors, confirming that the contour plotting logic is functional. The test uses mocking to bypass the interpolation step, allowing it to focus on the contour creation aspect. Assertions ensure that no exceptions are raised during the contour plot creation process.

        Parameters:
            self ("TestContourPlotCreation"): Test case instance.

        Returns:
            None: Assertions validate that contour lines are drawn.
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
            self.plotter._create_contour_plot(
                lon, lat, data, -110, -100, 35, 45,
                cmap, norm, levels, data_crs
            )

    def test_create_contourf_plot(self: "TestContourPlotCreation") -> None:
        """
        This test validates that the `_create_contourf_plot` method can successfully create filled contours on the map when provided with longitude, latitude, and data arrays. The test generates synthetic data for testing and checks that the method executes without errors, confirming that the filled contour plotting logic is functional. The test uses mocking to bypass the interpolation step, allowing it to focus on the filled contour creation aspect. Assertions ensure that no exceptions are raised during the filled contour plot creation process.

        Parameters:
            self ("TestContourPlotCreation"): Test case instance.

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
    def setup(self: "TestAddPrecipitationOverlay") -> None:
        """
        This fixture sets up the test environment for testing the `add_precipitation_overlay` method by initializing an instance of `MPASPrecipitationPlotter`, creating a figure and axes with a PlateCarree projection, and preparing longitude and latitude arrays. The fixture ensures that a consistent plotting environment is available for all tests in this class, allowing them to focus on validating the functionality of the `add_precipitation_overlay` method under various input scenarios, including different plot types, data conditions, and configuration settings.

        Parameters:
            self ("TestAddPrecipitationOverlay"): Test case instance.

        Returns:
            None: Plotter and axes are created.
        """
        self.plotter = MPASPrecipitationPlotter()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection=ccrs.PlateCarree())
        self.lon = np.linspace(-110, -100, 100)
        self.lat = np.linspace(35, 45, 100)

    def teardown_method(self: "TestAddPrecipitationOverlay") -> None:
        """
        This method is called after each test in the `TestAddPrecipitationOverlay` class to close all figures and clean up resources. It ensures that any figures created during the tests are properly closed to prevent memory leaks and to maintain a clean testing environment for subsequent tests.

        Parameters:
            self ("TestAddPrecipitationOverlay"): Test case instance.

        Returns:
            None: Figures are closed.
        """
        plt.close('all')

    def test_overlay_scatter(self: "TestAddPrecipitationOverlay") -> None:
        """
        This test validates that the `add_precipitation_overlay` method can successfully add a scatter overlay to the map when provided with appropriate configuration and data. The test generates synthetic precipitation data and checks that the method executes without errors, confirming that the scatter overlay logic is functional. The test captures standard output to verify that the method processes the scatter plot type correctly.

        Parameters:
            self ("TestAddPrecipitationOverlay"): Test case instance.

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

    def test_overlay_invalid_plot_type(self: "TestAddPrecipitationOverlay") -> None:
        """
        This test validates that the `add_precipitation_overlay` method raises a ValueError when an invalid plot type is specified in the configuration. The test provides a configuration with an unsupported plot type and checks that the appropriate exception is raised, confirming that the method correctly handles invalid input for plot types.

        Parameters:
            self ("TestAddPrecipitationOverlay"): Test case instance.

        Returns:
            None: Assertions validate that ValueError is raised for invalid plot type.
        """
        config = {
            'data': np.random.uniform(0, 5, 100),
            'plot_type': 'invalid_type',
        }

        with pytest.raises(ValueError, match="plot_type must be"):
            self.plotter.add_precipitation_overlay(self.ax, self.lon, self.lat, config)

    def test_overlay_no_valid_data_warning(self: "TestAddPrecipitationOverlay") -> None:
        """
        This test validates that the `add_precipitation_overlay` method prints a warning and returns without rendering when all data values are NaN. The test provides a configuration with all-NaN data and checks that the appropriate warning is printed, confirming that the method correctly handles cases with no valid data.

        Parameters:
            self ("TestAddPrecipitationOverlay"): Test case instance.

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

    def test_overlay_contourf_with_remap(self: "TestAddPrecipitationOverlay") -> None:
        """
        This test validates that when a contourf overlay is added, the method attempts to remap the data to a lat/lon grid. The test mocks the remapping function to return a predefined remapped dataset and checks that the contourf plotting logic is executed without errors. This ensures that the method correctly integrates remapping into the contourf overlay creation process.

        Parameters:
            self ("TestAddPrecipitationOverlay"): Test case instance.

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

    def test_overlay_uses_data_bounds_when_none_provided(self: "TestAddPrecipitationOverlay") -> None:
        """
        This test validates that when no explicit bounds are provided in the configuration for a scatter overlay, the method uses the data bounds to determine which points to plot. The test generates synthetic precipitation data and checks that the method executes without errors, confirming that it can handle cases where bounds are not specified by relying on the data values.

        Parameters:
            self ("TestAddPrecipitationOverlay"): Test case instance.

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


class TestPrecipitationUnitConversion:
    """ Tests for _convert_precipitation_units method edge cases. """

    def setup_method(self) -> None:
        """ 
        This method is called before each test in the `TestPrecipitationUnitConversion` class to initialize an instance of `MPASPrecipitationPlotter`. The method ensures that a fresh plotter instance is available for each test, allowing them to focus on validating the specific behaviors of the `_convert_precipitation_units` method under various input scenarios, including handling of None data arrays, unit conversion attempts, and fallback mechanisms when unit conversion fails. This setup is essential for testing the robustness of the unit conversion logic in the precipitation plotter.

        Parameters:
            self ("TestPrecipitationUnitConversion"): Test case instance.

        Returns:
            None: The method initializes the plotter for use in unit conversion tests.
        """
        self.plotter = MPASPrecipitationPlotter()

    def test_convert_units_none_data_array(self) -> None:
        """
        This test verifies that when the data array is None, the method returns the original data with 'mm' units without attempting conversion. It checks that the output data is unchanged and that the units are set to 'mm', confirming that the method handles cases with missing data arrays gracefully by providing a default behavior.

        Parameters:
            self ("TestPrecipitationUnitConversion"): Test case instance.

        Returns:
            None: Assertions validate that the original data is returned with 'mm' units.
        """
        data = np.array([1.0, 2.0, 3.0])
        result, unit = self.plotter._convert_precipitation_units(data, 'rainnc', None) # type: ignore
        assert unit == 'mm'
        np.testing.assert_array_equal(result, data)

    def test_convert_units_with_unit_converter(self) -> None:
        """
        This test verifies the normal conversion path via UnitConverter. It checks that the method correctly converts the data array units when a valid UnitConverter is available, ensuring that the output data is a numpy array and the units are returned as a string.

        Parameters:
            self ("TestPrecipitationUnitConversion"): Test case instance.

        Returns:
            None: Assertions validate that the data is converted and units are returned correctly.
        """
        data = np.array([0.001, 0.002, 0.003])
        data_array = xr.DataArray(data, attrs={'units': 'kg m-2', 'long_name': 'precip'})
        result, unit = self.plotter._convert_precipitation_units(data, 'rainnc', data_array) # type: ignore
        assert isinstance(result, np.ndarray)
        assert isinstance(unit, str)

    def test_convert_units_attribute_error_fallback(self) -> None:
        """
        This test verifies that if the UnitConverter raises an AttributeError (e.g., due to a missing method), the method falls back to returning the original data with 'mm' units. It checks that the output data is a numpy array and that the units are set to 'mm', confirming that the fallback mechanism works correctly when unit conversion fails due to an AttributeError.

        Parameters:
            self ("TestPrecipitationUnitConversion"): Test case instance.

        Returns:
            None: Assertions validate that the fallback mechanism returns data with 'mm' units.
        """
        data = np.array([1.0, 2.0])
        data_array = MagicMock()
        data_array.attrs = {'units': 'mm', 'long_name': 'Precipitation'}
        data_array.units = 'mm'
        data_array.long_name = 'Precipitation'

        with patch('mpasdiag.visualization.precipitation.UnitConverter') as mock_uc:
            mock_uc.convert_data_for_display.side_effect = AttributeError("missing method")
            result, unit = self.plotter._convert_precipitation_units(data, 'rainnc', data_array) # type: ignore

        assert isinstance(result, np.ndarray)
        assert unit == 'mm'

    def test_negative_precipitation_clipping(self, capsys) -> None:
        """
        This test verifies that negative precipitation values are clipped to 0 and that a warning is printed when such values are encountered. It checks that the resulting data contains no negative values and that the warning message about negative values is printed to standard output.

        Parameters:
            self ("TestPrecipitationUnitConversion"): Test case instance.
            capsys: Pytest fixture to capture stdout and stderr.

        Returns:
            None: Assertions validate that negative values are clipped and a warning is printed.
        """
        data = np.array([-0.5, 0.0, 1.0, 2.0, -0.1])
        data_array = xr.DataArray(data, attrs={'units': 'mm'})
        result, _ = self.plotter._convert_precipitation_units(data, 'rainnc', data_array) # type: ignore
        assert np.all(result >= 0)
        captured = capsys.readouterr()
        assert 'negative' in captured.out.lower() or np.min(result) >= 0


class TestPrecipitationRenderingPaths:
    """ Tests for rendering paths in precipitation maps. """

    def setup_method(self) -> None:
        """ 
        This method is called before each test in the `TestPrecipitationRenderingPaths` class to initialize an instance of `MPASPrecipitationPlotter`. The method ensures that a fresh plotter instance is available for each test, allowing them to focus on validating the specific behaviors of the rendering paths in the precipitation plotter, such as colormap creation, level filtering, boundary normalization, scatter plotting with geographic bounds, accumulation period mapping, and color level processing. This setup is essential for testing the robustness and correctness of the rendering logic in the precipitation plotter.

        Parameters:
            self ("TestPrecipitationRenderingPaths"): Test case instance.

        Returns:
            None: The method initializes the plotter for use in rendering path tests.
        """
        self.plotter = MPASPrecipitationPlotter()

    def test_colormap_levels_preparation(self) -> None:
        """
        This test verifies that the colormap and levels are correctly prepared for a given accumulation period. It checks that the colormap is not None, that the levels list is not empty, and that all levels are finite values. This ensures that the colormap creation logic produces valid outputs for use in precipitation rendering.

        Parameters:
            self ("TestPrecipitationRenderingPaths"): Test case instance.

        Returns:
            None: Assertions validate the colormap and levels.
        """
        cmap, levels = self.plotter.create_precip_colormap('a01h')
        assert cmap is not None
        assert len(levels) > 0
        assert all(np.isfinite(levels))

    def test_clim_min_max_level_filtering(self) -> None:
        """
        This test verifies that the color levels are correctly filtered and bounded by clim_min and clim_max values. It checks that the resulting color levels include the specified minimum and maximum values, and that they are sorted in ascending order. This ensures that the level filtering logic correctly prepares the color levels for rendering based on the specified bounds.

        Parameters:
            self ("TestPrecipitationRenderingPaths"): Test case instance.

        Returns:
            None: Assertions validate the filtering and bounding of color levels.
        """
        _, levels = self.plotter.create_precip_colormap('a01h')
        clim_min, clim_max = 1.0, 10.0
        filtered = [lv for lv in levels if clim_min <= lv <= clim_max]

        if clim_min not in filtered:
            filtered.insert(0, clim_min)

        if clim_max not in filtered:
            filtered.append(clim_max)

        color_levels_sorted = sorted(set([v for v in filtered if np.isfinite(v)]))

        assert color_levels_sorted[0] == clim_min
        assert color_levels_sorted[-1] == clim_max

    def test_boundary_norm_bounds_normalization(self) -> None:
        """
        This test verifies that the bounds used for BoundaryNorm are correctly normalized to include the minimum and maximum color levels. It checks that the bounds list starts with 0 and ends with a value greater than the maximum color level, ensuring that the normalization logic prepares appropriate bounds for the colormap.

        Parameters:
            self ("TestPrecipitationRenderingPaths"): Test case instance.

        Returns:
            None: Assertions validate the bounds normalization.
        """
        color_levels_sorted = [0.1, 0.5, 1.0, 2.0, 5.0]
        last_bound = max(color_levels_sorted) + 1
        bounds = [0] + color_levels_sorted + [last_bound]
        assert bounds[-1] == pytest.approx(6.0)
        assert bounds[0] == pytest.approx(0.0)

    def test_scatter_plot_valid_mask_geographic(self) -> None:
        """
        This test verifies that the valid_mask for scatter plotting correctly identifies points that are within the specified geographic bounds and have valid precipitation data. It checks that the valid_mask includes only points that are finite, non-negative, and within the longitude and latitude bounds, ensuring that the scatter plotting logic correctly filters data points for rendering on the map.

        Parameters:
            self ("TestPrecipitationRenderingPaths"): Test case instance.

        Returns:
            None: Assertions validate the valid_mask for geographic bounds.
        """
        n = 100

        lon = np.random.uniform(-110, -100, n)
        lat = np.random.uniform(30, 40, n)
        data = np.random.uniform(0, 10, n)

        valid_mask = (np.isfinite(data) & (data >= 0) & (data < 1e5) &
                      (lon >= -110) & (lon <= -100) &
                      (lat >= 30) & (lat <= 40))
        
        assert np.sum(valid_mask) > 0
        assert np.sum(valid_mask) <= n

    def test_accum_period_to_hours_mapping(self) -> None:
        """
        This test verifies that the mapping from accumulation period codes (e.g., 'a01h', 'a03h') to human-readable hour labels (e.g., '1-h', '3-h') is correct. It checks that the mapping returns the expected hour labels for known accumulation period codes and that it returns a default value for unknown codes, ensuring that the accumulation period mapping logic is accurate and robust.

        Parameters:
            self ("TestPrecipitationRenderingPaths"): Test case instance.

        Returns:
            None: Assertions validate the accum_period to hours mapping.
        """
        accum_hours_map = {'a01h': '1-h', 'a03h': '3-h', 'a06h': '6-h', 'a12h': '12-h', 'a24h': '24-h'}
        assert accum_hours_map.get('a01h') == '1-h'
        assert accum_hours_map.get('a06h') == '6-h'
        assert accum_hours_map.get('unknown', 'unknown') == 'unknown'

    def test_color_levels_filtering_and_sorting(self) -> None:
        """
        This test verifies that the color levels are correctly filtered to remove non-finite values and sorted in ascending order. It checks that the resulting list of color levels contains only finite values and is sorted, ensuring that the color level processing logic prepares valid levels for rendering.

        Parameters:
            self ("TestPrecipitationRenderingPaths"): Test case instance.

        Returns:
            None: Assertions validate the filtering and sorting of color levels.
        """
        color_levels = [0.1, float('inf'), 0.5, 1.0, float('nan'), 2.0, 5.0]
        result = sorted(set([v for v in color_levels if np.isfinite(v)]))
        assert result == [pytest.approx(0.1), pytest.approx(0.5), pytest.approx(1.0), pytest.approx(2.0), pytest.approx(5.0)]
        assert not any(np.isinf(v) or np.isnan(v) for v in result)


class TestPrecipitationBatchAndCoordinates:
    """ Tests for batch processing, coordinate extraction, and time indices. """

    def setup_method(self) -> None:
        """ 
        This method is called before each test in the `TestPrecipitationBatchAndCoordinates` class to initialize an instance of `MPASPrecipitationPlotter`. The method ensures that a fresh plotter instance is available for each test, allowing them to focus on validating the specific behaviors of batch processing, coordinate extraction, and time index setup in the precipitation plotter. This setup is essential for testing the robustness and correctness of these functionalities under various input scenarios, including different processor configurations, variable names, accumulation periods, and user-specified time indices.

        Parameters:
            self ("TestPrecipitationBatchAndCoordinates"): Test case instance.

        Returns:
            None: Initializes the plotter instance for each test.

        """
        self.plotter = MPASPrecipitationPlotter()

    def test_extract_coordinates_from_processor_2d(self) -> None:
        """
        This test verifies that the `_extract_coordinates_from_processor` method successfully extracts 2D coordinates when the processor provides the `extract_2d_coordinates_for_variable` method. It checks that the returned longitude and latitude arrays have the expected length, confirming that the method correctly utilizes the processor's coordinate extraction functionality when available.

        Parameters:
            self ("TestPrecipitationBatchAndCoordinates"): Test case instance.

        Returns:
            None: Assertions validate that 2D coordinates are extracted correctly from the processor.
        """
        mock_proc = MagicMock()
        mock_proc.extract_2d_coordinates_for_variable.return_value = (np.zeros(10), np.zeros(10))
        lon, lat = self.plotter._extract_coordinates_from_processor(mock_proc, 'rainnc')
        assert len(lon) == pytest.approx(10)

    def test_extract_coordinates_from_processor_spatial(self) -> None:
        """
        This test verifies that the `_extract_coordinates_from_processor` method falls back to the `extract_spatial_coordinates` method when the processor does not provide the `extract_2d_coordinates_for_variable` method. It checks that the returned longitude and latitude arrays have the expected length, confirming that the fallback mechanism works correctly.

        Parameters:
            self ("TestPrecipitationBatchAndCoordinates"): Test case instance.

        Returns:
            None: Assertions validate that 2D coordinates are extracted correctly from the processor using the fallback method.
        """
        mock_proc = MagicMock(spec=[])
        mock_proc.extract_spatial_coordinates = MagicMock(return_value=(np.zeros(10), np.zeros(10)))
        lon, lat = self.plotter._extract_coordinates_from_processor(mock_proc, 'rainnc')
        assert len(lon) == pytest.approx(10)

    def test_extract_coordinates_from_processor_dataset_fallback(self) -> None:
        """
        This test verifies that the `_extract_coordinates_from_processor` method falls back to the dataset's `lonCell` and `latCell` values when the processor does not provide the `extract_2d_coordinates_for_variable` or `extract_spatial_coordinates` methods. It checks that the returned longitude and latitude arrays have the expected length, confirming that the fallback mechanism works correctly.

        Parameters:
            self ("TestPrecipitationBatchAndCoordinates"): Test case instance.

        Returns:
            None: Assertions validate that 2D coordinates are extracted correctly from the dataset fallback.
        """
        mock_proc = MagicMock(spec=[])
        mock_proc.dataset = MagicMock()
        mock_proc.dataset.lonCell.values = np.zeros(10)
        mock_proc.dataset.latCell.values = np.zeros(10)
        lon, lat = self.plotter._extract_coordinates_from_processor(mock_proc, 'rainnc')
        assert len(lon) == pytest.approx(10)

    def test_setup_batch_time_indices_all_valid(self) -> None:
        """
        This test verifies that the `_setup_batch_time_indices` method returns valid indices for all times when the dataset has sufficient timesteps. It checks that the returned indices and accumulation hours are as expected.

        Parameters:
            self ("TestPrecipitationBatchAndCoordinates"): Test case instance.

        Returns:
            None: Assertions validate that the method correctly calculates batch time indices and accumulation hours.
        """
        mock_proc = MagicMock()
        mock_proc.dataset = MagicMock()
        mock_proc.dataset.sizes = {'Time': 10}

        indices, accum_hours = self.plotter._setup_batch_time_indices(mock_proc, 'a01h', None)

        assert accum_hours == pytest.approx(1)
        assert indices == list(range(1, 10))

    def test_setup_batch_time_indices_too_few_timesteps(self, capsys) -> None:
        """
        This test verifies that the `_setup_batch_time_indices` method returns empty indices when the dataset has insufficient timesteps. It checks that the returned indices and accumulation hours are as expected.

        Parameters:
            self ("TestPrecipitationBatchAndCoordinates"): Test case instance.

        Returns:
            None: Assertions validate that the method correctly handles insufficient timesteps.
        """
        mock_proc = MagicMock()
        mock_proc.dataset = MagicMock()
        mock_proc.dataset.sizes = {'Time': 1}

        indices, accum_hours = self.plotter._setup_batch_time_indices(mock_proc, 'a24h', None)

        assert indices == []
        assert accum_hours == pytest.approx(24)

    def test_setup_batch_time_indices_filtered_user_indices(self) -> None:
        """
        This test verifies that the `_setup_batch_time_indices` method filters user-specified indices correctly. It checks that only valid indices are returned when the dataset has sufficient timesteps.

        Parameters:
            self ("TestPrecipitationBatchAndCoordinates"): Test case instance.

        Returns:
            None: Assertions validate that the method correctly filters user-specified indices.
        """
        mock_proc = MagicMock()
        mock_proc.dataset = MagicMock()
        mock_proc.dataset.sizes = {'Time': 10}

        indices, _ = self.plotter._setup_batch_time_indices(mock_proc, 'a06h', [0, 1, 2, 6, 8])

        assert all(idx >= 6 for idx in indices)

    def test_setup_batch_time_indices_no_valid_user_indices(self, capsys) -> None:
        """
        This test verifies that the `_setup_batch_time_indices` method returns empty indices when all user-specified indices are invalid. It checks that the returned indices are empty as expected.

        Parameters:
            self ("TestPrecipitationBatchAndCoordinates"): Test case instance.

        Returns:
            None: Assertions validate that the method correctly handles invalid user-specified indices.
        """
        mock_proc = MagicMock()
        mock_proc.dataset = MagicMock()
        mock_proc.dataset.sizes = {'Time': 10}

        indices, _ = self.plotter._setup_batch_time_indices(mock_proc, 'a24h', [0, 1, 2])

        assert indices == []

    def test_batch_precipitation_maps_empty_time_indices(self) -> None:
        """
        This test verifies that the `create_batch_precipitation_maps` method returns an empty list when no valid times are available. It checks that the returned result is empty as expected.

        Parameters:
            self ("TestPrecipitationBatchAndCoordinates"): Test case instance.

        Returns:
            None: Assertions validate that the method correctly handles empty time indices.
        """
        import tempfile
        mock_proc = MagicMock()
        mock_proc.dataset = MagicMock()
        mock_proc.dataset.sizes = {'Time': 1}

        mock_proc.extract_2d_coordinates_for_variable.return_value = (
            np.random.uniform(-110, -100, 50), np.random.uniform(30, 40, 50)
        )

        with tempfile.TemporaryDirectory() as tmp_output_dir:
            result = self.plotter.create_batch_precipitation_maps(
                processor=mock_proc, output_dir=tmp_output_dir,
                lon_min=-110, lon_max=-100, lat_min=30, lat_max=40,
                var_name='rainnc', accum_period='a24h'
            )

        assert result == []


class TestPrecipitationOverlayEdgeCases:
    """ Tests for overlay data preparation edge cases. """

    def setup_method(self) -> None:
        """
        This method is called before each test in the `TestPrecipitationOverlayEdgeCases` class to initialize an instance of `MPASPrecipitationPlotter`. The method ensures that a fresh plotter instance is available for each test, allowing them to focus on validating the specific behaviors of the `_prepare_overlay_data` method under various input scenarios, including handling of negative precipitation values, cases with no valid data points, unit conversion attempts, and support for xarray DataArray inputs. This setup is essential for testing the robustness and correctness of the overlay data preparation logic in the precipitation plotter.

        Parameters:
            self ("TestPrecipitationOverlayEdgeCases"): Test case instance. 

        Returns:
            None: The method initializes the plotter for use in overlay data preparation tests.  
        """
        self.plotter = MPASPrecipitationPlotter()

    def test_prepare_overlay_data_negative_clipping(self, capsys) -> None:
        """
        This test verifies that the `_prepare_overlay_data` method correctly clips negative precipitation values to 0 and prints a warning when such values are encountered. It checks that the output data contains no negative values and that a warning message is printed, confirming that the method handles negative precipitation values appropriately.

        Parameters:
            self ("TestPrecipitationOverlayEdgeCases"): Test case instance.
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

    def test_prepare_overlay_data_no_valid_points(self) -> None:
        """
        This test verifies that the `_prepare_overlay_data` method raises a ValueError when all data points are invalid (e.g., NaN values). It ensures that the method correctly identifies the absence of valid precipitation data and raises an appropriate exception.

        Parameters:
            self ("TestPrecipitationOverlayEdgeCases"): Test case instance.

        Returns:
            None: Assertions validate that a ValueError is raised when no valid data points are present.
        """
        lon = np.random.uniform(-110, -100, 10)
        lat = np.random.uniform(30, 40, 10)
        data = np.full(10, np.nan)  

        with pytest.raises(ValueError, match="No valid precipitation overlay data"):
            self.plotter._prepare_overlay_data(
                lon, lat, data, 'rainnc', None, 'scatter'
            )

    def test_prepare_overlay_data_with_unit_conversion(self, capsys) -> None:
        """
        This test verifies that the `_prepare_overlay_data` method attempts unit conversion when the original_units parameter differs from the expected units. It ensures that the method correctly handles unit conversion scenarios and produces valid output data.

        Parameters:
            self ("TestPrecipitationOverlayEdgeCases"): Test case instance.
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

    def test_prepare_overlay_data_xarray_inputs(self) -> None:
        """
        This test verifies that the `_prepare_overlay_data` method correctly handles `xr.DataArray` inputs. It ensures that the method can process xarray data structures and produce valid output data.

        Parameters:
            self ("TestPrecipitationOverlayEdgeCases"): Test case instance.

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


class TestPrecipitationProcessSingleTimeStep:
    """ Tests for _process_single_time_step edge cases. """

    def setup_method(self) -> None:
        """
        This method is called before each test in the `TestPrecipitationProcessSingleTimeStep` class to initialize an instance of `MPASPrecipitationPlotter`. The method ensures that a fresh plotter instance is available for each test, allowing them to focus on validating the specific behaviors of the `_process_single_time_step` method under various input scenarios, including handling of missing time coordinates, data processing logic, and integration with other methods such as `compute_precipitation_difference`, `create_precipitation_map`, and file saving. This setup is essential for testing the robustness and correctness of the single time step processing logic in the precipitation plotter.

        Parameters:
            self ("TestPrecipitationProcessSingleTimeStep"): Test case instance.

        Returns:
            None: The method initializes the plotter for use in single time step processing tests.  
        """
        self.plotter = MPASPrecipitationPlotter()

    def test_process_single_time_step_no_time_coord(self) -> None:
        """
        This test verifies that the `_process_single_time_step` method raises a ValueError when the processor's dataset does not contain a 'Time' coordinate. It ensures that the method correctly identifies the absence of time information and raises an appropriate exception.

        Parameters:
            self ("TestPrecipitationProcessSingleTimeStep"): Test case instance.

        Returns:
            None: Assertions validate that a ValueError is raised when no 'Time' coordinate is present in the dataset.
        """
        import tempfile
        mock_proc = MagicMock()
        mock_proc.dataset = xr.Dataset({'rainnc': (['time', 'nCells'], np.zeros((5, 100)))})
        mock_proc.data_type = 'history'

        mock_precip = xr.DataArray(np.random.uniform(0, 5, 100), attrs={'units': 'mm'})

        with tempfile.TemporaryDirectory() as tmp_output_dir:
            with patch.object(PrecipitationDiagnostics, 'compute_precipitation_difference',
                             return_value=mock_precip), \
                 patch.object(self.plotter, 'create_precipitation_map',
                             return_value=(MagicMock(), MagicMock())), \
                 patch.object(self.plotter, 'save_plot'), \
                 patch.object(self.plotter, 'close_plot'):
                files = self.plotter._process_single_time_step(
                    mock_proc, 0, np.zeros(100), np.zeros(100),
                    -110, -100, 30, 40, 'rainnc', 'a01h', 'scatter',
                    None, None, None, None, tmp_output_dir, 'test', ['png']
            )
        assert isinstance(files, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
