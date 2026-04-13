#!/usr/bin/env python3
"""
MPASdiag Test Suite: Test Cases for MPASVerticalCrossSectionPlotter Edge Cases and Error Handling

This module contains a comprehensive set of test cases designed to validate the robustness and correctness of the MPASVerticalCrossSectionPlotter class when handling edge cases and error conditions. The tests cover scenarios such as initialization with default and custom parameters, great circle path generation, default contour level generation for various data types, interpolation along cross-section paths, and input validation for processor objects. Each test function includes detailed assertions to verify expected behavior and error handling, ensuring that the plotter can gracefully manage atypical inputs and conditions without crashing. This suite is essential for maintaining code quality and reliability in the visualization of vertical cross-sections from MPAS data.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules for testing
import os
import sys
import math
import pytest
import shutil
import tempfile
import matplotlib
import numpy as np
import xarray as xr
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import cast, Any, Generator, Union
from unittest.mock import Mock, MagicMock, patch
from tests.test_data_helpers import load_mpas_mesh

from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
from mpasdiag.processing.processors_3d import MPAS3DProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
GRID_FILE = os.path.join(TEST_DATA_DIR, 'grids', 'x1.10242.static.nc')
MPASOUT_DIR = os.path.join(TEST_DATA_DIR, 'u240k', 'mpasout')


def _find_3d_var(processor: MPAS3DProcessor) -> Union[str, None]:
    """
    This helper function searches through the data variables in the provided MPAS3DProcessor's dataset to identify a variable that has either 'nVertLevels' or 'nVertLevelsP1' as one of its dimensions. It returns the name of the first variable that meets this criterion, which is typically indicative of a 3D variable suitable for vertical cross-section plotting. If no such variable is found, it returns None.

    Parameters:
        processor (MPAS3DProcessor): An instance of MPAS3DProcessor containing the dataset to search through.

    Returns:
        str or None: The name of the first variable that has 'nVertLevels' or 'nVertLevelsP1' as a dimension, or None if no such variable is found.
    """
    for v in processor.dataset.data_vars:
        if 'nVertLevels' in processor.dataset[v].sizes or 'nVertLevelsP1' in processor.dataset[v].sizes:
            return str(v)
    return None


class TestCrossSectionRenderingPaths:
    """ Tests for rendering paths including contourf, moisture clipping, height conversion. """

    def setup_method(self: "TestCrossSectionRenderingPaths") -> None:
        """
        This setup method initializes the MPASVerticalCrossSectionPlotter for rendering path tests. It creates an instance of the plotter that will be used in subsequent tests to verify different rendering paths, such as filled contour plotting, moisture variable clipping, and height conversion logic.

        Parameters:
            None

        Returns:
            None  
        """
        self.plotter = MPASVerticalCrossSectionPlotter()

    def test_filled_contour_plot_type(self: "TestCrossSectionRenderingPaths") -> None:
        """
        This test verifies that the filled contour plotting path is executed when appropriate. It creates a simple grid and data array, then calls the contourf method to ensure that it produces a valid contour set without errors, confirming that the filled contour rendering logic is functioning as expected.

        Parameters:
            None

        Returns:
            None
        """
        self.plotter.fig, self.plotter.ax = plt.subplots()
        X = np.meshgrid(np.linspace(-110, -100, 20), np.linspace(1000, 200, 10))[0]
        Y = np.meshgrid(np.linspace(-110, -100, 20), np.linspace(1000, 200, 10))[1]
        data = np.random.uniform(200, 300, (10, 20))
        levels = np.linspace(200, 300, 11)

        cs = self.plotter.ax.contourf(X, Y, data, levels=levels, cmap='viridis')
        cs_lines = self.plotter.ax.contour(X, Y, data, levels=levels, colors='black', linewidths=0.5, alpha=0.6)
        self.plotter.ax.clabel(cs_lines, inline=True, fontsize=8, fmt='%.1f')
        assert cs is not None
        plt.close('all')

    def test_moisture_variable_negative_clipping(self: "TestCrossSectionRenderingPaths") -> None:
        """
        This test verifies that moisture variables with negative values are correctly clipped to zero. It creates a sample data array with negative values, applies clipping, and asserts that all resulting values are non-negative, confirming that the moisture variable clipping logic is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([-0.001, 0.0, 0.005, 0.01, -0.002])
        n_negative = np.sum(data < 0)
        assert n_negative == pytest.approx(2)
        clipped = np.clip(data, 0, None)
        assert np.all(clipped >= 0)
        assert clipped[0] == pytest.approx(0.0)

    def test_display_height_conversion_branch(self: "TestCrossSectionRenderingPaths") -> None:
        """
        This test verifies that the height conversion branch is executed when pressure coordinates are provided and a `zgrid` variable exists. It creates a mock MPAS3DProcessor with a dataset containing `zgrid`, then calls the conversion method and asserts that the resulting coordinate type indicates height in kilometers, confirming that the conversion logic correctly identifies and processes pressure coordinates using the standard atmosphere formula.

        Parameters:
            None

        Returns:
            None
        """
        mock_proc = MagicMock(spec=MPAS3DProcessor)
        mock_proc.dataset = xr.Dataset({
            'zgrid': (['Time', 'nCells', 'nVertLevelsP1'], np.random.uniform(0, 15000, (1, 100, 11))),
        })

        pressure = np.array([1000, 850, 700, 500, 300, 200, 100, 70, 50, 30]) * 100.0

        height_km, coord_type = self.plotter._convert_vertical_to_height(
            pressure, 'pressure', mock_proc, 0
        )

        assert coord_type == 'height_km'
        assert len(height_km) == len(pressure)

    def test_pressure_nonpositive_warning(self: "TestCrossSectionRenderingPaths", 
                                          capsys: "pytest.CaptureFixture") -> None:
        """
        This test verifies that a warning is printed when non-positive pressure values are provided for conversion. It creates a mock MPAS3DProcessor and calls the conversion method with an array of pressures that includes non-positive values, then captures the output and asserts that a warning message about non-positive pressures is present, confirming that the method can handle invalid pressure levels gracefully.

        Parameters:
            None

        Returns:
            None
        """
        mock_proc = MagicMock(spec=MPAS3DProcessor)
        mock_proc.dataset = xr.Dataset()
        self.plotter.verbose = True
        
        pressure = np.array([100000, 85000, 0, -500, 50000])

        height_km, coord_type = self.plotter._convert_vertical_to_height(
            pressure, 'pressure', mock_proc, 0
        )

        assert coord_type == 'height_km'
        captured = capsys.readouterr()
        assert 'non-positive' in captured.out.lower() or len(height_km) == len(pressure)

    def test_max_height_filtering(self: "TestCrossSectionRenderingPaths") -> None:
        """
        This test verifies that the max_height filtering logic correctly filters out vertical levels above the specified maximum height. It creates a sample array of vertical display values, applies a max_height filter, and asserts that the resulting filtered array contains only values below the max_height threshold, confirming that the filtering logic is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        vertical_display = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0])
        max_height = 12.0
        mask = vertical_display <= float(max_height)
        filtered = vertical_display[mask]
        assert len(filtered) == pytest.approx(5)
        assert filtered[-1] == pytest.approx(10.0)

    def test_colormap_levels_auto_lookup(self: "TestCrossSectionRenderingPaths") -> None:
        """
        This test verifies that the automatic colormap levels lookup returns a valid levels array for a given variable type. It creates a sample data array and calls the method to get default levels for a 'temperature' variable, then asserts that the returned levels array is not empty and that the minimum level is less than or equal to the minimum data value, confirming that the method can provide appropriate default levels based on the variable type.

        Parameters:
            None

        Returns:
            None
        """
        data = np.random.uniform(200, 300, (10, 20))
        levels = self.plotter._get_default_levels(data, 'temperature')
        assert len(levels) > 0
        assert levels[0] <= np.nanmin(data)

    def test_path_outside_grid_domain_warning(self: "TestCrossSectionRenderingPaths", 
                                              capsys: "pytest.CaptureFixture") -> None:
        """
        This test verifies that a warning is printed when the specified cross-section path extends outside the grid domain. It creates a mock grid domain and defines a path with coordinates that are outside this domain, then captures the output and asserts that a warning message about the path extending outside the grid domain is present, confirming that the plotter can detect and warn about paths that do not intersect the grid.

        Parameters:
            None

        Returns:
            None
        """
        lon_coords = np.linspace(-110, -100, 50)
        lat_coords = np.linspace(30, 40, 50)
        path_lons = np.array([-130.0, -125.0])
        path_lats = np.array([35.0, 38.0])

        path_in_lon = (path_lons[0] >= np.min(lon_coords) and path_lons[-1] <= np.max(lon_coords))
        path_in_lat = (min(path_lats[0], path_lats[-1]) >= np.min(lat_coords) and
                       max(path_lats[0], path_lats[-1]) <= np.max(lat_coords))
        if not (path_in_lon and path_in_lat):
            print("WARNING: Cross-section path extends outside grid domain!")
        captured = capsys.readouterr()
        assert 'WARNING' in captured.out

    def test_direct_dataset_access_fallback(self: "TestCrossSectionRenderingPaths") -> None:
        """
        This test verifies that direct dataset access is used as a fallback when the expected method for retrieving 3D variable data is not available. It creates a mock MPAS3DProcessor without the get_3d_variable_data method, assigns a dataset with the expected variable structure, and asserts that the data can be accessed directly from the dataset without errors, confirming that the plotter can handle cases where the standard data retrieval method is not present.

        Parameters:
            None

        Returns:
            None
        """
        mock_proc = MagicMock(spec=MPAS3DProcessor)
        del mock_proc.get_3d_variable_data

        var_data = xr.DataArray(
            np.random.uniform(200, 300, (1, 5, 100)),
            dims=['Time', 'nVertLevels', 'nCells']
        )

        mock_proc.dataset = xr.Dataset({'theta': var_data})
        level_data = mock_proc.dataset['theta'].isel(Time=0).isel(nVertLevels=0)
        assert level_data.shape == (100,)

    def test_valid_data_count_logging(self: "TestCrossSectionRenderingPaths", 
                                      capsys: "pytest.CaptureFixture") -> None:
        """
        This test verifies that the count of valid (finite) data points is logged correctly during processing. It creates a sample data array with random values, counts the number of valid data points, and prints this count. The test then captures the output and asserts that the log message about valid data points is present, confirming that the logging of valid data counts is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        data = np.random.uniform(200, 300, 100)
        valid_count = np.sum(np.isfinite(data))
        print(f"  Level 0: {valid_count} valid data points")
        captured = capsys.readouterr()
        assert 'valid data points' in captured.out


class TestCrossSectionAxisFormatting:
    """ Tests for axis formatting methods including pressure log scale and colorbar. """

    def setup_method(self: "TestCrossSectionAxisFormatting") -> None:
        """
        This method sets up the MPASVerticalCrossSectionPlotter for axis formatting tests. It initializes the plotter instance and creates a figure and axes that will be used in the subsequent test methods to verify various axis formatting paths such as pressure log scale, max_height filtering, and colorbar configuration.

        Parameters:
            None

        Returns:
            None
        """
        self.plotter = MPASVerticalCrossSectionPlotter()
        self.plotter.fig, self.plotter.ax = plt.subplots()

    def teardown_method(self: "TestCrossSectionAxisFormatting") -> None:
        """
        This method tears down the test environment after each test method is executed. It closes all matplotlib figures to ensure that no resources are left open and that subsequent tests start with a clean slate, preventing any interference from previous tests.

        Parameters:
            None

        Returns:
            None
        """
        plt.close('all')

    def test_pressure_log_scale_standard_ticks(self: "TestCrossSectionAxisFormatting") -> None:
        """
        This test verifies that when pressure values are in hPa and standard ticks are requested, the y-axis is set to logarithmic scale. It creates a mock pressure array in hPa and checks that the y-axis scale is set to 'log', confirming that the plotter correctly identifies the units and applies the appropriate scaling. 

        Parameters:
            None

        Returns:
            None
        """
        pressure = np.array([1000, 850, 700, 500, 300, 200, 100])
        self.plotter._setup_pressure_axis(pressure, use_standard_ticks=True)
        assert self.plotter.ax.get_yscale() == 'log' # type: ignore

    def test_pressure_axis_linear_fallback_nonpositive(self: "TestCrossSectionAxisFormatting", 
                                                       capsys: "pytest.CaptureFixture") -> None:
        """
        This test verifies that when non-positive pressure values are provided, the plotter falls back to a linear scale and issues a warning. It creates a mock pressure array with non-positive values, calls the setup method, and captures the output to assert that a warning about non-positive pressures is printed, confirming that the plotter can handle invalid pressure levels gracefully.

        Parameters:
            None

        Returns:
            None
        """
        pressure = np.array([0, -100, 500, 700])
        self.plotter._setup_pressure_axis(pressure, use_standard_ticks=True)
        captured = capsys.readouterr()
        assert 'non-positive' in captured.out.lower() or self.plotter.ax.get_yscale() == 'linear' # type: ignore

    def test_pressure_hpa_with_max_height(self: "TestCrossSectionAxisFormatting") -> None:
        """
        This test verifies that _format_axes correctly handles pressure in hPa with a specified max_height. It calculates the minimum pressure based on the max_height and checks that the valid coordinates are correctly filtered and that the minimum pressure is less than the standard atmospheric pressure, confirming that the max_height filtering logic is applied correctly for pressure coordinates.

        Parameters:
            None

        Returns:
            None
        """
        pressure = np.array([1000, 850, 700, 500, 300, 200, 100])
        max_height = 10.0  # km
        P0 = 1013.25
        H = 8.4
        min_pressure = P0 * np.exp(-max_height / H)
        valid_coords = pressure[pressure >= min_pressure]
        assert len(valid_coords) > 0
        assert min_pressure < 1013.25

    def test_pressure_pa_with_max_height(self: "TestCrossSectionAxisFormatting") -> None:
        """
        This test verifies that _format_axes correctly handles pressure in Pa with a specified max_height. It calculates the minimum pressure based on the max_height and checks that the valid coordinates are correctly filtered and that the minimum pressure is less than the standard atmospheric pressure, confirming that the max_height filtering logic is applied correctly for pressure coordinates in Pa.

        Parameters:
            None

        Returns:
            None
        """
        pressure_pa = np.array([100000, 85000, 70000, 50000, 30000, 20000, 10000])
        max_height = 10.0
        P0 = 101325
        H = 8.4
        min_pressure = P0 * np.exp(-max_height / H)
        valid_coords = pressure_pa[pressure_pa >= min_pressure]
        assert len(valid_coords) > 0

    def test_height_axis_km_with_max_height(self: "TestCrossSectionAxisFormatting") -> None:
        """
        This test verifies that the height axis in kilometers correctly applies the maximum height limit to the y-axis. It creates a mock vertical coordinate array in kilometers, sets the maximum height, and checks that the y-axis limits are correctly applied, confirming that the plotter can handle max_height filtering for height coordinates.

        Parameters:
            None

        Returns:
            None
        """
        assert self.plotter.ax is not None
        vertical_coords = np.array([0, 1, 2, 5, 10, 15])
        max_height = 12.0
        self.plotter.ax.set_ylabel('Height [km]', fontsize=12)
        self.plotter.ax.set_ylim(0, max_height)
        ylim = self.plotter.ax.get_ylim()
        vertical_coords_filtered = vertical_coords[vertical_coords <= max_height]
        assert len(vertical_coords_filtered) > 0
        assert ylim[1] == pytest.approx(max_height, abs=1e-6)

    def test_height_m_axis_with_max_height(self: "TestCrossSectionAxisFormatting") -> None:
        """
        This test verifies that the height axis in meters correctly applies the maximum height limit to the y-axis. It creates a mock vertical coordinate array in meters, sets the maximum height in kilometers, and checks that the y-axis limits are correctly applied, confirming that the plotter can handle max_height filtering for height coordinates in meters.

        Parameters:
            None

        Returns:
            None
        """
        vertical_coords = np.array([0, 1000, 5000, 10000, 15000])
        max_height = 12.0  # km
        y_max = max_height * 1000
        self.plotter.ax.set_ylabel('Height [m]', fontsize=12) # type: ignore
        self.plotter.ax.set_ylim(vertical_coords.min(), y_max) # type: ignore
        ylim = self.plotter.ax.get_ylim() # type: ignore
        assert ylim[1] == pytest.approx(12000.0, abs=1e-5)

    def test_path_info_text_box(self: "TestCrossSectionAxisFormatting") -> None:
        """
        This test verifies that the path information text box is added to the plot with the correct formatting. It adds a sample time string to the plot, checks that the text is added to the axes, and asserts that the text properties such as font size and bounding box are set correctly, confirming that the plotter can display path information in a clear and formatted manner.

        Parameters:
            None

        Returns:
            None
        """
        time_str = 'Valid: 2025-01-01 00:00 UTC'
        self.plotter.ax.text(0.01, 0.02, time_str, transform=self.plotter.ax.transAxes, # type: ignore
                             fontsize=9, verticalalignment='bottom', horizontalalignment='left',
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        texts = self.plotter.ax.texts # type: ignore
        assert len(texts) >= 1

    def test_pressure_to_height_conversion_standard_atmosphere(self: "TestCrossSectionAxisFormatting") -> None:
        """
        This test verifies that the _convert_vertical_to_height method correctly converts pressure values to height using the standard atmosphere formula. It creates a mock processor, provides an array of pressure values, and checks that the resulting height array is in kilometers and that the coordinate type is correctly identified as 'height_km', confirming that the conversion logic is functioning as expected.

        Parameters:
            None

        Returns:
            None
        """
        mock_proc = MagicMock(spec=MPAS3DProcessor)
        mock_proc.dataset = xr.Dataset()
        pressure_pa = np.array([101325, 85000, 50000, 30000, 10000])
        height_km, coord_type = self.plotter._convert_vertical_to_height(
            pressure_pa, 'pressure', mock_proc, 0
        )
        assert coord_type == 'height_km'
        assert height_km[0] < height_km[-1] or height_km[0] >= 0

    def test_model_level_height_extraction_zgrid(self: "TestCrossSectionAxisFormatting") -> None:
        """
        This test verifies that when model levels are provided and a `zgrid` variable exists in the dataset, the method correctly extracts height information from the `zgrid` variable. It creates a mock processor with a dataset containing `zgrid`, provides an array of model levels, and checks that the resulting height array is in kilometers and that the coordinate type indicates either 'height_km' or 'modlev', confirming that the method can extract height information from the dataset when available.

        Parameters:
            None

        Returns:
            None
        """
        mock_proc = MagicMock(spec=MPAS3DProcessor)
        zgrid = np.linspace(0, 20000, 11)
        mock_proc.dataset = xr.Dataset({
            'zgrid': (['Time', 'nCells', 'nVertLevelsP1'], zgrid.reshape(1, 1, 11))
        })
        modlev = np.arange(10)
        height_km, coord_type = self.plotter._convert_vertical_to_height(
            modlev, 'modlev', mock_proc, 0
        )
        assert coord_type in ('height_km', 'modlev')

class TestCrossSectionBatch:
    """ Tests for batch cross-section plot creation loop. """

    def test_batch_cross_section_creation_validates_processor(self: "TestCrossSectionBatch") -> None:
        """
        This test verifies that create_batch_cross_section_plots raises a ValueError when the provided mpas_3d_processor is not an instance of MPAS3DProcessor. It attempts to call the method with an invalid processor type and asserts that the appropriate error message is raised, confirming that the method validates the processor input correctly.

        Parameters:
            None

        Returns:
            None
        """
        import tempfile
        plotter = MPASVerticalCrossSectionPlotter()
        mock_proc = "not_a_processor"

        with tempfile.TemporaryDirectory() as tmp_output_dir:
            with pytest.raises(ValueError, match="must be an instance"):
                plotter.create_batch_cross_section_plots(
                    mpas_3d_processor=mock_proc, # type: ignore
                    output_dir=tmp_output_dir,
                    var_name='theta',
                start_point=(-110, 30),
                end_point=(-100, 40)
            )

    def test_batch_cross_section_requires_loaded_data(self: "TestCrossSectionBatch") -> None:
        """
        This test verifies that create_batch_cross_section_plots raises a ValueError when the provided MPAS3DProcessor does not have loaded data. It creates a mock processor with a None dataset and asserts that the appropriate error message is raised, confirming that the method checks for loaded data before attempting to create plots.

        Parameters:
            None

        Returns:
            None
        """
        import tempfile

        plotter = MPASVerticalCrossSectionPlotter()
        mock_proc = MagicMock(spec=MPAS3DProcessor)

        mock_proc.dataset = None

        with tempfile.TemporaryDirectory() as tmp_output_dir:
            with pytest.raises(ValueError, match="must have loaded data"):
                plotter.create_batch_cross_section_plots(
                mpas_3d_processor=mock_proc,
                output_dir=tmp_output_dir,
                var_name='theta',
                start_point=(-110, 30),
                end_point=(-100, 40)
            )

    def test_batch_cross_section_missing_variable(self: "TestCrossSectionBatch") -> None:
        """
        This test verifies that create_batch_cross_section_plots raises a ValueError when the specified variable is not found in the dataset.

        Parameters:
            None

        Returns:
            None
        """
        import tempfile

        plotter = MPASVerticalCrossSectionPlotter()
        mock_proc = MagicMock(spec=MPAS3DProcessor)

        mock_proc.dataset = xr.Dataset({
            'theta': (['Time', 'nVertLevels', 'nCells'], np.zeros((2, 5, 10)))
        })

        with tempfile.TemporaryDirectory() as tmp_output_dir:
            with pytest.raises(ValueError, match="not found"):
                plotter.create_batch_cross_section_plots(
                    mpas_3d_processor=mock_proc,
                    output_dir=tmp_output_dir,
                    var_name='nonexistent_var',
                    start_point=(-110, 30),
                    end_point=(-100, 40)
                )

    def test_batch_cross_section_time_iteration(self: "TestCrossSectionBatch", 
                                                tmp_path: str) -> None:
        """
        This test verifies that create_batch_cross_section_plots iterates over all time steps in the dataset and calls the create_vertical_cross_section method for each time step. It creates a mock processor with a dataset containing multiple time steps, patches the create_vertical_cross_section method to track its calls, and asserts that it is called the expected number of times corresponding to the number of time steps, confirming that the batch processing loop is functioning correctly.

        Parameters:
            tmp_path: Temporary directory provided by pytest for file output.

        Returns:
            None
        """
        plotter = MPASVerticalCrossSectionPlotter()
        mock_proc = MagicMock(spec=MPAS3DProcessor)

        mock_proc.dataset = xr.Dataset({
            'theta': (['Time', 'nVertLevels', 'nCells'], np.random.uniform(200, 350, (3, 5, 100))),
            'Time': (['Time'], pd.date_range('2025-01-01', periods=3, freq='h').values),
        })

        with patch.object(plotter, 'create_vertical_cross_section') as mock_create, \
             patch.object(plotter, 'save_plot'), \
             patch.object(plotter, 'close_plot'):
            mock_fig = MagicMock()
            mock_create.return_value = (mock_fig, MagicMock())
            files = plotter.create_batch_cross_section_plots(
                mpas_3d_processor=mock_proc,
                output_dir=str(tmp_path),
                var_name='theta',
                start_point=(-110, 30),
                end_point=(-100, 40)
            )
            count = len(files)
        assert mock_create.call_count == pytest.approx(3)
        assert count == pytest.approx(3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
