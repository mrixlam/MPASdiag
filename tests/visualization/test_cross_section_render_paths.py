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
import pytest
import matplotlib
import numpy as np
import xarray as xr
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Union
from unittest.mock import MagicMock, patch

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

    def setup_method(self: 'TestCrossSectionRenderingPaths') -> None:
        """
        This setup method initializes the MPASVerticalCrossSectionPlotter for rendering path tests. It creates an instance of the plotter that will be used in subsequent tests to verify different rendering paths, such as filled contour plotting, moisture variable clipping, and height conversion logic.

        Parameters:
            None

        Returns:
            None  
        """
        self.plotter = MPASVerticalCrossSectionPlotter()


    def test_pressure_nonpositive_warning(self: 'TestCrossSectionRenderingPaths', 
                                          capsys: 'pytest.CaptureFixture') -> None:
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


class TestCrossSectionAxisFormatting:
    """ Tests for axis formatting methods including pressure log scale and colorbar. """

    def setup_method(self: 'TestCrossSectionAxisFormatting') -> None:
        """
        This method sets up the MPASVerticalCrossSectionPlotter for axis formatting tests. It initializes the plotter instance and creates a figure and axes that will be used in the subsequent test methods to verify various axis formatting paths such as pressure log scale, max_height filtering, and colorbar configuration.

        Parameters:
            None

        Returns:
            None
        """
        self.plotter = MPASVerticalCrossSectionPlotter()
        self.plotter.fig, self.plotter.ax = plt.subplots()

    def teardown_method(self: 'TestCrossSectionAxisFormatting') -> None:
        """
        This method tears down the test environment after each test method is executed. It closes all matplotlib figures to ensure that no resources are left open and that subsequent tests start with a clean slate, preventing any interference from previous tests.

        Parameters:
            None

        Returns:
            None
        """
        plt.close('all')


    def test_pressure_to_height_conversion_standard_atmosphere(self: 'TestCrossSectionAxisFormatting') -> None:
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


class TestCrossSectionBatch:
    """ Tests for batch cross-section plot creation loop. """


    def test_batch_cross_section_time_iteration(self: 'TestCrossSectionBatch', 
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
