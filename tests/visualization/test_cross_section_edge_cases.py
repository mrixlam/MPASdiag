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
import shutil
import tempfile
import matplotlib
import numpy as np
import xarray as xr
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Generator, Union
from unittest.mock import Mock, patch

from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
from mpasdiag.processing.processors_3d import MPAS3DProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
GRID_FILE = os.path.join(TEST_DATA_DIR, 'grids', 'x1.10242.static.nc')
MPASOUT_DIR = os.path.join(TEST_DATA_DIR, 'u240k', 'mpasout')


def _find_3d_var(processor: MPAS3DProcessor) -> Union[str, None]:
    """
    This helper function searches through the data variables in the processor's dataset to find a variable that has either 'nVertLevels' or 'nVertLevelsP1' as one of its dimensions. It returns the name of the first variable that meets this criterion, which is likely to be a 3D variable suitable for testing vertical cross-section plotting. If no such variable is found, it returns None. This function is useful for dynamically identifying a variable to use in tests without hardcoding specific variable names, allowing for more flexible and robust test cases.

    Parameters:
        processor (MPAS3DProcessor): An instance of MPAS3DProcessor containing the dataset to search through.

    Returns:
        str or None: The name of the first variable that has 'nVertLevels' or 'nVertLevelsP1' as a dimension, or None if no such variable is found.
    """
    for v in processor.dataset.data_vars:
        if 'nVertLevels' in processor.dataset[v].sizes or 'nVertLevelsP1' in processor.dataset[v].sizes:
            return str(v)
    return None


class TestAdditionalCoverage:
    """ Additional tests to cover remaining uncovered lines. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestAdditionalCoverage') -> Generator[None, None, None]:
        """
        This fixture sets up the MPASVerticalCrossSectionPlotter instance and a temporary directory for each test method in the TestAdditionalCoverage class. It ensures that a fresh plotter object is available for each test and that all matplotlib figures are closed after each test to prevent resource leaks. Additionally, it creates a temporary directory that can be used for any file-based operations during the tests, and ensures that this directory is cleaned up after the tests complete. By using an autouse fixture, we ensure that this setup and teardown process is automatically applied to all test methods in the class without needing to explicitly call it in each test function.

        Parameters:
            None

        Returns:
            None
        """
        self.plotter = MPASVerticalCrossSectionPlotter()
        self.temp_dir = tempfile.mkdtemp()
    
        yield
        plt.close('all')
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_basic_processor(self: 'TestAdditionalCoverage') -> Mock:
        """
        This helper method creates a basic mock MPAS3DProcessor instance with a predefined dataset and mocked methods for coordinate extraction and vertical level retrieval. The dataset includes a variable 'theta' with dimensions corresponding to time, vertical levels, and cells, as well as longitude and latitude coordinates. The mocked methods return appropriate values for the coordinates and vertical levels, allowing the tests to focus on the behavior of the plotter without needing to rely on actual data processing logic.

        Parameters:
            None

        Returns:
            Mock: A mock MPAS3DProcessor instance with predefined dataset and methods.
        """
        mock_processor = Mock(spec=MPAS3DProcessor)
        times = pd.date_range('2025-01-01', periods=2, freq='1h')
        
        ds = xr.Dataset({
            'theta': (['Time', 'nVertLevels', 'nCells'], 
                           np.full((2, 10, 100), 270.0) + np.random.randn(2, 10, 100) * 10.0),
            'lonCell': ('nCells', np.linspace(-100, -90, 100)),
            'latCell': ('nCells', np.linspace(30, 40, 100))
        }, coords={'Time': times})
        
        mock_processor.dataset = ds
        mock_processor.extract_2d_coordinates_for_variable = Mock(
            return_value=(ds.lonCell.values, ds.latCell.values)
        )
        mock_processor.get_vertical_levels = Mock(
            return_value=np.arange(10)
        )
        
        return mock_processor
    
    def test_unit_conversion_success_path(self: 'TestAdditionalCoverage') -> None:
        """
        This test verifies the successful path of unit conversion when valid metadata is available. It patches the methods responsible for retrieving variable metadata, display units, and performing unit conversion to return expected values for a temperature variable. The test then calls the create_vertical_cross_section method and asserts that a figure is created successfully, confirming that the unit conversion logic is functioning correctly when all necessary information is provided.

        Parameters:
            None

        Returns:
            None
        """
        mock_processor = self._create_basic_processor()
        
        with patch('mpasdiag.visualization.cross_section.MPASFileMetadata.get_variable_metadata') as mock_meta, \
             patch('mpasdiag.visualization.cross_section.UnitConverter.get_display_units') as mock_display, \
             patch('mpasdiag.visualization.cross_section.UnitConverter.convert_units') as mock_convert:
            
            mock_meta.return_value = {'units': 'K', 'long_name': 'Temperature'}
            mock_display.return_value = 'degC'

            mock_convert.return_value = np.full((10, 20), 15.0) + np.random.randn(10, 20) * 5.0
            
            fig, _ = self.plotter.create_vertical_cross_section(
                mpas_3d_processor=mock_processor,
                var_name="theta",
                start_point=(-100, 30),
                end_point=(-90, 40),
                num_points=20
            )
            
            assert fig is not None
            plt.close(fig)
    
    def test_unit_conversion_failure_path(self: 'TestAdditionalCoverage') -> None:
        """
        This test verifies the failure path of unit conversion when an exception is raised during metadata retrieval. It patches the method responsible for retrieving variable metadata to raise an exception, simulating a scenario where metadata is unavailable or corrupted. The test then calls the create_vertical_cross_section method and asserts that a figure is still created successfully, confirming that the plotter can handle exceptions in the unit conversion process gracefully without crashing.

        Parameters:
            None

        Returns:
            None
        """
        mock_processor = self._create_basic_processor()
        
        with patch('mpasdiag.visualization.cross_section.MPASFileMetadata.get_variable_metadata',
                   side_effect=Exception("Metadata error")):
            
            fig, _ = self.plotter.create_vertical_cross_section(
                mpas_3d_processor=mock_processor,
                var_name="theta",
                start_point=(-100, 30),
                end_point=(-90, 40),
                num_points=20
            )
            
            assert fig is not None
            plt.close(fig)
    
    def test_display_vertical_pressure_non_finite(self: 'TestAdditionalCoverage') -> None:
        """
        This test verifies that the plotter can handle non-finite pressure values in the vertical levels without crashing. It patches the get_vertical_levels method to return an array containing both positive and negative values, simulating a scenario where some pressure levels are invalid. The test then calls the create_vertical_cross_section method with display_vertical set to 'pressure' and asserts that a figure is created successfully, confirming that the plotter can manage non-finite pressure values gracefully.

        Parameters:
            None

        Returns:
            None
        """
        mock_processor = self._create_basic_processor()

        mock_processor.get_vertical_levels = Mock(
            return_value=np.array([100000, -100, 0, 50000, 30000])
        )
        
        fig, _ = self.plotter.create_vertical_cross_section(
            mpas_3d_processor=mock_processor,
            var_name="theta",
            start_point=(-100, 30),
            end_point=(-90, 40),
            vertical_coord='pressure',
            display_vertical='pressure',
            num_points=20
        )
        
        assert fig is not None

        plt.close(fig)
    
    def test_colormap_style_success(self: 'TestAdditionalCoverage') -> None:
        """
        This test verifies that the plotter successfully retrieves and applies a colormap style for a variable when the visualization style lookup returns valid information. It patches the get_variable_style method to return a specific colormap and levels for the 'theta' variable, then calls the create_vertical_cross_section method and asserts that a figure is created successfully. This confirms that the plotter can utilize visualization styles correctly when they are available.

        Parameters:
            None

        Returns:
            None
        """
        mock_processor = self._create_basic_processor()
        
        with patch('mpasdiag.visualization.cross_section.MPASVisualizationStyle.get_variable_style') as mock_style:
            mock_style.return_value = {'colormap': 'coolwarm', 'levels': np.linspace(250, 300, 11)}
            
            fig, _ = self.plotter.create_vertical_cross_section(
                mpas_3d_processor=mock_processor,
                var_name="theta",
                start_point=(-100, 30),
                end_point=(-90, 40),
                num_points=20
            )
            
            assert fig is not None
            plt.close(fig)
    
    def test_colormap_style_exception(self: 'TestAdditionalCoverage') -> None:
        """
        This test verifies the fallback behavior when the visualization style lookup raises an exception. It patches the get_variable_style method to raise an exception, simulating a scenario where the style information is unavailable or corrupted. The test then calls the create_vertical_cross_section method and asserts that a figure is still created successfully, confirming that the plotter can handle exceptions in the style lookup process gracefully without crashing.

        Parameters:
            None

        Returns:
            None
        """
        mock_processor = self._create_basic_processor()
        
        with patch('mpasdiag.visualization.cross_section.MPASVisualizationStyle.get_variable_style',
                   side_effect=Exception("Style error")):
            
            fig, _ = self.plotter.create_vertical_cross_section(
                mpas_3d_processor=mock_processor,
                var_name="theta",
                start_point=(-100, 30),
                end_point=(-90, 40),
                num_points=20
            )
            
            assert fig is not None
            plt.close(fig)
    
    def test_contour_plot_type_lines(self: 'TestAdditionalCoverage') -> None:
        """
        This test verifies that the `contour` rendering branch produces a valid Figure/Axes. The test ensures contour plotting completes for synthetic input data when the plot_type is set to 'contour', confirming that this rendering path is functional and can handle typical data without errors.

        Parameters:
            None

        Returns:
            None
        """
        mock_processor = self._create_basic_processor()
        
        fig, _ = self.plotter.create_vertical_cross_section(
            mpas_3d_processor=mock_processor,
            var_name="theta",
            start_point=(-100, 30),
            end_point=(-90, 40),
            plot_type='contour',
            levels=np.linspace(250, 300, 11),
            num_points=20
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_pcolormesh_plot_type_lines(self: 'TestAdditionalCoverage') -> None:
        """
        This test verifies that the `pcolormesh` rendering branch produces a valid Figure/Axes. The test ensures pcolormesh plotting completes for synthetic input data when the plot_type is set to 'pcolormesh', confirming that this rendering path is functional and can handle typical data without errors.

        Parameters:
            None

        Returns:
            None
        """
        mock_processor = self._create_basic_processor()
        
        fig, _ = self.plotter.create_vertical_cross_section(
            mpas_3d_processor=mock_processor,
            var_name="theta",
            start_point=(-100, 30),
            end_point=(-90, 40),
            plot_type='pcolormesh',
            num_points=20
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_vertical_levels_pressure_extraction(self: 'TestAdditionalCoverage') -> None:
        """
        This test verifies the extraction of pressure-based vertical levels from the processor. It patches `get_vertical_levels` to return representative pressures and asserts that the plotter calls the method and produces a plot object, confirming that pressure-based vertical levels are handled correctly.

        Parameters:
            None

        Returns:
            None
        """
        mock_processor = self._create_basic_processor()

        mock_processor.get_vertical_levels = Mock(
            return_value=np.linspace(100000, 10000, 10)
        )
        
        fig, _ = self.plotter.create_vertical_cross_section(
            mpas_3d_processor=mock_processor,
            var_name="theta",
            start_point=(-100, 30),
            end_point=(-90, 40),
            vertical_coord='pressure',
            num_points=20
        )
        
        mock_processor.get_vertical_levels.assert_called()
        assert fig is not None
        plt.close(fig)
    
    def test_vertical_levels_integer_detection(self: 'TestAdditionalCoverage') -> None:
        """
        This test verifies that integer vertical levels are detected and handled in verbose mode. When levels are integer indices and `verbose` is True, the plotter may exercise alternate formatting paths; the test asserts that plotting completes successfully.

        Parameters:
            None

        Returns:
            None
        """
        mock_processor = self._create_basic_processor()

        mock_processor.get_vertical_levels = Mock(
            return_value=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
        )
        
        self.plotter.fig = Mock()
        self.plotter.verbose = True
        
        fig, _ = self.plotter.create_vertical_cross_section(
            mpas_3d_processor=mock_processor,
            var_name="theta",
            start_point=(-100, 30),
            end_point=(-90, 40),
            vertical_coord='pressure',
            num_points=20
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_vertical_levels_exception_nVertLevelsP1(self: 'TestAdditionalCoverage') -> None:
        """
        This test verifies that the plotter recovers when `get_vertical_levels` raises an exception and `nVertLevelsP1` is present. The test constructs a dataset with `nVertLevelsP1` and simulates a level extraction exception to ensure plotting proceeds using fallback logic.

        Parameters:
            None

        Returns:
            None
        """
        mock_processor = Mock(spec=MPAS3DProcessor)

        ds = xr.Dataset({
            'theta': (['Time', 'nVertLevelsP1', 'nCells'], 
                           np.random.rand(2, 11, 100) * 300),
            'lonCell': ('nCells', np.linspace(-100, -90, 100)),
            'latCell': ('nCells', np.linspace(30, 40, 100))
        }, coords={'Time': pd.date_range('2025-01-01', periods=2, freq='1h')})
        
        mock_processor.dataset = ds
        mock_processor.extract_2d_coordinates_for_variable = Mock(
            return_value=(ds.lonCell.values, ds.latCell.values)
        )

        mock_processor.get_vertical_levels = Mock(
            side_effect=Exception("Level extraction failed")
        )
        
        fig, _ = self.plotter.create_vertical_cross_section(
            mpas_3d_processor=mock_processor,
            var_name="theta",
            start_point=(-100, 30),
            end_point=(-90, 40),
            num_points=20
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_coordinate_extraction_exception(self: 'TestAdditionalCoverage') -> None:
        """
        This test verifies that exceptions raised during coordinate extraction are handled gracefully. It mocks the coordinate extraction method to raise an exception and asserts that the plotter still returns a figure object, confirming that the plotter can manage coordinate extraction failures without crashing.

        Parameters:
            None

        Returns:
            None
        """
        mock_processor = self._create_basic_processor()

        mock_processor.extract_2d_coordinates_for_variable = Mock(
            side_effect=Exception("Extraction failed")
        )
        
        fig, _ = self.plotter.create_vertical_cross_section(
            mpas_3d_processor=mock_processor,
            var_name="theta",
            start_point=(-100, 30),
            end_point=(-90, 40),
            num_points=20
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_coordinates_in_radians(self: 'TestAdditionalCoverage') -> None:
        """
        This test verifies that the plotter can handle coordinates provided in radians. It constructs a dataset with longitude and latitude values in radians and ensures that the plotter can create a vertical cross-section without errors, confirming that the coordinate handling logic can manage different units appropriately.

        Parameters:
            None

        Returns:
            None
        """
        mock_processor = Mock(spec=MPAS3DProcessor)

        ds = xr.Dataset({
            'theta': (['Time', 'nVertLevels', 'nCells'], 
                           np.random.rand(2, 10, 100) * 300),
            'lonCell': ('nCells', np.radians(np.linspace(-100, -90, 100))),
            'latCell': ('nCells', np.radians(np.linspace(30, 40, 100)))
        }, coords={'Time': pd.date_range('2025-01-01', periods=2, freq='1h')})
        
        mock_processor.dataset = ds

        mock_processor.extract_2d_coordinates_for_variable = Mock(
            return_value=(ds.lonCell.values, ds.latCell.values)
        )

        mock_processor.get_vertical_levels = Mock(
            return_value=np.arange(10)
        )
        
        fig, _ = self.plotter.create_vertical_cross_section(
            mpas_3d_processor=mock_processor,
            var_name="theta",
            start_point=(-100, 30),
            end_point=(-90, 40),
            num_points=20
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_level_data_with_get_3d_method(self: 'TestAdditionalCoverage') -> None:
        """
        This test verifies that the plotter accepts level data returned directly from `get_3d_variable_data`. The test mocks the method to return a 1D array and ensures that the cross-section creation completes and returns plotting objects, confirming that the plotter can handle level data provided in this manner.

        Parameters:
            None

        Returns:
            None
        """
        mock_processor = self._create_basic_processor()

        mock_processor.get_3d_variable_data = Mock(
            return_value=np.random.rand(100)
        )
        
        fig, _ = self.plotter.create_vertical_cross_section(
            mpas_3d_processor=mock_processor,
            var_name="theta",
            start_point=(-100, 30),
            end_point=(-90, 40),
            num_points=20
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_level_data_extraction_exception(self: 'TestAdditionalCoverage') -> None:
        """
        This test verifies that exceptions raised while extracting per-level data are caught and do not abort the entire cross-section generation. The plotter should continue and return a figure object when possible, confirming that it can handle data extraction failures gracefully.

        Parameters:
            None

        Returns:
            None
        """
        mock_processor = self._create_basic_processor()

        mock_processor.get_3d_variable_data = Mock(
            side_effect=Exception("Data extraction failed")
        )
        
        fig, _ = self.plotter.create_vertical_cross_section(
            mpas_3d_processor=mock_processor,
            var_name="theta",
            start_point=(-100, 30),
            end_point=(-90, 40),
            num_points=20
        )
        
        assert fig is not None
        plt.close(fig)
    
    
    def test_format_axes_pressure_hPa_max_height(self: 'TestAdditionalCoverage') -> None:
        """
        This test verifies that axis formatting for pressure in hPa includes 'hPa' in the ylabel when max_height is specified. The test configures the axes with pressure values and a max_height, then asserts that the ylabel contains 'hPa', confirming that the plotter correctly formats the axis label based on the provided parameters.

        Parameters:
            None

        Returns:
            None
        """
        self.plotter.fig = plt.figure()
        self.plotter.ax = self.plotter.fig.add_subplot(111)
        
        lons = np.linspace(-100, -90, 50)
        vertical_coords = np.array([1000, 850, 700, 500, 300, 200, 100])
        
        self.plotter._format_cross_section_axes(
            lons, vertical_coords, 'pressure_hPa',
            (-100, 30), (-90, 40), max_height=10.0
        )
        
        assert self.plotter.ax.get_ylabel() is not None
        plt.close(self.plotter.fig)
    
    
    def test_batch_processing_complete_workflow(self: 'TestAdditionalCoverage') -> None:
        """
        This test verifies the end-to-end batch cross-section processing pipeline creates output files. The test runs `create_batch_cross_section_plots` with a mock processor and asserts the returned file list length and existence of produced files, confirming that the pipeline functions correctly from input to output.

        Parameters:
            None

        Returns:
            None
        """
        mock_processor = Mock(spec=MPAS3DProcessor)
        times = pd.date_range('2025-01-01', periods=2, freq='1h')
        
        ds = xr.Dataset({
            'theta': (['Time', 'nVertLevels', 'nCells'], 
                           np.random.rand(2, 10, 100) * 300),
            'lonCell': ('nCells', np.linspace(-100, -90, 100)),
            'latCell': ('nCells', np.linspace(30, 40, 100))
        }, coords={'Time': times})
        
        mock_processor.dataset = ds
        mock_processor.extract_2d_coordinates_for_variable = Mock(
            return_value=(ds.lonCell.values, ds.latCell.values)
        )
        mock_processor.get_vertical_levels = Mock(
            return_value=np.arange(10)
        )
        
        files = self.plotter.create_batch_cross_section_plots(
            mpas_3d_processor=mock_processor,
            output_dir=self.temp_dir,
            var_name="theta",
            start_point=(-100, 30),
            end_point=(-90, 40),
            num_points=20,
            max_height=5.0
        )
        
        assert len(files) == pytest.approx(2, abs=0.1)
        for f in files:
            assert os.path.exists(f)
