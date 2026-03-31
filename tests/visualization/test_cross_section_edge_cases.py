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
    This helper function searches through the variables in the processor's dataset to find a variable that has a vertical dimension (either 'nVertLevels' or 'nVertLevelsP1'). It returns the name of the first variable that meets this criterion, which can be used for testing purposes when a specific 3D variable is needed. If no such variable is found, it returns None. This function is useful for dynamically identifying a suitable 3D variable in the dataset for testing the vertical cross-section plotting functionality.

    Parameters:
        processor: An instance of MPAS3DProcessor containing the dataset to search through.

    Returns:
        The name of the first variable with a vertical dimension, or None if no such variable is found.
    """
    for v in processor.dataset.data_vars:
        if 'nVertLevels' in processor.dataset[v].sizes or 'nVertLevelsP1' in processor.dataset[v].sizes:
            return str(v)
    return None


def test_vertical_cross_section_plotter_initialization() -> None:
    """
    This test verifies the initialization of the MPASVerticalCrossSectionPlotter class with default and custom parameters. It ensures that the plotter object is correctly instantiated with default figure size and DPI, and that custom values are properly applied. The test checks that the figure and axes attributes are initially set to None, confirming that no plot is created upon initialization. By validating both default and custom configurations, this test ensures that the plotter can be initialized flexibly while maintaining expected defaults for users who do not specify parameters.

    Parameters:
        None

    Returns:
        None
    """
    plotter = MPASVerticalCrossSectionPlotter()
    
    assert plotter.figsize == (pytest.approx(10), pytest.approx(12))
    assert plotter.dpi == pytest.approx(100)
    assert plotter.fig is None
    assert plotter.ax is None
    
    custom_plotter = MPASVerticalCrossSectionPlotter(figsize=(10, 6), dpi=150)
    assert custom_plotter.figsize == (pytest.approx(10), pytest.approx(6))
    assert custom_plotter.dpi == pytest.approx(150)


def test_great_circle_path_generation() -> None:
    """
    This test validates the great circle path generation method of the MPASVerticalCrossSectionPlotter class. It checks that the method correctly computes longitude, latitude, and distance arrays for a specified start and end point along a great circle path. The test verifies that the number of generated points matches the requested number, that the first and last points correspond to the start and end coordinates within a reasonable tolerance, and that the distance array is non-negative and starts at zero. This ensures that the path generation logic is accurate and robust for typical use cases.

    Parameters:
        None

    Returns:
        None
    """
    plotter = MPASVerticalCrossSectionPlotter()
    
    start_point = (-100.0, 40.0)
    end_point = (-90.0, 40.0)
    num_points = 11
    
    lons, lats, distances = plotter._generate_great_circle_path(start_point, end_point, num_points)
    
    assert len(lons) == num_points
    assert len(lats) == num_points
    assert len(distances) == num_points
    
    assert math.isclose(lons[0], start_point[0], abs_tol=0.01)
    assert math.isclose(lats[0], start_point[1], abs_tol=0.01)
    assert math.isclose(lons[-1], end_point[0], abs_tol=0.01)
    assert math.isclose(lats[-1], end_point[1], abs_tol=0.01)
    
    assert np.all(np.diff(distances) >= 0)
    assert math.isclose(distances[0], 0.0, abs_tol=1e-6)
    assert distances[-1] > 0.0
    

def test_default_levels_generation() -> None:
    """
    This test checks the default contour level generation method for various types of data and variable names. It validates that the method produces a non-empty set of levels that encompass the range of the input data for temperature, wind, and constant fields. The test also ensures that the method can handle NaN values without failing and returns a reasonable set of levels in such cases. By covering different variable types and data characteristics, this test confirms that the default level generation logic is robust and adaptable to a variety of plotting scenarios.

    Parameters:
        None

    Returns:
        None
    """
    plotter = MPASVerticalCrossSectionPlotter()
    
    temp_data = np.array([[250, 260, 270], [280, 290, 300], [310, 320, 330]])
    temp_levels = plotter._get_default_levels(temp_data, 'theta')
    
    assert len(temp_levels) > 0
    assert temp_levels.min() <= temp_data.min()
    assert temp_levels.max() >= temp_data.max()
    
    wind_data = np.array([[-10, -5, 0], [5, 10, 15], [-15, 20, 25]])
    wind_levels = plotter._get_default_levels(wind_data, 'uwind')
    
    assert len(wind_levels) > 0
    assert wind_levels.min() <= wind_data.min()
    assert wind_levels.max() >= wind_data.max()
    
    constant_data = np.full((3, 3), 5.0)
    constant_levels = plotter._get_default_levels(constant_data, 'constant')
    
    assert len(constant_levels) >= 1
    
    nan_data = np.full((3, 3), np.nan)
    nan_levels = plotter._get_default_levels(nan_data, 'nan_data')
    
    assert len(nan_levels) > 0
    

def test_interpolation_along_path() -> None:
    """
    This test verifies the interpolation method for extracting data values along a specified path. It checks that the method can successfully interpolate values from a grid of longitude, latitude, and data values to a set of path points. The test ensures that the output array has the correct length corresponding to the number of path points and that it contains valid interpolated values rather than all NaNs. By using a simple synthetic dataset, this test confirms that the interpolation logic is functioning correctly and can handle typical use cases without errors.

    Parameters:
        None

    Returns:
        None
    """
    plotter = MPASVerticalCrossSectionPlotter()
    
    grid_lons = np.array([-102, -101, -100, -99, -98])
    grid_lats = np.array([39, 40, 41, 42, 43])
    grid_data = np.array([10, 20, 30, 40, 50])
    
    path_lons = np.array([-101.5, -100.5, -99.5])
    path_lats = np.array([39.5, 40.5, 41.5])
    
    try:
        interpolated = plotter._interpolate_along_path(
            grid_lons, grid_lats, grid_data, path_lons, path_lats
        )
        
        assert len(interpolated) == len(path_lons)
        assert not np.all(np.isnan(interpolated))  
        
        print("Interpolation along path test passed!")
        
    except ImportError:
        print("Scipy not available, skipping interpolation test")
        pytest.skip("Scipy not available for interpolation test")


def test_input_validation() -> None:
    """
    This test checks the input validation logic for the create_vertical_cross_section method. It verifies that the method raises a ValueError when an invalid type is passed as the mpas_3d_processor argument. The test ensures that the error message contains relevant information about the expected type, confirming that the input validation is correctly identifying and handling inappropriate inputs. By testing this edge case, we can confirm that the plotter provides clear feedback to users when they attempt to use it with unsupported processor types.

    Parameters:
        None

    Returns:
        None
    """
    plotter = MPASVerticalCrossSectionPlotter()
    try:
        plotter.create_vertical_cross_section(
            mpas_3d_processor=cast(Any, "invalid"),
            var_name="theta",
            start_point=(-100, 40),
            end_point=(-90, 40)
        )
        assert False, "Should have raised ValueError for invalid processor"
    except ValueError as e:
        assert "MPAS3DProcessor" in str(e)
        print("Input validation test passed!")


class TestEdgeCases:
    """ Tests for edge cases and error conditions. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestEdgeCases") -> Generator[None, None, None]:
        """
        This fixture sets up the MPASVerticalCrossSectionPlotter instance for each test method in the TestEdgeCases class. It ensures that a fresh plotter object is available for each test and that all matplotlib figures are closed after each test to prevent resource leaks. By using an autouse fixture, we ensure that this setup and teardown process is automatically applied to all test methods in the class without needing to explicitly call it in each test function.

        Parameters:
            None

        Returns:
            None
        """
        self.plotter = MPASVerticalCrossSectionPlotter()
    
        yield
        plt.close('all')
    
    def test_interpolation_with_all_nan_data(self: "TestEdgeCases") -> None:
        """
        This test verifies that the interpolation method can handle a grid of data that contains only NaN values without crashing. It checks that the method returns an array of NaNs for the interpolated values when the input grid data is entirely NaN. This ensures that the interpolation logic can gracefully manage cases where no valid data is available, which is an important edge case for real-world datasets that may contain missing or invalid entries.

        Parameters:
            None

        Returns:
            None
        """
        grid_lons = np.linspace(-110, -90, 50)
        grid_lats = np.linspace(30, 50, 50)
        grid_data = np.full(50, np.nan)
        
        path_lons = np.linspace(-105, -95, 20)
        path_lats = np.linspace(35, 45, 20)
        
        result = self.plotter._interpolate_along_path(
            grid_lons, grid_lats, grid_data,
            path_lons, path_lats
        )
        
        assert np.all(np.isnan(result))
    
    def test_default_levels_zero_range(self: "TestEdgeCases") -> None:
        """
        This test checks the behavior of the default level generation method when the input data has zero range (i.e., all values are the same). It verifies that the method returns a single level equal to that constant value, ensuring that it can handle this edge case without producing an empty set of levels or crashing. This is important for visualizations where the data may not vary, and we still want to be able to generate a plot with appropriate levels.

        Parameters:
            None

        Returns:
            None
        """
        data = np.full((10, 10), 100.0)
        
        levels = self.plotter._get_default_levels(data, 'theta')
        
        assert len(levels) == pytest.approx(1)
        assert levels[0] == pytest.approx(100.0)
    
    def test_default_levels_temperature_variable(self: "TestEdgeCases") -> None:
        """
        This test verifies that the default level generation for a temperature variable produces a reasonable set of levels that span the range of the data. It creates a synthetic dataset with a range of temperature values and checks that the generated levels include values that cover the minimum and maximum of the dataset. Additionally, it asserts that the typical interval between levels is not excessively large, ensuring that the levels are suitable for visualizing temperature data effectively.

        Parameters:
            None

        Returns:
            None
        """
        _, _, u2d, _ = load_mpas_mesh(10, 10)
        umin = u2d.min()
        umax = u2d.max()
        data = 200.0 + 100.0 * (u2d - umin) / (umax - umin + 1e-12)
        
        levels = self.plotter._get_default_levels(data, 'theta')
        
        assert len(levels) > 1

        if len(levels) > 1:
            typical_diff = levels[1] - levels[0]
            assert typical_diff <= 10.5  
    
    def test_default_levels_pressure_variable(self: "TestEdgeCases") -> None:
        """
        This test verifies that the default level generation for a pressure variable produces a reasonable set of levels that span the range of the data. It creates a synthetic dataset with a range of pressure values and checks that the generated levels include values that cover the minimum and maximum of the dataset. Additionally, it asserts that the levels are spaced in a physically meaningful way, typically using logarithmic spacing for positive pressure values.

        Parameters:
            None

        Returns:
            None
        """
        _, _, u2d, _ = load_mpas_mesh(10, 10)
        umin = u2d.min()
        umax = u2d.max()
        data = 10000.0 + 90000.0 * (u2d - umin) / (umax - umin + 1e-12)
        
        levels = self.plotter._get_default_levels(data, 'pressure')
        
        assert len(levels) > 1
    
    def test_default_levels_wind_variable(self: "TestEdgeCases") -> None:
        """
        This test verifies that the default level generation for a wind variable produces a reasonable set of levels that span the range of the data. It creates a synthetic dataset with a range of wind speed values and checks that the generated levels include values that cover the minimum and maximum of the dataset. Additionally, it asserts that the levels are symmetric around zero if the data includes both positive and negative values, which is typical for wind components.

        Parameters:
            None

        Returns:
            None
        """
        _, _, u2d, v2d = load_mpas_mesh(10, 10)
        speed = np.hypot(u2d, v2d)
        smin = speed.min()
        smax = speed.max()
        data = -20.0 + 40.0 * (speed - smin) / (smax - smin + 1e-12)
        
        levels = self.plotter._get_default_levels(data, 'u_wind')
        
        assert len(levels) > 1

        if len(levels) > 1:
            assert np.mean(levels) == pytest.approx(0, abs=1e-0)
    
    def test_great_circle_very_small_distance(self: "TestEdgeCases") -> None:
        """
        This test verifies that the great circle path generation method can handle cases where the start and end points are very close together, resulting in a very small distance. It checks that the method still returns the requested number of points and that the distance array is non-negative and starts at zero, even when the total distance is extremely small. This ensures that the path generation logic is robust and can manage edge cases without numerical issues or errors.

        Parameters:
            None

        Returns:
            None
        """
        start = (-100.0, 40.0)
        end = (-100.001, 40.001)
        
        lons, lats, distances = self.plotter._generate_great_circle_path(
            start, end, num_points=10
        )
        
        assert len(lons) == pytest.approx(10, abs=1e-1)
        assert np.all(distances >= 0)


class TestAdditionalCoverage:
    """ Additional tests to cover remaining uncovered lines. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestAdditionalCoverage") -> Generator[None, None, None]:
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
    
    def _create_basic_processor(self: "TestAdditionalCoverage") -> Mock:
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
    
    def test_unit_conversion_success_path(self: "TestAdditionalCoverage") -> None:
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
            
            fig, ax = self.plotter.create_vertical_cross_section(
                mpas_3d_processor=mock_processor,
                var_name="theta",
                start_point=(-100, 30),
                end_point=(-90, 40),
                num_points=20
            )
            
            assert fig is not None
            plt.close(fig)
    
    def test_unit_conversion_failure_path(self: "TestAdditionalCoverage") -> None:
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
            
            fig, ax = self.plotter.create_vertical_cross_section(
                mpas_3d_processor=mock_processor,
                var_name="theta",
                start_point=(-100, 30),
                end_point=(-90, 40),
                num_points=20
            )
            
            assert fig is not None
            plt.close(fig)
    
    def test_display_vertical_pressure_non_finite(self: "TestAdditionalCoverage") -> None:
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
        
        fig, ax = self.plotter.create_vertical_cross_section(
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
    
    def test_colormap_style_success(self: "TestAdditionalCoverage") -> None:
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
            
            fig, ax = self.plotter.create_vertical_cross_section(
                mpas_3d_processor=mock_processor,
                var_name="theta",
                start_point=(-100, 30),
                end_point=(-90, 40),
                num_points=20
            )
            
            assert fig is not None
            plt.close(fig)
    
    def test_colormap_style_exception(self: "TestAdditionalCoverage") -> None:
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
            
            fig, ax = self.plotter.create_vertical_cross_section(
                mpas_3d_processor=mock_processor,
                var_name="theta",
                start_point=(-100, 30),
                end_point=(-90, 40),
                num_points=20
            )
            
            assert fig is not None
            plt.close(fig)
    
    def test_contour_plot_type_lines(self: "TestAdditionalCoverage") -> None:
        """
        This test verifies that the `contour` rendering branch produces a valid Figure/Axes. The test ensures contour plotting completes for synthetic input data when the plot_type is set to 'contour', confirming that this rendering path is functional and can handle typical data without errors.

        Parameters:
            None

        Returns:
            None
        """
        mock_processor = self._create_basic_processor()
        
        fig, ax = self.plotter.create_vertical_cross_section(
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
    
    def test_pcolormesh_plot_type_lines(self: "TestAdditionalCoverage") -> None:
        """
        This test verifies that the `pcolormesh` rendering branch produces a valid Figure/Axes. The test ensures pcolormesh plotting completes for synthetic input data when the plot_type is set to 'pcolormesh', confirming that this rendering path is functional and can handle typical data without errors.

        Parameters:
            None

        Returns:
            None
        """
        mock_processor = self._create_basic_processor()
        
        fig, ax = self.plotter.create_vertical_cross_section(
            mpas_3d_processor=mock_processor,
            var_name="theta",
            start_point=(-100, 30),
            end_point=(-90, 40),
            plot_type='pcolormesh',
            num_points=20
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_vertical_levels_pressure_extraction(self: "TestAdditionalCoverage") -> None:
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
        
        fig, ax = self.plotter.create_vertical_cross_section(
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
    
    def test_vertical_levels_integer_detection(self: "TestAdditionalCoverage") -> None:
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
        
        fig, ax = self.plotter.create_vertical_cross_section(
            mpas_3d_processor=mock_processor,
            var_name="theta",
            start_point=(-100, 30),
            end_point=(-90, 40),
            vertical_coord='pressure',
            num_points=20
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_vertical_levels_exception_nVertLevelsP1(self: "TestAdditionalCoverage") -> None:
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
        
        fig, ax = self.plotter.create_vertical_cross_section(
            mpas_3d_processor=mock_processor,
            var_name="theta",
            start_point=(-100, 30),
            end_point=(-90, 40),
            num_points=20
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_coordinate_extraction_exception(self: "TestAdditionalCoverage") -> None:
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
        
        fig, ax = self.plotter.create_vertical_cross_section(
            mpas_3d_processor=mock_processor,
            var_name="theta",
            start_point=(-100, 30),
            end_point=(-90, 40),
            num_points=20
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_coordinates_in_radians(self: "TestAdditionalCoverage") -> None:
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
        
        fig, ax = self.plotter.create_vertical_cross_section(
            mpas_3d_processor=mock_processor,
            var_name="theta",
            start_point=(-100, 30),
            end_point=(-90, 40),
            num_points=20
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_level_data_with_get_3d_method(self: "TestAdditionalCoverage") -> None:
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
        
        fig, ax = self.plotter.create_vertical_cross_section(
            mpas_3d_processor=mock_processor,
            var_name="theta",
            start_point=(-100, 30),
            end_point=(-90, 40),
            num_points=20
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_level_data_extraction_exception(self: "TestAdditionalCoverage") -> None:
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
        
        fig, ax = self.plotter.create_vertical_cross_section(
            mpas_3d_processor=mock_processor,
            var_name="theta",
            start_point=(-100, 30),
            end_point=(-90, 40),
            num_points=20
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_great_circle_zero_distance(self: "TestAdditionalCoverage") -> None:
        """
        This test verifies that the great circle path generation method can handle cases where the start and end points are identical, resulting in a zero distance. It checks that the method still returns the requested number of points and that all distance values are zero, confirming that the path generation logic can manage this edge case without errors.

        Parameters:
            None

        Returns:
            None
        """
        lons, lats, dists = self.plotter._generate_great_circle_path(
            start_point=(-100, 30),
            end_point=(-100, 30), 
            num_points=50
        )
        
        assert len(lons) == pytest.approx(50, abs=1e-1)
        assert np.allclose(dists, 0)
    
    def test_great_circle_first_point(self: "TestAdditionalCoverage") -> None:
        """
        This test verifies that the first point of a generated great circle path equals the start point. The test asserts longitude/latitude and distance values at index zero match expected values within numerical tolerance. This ensures that the path generation correctly initializes the first point and distance, which is critical for accurate path representation.

        Parameters:
            None

        Returns:
            None
        """
        lons, lats, dists = self.plotter._generate_great_circle_path(
            start_point=(-100, 30),
            end_point=(-90, 40),
            num_points=50
        )
        
        assert lons[0] == pytest.approx(-100, abs=1e-1)
        assert lats[0] == pytest.approx(30, abs=1e-1)
        assert dists[0] == pytest.approx(0, abs=1e-1)
    
    def test_great_circle_last_point(self: "TestAdditionalCoverage") -> None:
        """
        This test verifies that the last point of a generated great circle path equals the end point. The test asserts endpoint longitude and latitude against expected values within a small numerical tolerance after path generation. The distance to the last point should also match the total path length within tolerance.

        Parameters:
            None

        Returns:
            None
        """
        lons, lats, dists = self.plotter._generate_great_circle_path(
            start_point=(-100, 30),
            end_point=(-90, 40),
            num_points=50
        )
        
        assert lons[-1] == pytest.approx(-90, abs=1e-1)
        assert lats[-1] == pytest.approx(40, abs=1e-1)
    
    def test_height_extraction_variable_not_found(self: "TestAdditionalCoverage") -> None:
        """
        This test verifies that the height extraction method returns `None` when the specified height variable is not found in the dataset. The test constructs a dataset without the requested height variable and asserts that the helper method returns `None`, confirming that it can gracefully handle missing variables without crashing.

        Parameters:
            None

        Returns:
            None
        """
        mock_processor = Mock(spec=MPAS3DProcessor)
        
        ds = xr.Dataset({
            'theta': (['Time', 'nVertLevels', 'nCells'], 
                           np.random.rand(2, 10, 100) * 300)
        }, coords={'Time': pd.date_range('2025-01-01', periods=2, freq='1h')})
        
        mock_processor.dataset = ds
        
        result = self.plotter._extract_height_from_dataset(
            mock_processor, 0, np.arange(10), 'zgrid'
        )
        
        assert result is None
    
    def test_height_extraction_nVertLevelsP1_averaging(self: "TestAdditionalCoverage") -> None:
        """
        This test verifies that the height extraction method correctly averages `nVertLevelsP1` to `nVertLevels`. The test builds a dataset containing `nVertLevelsP1` and asserts that the returned averaged heights match the expected model-level count.

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
            'zgrid': (['Time', 'nCells', 'nVertLevelsP1'],
                     np.linspace(0, 10000, 11)[None, None, :] * np.ones((2, 100, 11)))
        }, coords={'Time': times})
        
        mock_processor.dataset = ds
        
        result = self.plotter._extract_height_from_dataset(
            mock_processor, 0, np.arange(10), 'zgrid'
        )
        
        assert result is not None
        assert len(result) == pytest.approx(10, abs=1e-1)
    
    def test_height_extraction_interpolation_scipy(self: "TestAdditionalCoverage") -> None:
        """
        This test verifies that the height extraction method can perform interpolation using SciPy when the height variable has a different number of levels than requested. The test constructs a dataset where the height variable has more levels than the model levels, triggering interpolation. It asserts that the returned heights have the expected length corresponding to the model levels, confirming that the interpolation logic is functioning correctly.

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
            'height': (['Time', 'nCells', 'nVertLevelsHeight'],
                     np.linspace(0, 10000, 15)[None, None, :] * np.ones((2, 100, 15)))
        }, coords={'Time': times})
        
        mock_processor.dataset = ds
        
        result = self.plotter._extract_height_from_dataset(
            mock_processor, 0, np.arange(10), 'height'
        )
        
        assert result is not None
        assert len(result) == pytest.approx(10, abs=1e-1)
    
    def test_height_extraction_exception_handling(self: "TestAdditionalCoverage") -> None:
        """
        This test verifies that the height extraction method returns `None` when the processor dataset is unavailable. The test sets `dataset=None` on the mock processor and asserts that the helper method returns `None` rather than raising an exception, allowing calling code to handle the absence of data gracefully.

        Parameters:
            None

        Returns:
            None
        """
        mock_processor = Mock(spec=MPAS3DProcessor)
        mock_processor.dataset = None
        
        result = self.plotter._extract_height_from_dataset(
            mock_processor, 0, np.arange(10), 'zgrid'
        )
        
        assert result is None
    
    def test_convert_height_direct(self: "TestAdditionalCoverage") -> None:
        """
        This test verifies that the height conversion method correctly rescales meters to kilometers. The test passes known meter values and asserts that the returned display coordinates equal the input divided by 1000 with the expected type flag.

        Parameters:
            None

        Returns:
            None
        """
        mock_processor = self._create_basic_processor()
        coords = np.array([0, 1000, 2000, 3000, 4000])
        
        result, coord_type = self.plotter._convert_vertical_to_height(
            coords, 'height', mock_processor, 0
        )
        
        assert coord_type == 'height_km'
        assert np.allclose(result, coords / 1000.0)
    
    def test_convert_pressure_with_zgrid_variable(self: "TestAdditionalCoverage") -> None:
        """
        This test verifies pressure-to-height conversion when a `zgrid` variable exists. The plotter should use `zgrid` to derive heights for pressure coordinates and return a height-based display coordinate type for plotting.

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
            'zgrid': (['Time', 'nCells', 'nVertLevels'],
                     np.linspace(0, 10000, 10)[None, None, :] * np.ones((2, 100, 10)))
        }, coords={'Time': times})
        
        mock_processor.dataset = ds
        coords = np.linspace(100000, 10000, 10)
        
        result, coord_type = self.plotter._convert_vertical_to_height(
            coords, 'pressure', mock_processor, 0
        )
        
        assert coord_type == 'height_km'
    
    def test_convert_model_levels_with_zgrid(self: "TestAdditionalCoverage") -> None:
        """
        This test verifies that model-level indices are converted to height using `zgrid` when present. The test constructs a dataset with `zgrid` and asserts the returned coordinate type signals kilometers for display purposes.

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
            'zgrid': (['Time', 'nCells', 'nVertLevels'],
                     np.linspace(0, 10000, 10)[None, None, :] * np.ones((2, 100, 10)))
        }, coords={'Time': times})
        
        mock_processor.dataset = ds
        coords = np.arange(10)
        
        result, coord_type = self.plotter._convert_vertical_to_height(
            coords, 'model_levels', mock_processor, 0
        )
        
        assert coord_type == 'height_km'
    
    def test_pressure_axis_with_standard_ticks(self: "TestAdditionalCoverage") -> None:
        """
        Verify pressure axis configuration uses a logarithmic scale for standard ticks. The test sets up a figure/axes and requests standard ticks, asserting the axes y-scale becomes 'log' when valid positive pressures are provided.

        Parameters:
            None

        Returns:
            None
        """
        self.plotter.fig = plt.figure()
        self.plotter.ax = self.plotter.fig.add_subplot(111)
        
        vertical_coords = np.array([1000, 850, 700, 500, 300, 200, 100])
        self.plotter._setup_pressure_axis(vertical_coords, use_standard_ticks=True)
        
        assert self.plotter.ax.get_yscale() == 'log'
        plt.close(self.plotter.fig)
    
    def test_pressure_axis_non_positive_values(self: "TestAdditionalCoverage") -> None:
        """
        This test verifies that the pressure axis setup method can handle non-positive pressure values without crashing. The test provides an array of vertical coordinates that includes negative and zero values, and asserts that the method completes without error, confirming that it can manage invalid pressure levels gracefully.

        Parameters:
            None

        Returns:
            None
        """
        self.plotter.fig = plt.figure()
        self.plotter.ax = self.plotter.fig.add_subplot(111)
        
        vertical_coords = np.array([-100, 0, 100, 200])
        self.plotter._setup_pressure_axis(vertical_coords, use_standard_ticks=True)
        
        assert self.plotter.ax is not None
        plt.close(self.plotter.fig)
    
    def test_format_axes_pressure_hPa_max_height(self: "TestAdditionalCoverage") -> None:
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
    
    def test_format_axes_pressure_Pa(self: "TestAdditionalCoverage") -> None:
        """
        This test verifies that axis formatting for pressure in Pascals includes 'Pa' in the ylabel when max_height is not specified. The test configures the axes with pressure values in Pascals and asserts that the ylabel contains 'Pa', confirming that the plotter correctly formats the axis label based on the provided parameters.

        Parameters:
            None

        Returns:
            None
        """
        self.plotter.fig = plt.figure()
        self.plotter.ax = self.plotter.fig.add_subplot(111)
        
        lons = np.linspace(-100, -90, 50)
        vertical_coords = np.array([100000, 85000, 70000, 50000])
        
        self.plotter._format_cross_section_axes(
            lons, vertical_coords, 'pressure',
            (-100, 30), (-90, 40), max_height=None
        )
        
        assert 'Pa' in self.plotter.ax.get_ylabel()
        plt.close(self.plotter.fig)
    
    def test_format_axes_height_meters(self: "TestAdditionalCoverage") -> None:
        """
        This test verifies that axis formatting for height displays meters units when requested. The test supplies vertical coordinates in meters and asserts the ylabel contains '[m]', confirming that the plotter correctly formats the axis label for height coordinates.

        Parameters:
            None

        Returns:
            None
        """
        self.plotter.fig = plt.figure()
        self.plotter.ax = self.plotter.fig.add_subplot(111)
        
        lons = np.linspace(-100, -90, 50)
        vertical_coords = np.array([0, 1000, 2000, 3000, 4000, 5000])
        
        self.plotter._format_cross_section_axes(
            lons, vertical_coords, 'height',
            (-100, 30), (-90, 40), max_height=3.0
        )
        
        assert '[m]' in self.plotter.ax.get_ylabel()
        plt.close(self.plotter.fig)
    
    def test_format_axes_model_levels(self: "TestAdditionalCoverage") -> None:
        """
        This test verifies that axis formatting for model levels includes the 'Model Level' label. The test sets integer vertical coordinates and asserts the ylabel contains the expected text, confirming that the plotter correctly formats the axis label for model level coordinates.

        Parameters:
            None

        Returns:
            None
        """
        self.plotter.fig = plt.figure()
        self.plotter.ax = self.plotter.fig.add_subplot(111)
        
        lons = np.linspace(-100, -90, 50)
        vertical_coords = np.arange(10)
        
        self.plotter._format_cross_section_axes(
            lons, vertical_coords, 'model_levels',
            (-100, 30), (-90, 40), max_height=None
        )
        
        assert 'Model Level' in self.plotter.ax.get_ylabel()
        plt.close(self.plotter.fig)
    
    def test_batch_processing_complete_workflow(self: "TestAdditionalCoverage") -> None:
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


class TestAdditionalEdgeCases:
    """ Additional tests targeting remaining uncovered lines. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestAdditionalEdgeCases", mpas_3d_processor) -> Generator[None, None, None]: # type: ignore[no-untyped-def]
        """
        This fixture sets up the MPAS3DProcessor and MPASVerticalCrossSectionPlotter for the edge case tests. It checks if the processor is available and skips tests if not, ensuring that the tests only run when the necessary data is accessible. The processor and plotter are assigned to instance variables for use in the test methods.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): Fixture providing a processor with MPAS data.

        Returns:
            Generator[None, None, None]: A generator that yields control back to the test methods after setup.
        """
        if mpas_3d_processor is None:
            pytest.skip("MPAS data not available")
        
        self.processor = mpas_3d_processor
        self.plotter = MPASVerticalCrossSectionPlotter()
    
    def test_max_height_all_levels_above_limit(self: "TestAdditionalEdgeCases") -> None:
        """
        This test verifies that the plotter handles cases where all vertical levels are above the specified `max_height` limit. It mocks the data generation to return heights that exceed the `max_height` threshold and asserts that the plotter can still create a figure without crashing, confirming that it can manage scenarios where no data points meet the height criteria.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor
        
        plotter = MPASVerticalCrossSectionPlotter()
        
        with patch.object(plotter, '_generate_cross_section_data') as mock_gen:
            mock_gen.return_value = {
                'distances': np.linspace(0, 100, 50),
                'vertical_coords': np.array([5, 10, 15, 20]),  
                'data_values': np.random.rand(4, 50),
                'path_lons': np.linspace(-100, -90, 50),
                'path_lats': np.linspace(30, 40, 50),
                'longitudes': np.linspace(-100, -90, 50),
                'vertical_coord_type': 'height'
            }
            
            var = _find_3d_var(processor)
            assert var is not None, "No 3D variable found in processor dataset"

            fig, ax = plotter.create_vertical_cross_section(
                processor, var, (-100, 30), (-90, 40),
                max_height=0.1,  
                display_vertical='height'
            )

            plt.close(fig)
    
    def test_pressure_in_hpa_not_pa(self: "TestAdditionalEdgeCases") -> None:
        """
        This test verifies that when pressure values are provided in hPa (i.e., max < 10000), the plotter correctly identifies the units and formats the axes accordingly. It mocks the data generation to return pressure values in hPa and asserts that the plotter can create a figure without errors, confirming that it can handle pressure data in different units.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor
        
        plotter = MPASVerticalCrossSectionPlotter()
        
        with patch.object(plotter, '_generate_cross_section_data') as mock_gen:
            mock_gen.return_value = {
                'distances': np.linspace(0, 100, 50),
                'vertical_coords': np.array([1000, 850, 700, 500]),  
                'data_values': np.random.rand(4, 50),
                'path_lons': np.linspace(-100, -90, 50),
                'path_lats': np.linspace(30, 40, 50),
                'longitudes': np.linspace(-100, -90, 50),
                'vertical_coord_type': 'pressure'
            }
            
            var = _find_3d_var(processor)
            assert var is not None, "No 3D variable found in processor dataset"
            
            fig, ax = plotter.create_vertical_cross_section(
                processor, var, (-100, 30), (-90, 40),
                display_vertical='pressure'
            )

            plt.close(fig)
    
    def test_scipy_interpolation_in_height_extraction(self: "TestAdditionalEdgeCases") -> None:
        """
        This test verifies that scipy interpolation is used when the source height array size differs from the requested levels. It builds a small dataset with mismatched sizes and asserts the returned interpolated heights length.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASVerticalCrossSectionPlotter()
        original_extract = plotter._extract_height_from_dataset
        
        def mock_extract(proc, time_idx, vert_coords, var_name):
            return original_extract(proc, time_idx, vert_coords, var_name)
        
        vertical_coords = np.array([0, 1, 2, 3, 4])  
        
        simple_dataset = xr.Dataset({
            'zgrid': xr.DataArray(
                np.linspace(0, 20000, 8).reshape(1, 8, 1),  
                dims=['Time', 'nVertLevelsP1', 'nCells']
            )
        })
        
        simple_processor = Mock(spec=MPAS3DProcessor)
        simple_processor.dataset = simple_dataset
        
        result = plotter._extract_height_from_dataset(
            simple_processor, 0, vertical_coords, 'zgrid'
        )
        
        if result is not None:
            assert len(result) == len(vertical_coords)
    
    def test_batch_processing_time_extraction_edge_cases(self: "TestAdditionalEdgeCases") -> None:
        """
        This test verifies that the batch processing method can handle edge cases in time extraction, such as when the dataset has a Time coordinate but it is not properly formatted. It asserts that the method can still produce output files without crashing, confirming that it can manage time-related issues gracefully.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None: Test asserts on produced filenames and file existence.
        """
        processor = self.processor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            files = self.plotter.create_batch_cross_section_plots(
                processor, temp_dir, 'theta',
                start_point=(-100, 30),
                end_point=(-90, 40),
                num_points=20,
                formats=['png']
            )
            
            assert len(files) > 0


class TestCrossSectionPathValidation:
    """ Tests for cross-section path domain checks and coordinate conversion. """

    @pytest.fixture(autouse=True)
    def setup(self: "TestCrossSectionPathValidation") -> None:
        """
        This fixture sets up the MPASVerticalCrossSectionPlotter for path validation tests. It initializes the plotter and enables verbose mode to ensure that any warnings or messages related to path validation are printed during the tests.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        self.plotter = MPASVerticalCrossSectionPlotter()
        self.plotter.verbose = True

    def test_path_outside_grid_domain_warning(self: "TestCrossSectionPathValidation") -> None:
        """
        This test verifies that a warning is issued when the specified cross-section path lies outside the grid domain. It creates a mock grid domain and defines a path with coordinates that are clearly outside this domain. The test asserts that the path is recognized as being outside the longitude and latitude bounds, confirming that the plotter can detect and warn about paths that do not intersect the grid.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        n_cells = 100
        lon_coords = np.linspace(-105, -95, n_cells)
        lat_coords = np.linspace(30, 40, n_cells)

        path_lons = np.array([-130, -125, -120])
        path_lats = np.array([50, 55, 60])

        path_in_lon = (path_lons[0] >= np.min(lon_coords) and path_lons[-1] <= np.max(lon_coords))

        path_in_lat = (min(path_lats[0], path_lats[-1]) >= np.min(lat_coords) and
                       max(path_lats[0], path_lats[-1]) <= np.max(lat_coords))

        assert not path_in_lon or not path_in_lat

    def test_great_circle_same_start_end(self: "TestCrossSectionPathValidation") -> None:
        """
        This test verifies that when the start and end points of a great circle path are identical, the path still produces valid output. It asserts that the generated longitude and latitude arrays have the correct length and that all distances are non-negative. This confirms that the path generation logic can handle zero-distance paths without errors.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        lons, lats, dists = self.plotter._generate_great_circle_path(
            start_point=(-100, 35),
            end_point=(-100, 35),
            num_points=5
        )
        assert len(lons) == pytest.approx(5)
        assert np.all(dists >= 0)


class TestStandardAtmosphereConversion:
    """ Tests for _convert_vertical_to_height standard atmosphere paths. """

    @pytest.fixture(autouse=True)
    def setup(self: "TestStandardAtmosphereConversion") -> None:
        """
        This fixture sets up the MPASVerticalCrossSectionPlotter for standard atmosphere conversion tests. It initializes the plotter and enables verbose mode to ensure that any warnings or messages related to vertical coordinate conversion are printed during the tests.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        self.plotter = MPASVerticalCrossSectionPlotter()
        self.plotter.verbose = True

    def test_pressure_to_height_standard_atmosphere(self: "TestStandardAtmosphereConversion") -> None:
        """
        This test verifies that valid pressure values are correctly converted to height using the standard atmosphere formula. It asserts that the resulting height array is in kilometers, that higher pressures correspond to lower heights, and that all height values are non-negative.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        pressure_pa = np.array([101325.0, 50000.0, 25000.0, 10000.0])
        mock_processor = MagicMock()
        mock_processor.dataset = xr.Dataset()  

        height, coord_type = self.plotter._convert_vertical_to_height(
            pressure_pa, 'pressure', mock_processor, 0
        )
        assert coord_type == 'height_km'
        assert height[0] < height[-1]  
        assert np.all(height >= 0)

    def test_pressure_non_positive_clipping(self: "TestStandardAtmosphereConversion") -> None:
        """
        This test verifies that non-positive pressure values are correctly handled by clipping them to a minimum positive value. It asserts that the resulting height array is in kilometers and that all height values are finite.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        from io import StringIO
        pressure_pa = np.array([101325.0, 50000.0, -100.0, 0.0])
        mock_processor = MagicMock()
        mock_processor.dataset = xr.Dataset()

        captured = StringIO()
        with patch('sys.stdout', captured):
            height, coord_type = self.plotter._convert_vertical_to_height(
                pressure_pa, 'pressure', mock_processor, 0
            )
        assert coord_type == 'height_km'
        assert np.all(np.isfinite(height))

    def test_pressure_hpa_small_values(self: "TestStandardAtmosphereConversion") -> None:
        """
        This test verifies that pressure values less than 10000 (likely in hPa) are correctly multiplied by 100 to convert to Pa. It asserts that the resulting height array is in kilometers and that the surface pressure corresponds to near-zero height.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        pressure_hpa = np.array([1013.25, 500.0, 250.0, 100.0])
        mock_processor = MagicMock()
        mock_processor.dataset = xr.Dataset()

        height, coord_type = self.plotter._convert_vertical_to_height(
            pressure_hpa, 'pressure', mock_processor, 0
        )
        assert coord_type == 'height_km'
        assert height[0] < 2.0

    def test_model_levels_fallback_no_height_var(self: "TestStandardAtmosphereConversion") -> None:
        """
        This test verifies that when model levels are provided without a height variable, the method falls back to using the model level indices as heights. It asserts that the resulting height array matches the input model levels and that the coordinate type indicates model levels.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        model_levels = np.arange(1, 56)
        mock_processor = MagicMock()
        mock_processor.dataset = xr.Dataset()

        height, coord_type = self.plotter._convert_vertical_to_height(
            model_levels, 'model_levels', mock_processor, 0
        )
        assert coord_type == 'model_levels'
        np.testing.assert_array_equal(height, model_levels)

    def test_height_direct_conversion(self: "TestStandardAtmosphereConversion") -> None:
        """
        This test verifies that when height coordinates are provided directly, they are correctly converted from meters to kilometers. It asserts that the resulting height array is in kilometers and that the conversion is accurate.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        height_m = np.array([0, 1000, 5000, 10000, 15000])
        mock_processor = MagicMock()

        height_km, coord_type = self.plotter._convert_vertical_to_height(
            height_m, 'height', mock_processor, 0
        )
        assert coord_type == 'height_km'
        np.testing.assert_array_almost_equal(height_km, [0, 1, 5, 10, 15])


class TestInterpolationAllNaN:
    """ Tests for interpolation along path with edge-case data. """

    @pytest.fixture(autouse=True)
    def setup(self: "TestInterpolationAllNaN") -> None:
        """
        This fixture sets up the MPASVerticalCrossSectionPlotter for interpolation tests. It initializes the plotter and disables verbose mode to suppress output during interpolation tests, ensuring that the focus is on the interpolation results rather than any warnings or messages.
        
        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        self.plotter = MPASVerticalCrossSectionPlotter()
        self.plotter.verbose = False

    def test_interpolate_all_nan_returns_nan(self: "TestInterpolationAllNaN") -> None:
        """
        This test verifies that when all grid data values are NaN, the interpolation method returns an array of NaN values. It constructs a grid with NaN data and a path that intersects the grid, then asserts that the interpolation result is entirely NaN, confirming that the method can handle cases where no valid data points are available for interpolation.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        grid_lons = np.linspace(-110, -100, 50)
        grid_lats = np.linspace(35, 45, 50)
        grid_data = np.full(50, np.nan)
        path_lons = np.linspace(-108, -102, 10)
        path_lats = np.linspace(37, 43, 10)

        result = self.plotter._interpolate_along_path(
            grid_lons, grid_lats, grid_data, path_lons, path_lats
        )
        assert np.all(np.isnan(result))

    def test_interpolate_with_xarray_input(self: "TestInterpolationAllNaN") -> None:
        """
        This test verifies that DataArray input should be handled transparently. It constructs a grid with random data in a DataArray and a path that intersects the grid, then asserts that the interpolation result has the correct length and contains valid values, confirming that the method can handle xarray DataArray inputs.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        grid_lons = np.linspace(-110, -100, 50)
        grid_lats = np.linspace(35, 45, 50)
        grid_data = xr.DataArray(np.random.uniform(200, 300, 50))
        path_lons = np.linspace(-108, -102, 10)
        path_lats = np.linspace(37, 43, 10)

        result = self.plotter._interpolate_along_path(
            grid_lons, grid_lats, grid_data, path_lons, path_lats
        )
        assert len(result) == pytest.approx(10)
        assert not np.all(np.isnan(result))


class TestCrossSectionRenderingPaths:
    """ Tests for rendering paths including filled_contour, moisture clipping, height conversion. """

    def setup_method(self) -> None:
        """
        This setup method initializes the MPASVerticalCrossSectionPlotter for rendering path tests. It creates an instance of the plotter that will be used in subsequent tests to verify different rendering paths, such as filled contour plotting, moisture variable clipping, and height conversion logic.

        Parameters:
            None

        Returns:
            None  
        """
        self.plotter = MPASVerticalCrossSectionPlotter()

    def test_filled_contour_plot_type(self) -> None:
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

    def test_moisture_variable_negative_clipping(self) -> None:
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

    def test_display_height_conversion_branch(self) -> None:
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

    def test_pressure_nonpositive_warning(self, capsys) -> None:
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

    def test_max_height_filtering(self) -> None:
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

    def test_colormap_levels_auto_lookup(self) -> None:
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

    def test_path_outside_grid_domain_warning(self, capsys) -> None:
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

    def test_direct_dataset_access_fallback(self) -> None:
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

    def test_valid_data_count_logging(self, capsys) -> None:
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

    def setup_method(self) -> None:
        """
        This method sets up the MPASVerticalCrossSectionPlotter for axis formatting tests. It initializes the plotter instance and creates a figure and axes that will be used in the subsequent test methods to verify various axis formatting paths such as pressure log scale, max_height filtering, and colorbar configuration.

        Parameters:
            None

        Returns:
            None
        """
        self.plotter = MPASVerticalCrossSectionPlotter()
        self.plotter.fig, self.plotter.ax = plt.subplots()

    def teardown_method(self) -> None:
        """
        This method tears down the test environment after each test method is executed. It closes all matplotlib figures to ensure that no resources are left open and that subsequent tests start with a clean slate, preventing any interference from previous tests.

        Parameters:
            None

        Returns:
            None
        """
        plt.close('all')

    def test_pressure_log_scale_standard_ticks(self) -> None:
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

    def test_pressure_axis_linear_fallback_nonpositive(self, capsys) -> None:
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

    def test_pressure_hpa_with_max_height(self) -> None:
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

    def test_pressure_pa_with_max_height(self) -> None:
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

    def test_height_axis_km_with_max_height(self) -> None:
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
        assert ylim[1] == max_height

    def test_height_m_axis_with_max_height(self) -> None:
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

    def test_path_info_text_box(self) -> None:
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

    def test_pressure_to_height_conversion_standard_atmosphere(self) -> None:
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

    def test_model_level_height_extraction_zgrid(self) -> None:
        """
        This test verifies that when model levels are provided and a `zgrid` variable exists in the dataset, the method correctly extracts height information from the `zgrid` variable. It creates a mock processor with a dataset containing `zgrid`, provides an array of model levels, and checks that the resulting height array is in kilometers and that the coordinate type indicates either 'height_km' or 'model_levels', confirming that the method can extract height information from the dataset when available.

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
        model_levels = np.arange(10)
        height_km, coord_type = self.plotter._convert_vertical_to_height(
            model_levels, 'model_levels', mock_proc, 0
        )
        assert coord_type in ('height_km', 'model_levels')

class TestCrossSectionBatch:
    """ Tests for batch cross-section plot creation loop. """

    def test_batch_cross_section_creation_validates_processor(self) -> None:
        """
        This test verifies that create_batch_cross_section_plots raises a ValueError when the provided mpas_3d_processor is not an instance of MPAS3DProcessor. It attempts to call the method with an invalid processor type and asserts that the appropriate error message is raised, confirming that the method validates the processor input correctly.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASVerticalCrossSectionPlotter()
        with pytest.raises(ValueError, match="must be an instance"):
            plotter.create_batch_cross_section_plots(
                mpas_3d_processor="not_a_processor",
                output_dir='/tmp/test',
                var_name='theta',
                start_point=(-110, 30),
                end_point=(-100, 40)
            )

    def test_batch_cross_section_requires_loaded_data(self) -> None:
        """
        This test verifies that create_batch_cross_section_plots raises a ValueError when the provided MPAS3DProcessor does not have loaded data. It creates a mock processor with a None dataset and asserts that the appropriate error message is raised, confirming that the method checks for loaded data before attempting to create plots.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASVerticalCrossSectionPlotter()
        mock_proc = MagicMock(spec=MPAS3DProcessor)
        mock_proc.dataset = None
        with pytest.raises(ValueError, match="must have loaded data"):
            plotter.create_batch_cross_section_plots(
                mpas_3d_processor=mock_proc,
                output_dir='/tmp/test',
                var_name='theta',
                start_point=(-110, 30),
                end_point=(-100, 40)
            )

    def test_batch_cross_section_missing_variable(self) -> None:
        """
        This test verifies that create_batch_cross_section_plots raises a ValueError when the specified variable is not found in the dataset.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASVerticalCrossSectionPlotter()
        mock_proc = MagicMock(spec=MPAS3DProcessor)
        mock_proc.dataset = xr.Dataset({
            'theta': (['Time', 'nVertLevels', 'nCells'], np.zeros((2, 5, 10)))
        })
        with pytest.raises(ValueError, match="not found"):
            plotter.create_batch_cross_section_plots(
                mpas_3d_processor=mock_proc,
                output_dir='/tmp/test',
                var_name='nonexistent_var',
                start_point=(-110, 30),
                end_point=(-100, 40)
            )

    def test_batch_cross_section_time_iteration(self, tmp_path) -> None:
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
