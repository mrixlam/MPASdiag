#!/usr/bin/env python3
"""
MPASdiag Test Suite: Test MPASVerticalCrossSectionPlotter Functionality

This test suite validates the core functionality of the MPASVerticalCrossSectionPlotter class, which is responsible for generating vertical cross-section visualizations from MPAS model output. The tests cover key aspects including plotter initialization with default and custom parameters, great circle path generation between geographic endpoints, automatic contour level generation for various data types, spatial interpolation of irregular grid data onto cross-section paths, and robust input validation for processor objects. By systematically verifying these components, the test suite ensures that the plotter can be reliably used for creating accurate and informative vertical cross-section visualizations in meteorological analyses.

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
import matplotlib
import numpy as np
matplotlib.use('Agg')
from typing import cast, Any
import matplotlib.pyplot as plt

from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
GRID_FILE = os.path.join(TEST_DATA_DIR, 'grids', 'x1.10242.static.nc')
MPASOUT_DIR = os.path.join(TEST_DATA_DIR, 'u240k', 'mpasout')


def test_vertical_cross_section_plotter_initialization() -> None:
    """
    This test validates the initialization of the MPASVerticalCrossSectionPlotter class, ensuring that default parameters are set correctly and that custom parameters can be applied. The test checks that the default figure size is (10, 12) inches and the default DPI is 100. It also verifies that the figure and axes attributes are initialized to None. When custom parameters are provided, the test confirms that they are correctly assigned to the plotter instance.

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
    This test verifies the correctness of the great circle path generation method in the MPASVerticalCrossSectionPlotter class. It checks that the generated longitude and latitude arrays have the correct length corresponding to the specified number of points along the path. The test also confirms that the starting and ending points of the generated path closely match the input coordinates within a reasonable tolerance. Additionally, it validates that the distance array is monotonically increasing and that the first distance value is zero while the last distance value is greater than zero, indicating a valid path length.

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
    This test validates the default contour level generation method in the MPASVerticalCrossSectionPlotter class for different types of data. It checks that the generated levels are appropriate for typical meteorological variables such as potential temperature (theta) and wind components (uwind). The test confirms that the levels cover the range of the input data and that they are not empty. It also verifies that the method can handle constant data and NaN values without errors, ensuring that it returns a reasonable set of levels in these edge cases.

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
    This test verifies the interpolation of grid data along a specified path defined by longitude and latitude coordinates. It checks that the interpolated values are returned for each point along the path and that they are not all NaN, indicating that the interpolation is functioning correctly. The test uses a simple synthetic dataset to validate the interpolation logic, ensuring that the method can handle typical scenarios encountered in cross-section plotting where data values need to be estimated at specific locations along the path. This validation confirms that the plotter can accurately interpolate data for visualization purposes, enhancing the quality and reliability of the generated cross-section plots.

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
    This test validates the input handling of the create_vertical_cross_section method in the MPASVerticalCrossSectionPlotter class. It checks that the method raises a ValueError when an invalid MPAS3DProcessor object is passed as an argument. The test ensures that the error message contains relevant information about the expected processor type, confirming that the input validation logic is correctly implemented to prevent misuse of the plotting function with incompatible data processors. This validation is crucial for maintaining the robustness and reliability of the plotter when integrated into larger workflows where user input may vary.

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


class TestInterpolationEdgeCases:
    """ Tests for data interpolation edge cases. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestInterpolationEdgeCases") -> None:
        """
        This setup method initializes the MPASVerticalCrossSectionPlotter instance for use in interpolation edge case tests. It ensures that a fresh plotter object is available for each test method, allowing for consistent testing of interpolation behavior under various edge case scenarios such as empty data, constant values, and NaN-filled arrays. By using an autouse fixture, this setup is automatically applied to all test methods within the TestInterpolationEdgeCases class without requiring explicit calls, streamlining the test code and ensuring that the necessary plotter instance is always ready for interpolation tests.

        Parameters:
            self (TestInterpolationEdgeCases): The test case instance for which the plotter is being initialized.

        Returns:
            None
        """
        self.plotter = MPASVerticalCrossSectionPlotter()
    
    def test_empty_valid_data_returns_nan(self: "TestInterpolationEdgeCases") -> None:
        """
        This test verifies that the interpolation method returns NaN values when provided with empty input data. It checks that when the grid data is empty, the interpolation along the specified path results in an array of NaN values, indicating that no valid interpolation can be performed. This test ensures that the interpolation method correctly handles cases where there is no data to interpolate, preventing potential errors or misleading results in the generated cross-section plots.

        Parameters:
            None

        Returns:
            None
        """
        grid_lons = np.array([1.0, 2.0, 3.0])
        grid_lats = np.array([1.0, 2.0, 3.0])
        grid_data = np.array([np.nan, np.nan, np.nan])
        path_lons = np.array([1.5, 2.5])
        path_lats = np.array([1.5, 2.5])
        
        result = self.plotter._interpolate_along_path(
            grid_lons, grid_lats, grid_data, path_lons, path_lats
        )
        
        assert np.all(np.isnan(result))
        assert len(result) == len(path_lons)


class TestInterpolationWithRealData:
    """ Tests for interpolation with real data. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestInterpolationWithRealData", mpas_3d_processor) -> None:
        """
        This setup method initializes the MPASVerticalCrossSectionPlotter instance and assigns the shared session-scoped MPAS3DProcessor instance for use in tests that involve interpolation with real MPAS data. It checks for the availability of the necessary data files and skips the tests if the data is not available, ensuring that the tests are only run when valid MPAS data can be accessed. By using an autouse fixture, this setup is automatically applied to all test methods within the TestInterpolationWithRealData class, providing a consistent testing environment for interpolation tests that rely on real data.

        Parameters:
            self (TestInterpolationWithRealData): The test case instance for which the plotter and processor are being initialized.
            mpas_3d_processor (Any): The shared session-scoped MPAS3DProcessor instance.

        Returns:
            None
        """
        if mpas_3d_processor is None or not os.path.exists(GRID_FILE) or not os.path.exists(MPASOUT_DIR):
            pytest.skip("Real MPAS data not available")
        
        self.plotter = MPASVerticalCrossSectionPlotter()
        self.processor = mpas_3d_processor
    
    def test_interpolate_with_sparse_points(self: "TestInterpolationWithRealData") -> None:
        """
        This test verifies the interpolation of real MPAS data along a specified path with a very limited number of points. It checks that the interpolation method can still produce valid results when only a few points are available along the path, ensuring that the plotter can handle scenarios where data may be sparse or where users may choose to generate cross-sections with fewer points for performance reasons. The test confirms that the resulting plot is created successfully without errors, validating the robustness of the interpolation method under these conditions.

        Parameters:
            self (Any): Test case instance with `processor` and `plotter` fixtures.

        Returns:
            None: Asserts that the plot is created successfully.
        """
        fig, ax = self.plotter.create_vertical_cross_section(
            self.processor,
            'theta',
            (-10, 30),
            (20, 50),
            vertical_coord='pressure',
            num_points=5,  
            time_index=0
        )
        
        assert fig is not None
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
