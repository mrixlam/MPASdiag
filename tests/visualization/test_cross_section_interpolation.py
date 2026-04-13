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
import pytest
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.visualization.cross_section_test_helpers import (
    GRID_FILE, MPASOUT_DIR,
    check_plotter_initialization, check_great_circle_path,
    check_default_levels, check_interpolation_along_path,
    check_input_validation,
)


def test_vertical_cross_section_plotter_initialization() -> None:
    """
    This test verifies that the MPASVerticalCrossSectionPlotter can be initialized with default parameters and that its attributes are set correctly. It checks that the plotter instance is created successfully and that the default values for attributes such as the colormap, contour levels, and interpolation method are assigned as expected. This test ensures that the plotter is ready for use in generating vertical cross-section visualizations without requiring additional configuration.

    Parameters:
        None

    Returns:
        None
    """
    check_plotter_initialization()


def test_great_circle_path_generation() -> None:
    """
    This test validates the functionality of the great circle path generation method in the MPASVerticalCrossSectionPlotter. It checks that the method correctly computes the great circle path between two geographic endpoints, ensuring that the generated path follows the shortest route on the Earth's surface. The test verifies that the computed path points are accurate and consistent with expected values for given input coordinates, confirming that the plotter can generate correct cross-section paths for visualization. 

    Parameters:
        None

    Returns:
        None
    """
    check_great_circle_path()


def test_default_levels_generation() -> None:
    """
    This test verifies that the MPASVerticalCrossSectionPlotter can generate appropriate default contour levels for different types of data, such as temperature, pressure, and wind speed. It checks that the generated levels are suitable for visualizing the data effectively, ensuring that they provide sufficient resolution and coverage for typical ranges of values encountered in meteorological analyses. This test confirms that the plotter can automatically determine reasonable contour levels when explicit levels are not provided by the user. 

    Parameters:
        None

    Returns:
        None
    """
    check_default_levels()


def test_interpolation_along_path() -> None:
    """
    This test validates the interpolation functionality of the MPASVerticalCrossSectionPlotter when interpolating data along a specified path. It checks that the interpolation method correctly handles irregular grid data and produces accurate interpolated values at the points along the cross-section path. The test verifies that the interpolation results are consistent with expected values based on known input data, ensuring that the plotter can effectively interpolate data for generating accurate vertical cross-section visualizations. 

    Parameters:
        None

    Returns:
        None
    """
    check_interpolation_along_path()


def test_input_validation() -> None:
    """
    This test ensures that the MPASVerticalCrossSectionPlotter's input validation correctly identifies and handles invalid processor objects. It checks that when an invalid processor (e.g., one that does not conform to the expected interface or lacks necessary attributes) is passed to the plotter, the appropriate exceptions are raised with informative error messages. This test confirms that the plotter can robustly handle incorrect inputs, preventing potential issues during plot generation and guiding users towards providing valid processor objects for successful visualization. 

    Parameters:
        None

    Returns:
        None
    """
    check_input_validation()


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
            return
        
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
