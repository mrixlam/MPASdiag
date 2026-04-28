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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.visualization.cross_section_test_helpers import (
    GRID_FILE, MPASOUT_DIR,
    check_great_circle_path,
)


def test_great_circle_path_generation() -> None:
    """
    This test validates the functionality of the great circle path generation method in the MPASVerticalCrossSectionPlotter. It checks that the method correctly computes the great circle path between two geographic endpoints, ensuring that the generated path follows the shortest route on the Earth's surface. The test verifies that the computed path points are accurate and consistent with expected values for given input coordinates, confirming that the plotter can generate correct cross-section paths for visualization. 

    Parameters:
        None

    Returns:
        None
    """
    check_great_circle_path()


class TestInterpolationWithRealData:
    """ Tests for interpolation with real data. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestInterpolationWithRealData', mpas_3d_processor) -> None:
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
    
    def test_interpolate_with_sparse_points(self: 'TestInterpolationWithRealData') -> None:
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
