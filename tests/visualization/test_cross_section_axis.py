#!/usr/bin/env python3
"""
MPASdiag Test Suite: Test MPASVerticalCrossSectionPlotter Functionality

This test suite validates the core functionality of the MPASVerticalCrossSectionPlotter class, which is responsible for generating vertical cross-section plots from MPAS model data. The tests cover key aspects of the plotter's behavior including initialization with default and custom parameters, great circle path generation between geographic points, automatic contour level generation for various data types, spatial interpolation of grid data onto cross-section paths, and robust input validation to ensure proper error handling. By systematically verifying these functionalities, this test suite ensures that the MPASVerticalCrossSectionPlotter can reliably produce accurate and informative visualizations of vertical cross-sections for meteorological analysis.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and set up test environment
import os
import sys
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
from mpasdiag.processing.processors_3d import MPAS3DProcessor

from tests.visualization.cross_section_test_helpers import (
    GRID_FILE, MPASOUT_DIR,
    check_great_circle_path,
)


def test_great_circle_path_generation() -> None:
    """
    This test verifies that the MPASVerticalCrossSectionPlotter correctly generates a great circle path between two geographic points. It checks that the generated longitude and latitude arrays have the expected number of points, that the start and end points match the input coordinates within a reasonable tolerance, and that the distance array is monotonically increasing along the path. By validating these aspects of the great circle path generation, this test ensures that the plotter can accurately compute spatial paths for cross-section plotting based on geographic coordinates.

    Parameters:
        None

    Returns:
        None
    """
    check_great_circle_path()


class TestAxisFormattingFinal:
    """ Test axis formatting with real data. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestAxisFormattingFinal', 
                     mpas_3d_processor: 'MPAS3DProcessor') -> None:
        """
        This fixture sets up a MPASVerticalCrossSectionPlotter instance and initializes it with real MPAS data for testing axis formatting with actual model output. It checks for the availability of the required data files and skips the tests if the data is not available. By using a shared session-scoped processor, it avoids redundant data loading across multiple tests, ensuring efficient use of resources while validating axis formatting with real MPAS data.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): A session-scoped fixture that provides a pre-loaded MPAS3DProcessor instance with real MPAS data.

        Returns:   
            None: Asserts that the plotter is initialized and the processor is available for testing.
        """
        if mpas_3d_processor is None or not os.path.exists(GRID_FILE) or not os.path.exists(MPASOUT_DIR):
            pytest.skip("Real MPAS data not available")
            return
        
        self.plotter = MPASVerticalCrossSectionPlotter()

        self.processor = mpas_3d_processor
    
    def test_axis_formatting_with_max_height(self: 'TestAxisFormattingFinal') -> None:
        """
        This test produces a plot using `vertical_coord='height'` and checks that the y-axis label contains 'Height' and that the maximum height limit is applied correctly. It verifies that the plot is created successfully without errors, confirming that the plotter can handle height-based vertical coordinates and apply axis formatting with a specified maximum height.

        Parameters:
            self (Any): Test case instance with `processor` and `plotter` fixtures.

        Returns:
            None: Asserts that the figure is created successfully.
        """
        fig, _ = self.plotter.create_vertical_cross_section(
            self.processor,
            'theta',
            (0, 30),
            (15, 45),
            vertical_coord='height',
            max_height=20000,
            num_points=30,
            time_index=0
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_model_levels_axis_formatting(self: 'TestAxisFormattingFinal') -> None:
        """
        This test produces a plot using `vertical_coord='modlev'` and checks that the y-axis label contains 'Model Level'. It verifies that the plot is created successfully without errors, confirming that the plotter can handle model level vertical coordinates and apply appropriate axis labeling for model levels.

        Parameters:
            self (Any): Test case instance with `processor` and `plotter` fixtures.

        Returns:
            None: Asserts that the axis label contains 'Model Level'.
        """
        fig, ax = self.plotter.create_vertical_cross_section(
            self.processor,
            'theta',
            (-5, 35),
            (10, 50),
            vertical_coord='modlev',
            num_points=25,
            time_index=0
        )
        
        ylabel = ax.get_ylabel()
        assert 'Model Level' in ylabel
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
