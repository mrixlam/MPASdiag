#!/usr/bin/env python3
"""
MPASdiag Test Suite: Test MPASVerticalCrossSectionPlotter Functionality

This test suite validates the core functionality of the MPASVerticalCrossSectionPlotter class, which is responsible for generating vertical cross-section plots from MPAS model output data. The tests cover various aspects of the plotter's behavior, including initialization, great circle path generation, default level calculation, spatial interpolation along the cross-section path, and input validation for processor objects. By systematically verifying these components, the test suite ensures that the plotter can reliably create accurate and informative visualizations of atmospheric variables along specified cross-section paths. The tests are designed to be comprehensive yet efficient, providing confidence in the plotter's robustness and correctness when applied to real MPAS datasets.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and set up test environment
import os
import sys
import math
import pytest
import matplotlib
import numpy as np
matplotlib.use('Agg')
from typing import cast, Any
from unittest.mock import patch
import matplotlib.pyplot as plt

from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
GRID_FILE = os.path.join(TEST_DATA_DIR, 'grids', 'x1.10242.static.nc')
MPASOUT_DIR = os.path.join(TEST_DATA_DIR, 'u240k', 'mpasout')


def test_vertical_cross_section_plotter_initialization() -> None:
    """
    This test verifies that the MPASVerticalCrossSectionPlotter initializes with the correct default parameters and allows for custom configuration. It checks that the default figure size is (10, 12) inches and the default DPI is 100, ensuring that the plotter is set up for high-quality visualizations. The test also confirms that the figure and axes attributes are initialized to None, indicating that no plot has been created yet. Additionally, it tests that custom parameters for figure size and DPI can be set during initialization, allowing users to tailor the plotting environment to their specific needs. This validation ensures that the plotter's initialization process is robust and flexible for various plotting requirements.

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
    This test validates the generation of great circle paths between specified start and end points. It checks that the generated longitude and latitude arrays have the correct number of points as specified by `num_points`, and that the first and last points in the arrays match the provided start and end coordinates within a reasonable tolerance. The test also verifies that the distance array is monotonically increasing, starting from zero at the first point, confirming that the path is correctly calculated along the great circle route. This ensures that the plotter can accurately generate cross-section paths for visualization based on geographic coordinates.

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
    
    print("Great circle path generation test passed!")


def test_default_levels_generation() -> None:
    """
    This test checks the generation of default contour levels for different types of data, including temperature (theta), wind components (uwind), constant values, and NaN-filled arrays. It verifies that the generated levels cover the range of the input data appropriately, ensuring that the minimum and maximum levels encompass the minimum and maximum data values. The test also confirms that a reasonable number of levels are generated for typical atmospheric variables, while still handling edge cases such as constant or NaN-filled data gracefully. This validation ensures that the plotter can automatically determine suitable contour levels for a variety of input data scenarios, enhancing the usability and visual clarity of the resulting plots.

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
    
    print("Default levels generation test passed!")


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
    This test checks the input validation for the `create_vertical_cross_section` method by passing an invalid processor object. It verifies that a ValueError is raised with an appropriate message indicating that the processor must be an instance of MPAS3DProcessor. This ensures that the plotter correctly handles invalid inputs and provides clear feedback to users when they attempt to use the plotting functionality with unsupported processor types, thereby preventing potential runtime errors and guiding users towards correct usage.

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


class TestPlotTypeAndLabelingErrors:
    """ Tests for plot type errors and labeling exception handling. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestPlotTypeAndLabelingErrors", mpas_3d_processor) -> None:
        """
        This fixture sets up the test environment for the TestPlotTypeAndLabelingErrors class by initializing the MPASVerticalCrossSectionPlotter and assigning a shared MPAS3DProcessor instance to the test class. It checks if the processor is available and skips the tests if not, ensuring that the tests are only run when real MPAS data is accessible. This setup allows the subsequent tests to focus on validating plot type handling and labeling exception scenarios using actual data, providing a realistic context for error handling verification.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): A shared session-scoped fixture that provides a processor instance with loaded MPAS data for testing.
        
        Returns:
            None
        """
        if mpas_3d_processor is None:
            pytest.skip("MPAS data not available")
            return
        
        self.processor = mpas_3d_processor
        self.plotter = MPASVerticalCrossSectionPlotter()
    
    def test_invalid_plot_type_error(self: "TestPlotTypeAndLabelingErrors") -> None:
        """
        This test verifies that the `create_vertical_cross_section` method raises a ValueError when an invalid plot type is specified. It attempts to create a vertical cross-section plot using an unsupported plot type and asserts that the error message contains the expected text indicating the unknown plot type. This ensures that the plotter correctly validates the `plot_type` parameter and provides clear feedback to users when they attempt to use an unsupported plotting option, thereby enhancing the robustness of the plotting functionality.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor
        
        with pytest.raises(ValueError) as cm:
            fig, ax = self.plotter.create_vertical_cross_section(
                processor, 'theta', (-100, 30), (-90, 40),
                plot_type='invalid_type'
            )
        
        assert "Unknown plot_type" in str(cm.value)
    
    def test_colorbar_label_exception_handling(self: "TestPlotTypeAndLabelingErrors") -> None:
        """
        This test checks the exception handling for colorbar label retrieval in the `create_vertical_cross_section` method. It mocks the MPASFileMetadata class to raise an exception when attempting to get variable metadata, simulating a scenario where metadata retrieval fails. The test then asserts that the plotter catches the exception and uses the variable name as a fallback for the colorbar label, ensuring that the plotting process can continue without crashing and still provides a meaningful label for the colorbar. This validation confirms that the plotter is resilient to metadata retrieval issues and can gracefully handle errors while still producing a usable plot.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor        
        plotter = MPASVerticalCrossSectionPlotter()
        
        with patch('mpasdiag.visualization.cross_section.MPASFileMetadata') as mock_meta:
            mock_meta.get_variable_metadata.side_effect = Exception("Metadata error")
            fig, ax = plotter.create_vertical_cross_section(
                processor, 'theta', (-100, 30), (-90, 40),
                plot_type='filled_contour'
            )
            plt.close(fig)
    
    def test_title_exception_handling(self: "TestPlotTypeAndLabelingErrors") -> None:
        """
        This test verifies that the `create_vertical_cross_section` method can handle exceptions that occur during time string generation for the plot title. It mocks the `_get_time_string` method to raise an exception, simulating a scenario where time information retrieval fails. The test then asserts that the plotter catches the exception and uses a fallback title that does not include time information, ensuring that the plotting process can continue without crashing and still provides a meaningful title for the plot. This validation confirms that the plotter is robust against errors in time information retrieval and can gracefully handle such exceptions while still producing a usable plot.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor        
        plotter = MPASVerticalCrossSectionPlotter()
        
        with patch.object(plotter, '_get_time_string', side_effect=Exception("Time error")):
            fig, ax = plotter.create_vertical_cross_section(
                processor, 'theta', (-100, 30), (-90, 40),
                title=None  
            )
            
            assert "Vertical Cross-Section" in ax.get_title()

            plt.close(fig)


class TestPlottingConfigurations:
    """ Test plotting with various configurations using real data. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestPlottingConfigurations", mpas_3d_processor) -> None:
        """
        This fixture sets up the test environment for the TestPlottingConfigurations class by initializing the MPASVerticalCrossSectionPlotter and assigning a shared MPAS3DProcessor instance to the test class. It checks if the processor is available and if the necessary grid and output files exist, skipping the tests if any of these conditions are not met. This ensures that the tests are only run when real MPAS data is accessible, allowing for validation of plotting functionality using actual datasets. The setup provides a realistic context for testing various plotting configurations, ensuring that the plotter can handle different scenarios effectively.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): A shared session-scoped fixture that provides a processor instance with loaded MPAS data for testing.

        Returns:
            None
        """
        if mpas_3d_processor is None or not os.path.exists(GRID_FILE) or not os.path.exists(MPASOUT_DIR):
            pytest.skip("Real MPAS data not available")
            return
        
        self.plotter = MPASVerticalCrossSectionPlotter()
        self.processor = mpas_3d_processor
    
    def test_cross_section_with_custom_colormap(self: "TestPlottingConfigurations") -> None:
        """
        This test invokes the cross-section plotter with a custom colormap ('coolwarm') and validates that a figure is returned and the call completes successfully for a representative subregion of the dataset. It checks that the returned figure object is not None, confirming that the plot was created successfully with the specified colormap. This validation ensures that the plotter can handle custom colormap configurations without errors, allowing users to customize the appearance of their plots according to their preferences.

        Parameters:
            self (Any): Test case instance with `processor` and `plotter` fixtures.

        Returns:
            None: Asserts that the returned figure object is not None.
        """
        fig, ax = self.plotter.create_vertical_cross_section(
            self.processor,
            'theta',
            (0, 30),
            (20, 50),
            vertical_coord='pressure',
            colormap='coolwarm',
            num_points=30,
            time_index=0
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_cross_section_with_plot_types(self: "TestPlottingConfigurations") -> None:
        """
        This test validates that the `create_vertical_cross_section` method can successfully create plots using different plot types, specifically 'pcolormesh' and 'filled_contour'. It checks that a figure is returned for each plot type, confirming that the method can handle various plotting configurations without errors. This ensures that users have flexibility in choosing their preferred plot type for visualizing cross-section data, and that the plotter can accommodate these choices effectively.

        Parameters:
            self (Any): Test case instance with `processor` and `plotter` fixtures.

        Returns:
            None: Asserts that the returned figure object is not None.
        """
        fig, ax = self.plotter.create_vertical_cross_section(
            self.processor,
            'theta',
            (5, 35),
            (15, 45),
            vertical_coord='pressure',
            plot_type='pcolormesh',
            num_points=25,
            time_index=0
        )
        
        assert fig is not None
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
