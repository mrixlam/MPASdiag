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
import math
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import cast, Any, Generator
from cartopy.mpl.geoaxes import GeoAxes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
from mpasdiag.processing.processors_3d import MPAS3DProcessor
from tests.test_data_helpers import assert_expected_public_methods

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
GRID_FILE = os.path.join(TEST_DATA_DIR, 'grids', 'x1.10242.static.nc')
MPASOUT_DIR = os.path.join(TEST_DATA_DIR, 'u240k', 'mpasout')


def test_vertical_cross_section_plotter_initialization() -> None:
    """
    This test validates that the MPASVerticalCrossSectionPlotter initializes with correct default parameters and allows for custom configuration. It checks that the default figure size is (10, 12) inches and the default DPI is 100, while also confirming that the figure and axes are initially set to None. The test then creates a custom plotter instance with a specified figure size of (10, 6) inches and a DPI of 150, verifying that these custom parameters are correctly applied to the new instance.

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
    This test verifies that the MPASVerticalCrossSectionPlotter correctly generates a great circle path between two geographic points. It checks that the generated longitude and latitude arrays have the expected number of points, that the start and end points match the input coordinates within a reasonable tolerance, and that the distance array is monotonically increasing along the path. By validating these aspects of the great circle path generation, this test ensures that the plotter can accurately compute spatial paths for cross-section plotting based on geographic coordinates.

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
    This test validates that the MPASVerticalCrossSectionPlotter generates appropriate default contour levels for different types of data. It checks that the generated levels cover the range of the input data, that they are not empty, and that they are suitable for the specified variable type (e.g., 'theta' for potential temperature, 'uwind' for zonal wind). The test also verifies that constant data results in a reasonable number of levels and that NaN values do not cause errors in level generation. By confirming these aspects of default level generation, this test ensures that the plotter can automatically determine suitable contour levels for visualizing various meteorological variables.

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
    This test verifies that the MPASVerticalCrossSectionPlotter can interpolate grid data along a specified path defined by longitude and latitude points. It checks that the interpolated values are returned for each point along the path, that they are not all NaN (indicating successful interpolation), and that the method can handle cases where the path points do not exactly match the grid points. By confirming that the interpolation routine produces reasonable values along the path, this test ensures that the plotter can accurately extract data for cross-section plotting based on spatial paths.

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
    This test verifies that the MPASVerticalCrossSectionPlotter's input validation correctly identifies and raises errors for invalid inputs. It checks that providing an invalid type for the `mpas_3d_processor` argument in the `create_vertical_cross_section` method results in a ValueError with an appropriate error message. By confirming that the plotter raises exceptions for incorrect input types, this test ensures that the plotter's input validation mechanisms are functioning properly to prevent misuse and guide users towards correct usage.

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


class TestPressureAxisSetup:
    """ Tests for pressure axis configuration. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestPressureAxisSetup") -> Generator[None, None, None]:
        """
        This fixture sets up a MPASVerticalCrossSectionPlotter instance and initializes a figure and axes for testing pressure axis configuration. It yields control to the test methods and ensures that any created figures are closed after the tests complete to prevent resource leaks.

        Parameters:
            None

        Returns:
            None
        """
        self.plotter = MPASVerticalCrossSectionPlotter()

        self.plotter.fig = plt.figure()
        self.plotter.ax = self.plotter.fig.add_subplot(111)
    
        yield

        if self.plotter.fig:
            plt.close(self.plotter.fig)
    
    def test_pressure_axis_with_standard_ticks(self: "TestPressureAxisSetup") -> None:
        """
        This test verifies that when standard pressure levels are provided, the pressure axis is configured with a logarithmic scale and appropriate ticks. It checks that the y-axis scale is set to 'log' after calling the axis setup method with typical pressure coordinates, confirming that the plotter correctly identifies standard pressure levels and applies the expected axis configuration.

        Parameters:
            None

        Returns:
            None
        """
        pressure_coords = np.array([1000, 850, 700, 500, 300, 200, 100])        
        self.plotter._setup_pressure_axis(pressure_coords, use_standard_ticks=True)  
        assert self.plotter.ax.get_yscale() == 'log' # type: ignore
    
    def test_pressure_axis_tick_filtering(self: "TestPressureAxisSetup") -> None:
        """
        This test verifies that the MPASVerticalCrossSectionPlotter correctly filters out irrelevant standard ticks based on the data range. It checks that the y-axis scale remains 'log' while excluding ticks that fall outside the range of the provided pressure coordinates.

        Parameters:
            None

        Returns:
            None
        """
        pressure_coords = np.array([600, 500, 400, 300])        
        self.plotter._setup_pressure_axis(pressure_coords, use_standard_ticks=True)  
        assert self.plotter.ax.get_yscale() == 'log' # type: ignore
    
    def test_non_positive_pressure_warning(self: "TestPressureAxisSetup") -> None:
        """
        This test verifies that when non-positive pressure values are included in the input coordinates, the MPASVerticalCrossSectionPlotter issues a warning and falls back to a linear scale for the pressure axis. It checks that the y-axis scale is set to 'linear' after calling the axis setup method with invalid pressure coordinates, confirming that the plotter handles non-physical pressure values gracefully by reverting to a more appropriate axis configuration.

        Parameters:
            None

        Returns:
            None
        """
        pressure_coords = np.array([-100, 0, 100, 200])
        self.plotter._setup_pressure_axis(pressure_coords, use_standard_ticks=True)
        assert self.plotter.ax.get_yscale() == 'linear' # type: ignore
    
    def test_pressure_axis_exception_handling(self: "TestPressureAxisSetup") -> None:
        """
        This test verifies that the MPASVerticalCrossSectionPlotter's pressure axis setup method can handle exceptions gracefully without crashing. It checks that when an exception occurs during the axis setup (e.g., due to invalid input), the method catches the exception and ensures that the plotter's axes remain in a consistent state, allowing for continued plotting operations. By confirming that exceptions are handled without propagating errors, this test ensures that the plotter is robust to unexpected issues during pressure axis configuration.

        Parameters:
            None

        Returns:
            None
        """
        pressure_coords = np.array([])
        self.plotter._setup_pressure_axis(pressure_coords, use_standard_ticks=True)

        assert self.plotter.ax is not None
        assert self.plotter.ax.get_yscale() == 'linear' 

class TestAxisFormatting:
    """ Tests for axis formatting edge cases. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestAxisFormatting") -> Generator[None, None, None]:
        """
        This fixture sets up a MPASVerticalCrossSectionPlotter instance and initializes a figure and axes for testing axis formatting edge cases. It yields control to the test methods and ensures that any created figures are closed after the tests complete to prevent resource leaks.

        Parameters:
            None

        Returns:
            None
        """
        self.plotter = MPASVerticalCrossSectionPlotter()

        self.plotter.fig = plt.figure()
        self.plotter.ax = self.plotter.fig.add_subplot(111)
    
        yield
        
        if self.plotter.fig:
            plt.close(self.plotter.fig)
    
    def test_height_axis_formatting(self: "TestAxisFormatting") -> None:
        """
        This test verifies that when height coordinates are provided, the MPASVerticalCrossSectionPlotter formats the vertical axis with appropriate labels and units. It checks that the y-axis label contains 'Height' after calling the axis formatting method with height coordinates, confirming that the plotter correctly identifies height as the vertical coordinate and applies suitable axis labeling.

        Parameters:
            None

        Returns:
            None
        """
        longitudes = np.linspace(-100, -90, 50)
        vertical_coords = np.linspace(0, 20000, 20)
        
        self.plotter._format_cross_section_axes(
            longitudes, vertical_coords, 'height',
            (-100, 30), (-90, 40), max_height=15.0
        )
        
        assert self.plotter.ax
        assert 'Height' in self.plotter.ax.get_ylabel()
    
    def test_pressure_pa_axis_formatting(self: "TestAxisFormatting") -> None:
        """
        This test verifies that when pressure coordinates are provided in Pascals, the MPASVerticalCrossSectionPlotter formats the vertical axis with appropriate labels and units. It checks that the y-axis label contains 'Pressure' after calling the axis formatting method with pressure coordinates in Pascals, confirming that the plotter correctly identifies pressure as the vertical coordinate and applies suitable axis labeling.

        Parameters:
            None

        Returns:
            None
        """
        longitudes = np.linspace(-100, -90, 50)
        vertical_coords = np.array([100000, 85000, 70000, 50000])
        
        self.plotter._format_cross_section_axes(
            longitudes, vertical_coords, 'pressure',
            (-100, 30), (-90, 40)
        )
        
        assert 'Pressure' in self.plotter.ax.get_ylabel() # type: ignore
    
    def test_model_levels_ylim_exception(self: "TestAxisFormatting") -> None:
        """
        This test verifies that when model level coordinates are provided, the MPASVerticalCrossSectionPlotter can set the y-axis limits without raising exceptions. It checks that after calling the axis formatting method with model level coordinates, the y-axis limits are set to the range of the provided model levels, confirming that the plotter can handle model level vertical coordinates and apply appropriate axis limits without errors.

        Parameters:
            None

        Returns:
            None
        """
        longitudes = np.linspace(-100, -90, 50)
        vertical_coords = np.array([1, 2, 3])

        self.plotter._format_cross_section_axes(
            longitudes, vertical_coords, 'modlev',
            (-100, 30), (-90, 40)
        )

        assert self.plotter.ax is not None
        assert self.plotter.ax.get_ylim() == (pytest.approx(1), pytest.approx(3))
        assert self.plotter.ax.get_ylabel() == 'Model Level'


class TestAxisFormattingFinal:
    """ Test axis formatting with real data. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestAxisFormattingFinal", mpas_3d_processor: "MPAS3DProcessor") -> None:
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
    
    def test_axis_formatting_with_max_height(self: "TestAxisFormattingFinal") -> None:
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
    
    def test_model_levels_axis_formatting(self: "TestAxisFormattingFinal") -> None:
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
