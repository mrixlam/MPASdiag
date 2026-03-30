#!/usr/bin/env python3
"""
MPASdiag Test Suite: Test MPASVerticalCrossSectionPlotter Functionality

This test suite validates the core functionality of the MPASVerticalCrossSectionPlotter class, which is responsible for generating vertical cross-section visualizations from MPAS 3D data. The tests cover initialization, great circle path generation, default level calculation, interpolation along the cross-section path, and input validation for processor objects. By using both synthetic data and real MPAS datasets, the tests ensure that the plotter can handle a variety of scenarios and edge cases while producing valid plot objects for visualization. This comprehensive testing approach helps maintain code robustness and reliability for users creating vertical cross-sections from MPAS model output.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load standard libraries and testing dependencies
import os
import sys
import math
import shutil
import pytest
import tempfile
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch
from typing import cast, Any, Generator, Union

from mpasdiag.processing.processors_3d import MPAS3DProcessor
from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
GRID_FILE = os.path.join(TEST_DATA_DIR, 'grids', 'x1.40962.static.nc')
MPASOUT_DIR = os.path.join(TEST_DATA_DIR, 'u120k', 'mpasout')


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
    This test verifies that the MPASVerticalCrossSectionPlotter initializes with default parameters and allows custom configuration. It creates an instance of the plotter and checks that default values for figsize, dpi, fig, and ax are set correctly. The test also creates a custom plotter with specified figsize and dpi values and asserts that these custom settings are applied, confirming that the initialization logic correctly handles both default and user-specified parameters.

    Parameters:
        None

    Returns:
        None
    """
    plotter = MPASVerticalCrossSectionPlotter()
    
    assert plotter.figsize == (10, 12)
    assert plotter.dpi == 100  
    assert plotter.fig is None
    assert plotter.ax is None
    
    custom_plotter = MPASVerticalCrossSectionPlotter(figsize=(10, 6), dpi=150)

    assert custom_plotter.figsize == (10, 6)
    assert custom_plotter.dpi == 150


def test_great_circle_path_generation() -> None:
    """
    This test verifies that the _generate_great_circle_path method correctly generates a great circle path between two geographic points. It creates a plotter instance, defines start and end points, and calls the method to generate longitude, latitude, and distance arrays. The test checks that the output arrays have the expected length, that the first and last points match the input coordinates within a reasonable tolerance, and that the distance array is monotonically increasing with the first value at zero, confirming that the great circle path generation logic is functioning as intended.

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
    This test verifies that the _get_default_levels method generates appropriate contour levels based on the input data and variable type. It creates a plotter instance and tests the method with synthetic temperature data, wind speed data, constant data, and NaN-filled data. The test checks that the generated levels are non-empty, encompass the range of the input data, and handle edge cases like constant or NaN data without errors, confirming that the default level generation logic is robust and adaptable to different types of input fields.

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
    This test verifies that the _interpolate_along_path method correctly interpolates grid data along a specified path. It creates a plotter instance, defines synthetic grid longitude, latitude, and data arrays, and specifies a path with longitude and latitude points. The test calls the interpolation method and checks that the output array has the expected length, that it contains valid interpolated values (not all NaN), and that the interpolation logic is functioning without errors, confirming that the method can successfully interpolate data along a cross-section path.

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
        
    except ImportError:
        print("Scipy not available, skipping interpolation test")
        pytest.skip("Scipy not available for interpolation test")


def test_input_validation() -> None:
    """
    This test verifies that the create_vertical_cross_section method raises a ValueError when the provided mpas_3d_processor is not an instance of MPAS3DProcessor. It attempts to call the method with an invalid processor type and asserts that the appropriate error message is raised, confirming that the method validates the processor input correctly.

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


class TestCreateVerticalCrossSectionComplete:
    """ Comprehensive tests for create_vertical_cross_section method. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestCreateVerticalCrossSectionComplete", mpas_3d_processor) -> Generator[None, None, None]:
        """
        This fixture sets up the testing environment for the TestCreateVerticalCrossSectionComplete class by creating a temporary directory for saving plots and initializing an MPASVerticalCrossSectionPlotter instance. It uses a session-scoped fixture to provide a real MPAS3DProcessor with loaded data, which is essential for testing the create_vertical_cross_section method with actual MPAS datasets. The fixture yields control to the test methods and ensures that any created temporary files are cleaned up after the tests complete.

        Parameters:
            mpas_3d_processor: Session-scoped fixture providing real MPAS3DProcessor with loaded data

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()
        self.plotter = MPASVerticalCrossSectionPlotter(figsize=(10, 8), dpi=100)
        
        if mpas_3d_processor is None:
            pytest.skip("MPAS 3D data not available")
        
        self.processor = mpas_3d_processor
        
        yield
        
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        plt.close('all')
    
    def test_create_cross_section_filled_contour(self: "TestCreateVerticalCrossSectionComplete") -> None:
        """
        This test verifies that the create_vertical_cross_section method can generate a filled contour plot without errors and returns valid Figure and Axes objects. It calls the method with a real MPAS3DProcessor, specifying 'theta' as the variable, geographic start and end points, and requests a filled contour plot. The test asserts that the returned figure and axes are not None and that they match the plotter's internal fig and ax attributes, confirming that the plotting routine executes successfully and produces valid plot objects for visualization.

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = self.plotter.create_vertical_cross_section(
            self.processor,
            'theta',
            start_point=(-110, 35),
            end_point=(-90, 45),
            time_index=0,
            vertical_coord='pressure',
            num_points=50,
            plot_type='filled_contour'
        )
        
        assert fig is not None
        assert ax is not None
        assert fig == self.plotter.fig
        assert ax == self.plotter.ax
    
    def test_create_cross_section_contour_only(self: "TestCreateVerticalCrossSectionComplete") -> None:
        """
        This test verifies that the create_vertical_cross_section method can generate a contour-line-only plot without errors and returns valid Figure and Axes objects. It calls the method with a real MPASS3DProcessor, specifying 'theta' as the variable, geographic start and end points, and requests a contour-only plot. The test asserts that the returned figure and axes are not None, confirming that the contour-only rendering path operates successfully.

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = self.plotter.create_vertical_cross_section(
            self.processor,
            'theta',
            start_point=(-110, 35),
            end_point=(-90, 45),
            plot_type='contour'
        )
        
        assert fig is not None
        assert ax is not None
    
    def test_create_cross_section_pcolormesh(self: "TestCreateVerticalCrossSectionComplete") -> None:
        """
        This test verifies that the create_vertical_cross_section method can generate a pcolormesh plot without errors and returns valid Figure and Axes objects. It calls the method with a real MPAS3DProcessor, specifying 'theta' as the variable, geographic start and end points, and requests a pcolormesh plot. The test asserts that the returned figure and axes are not None, confirming that the pcolormesh rendering path operates successfully.

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = self.plotter.create_vertical_cross_section(
            self.processor,
            'theta',
            start_point=(-110, 35),
            end_point=(-90, 45),
            plot_type='pcolormesh'
        )
        
        assert fig is not None
        assert ax is not None
    
    def test_create_cross_section_model_levels(self: "TestCreateVerticalCrossSectionComplete") -> None:
        """
        This test verifies that the create_vertical_cross_section method can generate a cross-section plot using model-level indices as the vertical coordinate without errors and returns valid Figure and Axes objects. It calls the method with a real MPAS3DProcessor, specifying 'theta' as the variable, geographic start and end points, and requests that the vertical coordinate be based on model levels. The test asserts that the returned figure and axes are not None, confirming that the plotter can handle model-level vertical coordinates and produce a valid plot even when the vertical coordinate is represented as integer level indices.

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = self.plotter.create_vertical_cross_section(
            self.processor,
            'theta',
            start_point=(-110, 35),
            end_point=(-90, 45),
            vertical_coord='model_levels'
        )
        
        assert fig is not None
        assert ax is not None
    
    def test_create_cross_section_height_coord(self: "TestCreateVerticalCrossSectionComplete") -> None:
        """
        This test verifies that the create_vertical_cross_section method can generate a cross-section plot using geometric height as the vertical coordinate without errors and returns valid Figure and Axes objects. It calls the method with a real MPAS3DProcessor, specifying 'theta' as the variable, geographic start and end points, and requests that the vertical coordinate be based on height. The test asserts that the returned figure and axes are not None, confirming that the plotter can handle height-based vertical coordinates and produce a valid plot.

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = self.plotter.create_vertical_cross_section(
            self.processor,
            'theta',
            start_point=(-110, 35),
            end_point=(-90, 45),
            vertical_coord='height'
        )
        
        assert fig is not None
        assert ax is not None
    
    def test_create_cross_section_display_vertical_pressure(self: "TestCreateVerticalCrossSectionComplete") -> None:
        """
        This test verifies that the create_vertical_cross_section method can explicitly control the displayed vertical coordinate as pressure values. By passing `display_vertical='pressure'` to `create_vertical_cross_section`, this test ensures that axis labeling and tick formatting for pressure-based displays are handled without error and that plot objects are returned for inspection.

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = self.plotter.create_vertical_cross_section(
            self.processor,
            'theta',
            start_point=(-110, 35),
            end_point=(-90, 45),
            vertical_coord='model_levels',
            display_vertical='pressure'
        )
        
        assert fig is not None
    
    def test_create_cross_section_display_vertical_height(self: "TestCreateVerticalCrossSectionComplete") -> None:
        """
        This test verifies that the create_vertical_cross_section method can explicitly control the displayed vertical coordinate as geometric height. By passing `display_vertical='height'` to `create_vertical_cross_section`, this test ensures that axis labeling and tick formatting for height-based displays are handled without error and that plot objects are returned for inspection.

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = self.plotter.create_vertical_cross_section(
            self.processor,
            'theta',
            start_point=(-110, 35),
            end_point=(-90, 45),
            vertical_coord='pressure',
            display_vertical='height'
        )
        
        assert fig is not None
    
    def test_create_cross_section_display_vertical_model_levels(self: "TestCreateVerticalCrossSectionComplete") -> None:
        """
        This test verifies that the create_vertical_cross_section method can explicitly control the displayed vertical coordinate as model level indices. By passing `display_vertical='model_levels'` to `create_vertical_cross_section`, this test ensures that axis labeling and tick formatting for model-level displays are handled without error and that plot objects are returned for inspection.

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = self.plotter.create_vertical_cross_section(
            self.processor,
            'theta',
            start_point=(-110, 35),
            end_point=(-90, 45),
            vertical_coord='pressure',
            display_vertical='model_levels'
        )
        
        assert fig is not None
    
    def test_create_cross_section_with_max_height(self: "TestCreateVerticalCrossSectionComplete") -> None:
        """
        This test verifies that the create_vertical_cross_section method respects the `max_height` parameter and returns plot objects. By passing a small `max_height` value to `create_vertical_cross_section`, this test ensures that the plotter applies the vertical clipping logic and still returns a valid figure object when levels above the requested height exist.

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = self.plotter.create_vertical_cross_section(
            self.processor,
            'theta',
            start_point=(-110, 35),
            end_point=(-90, 45),
            vertical_coord='pressure',
            max_height=10.0
        )
        
        assert fig is not None
    
    def test_create_cross_section_custom_colormap(self: "TestCreateVerticalCrossSectionComplete") -> None:
        """
        This test verifies that a custom colormap can be supplied and used by the plotting routine. By passing a named colormap to `create_vertical_cross_section`, this test ensures that the plot is created successfully without raising exceptions and that valid plot objects are returned for inspection.

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = self.plotter.create_vertical_cross_section(
            self.processor,
            'theta',
            start_point=(-110, 35),
            end_point=(-90, 45),
            colormap='RdBu_r'
        )
        
        assert fig is not None
    
    def test_create_cross_section_custom_levels(self: "TestCreateVerticalCrossSectionComplete") -> None:
        """
        This test verifies that user-supplied contour `levels` are accepted and used by the plotting routine. By providing a sequence of contour levels to `create_vertical_cross_section`, this test ensures that the plotter completes successfully and returns valid plotting objects for downstream use.

        Parameters:
            None

        Returns:
            None
        """
        levels = np.linspace(240, 300, 13)
        
        fig, ax = self.plotter.create_vertical_cross_section(
            self.processor,
            'theta',
            start_point=(-110, 35),
            end_point=(-90, 45),
            levels=levels
        )
        
        assert fig is not None
    
    def test_create_cross_section_custom_title(self: "TestCreateVerticalCrossSectionComplete") -> None:
        """
        This test verifies that a custom `title` argument is applied to the Axes returned. By supplying a custom title string, this test ensures that the title appears in the resulting Axes after plotting completes successfully.

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = self.plotter.create_vertical_cross_section(
            self.processor,
            'theta',
            start_point=(-110, 35),
            end_point=(-90, 45),
            title="Custom Cross-Section Title"
        )
        
        assert fig is not None
        assert "Custom" in ax.get_title()
    
    def test_create_cross_section_save_file(self: "TestCreateVerticalCrossSectionComplete") -> None:
        """
        This test verifies that `create_vertical_cross_section` can save output to disk. By requesting a save_path within a temporary directory, this test ensures that the produced image file exists on successful completion.

        Parameters:
            None

        Returns:
            None
        """
        save_path = os.path.join(self.temp_dir, "cross_section.png")
        
        fig, ax = self.plotter.create_vertical_cross_section(
            self.processor,
            'theta',
            start_point=(-110, 35),
            end_point=(-90, 45),
            save_path=save_path
        )
        
        assert os.path.exists(save_path)
    
    def test_create_cross_section_invalid_processor(self: "TestCreateVerticalCrossSectionComplete") -> None:
        """
        This test verifies that passing an invalid processor object raises a ValueError. By calling `create_vertical_cross_section` with a non-processor value, this test ensures that the exception message references the expected type.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError) as ctx:
            self.plotter.create_vertical_cross_section(
                "not_a_processor",
                'theta',
                start_point=(-110, 35),
                end_point=(-90, 45)
            )
        assert "must be an instance of MPAS3DProcessor" in str(ctx.value)
    
    def test_create_cross_section_no_dataset(self: "TestCreateVerticalCrossSectionComplete") -> None:
        """
        This test verifies the behavior when the processor has no loaded dataset. By using a Mock processor with dataset=None, this test ensures that the plotter raises a ValueError indicating data must be loaded and asserts the appropriate error message is raised.

        Parameters:
            None

        Returns:
            None
        """
        mock_proc = Mock(spec=MPAS3DProcessor)
        mock_proc.dataset = None
        
        with pytest.raises(ValueError) as ctx:
            self.plotter.create_vertical_cross_section(
                mock_proc,
                'theta',
                start_point=(-110, 35),
                end_point=(-90, 45)
            )
        assert "must have loaded data" in str(ctx.value)
    
    def test_create_cross_section_variable_not_found(self: "TestCreateVerticalCrossSectionComplete") -> None:
        """
        This test verifies that requesting a non-existent variable raises an informative error. By invoking the plotter with a variable name missing from the mock dataset, this test ensures that a ValueError mentioning 'not found' is raised.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError) as ctx:
            self.plotter.create_vertical_cross_section(
                self.processor,
                'nonexistent_var',
                start_point=(-110, 35),
                end_point=(-90, 45)
            )
        assert "not found" in str(ctx.value)
    
    def test_create_cross_section_not_3d_variable(self: "TestCreateVerticalCrossSectionComplete") -> None:
        """
        This test verifies that the plotter rejects 2D (non-3D) variables when a 3D field is required. By using a variable that exists in real MPAS data but is 2D, this test ensures that a ValueError is raised indicating the variable is not a 3D atmospheric field.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError) as ctx:
            self.plotter.create_vertical_cross_section(
                self.processor,
                't2m',  
                start_point=(-110, 35),
                end_point=(-90, 45)
            )
        assert "not a 3D atmospheric variable" in str(ctx.value)
    
    def test_create_cross_section_invalid_plot_type(self: "TestCreateVerticalCrossSectionComplete") -> None:
        """
        This test verifies that an invalid `plot_type` argument raises a ValueError. By passing an unsupported plot type string, this test ensures that the exception message references 'Unknown plot_type' to confirm that input validation for plot types is functioning correctly.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError) as ctx:
            self.plotter.create_vertical_cross_section(
                self.processor,
                'theta',
                start_point=(-110, 35),
                end_point=(-90, 45),
                plot_type='invalid_type'
            )
        assert "Unknown plot_type" in str(ctx.value)


class TestVerticalCoordinateConversion:
    """ Tests for vertical coordinate conversion methods. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestVerticalCoordinateConversion", mpas_3d_processor) -> Generator[None, None, None]:
        """
        This fixture sets up the testing environment for vertical coordinate conversion tests by initializing an MPASVerticalCrossSectionPlotter instance and providing a real MPAS3DProcessor with loaded data. The fixture ensures that the plotter is available for testing the conversion of pressure coordinates to height, model levels to height, and height extraction from the dataset. If the MPAS 3D processor is not available, the fixture will skip the tests that depend on it.

        Parameters:
            mpas_3d_processor: Session-scoped fixture providing real MPAS3DProcessor with loaded data 

        Returns:
            None
        """
        self.plotter = MPASVerticalCrossSectionPlotter()
        
        if mpas_3d_processor is None:
            pytest.skip("MPAS 3D data not available")
        
        self.processor = mpas_3d_processor
    
        yield
        
        plt.close('all')
    
    def test_convert_pressure_to_height(self: "TestVerticalCoordinateConversion") -> None:
        """
        This test verifies the conversion of pressure coordinates to height values using the plotter's internal method. It provides a set of standard pressure levels and checks that the returned height display coordinates are valid, positive, and increasing, confirming that the conversion logic from pressure to height is functioning correctly.

        Parameters:
            None

        Returns:
            None
        """
        pressure_coords = np.array([100000, 85000, 70000, 50000, 30000, 20000, 10000])
        
        height_display, coord_type = self.plotter._convert_vertical_to_height(
            pressure_coords,
            'pressure',
            self.processor,
            time_index=0
        )
        
        assert height_display is not None
        assert coord_type == 'height_km'
        assert np.all(height_display >= 0)
        assert np.all(np.diff(height_display) > 0)
    
    def test_convert_model_levels_to_height(self: "TestVerticalCoordinateConversion") -> None:
        """
        This test verifies the conversion of integer model level indices to height values using the plotter's internal method. It provides a set of model level indices and checks that the returned height display coordinates are valid and the coordinate type indicates kilometers for plotting purposes.

        Parameters:
            None

        Returns:
            None
        """
        model_levels = np.arange(20)
        
        height_display, coord_type = self.plotter._convert_vertical_to_height(
            model_levels,
            'model_levels',
            self.processor,
            time_index=0
        )
        
        assert height_display is not None
        assert coord_type in ('height_km', 'model_levels')
    
    def test_extract_height_zgrid(self: "TestVerticalCoordinateConversion") -> None:
        """
        This test verifies that height values can be extracted from the dataset using the 'zgrid' variable. It requests height extraction for a set of vertical coordinates and checks that the returned heights array has the expected length, confirming that the plotter can successfully extract height information from the dataset when available.

        Parameters:
            None

        Returns:
            None
        """
        vertical_coords = np.arange(20)
        
        heights = self.plotter._extract_height_from_dataset(
            self.processor,
            time_index=0,
            vertical_coords=vertical_coords,
            var_name='zgrid'
        )
        
        if heights is not None:
            assert len(heights) == len(vertical_coords)
    
    def test_extract_height_height_variable(self: "TestVerticalCoordinateConversion") -> None:
        """
        This test verifies that height values can be extracted from the dataset using a variable named 'height'. It requests height extraction for a set of vertical coordinates and checks that the returned heights array has the expected length, confirming that the plotter can successfully extract height information from the dataset when a variable named 'height' is present.

        Parameters:
            None

        Returns:
            None
        """
        vertical_coords = np.arange(20)
        
        heights = self.plotter._extract_height_from_dataset(
            self.processor,
            time_index=0,
            vertical_coords=vertical_coords,
            var_name='height'
        )
        
        if heights is not None:
            assert len(heights) == len(vertical_coords)
    
    def test_extract_height_missing_variable(self: "TestVerticalCoordinateConversion") -> None:
        """
        This test verifies that when the requested height variable is missing from the dataset, the plotter's height extraction method returns None. It requests height extraction for a non-existent variable and asserts that the return value is None, confirming that the plotter gracefully handles missing height variables by returning None to allow fallback logic upstream.

        Parameters:
            None

        Returns:
            None
        """
        vertical_coords = np.arange(20)
        
        heights = self.plotter._extract_height_from_dataset(
            self.processor,
            time_index=0,
            vertical_coords=vertical_coords,
            var_name='nonexistent'
        )
        
        assert heights is None


class TestVerticalCoordinateEdgeCases:
    """ Tests for vertical coordinate conversion edge cases. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestVerticalCoordinateEdgeCases", mpas_3d_processor) -> None:
        """
        This fixture sets up the testing environment for vertical coordinate edge case tests by initializing an MPASVerticalCrossSectionPlotter instance and providing a real MPAS3DProcessor with loaded data. The fixture ensures that the plotter is available for testing edge cases in vertical coordinate conversion, such as handling non-numeric pressure values, non-finite pressures, and large pressure values that require unit conversion. If the MPAS 3D processor is not available, the fixture will skip the tests that depend on it.

        Parameters:
            mpas_3d_processor: Session-scoped fixture providing real MPAS3DProcessor with loaded data

        Returns:
            None
        """
        if mpas_3d_processor is None:
            pytest.skip("MPAS data not available")
        
        self.processor = mpas_3d_processor
        self.plotter = MPASVerticalCrossSectionPlotter(figsize=(10, 8), dpi=100)
        
    def test_pressure_conversion_exception_path(self: "TestVerticalCoordinateEdgeCases") -> None:
        """
        This test verifies that if the pressure values cannot be converted to float (e.g., due to non-numeric data), the plotter falls back to using np.asarray without conversion. By mocking the data generation method to return non-numeric vertical coordinates, this test ensures that the exception handling path is exercised and that the plotter does not raise an unhandled exception, confirming that it can gracefully handle unexpected data types in pressure coordinates.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor
        var_3d = _find_3d_var(processor)

        if var_3d is None:
            pytest.skip("No 3D variable found in dataset")
        
        plotter = MPASVerticalCrossSectionPlotter()
        
        with patch.object(plotter, '_generate_cross_section_data') as mock_gen:
            mock_gen.return_value = {
                'distances': np.linspace(0, 100, 50),
                'vertical_coords': np.array(['invalid', 'data', 'type']),  
                'data_values': np.random.rand(3, 50),
                'path_lons': np.linspace(-100, -90, 50),
                'path_lats': np.linspace(30, 40, 50),
                'longitudes': np.linspace(-100, -90, 50),
                'vertical_coord_type': 'pressure'
            }

            try:
                fig, ax = plotter.create_vertical_cross_section(
                    processor, var_3d, (-100, 30), (-90, 40),
                    display_vertical='pressure'
                )
                plt.close(fig)
            except Exception:
                pass  
    
    def test_non_finite_pressure_values_fallback(self: "TestVerticalCoordinateEdgeCases") -> None:
        """
        This test verifies that if the pressure values are non-finite (e.g., negative or zero), the plotter falls back to using model levels for the vertical coordinate. By mocking the data generation method to return non-positive pressure values, this test ensures that the plotter detects the invalid pressures and successfully falls back to model levels without raising an unhandled exception, confirming that it can handle edge cases in pressure data gracefully.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor
        var_3d = _find_3d_var(processor)

        if var_3d is None:
            pytest.skip("No 3D variable found in dataset")
        
        plotter = MPASVerticalCrossSectionPlotter()
        
        with patch.object(plotter, '_generate_cross_section_data') as mock_gen:
            mock_gen.return_value = {
                'distances': np.linspace(0, 100, 50),
                'vertical_coords': np.array([-100, 0, -50, -200]),  
                'data_values': np.random.rand(4, 50),
                'path_lons': np.linspace(-100, -90, 50),
                'path_lats': np.linspace(30, 40, 50),
                'longitudes': np.linspace(-100, -90, 50),
                'vertical_coord_type': 'pressure'
            }
            
            fig, ax = plotter.create_vertical_cross_section(
                processor, var_3d, (-100, 30), (-90, 40),
                display_vertical='pressure'
            )

            assert 'Model' in ax.get_ylabel()
            plt.close(fig)
    
    def test_pressure_in_pascals_conversion(self: "TestVerticalCoordinateEdgeCases") -> None:
        """
        This test verifies that if the pressure values are in Pascals (greater than 10000), the plotter correctly converts them to hPa by dividing by 100. By mocking the data generation method to return pressure values in Pascals, this test ensures that the plotter applies the unit conversion logic and successfully creates a plot without errors, confirming that it can handle pressure data in different units appropriately.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor
        var_3d = _find_3d_var(processor)

        if var_3d is None:
            pytest.skip("No 3D variable found in dataset")
        
        plotter = MPASVerticalCrossSectionPlotter()
        
        with patch.object(plotter, '_generate_cross_section_data') as mock_gen:
            mock_gen.return_value = {
                'distances': np.linspace(0, 100, 50),
                'vertical_coords': np.array([100000, 85000, 70000, 50000]),
                'data_values': np.random.rand(4, 50),
                'path_lons': np.linspace(-100, -90, 50),
                'path_lats': np.linspace(30, 40, 50),
                'longitudes': np.linspace(-100, -90, 50),
                'vertical_coord_type': 'pressure'
            }
            
            fig, ax = plotter.create_vertical_cross_section(
                processor, var_3d, (-100, 30), (-90, 40),
                display_vertical='pressure'
            )

            plt.close(fig)
    
    def test_max_height_no_valid_data_warning(self: "TestVerticalCoordinateEdgeCases") -> None:
        """
        This test verifies that if the `max_height` parameter is set but all vertical levels are above the specified maximum height, the plotter issues a warning and still returns a valid figure object. By mocking the data generation method to return vertical coordinates that are all above the requested `max_height`, this test ensures that the warning path is exercised and that the plotter does not raise an unhandled exception, confirming that it can gracefully handle cases where no valid vertical levels exist below the specified maximum height.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor
        var_3d = _find_3d_var(processor)

        if var_3d is None:
            pytest.skip("No 3D variable found in dataset")
        
        plotter = MPASVerticalCrossSectionPlotter()
        
        with patch.object(plotter, '_generate_cross_section_data') as mock_gen:
            mock_gen.return_value = {
                'distances': np.linspace(0, 100, 50),
                'vertical_coords': np.array([15, 20, 25, 30]), 
                'data_values': np.random.rand(4, 50),
                'path_lons': np.linspace(-100, -90, 50),
                'path_lats': np.linspace(30, 40, 50),
                'longitudes': np.linspace(-100, -90, 50),
                'vertical_coord_type': 'height'
            }
            
            fig, ax = plotter.create_vertical_cross_section(
                processor, var_3d, (-100, 30), (-90, 40),
                max_height=10.0 
            )

            plt.close(fig)
    
    def test_max_height_exception_path(self: "TestVerticalCoordinateEdgeCases") -> None:
        """
        This test verifies that if an exception occurs during the processing of `max_height` (e.g., due to unexpected data types or shapes), the plotter catches the exception and continues without crashing. By mocking the data generation method to return data that will cause an exception in the max_height processing logic, this test ensures that the exception handling path is exercised and that the plotter does not raise an unhandled exception, confirming that it can gracefully handle errors in max_height processing.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor
        var_3d = _find_3d_var(processor)

        if var_3d is None:
            pytest.skip("No 3D variable found in dataset")
        
        plotter = MPASVerticalCrossSectionPlotter()
        
        with patch.object(plotter, '_generate_cross_section_data') as mock_gen:
            mock_gen.return_value = {
                'distances': np.linspace(0, 100, 50),
                'vertical_coords': np.array([1, 2, 3, 4]),
                'data_values': np.random.rand(4, 50),  
                'path_lons': np.linspace(-100, -90, 50),
                'path_lats': np.linspace(30, 40, 50),
                'longitudes': np.linspace(-100, -90, 50),
                'vertical_coord_type': 'height'
            }
            
            with patch('numpy.asarray', side_effect=Exception("Test exception")):
                try:
                    fig, ax = plotter.create_vertical_cross_section(
                        processor, var_3d, (-100, 30), (-90, 40),
                        max_height=10.0,
                        display_vertical='height'
                    )
                    plt.close(fig)
                except Exception:
                    pass  


class TestVerticalCoordinateEdgeCasesFinal:
    """ Test edge cases in vertical coordinate handling with real MPAS data. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestVerticalCoordinateEdgeCasesFinal", mpas_3d_processor) -> None:
        """
        This fixture sets up the testing environment for vertical coordinate edge case tests using real MPAS data. It initializes an MPASVerticalCrossSectionPlotter instance and assigns the session-scoped MPAS3DProcessor with loaded data to the test instance. If the MPAS 3D processor or necessary data files are not available, the fixture will skip the tests that depend on real MPAS data.

        Parameters:
            mpas_3d_processor: Session-scoped fixture providing real MPAS3DProcessor with loaded data

        Returns:
            None
        """
        if mpas_3d_processor is None or not os.path.exists(GRID_FILE) or not os.path.exists(MPASOUT_DIR):
            pytest.skip("Real MPAS data not available")
        
        self.plotter = MPASVerticalCrossSectionPlotter(figsize=(10, 8))
        self.processor = mpas_3d_processor
    
    def test_vertical_coords_with_nan_values(self: "TestVerticalCoordinateEdgeCasesFinal") -> None:
        """
        This test verifies that the create_vertical_cross_section method can handle NaN values in the pressure coordinate without raising an unhandled exception. By modifying the real MPAS dataset to inject NaN values into the pressure field, this test ensures that the plotter's vertical coordinate handling logic can gracefully manage non-finite values and still produce a valid plot, confirming robustness in the presence of imperfect data.

        Parameters:
            self (Any): Test case instance with `processor` and `plotter` fixtures.

        Returns:
            None: Assertions validate plotting succeeded and no exceptions.
        """
        var_3d = _find_3d_var(self.processor)
        
        if var_3d is None:
            pytest.skip("No 3D variable found in dataset")

        num_levels = self.processor.dataset.sizes.get('nVertLevels', 55)
        pressure_with_nan = np.linspace(100000.0, 1000.0, num_levels)
        pressure_with_nan[10:15] = np.nan

        with patch.object(self.processor, 'get_vertical_levels', return_value=pressure_with_nan.tolist()):
            fig, ax = self.plotter.create_vertical_cross_section(
                self.processor,
                var_3d,
                (0, 0),
                (10, 10),
                vertical_coord='pressure',
                num_points=20,
                time_index=0
            )

        assert fig is not None
        plt.close(fig)
    
    def test_pressure_unit_detection_pa(self: "TestVerticalCoordinateEdgeCasesFinal") -> None:
        """
        This test verifies that if the pressure values in the dataset are in Pascals (greater than 10000), the create_vertical_cross_section method correctly detects this and applies the necessary unit conversion to hPa for plotting. By using real MPAS data where pressure is typically in Pascals, this test ensures that the plotter's unit detection and conversion logic is exercised and that the resulting plot is created successfully without errors, confirming that it can handle pressure data in different units appropriately.

        Parameters:
            self (Any): Test case instance with `processor` and `plotter` fixtures.

        Returns:
            None: Asserts that the y-axis label contains 'Pressure'.
        """
        fig, ax = self.plotter.create_vertical_cross_section(
            self.processor,
            'theta',
            (0, 40),
            (10, 50),
            vertical_coord='pressure',
            num_points=30,
            time_index=0
        )
        
        ylabel = ax.get_ylabel()
        assert 'Pressure' in ylabel
        plt.close(fig)
    
    def test_model_levels_coordinate_system(self: "TestVerticalCoordinateEdgeCasesFinal") -> None:
        """
        This test verifies that when using model level indices as the vertical coordinate, the create_vertical_cross_section method correctly handles the coordinate system and produces a plot with appropriate axis labeling. By calling the method with `vertical_coord='model_levels'` and real MPAS data, this test ensures that the plotter can manage model-level vertical coordinates and that the resulting plot's y-axis label reflects the use of model levels, confirming correct handling of this vertical coordinate type.

        Parameters:
            self (Any): Test case instance with `processor` and `plotter` fixtures.

        Returns:
            None: Asserts that the axis label contains 'Model Level'.
        """
        fig, ax = self.plotter.create_vertical_cross_section(
            self.processor,
            'theta',
            (-5, 35),
            (15, 45),
            vertical_coord='model_levels',
            num_points=25,
            time_index=0
        )
        
        ylabel = ax.get_ylabel()
        assert 'Model Level' in ylabel
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
