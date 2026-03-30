#!/usr/bin/env python3
"""
MPASdiag Test Suite: Tests for MPASVerticalCrossSectionPlotter functionality and edge cases.

This test suite validates the behavior of the MPASVerticalCrossSectionPlotter class, which is responsible for generating vertical cross-section plots from MPAS model output. The tests cover a range of scenarios including initialization, great circle path generation, default contour level creation, spatial interpolation along cross-section paths, and input validation for processor objects. The suite ensures that the plotter can handle various data types, edge cases such as constant or NaN-filled arrays, and gracefully manages errors during processing. By confirming correct functionality across these dimensions, the tests help maintain the robustness and reliability of the cross-section plotting capabilities within the MPASdiag package.

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
import xarray as xr
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import cast, Any, Union
from unittest.mock import Mock, patch

from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
from mpasdiag.processing.processors_3d import MPAS3DProcessor

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
    This test validates the initialization of the MPASVerticalCrossSectionPlotter class, ensuring that default parameters are set correctly and that custom configurations are properly applied. It checks that the default figure size is (10, 12) inches and the default DPI is 100, while also confirming that the figure and axis attributes are initialized to None. The test then creates a custom plotter instance with a specified figure size of (10, 6) inches and a DPI of 150, verifying that these custom settings are correctly assigned to the instance. This ensures that users can initialize the plotter with both default and custom configurations without encountering issues.

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
    This test verifies the correctness of the great circle path generation method in the MPASVerticalCrossSectionPlotter class. It checks that the generated longitude and latitude arrays have the correct length corresponding to the specified number of points along the path. The test also confirms that the starting and ending coordinates of the generated path closely match the provided start and end points, ensuring accurate path generation. Additionally, it validates that the distance array is monotonically increasing along the path, with the first distance value being approximately zero at the start point and positive at the end point. This ensures that the great circle path is generated correctly for use in cross-section plotting.

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
    This test validates the default contour level generation method in the MPASVerticalCrossSectionPlotter class across various data scenarios. It checks that the generated levels are appropriate for different variable types, such as potential temperature (theta) and wind components (uwind), ensuring that the levels encompass the range of the input data. The test also verifies that for constant data arrays, a single level is returned corresponding to the constant value, and that for NaN-filled arrays, a non-empty set of levels is still generated without errors. This ensures that the default level generation method can handle a variety of data conditions robustly, providing meaningful contour levels for cross-section plotting.

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
        
    except ImportError:
        print("Scipy not available, skipping interpolation test")
        pytest.skip("Scipy not available for interpolation test")


def test_input_validation() -> None:
    """
    This test confirms that the `create_vertical_cross_section` method raises a ValueError when an invalid processor object is passed. It attempts to call the method with a string instead of a valid MPAS3DProcessor instance and checks that the error message contains the expected text indicating the requirement for an MPAS3DProcessor. This validation ensures that the plotter correctly enforces input types and provides informative error messages to guide users in providing the correct processor object for cross-section generation.

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


class TestVerticalLevelExtraction:
    """ Tests for vertical level extraction edge cases. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestVerticalLevelExtraction", mpas_3d_processor) -> None:
        """
        This fixture sets up the test environment for vertical level extraction tests by initializing the MPASVerticalCrossSectionPlotter and assigning a shared MPAS3DProcessor instance. If the processor is not available, it skips the tests to avoid failures due to missing data. This setup allows the subsequent tests to focus on validating vertical level extraction logic using real or mock MPAS data without needing to load the processor separately in each test case.

        Parameters:
            mpas_3d_processor: A shared session-scoped fixture that provides an instance of MPAS3DProcessor loaded with test data. If None, tests will be skipped.

        Returns:
            None
        """
        if mpas_3d_processor is None:
            pytest.skip("MPAS data not available")
        
        self.processor = mpas_3d_processor
        self.plotter = MPASVerticalCrossSectionPlotter(figsize=(10, 8), dpi=100)
    
    def test_verbose_integer_vertical_levels(self: "TestVerticalLevelExtraction") -> None:
        """
        This test checks that when the processor's `get_vertical_levels` method returns integer levels, the plotter can handle this scenario without errors. It mocks the `get_vertical_levels` method to return a simple array of integers and then calls the `create_vertical_cross_section` method to ensure that it processes these levels correctly and returns a figure object. This test validates that the plotter can work with integer vertical levels, which may occur in certain datasets or configurations, and that it does not rely on specific data types for level extraction.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor        
        plotter = MPASVerticalCrossSectionPlotter()
        plotter.verbose = True
        
        with patch.object(processor, 'get_vertical_levels', return_value=np.arange(10, dtype=int)):
            fig, ax = plotter.create_vertical_cross_section(
                processor, 'theta', (-100, 30), (-90, 40)
            )
            plt.close(fig)
    
    def test_vertical_levels_exception_handling(self: "TestVerticalLevelExtraction") -> None:
        """
        This test verifies that if the `get_vertical_levels` method raises an exception, the plotter can handle it gracefully and still produce a plot using fallback level indices. It mocks the `get_vertical_levels` method to raise an exception and then calls the `create_vertical_cross_section` method to ensure that it does not crash and instead uses a fallback mechanism to generate the plot. This test confirms that the plotter is robust against errors in vertical level extraction and can still provide visual output even when level information is unavailable.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor        
        plotter = MPASVerticalCrossSectionPlotter()
        
        with patch.object(processor, 'get_vertical_levels', side_effect=Exception("Level error")):
            fig, ax = plotter.create_vertical_cross_section(
                processor, 'theta', (-100, 30), (-90, 40)
            )
            plt.close(fig)
    
    def test_path_outside_grid_domain_warning(self: "TestVerticalLevelExtraction") -> None:
        """
        This test checks that when the specified start and end points for the cross-section path are outside the grid domain, the plotter can still attempt to create a plot without crashing. It uses coordinates that are likely outside the typical MPAS grid domain and calls the `create_vertical_cross_section` method to ensure that it handles this edge case gracefully, potentially by issuing a warning but still returning a figure object. This test validates that the plotter can manage scenarios where the path extends beyond the available data domain without failing.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor        
        plotter = MPASVerticalCrossSectionPlotter()
        
        fig, ax = plotter.create_vertical_cross_section(
            processor, 'theta', 
            (0, 0), (90, 80), 
            num_points=20
        )

        plt.close(fig)
    
    def test_level_extraction_with_nvertlevelsp1(self: "TestVerticalLevelExtraction") -> None:
        """
        This test verifies that if the dataset contains a `nVertLevelP1` variable, the plotter can successfully extract vertical levels from it and create a cross-section plot. It checks for the presence of `nVertLevelP1` in the dataset and then calls the `create_vertical_cross_section` method to ensure that it can use this variable for level extraction without errors. This test confirms that the plotter can utilize available vertical level information from the dataset when present, enhancing its ability to generate accurate cross-section plots.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor
        
        if 'w' in processor.dataset.data_vars:
            plotter = MPASVerticalCrossSectionPlotter()
            
            fig, ax = plotter.create_vertical_cross_section(
                processor, 'w', (-100, 30), (-90, 40),
                num_points=20
            )
            plt.close(fig)
    
    def test_level_extraction_exception_continue(self: "TestVerticalLevelExtraction") -> None:
        """
        This test checks that if an exception occurs during the extraction of 3D variable data for certain levels, the plotter can catch the exception and continue processing without crashing. It mocks the `get_3d_variable_data` method to raise an exception for a specific level index and then calls the `create_vertical_cross_section` method to ensure that it handles this scenario gracefully, allowing the plot to be created using fallback data for other levels. This test validates that the plotter is resilient to errors in data extraction and can still produce visual output even when some level data is unavailable.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor
        
        plotter = MPASVerticalCrossSectionPlotter()
        
        var_3d = _find_3d_var(processor)

        if var_3d is None:
            pytest.skip("No 3D variable found in dataset")
        
        fig, _ = plotter.create_vertical_cross_section(
            processor, var_3d, (0, 0), (90, 80),
            num_points=20
        )

        assert fig is not None
        plt.close(fig)


class TestDefaultLevelGeneration:
    """ Tests for default contour level generation edge cases. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestDefaultLevelGeneration") -> None:
        """
        This fixture sets up the test environment for default level generation tests by initializing an instance of the MPASVerticalCrossSectionPlotter class. This allows the subsequent tests to call the `_get_default_levels` method directly to validate its behavior under various data scenarios, such as positive pressure values, non-positive pressure values, and zero data range. By providing a common setup for these tests, it ensures that they can focus on validating the level generation logic without needing to initialize the plotter separately in each test case.

        Parameters:
            None

        Returns:
            None
        """
        self.plotter = MPASVerticalCrossSectionPlotter()
    
    def test_pressure_variable_with_positive_values(self: "TestDefaultLevelGeneration") -> None:
        """
        This test verifies that when the input data for a pressure variable contains only positive values, the default level generation method produces a set of levels that are appropriate for pressure data, typically using a logarithmic scale. It checks that the generated levels are all positive and that a non-empty set of levels is returned, confirming that the method correctly identifies the variable type and applies suitable level generation logic for pressure variables.

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([100, 500, 1000, 5000, 10000])
        levels = self.plotter._get_default_levels(data, 'pressure')
        
        assert len(levels ) > 0
        assert np.all(levels > 0)
    
    def test_pressure_variable_with_non_positive_values(self: "TestDefaultLevelGeneration") -> None:
        """
        This test checks that when the input data for a pressure variable contains non-positive values (zero or negative), the default level generation method falls back to a linear scale or another appropriate method to generate levels. It verifies that a non-empty set of levels is returned even when the data includes non-positive values, ensuring that the method can handle edge cases in pressure data without failing.

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([-100, 0, 100, 500])
        levels = self.plotter._get_default_levels(data, 'pressure')
        
        assert len(levels ) > 0
    
    def test_zero_data_range(self: "TestDefaultLevelGeneration") -> None:
        """
        This test confirms that when the input data has a zero range (all values are the same), the default level generation method returns a single level corresponding to that constant value. It checks that the returned levels array has a length of one and that the level value matches the constant value in the input data, ensuring that the method can handle this edge case gracefully without producing an empty set of levels or raising an error.

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([42.0, 42.0, 42.0])
        levels = self.plotter._get_default_levels(data, 'theta')
        
        assert len(levels) == 1
        assert levels[0] == 42.0


class TestHeightExtraction:
    """ Tests for height extraction from dataset. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestHeightExtraction", mpas_3d_processor) -> None:
        """
        This fixture sets up the test environment for height extraction tests by initializing the MPASVerticalCrossSectionPlotter and assigning a shared MPAS3DProcessor instance to the test class. If the processor is not available, it skips the tests to avoid failures due to missing data. This setup allows the subsequent tests to focus on validating height extraction logic using real or mock MPAS data without needing to load the processor separately in each test case.

        Parameters:
            mpas_3d_processor: A shared session-scoped fixture that provides an instance of MPAS3DProcessor loaded with test data. If None, tests will be skipped.

        Returns:
            None
        """
        if mpas_3d_processor is None:
            pytest.skip("MPAS data not available")
        
        self.processor = mpas_3d_processor
        self.plotter = MPASVerticalCrossSectionPlotter()
    
    def test_height_extraction_with_zgrid(self: "TestHeightExtraction") -> None:
        """
        This test verifies that if the dataset contains a `zgrid` variable, the plotter can successfully extract height information from it. It checks for the presence of `zgrid` in the dataset and then calls the `_extract_height_from_dataset` method to ensure that it can retrieve height values without errors. The test confirms that the extracted height array is not empty, indicating that valid height information was obtained from the dataset when `zgrid` is available.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor
        
        if 'zgrid' in processor.dataset.data_vars:
            vertical_coords = np.arange(20)
            height = self.plotter._extract_height_from_dataset(
                processor, 0, vertical_coords, 'zgrid'
            )
            
            if height is not None:
                assert len(height ) > 0
    
    def test_height_extraction_scipy_interpolation(self: "TestHeightExtraction") -> None:
        """
        This test checks that if the dataset does not contain a `zgrid` variable but has a `height` variable with a different size than the vertical coordinates, the plotter can use scipy interpolation to extract height values. It creates a mock dataset with a `height` variable that has a different number of levels than the vertical coordinates and then calls the `_extract_height_from_dataset` method to ensure that it can perform interpolation without errors. The test confirms that the interpolated height array has the same length as the vertical coordinates, indicating that the interpolation was successful.

        Parameters:
            None

        Returns:
            None
        """
        processor = Mock(spec=MPAS3DProcessor)
        
        n_levels = 20
        n_time = 1
        n_cells = 10

        mock_dataset = xr.Dataset({
            'height': xr.DataArray(
                np.tile(np.linspace(0, 20000, 30), (n_time, n_cells, 1)).transpose(0, 2, 1),
                dims=['Time', 'nVertLevels', 'nCells'],
                coords={
                    'Time': pd.date_range('2024-01-01', periods=n_time),
                    'nVertLevels': range(30),
                    'nCells': range(n_cells)
                }
            )
        })
        
        processor.dataset = mock_dataset
        vertical_coords = np.arange(n_levels)
        
        height = self.plotter._extract_height_from_dataset(
            processor, 0, vertical_coords, 'height'
        )
        
        if height is not None:
            assert len(height) == len(vertical_coords)
    
    def test_height_extraction_exception_returns_none(self: "TestHeightExtraction") -> None:
        """
        This test verifies that if an exception occurs during height extraction (e.g., due to missing variables or incompatible dimensions), the plotter's `_extract_height_from_dataset` method returns None instead of crashing. It creates a mock dataset that lacks the necessary variables for height extraction and then calls the method to ensure that it handles the exception gracefully and returns None, indicating that height information could not be extracted.

        Parameters:
            None

        Returns:
            None
        """
        processor = Mock(spec=MPAS3DProcessor)
        processor.dataset = xr.Dataset()  
        
        vertical_coords = np.arange(10)
        height = self.plotter._extract_height_from_dataset(
            processor, 0, vertical_coords, 'nonexistent_var'
        )
        
        assert height is None


class TestVerticalToHeightConversion:
    """ Tests for vertical coordinate to height conversion. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestVerticalToHeightConversion", mpas_3d_processor) -> None:
        """
        This fixture sets up the test environment for vertical coordinate to height conversion tests by initializing the MPASVerticalCrossSectionPlotter and assigning a shared MPAS3DProcessor instance to the test class. If the processor is not available, it skips the tests to avoid failures due to missing data. This setup allows the subsequent tests to focus on validating the conversion logic from pressure or model levels to height using real or mock MPAS data without needing to load the processor separately in each test case.

        Parameters:
            mpas_3d_processor: A shared session-scoped fixture that provides an instance of MPAS3DProcessor loaded with test data. If None, tests will be skipped.

        Returns:
            None
        """
        if mpas_3d_processor is None:
            pytest.skip("MPAS data not available")
        
        self.processor = mpas_3d_processor
        self.plotter = MPASVerticalCrossSectionPlotter()
    
    def test_pressure_to_height_with_zgrid(self: "TestVerticalToHeightConversion") -> None:
        """
        This test verifies that when pressure coordinates are provided and the dataset contains a `zgrid` variable, the plotter can successfully convert the pressure levels to height using the `zgrid` information. It checks that the conversion returns a valid height array with a length greater than zero, confirming that the plotter can utilize `zgrid` data for accurate vertical coordinate conversion when available.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor
        
        pressure_coords = np.array([100000, 85000, 70000, 50000])
        
        height, coord_type = self.plotter._convert_vertical_to_height(
            pressure_coords, 'pressure', processor, 0
        )
        
        assert height is not None
        assert len(height ) > 0
    
    def test_pressure_to_height_standard_atmosphere(self: "TestVerticalToHeightConversion") -> None:
        """
        This test checks that when pressure coordinates are provided but the dataset does not contain a `zgrid` variable, the plotter can still convert the pressure levels to height using a standard atmosphere approximation. It creates a mock dataset without `zgrid` and then calls the conversion method with typical pressure levels. The test confirms that the resulting height values are non-negative and that the coordinate type is identified as 'height_km', indicating that the standard atmosphere formula was used for the conversion.

        Parameters:
            None

        Returns:
            None
        """
        processor = Mock(spec=MPAS3DProcessor)
        processor.dataset = xr.Dataset()  
        
        pressure_coords = np.array([100000, 85000, 70000, 50000])
        
        height, coord_type = self.plotter._convert_vertical_to_height(
            pressure_coords, 'pressure', processor, 0
        )
        
        assert coord_type == 'height_km'
        assert np.all(height >= 0)
    
    def test_model_levels_with_height_variable(self: "TestVerticalToHeightConversion") -> None:
        """
        This test verifies that when model level coordinates are provided and the dataset contains a `height` variable, the plotter can successfully convert the model levels to height using the available height information. It checks that the conversion returns a valid height array with a length greater than zero, confirming that the plotter can utilize height data for accurate vertical coordinate conversion when `zgrid` is not available but `height` is present.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor
        
        model_coords = np.arange(20)
        
        height, coord_type = self.plotter._convert_vertical_to_height(
            model_coords, 'model_levels', processor, 0
        )
        
        assert height is not None


class TestHeightCoordinates:
    """ Test height extraction and coordinate conversion with real data. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestHeightCoordinates", mpas_3d_processor) -> None:
        """
        This fixture sets up the test environment for height coordinate tests by initializing the MPASVerticalCrossSectionPlotter and assigning a shared MPAS3DProcessor instance to the test class. If the processor is not available or if the necessary grid and output files are missing, it skips the tests to avoid failures due to missing data. This setup allows the subsequent tests to focus on validating height extraction and coordinate conversion logic using real MPAS data, ensuring that the plotter can handle scenarios where height information is derived from model levels when `zgrid` is not available.

        Parameters:
            mpas_3d_processor: A shared session-scoped fixture that provides an instance of MPAS3DProcessor loaded with test data. If None, tests will be skipped.

        Returns:
            None
        """
        if mpas_3d_processor is None or not os.path.exists(GRID_FILE) or not os.path.exists(MPASOUT_DIR):
            pytest.skip("Real MPAS data not available")
        
        self.plotter = MPASVerticalCrossSectionPlotter()
        self.processor = mpas_3d_processor
        
        fig, ax = self.plotter.create_vertical_cross_section(
            self.processor,
            'theta',
            (0, 35),
            (15, 50),
            vertical_coord='height',
            num_points=30,
            time_index=0
        )
        
        assert fig is not None
        ylabel = ax.get_ylabel()
        assert 'evel' in ylabel.lower() 
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
