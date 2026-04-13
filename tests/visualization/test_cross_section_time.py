#!/usr/bin/env python3
"""
MPASdiag Test Suite: Test MPASVerticalCrossSectionPlotter Functionality

This test suite validates the core functionality of the MPASVerticalCrossSectionPlotter class, which is responsible for generating vertical cross-section visualizations from MPAS model output. The tests cover key aspects including plotter initialization with default and custom parameters, great circle path generation between geographic endpoints, automatic contour level generation for various data types, spatial interpolation of irregular grid data onto cross-section paths, and robust input validation for processor objects. Each test function asserts expected behaviors and handles edge cases to ensure the plotter operates correctly under diverse conditions. The suite also includes a comprehensive test runner that executes all tests with clear reporting of successes and failures.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import pytest
import matplotlib
import xarray as xr
matplotlib.use('Agg')
from unittest.mock import Mock

from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
from mpasdiag.processing.processors_3d import MPAS3DProcessor

from tests.visualization.cross_section_test_helpers import (
    check_plotter_initialization, 
    check_great_circle_path,
    check_default_levels, 
    check_interpolation_along_path,
    check_input_validation,
)


def test_vertical_cross_section_plotter_initialization() -> None:
    """
    This test validates the initialization of the MPASVerticalCrossSectionPlotter class, ensuring that default parameters are set correctly and that custom parameters can be applied. The test creates an instance of the plotter and asserts that the default figure size is (10, 12) inches and the default DPI is 100. It also checks that the figure and axes attributes are initialized to None. Then, it creates another instance with custom parameters (figsize=(10, 6) and dpi=150) and verifies that these custom settings are correctly applied.

    Parameters:
        None

    Returns:
        None
    """
    check_plotter_initialization()


def test_great_circle_path_generation() -> None:
    """
    This test verifies the correctness of the great circle path generation method in the MPASVerticalCrossSectionPlotter class. The test defines a start point at (-100.0, 40.0) and an end point at (-90.0, 40.0) with a specified number of points (11) along the path. It asserts that the generated longitude and latitude arrays have the correct length, that the first and last points match the specified start and end coordinates within a reasonable tolerance, and that the distance array is monotonically increasing with the first distance being zero and the last distance being greater than zero. This ensures that the great circle path is generated correctly between the two geographic points.

    Parameters:
        None

    Returns:
        None
    """
    check_great_circle_path()


def test_default_levels_generation() -> None:
    """
    This test validates the default contour level generation method in the MPASVerticalCrossSectionPlotter class for different types of data. The test creates synthetic datasets representing potential temperature (theta), wind speed (uwind), a constant field, and a dataset with NaN values. It asserts that the generated levels for each dataset are appropriate, ensuring that the levels cover the range of the data and that they are not empty. For the constant field, it checks that at least one level is generated, and for the NaN dataset, it verifies that levels are still generated without errors. This ensures that the default level generation method can handle various data scenarios robustly.

    Parameters:
        None

    Returns:
        None
    """
    check_default_levels()


def test_interpolation_along_path() -> None:
    """
    This test verifies the interpolation method in the MPASVerticalCrossSectionPlotter class that interpolates grid data along a specified path. The test creates synthetic grid longitude and latitude arrays, corresponding grid data values, and a set of path longitude and latitude points. It asserts that the interpolated values along the path have the correct length, that they are not all NaN (indicating successful interpolation), and that the method can handle cases where the path points do not exactly match the grid points. This ensures that the interpolation method can effectively interpolate data from an irregular grid onto a defined cross-section path.

    Parameters:
        None

    Returns:
        None
    """
    check_interpolation_along_path()


def test_input_validation() -> None:
    """
    This test validates the input validation logic in the MPASVerticalCrossSectionPlotter class, specifically ensuring that a ValueError is raised when an invalid processor object is passed to the `create_vertical_cross_section` method. The test attempts to call the method with an invalid processor (a string instead of an MPAS3DProcessor instance) and asserts that a ValueError is raised with an appropriate error message indicating the expected type. This ensures that the plotter's input validation correctly identifies and handles invalid processor inputs, preventing downstream errors during cross-section generation.

    Parameters:
        None

    Returns:
        None
    """
    check_input_validation()


class TestTimeStringExtraction:
    """ Tests for time string extraction with fallbacks. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestTimeStringExtraction", 
                     mpas_3d_processor: "MPAS3DProcessor") -> None:
        """
        This setup method initializes the MPASVerticalCrossSectionPlotter and assigns the provided `mpas_3d_processor` fixture to the test instance. If the processor is not available, it skips the tests in this class. This setup ensures that each test has access to a valid processor object for testing time string extraction functionality.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): A fixture providing a processor object for testing.
        
        Returns:
            None
        """
        if mpas_3d_processor is None:
            pytest.skip("MPAS data not available")
            return
        
        self.processor = mpas_3d_processor
        self.plotter = MPASVerticalCrossSectionPlotter()
    
    def test_time_string_with_get_time_info(self: "TestTimeStringExtraction") -> None:
        """
        This test verifies that the `_get_time_string` method correctly extracts time information from the processor's `get_time_info` method when it is available. The test mocks the `get_time_info` method to return a specific time string and asserts that the returned time string from `_get_time_string` includes the expected year, confirming that the method correctly utilizes the processor's time information.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor        
        processor.get_time_info = Mock(return_value="Valid: 2024-09-17 00:00 UTC")        
        time_str = self.plotter._get_time_string(processor, 0)
        
        assert "2024" in time_str
    
    def test_time_string_with_time_coordinate(self: "TestTimeStringExtraction") -> None:
        """
        This test confirms that the `_get_time_string` method can extract time information from the `Time` coordinate of the processor's dataset when the `get_time_info` method is not available. The test ensures that the method correctly falls back to using the `Time` coordinate and that the extracted time string includes the expected year, demonstrating that the method can handle cases where time information is provided through dataset coordinates.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor
        time_str = self.plotter._get_time_string(processor, 0)
        assert len(time_str ) > 0
        assert "2024" in time_str
    
    def test_time_string_fallback_to_index(self: "TestTimeStringExtraction") -> None:
        """
        This test simulates a processor object that lacks both the `get_time_info` method and a `Time` coordinate in its dataset, verifying that the `_get_time_string` method correctly falls back to returning a string representation based on the provided time index. The test creates a mock processor without these attributes and asserts that the returned time string includes the expected index-based representation, ensuring that the method can gracefully handle cases where time information is not available through standard means.

        Parameters:
            None

        Returns:
            None
        """
        processor = Mock(spec=MPAS3DProcessor)
        processor.dataset = xr.Dataset() 
        
        if hasattr(processor, 'get_time_info'):
            delattr(processor, 'get_time_info')
        
        time_str = self.plotter._get_time_string(processor, 5)
        assert "5" in time_str


class TestTimeStringHandling:
    """ Test time string generation. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestTimeStringHandling") -> None:
        """
        This setup method initializes the MPASVerticalCrossSectionPlotter and creates a mock processor object for testing time string handling. The mock processor is configured to simulate scenarios where the `get_time_info` method may not be available or may raise an exception, allowing the tests to verify that the `_get_time_string` method correctly falls back to index-based time string generation without raising errors.

        Parameters:
            None

        Returns:
            None
        """
        self.plotter = MPASVerticalCrossSectionPlotter()
        self.processor = Mock(spec=MPAS3DProcessor)
        
    def test_time_string_without_get_time_info(self: "TestTimeStringHandling") -> None:
        """
        This test verifies that the `_get_time_string` method correctly falls back to an index-based time string representation when the processor object does not have the `get_time_info` method. The test creates a mock processor without this method and asserts that the returned time string includes the expected index-based representation, confirming that the method can handle cases where time information is not available through the standard `get_time_info` method.

        Parameters:
            self (Any): Test case instance with a mocked `processor`.

        Returns:
            None: Asserts that the returned string contains 'Time Index'.
        """
        if hasattr(self.processor, 'get_time_info'):
            delattr(self.processor, 'get_time_info')
        
        time_str = self.plotter._get_time_string(self.processor, 0)
        
        assert 'Time Index' in time_str
    
    def test_time_string_with_exception(self: "TestTimeStringHandling") -> None:
        """
        This test simulates a scenario where the processor's `get_time_info` method raises an exception, verifying that the `_get_time_string` method correctly handles the exception and falls back to returning an index-based time string representation. The test mocks the `get_time_info` method to raise an exception and asserts that the returned time string includes the expected index-based representation, ensuring that the method can gracefully handle errors in time information retrieval without crashing.

        Parameters:
            self (Any): Test case instance with a mocked `processor`.

        Returns:
            None: Asserts that the returned string contains 'Time Index'.
        """
        self.processor.get_time_info = Mock(side_effect=Exception("Time error"))        
        time_str = self.plotter._get_time_string(self.processor, 0)
        assert 'Time Index' in time_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
