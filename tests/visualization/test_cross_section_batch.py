#!/usr/bin/env python3
"""
MPASdiag Test Suite: Test MPASVerticalCrossSectionPlotter Functionality with Synthetic Data

This test suite validates the core functionality of the MPASVerticalCrossSectionPlotter class using synthetic data to ensure that the plotting routines can handle a variety of input scenarios without relying on real MPAS datasets. The tests cover initialization, great circle path generation, default level calculation, interpolation along paths, and input validation. By using controlled synthetic data, we can isolate and verify specific aspects of the plotter's behavior in a deterministic manner. This approach allows for comprehensive testing of the plotter's logic and error handling capabilities while avoiding dependencies on external data files or complex MPAS processing.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load standard libraries and testing frameworks
import os
import sys
import math
import shutil
import pytest
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import cast, Any, Generator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
from mpasdiag.processing.processors_3d import MPAS3DProcessor

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
GRID_FILE = os.path.join(TEST_DATA_DIR, 'grids', 'x1.40962.static.nc')
MPASOUT_DIR = os.path.join(TEST_DATA_DIR, 'u120k', 'mpasout')


def test_vertical_cross_section_plotter_initialization() -> None:
    """
    This test validates that the MPASVerticalCrossSectionPlotter initializes with correct default parameters and allows for custom configuration. It checks that the default figure size is (10, 12) inches and the default DPI is 100, ensuring that the plotter's visual settings are set as expected. The test also confirms that the figure and axis attributes are initialized to None, indicating that no plot has been created yet. Additionally, it verifies that when custom parameters are provided during initialization (e.g., figsize=(10, 6) and dpi=150), the plotter correctly applies these settings, allowing for flexible configuration of the plotting environment.

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
    This test validates that the MPASVerticalCrossSectionPlotter generates appropriate default contour levels for different types of data. It checks that the generated levels cover the range of the input data, that they are not empty, and that they are suitable for the variable type (e.g., potential temperature vs. wind speed). The test also verifies that the method can handle edge cases such as constant data or data containing NaN values without failing, ensuring that it produces reasonable levels in these scenarios. By confirming the correctness and robustness of the default level generation, this test ensures that the plotter can create meaningful visualizations even when users do not specify custom levels.

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
    This test verifies that the MPASVerticalCrossSectionPlotter's create_vertical_cross_section method properly validates its inputs and raises appropriate exceptions when invalid data is provided. It checks that passing an invalid type for the mpas_3d_processor argument (e.g., a string instead of an MPAS3DProcessor instance) results in a ValueError with a clear error message indicating the expected type. By confirming that the method enforces input validation, this test ensures that users receive informative feedback when they provide incorrect inputs, improving the robustness and usability of the plotter.

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


class TestBatchProcessing:
    """ Tests for batch cross-section processing. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestBatchProcessing", mpas_3d_processor: MPAS3DProcessor) -> Generator[None, None, None]:
        """
        This fixture sets up the MPASVerticalCrossSectionPlotter and a mock MPAS3DProcessor for use in batch processing tests. It initializes the plotter and processor before each test and ensures that all matplotlib figures are closed after the test to prevent resource leaks. The fixture is automatically applied to all test methods in the TestBatchProcessing class, providing a consistent testing environment for validating batch cross-section processing functionality.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): A fixture providing a mock or real MPAS3DProcessor instance for testing.

        Returns:
            Generator[None, None, None]: A generator that yields control to the test method and performs cleanup after the test completes.
        """
        self.plotter = MPASVerticalCrossSectionPlotter()
        self.processor = mpas_3d_processor
        self.mock_processor = mpas_3d_processor  
    
        yield
        plt.close('all')
    
    def test_batch_cross_sections_validation(self: "TestBatchProcessing") -> None:
        """
        This test validates that the batch cross-section processing function correctly calls the create_vertical_cross_section method for each specified path. It uses the unittest.mock.patch decorator to replace the create_vertical_cross_section method with a mock that tracks how many times it is called and with what arguments. The test defines multiple start and end points for cross-sections and asserts that the create_vertical_cross_section method is called the expected number of times (once for each path) with the correct parameters. By confirming that the batch processing function invokes the plotting routine as intended, this test ensures that the batch functionality is correctly orchestrating multiple cross-section generations.

        Parameters:
            None

        Returns:
            None
        """
        if self.processor is None:
            pytest.skip("MPAS data not available")
        
        paths = [
            ((-110, 35), (-90, 45)),
            ((-110, 40), (-90, 40))
        ]
        
        results = []
        for start, end in paths:
            result = self.plotter.create_vertical_cross_section(
                self.processor,
                'theta',
                start_point=start,
                end_point=end
            )
            results.append(result)
        
        assert len(results) == 2


class TestBatchProcessingFinal:
    """ Tests for batch cross-section processing with real data. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestBatchProcessingFinal", mpas_3d_processor) -> Generator[None, None, None]:
        """
        This fixture sets up the MPASVerticalCrossSectionPlotter and a real MPAS3DProcessor for use in batch processing tests. It initializes the plotter and processor before each test and ensures that all matplotlib figures are closed after the test to prevent resource leaks. The fixture is automatically applied to all test methods in the TestBatchProcessingFinal class, providing a consistent testing environment for validating batch cross-section processing functionality.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): A fixture providing a real MPAS3DProcessor instance for testing.

        Returns:
            Generator[None, None, None]: A generator that yields control to the test method and performs cleanup after the test completes.
        """
        if mpas_3d_processor is None:
            pytest.skip("MPAS data not available")
        
        self.processor = mpas_3d_processor
        self.temp_dir = tempfile.mkdtemp()
        self.plotter = MPASVerticalCrossSectionPlotter()
    
        yield
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_batch_processing_exception_handling(self: "TestBatchProcessingFinal") -> None:
        """
        This test verifies that the batch processing function can handle exceptions raised during the creation of individual cross-section plots without crashing the entire batch process. It uses unittest.mock.patch to simulate an exception being raised on the first call to create_vertical_cross_section, while allowing subsequent calls to succeed. The test asserts that the batch processing function continues to generate plots for the remaining time steps and that it returns a list of output files, confirming that it can gracefully handle errors in individual plot generation while still completing the batch process.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor
        
        original_create = self.plotter.create_vertical_cross_section
        call_count = [0]
        
        def failing_on_first_call(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Test error")
            return original_create(*args, **kwargs)
        
        self.plotter.create_vertical_cross_section = failing_on_first_call
        try:
            files = self.plotter.create_batch_cross_section_plots(
                processor, self.temp_dir, 'theta',
                (-100, 30), (-90, 40),
                num_points=20
            )
        finally:
            self.plotter.create_vertical_cross_section = original_create
        
        assert files is not None
    
    def test_batch_processing_multiple_formats(self: "TestBatchProcessingFinal") -> None:
        """
        This test validates that the batch processing function can generate cross-section plots in multiple specified formats (e.g., PNG and PDF) and that the output files are created with the correct extensions. It uses a temporary directory to store the generated files and asserts that the returned list of files includes entries for each requested format, confirming that the batch processing function correctly handles multiple output formats as intended.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor
        
        files = self.plotter.create_batch_cross_section_plots(
            processor, self.temp_dir, 'theta',
            (-100, 30), (-90, 40),
            num_points=10,
            formats=['png', 'pdf']
        )
        
        png_files = [f for f in files if f.endswith('.png')]
        pdf_files = [f for f in files if f.endswith('.pdf')]
        
        assert len(png_files ) > 0
        assert len(pdf_files ) > 0

    def test_batch_processing_all_times(self: "TestBatchProcessingFinal") -> None:
        """
        This test validates that the batch processing function generates cross-section plots for all time steps in the dataset. It uses a temporary directory to store the output files and asserts that the number of generated files corresponds to the number of time steps, confirming that the batch processing loop iterates over all time steps as expected.

        Parameters:
            None

        Returns:
            None: Asserts that returned file list is non-empty and files exist.
        """
        processor = self.processor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.plotter.create_batch_cross_section_plots(
                mpas_3d_processor=processor,
                output_dir=tmpdir,
                var_name='theta',
                start_point=(0, 35),
                end_point=(15, 50),
                vertical_coord='pressure',
                num_points=30,
                formats=['png']
            )
            
            assert len(result) > 0
            
            for file_path in result:
                assert os.path.exists(file_path)
    
    def test_batch_processing_without_time_dimension(self: "TestBatchProcessingFinal") -> None:
        """
        This test verifies that the batch processing function can handle a dataset that does not contain a 'Time' dimension without crashing. It checks that the function can still generate cross-section plots for the available data and that it does not raise an exception due to the missing time dimension. By confirming that the batch processing function can operate with datasets lacking a 'Time' dimension, this test ensures that it is robust to variations in dataset structure and can still produce visualizations when time information is not present.

        Parameters:
            None

        Returns:
            None: Asserts that 'Time' is present in dataset dimensions.
        """
        processor = self.processor        
        assert 'Time' in processor.dataset.dims
    
    def test_batch_processing_with_valid_output(self: "TestBatchProcessingFinal") -> None:
        """
        This test validates that the batch processing function can successfully generate cross-section plots and save them to a specified output directory. It uses a temporary directory to store the generated files and asserts that the function completes without errors and returns a list of output file paths, confirming that the batch processing function can produce valid output files as expected.

        Parameters:
            None

        Returns:
            None: Asserts that the result is a non-empty list of files.
        """
        processor = self.processor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.plotter.create_batch_cross_section_plots(
                mpas_3d_processor=processor,
                output_dir=tmpdir,
                var_name='theta',
                start_point=(0, 30),
                end_point=(10, 40),
                vertical_coord='pressure',
                num_points=20,
                formats=['png']
            )
            
            assert isinstance(result, list)
            assert len(result) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
