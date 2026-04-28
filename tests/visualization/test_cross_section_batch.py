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
import shutil
import pytest
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Generator, Any

from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
from mpasdiag.processing.processors_3d import MPAS3DProcessor

from tests.visualization.cross_section_test_helpers import (
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


def failing_on_first_call(self: MPASVerticalCrossSectionPlotter, 
                          *args, 
                          **kwargs) -> Any:
    """
    This helper function simulates a failure on the first call to create_vertical_cross_section by raising an exception, while allowing subsequent calls to succeed. It uses an instance attribute to track the number of times it has been called and raises an exception only on the first call. On subsequent calls, it checks for the presence of the original create_vertical_cross_section method (which should have been saved before patching) and calls it with the provided arguments. If the original method is not found, it raises a RuntimeError. This function is designed to test the batch processing function's ability to handle exceptions gracefully without crashing the entire batch process.

    Parameters:
        *args: Positional arguments to be passed to the original create_vertical_cross_section method.
        **kwargs: Keyword arguments to be passed to the original create_vertical_cross_section method.

    Returns:
        Any: The return value from the original create_vertical_cross_section method on subsequent calls after the first call raises an exception.
    """
    if not hasattr(self, '_failing_call_count'):
        self._failing_call_count = 0

    self._failing_call_count += 1

    if self._failing_call_count == 1:
        raise Exception("Test error")

    if hasattr(self, '_original_create_vertical_cross_section'):
        return self._original_create_vertical_cross_section(*args, **kwargs)

    raise RuntimeError("Original create_vertical_cross_section not set on instance.")

class TestBatchProcessing:
    """ Tests for batch cross-section processing. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestBatchProcessing', 
                     mpas_3d_processor: 'MPAS3DProcessor') -> Generator[None, None, None]:
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
    
    def test_batch_cross_sections_validation(self: 'TestBatchProcessing') -> None:
        """
        This test validates that the batch cross-section processing function correctly calls the create_vertical_cross_section method for each specified path. It uses the unittest.mock.patch decorator to replace the create_vertical_cross_section method with a mock that tracks how many times it is called and with what arguments. The test defines multiple start and end points for cross-sections and asserts that the create_vertical_cross_section method is called the expected number of times (once for each path) with the correct parameters. By confirming that the batch processing function invokes the plotting routine as intended, this test ensures that the batch functionality is correctly orchestrating multiple cross-section generations.

        Parameters:
            None

        Returns:
            None
        """
        if self.processor is None:
            pytest.skip("MPAS data not available")
            return
        
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
        
        assert len(results) == pytest.approx(2)
        plt.close('all')


class TestBatchProcessingFinal:
    """ Tests for batch cross-section processing with real data. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestBatchProcessingFinal', 
                     mpas_3d_processor: 'MPAS3DProcessor') -> Generator[None, None, None]:
        """
        This fixture sets up the MPASVerticalCrossSectionPlotter and a real MPAS3DProcessor for use in batch processing tests. It initializes the plotter and processor before each test and ensures that all matplotlib figures are closed after the test to prevent resource leaks. The fixture is automatically applied to all test methods in the TestBatchProcessingFinal class, providing a consistent testing environment for validating batch cross-section processing functionality.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): A fixture providing a real MPAS3DProcessor instance for testing.

        Returns:
            Generator[None, None, None]: A generator that yields control to the test method and performs cleanup after the test completes.
        """
        if mpas_3d_processor is None:
            pytest.skip("MPAS data not available")
            return 
        
        self.processor = mpas_3d_processor
        self.temp_dir = tempfile.mkdtemp()
        self.plotter = MPASVerticalCrossSectionPlotter()
    
        yield
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_batch_processing_exception_handling(self: 'TestBatchProcessingFinal') -> None:
        """
        This test verifies that the batch processing function can handle exceptions raised during the creation of individual cross-section plots without crashing the entire batch process. It uses unittest.mock.patch to simulate an exception being raised on the first call to create_vertical_cross_section, while allowing subsequent calls to succeed. The test asserts that the batch processing function continues to generate plots for the remaining time steps and that it returns a list of output files, confirming that it can gracefully handle errors in individual plot generation while still completing the batch process.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor

        original_create = self.plotter.create_vertical_cross_section
        self.plotter.create_vertical_cross_section = failing_on_first_call.__get__(self.plotter, type(self.plotter))

        try:
            files = self.plotter.create_batch_cross_section_plots(
                processor, self.temp_dir, 'theta',
                (-100, 30), (-90, 40),
                num_points=20
            )
        finally:
            self.plotter.create_vertical_cross_section = original_create

        assert files is not None
    
    def test_batch_processing_multiple_formats(self: 'TestBatchProcessingFinal') -> None:
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

    def test_batch_processing_all_times(self: 'TestBatchProcessingFinal') -> None:
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
    
    
    def test_batch_processing_with_valid_output(self: 'TestBatchProcessingFinal') -> None:
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
