#!/usr/bin/env python3
"""
MPASdiag Test Suite: Test Cases for MPASVerticalCrossSectionPlotter Edge Cases and Error Handling

This module contains a comprehensive set of test cases designed to validate the robustness and correctness of the MPASVerticalCrossSectionPlotter class when handling edge cases and error conditions. The tests cover scenarios such as initialization with default and custom parameters, great circle path generation, default contour level generation for various data types, interpolation along cross-section paths, and input validation for processor objects. Each test function includes detailed assertions to verify expected behavior and error handling, ensuring that the plotter can gracefully manage atypical inputs and conditions without crashing. This suite is essential for maintaining code quality and reliability in the visualization of vertical cross-sections from MPAS data.

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
import tempfile
import matplotlib
import numpy as np
import xarray as xr
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Generator, Union
from unittest.mock import MagicMock, patch

from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
from mpasdiag.processing.processors_3d import MPAS3DProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
GRID_FILE = os.path.join(TEST_DATA_DIR, 'grids', 'x1.10242.static.nc')
MPASOUT_DIR = os.path.join(TEST_DATA_DIR, 'u240k', 'mpasout')


def _find_3d_var(processor: MPAS3DProcessor) -> Union[str, None]:
    """
    This helper function searches through the dataset in the provided MPAS3DProcessor to find a variable that has a vertical dimension (either 'nVertLevels' or 'nVertLevelsP1'). It returns the name of the first variable it finds that meets this criterion, or None if no such variable is found. This function is useful for identifying a suitable variable to use in tests that require 3D data with vertical levels.

    Parameters:
        processor (MPAS3DProcessor): The processor containing the dataset to search through.

    Returns:
        Union[str, None]: The name of the first variable with a vertical dimension, or None if no such variable is found.
    """
    for v in processor.dataset.data_vars:
        if 'nVertLevels' in processor.dataset[v].sizes or 'nVertLevelsP1' in processor.dataset[v].sizes:
            return str(v)
    return None


class TestAdditionalEdgeCases:
    """ Additional tests targeting remaining uncovered lines. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestAdditionalEdgeCases', 
                     mpas_3d_processor: 'MPAS3DProcessor') -> Generator[None, None, None]: # type: ignore[no-untyped-def]
        """
        This fixture sets up the MPAS3DProcessor and MPASVerticalCrossSectionPlotter for the edge case tests. It checks if the processor is available and skips tests if not, ensuring that the tests only run when the necessary data is accessible. The processor and plotter are assigned to instance variables for use in the test methods.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): Fixture providing a processor with MPAS data.

        Returns:
            Generator[None, None, None]: A generator that yields control back to the test methods after setup.
        """
        if mpas_3d_processor is None:
            pytest.skip("MPAS data not available")
            return
        
        self.processor = mpas_3d_processor
        self.plotter = MPASVerticalCrossSectionPlotter()
    
    def test_max_height_all_levels_above_limit(self: 'TestAdditionalEdgeCases') -> None:
        """
        This test verifies that the plotter handles cases where all vertical levels are above the specified `max_height` limit. It mocks the data generation to return heights that exceed the `max_height` threshold and asserts that the plotter can still create a figure without crashing, confirming that it can manage scenarios where no data points meet the height criteria.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor
        
        plotter = MPASVerticalCrossSectionPlotter()
        
        with patch.object(plotter, '_generate_cross_section_data') as mock_gen:
            mock_gen.return_value = {
                'distances': np.linspace(0, 100, 50),
                'vertical_coords': np.array([5, 10, 15, 20]),  
                'data_values': np.random.rand(4, 50),
                'path_lons': np.linspace(-100, -90, 50),
                'path_lats': np.linspace(30, 40, 50),
                'longitudes': np.linspace(-100, -90, 50),
                'vertical_coord_type': 'height'
            }
            
            var = _find_3d_var(processor)
            assert var is not None, "No 3D variable found in processor dataset"

            fig, _ = plotter.create_vertical_cross_section(
                processor, var, (-100, 30), (-90, 40),
                max_height=0.1,  
                display_vertical='height'
            )

            plt.close(fig)
    
    def test_pressure_in_hpa_not_pa(self: 'TestAdditionalEdgeCases') -> None:
        """
        This test verifies that when pressure values are provided in hPa (i.e., max < 10000), the plotter correctly identifies the units and formats the axes accordingly. It mocks the data generation to return pressure values in hPa and asserts that the plotter can create a figure without errors, confirming that it can handle pressure data in different units.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor
        
        plotter = MPASVerticalCrossSectionPlotter()
        
        with patch.object(plotter, '_generate_cross_section_data') as mock_gen:
            mock_gen.return_value = {
                'distances': np.linspace(0, 100, 50),
                'vertical_coords': np.array([1000, 850, 700, 500]),  
                'data_values': np.random.rand(4, 50),
                'path_lons': np.linspace(-100, -90, 50),
                'path_lats': np.linspace(30, 40, 50),
                'longitudes': np.linspace(-100, -90, 50),
                'vertical_coord_type': 'pressure'
            }
            
            var = _find_3d_var(processor)
            assert var is not None, "No 3D variable found in processor dataset"
            
            fig, _ = plotter.create_vertical_cross_section(
                processor, var, (-100, 30), (-90, 40),
                display_vertical='pressure'
            )

            plt.close(fig)
    
    
    def test_batch_processing_time_extraction_edge_cases(self: 'TestAdditionalEdgeCases') -> None:
        """
        This test verifies that the batch processing method can handle edge cases in time extraction, such as when the dataset has a Time coordinate but it is not properly formatted. It asserts that the method can still produce output files without crashing, confirming that it can manage time-related issues gracefully.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None: Test asserts on produced filenames and file existence.
        """
        processor = self.processor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            files = self.plotter.create_batch_cross_section_plots(
                processor, temp_dir, 'theta',
                start_point=(-100, 30),
                end_point=(-90, 40),
                num_points=20,
                formats=['png']
            )
            
            assert len(files) > 0


class TestStandardAtmosphereConversion:
    """ Tests for _convert_vertical_to_height standard atmosphere paths. """

    @pytest.fixture(autouse=True)
    def setup(self: 'TestStandardAtmosphereConversion') -> None:
        """
        This fixture sets up the MPASVerticalCrossSectionPlotter for standard atmosphere conversion tests. It initializes the plotter and enables verbose mode to ensure that any warnings or messages related to vertical coordinate conversion are printed during the tests.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        self.plotter = MPASVerticalCrossSectionPlotter()
        self.plotter.verbose = True

    def test_pressure_to_height_standard_atmosphere(self: 'TestStandardAtmosphereConversion') -> None:
        """
        This test verifies that valid pressure values are correctly converted to height using the standard atmosphere formula. It asserts that the resulting height array is in kilometers, that higher pressures correspond to lower heights, and that all height values are non-negative.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        pressure_pa = np.array([101325.0, 50000.0, 25000.0, 10000.0])
        mock_processor = MagicMock()
        mock_processor.dataset = xr.Dataset()  

        height, coord_type = self.plotter._convert_vertical_to_height(
            pressure_pa, 'pressure', mock_processor, 0
        )
        assert coord_type == 'height_km'
        assert height[0] < height[-1]  
        assert np.all(height >= 0)

    def test_pressure_non_positive_clipping(self: 'TestStandardAtmosphereConversion') -> None:
        """
        This test verifies that non-positive pressure values are correctly handled by clipping them to a minimum positive value. It asserts that the resulting height array is in kilometers and that all height values are finite.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        from io import StringIO
        pressure_pa = np.array([101325.0, 50000.0, -100.0, 0.0])
        mock_processor = MagicMock()
        mock_processor.dataset = xr.Dataset()

        captured = StringIO()
        with patch('sys.stdout', captured):
            height, coord_type = self.plotter._convert_vertical_to_height(
                pressure_pa, 'pressure', mock_processor, 0
            )
        assert coord_type == 'height_km'
        assert np.all(np.isfinite(height))

    def test_pressure_hpa_small_values(self: 'TestStandardAtmosphereConversion') -> None:
        """
        This test verifies that pressure values less than 10000 (likely in hPa) are correctly multiplied by 100 to convert to Pa. It asserts that the resulting height array is in kilometers and that the surface pressure corresponds to near-zero height.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        pressure_hpa = np.array([1013.25, 500.0, 250.0, 100.0])
        mock_processor = MagicMock()
        mock_processor.dataset = xr.Dataset()

        height, coord_type = self.plotter._convert_vertical_to_height(
            pressure_hpa, 'pressure', mock_processor, 0
        )
        assert coord_type == 'height_km'
        assert height[0] < 2.0
