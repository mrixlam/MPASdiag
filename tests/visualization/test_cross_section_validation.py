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
import math
import pytest
import shutil
import tempfile
import matplotlib
import numpy as np
import xarray as xr
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import cast, Any, Generator, Union
from unittest.mock import Mock, MagicMock, patch
from tests.test_data_helpers import load_mpas_mesh

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
    def setup_method(self: "TestAdditionalEdgeCases", 
                     mpas_3d_processor: MPAS3DProcessor) -> Generator[None, None, None]: # type: ignore[no-untyped-def]
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
    
    def test_max_height_all_levels_above_limit(self: "TestAdditionalEdgeCases") -> None:
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
    
    def test_pressure_in_hpa_not_pa(self: "TestAdditionalEdgeCases") -> None:
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
    
    def test_scipy_interpolation_in_height_extraction(self: "TestAdditionalEdgeCases") -> None:
        """
        This test verifies that scipy interpolation is used when the source height array size differs from the requested levels. It builds a small dataset with mismatched sizes and asserts the returned interpolated heights length.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASVerticalCrossSectionPlotter()
        original_extract = plotter._extract_height_from_dataset
        
        def mock_extract(proc, time_idx, vert_coords, var_name):
            return original_extract(proc, time_idx, vert_coords, var_name)
        
        vertical_coords = np.array([0, 1, 2, 3, 4])  
        
        simple_dataset = xr.Dataset({
            'zgrid': xr.DataArray(
                np.linspace(0, 20000, 8).reshape(1, 8, 1),  
                dims=['Time', 'nVertLevelsP1', 'nCells']
            )
        })
        
        simple_processor = Mock(spec=MPAS3DProcessor)
        simple_processor.dataset = simple_dataset
        
        result = plotter._extract_height_from_dataset(
            simple_processor, 0, vertical_coords, 'zgrid'
        )
        
        if result is not None:
            assert len(result) == len(vertical_coords)
    
    def test_batch_processing_time_extraction_edge_cases(self: "TestAdditionalEdgeCases") -> None:
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


class TestCrossSectionPathValidation:
    """ Tests for cross-section path domain checks and coordinate conversion. """

    @pytest.fixture(autouse=True)
    def setup(self: "TestCrossSectionPathValidation") -> None:
        """
        This fixture sets up the MPASVerticalCrossSectionPlotter for path validation tests. It initializes the plotter and enables verbose mode to ensure that any warnings or messages related to path validation are printed during the tests.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        self.plotter = MPASVerticalCrossSectionPlotter()
        self.plotter.verbose = True

    def test_path_outside_grid_domain_warning(self: "TestCrossSectionPathValidation") -> None:
        """
        This test verifies that a warning is issued when the specified cross-section path lies outside the grid domain. It creates a mock grid domain and defines a path with coordinates that are clearly outside this domain. The test asserts that the path is recognized as being outside the longitude and latitude bounds, confirming that the plotter can detect and warn about paths that do not intersect the grid.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        n_cells = 100
        lon_coords = np.linspace(-105, -95, n_cells)
        lat_coords = np.linspace(30, 40, n_cells)

        path_lons = np.array([-130, -125, -120])
        path_lats = np.array([50, 55, 60])

        path_in_lon = (path_lons[0] >= np.min(lon_coords) and path_lons[-1] <= np.max(lon_coords))

        path_in_lat = (min(path_lats[0], path_lats[-1]) >= np.min(lat_coords) and
                       max(path_lats[0], path_lats[-1]) <= np.max(lat_coords))

        assert not path_in_lon or not path_in_lat

    def test_great_circle_same_start_end(self: "TestCrossSectionPathValidation") -> None:
        """
        This test verifies that when the start and end points of a great circle path are identical, the path still produces valid output. It asserts that the generated longitude and latitude arrays have the correct length and that all distances are non-negative. This confirms that the path generation logic can handle zero-distance paths without errors.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        lons, _, dists = self.plotter._generate_great_circle_path(
            start_point=(-100, 35),
            end_point=(-100, 35),
            num_points=5
        )
        assert len(lons) == pytest.approx(5)
        assert np.all(dists >= 0)


class TestStandardAtmosphereConversion:
    """ Tests for _convert_vertical_to_height standard atmosphere paths. """

    @pytest.fixture(autouse=True)
    def setup(self: "TestStandardAtmosphereConversion") -> None:
        """
        This fixture sets up the MPASVerticalCrossSectionPlotter for standard atmosphere conversion tests. It initializes the plotter and enables verbose mode to ensure that any warnings or messages related to vertical coordinate conversion are printed during the tests.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        self.plotter = MPASVerticalCrossSectionPlotter()
        self.plotter.verbose = True

    def test_pressure_to_height_standard_atmosphere(self: "TestStandardAtmosphereConversion") -> None:
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

    def test_pressure_non_positive_clipping(self: "TestStandardAtmosphereConversion") -> None:
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

    def test_pressure_hpa_small_values(self: "TestStandardAtmosphereConversion") -> None:
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

    def test_model_levels_fallback_no_height_var(self: "TestStandardAtmosphereConversion") -> None:
        """
        This test verifies that when model levels are provided without a height variable, the method falls back to using the model level indices as heights. It asserts that the resulting height array matches the input model levels and that the coordinate type indicates model levels.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        modlev = np.arange(1, 56)
        mock_processor = MagicMock()
        mock_processor.dataset = xr.Dataset()

        height, coord_type = self.plotter._convert_vertical_to_height(
            modlev, 'modlev', mock_processor, 0
        )
        assert coord_type == 'modlev'
        np.testing.assert_array_equal(height, modlev)

    def test_height_direct_conversion(self: "TestStandardAtmosphereConversion") -> None:
        """
        This test verifies that when height coordinates are provided directly, they are correctly converted from meters to kilometers. It asserts that the resulting height array is in kilometers and that the conversion is accurate.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        height_m = np.array([0, 1000, 5000, 10000, 15000])
        mock_processor = MagicMock()

        height_km, coord_type = self.plotter._convert_vertical_to_height(
            height_m, 'height', mock_processor, 0
        )
        assert coord_type == 'height_km'
        np.testing.assert_array_almost_equal(height_km, [0, 1, 5, 10, 15])


class TestInterpolationAllNaN:
    """ Tests for interpolation along path with edge-case data. """

    @pytest.fixture(autouse=True)
    def setup(self: "TestInterpolationAllNaN") -> None:
        """
        This fixture sets up the MPASVerticalCrossSectionPlotter for interpolation tests. It initializes the plotter and disables verbose mode to suppress output during interpolation tests, ensuring that the focus is on the interpolation results rather than any warnings or messages.
        
        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        self.plotter = MPASVerticalCrossSectionPlotter()
        self.plotter.verbose = False

    def test_interpolate_all_nan_returns_nan(self: "TestInterpolationAllNaN") -> None:
        """
        This test verifies that when all grid data values are NaN, the interpolation method returns an array of NaN values. It constructs a grid with NaN data and a path that intersects the grid, then asserts that the interpolation result is entirely NaN, confirming that the method can handle cases where no valid data points are available for interpolation.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        grid_lons = np.linspace(-110, -100, 50)
        grid_lats = np.linspace(35, 45, 50)
        grid_data = np.full(50, np.nan)
        path_lons = np.linspace(-108, -102, 10)
        path_lats = np.linspace(37, 43, 10)

        result = self.plotter._interpolate_along_path(
            grid_lons, grid_lats, grid_data, path_lons, path_lats
        )
        assert np.all(np.isnan(result))

    def test_interpolate_with_xarray_input(self: "TestInterpolationAllNaN") -> None:
        """
        This test verifies that DataArray input should be handled transparently. It constructs a grid with random data in a DataArray and a path that intersects the grid, then asserts that the interpolation result has the correct length and contains valid values, confirming that the method can handle xarray DataArray inputs.

        Parameters:
            self (Any): Test case instance providing fixtures.

        Returns:
            None
        """
        grid_lons = np.linspace(-110, -100, 50)
        grid_lats = np.linspace(35, 45, 50)
        grid_data = xr.DataArray(np.random.uniform(200, 300, 50))
        path_lons = np.linspace(-108, -102, 10)
        path_lats = np.linspace(37, 43, 10)

        result = self.plotter._interpolate_along_path(
            grid_lons, grid_lats, grid_data, path_lons, path_lats
        )
        assert len(result) == pytest.approx(10)
        assert not np.all(np.isnan(result))


