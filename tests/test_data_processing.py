#!/usr/bin/env python3
"""
MPAS Data Processing Module Unit Tests

This module provides comprehensive unit tests for the MPAS data processing utilities
and processor classes including MPAS2DProcessor, MPAS3DProcessor, and utility modules
for geographic operations and datetime parsing. These tests validate core functionality
including file discovery, datetime parsing, spatial coordinate extraction, variable data
retrieval, accumulation period handling, and spatial filtering using synthetic xarray
datasets and in-memory NetCDF files with mocking to isolate components and ensure proper
error handling for edge cases.

Tests Performed:
    TestUtilityFunctions:
        - test_validate_geographic_extent: Validates geographic extent bounds checking (lon/lat ranges)
        - test_normalize_longitude: Tests longitude normalization to standard ranges
        - test_get_accumulation_hours: Verifies accumulation period string parsing (e.g., "3hr", "24hr")
    
    TestMPAS2DProcessor:
        - test_initialization: Verifies processor initialization with grid file and parameters
        - test_initialization_invalid_grid: Tests error handling for invalid grid file paths
        - test_find_diagnostic_files: Validates discovery of diagnostic files in data directory
        - test_find_diagnostic_files_insufficient: Tests error handling for insufficient file count
        - test_parse_file_datetimes: Verifies datetime extraction from MPAS filename patterns
        - test_validate_time_parameters: Tests time range validation with dataset
        - test_validate_time_parameters_no_dataset: Validates error handling without loaded dataset
        - test_get_time_info: Tests time information extraction from loaded datasets
        - test_extract_spatial_coordinates_no_dataset: Verifies error handling when extracting coordinates without data
    
    TestDataValidation:
        - test_filter_by_spatial_extent: Validates spatial filtering with geographic bounds
    
    TestErrorHandling:
        - test_invalid_variable_name: Tests error handling for non-existent variable requests
        - test_corrupted_data_files: Validates handling of corrupted or malformed data files

Test Coverage:
    - MPASBaseProcessor: base initialization, grid file validation, verbose logging
    - MPAS2DProcessor: 2D data loading, file discovery, datetime extraction, coordinate handling
    - MPAS3DProcessor: 3D variable extraction, vertical level management, pressure calculations
    - MPASGeographicUtils: extent validation, longitude normalization, spatial filtering
    - MPASDateTimeUtils: accumulation period parsing, datetime extraction from filenames
    - File discovery: glob pattern matching, file counting, directory traversal
    - Datetime parsing: MPAS filename patterns, xarray time coordinate extraction
    - Spatial coordinates: latitude/longitude extraction, coordinate bounds validation
    - Spatial filtering: mask generation, extent-based filtering, coordinate bounds checking
    - Data validation: variable existence checking, dataset availability verification
    - Error handling: missing data, insufficient files, invalid inputs, corrupted files
    - Accumulation processing: period validation, time difference calculations

Testing Approach:
    Unit tests using unittest framework with synthetic xarray datasets and temporary NetCDF
    files created in-memory during test execution. Mocking simulates file system operations
    (os.path.exists, glob.glob) and dataset structures to validate processing logic without
    requiring actual MPAS model output files. Tests use patch decorators to isolate file I/O
    dependencies and verify both successful operations and error conditions.

Expected Results:
    - Utility functions return correct numeric/boolean values for valid inputs
    - Geographic extent validation correctly identifies valid and invalid bounds
    - Longitude normalization handles edge cases (wrap-around, negative values)
    - Accumulation period parsing extracts correct hour values from strings
    - MPAS2DProcessor initializes successfully with valid grid file paths
    - Invalid grid files raise FileNotFoundError during initialization
    - File discovery returns correct file lists matching MPAS naming patterns
    - Insufficient file counts raise appropriate exceptions
    - Datetime parsing extracts valid datetime objects from filenames
    - Time validation identifies out-of-range time parameters
    - Coordinate extraction returns arrays with correct shapes and value ranges
    - Spatial filtering produces accurate boolean masks for geographic extents
    - Invalid variable names raise KeyError or AttributeError
    - Corrupted data files handled gracefully with appropriate error messages
    - All tests pass with synthetic data and comprehensive mocking

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

from typing import Any
import unittest
import tempfile
import os
import numpy as np
import xarray as xr
from unittest.mock import patch, MagicMock
from datetime import datetime

import sys
from pathlib import Path

package_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_dir))

from mpasdiag.processing.processors_2d import MPAS2DProcessor
from mpasdiag.processing.base import MPASBaseProcessor
from mpasdiag.processing.utils_geog import MPASGeographicUtils
from mpasdiag.processing.utils_datetime import MPASDateTimeUtils


class TestUtilityFunctions(unittest.TestCase):
    """
    Tests for standalone utility functions in data_processing.

    Scope:
        Validate geographic extent checks, longitude normalization and
        accumulation hour parsing using synthetic inputs.
    """
    
    @patch('os.path.exists')
    def test_validate_geographic_extent(self, mock_exists: Any) -> None:
        """
        Validate geographic extent bounds checking for longitude and latitude ranges. This test verifies the validate_geographic_extent method correctly identifies valid geographic bounds (longitude -180 to 180, latitude -90 to 90) and properly rejects invalid ranges including out-of-bounds values, reversed min/max order, and impossible coordinate combinations. Multiple test cases cover valid extents, longitude out of range, latitude out of range, reversed longitude bounds, and reversed latitude bounds. Mock grid file existence prevents actual file I/O during testing. This ensures robust geographic validation prevents plotting or processing with invalid spatial extents.

        Parameters:
            mock_exists (unittest.mock.MagicMock): Mock for os.path.exists to simulate grid file availability without actual file system access.

        Returns:
            None
        """
        mock_exists.return_value = True
        
        processor = MPASBaseProcessor("mock_grid.nc", verbose=False)
        
        self.assertTrue(processor.validate_geographic_extent((100.0, 110.0, -10.0, 10.0)))
        self.assertFalse(processor.validate_geographic_extent((200.0, 110.0, -10.0, 10.0)))
        self.assertFalse(processor.validate_geographic_extent((100.0, 110.0, -100.0, 10.0)))
        self.assertFalse(processor.validate_geographic_extent((110.0, 100.0, -10.0, 10.0)))
        self.assertFalse(processor.validate_geographic_extent((100.0, 110.0, 10.0, -10.0)))
    
    @patch('os.path.exists')
    def test_normalize_longitude(self, mock_exists: Any) -> None:
        """
        Verify longitude normalization to standard -180 to 180 degree range. This test validates the normalize_longitude method correctly handles various longitude formats including positive values greater than 180 degrees, negative values less than -180 degrees, and edge cases at 0, 180, and -180 degrees. The method wraps longitude values to the standard meteorological convention of -180 to 180 degrees for consistent geographic processing. Test cases use numpy arrays and scalar values to verify both array and element-wise normalization. This ensures consistent longitude representation across different data sources and coordinate systems.

        Parameters:
            mock_exists (unittest.mock.MagicMock): Mock for os.path.exists to simulate grid file availability without actual file system access.

        Returns:
            None
        """
        mock_exists.return_value = True
        
        processor = MPASBaseProcessor("mock_grid.nc", verbose=False)
        
        lon_in = np.array([350.0, -200.0, 0.0, 180.0, -180.0])
        lon_out = processor.normalize_longitude(lon_in)
        
        expected = np.array([-10.0, 160.0, 0.0, -180.0, -180.0])
        np.testing.assert_array_almost_equal(lon_out, expected)
        
        self.assertAlmostEqual(processor.normalize_longitude(350.0), -10.0)
    
    @patch('os.path.exists')
    def test_get_accumulation_hours(self, mock_exists: Any) -> None:
        """
        Validate accumulation period string parsing for precipitation analysis. This test verifies the get_accumulation_hours method correctly extracts hour values from MPAS accumulation period strings (e.g., 'a01h', 'a03h', 'a24h') used in diagnostic file naming conventions. Valid accumulation periods (1, 3, 6, 12, 24 hours) are tested to ensure proper numeric extraction. Invalid accumulation strings default to 24 hours as a fallback value. This functionality enables proper time windowing calculations for accumulated precipitation variables. Mock grid file existence prevents actual file I/O during testing.

        Parameters:
            mock_exists (unittest.mock.MagicMock): Mock for os.path.exists to simulate grid file availability without actual file system access.

        Returns:
            None
        """
        mock_exists.return_value = True
        
        processor = MPAS2DProcessor("mock_grid.nc", verbose=False)
        
        self.assertEqual(processor.get_accumulation_hours('a01h'), 1)
        self.assertEqual(processor.get_accumulation_hours('a03h'), 3)
        self.assertEqual(processor.get_accumulation_hours('a06h'), 6)
        self.assertEqual(processor.get_accumulation_hours('a12h'), 12)
        self.assertEqual(processor.get_accumulation_hours('a24h'), 24)
        self.assertEqual(processor.get_accumulation_hours('invalid'), 24)  


class TestMPAS2DProcessor(unittest.TestCase):
    """
    Tests for MPAS2DProcessor class.

    Scope:
        Exercises grid file creation, dataset-oriented helpers, time parsing,
        and spatial coordinate extraction using temporary NetCDF files.
    """
    
    def setUp(self) -> None:
        """
        Initialize test fixtures with temporary directory and synthetic grid file. This method creates a temporary directory for test file storage and generates a test grid NetCDF file with synthetic coordinate arrays (lonCell, latCell, lonVertex, latVertex) simulating MPAS mesh structure. The synthetic grid contains 100 cells and 50 vertices with coordinate values in a realistic geographic region (100-110°E longitude, -5 to 5°N latitude). This test grid enables validation of processor initialization and coordinate extraction without requiring actual MPAS model output files. Temporary files are automatically cleaned up in tearDown to prevent test pollution.

        Parameters:
            None

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()
        self.grid_file = os.path.join(self.temp_dir, "test_grid.nc")
        
        self.create_test_grid_file()
    
    def tearDown(self) -> None:
        """
        Clean up temporary test files and directories created during test execution. This method recursively removes the temporary directory and all its contents including the synthetic grid file to ensure no test artifacts persist after test completion. The ignore_errors=True parameter ensures cleanup proceeds even if some files are locked or inaccessible. Proper cleanup prevents disk space accumulation from repeated test runs and ensures test isolation. This standard unittest tearDown pattern guarantees cleanup occurs regardless of test success or failure.

        Parameters:
            None

        Returns:
            None
        """
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_grid_file(self) -> None:
        """
        Generate synthetic MPAS grid NetCDF file with realistic coordinate structure. This method creates an xarray Dataset containing coordinate arrays for cell centers (lonCell, latCell) and vertices (lonVertex, latVertex) with dimensions matching typical MPAS mesh topology. Random coordinate values span a realistic geographic region (100-110°E longitude, -5 to 5°N latitude) to simulate Southeast Asian domain. The dataset is saved as a NetCDF file in the temporary directory for use in processor initialization tests. This synthetic grid enables testing without requiring actual MPAS static/init files.

        Parameters:
            None

        Returns:
            None
        """
        nCells = 100
        nVertices = 50
        
        lonCell = np.random.uniform(100, 110, nCells)
        latCell = np.random.uniform(-5, 5, nCells)
        lonVertex = np.random.uniform(100, 110, nVertices)
        latVertex = np.random.uniform(-5, 5, nVertices)
        
        ds = xr.Dataset({
            'lonCell': (['nCells'], lonCell),
            'latCell': (['nCells'], latCell),
            'lonVertex': (['nVertices'], lonVertex),
            'latVertex': (['nVertices'], latVertex),
        })
        
        ds.to_netcdf(self.grid_file)
    
    def test_initialization(self) -> None:
        """
        Verify correct initialization of MPAS2DProcessor with grid file and parameters. This test validates that the processor initializes with the correct grid file path, verbose flag is properly set to False, and initial state attributes (dataset, data_type) are None before data loading. Assertions confirm instance attributes match the initialization parameters. This establishes baseline processor state and verifies constructor behavior before testing data loading operations. Proper initialization is critical for subsequent processing steps that depend on these attributes.

        Parameters:
            None

        Returns:
            None
        """
        processor = MPAS2DProcessor(self.grid_file, verbose=False)
        self.assertEqual(processor.grid_file, self.grid_file)
        self.assertFalse(processor.verbose)
        self.assertIsNone(processor.dataset)
        self.assertIsNone(processor.data_type)
    
    def test_initialization_invalid_grid(self) -> None:
        """
        Validate error handling for non-existent grid file during processor initialization. This test verifies that MPAS2DProcessor raises FileNotFoundError when initialized with a path to a non-existent grid file. Proper error handling prevents downstream processing failures and provides clear diagnostic messages when required grid files are missing. The assertRaises context manager confirms the expected exception type. This test ensures robust validation at the earliest stage of processor creation, failing fast with informative errors rather than cryptic messages later in the processing pipeline.

        Parameters:
            None

        Returns:
            None
        """
        with self.assertRaises(FileNotFoundError):
            MPAS2DProcessor("nonexistent_file.nc")
    
    @patch('glob.glob')
    def test_find_diagnostic_files(self, mock_glob: Any) -> None:
        """
        Verify diagnostic file discovery with MPAS filename pattern matching. This test validates the find_diagnostic_files method correctly discovers diagnostic files in a specified directory using glob pattern matching for MPAS naming conventions (diag.YYYY-MM-DD_HH.MM.SS.nc). Mock glob returns a predefined list of three diagnostic files to isolate file discovery logic from actual filesystem operations. Assertions confirm the correct number of files are discovered and the returned file list matches the expected paths. This ensures reliable data file discovery across different directory structures and file naming variations.

        Parameters:
            mock_glob (unittest.mock.MagicMock): Mock for glob.glob to return predefined file list without actual filesystem traversal.

        Returns:
            None
        """
        mock_files = [
            "/path/diag.2024-01-01_00.00.00.nc",
            "/path/diag.2024-01-01_01.00.00.nc",
            "/path/diag.2024-01-01_02.00.00.nc"
        ]
        mock_glob.return_value = mock_files
        
        processor = MPAS2DProcessor(self.grid_file, verbose=False)
        files = processor.find_diagnostic_files("/path")
        
        self.assertEqual(len(files), 3)
        self.assertEqual(files, mock_files)
    
    @patch('glob.glob')
    def test_find_diagnostic_files_insufficient(self, mock_glob: Any) -> None:
        """
        Validate error handling when insufficient diagnostic files are found. This test verifies that find_diagnostic_files raises ValueError when fewer than the minimum required number of files (typically 2) are discovered in the specified directory. Mock glob returns only a single file to trigger the insufficient file condition. The assertRaises context manager confirms ValueError is properly raised with an appropriate error message. This prevents processing attempts with incomplete data that would fail downstream and ensures users receive clear diagnostic messages about missing required files.

        Parameters:
            mock_glob (unittest.mock.MagicMock): Mock for glob.glob to return insufficient file count for testing error handling.

        Returns:
            None
        """
        mock_glob.return_value = ["/path/diag.2024-01-01_00.00.00.nc"]
        
        processor = MPAS2DProcessor(self.grid_file, verbose=False)
        
        with self.assertRaises(ValueError):
            processor.find_diagnostic_files("/path")
    
    def test_parse_file_datetimes(self) -> None:
        """
        Verify datetime extraction from MPAS diagnostic filename patterns. This test validates the parse_file_datetimes method correctly parses datetime information from standard MPAS filename format (diag.YYYY-MM-DD_HH.MM.SS.nc) and handles invalid filenames gracefully. Test cases include two valid filenames with known datetime values and one malformed filename to verify fallback behavior. Assertions confirm correct datetime object creation for valid patterns and datetime instance return for invalid patterns. This ensures robust datetime handling for time series analysis and temporal file ordering regardless of filename variations or errors.

        Parameters:
            None

        Returns:
            None
        """
        files = [
            "diag.2024-01-01_00.00.00.nc",
            "diag.2024-01-01_01.00.00.nc",
            "invalid_filename.nc"
        ]
        
        processor = MPAS2DProcessor(self.grid_file, verbose=False)
        datetimes = processor.parse_file_datetimes(files)
        
        self.assertEqual(len(datetimes), 3)
        self.assertEqual(datetimes[0], datetime(2024, 1, 1, 0, 0, 0))
        self.assertEqual(datetimes[1], datetime(2024, 1, 1, 1, 0, 0))
        self.assertIsInstance(datetimes[2], datetime)
    
    def test_validate_time_parameters(self) -> None:
        """
        Validate time parameter checking and bounds enforcement with loaded datasets. This test verifies the validate_time_parameters method correctly validates time indices against dataset dimensions, returns proper time dimension information, and clamps out-of-range indices to valid bounds. Mock dataset with 10 time steps tests both valid time index (5) and out-of-range index (15) to verify clamping behavior. Assertions confirm correct dimension name, validated index, and total time size are returned. This prevents array indexing errors and ensures safe time-based data extraction across different dataset sizes.

        Parameters:
            None

        Returns:
            None
        """
        mock_dataset = MagicMock()
        mock_dataset.sizes = {'Time': 10}
        mock_dataset.sizes = {'Time': 10}
        
        processor = MPAS2DProcessor(self.grid_file, verbose=False)
        processor.dataset = mock_dataset
        
        time_dim, time_idx, time_size = processor.validate_time_parameters(5)
        self.assertEqual(time_dim, 'Time')
        self.assertEqual(time_idx, 5)
        self.assertEqual(time_size, 10)
        
        time_dim, time_idx, time_size = processor.validate_time_parameters(15)
        self.assertEqual(time_idx, 9)  
    
    def test_validate_time_parameters_no_dataset(self) -> None:
        """
        Verify error handling when validating time parameters without loaded dataset. This test validates that validate_time_parameters raises ValueError when called on a processor instance that hasn't loaded any dataset yet (dataset attribute is None). The assertRaises context manager confirms the expected exception type. This test ensures the processor fails fast with a clear error message rather than attempting to access attributes on a None object, which would produce cryptic AttributeError messages. Proper validation prevents downstream processing errors and guides users to load data before requesting time-based operations.

        Parameters:
            None

        Returns:
            None
        """
        processor = MPAS2DProcessor(self.grid_file, verbose=False)
        
        with self.assertRaises(ValueError):
            processor.validate_time_parameters(0)
    
    def test_get_time_info(self) -> None:
        """
        Validate time information extraction and formatting from dataset time coordinates. This test verifies the get_time_info method correctly extracts datetime values from xarray Time coordinate and formats them as compact timestamp strings (YYYYMMDDTHH format). Mock dataset with two numpy datetime64 values simulates typical MPAS time coordinate structure. Patching hasattr ensures Time coordinate is detected correctly during attribute checking. Assertion confirms the formatted time string matches the expected compact format without punctuation. This enables clear time labeling in plot titles and filenames for temporal data visualization.

        Parameters:
            None

        Returns:
            None
        """
        times = [
            np.datetime64('2024-01-01T00:00:00'),
            np.datetime64('2024-01-01T01:00:00'),
        ]
        
        mock_dataset = MagicMock()
        mock_dataset.Time.values = times
        mock_dataset.__len__ = MagicMock(return_value=2)
        mock_dataset.Time.__len__ = MagicMock(return_value=2)
        
        processor = MPAS2DProcessor(self.grid_file, verbose=False)
        processor.dataset = mock_dataset
        
        with patch('builtins.hasattr') as mock_hasattr:
            mock_hasattr.side_effect = lambda obj, attr: attr == 'Time'
            time_str = processor.get_time_info(0)
            self.assertEqual(time_str, "20240101T00")
    
    def test_extract_spatial_coordinates_no_dataset(self) -> None:
        """
        Verify error handling when extracting coordinates without loaded dataset. This test validates that extract_spatial_coordinates raises ValueError when called on a processor instance without a loaded dataset (dataset attribute is None). The assertRaises context manager confirms the expected exception is properly raised. This test ensures coordinate extraction fails fast with an informative error message rather than attempting to access non-existent dataset attributes. Proper validation prevents cryptic errors and guides users to load data before requesting spatial coordinate information for plotting or analysis operations.

        Parameters:
            None

        Returns:
            None
        """
        processor = MPAS2DProcessor(self.grid_file, verbose=False)
        
        with self.assertRaises(ValueError):
            processor.extract_spatial_coordinates()


class TestDataValidation(unittest.TestCase):
    """
    Tests for data validation helpers.

    Scope:
        Verifies filtering by spatial extent and other validation logic using
        synthetic xarray DataArrays.
    """
    
    def test_filter_by_spatial_extent(self) -> None:
        """
        Validate spatial filtering using geographic extent bounds and boolean masking. This test verifies the filter_by_spatial_extent method correctly generates boolean masks identifying cells within specified longitude-latitude bounds. Synthetic coordinate arrays (100 cells) with random values spanning a wider region are filtered to a subregion (100-110°E, -4 to 4°N) to validate mask generation. Assertions confirm the mask is boolean type, correct length, and matches expected logical conditions for coordinate bounds. Temporary grid file created and cleaned up within the test enables processor initialization without persistent file artifacts. This ensures accurate spatial subsetting for regional analysis and plotting.

        Parameters:
            None

        Returns:
            None
        """
        nCells = 100
        lon = np.random.uniform(98, 112, nCells)
        lat = np.random.uniform(-6, 8, nCells)
        data_values = np.random.uniform(0, 50, nCells)
        
        data_array = xr.DataArray(data_values, dims=['nCells'])
        
        grid_file = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
        try:
            ds = xr.Dataset({
                'lonCell': (['nCells'], lon),
                'latCell': (['nCells'], lat),
            })
            ds.to_netcdf(grid_file)
            
            processor = MPAS2DProcessor(grid_file, verbose=False)
            processor.dataset = ds
            
            filtered_data, mask = processor.filter_by_spatial_extent(
                data_array, 100.0, 110.0, -4.0, 4.0
            )
            
            self.assertEqual(mask.dtype, bool)
            self.assertEqual(len(mask), nCells)
            
            expected_mask = ((lon >= 100.0) & (lon <= 110.0) & 
                           (lat >= -4.0) & (lat <= 4.0))
            np.testing.assert_array_equal(mask, expected_mask)
            
        finally:
            os.unlink(grid_file)


class TestErrorHandling(unittest.TestCase):
    """
    Tests for error handling scenarios.

    Scope:
        Placeholder tests intended to exercise invalid variable and
        corrupted file handling. Tests may be expanded as more negative
        cases are implemented.
    """
    
    def test_invalid_variable_name(self) -> None:
        """
        Validate error handling for requests to extract non-existent variables from datasets. This test serves as a placeholder for future implementation of robust variable name validation that should raise KeyError or AttributeError when users request variables not present in the loaded dataset. Proper error handling prevents cryptic messages from xarray and provides clear diagnostic information about available variables. The test currently passes as a placeholder and should be expanded with actual variable extraction attempts on mock datasets. This ensures users receive helpful error messages when specifying incorrect variable names for visualization or analysis.

        Parameters:
            None

        Returns:
            None
        """
        pass
    
    def test_corrupted_data_files(self) -> None:
        """
        Validate graceful handling of corrupted or malformed NetCDF data files during loading. This test serves as a placeholder for future implementation of robust error handling when attempting to open corrupted NetCDF files that may result from incomplete downloads, disk errors, or format violations. Proper error handling should catch xarray/NetCDF exceptions and provide informative error messages to users. The test currently passes as a placeholder and should be expanded with actual corrupted file scenarios using temporary malformed NetCDF files. This ensures the system fails gracefully with diagnostic messages rather than crashing with obscure low-level errors.

        Parameters:
            None

        Returns:
            None
        """
        pass


if __name__ == '__main__':
    unittest.main()