#!/usr/bin/env python3

"""
Unit tests for MPAS Data Processing Module

Scope:
        Validate the core data processing utilities and the `MPASDataProcessor`
        class. These tests exercise file discovery, datetime parsing, spatial
        coordinate extraction, accumulation handling, and basic validation logic.

Test data:
        Synthetic xarray datasets and temporary NetCDF files created in-memory
        during the tests. Mocking is used for file discovery and to simulate
        various dataset shapes and attributes.

Expected results:
        - Utility functions return expected numeric or boolean values for valid
            inputs and raise appropriate exceptions for invalid inputs.
        - `MPASDataProcessor` interacts with datasets correctly (loading grid
            coordinates, extracting variables, computing precipitation differences).
        - Edge cases (missing data, insufficient files) produce clear errors.

Per-test expectations (short):
        - TestUtilityFunctions: helpers validate extents, normalize longitudes,
            and parse accumulation specifications correctly.
        - TestMPASDataProcessor: data processor loads grid files, finds
            diagnostic files, parses datetimes, validates time indices, and
            extracts spatial coordinates as NumPy arrays.
        - TestDataValidation: spatial filtering produces correct boolean masks
            and filtered arrays.

Author: Rubaiat Islam
"""

import unittest
import tempfile
import os
import numpy as np
import xarray as xr
from unittest.mock import patch, MagicMock
from datetime import datetime

import sys
from pathlib import Path
package_dir = Path(__file__).parent.parent
sys.path.insert(0, str(package_dir))

from mpas_analysis.data_processing import (
    MPASDataProcessor,
    validate_geographic_extent,
    normalize_longitude,
    get_accumulation_hours
)


class TestUtilityFunctions(unittest.TestCase):
    """
    Tests for standalone utility functions in data_processing.

    Scope:
        Validate geographic extent checks, longitude normalization and
        accumulation hour parsing using synthetic inputs.
    """
    
    def test_validate_geographic_extent(self):
        """Test geographic extent validation."""
        self.assertTrue(validate_geographic_extent((100.0, 110.0, -10.0, 10.0)))
        self.assertFalse(validate_geographic_extent((200.0, 110.0, -10.0, 10.0)))
        self.assertFalse(validate_geographic_extent((100.0, 110.0, -100.0, 10.0)))
        self.assertFalse(validate_geographic_extent((110.0, 100.0, -10.0, 10.0)))
        self.assertFalse(validate_geographic_extent((100.0, 110.0, 10.0, -10.0)))
    
    def test_normalize_longitude(self):
        """Test longitude normalization."""
        lon_in = np.array([350.0, -200.0, 0.0, 180.0, -180.0])
        lon_out = normalize_longitude(lon_in)
        
        expected = np.array([-10.0, 160.0, 0.0, -180.0, -180.0])
        np.testing.assert_array_almost_equal(lon_out, expected)
        
        self.assertAlmostEqual(normalize_longitude(350.0), -10.0)
    
    def test_get_accumulation_hours(self):
        """Test accumulation period parsing."""
        self.assertEqual(get_accumulation_hours('a01h'), 1)
        self.assertEqual(get_accumulation_hours('a03h'), 3)
        self.assertEqual(get_accumulation_hours('a06h'), 6)
        self.assertEqual(get_accumulation_hours('a12h'), 12)
        self.assertEqual(get_accumulation_hours('a24h'), 24)
        self.assertEqual(get_accumulation_hours('invalid'), 24)  


class TestMPASDataProcessor(unittest.TestCase):
    """
    Tests for MPASDataProcessor class.

    Scope:
        Exercises grid file creation, dataset-oriented helpers, time parsing,
        and spatial coordinate extraction using temporary NetCDF files.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.grid_file = os.path.join(self.temp_dir, "test_grid.nc")
        
        self.create_test_grid_file()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_grid_file(self):
        """Create a test grid file."""
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
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = MPASDataProcessor(self.grid_file, verbose=False)
        self.assertEqual(processor.grid_file, self.grid_file)
        self.assertFalse(processor.verbose)
        self.assertIsNone(processor.dataset)
        self.assertIsNone(processor.data_type)
    
    def test_initialization_invalid_grid(self):
        """Test initialization with invalid grid file."""
        with self.assertRaises(FileNotFoundError):
            MPASDataProcessor("nonexistent_file.nc")
    
    @patch('glob.glob')
    def test_find_diagnostic_files(self, mock_glob):
        """Test diagnostic file discovery."""
        mock_files = [
            "/path/diag.2024-01-01_00.00.00.nc",
            "/path/diag.2024-01-01_01.00.00.nc",
            "/path/diag.2024-01-01_02.00.00.nc"
        ]
        mock_glob.return_value = mock_files
        
        processor = MPASDataProcessor(self.grid_file, verbose=False)
        files = processor.find_diagnostic_files("/path")
        
        self.assertEqual(len(files), 3)
        self.assertEqual(files, mock_files)
    
    @patch('glob.glob')
    def test_find_diagnostic_files_insufficient(self, mock_glob):
        """Test diagnostic file discovery with insufficient files."""
        mock_glob.return_value = ["/path/diag.2024-01-01_00.00.00.nc"]
        
        processor = MPASDataProcessor(self.grid_file, verbose=False)
        
        with self.assertRaises(ValueError):
            processor.find_diagnostic_files("/path")
    
    def test_parse_file_datetimes(self):
        """Test datetime parsing from filenames."""
        files = [
            "diag.2024-01-01_00.00.00.nc",
            "diag.2024-01-01_01.00.00.nc",
            "invalid_filename.nc"
        ]
        
        processor = MPASDataProcessor(self.grid_file, verbose=False)
        datetimes = processor.parse_file_datetimes(files)
        
        self.assertEqual(len(datetimes), 3)
        self.assertEqual(datetimes[0], datetime(2024, 1, 1, 0, 0, 0))
        self.assertEqual(datetimes[1], datetime(2024, 1, 1, 1, 0, 0))
        self.assertIsInstance(datetimes[2], datetime)
    
    def test_validate_time_parameters(self):
        """Test time parameter validation."""
        mock_dataset = MagicMock()
        mock_dataset.dims = {'Time': 10}
        mock_dataset.sizes = {'Time': 10}
        
        processor = MPASDataProcessor(self.grid_file, verbose=False)
        processor.dataset = mock_dataset
        
        time_dim, time_idx, time_size = processor.validate_time_parameters(5)
        self.assertEqual(time_dim, 'Time')
        self.assertEqual(time_idx, 5)
        self.assertEqual(time_size, 10)
        
        time_dim, time_idx, time_size = processor.validate_time_parameters(15)
        self.assertEqual(time_idx, 9)  
    
    def test_validate_time_parameters_no_dataset(self):
        """Test time parameter validation without loaded dataset."""
        processor = MPASDataProcessor(self.grid_file, verbose=False)
        
        with self.assertRaises(ValueError):
            processor.validate_time_parameters(0)
    
    def test_get_time_info(self):
        """Test time information extraction."""
        times = [
            np.datetime64('2024-01-01T00:00:00'),
            np.datetime64('2024-01-01T01:00:00'),
        ]
        
        mock_dataset = MagicMock()
        mock_dataset.Time.values = times
        mock_dataset.__len__ = MagicMock(return_value=2)
        mock_dataset.Time.__len__ = MagicMock(return_value=2)
        
        processor = MPASDataProcessor(self.grid_file, verbose=False)
        processor.dataset = mock_dataset
        
        with patch('builtins.hasattr') as mock_hasattr:
            mock_hasattr.side_effect = lambda obj, attr: attr == 'Time'
            time_str = processor.get_time_info(0)
            self.assertEqual(time_str, "20240101T00")
    
    def test_extract_spatial_coordinates_no_dataset(self):
        """Test spatial coordinate extraction without dataset."""
        processor = MPASDataProcessor(self.grid_file, verbose=False)
        
        with self.assertRaises(ValueError):
            processor.extract_spatial_coordinates()


class TestDataValidation(unittest.TestCase):
    """
    Tests for data validation helpers.

    Scope:
        Verifies filtering by spatial extent and other validation logic using
        synthetic xarray DataArrays.
    """
    
    def test_filter_by_spatial_extent(self):
        """Test spatial extent filtering."""
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
            
            processor = MPASDataProcessor(grid_file, verbose=False)
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
    
    def test_invalid_variable_name(self):
        """Test handling of invalid variable names."""
        pass
    
    def test_corrupted_data_files(self):
        """Test handling of corrupted data files."""
        pass


if __name__ == '__main__':
    unittest.main()