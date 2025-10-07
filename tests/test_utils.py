#!/usr/bin/env python3

"""
Unit tests for MPAS Utils Module

Scope:
        Test utility classes and helpers used across the package: configuration
        dataclass, logging helper, file manager, data validator, performance
        monitor, argument parser, and small helpers for filesystem/memory.

Test data:
        Uses temporary files/directories and synthetic NumPy arrays. YAML files
        are created as needed for configuration load/save tests. Mocking is
        used to simulate system resources and file I/O where appropriate.

Expected results:
        - Config serialization/deserialization preserves expected fields.
        - Logger writes messages to file and sets up handlers correctly.
        - File manager finds files and reports metadata accurately.
        - Data validator detects invalid arrays and reports summary statistics.
        - Performance monitor records timings and returns a small summary.
        - Argument parser builds parsers that accept expected flags.

Per-test expectations (short):
        - TestMPASConfig: default and custom initialization behave as expected;
            saving and loading configuration preserves fields.
        - TestMPASLogger: logger writes messages and attaches handlers properly.
        - TestFileManager: file creation, search, and metadata functions behave
            correctly on disk.
        - TestDataValidator: validation flags inconsistent arrays and reports
            issues and statistics.
        - TestArgumentParser: parser creation functions return parsers with the
            expected options and argument parsing maps to configuration.

Author: Rubaiat Islam
"""

import unittest
import tempfile
import os
import yaml
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime

import sys
from pathlib import Path
package_dir = Path(__file__).parent.parent
sys.path.insert(0, str(package_dir))

from mpas_analysis.utils import (
    MPASConfig,
    MPASLogger,
    FileManager,
    DataValidator,
    PerformanceMonitor,
    ArgumentParser,
    get_available_memory,
    format_file_size,
    create_output_filename
)


class TestMPASConfig(unittest.TestCase):
    """
    Tests for MPASConfig dataclass behavior.

    Scope:
        Verifies default values, custom initialization, and serialization
        (save/load) using temporary YAML files.
    Test data:
        Synthetic config dictionaries and temporary files on disk.
    """
    
    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = MPASConfig()
        
        self.assertEqual(config.variable, "rainnc")
        self.assertEqual(config.accumulation_period, "a01h")
        self.assertEqual(config.dpi, 300)
        self.assertFalse(config.use_pure_xarray)
        self.assertTrue(config.verbose)
        self.assertFalse(config.quiet)
        self.assertEqual(config.output_formats, ["png"])
    
    def test_custom_initialization(self):
        """Test custom configuration initialization."""
        config = MPASConfig(
            grid_file="/path/to/grid.nc",
            data_dir="/path/to/data",
            variable="rainc",
            dpi=600,
            output_formats=["png", "pdf"]
        )
        
        self.assertEqual(config.grid_file, "/path/to/grid.nc")
        self.assertEqual(config.data_dir, "/path/to/data")
        self.assertEqual(config.variable, "rainc")
        self.assertEqual(config.dpi, 600)
        self.assertEqual(config.output_formats, ["png", "pdf"])
    
    def test_invalid_spatial_extent(self):
        """Test validation of invalid spatial extent."""
        with self.assertRaises(ValueError):
            MPASConfig(
                lon_min=110.0,  
                lon_max=100.0,
                lat_min=-10.0,
                lat_max=10.0
            )
    
    def test_to_dict(self):
        """Test configuration conversion to dictionary."""
        config = MPASConfig(variable="rainc", dpi=400)
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["variable"], "rainc")
        self.assertEqual(config_dict["dpi"], 400)
    
    def test_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "variable": "total",
            "dpi": 500,
            "verbose": False
        }
        
        config = MPASConfig.from_dict(config_dict)
        
        self.assertEqual(config.variable, "total")
        self.assertEqual(config.dpi, 500)
        self.assertFalse(config.verbose)
    
    def test_save_and_load_file(self):
        """Test saving and loading configuration files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_file = f.name
        
        try:
            config = MPASConfig(
                variable="rainc",
                dpi=350,
                output_formats=["png", "svg"]
            )
            config.save_to_file(config_file)
            
            loaded_config = MPASConfig.load_from_file(config_file)
            
            self.assertEqual(loaded_config.variable, "rainc")
            self.assertEqual(loaded_config.dpi, 350)
            self.assertEqual(loaded_config.output_formats, ["png", "svg"])
            
        finally:
            os.unlink(config_file)


class TestMPASLogger(unittest.TestCase):
    """
    Tests for MPASLogger behavior.

    Scope:
        Ensures logger attaches handlers and writes messages to files when
        requested. Uses temporary files for verification.
    """
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        logger = MPASLogger("test_logger", verbose=True)
        
        self.assertEqual(logger.logger.name, "test_logger")
        self.assertTrue(len(logger.logger.handlers) > 0)
    
    def test_logger_with_file(self):
        """Test logger with file output."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            log_file = f.name
        
        try:
            logger = MPASLogger("test_logger", log_file=log_file, verbose=True)
            logger.info("Test message")
            
            self.assertTrue(os.path.exists(log_file))
            with open(log_file, 'r') as f:
                content = f.read()
                self.assertIn("Test message", content)
                
        finally:
            os.unlink(log_file)
    
    def test_log_levels(self):
        """Test different log levels."""
        logger = MPASLogger("test_logger", verbose=False)  
        
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.debug("Debug message")


class TestFileManager(unittest.TestCase):
    """
    Tests for FileManager utilities.

    Scope:
        File/directory creation, search, and metadata extraction using
        temporary directories and files.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ensure_directory(self):
        """Test directory creation."""
        test_dir = os.path.join(self.temp_dir, "subdir", "subsubdir")
        
        FileManager.ensure_directory(test_dir)
        
        self.assertTrue(os.path.exists(test_dir))
        self.assertTrue(os.path.isdir(test_dir))
    
    def test_find_files(self):
        """Test file finding functionality."""
        test_files = ["test1.nc", "test2.nc", "other.txt"]
        for filename in test_files:
            with open(os.path.join(self.temp_dir, filename), 'w') as f:
                f.write("test content")
        
        nc_files = FileManager.find_files(self.temp_dir, "*.nc")
        
        self.assertEqual(len(nc_files), 2)
        self.assertTrue(any("test1.nc" in f for f in nc_files))
        self.assertTrue(any("test2.nc" in f for f in nc_files))
    
    def test_get_file_info(self):
        """Test file information extraction."""
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("Hello, World!")
        
        file_info = FileManager.get_file_info(test_file)
        
        self.assertTrue(file_info["exists"])
        self.assertGreater(file_info["size"], 0)
        self.assertIsInstance(file_info["modified"], datetime)
        self.assertIsInstance(file_info["created"], datetime)
    
    def test_get_file_info_nonexistent(self):
        """Test file info for nonexistent file."""
        file_info = FileManager.get_file_info("/nonexistent/file.txt")
        
        self.assertFalse(file_info["exists"])


class TestDataValidator(unittest.TestCase):
    """
    Tests for DataValidator utility functions.

    Scope:
        Validates coordinate arrays, data arrays and error reporting on
        invalid inputs using synthetic NumPy arrays.
    """
    
    def test_validate_coordinates_valid(self):
        """Test coordinate validation with valid data."""
        lon = np.array([100.0, 105.0, 110.0])
        lat = np.array([-5.0, 0.0, 5.0])
        
        self.assertTrue(DataValidator.validate_coordinates(lon, lat))
    
    def test_validate_coordinates_invalid_length(self):
        """Test coordinate validation with mismatched lengths."""
        lon = np.array([100.0, 105.0, 110.0])
        lat = np.array([-5.0, 0.0])  
        
        self.assertFalse(DataValidator.validate_coordinates(lon, lat))
    
    def test_validate_coordinates_invalid_values(self):
        """Test coordinate validation with invalid values."""
        lon = np.array([200.0, 105.0, 110.0])
        lat = np.array([-5.0, 0.0, 5.0])
        
        self.assertFalse(DataValidator.validate_coordinates(lon, lat))
        
        lon = np.array([100.0, 105.0, 110.0])
        lat = np.array([-100.0, 0.0, 5.0])
        
        self.assertFalse(DataValidator.validate_coordinates(lon, lat))
    
    def test_validate_data_array_valid(self):
        """Test data array validation with valid data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = DataValidator.validate_data_array(data, min_val=0.0, max_val=10.0)
        
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["issues"]), 0)
        self.assertEqual(result["stats"]["total_points"], 5)
        self.assertEqual(result["stats"]["finite_points"], 5)
        self.assertEqual(result["stats"]["min"], 1.0)
        self.assertEqual(result["stats"]["max"], 5.0)
    
    def test_validate_data_array_with_issues(self):
        """Test data array validation with issues."""
        data = np.array([1.0, np.nan, 15.0, 4.0, -1.0])  
        
        result = DataValidator.validate_data_array(data, min_val=0.0, max_val=10.0)
        
        self.assertFalse(result["valid"])
        self.assertGreater(len(result["issues"]), 0)
        self.assertEqual(result["stats"]["total_points"], 5)
        self.assertEqual(result["stats"]["finite_points"], 4) 
    
    def test_validate_data_array_all_identical(self):
        """Test data array validation with identical values."""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        
        result = DataValidator.validate_data_array(data)
        
        self.assertFalse(result["valid"])
        self.assertIn("All values are identical", result["issues"])
        self.assertEqual(result["stats"]["std"], 0.0)


class TestPerformanceMonitor(unittest.TestCase):
    """
    Tests for PerformanceMonitor timing utilities.

    Scope:
        Uses short sleep intervals to assert timer context manager records
        timings and returns summary values.
    """
    
    def test_timer_context_manager(self):
        """Test timer context manager."""
        monitor = PerformanceMonitor()
        
        with monitor.timer("test_operation"):
            import time
            time.sleep(0.01)  
        
        summary = monitor.get_summary()
        
        self.assertIn("test_operation", summary)
        self.assertGreater(summary["test_operation"], 0.0)
        self.assertLess(summary["test_operation"], 1.0)  
    
    def test_multiple_operations(self):
        """Test monitoring multiple operations."""
        monitor = PerformanceMonitor()
        
        with monitor.timer("operation1"):
            import time
            time.sleep(0.01)
        
        with monitor.timer("operation2"):
            import time
            time.sleep(0.01)
        
        summary = monitor.get_summary()
        
        self.assertEqual(len(summary), 2)
        self.assertIn("operation1", summary)
        self.assertIn("operation2", summary)


class TestArgumentParser(unittest.TestCase):
    """
    Tests for argument parser factories and argument->config mapping.

    Scope:
        Ensures parsers expose expected flags and that parsing maps to the
        configuration dataclass.
    """
    
    def test_create_parser(self):
        """Test parser creation."""
        parser = ArgumentParser.create_parser()
        
        self.assertIsNotNone(parser)
        
        args = parser.parse_args([
            '--grid-file', 'grid.nc',
            '--data-dir', './data',
            '--var', 'rainc'
        ])
        
        self.assertEqual(args.grid_file, 'grid.nc')
        self.assertEqual(args.data_dir, './data')
        self.assertEqual(args.var, 'rainc')
    
    def test_parse_args_to_config(self):
        """Test conversion of arguments to config."""
        parser = ArgumentParser.create_parser()
        args = parser.parse_args([
            '--grid-file', 'grid.nc',
            '--data-dir', './data',
            '--var', 'total',
            '--dpi', '400',
            '--verbose'
        ])
        
        config = ArgumentParser.parse_args_to_config(args)
        
        self.assertEqual(config.grid_file, 'grid.nc')
        self.assertEqual(config.data_dir, './data')
        self.assertEqual(config.variable, 'total')
        self.assertEqual(config.dpi, 400)
        self.assertTrue(config.verbose)


class TestUtilityFunctions(unittest.TestCase):
    """Test standalone utility functions."""
    
    def test_format_file_size(self):
        """Test file size formatting."""
        self.assertEqual(format_file_size(512), "512.0 B")
        self.assertEqual(format_file_size(1024), "1.0 KB")
        self.assertEqual(format_file_size(1024 * 1024), "1.0 MB")
        self.assertEqual(format_file_size(1024 * 1024 * 1024), "1.0 GB")
    
    def test_create_output_filename(self):
        """Test output filename creation."""
        filename = create_output_filename(
            "precipitation_map", "20240101T12", "rainc", "a01h", "png"
        )
        
        expected = "precipitation_map_vartype_rainc_acctype_a01h_valid_20240101T12_point.png"
        self.assertEqual(filename, expected)
    
    @patch('psutil.virtual_memory')
    def test_get_available_memory_with_psutil(self, mock_memory):
        """Test memory detection with psutil available."""
        mock_memory.return_value.available = 8 * (1024**3)  
        
        memory = get_available_memory()
        self.assertEqual(memory, 8.0)
    
    def test_get_available_memory_without_psutil(self):
        """Test memory detection without psutil."""
        memory = get_available_memory()
        self.assertIsInstance(memory, float)
        self.assertGreaterEqual(memory, 0.0)


if __name__ == '__main__':
    unittest.main()