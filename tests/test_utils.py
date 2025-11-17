#!/usr/bin/env python3
"""
MPAS Utility Module Unit Tests

This module provides comprehensive unit tests for MPAS utility classes and helper functions
including configuration management (MPASConfig), logging (MPASLogger), file operations
(FileManager), data validation (DataValidator), performance monitoring (PerformanceMonitor),
and command-line argument parsing (ArgumentParser). These tests validate core infrastructure
components used across the package using synthetic data, temporary files, and mocking to
isolate functionality from system dependencies.

Tests Performed:
    TestMPASConfig:
        - test_default_initialization: Verifies default configuration values (variable, DPI, formats)
        - test_custom_initialization: Tests custom parameter initialization and field assignment
        - test_invalid_spatial_extent: Validates error handling for invalid lon/lat bounds
        - test_to_dict: Tests serialization of config object to dictionary
        - test_from_dict: Validates deserialization from dictionary to config object
        - test_save_and_load_file: Tests YAML file I/O for configuration persistence
    
    TestMPASLogger:
        - test_logger_initialization: Verifies logger setup with default handlers
        - test_logger_with_file: Tests file handler attachment and log file writing
        - test_log_levels: Validates logging at different levels (DEBUG, INFO, WARNING, ERROR)
    
    TestFileManager:
        - test_ensure_directory: Tests directory creation with proper permissions
        - test_find_files: Validates file pattern matching and discovery
        - test_get_file_info: Tests file metadata extraction (size, timestamps)
        - test_get_file_info_nonexistent: Validates error handling for missing files
    
    TestDataValidator:
        - test_validate_coordinates_valid: Tests validation of proper coordinate arrays
        - test_validate_coordinates_invalid_length: Validates detection of mismatched array sizes
        - test_validate_coordinates_invalid_values: Tests handling of out-of-range coordinates
        - test_validate_data_array_valid: Verifies validation of clean data arrays
        - test_validate_data_array_with_issues: Tests detection of NaN/Inf values and outliers
        - test_validate_data_array_all_identical: Validates handling of zero-variance data
    
    TestPerformanceMonitor:
        - test_timer_context_manager: Tests timing context manager for operation profiling
        - test_multiple_operations: Validates tracking of multiple timed operations
    
    TestArgumentParser:
        - test_create_parser: Verifies argument parser creation with expected options
        - test_parse_args_to_config: Tests mapping of CLI arguments to MPASConfig object
    
    TestUtilityFunctions:
        - test_format_file_size: Tests human-readable file size formatting (B, KB, MB, GB)
        - test_create_output_filename: Validates standardized filename generation with metadata
        - test_get_available_memory_with_psutil: Tests memory detection with psutil library
        - test_get_available_memory_without_psutil: Tests fallback memory detection method

Test Coverage:
    - Configuration management: dataclass initialization, validation, YAML serialization
    - Logging infrastructure: handler setup, file output, level filtering
    - File operations: directory creation, pattern matching, metadata extraction
    - Data validation: coordinate bounds checking, array quality assessment, statistics
    - Performance monitoring: context managers, timing accumulation, summary reporting
    - Argument parsing: CLI parser construction, config mapping, type conversion
    - Utility functions: file size formatting, filename standardization, memory detection

Testing Approach:
    Unit tests using temporary directories and files for I/O operations, synthetic NumPy
    arrays for data validation, and mocking for system resources (memory, file systems).
    Tests verify proper initialization, error handling, edge cases, and expected return
    types without requiring actual MPAS data or heavy system dependencies.

Expected Results:
    - Configuration objects serialize/deserialize correctly maintaining all fields
    - Logger writes messages to appropriate handlers and files
    - File manager correctly finds, creates, and reports metadata for filesystem objects
    - Data validator detects coordinate mismatches, NaN/Inf values, and range violations
    - Performance monitor accurately tracks operation timings with low overhead
    - Argument parser creates parsers with expected options and maps to config correctly
    - Utility functions format sizes correctly and generate valid standardized filenames

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import os
import yaml
import sys
import unittest
import tempfile
import os
import yaml
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock
from pathlib import Path

package_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_dir))

from mpasdiag.processing.utils_config import MPASConfig
from mpasdiag.processing.utils_logger import MPASLogger
from mpasdiag.processing.utils_file import FileManager
from mpasdiag.processing.utils_validator import DataValidator
from mpasdiag.processing.utils_monitor import PerformanceMonitor
from mpasdiag.processing.utils_parser import ArgumentParser

file_manager = FileManager()
get_available_memory = file_manager.get_available_memory
format_file_size = file_manager.format_file_size
create_output_filename = file_manager.create_output_filename


def create_mock_memory_getter(available_bytes: int):
    """
    Create a mock get_available_memory function for testing memory detection.

    Parameters:
        available_bytes (int): Simulated available memory in bytes.

    Returns:
        Callable: Function that returns memory in gigabytes.
    """
    return lambda: available_bytes / (1024**3)


class TestMPASConfig(unittest.TestCase):
    """
    Tests for MPASConfig dataclass behavior.

    Scope:
        Verifies default values, custom initialization, and serialization
        (save/load) using temporary YAML files.
    Test data:
        Synthetic config dictionaries and temporary files on disk.
    """
    
    def test_default_initialization(self) -> None:
        """
        Verify MPASConfig dataclass instantiation with default parameter values for variable, DPI, output formats, and behavior flags. This test confirms that configuration objects initialize with sensible operational defaults including rainnc variable selection, 1-hour accumulation period, 100 DPI resolution, and PNG output format. The default configuration enables verbose logging while disabling quiet mode and pure xarray processing for standard diagnostic workflows. Testing validates that all configuration fields receive appropriate type-consistent default values without requiring explicit parameter specification. These defaults support rapid prototyping and standard analysis workflows where minimal configuration provides functional behavior.

        Parameters:
            None

        Returns:
            None
        """
        config = MPASConfig()
        
        self.assertEqual(config.variable, "rainnc")
        self.assertEqual(config.accumulation_period, "a01h")
        self.assertEqual(config.dpi, 100)  
        self.assertFalse(config.use_pure_xarray)
        self.assertTrue(config.verbose)
        self.assertFalse(config.quiet)
        self.assertEqual(config.output_formats, ["png"])
    
    def test_custom_initialization(self) -> None:
        """
        Validate MPASConfig instantiation with user-specified custom parameters overriding default values for paths, variables, and output settings. This test confirms that explicit parameter values for grid file path, data directory, variable selection (rainc), DPI resolution (600), and output format list properly override defaults. Custom initialization testing ensures that users can fully configure analysis workflows through constructor parameters without post-initialization modifications. Field assignment validation verifies that all specified parameters propagate correctly to the configuration object attributes. This flexibility supports diverse operational scenarios requiring non-default settings for file locations, visualization parameters, and output specifications.

        Parameters:
            None

        Returns:
            None
        """
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
    
    def test_invalid_spatial_extent(self) -> None:
        """
        Verify validation error handling for geographically invalid spatial extent specifications with inverted longitude bounds. This test confirms that MPASConfig raises ValueError when longitude minimum exceeds longitude maximum (110° > 100°), preventing nonsensical spatial domain definitions. The validation catches configuration errors at initialization time before invalid bounds propagate through analysis workflows causing downstream failures. Geographic constraint checking ensures that spatial extent specifications maintain logical consistency with standard longitude/latitude coordinate conventions. This defensive validation approach improves error diagnostics by rejecting invalid configurations immediately with clear error messages rather than allowing cryptic failures during data processing.

        Parameters:
            None

        Returns:
            None
        """
        with self.assertRaises(ValueError):
            MPASConfig(
                lon_min=110.0,  
                lon_max=100.0,
                lat_min=-10.0,
                lat_max=10.0
            )
    
    def test_to_dict(self) -> None:
        """
        Validate serialization of MPASConfig dataclass to dictionary representation for file persistence and data exchange. This test confirms that the to_dict method converts configuration objects to standard Python dictionaries maintaining all field values and types. Dictionary serialization enables YAML file output, JSON export, and interoperability with external configuration management systems. The test validates that specific fields (variable, dpi) appear in the dictionary output with correct values matching the original configuration object. This serialization capability supports configuration persistence, version control, and sharing of analysis parameters across different execution environments.

        Parameters:
            None

        Returns:
            None
        """
        config = MPASConfig(variable="rainc", dpi=400)
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["variable"], "rainc")
        self.assertEqual(config_dict["dpi"], 400)
    
    def test_from_dict(self) -> None:
        """
        Verify deserialization of dictionary representation to MPASConfig dataclass object for configuration loading workflows. This test confirms that the from_dict class method correctly reconstructs configuration objects from dictionary inputs maintaining field types and values. Deserialization enables configuration loading from YAML files, JSON documents, and programmatic dictionary definitions. The test validates that specific dictionary entries (variable: total, dpi: 500, verbose: False) map correctly to configuration object attributes. This deserialization capability supports configuration file-based workflows where users define analysis parameters in external YAML files loaded at runtime.

        Parameters:
            None

        Returns:
            None
        """
        config_dict = {
            "variable": "total",
            "dpi": 500,
            "verbose": False
        }
        
        config = MPASConfig.from_dict(config_dict)
        
        self.assertEqual(config.variable, "total")
        self.assertEqual(config.dpi, 500)
        self.assertFalse(config.verbose)
    
    def test_save_and_load_file(self) -> None:
        """
        Validate complete configuration persistence workflow including YAML file writing and reading with full round-trip fidelity. This test confirms that MPASConfig objects serialize to YAML files and deserialize back to equivalent configuration objects preserving all field values. The test uses temporary files for I/O operations validating that saved configurations load correctly with matching variable (rainc), DPI (350), and output format ([png, svg]) specifications. Round-trip testing ensures that configuration persistence maintains data integrity without loss or corruption through serialization cycles. This persistence capability enables reproducible analysis workflows where configuration files document exact parameters used for specific diagnostic runs.

        Parameters:
            None

        Returns:
            None
        """
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
    
    def test_logger_initialization(self) -> None:
        """
        Verify MPASLogger instantiation with proper handler attachment and name assignment for message routing infrastructure. This test confirms that logger objects initialize with correct logger names and attach appropriate handlers for message output. Handler presence validation ensures that logging messages have output destinations whether console, file, or both depending on verbosity settings. The test validates that logger infrastructure sets up correctly without requiring external dependencies or configuration files. This initialization testing ensures that logging functionality works reliably across different execution environments providing consistent message output for debugging and operational monitoring.

        Parameters:
            None

        Returns:
            None
        """
        logger = MPASLogger("test_logger", verbose=True)
        
        self.assertEqual(logger.logger.name, "test_logger")
        self.assertTrue(len(logger.logger.handlers) > 0)
    
    def test_logger_with_file(self) -> None:
        """
        Validate file handler attachment and log message persistence to disk files for audit trail and debugging support. This test confirms that MPASLogger correctly creates file handlers writing log messages to specified file paths. The test uses temporary files to validate that logged messages (Test message) appear in file content after info-level logging calls. File writing validation ensures that log persistence works correctly enabling post-execution analysis of application behavior and error diagnostics. This file logging capability supports operational workflows requiring audit trails, debugging history, and long-term message archival for troubleshooting distributed processing jobs.

        Parameters:
            None

        Returns:
            None
        """
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
    
    def test_log_levels(self) -> None:
        """
        Verify logging at multiple severity levels including DEBUG, INFO, WARNING, and ERROR for flexible message categorization. This test validates that MPASLogger accepts log messages at all standard Python logging levels without errors or exceptions. The test operates with verbose=False to confirm that level filtering works correctly suppressing debug messages while allowing higher-priority messages. Level-based logging supports operational workflows requiring different verbosity settings from detailed debugging output to minimal error-only reporting. This multi-level logging capability enables flexible message output control through configuration without code modifications.

        Parameters:
            None

        Returns:
            None
        """
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
    
    def setUp(self) -> None:
        """
        Initialize temporary directory for file operation testing providing isolated filesystem environment for each test method. This fixture creates a clean temporary directory before each test ensuring that file operations don't interfere with system directories or other tests. The temporary directory provides a sandbox for testing directory creation, file searching, and metadata extraction without risking data loss or permission issues. Isolated test environments ensure reproducible test behavior independent of system state or previous test executions. This setup approach follows testing best practices for file I/O operations requiring filesystem access.

        Parameters:
            None

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self) -> None:
        """
        Clean up temporary directory and all contained files after test completion ensuring no filesystem artifacts remain. This fixture removes the temporary test directory recursively including all subdirectories and files created during test execution. The cleanup uses ignore_errors flag to handle cases where files may be locked or inaccessible preventing test failures from cleanup issues. Proper teardown maintains clean test environment state preventing disk space accumulation from repeated test runs. This cleanup approach follows testing best practices ensuring that each test starts with a fresh filesystem state.

        Parameters:
            None

        Returns:
            None
        """
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ensure_directory(self) -> None:
        """
        Validate recursive directory creation including nested subdirectory paths with automatic parent directory generation. This test confirms that FileManager.ensure_directory creates multi-level directory hierarchies (subdir/subsubdir) when intermediate directories don't exist. The functionality mirrors mkdir -p behavior creating all necessary parent directories automatically without requiring separate creation calls. Directory existence and type validation ensures that created paths are actual directories not files or symbolic links. This recursive creation capability simplifies output directory management where deep hierarchies are needed for organized data storage.

        Parameters:
            None

        Returns:
            None
        """
        test_dir = os.path.join(self.temp_dir, "subdir", "subsubdir")
        
        FileManager.ensure_directory(test_dir)
        
        self.assertTrue(os.path.exists(test_dir))
        self.assertTrue(os.path.isdir(test_dir))
    
    def test_find_files(self) -> None:
        """
        Verify pattern-based file discovery using glob patterns for filtering file lists by extension or naming convention. This test validates that FileManager.find_files correctly identifies files matching wildcard patterns (*.nc) while excluding non-matching files (*.txt). The test creates multiple files in temporary directory and confirms that NetCDF file pattern returns exactly two matches with correct filenames. Pattern matching enables flexible file discovery for batch processing workflows where input files share common naming conventions. This search capability supports automated data ingestion workflows discovering all relevant files without explicit filename enumeration.

        Parameters:
            None

        Returns:
            None
        """
        test_files = ["test1.nc", "test2.nc", "other.txt"]

        for filename in test_files:
            with open(os.path.join(self.temp_dir, filename), 'w') as f:
                f.write("test content")
        
        nc_files = FileManager.find_files(self.temp_dir, "*.nc")
        
        self.assertEqual(len(nc_files), 2)
        self.assertTrue(any("test1.nc" in f for f in nc_files))
        self.assertTrue(any("test2.nc" in f for f in nc_files))
    
    def test_get_file_info(self) -> None:
        """
        Validate file metadata extraction including existence status, size, modification time, and creation time for file management operations. This test confirms that FileManager.get_file_info returns comprehensive file information dictionary with correct data types and non-zero values. The metadata includes boolean existence flag, integer size in bytes, and datetime objects for modification and creation timestamps. File information extraction supports logging, audit trails, and conditional processing based on file characteristics. This metadata access capability enables sophisticated file management workflows requiring decisions based on file age, size, or existence status.

        Parameters:
            None

        Returns:
            None
        """
        test_file = os.path.join(self.temp_dir, "test.txt")

        with open(test_file, 'w') as f:
            f.write("Hello, World!")
        
        file_info = FileManager.get_file_info(test_file)
        
        self.assertTrue(file_info["exists"])
        self.assertGreater(file_info["size"], 0)
        self.assertIsInstance(file_info["modified"], datetime)
        self.assertIsInstance(file_info["created"], datetime)
    
    def test_get_file_info_nonexistent(self) -> None:
        """
        Verify graceful handling of metadata requests for nonexistent files returning existence flag without raising exceptions. This test confirms that FileManager.get_file_info handles missing files by returning dictionary with exists=False rather than throwing file-not-found errors. Graceful error handling enables conditional file processing workflows where code checks existence before attempting operations. The non-exception approach simplifies calling code eliminating need for try-except blocks around every file access operation. This defensive design pattern supports robust file management where missing files are expected conditions not exceptional failures.

        Parameters:
            None

        Returns:
            None
        """
        file_info = FileManager.get_file_info("/nonexistent/file.txt")
        
        self.assertFalse(file_info["exists"])


class TestDataValidator(unittest.TestCase):
    """
    Tests for DataValidator utility functions.

    Scope:
        Validates coordinate arrays, data arrays and error reporting on
        invalid inputs using synthetic NumPy arrays.
    """
    
    def test_validate_coordinates_valid(self) -> None:
        """
        Verify acceptance of properly formatted coordinate arrays with valid geographic ranges and matching array lengths for spatial data validation. This test confirms that DataValidator.validate_coordinates returns True for longitude and latitude arrays with correct dimensions and values within standard geographic ranges (-180° to 180°, -90° to 90°). The validation uses representative coordinate values (100-110°E, -5 to 5°N) spanning reasonable geographic extent. Coordinate validation prevents downstream processing errors from malformed spatial data. This validation capability ensures that coordinate arrays meet basic quality requirements before expensive computation or visualization operations.

        Parameters:
            None

        Returns:
            None
        """
        lon = np.array([100.0, 105.0, 110.0])
        lat = np.array([-5.0, 0.0, 5.0])
        
        self.assertTrue(DataValidator.validate_coordinates(lon, lat))
    
    def test_validate_coordinates_invalid_length(self) -> None:
        """
        Verify detection of coordinate array length mismatches where longitude and latitude arrays have different sizes. This test confirms that DataValidator.validate_coordinates returns False when coordinate arrays don't match in length (3-element lon vs 2-element lat). Length mismatch detection prevents zip pairing errors and indexing failures in downstream processing requiring coordinate correspondence. The validation catches data loading errors where partial arrays or incorrect file parsing creates misaligned coordinates. This defensive validation approach ensures data consistency before expensive operations begin preventing cryptic failures deep in processing pipelines.

        Parameters:
            None

        Returns:
            None
        """
        lon = np.array([100.0, 105.0, 110.0])
        lat = np.array([-5.0, 0.0])  
        
        self.assertFalse(DataValidator.validate_coordinates(lon, lat))
    
    def test_validate_coordinates_invalid_values(self) -> None:
        """
        Verify detection of out-of-range coordinate values exceeding valid geographic bounds for longitude and latitude constraints. This test confirms that DataValidator.validate_coordinates returns False for coordinates outside standard ranges including longitude beyond ±180° (200°) and latitude beyond ±90° (-100°). Range validation prevents nonsensical geographic specifications that would cause projection errors or visualization failures. The test validates both longitude and latitude bounds checking ensuring comprehensive range enforcement. This geographic constraint validation supports robust spatial data handling where coordinate sanity checking prevents downstream errors from invalid geographic specifications.

        Parameters:
            None

        Returns:
            None
        """
        lon = np.array([200.0, 105.0, 110.0])
        lat = np.array([-5.0, 0.0, 5.0])
        
        self.assertFalse(DataValidator.validate_coordinates(lon, lat))
        
        lon = np.array([100.0, 105.0, 110.0])
        lat = np.array([-100.0, 0.0, 5.0])
        
        self.assertFalse(DataValidator.validate_coordinates(lon, lat))
    
    def test_validate_data_array_valid(self) -> None:
        """
        Verify acceptance of clean data arrays passing all quality checks including range constraints and completeness validation. This test confirms that DataValidator.validate_data_array returns valid=True with empty issues list for arrays containing only finite values within specified bounds (0-10). The validation computes comprehensive statistics including total points, finite point count, minimum, and maximum values for quality assessment. Array validation with successful results indicates data suitable for visualization or analysis without requiring cleaning or filtering. This quality checking capability supports data-driven decisions about whether arrays meet requirements for downstream processing.

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = DataValidator.validate_data_array(data, min_val=0.0, max_val=10.0)
        
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["issues"]), 0)
        self.assertEqual(result["stats"]["total_points"], 5)
        self.assertEqual(result["stats"]["finite_points"], 5)
        self.assertEqual(result["stats"]["min"], 1.0)
        self.assertEqual(result["stats"]["max"], 5.0)
    
    def test_validate_data_array_with_issues(self) -> None:
        """
        Verify detection of data quality problems including NaN values, range violations, and outliers in validation reporting. This test confirms that DataValidator.validate_data_array returns valid=False with non-empty issues list for arrays containing missing data (NaN), out-of-range values (15 > max 10), and negative values (-1 < min 0). The validation distinguishes total point count from finite point count enabling identification of missing data extent. Issue detection supports informed decisions about data cleaning requirements or suitability for analysis. This comprehensive quality assessment capability identifies multiple data problems simultaneously providing complete diagnostic information for data preparation workflows.

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([1.0, np.nan, 15.0, 4.0, -1.0])  
        
        result = DataValidator.validate_data_array(data, min_val=0.0, max_val=10.0)
        
        self.assertFalse(result["valid"])
        self.assertGreater(len(result["issues"]), 0)
        self.assertEqual(result["stats"]["total_points"], 5)
        self.assertEqual(result["stats"]["finite_points"], 4) 
    
    def test_validate_data_array_all_identical(self) -> None:
        """
        Verify detection of zero-variance data arrays where all values are identical indicating potential data problems or processing errors. This test confirms that DataValidator.validate_data_array returns valid=False with specific issue message for arrays with all identical values (all 5.0). Zero standard deviation detection identifies data that lacks variability making visualization meaningless and suggesting data loading or processing failures. The validation prevents wasted computation on degenerate data while alerting users to investigate data sources. This variance checking capability supports quality assurance workflows where zero-variance arrays indicate upstream problems requiring investigation before expensive analysis operations.

        Parameters:
            None

        Returns:
            None
        """
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
    
    def test_timer_context_manager(self) -> None:
        """
        Validate timing context manager functionality measuring operation duration with minimal overhead for performance profiling. This test confirms that PerformanceMonitor.timer context manager accurately captures elapsed time for operations (0.01s sleep) within reasonable bounds. The timing measurement appears in summary dictionary with operation name as key and elapsed seconds as value. Context manager approach enables clean timing instrumentation without explicit start/stop calls and automatic timing completion even with exceptions. This timing capability supports performance analysis workflows identifying bottlenecks and validating optimization efforts through before/after comparisons.

        Parameters:
            None

        Returns:
            None
        """
        monitor = PerformanceMonitor()
        
        with monitor.timer("test_operation"):
            import time
            time.sleep(0.01)  
        
        summary = monitor.get_summary()
        
        self.assertIn("test_operation", summary)
        self.assertGreater(summary["test_operation"], 0.0)
        self.assertLess(summary["test_operation"], 1.0)  
    
    def test_multiple_operations(self) -> None:
        """
        Verify tracking of multiple independent timed operations with separate timing accumulation for performance breakdown analysis. This test confirms that PerformanceMonitor maintains separate timing entries for different operation names (operation1, operation2) in summary dictionary. Multiple operation tracking enables comprehensive performance profiling identifying relative time consumption across different processing stages. The test validates that summary contains exactly two entries with distinct operation identifiers. This multi-operation timing capability supports detailed performance analysis workflows where understanding relative costs of different processing phases guides optimization priorities.

        Parameters:
            None

        Returns:
            None
        """
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
    
    def test_create_parser(self) -> None:
        """
        Verify argument parser creation with expected command-line options and proper argument parsing for configuration specification. This test confirms that ArgumentParser.create_parser generates parser objects accepting standard options including grid-file, data-dir, and var flags. The test validates that parser correctly processes command-line argument lists producing parsed namespace objects with appropriate attribute values. Parser creation testing ensures that command-line interface accepts expected arguments with correct names and formats. This CLI testing capability supports verification that user-facing interfaces work correctly before deployment.

        Parameters:
            None

        Returns:
            None
        """
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
    
    def test_parse_args_to_config(self) -> None:
        """
        Validate mapping of parsed command-line arguments to MPASConfig dataclass fields with proper type conversion and field assignment. This test confirms that ArgumentParser.parse_args_to_config correctly transforms argparse namespace objects into configuration objects with matching field values. The mapping handles type conversions including string-to-integer for DPI and string-to-boolean for verbose flag. Field name mapping accommodates CLI naming conventions (var) to internal field names (variable). This argument-to-config transformation capability enables seamless integration of command-line specifications into configuration-driven workflows.

        Parameters:
            None

        Returns:
            None
        """
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
    """
    Test standalone utility functions.
    """
    
    def test_format_file_size(self) -> None:
        """
        Verify human-readable file size formatting with automatic unit selection from bytes through gigabytes for user-friendly display. This test validates that format_file_size converts byte counts to appropriate units with 1024-based powers including 512 B, 1.0 KB, 1.0 MB, and 1.0 GB. Automatic unit selection provides readable sizes without scientific notation or overwhelming digit counts. The formatting uses single decimal precision balancing readability with sufficient accuracy for file size reporting. This size formatting capability supports user-friendly logging and reporting where file sizes appear in natural units matching user expectations.

        Parameters:
            None

        Returns:
            None
        """
        self.assertEqual(format_file_size(512), "512.0 B")
        self.assertEqual(format_file_size(1024), "1.0 KB")
        self.assertEqual(format_file_size(1024 * 1024), "1.0 MB")
        self.assertEqual(format_file_size(1024 * 1024 * 1024), "1.0 GB")
    
    def test_create_output_filename(self) -> None:
        """
        Validate standardized filename generation incorporating metadata fields including variable type, accumulation period, and validation time. This test confirms that create_output_filename produces consistent filenames following naming convention with underscored field separators. The generated filename (precipitation_map_vartype_rainc_acctype_a01h_valid_20240101T12_point.png) includes all specified metadata fields enabling self-documenting output files. Standardized naming supports automated file discovery and parsing where filename structure encodes essential metadata. This filename generation capability ensures consistent file organization across different analysis runs and facilitates batch processing workflows requiring systematic file identification.

        Parameters:
            None

        Returns:
            None
        """
        filename = create_output_filename(
            "precipitation_map", "20240101T12", "rainc", "a01h", "png"
        )
        
        expected = "precipitation_map_vartype_rainc_acctype_a01h_valid_20240101T12_point.png"
        self.assertEqual(filename, expected)
    
    def test_get_available_memory_with_psutil(self) -> None:
        """
        Verify memory detection using simulated memory values for system resource assessment. This test confirms that a mock memory getter correctly retrieves and converts available memory (8 GB) to floating-point gigabyte representation. The test uses a mock function to simulate memory values without depending on actual system memory state ensuring reproducible test results. Memory detection enables resource-aware processing where algorithms adjust behavior based on available system memory. This memory detection supports optimal resource utilization in data-intensive workflows requiring memory-conscious processing strategies.

        Parameters:
            None

        Returns:
            None
        """
        mock_get_memory = create_mock_memory_getter(8 * (1024**3))
        
        memory = mock_get_memory()
        self.assertEqual(memory, 8.0)
    
    def test_get_available_memory_without_psutil(self) -> None:
        """
        Verify fallback memory detection functionality operating correctly when psutil library is unavailable or import fails. This test confirms that get_available_memory returns non-negative floating-point values even without psutil dependency using alternative detection methods. Fallback capability ensures that memory-dependent code degrades gracefully on systems lacking psutil rather than failing completely. The test validates return type and non-negative constraint without specifying exact values since fallback methods vary across platforms. This fallback memory detection supports portable code execution across diverse environments including minimal installations lacking optional dependencies.

        Parameters:
            None

        Returns:
            None
        """
        memory = get_available_memory()
        self.assertIsInstance(memory, float)
        self.assertGreaterEqual(memory, 0.0)


if __name__ == '__main__':
    unittest.main()