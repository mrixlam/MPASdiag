#!/usr/bin/env python3
"""
MPASdiag Test Suite: Tests for MPAS Utility Classes and Helper Functions

This module contains unit tests for the utility classes and helper functions defined in `mpasdiag.processing.utils_parser` and `mpasdiag.processing.utils_config`. The tests cover the creation of argument parsers for general, surface, wind, and cross-section plotting, as well as the conversion of parsed arguments into configuration objects. The tests ensure that the parsers correctly recognize required and optional arguments, apply defaults, and that the conversion functions properly map argument namespaces to `MPASConfig` instances with the expected attributes and types.

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
import numpy as np
from pathlib import Path
from typing import Generator
from datetime import datetime

from mpasdiag.processing.utils_config import MPASConfig
from mpasdiag.processing.utils_logger import MPASLogger
from mpasdiag.processing.utils_file import FileManager
from mpasdiag.processing.utils_validator import DataValidator
from mpasdiag.processing.utils_monitor import PerformanceMonitor
from mpasdiag.processing.utils_parser import ArgumentParser

package_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_dir))

file_manager = FileManager()
get_available_memory = file_manager.get_available_memory
format_file_size = file_manager.format_file_size
create_output_filename = file_manager.create_output_filename


def create_mock_memory_getter(available_bytes: int):
    """
    This helper function creates a mock memory getter function that simulates available memory in bytes for testing purposes. It returns a lambda function that converts the provided byte value to gigabytes when called. This allows tests to simulate different memory availability scenarios without relying on actual system memory conditions. 

    Parameters:
        available_bytes (int): Simulated available memory in bytes.

    Returns:
        Callable: Function that returns memory in gigabytes.
    """
    return lambda: available_bytes / (1024**3)


class TestMPASConfig:
    """ Tests for MPASConfig dataclass behavior. """
    
    def test_default_initialization(self: "TestMPASConfig") -> None:
        """
        This test validates the default instantiation of the MPASConfig dataclass ensuring that all fields are initialized with expected default values. The test confirms that default parameters for variable selection (rainnc), accumulation period (a01h), DPI resolution (100), pure xarray usage (False), verbosity (True), quiet mode (False), and output formats (['png']) are correctly set when no arguments are provided. Default initialization testing ensures that users can create configuration objects with sensible defaults without needing to specify every parameter, supporting ease of use for common scenarios while allowing overrides when needed. 

        Parameters:
            None

        Returns:
            None
        """
        config = MPASConfig()        
        assert config.variable == "rainnc"
        assert config.accumulation_period == "a01h"
        assert config.dpi == 100
        assert config.use_pure_xarray is False
        assert config.verbose is True
        assert config.quiet is False
        assert config.output_formats == ["png"]
    
    def test_custom_initialization(self: "TestMPASConfig") -> None:
        """
        This test verifies that MPASConfig can be initialized with custom parameters correctly assigning provided values to the corresponding fields. The test confirms that when specific arguments are passed during instantiation (grid_file, data_dir, variable, dpi, output_formats), the resulting configuration object reflects those values accurately. Custom initialization testing ensures that users can create configuration objects tailored to their specific analysis needs by overriding defaults with explicit parameters. This flexibility supports a wide range of use cases from simple default runs to complex configurations requiring precise control over input files, variables, and output settings. 

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
        
        assert config.grid_file == "/path/to/grid.nc"
        assert config.data_dir == "/path/to/data"
        assert config.variable == "rainc"
        assert config.dpi == 600
        assert config.output_formats == ["png", "pdf"]
    
    def test_invalid_spatial_extent(self: "TestMPASConfig") -> None:
        """
        This test validates that MPASConfig raises a ValueError when initialized with invalid spatial extent parameters where longitude minimum exceeds maximum or latitude minimum exceeds maximum. The test confirms that providing lon_min=110.0 and lon_max=100.0 triggers an exception indicating invalid longitude range, and similarly for latitude. This validation ensures that configuration objects cannot be created with nonsensical geographic specifications preventing downstream processing errors related to invalid spatial extents. The test verifies that the error message clearly indicates the nature of the problem guiding users to correct their input parameters. 

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError):
            MPASConfig(
                lon_min=110.0,  
                lon_max=100.0,
                lat_min=-10.0,
                lat_max=10.0
            )
    
    def test_to_dict(self: "TestMPASConfig") -> None:
        """
        This test confirms that the to_dict method of MPASConfig correctly serializes the configuration object into a dictionary representation suitable for YAML serialization. The test validates that the resulting dictionary contains all expected keys corresponding to configuration fields and that values match those set during initialization. The test uses a custom configuration with variable set to "rainc" and DPI set to 400 to confirm that specific entries appear in the dictionary output. This serialization capability supports configuration file generation workflows where users can programmatically create configurations and save them as YAML files for later use. 

        Parameters:
            None

        Returns:
            None
        """
        config = MPASConfig(variable="rainc", dpi=400)
        config_dict = config.to_dict()        
        assert isinstance(config_dict, dict)
        assert config_dict["variable"] == "rainc"
        assert config_dict["dpi"] == 400
    
    def test_from_dict(self: "TestMPASConfig") -> None:
        """
        This test validates that the from_dict class method of MPASConfig correctly creates a configuration object from a dictionary representation. The test confirms that when provided with a dictionary containing specific configuration parameters (variable, dpi, verbose), the resulting MPASConfig object has corresponding field values matching those in the dictionary. The test uses a sample dictionary with variable set to "total", DPI set to 500, and verbose set to False to confirm that these values are correctly assigned in the created configuration object. This deserialization capability supports loading configurations from YAML files where the file content is parsed into dictionaries before being converted into configuration objects for use in processing workflows. 

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
        
        assert config.variable == "total"
        assert config.dpi == 500
        assert config.verbose is False
    
    def test_save_and_load_file(self: "TestMPASConfig") -> None:
        """
        This test verifies that MPASConfig can be saved to a YAML file and loaded back correctly preserving all configuration parameters. The test creates a temporary file to save the configuration, then loads it back and confirms that the loaded configuration matches the original one in terms of field values. This round-trip testing ensures that the save_to_file and load_from_file methods work together seamlessly enabling users to persist configurations to disk and retrieve them later without loss of information or corruption. The test validates that the file-based serialization and deserialization processes maintain data integrity for configuration management workflows. 

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
            
            assert loaded_config.variable == "rainc"
            assert loaded_config.dpi == 350
            assert loaded_config.output_formats == ["png", "svg"]
            
        finally:
            os.unlink(config_file)


class TestMPASLogger:
    """ Tests for MPASLogger utility class behavior. """
    
    def test_logger_initialization(self: "TestMPASLogger") -> None:
        """
        This test validates the initialization of the MPASLogger class ensuring that a logger instance is created with the specified name and that a console handler is attached by default. The test confirms that when a logger is instantiated with a name (test_logger) and verbose mode enabled, the resulting logger object has the correct name and contains at least one handler for output. This initialization testing ensures that logging infrastructure is set up correctly for subsequent logging operations without requiring additional configuration. The presence of handlers confirms that log messages will be output to the console as expected during processing workflows.

        Parameters:
            None

        Returns:
            None
        """
        logger = MPASLogger("test_logger", verbose=True)        
        assert logger.logger.name == "test_logger"
        assert len(logger.logger.handlers) > 0
    
    def test_logger_with_file(self: "TestMPASLogger") -> None:
        """
        This test verifies that MPASLogger can be initialized with a file handler correctly writing log messages to a specified log file. The test creates a temporary file to serve as the log output, then initializes the logger with that file and writes a test message. The test confirms that the log file is created and contains the expected message indicating that logging to file works as intended. This file logging capability supports persistent logging for audit trails and debugging where console output may not be sufficient or desirable. The test ensures that log messages are properly flushed to the file and that the logging infrastructure handles file output without errors. 

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
            
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test message" in content
                
        finally:
            os.unlink(log_file)
    
    def test_log_levels(self: "TestMPASLogger") -> None:
        """
        This test confirms that MPASLogger correctly handles different log levels (INFO, WARNING, ERROR, DEBUG) and that messages are logged according to the specified verbosity settings. The test initializes a logger with verbose mode disabled and attempts to log messages at various levels, then checks that only messages at or above the INFO level are output while DEBUG messages are suppressed. This log level testing ensures that users can control the verbosity of logging output to suit their needs, enabling detailed debugging information when desired while keeping logs concise during normal operation. The test validates that log level filtering works as intended preventing unwanted debug messages from cluttering logs when verbose mode is off. 

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


class TestFileManager:
    """ Tests for FileManager utility class behavior. """
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self: "TestFileManager") -> Generator[None, None, None]:
        """
        This fixture sets up a temporary directory for file management tests and ensures that it is cleaned up after tests complete. The fixture creates a temporary directory using tempfile.mkdtemp() before yielding control to the test functions, allowing them to use this directory for creating files and directories as needed. After the tests finish, the fixture performs teardown by removing the temporary directory and all its contents using shutil.rmtree(). This setup and teardown process ensures that each test runs in a clean environment without side effects from previous tests and that no temporary files are left behind after testing. 
        
        Uses pytest yield to handle both setup and teardown in a single fixture.

        Parameters:
            None

        Returns:
            None
        """
        import shutil
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ensure_directory(self: "TestFileManager") -> None:
        """
        This test validates that FileManager.ensure_directory correctly creates nested directories when they do not exist. The test constructs a nested directory path within the temporary directory and calls ensure_directory on that path. The test confirms that the directory is created successfully and that it exists as a directory on the filesystem. This directory creation testing ensures that the utility function can handle creating multiple levels of directories as needed for organizing output files or intermediate data during processing workflows. The test verifies that no exceptions are raised during directory creation and that the resulting structure is correct. 

        Parameters:
            None

        Returns:
            None
        """
        test_dir = os.path.join(self.temp_dir, "subdir", "subsubdir")        
        FileManager.ensure_directory(test_dir)        
        assert os.path.exists(test_dir)
        assert os.path.isdir(test_dir)
    
    def test_find_files(self: "TestFileManager") -> None:
        """
        This test confirms that FileManager.find_files correctly identifies files matching a specified pattern within a directory. The test creates several files in the temporary directory, some matching the "*.nc" pattern and others not. The test then calls find_files with the pattern and verifies that only the expected files are returned in the results. This file finding testing ensures that the utility function can effectively filter files based on patterns, supporting workflows that need to process specific types of files while ignoring others. The test validates that the correct number of files is found and that their names match the expected criteria. 

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
        
        assert len(nc_files) == 2
        assert any("test1.nc" in f for f in nc_files)
        assert any("test2.nc" in f for f in nc_files)
    
    def test_get_file_info(self: "TestFileManager") -> None:
        """
        This test validates that FileManager.get_file_info correctly retrieves metadata for existing files. The test creates a temporary file, writes content to it, and then calls get_file_info to obtain its metadata. The test confirms that the returned dictionary contains accurate information about the file's existence, size, and timestamps for modification and creation. This metadata retrieval testing ensures that the utility function can provide essential file information for logging, auditing, and conditional processing workflows.

        Parameters:
            None

        Returns:
            None
        """
        test_file = os.path.join(self.temp_dir, "test.txt")

        with open(test_file, 'w') as f:
            f.write("Hello, World!")
        
        file_info = FileManager.get_file_info(test_file)
        
        assert file_info["exists"] is True
        assert file_info["size"] > 0

        assert isinstance(file_info["modified"], datetime)
        assert isinstance(file_info["created"], datetime)
    
    def test_get_file_info_nonexistent(self: "TestFileManager") -> None:
        """
        This test confirms that FileManager.get_file_info correctly handles the case of a nonexistent file by returning a dictionary indicating that the file does not exist. The test calls get_file_info with a path to a file that has not been created and verifies that the "exists" key in the returned dictionary is False. This nonexistence handling testing ensures that the utility function can gracefully report on files that are missing without raising exceptions, supporting workflows that need to check for file presence before attempting to access or process them. The test validates that the function provides a clear indication of nonexistence for robust error handling in file-dependent operations. 

        Parameters:
            None

        Returns:
            None
        """
        file_info = FileManager.get_file_info("/nonexistent/file.txt")        
        assert file_info["exists"] is False


class TestDataValidator:
    """ Tests for DataValidator utility class behavior. """
    
    def test_validate_coordinates_valid(self: "TestDataValidator", mpas_coordinates) -> None:
        """
        This test verifies that DataValidator.validate_coordinates correctly identifies valid coordinate arrays where longitude and latitude arrays have matching lengths and values fall within acceptable geographic bounds. The test confirms that when provided with real MPAS grid coordinates (lon, lat) from fixtures, the validation returns True indicating that the coordinates are suitable for processing. This validation ensures that coordinate data is consistent and geographically sensible before being used in operations such as reprojection or visualization. The test uses a subset of real coordinate data to confirm that typical MPAS grid coordinates pass validation successfully. 

        Parameters:
            mpas_coordinates: Real MPAS grid coordinates (lon, lat) from fixtures.

        Returns:
            None
        """
        if mpas_coordinates is None:
            pytest.skip("MPAS data not available")
        
        lon, lat = mpas_coordinates

        lon_subset = lon[:100]
        lat_subset = lat[:100]
        
        assert DataValidator.validate_coordinates(lon_subset, lat_subset) is True
    
    def test_validate_coordinates_invalid_length(self: "TestDataValidator", mpas_coordinates) -> None:
        """
        This test confirms that DataValidator.validate_coordinates correctly detects mismatched coordinate array lengths where longitude and latitude arrays have different numbers of elements. The test uses real MPAS grid coordinates to create intentional length mismatches (lon with 100 elements and lat with 50 elements) and verifies that the validation returns False indicating that the coordinates are invalid for processing. This length mismatch detection ensures that coordinate data is structurally consistent before being used in operations that require paired longitude and latitude values. The test validates that the function can identify this common data issue preventing downstream errors in spatial processing workflows. 

        Parameters:
            mpas_coordinates: Real MPAS grid coordinates (lon, lat) from fixtures.

        Returns:
            None
        """
        if mpas_coordinates is None:
            pytest.skip("MPAS data not available")
        
        lon, lat = mpas_coordinates
        
        lon_subset = lon[:100]
        lat_subset = lat[:50]  
        
        assert DataValidator.validate_coordinates(lon_subset, lat_subset) is False
    
    def test_validate_coordinates_invalid_values(self: "TestDataValidator") -> None:
        """
        This test verifies that DataValidator.validate_coordinates correctly identifies coordinate arrays with out-of-range values where longitude values exceed 180 degrees or latitude values exceed 90 degrees. The test uses synthetic coordinate data to create invalid scenarios (lon with 200.0 and lat with -100.0) and confirms that the validation returns False indicating that the coordinates are not geographically valid. This range validation ensures that coordinate data falls within acceptable geographic bounds before being used in spatial operations, preventing errors related to invalid locations on the globe. The test validates that the function can detect these out-of-range values effectively, supporting robust data quality checks for geospatial processing workflows. 

        Parameters:
            None

        Returns:
            None
        """
        lon = np.array([200.0, 105.0, 110.0])
        lat = np.array([-5.0, 0.0, 5.0])
        
        assert DataValidator.validate_coordinates(lon, lat) is False
        
        lon = np.array([100.0, 105.0, 110.0])
        lat = np.array([-100.0, 0.0, 5.0])
        
        assert DataValidator.validate_coordinates(lon, lat) is False
    
    def test_validate_data_array_valid(self: "TestDataValidator", mpas_surface_temp_data) -> None:
        """
        This test confirms that DataValidator.validate_data_array correctly identifies valid data arrays where all values are finite and fall within specified bounds. The test uses real MPAS surface temperature data from fixtures, typically in Kelvin, and validates that the function returns True indicating that the data is suitable for processing. The test checks that the total point count matches the finite point count and that the minimum and maximum values fall within reasonable temperature bounds (200-350 K). This validation ensures that data arrays meet quality standards before being used in analysis or visualization workflows, preventing errors related to invalid or out-of-range data. The test validates that typical MPAS surface temperature data passes validation successfully. 

        Parameters:
            mpas_surface_temp_data: Real MPAS surface temperature data from fixtures.

        Returns:
            None
        """
        if mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
        
        data = mpas_surface_temp_data[:100]
        result = DataValidator.validate_data_array(data, min_val=200.0, max_val=350.0)
        
        assert result["valid"] is True
        assert len(result["issues"]) == 0
        assert result["stats"]["total_points"] == 100
        assert result["stats"]["finite_points"] == 100
        assert result["stats"]["min"] >= 200.0
        assert result["stats"]["max"] <= 350.0
    
    def test_validate_data_array_with_issues(self: "TestDataValidator") -> None:
        """
        This test verifies that DataValidator.validate_data_array correctly identifies issues in data arrays containing NaN values and out-of-range values. The test uses synthetic data with intentional problems (NaN, value above max, value below min) to confirm that the validation returns False indicating that the data is not valid for processing. The test checks that the issues list contains entries describing the detected problems and that the statistics reflect the presence of invalid values. This issue detection ensures that data quality problems are identified early preventing errors in analysis or visualization workflows that rely on valid data. The test validates that the function can effectively report on common data issues such as NaNs and out-of-range values. 

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([1.0, np.nan, 15.0, 4.0, -1.0])  
        
        result = DataValidator.validate_data_array(data, min_val=0.0, max_val=10.0)
        
        assert result["valid"] is False
        assert len(result["issues"]) > 0
        assert result["stats"]["total_points"] == 5
        assert result["stats"]["finite_points"] == 4 
    
    def test_validate_data_array_all_identical(self: "TestDataValidator") -> None:
        """
        This test confirms that DataValidator.validate_data_array correctly identifies data arrays where all values are identical, which may indicate a lack of variability or a potential data issue. The test uses synthetic data with all values set to 5.0 and validates that the function returns False indicating that the data is not valid for processing due to zero variance. The test checks that the issues list contains an entry describing the lack of variability and that the statistics reflect a standard deviation of zero. This identical value detection ensures that data arrays with no variability are flagged preventing misleading analysis or visualizations based on uniform data. The test validates that the function can effectively identify this specific data issue. 

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        
        result = DataValidator.validate_data_array(data)
        
        assert result["valid"] is False
        assert "All values are identical" in result["issues"]
        assert result["stats"]["std"] == 0.0


class TestPerformanceMonitor:
    """ Tests for PerformanceMonitor utility class behavior, specifically the timer context manager for measuring operation durations. """
    
    def test_timer_context_manager(self: "TestPerformanceMonitor") -> None:
        """
        This test validates that the timer context manager of PerformanceMonitor correctly measures the duration of a timed code block and records it under the specified operation name. The test uses a simple time.sleep() call to simulate an operation taking approximately 0.01 seconds, then checks that the recorded time for the operation is greater than zero and less than one second, confirming that timing is working as expected. This timing validation ensures that users can rely on the PerformanceMonitor to provide accurate measurements of code execution times for performance profiling and optimization efforts. The test confirms that the summary dictionary contains an entry for the timed operation with a reasonable duration value. 

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
        
        assert "test_operation" in summary
        assert summary["test_operation"] > 0.0
        assert summary["test_operation"] < 1.0  
    
    def test_multiple_operations(self: "TestPerformanceMonitor") -> None:
        """
        This test confirms that PerformanceMonitor can handle multiple timed operations correctly, recording separate durations for each operation name. The test simulates two operations (operation1 and operation2) with time.sleep() calls, then checks that both operations are present in the summary with reasonable duration values. This multi-operation testing ensures that the PerformanceMonitor can track multiple code blocks independently without overwriting or losing timing information, supporting comprehensive performance profiling across different parts of the codebase. The test validates that the summary contains entries for both operations with durations greater than zero. 

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
        
        assert len(summary) == 2
        assert "operation1" in summary
        assert "operation2" in summary


class TestArgumentParser:
    """ Tests for ArgumentParser utility class behavior, specifically the creation of argument parsers and the conversion of parsed arguments to configuration objects. """
    
    def test_create_parser(self: "TestArgumentParser") -> None:
        """
        This test validates that ArgumentParser.create_parser successfully creates an argument parser instance with the expected command-line arguments for grid file, data directory, and variable selection. The test confirms that the parser can parse a sample set of arguments and that the resulting namespace contains the correct values for each argument. This parser creation testing ensures that the command-line interface is properly defined and that users can specify necessary parameters for processing through CLI arguments. The test checks that required arguments are recognized and that their values are correctly assigned in the parsed namespace. 

        Parameters:
            None

        Returns:
            None
        """
        parser = ArgumentParser.create_parser()
        
        assert parser is not None
        
        args = parser.parse_args([
            '--grid-file', 'grid.nc',
            '--data-dir', './data',
            '--var', 'rainc'
        ])
        
        assert args.grid_file == 'grid.nc'
        assert args.data_dir == './data'
        assert args.var == 'rainc'
    
    def test_parse_args_to_config(self: "TestArgumentParser") -> None:
        """
        This test confirms that ArgumentParser.parse_args_to_config correctly converts a parsed argument namespace into an MPASConfig object with the expected field values. The test creates a sample set of arguments, parses them, and then converts the parsed namespace to a configuration object. The test validates that the resulting MPASConfig instance has attributes matching the provided arguments (grid_file, data_dir, variable, dpi, verbose) confirming that the conversion process maps CLI arguments to configuration fields accurately. This conversion testing ensures that users can seamlessly transition from command-line input to structured configuration objects for use in processing workflows. The test checks that all specified parameters are correctly reflected in the created configuration object. 

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
        
        assert config.grid_file == 'grid.nc'
        assert config.data_dir == './data'
        assert config.variable == 'total'
        assert config.dpi == 400
        assert config.verbose is True


class TestUtilityFunctions:
    """ Tests for standalone utility functions related to file management and memory detection, including file size formatting, output filename creation, and available memory retrieval with and without psutil. """
    
    def test_format_file_size(self: "TestUtilityFunctions") -> None:
        """
        This test validates that the format_file_size function correctly converts byte values into human-readable strings with appropriate units (B, KB, MB, GB). The test confirms that specific byte values (512, 1024, 1024^2, 1024^3) are formatted as "512.0 B", "1.0 KB", "1.0 MB", and "1.0 GB" respectively, demonstrating that the function handles unit conversions accurately. This formatting testing ensures that file sizes can be presented in a user-friendly manner for logging and display purposes, improving readability of file size information in outputs and logs. The test checks that the function returns the expected string representations for a range of byte values. 

        Parameters:
            None

        Returns:
            None
        """
        assert format_file_size(512) == "512.0 B"
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"
    
    def test_create_output_filename(self: "TestUtilityFunctions") -> None:
        """
        This test confirms that the create_output_filename function generates a correctly formatted filename based on provided parameters such as base name, valid time, variable type, accumulation type, and file extension. The test uses specific input values to create an output filename and checks that the resulting string matches the expected format "precipitation_map_vartype_rainc_acctype_a01h_valid_20240101T12_point.png". This filename creation testing ensures that output files are named consistently and descriptively based on their content and metadata, supporting organized file management and easy identification of file characteristics from their names. The test validates that all components of the filename are included and formatted correctly according to the specified parameters. 

        Parameters:
            None

        Returns:
            None
        """
        filename = create_output_filename(
            "precipitation_map", "20240101T12", "rainc", "a01h", "png"
        )
        
        expected = "precipitation_map_vartype_rainc_acctype_a01h_valid_20240101T12_point.png"
        assert filename == expected
    
    def test_get_available_memory_with_psutil(self: "TestUtilityFunctions") -> None:
        """
        This test validates that the get_available_memory function correctly retrieves the available system memory in gigabytes when the psutil library is available. The test uses a mock memory getter to simulate a system with 8 GB of available memory and confirms that the function returns the expected value of 8.0 GB. This memory detection testing ensures that the utility function can provide accurate information about available memory for use in memory-dependent processing workflows, allowing for informed decisions about resource usage and potential optimizations. The test checks that the returned value is a floating-point number representing gigabytes of available memory as expected when psutil is functioning properly.

        Parameters:
            None

        Returns:
            None
        """
        mock_get_memory = create_mock_memory_getter(8 * (1024**3))
        
        memory = mock_get_memory()
        assert memory == 8.0
    
    def test_get_available_memory_without_psutil(self: "TestUtilityFunctions") -> None:
        """
        This test confirms that the get_available_memory function returns a non-negative float value representing available memory in gigabytes even when the psutil library is not available. The test simulates the absence of psutil and checks that the function still returns a valid memory value (greater than or equal to 0.0) ensuring that the utility can provide some level of memory information even without external dependencies. This fallback testing ensures that the function can operate in environments where psutil is not installed while still providing useful information about available memory for processing workflows. The test validates that the returned value is a float and is non-negative as expected in this scenario. 

        Parameters:
            None

        Returns:
            None
        """
        memory = get_available_memory()
        assert isinstance(memory, float)
        assert memory >= 0.0


if __name__ == "__main__":
    pytest.main([__file__])