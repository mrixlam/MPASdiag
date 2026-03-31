#!/usr/bin/env python3
"""
MPASdiag Test Suite: Comprehensive tests for file utilities in MPASdiag

This module contains a comprehensive set of unit tests for the file utility functions defined in `mpasdiag.processing.utils_file`. The tests cover various aspects of file management, including finding files with specific patterns, cleaning up old files, formatting file sizes, retrieving available memory, printing system information, creating output filenames, loading configuration files, and validating input files. Each test case is designed to verify the expected behavior of the corresponding utility function under different scenarios, including edge cases and error conditions. The tests utilize temporary directories and files to ensure isolation and avoid side effects on the filesystem. By running this test suite, developers can ensure that the file utilities in MPASdiag are functioning correctly and robustly before integrating them into larger processing workflows.    

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules for testing
import os
import sys
import time
import pytest
import tempfile
from io import StringIO
from typing import Generator
from unittest.mock import patch

from mpasdiag.processing.utils_file import FileManager
from mpasdiag.processing.utils_config import MPASConfig

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestFindFilesComprehensive:
    """ Comprehensive tests for find_files method. """
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self: "TestFindFilesComprehensive") -> Generator[None, None, None]:
        """
        This fixture sets up a temporary directory for testing and ensures cleanup after tests. 

        Parameters:
            None

        Returns:
            None: Fixture yields no value; cleanup is performed after test.
        """
        self.temp_dir = tempfile.mkdtemp()
        yield
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_find_files_recursive(self: "TestFindFilesComprehensive") -> None:
        """
        This test verifies that `find_files` can locate files matching a pattern in nested directories when `recursive=True`. 

        Parameters:
            None

        Returns:
            None: Uses assertions to validate correct behavior.
        """
        subdir1 = os.path.join(self.temp_dir, "sub1")
        subdir2 = os.path.join(self.temp_dir, "sub1", "sub2")
        os.makedirs(subdir2)
        
        with open(os.path.join(self.temp_dir, "file1.nc"), 'w') as f:
            f.write("test")

        with open(os.path.join(subdir1, "file2.nc"), 'w') as f:
            f.write("test")

        with open(os.path.join(subdir2, "file3.nc"), 'w') as f:
            f.write("test")
        
        files = FileManager.find_files(self.temp_dir, "*.nc", recursive=True)
        assert len(files) == pytest.approx(3)
    
    def test_find_files_nonexistent_directory(self: "TestFindFilesComprehensive") -> None:
        """
        This test confirms that `find_files` returns an empty list when the specified directory does not exist. 

        Parameters:
            None

        Returns:
            None: Assertions confirm the returned value is an empty list.
        """
        files = FileManager.find_files("/nonexistent/path", "*.nc")
        assert files == []
    
    def test_find_files_no_matches(self: "TestFindFilesComprehensive") -> None:
        """
        This test ensures that `find_files` returns an empty list when no files match the specified pattern in the directory. 

        Parameters:
            None

        Returns:
            None: Validated via assertions.
        """
        with open(os.path.join(self.temp_dir, "test.txt"), 'w') as f:
            f.write("test")
        
        files = FileManager.find_files(self.temp_dir, "*.nc")
        assert files == []


class TestCleanupFiles:
    """ Tests for cleanup_files method. """
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self: "TestCleanupFiles") -> Generator[None, None, None]:
        """
        This fixture creates a temporary directory for testing file cleanup and ensures it is removed after tests are completed. 

        Parameters:
            None

        Returns:
            None: Fixture yields no value; cleanup happens after the test.
        """
        self.temp_dir = tempfile.mkdtemp()
        yield
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cleanup_old_files(self: "TestCleanupFiles") -> None:
        """
        This test verifies that `cleanup_files` correctly identifies and deletes files older than a specified number of days. 

        Parameters:
            None

        Returns:
            None: Verified through assertions on file existence and return value.
        """
        old_file = os.path.join(self.temp_dir, "old.tmp")
        new_file = os.path.join(self.temp_dir, "new.tmp")
        
        with open(old_file, 'w') as f:
            f.write("old")
        
        old_time = time.time() - (10 * 24 * 60 * 60)
        os.utime(old_file, (old_time, old_time))
        
        with open(new_file, 'w') as f:
            f.write("new")
        
        deleted_count = FileManager.cleanup_files(self.temp_dir, "*.tmp", older_than_days=7)
        
        assert deleted_count == pytest.approx(1)
        assert not os.path.exists(old_file)
        assert os.path.exists(new_file)
    
    def test_cleanup_no_old_files(self: "TestCleanupFiles") -> None:
        """
        This test confirms that `cleanup_files` does not delete files that are newer than the specified age threshold. 

        Parameters:
            None

        Returns:
            None: Confirmed via assertions.
        """
        new_file = os.path.join(self.temp_dir, "new.tmp")

        with open(new_file, 'w') as f:
            f.write("new")
        
        deleted_count = FileManager.cleanup_files(self.temp_dir, "*.tmp", older_than_days=7)
        
        assert deleted_count == pytest.approx(0)
        assert os.path.exists(new_file)


class TestFormatFileSize:
    """ Tests for format_file_size method. """
    
    def test_format_bytes(self: "TestFormatFileSize") -> None:
        """
        This test verifies that `format_file_size` correctly formats byte-scale values into `B` units. 

        Parameters:
            None

        Returns:
            None: Verified by assertions comparing expected strings.
        """
        assert FileManager.format_file_size(500) == "500.0 B"
        assert FileManager.format_file_size(1023) == "1023.0 B"
    
    def test_format_kilobytes(self: "TestFormatFileSize") -> None:
        """
        This test confirms that `format_file_size` formats values in the kilobyte range into `KB` units with one decimal place. 

        Parameters:
            None

        Returns:
            None: Verified via assertions on returned strings.
        """
        assert FileManager.format_file_size(1024) == "1.0 KB"
        assert FileManager.format_file_size(2048) == "2.0 KB"
    
    def test_format_megabytes(self: "TestFormatFileSize") -> None:
        """
        This test ensures that `format_file_size` correctly formats megabyte-scale values into `MB` units. 

        Parameters:
            None

        Returns:
            None: Verified via equality assertions.
        """
        assert FileManager.format_file_size(1024 * 1024) == "1.0 MB"
        assert FileManager.format_file_size(5 * 1024 * 1024) == "5.0 MB"
    
    def test_format_gigabytes(self: "TestFormatFileSize") -> None:
        """
        This test validates that `format_file_size` formats gigabyte-range values into `GB` units correctly.

        Parameters:
            None

        Returns:
            None: Verified via assertion.
        """
        assert FileManager.format_file_size(1024 * 1024 * 1024) == "1.0 GB"
    
    def test_format_terabytes(self: "TestFormatFileSize") -> None:
        """
        This test checks that `format_file_size` can handle terabyte-scale values and formats them into `TB` units as expected.

        Parameters:
            None

        Returns:
            None: Verified via assertion on the return value.
        """
        assert FileManager.format_file_size(1024 * 1024 * 1024 * 1024) == "1.0 TB"
    
    def test_format_petabytes(self: "TestFormatFileSize") -> None:
        """
        This test confirms that `format_file_size` can handle petabyte-scale values and formats them into `PB` units correctly. 

        Parameters:
            None

        Returns:
            None: Validated using assertions.
        """
        size_pb = 1024 * 1024 * 1024 * 1024 * 1024
        result = FileManager.format_file_size(size_pb)
        assert "PB" in result
        assert result.startswith("1.0")


class TestGetAvailableMemory:
    """ Tests for get_available_memory method. """
    
    def test_get_available_memory_success(self: "TestGetAvailableMemory") -> None:
        """
        This test verifies that `get_available_memory` returns a non-negative float representing available memory in GB when `psutil` is available. 

        Parameters:
            None

        Returns:
            None: Verified through type and numeric assertions.
        """
        memory = FileManager.get_available_memory()
        assert isinstance(memory, float)
        assert memory >= 0.0
    
    def test_get_available_memory_import_error(self: "TestGetAvailableMemory") -> None:
        """
        This test ensures that `get_available_memory` handles the absence of `psutil` gracefully by returning 0.0 and not raising an exception.

        Parameters:
            None

        Returns:
            None: Behavior asserted by checking returned value equals 0.0.
        """
        import sys
        psutil_backup = sys.modules.get('psutil')
        
        try:
            if 'psutil' in sys.modules:
                del sys.modules['psutil']
            
            with patch.dict('sys.modules', {'psutil': None}):
                import importlib
                from mpasdiag.processing import utils_file
                importlib.reload(utils_file)               
                memory = utils_file.FileManager.get_available_memory()
                assert memory == pytest.approx(0.0), "Should return 0.0 when psutil unavailable"
        finally:
            if psutil_backup is not None:
                sys.modules['psutil'] = psutil_backup
            import importlib
            from mpasdiag.processing import utils_file
            importlib.reload(utils_file)


class TestPrintSystemInfo:
    """ Tests for print_system_info method. """
    
    def test_print_system_info_output(self: "TestPrintSystemInfo") -> None:
        """
        This test verifies that `print_system_info` outputs expected system information to stdout. The test captures stdout and checks for key phrases that indicate the presence of system information such as Python version, platform details, current working directory, and available memory.

        Parameters:
            None

        Returns:
            None: Verified using assertions on captured stdout.
        """
        captured_output = StringIO()
        
        with patch('sys.stdout', new=captured_output):
            FileManager.print_system_info()
        
        output = captured_output.getvalue()
        
        assert "System Information" in output
        assert "Python version:" in output
        assert "Platform:" in output
        assert "Current working directory:" in output
        assert "Available memory:" in output


class TestCreateOutputFilename:
    """ Tests for create_output_filename method. """
    
    def test_create_output_filename_default_extension(self: "TestCreateOutputFilename") -> None:
        """
        This test validates that `create_output_filename` generates a filename with the default `.png` extension when no custom extension is provided. The test checks that the returned filename follows the expected naming convention based on the input parameters. 

        Parameters:
            None

        Returns:
            None: Verified via equality assertion.
        """
        filename = FileManager.create_output_filename(
            base_name="test_run",
            time_str="20260103T12",
            var_name="temperature",
            accum_period="1h"
        )
        
        expected = "test_run_vartype_temperature_acctype_1h_valid_20260103T12_point.png"
        assert filename == expected
    
    def test_create_output_filename_custom_extension(self: "TestCreateOutputFilename") -> None:
        """
        This test ensures that `create_output_filename` correctly incorporates a custom file extension when provided. The test calls the method with `extension='pdf'` and verifies that the resulting filename ends with `.pdf` and follows the expected naming pattern. 

        Parameters:
            None

        Returns:
            None: Verified via equality assertion.
        """
        filename = FileManager.create_output_filename(
            base_name="exp01",
            time_str="20260103T18",
            var_name="wind",
            accum_period="6h",
            extension="pdf"
        )
        
        expected = "exp01_vartype_wind_acctype_6h_valid_20260103T18_point.pdf"
        assert filename == expected
    
    def test_create_output_filename_netcdf_extension(self: "TestCreateOutputFilename") -> None:
        """
        This test confirms that `create_output_filename` can generate filenames with a `.nc` extension when specified. The test provides `extension='nc'` and checks that the output filename is correctly formatted according to the input parameters and ends with `.nc`. 

        Parameters:
            None

        Returns:
            None: Verified via equality assertion.
        """
        filename = FileManager.create_output_filename(
            base_name="model_output",
            time_str="20260103T00",
            var_name="precip",
            accum_period="24h",
            extension="nc"
        )
        
        expected = "model_output_vartype_precip_acctype_24h_valid_20260103T00_point.nc"
        assert filename == expected


class TestLoadConfigFile:
    """ Tests for load_config_file method. """
    
    def test_load_config_file_not_found(self: "TestLoadConfigFile") -> None:
        """
        This test verifies that `load_config_file` handles the case of a missing configuration file gracefully by returning a default `MPASConfig` instance and printing an appropriate error message. The test attempts to load a non-existent file and captures stdout to confirm that the expected messages about the missing file and fallback to default configuration are printed. 

        Parameters:
            None

        Returns:
            None: Verified by type check and captured stdout inspection.
        """
        captured_output = StringIO()
        
        with patch('sys.stdout', new=captured_output):
            config = FileManager.load_config_file("/nonexistent/config.yaml")
        
        assert isinstance(config, MPASConfig)
        
        output = captured_output.getvalue()
        assert "Configuration file not found" in output
        assert "Using default configuration" in output
    
    def test_load_config_file_parse_error(self: "TestLoadConfigFile") -> None:
        """
        This test ensures that `load_config_file` can handle YAML parsing errors gracefully by returning a default `MPASConfig` instance and printing an appropriate error message. The test creates a temporary file with invalid YAML content, attempts to load it, and captures stdout to confirm that the expected error messages about loading the configuration file and using the default configuration are printed. 

        Parameters:
            None

        Returns:
            None: Validated through return type and stdout contents.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: {unclosed")
            config_file = f.name
        
        try:
            captured_output = StringIO()
            
            with patch('sys.stdout', new=captured_output):
                config = FileManager.load_config_file(config_file)
            
            assert isinstance(config, MPASConfig)
            
            output = captured_output.getvalue()
            assert "Error loading configuration file" in output
            assert "Using default configuration" in output
        finally:
            os.unlink(config_file)
    
    def test_load_config_file_success(self: "TestLoadConfigFile") -> None:
        """
        This test confirms that `load_config_file` can successfully load a valid YAML configuration file and return an instance of `MPASConfig` populated with the expected values. The test creates a temporary YAML file with a simple configuration, loads it using the method, and asserts that the returned object is an instance of `MPASConfig`. 

        Parameters:
            None

        Returns:
            None: Verified by isinstance type check.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("grid_file: test.nc\n")
            config_file = f.name
        
        try:
            config = FileManager.load_config_file(config_file)
            assert isinstance(config, MPASConfig)
        finally:
            os.unlink(config_file)


class TestValidateInputFiles:
    """ Tests for validate_input_files method. """
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self: "TestValidateInputFiles") -> Generator[None, None, None]:
        """
        This fixture sets up a temporary directory structure with a grid file and a data directory for testing the validation of input files. It creates a grid file and an empty data directory before yielding control to the test functions, and ensures that all temporary files and directories are cleaned up after the tests are completed. 

        Parameters:
            None

        Returns:
            None: Fixture yields no value; performs cleanup after the test.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.grid_file = os.path.join(self.temp_dir, "grid.nc")
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(self.data_dir)
        
        with open(self.grid_file, 'w') as f:
            f.write("test grid data")
        
        yield
        
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_input_files_success(self: "TestValidateInputFiles") -> None:
        """
        This test verifies that `validate_input_files` returns True when both the grid file and at least one diagnostic file are present in the specified data directory. The test creates a valid grid file and a diagnostic file, then calls the validation method and asserts that it returns True, indicating successful validation of input files. 

        Parameters:
            None

        Returns:
            None: Confirmed via assertion that result is truthy.
        """
        diag_file = os.path.join(self.data_dir, "diag.2024-01-01_00.00.00.nc")

        with open(diag_file, 'w') as f:
            f.write("test diag data")
        
        config = MPASConfig()
        config.grid_file = self.grid_file
        config.data_dir = self.data_dir
        
        result = FileManager.validate_input_files(config)
        assert result
    
    def test_validate_input_files_missing_grid_file(self: "TestValidateInputFiles") -> None:
        """
        This test confirms that `validate_input_files` returns False when the specified `grid_file` does not exist. The test sets `config.grid_file` to a nonexistent path while providing a valid data directory with diagnostic files, then asserts that the validation method returns False and prints an appropriate message about the missing grid file.

        Parameters:
            None

        Returns:
            None: Verified by checking return value and stdout.
        """
        config = MPASConfig()
        config.grid_file = "/nonexistent/grid.nc"
        config.data_dir = self.data_dir
        
        diag_file = os.path.join(self.data_dir, "diag.2024-01-01_00.00.00.nc")

        with open(diag_file, 'w') as f:
            f.write("test")
        
        captured_output = StringIO()
        
        with patch('sys.stdout', new=captured_output):
            result = FileManager.validate_input_files(config)
        
        assert result is False
        output = captured_output.getvalue()
        assert "Grid file not found" in output
    
    def test_validate_input_files_missing_data_dir(self: "TestValidateInputFiles") -> None:
        """
        This test ensures that `validate_input_files` returns False when the specified `data_dir` does not exist. The test sets `config.data_dir` to a nonexistent path while providing a valid grid file, then asserts that the validation method returns False and prints an appropriate message about the missing data directory. 

        Parameters:
            None

        Returns:
            None: Verified via assertions on the returned boolean and stdout.
        """
        config = MPASConfig()
        config.grid_file = self.grid_file
        config.data_dir = "/nonexistent/data"
        
        captured_output = StringIO()
        
        with patch('sys.stdout', new=captured_output):
            result = FileManager.validate_input_files(config)
        
        assert result is False
        output = captured_output.getvalue()
        assert "Data directory not found" in output
    
    def test_validate_input_files_data_path_is_file(self: "TestValidateInputFiles") -> None:
        """
        This test verifies that `validate_input_files` returns False when the `data_dir` path is actually a file instead of a directory. The test creates a regular file at the location specified by `config.data_dir`, then calls the validation method and asserts that it returns False while printing an appropriate message indicating that the data path is not a directory. 

        Parameters:
            None

        Returns:
            None: Confirmed through assertions and stdout inspection.
        """
        data_file = os.path.join(self.temp_dir, "not_a_directory.txt")

        with open(data_file, 'w') as f:
            f.write("test")
        
        config = MPASConfig()
        config.grid_file = self.grid_file
        config.data_dir = data_file
        
        captured_output = StringIO()
        
        with patch('sys.stdout', new=captured_output):
            result = FileManager.validate_input_files(config)
        
        assert result is False
        output = captured_output.getvalue()
        assert "Data path is not a directory" in output
    
    def test_validate_input_files_no_diagnostic_files(self: "TestValidateInputFiles") -> None:
        """
        This test confirms that `validate_input_files` returns False when no diagnostic files are found in the specified data directory. The test sets up a valid grid file and an empty data directory, then calls the validation method and asserts that it returns False while printing a message indicating that no diagnostic files were found.

        Parameters:
            None

        Returns:
            None: Verified by return value and captured stdout.
        """
        config = MPASConfig()
        config.grid_file = self.grid_file
        config.data_dir = self.data_dir
        
        captured_output = StringIO()
        
        with patch('sys.stdout', new=captured_output):
            result = FileManager.validate_input_files(config)
        
        assert result is False
        output = captured_output.getvalue()
        assert "No diagnostic files found" in output
    
    def test_validate_input_files_diag_subdirectory(self: "TestValidateInputFiles") -> None:
        """
        This test verifies that `validate_input_files` can successfully find diagnostic files located in a `diag/` subdirectory within the specified data directory. The test creates a `diag/` subdirectory, places a diagnostic file inside it, and asserts that the validation method returns True, indicating that it correctly identified the presence of diagnostic files in the expected subdirectory structure. 

        Parameters:
            None

        Returns:
            None: Confirmed via assertion on the return value.
        """
        diag_subdir = os.path.join(self.data_dir, "diag")
        os.makedirs(diag_subdir)
        
        diag_file = os.path.join(diag_subdir, "diag.2024-01-01_00.00.00.nc")

        with open(diag_file, 'w') as f:
            f.write("test diag data")
        
        config = MPASConfig()
        config.grid_file = self.grid_file
        config.data_dir = self.data_dir
        
        result = FileManager.validate_input_files(config)
        assert result
    
    def test_validate_input_files_mpasout_subdirectory(self: "TestValidateInputFiles") -> None:
        """
        This test verifies that `validate_input_files` can successfully find diagnostic files located in a `mpasout/` subdirectory within the specified data directory. The test creates a `mpasout/` subdirectory, places a diagnostic file inside it, and asserts that the validation method returns True, indicating that it correctly identified the presence of diagnostic files in the expected subdirectory structure.

        Parameters:
            None

        Returns:
            None: Confirmed by assertion that `result` is truthy.
        """
        mpasout_subdir = os.path.join(self.data_dir, "mpasout")
        os.makedirs(mpasout_subdir)
        
        diag_file = os.path.join(mpasout_subdir, "diag.2024-01-01_00.00.00.nc")
        with open(diag_file, 'w') as f:
            f.write("test diag data")
        
        config = MPASConfig()
        config.grid_file = self.grid_file
        config.data_dir = self.data_dir
        
        result = FileManager.validate_input_files(config)
        assert result
    
    def test_validate_input_files_recursive_search(self: "TestValidateInputFiles") -> None:
        """
        This test confirms that `validate_input_files` can successfully find diagnostic files located in nested subdirectories within the specified data directory. The test creates a nested directory structure (e.g., `nested/deep/`), places a diagnostic file inside the deepest subdirectory, and asserts that the validation method returns True, indicating that it correctly performed a recursive search for diagnostic files. 

        Parameters:
            None

        Returns:
            None: Verified via assertion that result is True.
        """
        nested_dir = os.path.join(self.data_dir, "nested", "deep")
        os.makedirs(nested_dir)
        
        diag_file = os.path.join(nested_dir, "diag.2024-01-01_00.00.00.nc")

        with open(diag_file, 'w') as f:
            f.write("test diag data")
        
        config = MPASConfig()
        config.grid_file = self.grid_file
        config.data_dir = self.data_dir
        
        result = FileManager.validate_input_files(config)
        assert result is True
    
    def test_validate_input_files_no_grid_file_specified(self: "TestValidateInputFiles") -> None:
        """
        This test verifies that `validate_input_files` returns False when the `grid_file` is not specified in the configuration. The test sets up a valid data directory with diagnostic files but leaves `config.grid_file` unset, then asserts that the validation method returns False and prints an appropriate message about the missing grid file configuration. 

        Parameters:
            None

        Returns:
            None: Verified via assertions on the return value and stdout.
        """
        config = MPASConfig()
        config.data_dir = self.data_dir
        
        diag_file = os.path.join(self.data_dir, "diag.2024-01-01_00.00.00.nc")

        with open(diag_file, 'w') as f:
            f.write("test")
        
        captured_output = StringIO()
        
        with patch('sys.stdout', new=captured_output):
            result = FileManager.validate_input_files(config)
        
        assert result is False
        output = captured_output.getvalue()
        assert "Grid file not specified" in output
    
    def test_validate_input_files_no_data_dir_specified(self: "TestValidateInputFiles") -> None:
        """
        This test confirms that `validate_input_files` returns False when the `data_dir` is not specified in the configuration. The test sets up a valid grid file but leaves `config.data_dir` unset, then asserts that the validation method returns False and prints an appropriate message about the missing data directory configuration. 

        Parameters:
            None

        Returns:
            None: Confirmed through assertions and stdout inspection.
        """
        config = MPASConfig()
        config.grid_file = self.grid_file
        
        captured_output = StringIO()
        
        with patch('sys.stdout', new=captured_output):
            result = FileManager.validate_input_files(config)
        
        assert result is False
        output = captured_output.getvalue()
        assert "Data directory not specified" in output


if __name__ == "__main__":
    pytest.main([__file__])