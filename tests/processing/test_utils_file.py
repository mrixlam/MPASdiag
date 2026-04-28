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


class TestGetAvailableMemory:
    """ Tests for get_available_memory method. """
    
    
    def test_get_available_memory_import_error(self: 'TestGetAvailableMemory') -> None:
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


class TestValidateInputFiles:
    """ Tests for validate_input_files method. """
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self: 'TestValidateInputFiles') -> Generator[None, None, None]:
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
    
    def test_validate_input_files_success(self: 'TestValidateInputFiles') -> None:
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
    
    def test_validate_input_files_missing_grid_file(self: 'TestValidateInputFiles') -> None:
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
    
    
    def test_validate_input_files_data_path_is_file(self: 'TestValidateInputFiles') -> None:
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
    
    def test_validate_input_files_no_diagnostic_files(self: 'TestValidateInputFiles') -> None:
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
    
    def test_validate_input_files_diag_subdirectory(self: 'TestValidateInputFiles') -> None:
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
    
    def test_validate_input_files_mpasout_subdirectory(self: 'TestValidateInputFiles') -> None:
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
    
    def test_validate_input_files_recursive_search(self: 'TestValidateInputFiles') -> None:
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
    
    def test_validate_input_files_no_grid_file_specified(self: 'TestValidateInputFiles') -> None:
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
    

    def test_validate_no_data_dir_specified(self: 'TestValidateInputFiles') -> None:
        config = MPASConfig()
        config.grid_file = self.grid_file
        config.data_dir = ""
        captured_output = StringIO()
        with patch('sys.stdout', new=captured_output):
            result = FileManager.validate_input_files(config)
        assert result is False
        assert "Data directory not specified" in captured_output.getvalue()

    def test_validate_data_dir_not_found(self: 'TestValidateInputFiles') -> None:
        config = MPASConfig()
        config.grid_file = self.grid_file
        config.data_dir = "/nonexistent/dir/does/not/exist"
        captured_output = StringIO()
        with patch('sys.stdout', new=captured_output):
            result = FileManager.validate_input_files(config)
        assert result is False
        assert "Data directory not found" in captured_output.getvalue()


class TestEnsureDirectory:
    """Tests for FileManager.ensure_directory."""

    @pytest.fixture(autouse=True)
    def setup(self) -> Generator[None, None, None]:
        self.temp_dir = tempfile.mkdtemp()
        yield
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_creates_new_directory(self) -> None:
        new_path = os.path.join(self.temp_dir, "new", "nested", "dir")
        assert not os.path.exists(new_path)
        FileManager.ensure_directory(new_path)
        assert os.path.isdir(new_path)

    def test_no_error_if_already_exists(self) -> None:
        FileManager.ensure_directory(self.temp_dir)
        FileManager.ensure_directory(self.temp_dir)
        assert os.path.isdir(self.temp_dir)


class TestGetFileInfo:
    """Tests for FileManager.get_file_info."""

    @pytest.fixture(autouse=True)
    def setup(self) -> Generator[None, None, None]:
        self.temp_dir = tempfile.mkdtemp()
        yield
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_nonexistent_file_returns_exists_false(self) -> None:
        result = FileManager.get_file_info("/nonexistent/file.nc")
        assert result == {"exists": False}

    def test_existing_file_returns_metadata(self) -> None:
        from datetime import datetime as dt
        filepath = os.path.join(self.temp_dir, "test.nc")
        with open(filepath, "w") as f:
            f.write("data")
        result = FileManager.get_file_info(filepath)
        assert result["exists"] is True
        assert isinstance(result["size"], int)
        assert isinstance(result["size_mb"], float)
        assert isinstance(result["modified"], dt)
        assert isinstance(result["created"], dt)
        assert result["size"] > 0


class TestCleanupFiles:
    """Tests for FileManager.cleanup_files."""

    @pytest.fixture(autouse=True)
    def setup(self) -> Generator[None, None, None]:
        self.temp_dir = tempfile.mkdtemp()
        yield
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_deletes_old_files_and_returns_count(self) -> None:
        old_file = os.path.join(self.temp_dir, "old.tmp")
        new_file = os.path.join(self.temp_dir, "new.tmp")
        for path in (old_file, new_file):
            with open(path, "w") as f:
                f.write("x")
        eight_days_ago = time.time() - (8 * 24 * 3600)
        os.utime(old_file, (eight_days_ago, eight_days_ago))
        count = FileManager.cleanup_files(self.temp_dir, "*.tmp", older_than_days=7)
        assert count == 1
        assert not os.path.exists(old_file)
        assert os.path.exists(new_file)

    def test_no_files_deleted_when_all_fresh(self) -> None:
        fresh_file = os.path.join(self.temp_dir, "fresh.tmp")
        with open(fresh_file, "w") as f:
            f.write("x")
        count = FileManager.cleanup_files(self.temp_dir, "*.tmp", older_than_days=7)
        assert count == 0
        assert os.path.exists(fresh_file)


class TestFormatFileSize:
    """Tests for FileManager.format_file_size."""

    @pytest.mark.parametrize("size_bytes,expected", [
        (500, "500.0 B"),
        (1536, "1.5 KB"),
        (1572864, "1.5 MB"),
        (1610612736, "1.5 GB"),
        (1649267441664, "1.5 TB"),
        (1688849860263936, "1.5 PB"),
    ])
    def test_formats_correctly(self, size_bytes: int, expected: str) -> None:
        assert FileManager.format_file_size(size_bytes) == expected


class TestGetAvailableMemoryPsutilPresent:
    """Tests for FileManager.get_available_memory when psutil is available."""

    def test_returns_non_negative_float(self) -> None:
        result = FileManager.get_available_memory()
        assert isinstance(result, float)
        assert result >= 0.0


class TestPrintSystemInfo:
    """Tests for FileManager.print_system_info."""

    def test_prints_system_information(self) -> None:
        captured = StringIO()
        with patch('sys.stdout', new=captured):
            FileManager.print_system_info()
        output = captured.getvalue()
        assert "Python version" in output
        assert "Platform" in output
        assert "Available memory" in output


class TestCreateOutputFilename:
    """Tests for FileManager.create_output_filename."""

    def test_returns_formatted_filename(self) -> None:
        result = FileManager.create_output_filename(
            "mpas", "20240101_000000", "precip", "24hr"
        )
        assert result == "mpas_vartype_precip_acctype_24hr_valid_20240101_000000_point.png"

    def test_custom_extension(self) -> None:
        result = FileManager.create_output_filename(
            "mpas", "20240101_000000", "temp", "6hr", extension="pdf"
        )
        assert result.endswith(".pdf")


class TestLoadConfigFile:
    """Tests for FileManager.load_config_file."""

    @pytest.fixture(autouse=True)
    def setup(self) -> Generator[None, None, None]:
        self.temp_dir = tempfile.mkdtemp()
        yield
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_nonexistent_file_returns_default_config(self) -> None:
        captured = StringIO()
        with patch('sys.stdout', new=captured):
            result = FileManager.load_config_file("/nonexistent/config.yaml")
        assert isinstance(result, MPASConfig)
        assert "Error loading configuration file" in captured.getvalue()

    def test_existing_file_delegates_to_load_from_file(self) -> None:
        from unittest.mock import patch as mpatch
        config_path = os.path.join(self.temp_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("# placeholder\n")
        expected = MPASConfig()
        with mpatch.object(MPASConfig, "load_from_file", return_value=expected) as mock_load:
            result = FileManager.load_config_file(config_path)
        mock_load.assert_called_once_with(config_path)
        assert result is expected


class TestModuleLevelPrintSystemInfo:
    """Tests for the module-level print_system_info convenience function."""

    def test_delegates_to_file_manager(self) -> None:
        from mpasdiag.processing.utils_file import print_system_info as module_print_info
        captured = StringIO()
        with patch('sys.stdout', new=captured):
            module_print_info()
        assert "Python version" in captured.getvalue()


if __name__ == "__main__":
    pytest.main([__file__])
