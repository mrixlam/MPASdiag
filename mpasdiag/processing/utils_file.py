#!/usr/bin/env python3

"""
MPASdiag Core Processing Module: File Management Utilities

This module provides a set of utility functions for handling file and directory operations in the context of MPAS diagnostic processing. It includes methods for ensuring output directories exist, finding files based on glob patterns, retrieving file metadata, cleaning up old files, formatting file sizes, checking available memory, printing system information, creating standardized output filenames, loading configuration files, and validating input files. These utilities are designed to support the core processing workflows of MPAS diagnostics by providing robust and reusable functions for common file management tasks. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

from .utils_config import MPASConfig
from .constants import DIAG_GLOB


class FileManager:
    """ Utility class for handling file and directory operations in MPAS diagnostic processing. """
    
    @staticmethod
    def ensure_directory(directory: str) -> None:
        """
        This method ensures that the specified directory exists by creating it if it does not already exist. It uses pathlib.Path.mkdir with parents=True to create any necessary parent directories and exist_ok=True to avoid raising an error if the directory already exists. This utility is commonly used to prepare output directories for saving diagnostic results, ensuring that the processing workflow can write files without encountering "directory not found" errors. By centralizing this functionality, it promotes code reuse and simplifies directory management across different parts of the MPAS diagnostic processing codebase. 

        Parameters: 
            directory (str): Path to the directory to ensure exists, can be absolute or relative. 

        Returns:
            None
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def find_files(directory: str, 
                   pattern: str = "*.nc",
                   recursive: bool = False) -> List[str]:
        """
        This method searches for files in the specified directory that match a given glob pattern, with an option to perform a recursive search through subdirectories. It uses pathlib.Path.glob for non-recursive searches and pathlib.Path.rglob for recursive searches. The method returns a sorted list of matching file paths as strings. If the specified directory does not exist, it returns an empty list. This utility is essential for locating input diagnostic files based on common naming patterns, allowing the processing workflow to easily gather the necessary files for analysis. By providing both non-recursive and recursive search options, it offers flexibility in how users organize their input data directories. 

        Parameters:
            directory (str): Path to the directory to search for files, can be absolute or relative.
            pattern (str): Glob pattern to match files such as "*.nc" or "diag_*.nc" (default: "*.nc").
            recursive (bool): If True, search subdirectories recursively; if False, search only the specified directory (default: False). 

        Returns:
            List[str]: Sorted list of file paths matching the pattern, or an empty list if the directory does not exist or no files match. 
        """
        path = Path(directory)

        if not path.exists():
            return []

        if recursive:
            return sorted([str(p) for p in path.rglob(pattern)])
        return sorted([str(p) for p in path.glob(pattern)])
    
    @staticmethod
    def get_file_info(filepath: str) -> Dict[str, Any]:
        """
        This method retrieves metadata information about a specified file, including its existence, size in bytes and megabytes, and timestamps for last modification and creation. It uses pathlib.Path to access file properties and returns a dictionary containing this information. If the file does not exist, it returns a dictionary indicating that the file does not exist. This utility is useful for inspecting input files before processing, allowing users to verify that files are present and to understand their characteristics such as size and modification times. This information can assist with troubleshooting issues related to missing or unexpectedly large files in MPAS diagnostic workflows. 

        Parameters: 
            filepath (str): Path to the file to retrieve information about, can be absolute or relative. 

        Returns:
            Dict[str, Any]: Dictionary containing file metadata including existence, size in bytes and megabytes, and modification/creation timestamps. If the file does not exist, returns {"exists": False}. 
        """
        path = Path(filepath)

        if not path.exists():
            return {"exists": False}

        stat = path.stat()

        return {
            "exists": True,
            "size": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "created": datetime.fromtimestamp(stat.st_ctime),
        }
    
    @staticmethod
    def cleanup_files(directory: str, 
                      pattern: str = "*.tmp",
                      older_than_days: int = 7) -> int:
        """
        This method deletes files in the specified directory that match a given glob pattern and have a modification time older than a specified number of days. It calculates a cutoff time based on the current time minus the older_than_days parameter and iterates through matching files to check their modification times. If a file's modification time is older than the cutoff, it is deleted using pathlib.Path.unlink. The method keeps a count of how many files were successfully deleted and returns this count at the end. This utility is useful for cleaning up temporary or backup files that may accumulate over time during MPAS diagnostic processing, helping to manage disk space and maintain an organized file system. By allowing users to specify both the pattern and age threshold, it provides flexibility in how cleanup operations are performed. 

        Parameters:
            directory (str): Path to the directory to clean up, can be absolute or relative.
            pattern (str): Glob pattern to match files for deletion such as "*.tmp" or "backup_*.nc" (default: "*.tmp").
            older_than_days (int): Number of days to use as a threshold for deleting files based on their modification time (default: 7). 

        Returns:
            int: Count of files that were deleted based on the specified criteria. 
        """
        path = Path(directory)
        cutoff_time = datetime.now() - timedelta(days=older_than_days)
        deleted_count = 0

        for filepath in path.glob(pattern):
            if datetime.fromtimestamp(filepath.stat().st_mtime) < cutoff_time:
                filepath.unlink()
                deleted_count += 1

        return deleted_count
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        This method converts a file size in bytes into a human-readable string format with appropriate units (B, KB, MB, GB, TB). It iteratively divides the size by 1024 to determine the correct unit and formats the resulting size with one decimal place. If the size exceeds the largest unit (TB), it defaults to PB. This utility is helpful for displaying file sizes in a more understandable format when reporting on input files or output products in MPAS diagnostic processing workflows, allowing users to quickly grasp the scale of file sizes without needing to interpret large byte values. 

        Parameters:
            size_bytes (int): File size in bytes to be formatted into a human-readable string. 

        Returns:
            str: Formatted file size string with appropriate units (e.g., "1.5 MB", "200 KB"). 
        """
        size_float = float(size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_float < 1024.0:
                return f"{size_float:.1f} {unit}"
            size_float /= 1024.0
        return f"{size_float:.1f} PB"
    
    @staticmethod
    def get_available_memory() -> float:
        """
        This method retrieves the available system memory in gigabytes using the psutil library. It attempts to import psutil and access the virtual_memory().available attribute to get the available memory in bytes, which it then converts to gigabytes by dividing by 1024^3. If psutil is not installed or if there is an error accessing memory information, it catches the ImportError and returns 0.0 to indicate that available memory cannot be determined. This utility is useful for assessing system resources before running MPAS diagnostic processing workflows, allowing users to understand how much memory is available for processing large datasets and potentially adjust their workflow or input data accordingly. 

        Parameters:
            None

        Returns:
            float: Available system memory in gigabytes, or 0.0 if memory information cannot be retrieved. 
        """
        try:
            import psutil

            return psutil.virtual_memory().available / (1024**3)
        except ImportError:
            return 0.0
    
    @staticmethod
    def print_system_info() -> None:
        """
        This method prints basic system information to the console, including the Python version, platform, current working directory, and available memory. It uses the sys and os modules to retrieve this information and the get_available_memory method to report on available system memory. This utility is helpful for providing context about the environment in which MPAS diagnostic processing is being run, allowing users to verify that they are using the expected Python version and platform, and to understand the resources available for processing. By centralizing this information in a single method, it promotes consistency in how system information is reported across different parts of the MPAS diagnostic codebase. 

        Parameters:
            None

        Returns:
            None
        """
        print("=== System Information ===")
        print(f"Python version: {sys.version}")
        print(f"Platform: {sys.platform}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Available memory: {FileManager.get_available_memory():.1f} GB")
        print("=" * 30)
    
    @staticmethod
    def create_output_filename(base_name: str, 
                               time_str: str, 
                               var_name: str,
                               accum_period: str, 
                               extension: str = "png") -> str:
        """
        This method generates a standardized output filename based on a specified pattern that includes the base name, variable type, accumulation type, valid time, and file extension. The filename follows the format "base_vartype_var_acctype_accum_valid_time_point.ext", where each component is filled in with the corresponding parameters provided to the method. This utility is useful for creating consistent and descriptive filenames for output products generated during MPAS diagnostic processing, making it easier to identify the contents and context of each file based on its name. By centralizing the filename formatting logic in this method, it promotes consistency across different parts of the codebase and simplifies the process of generating output filenames. 

        Parameters:
            base_name (str): Base name to include in the filename, typically representing the model or experiment.
            time_str (str): Valid time string to include in the filename, formatted as "YYYYMMDD_HHMMSS" or similar.
            var_name (str): Variable name to include in the filename, representing the type of data (e.g., "temperature", "precipitation").
            accum_period (str): Accumulation period to include in the filename, indicating the time period over which data is accumulated (e.g., "24hr", "6hr").
            extension (str): File extension to use for the output file (default: "png").

        Returns:
            str: Generated output filename following the specified pattern. 
        """
        return f"{base_name}_vartype_{var_name}_acctype_{accum_period}_valid_{time_str}_point.{extension}"
    
    @staticmethod
    def load_config_file(config_file: str) -> MPASConfig:
        """
        This method loads a YAML configuration file and parses it into an MPASConfig object. It checks if the specified configuration file exists and attempts to load it using the MPASConfig.load_from_file method. If the file does not exist or if there is an error during loading, it catches the exception, prints an error message to the console, and returns a default MPASConfig instance instead. This utility is essential for initializing the configuration parameters for MPAS diagnostic processing workflows, allowing users to specify their settings in a YAML file while providing a fallback to default settings if the file cannot be loaded. By centralizing the configuration loading logic in this method, it promotes consistency and error handling when working with configuration files across different parts of the codebase. 

        Parameters:
            config_file (str): Path to the YAML configuration file to load, can be absolute or relative. 

        Returns:
            MPASConfig: An instance of MPASConfig containing the loaded configuration parameters. If the file cannot be loaded, returns a default MPASConfig instance. 
        """
        try:
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Configuration file not found: {config_file}")

            return MPASConfig.load_from_file(config_file)

        except Exception as e:
            print(f"Error loading configuration file: {e}")
            print("Using default configuration")
            return MPASConfig()
    
    @staticmethod
    def _find_diag_files(data_dir: str) -> List[str]:
        """
        This internal method searches for diagnostic files in the specified data directory using a predefined glob pattern (DIAG_GLOB). It first looks for files directly in the provided data directory, then checks common subdirectories such as "diag" and "mpasout". If no files are found in these locations, it performs a recursive search through all subdirectories. The method returns a list of matching file paths. This utility is used by the validate_input_files method to ensure that the necessary diagnostic files are present in the expected locations before processing. By centralizing this file searching logic, it promotes consistency in how diagnostic files are located across different parts of the MPAS diagnostic codebase.

        Parameters:
            data_dir (str): Path to the data directory to search for diagnostic files, can be absolute or relative.

        Returns:
            List[str]: List of file paths matching the diagnostic glob pattern, or an empty list if no files are found.
        """
        files = FileManager.find_files(data_dir, DIAG_GLOB, recursive=False)

        if not files:
            diag_sub = os.path.join(data_dir, 'diag')
            mpasout_sub = os.path.join(data_dir, 'mpasout')
            files = FileManager.find_files(diag_sub, DIAG_GLOB, recursive=False)

            if not files:
                files = FileManager.find_files(mpasout_sub, DIAG_GLOB, recursive=False)

        if not files:
            files = FileManager.find_files(data_dir, DIAG_GLOB, recursive=True)

        return files

    @staticmethod
    def validate_input_files(config: MPASConfig) -> bool:
        """
        This method performs validation checks on the input files specified in the configuration object. It verifies that the grid file exists, that the data directory exists and is a directory, and that the data directory contains diagnostic files matching the expected glob pattern. If any of these checks fail, it collects error messages and prints them to the console, returning False to indicate that validation failed. If all checks pass, it returns True to indicate that the input files are valid and ready for processing. This utility is crucial for ensuring that the necessary input files are present and correctly specified before running MPAS diagnostic processing workflows, helping to prevent errors later in the processing due to missing or misconfigured input files. By centralizing the validation logic in this method, it promotes consistency in how input files are checked across different parts of the codebase. 

        Parameters:
            config (MPASConfig): Configuration object containing the parameters to validate, including grid_file and data_dir attributes. 

        Returns:
            bool: True if all input files are valid, False if any validation checks fail. 
        """
        errors: List[str] = []

        if not getattr(config, "grid_file", None):
            errors.append("Grid file not specified")
        elif not os.path.exists(config.grid_file):
            errors.append(f"Grid file not found: {config.grid_file}")

        if not getattr(config, "data_dir", None):
            errors.append("Data directory not specified")
        elif not os.path.exists(config.data_dir):
            errors.append(f"Data directory not found: {config.data_dir}")
        elif not os.path.isdir(config.data_dir):
            errors.append(f"Data path is not a directory: {config.data_dir}")

        if getattr(config, "data_dir", None) and os.path.exists(config.data_dir):
            data_files = FileManager._find_diag_files(config.data_dir)
            if not data_files:
                errors.append(f"No diagnostic files found in: {config.data_dir}")

        if errors:
            print("Input validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False

        return True


def print_system_info() -> None:
    """
    Module-level convenience function that prints basic system information to the
    console, including the Python version, platform, current working directory,
    and available memory. Delegates to :meth:`FileManager.print_system_info`.

    Parameters:
        None

    Returns:
        None
    """
    FileManager.print_system_info()
