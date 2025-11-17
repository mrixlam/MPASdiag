#!/usr/bin/env python3

"""
MPAS File Management Utilities

This module provides comprehensive file and directory management functionality for MPAS data analysis workflows including file discovery, path validation, directory operations, and cleanup utilities. It implements the FileManager class with static methods for locating MPAS output files using glob pattern matching, validating input file paths and directories with existence checks, creating output directory structures with proper permissions, formatting file sizes for human-readable display, and performing age-based cleanup of temporary and output files. The utilities support common file operations needed across MPASdiag processing and visualization modules including recursive directory creation, file pattern matching with wildcards, metadata inspection for file size and modification time, and safe file deletion with validation checks. Core capabilities include automatic handling of file system errors with informative error messages, support for both absolute and relative paths with path normalization, integration with MPASConfig for consistent path management, and stateless design enabling usage without class instantiation suitable for both interactive scripts and automated batch processing pipelines.

Classes:
    FileManager: Centralized utility class providing static methods for file and directory operations in MPAS analysis workflows.

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


class FileManager:
    """
    Centralized file and directory management utilities for MPAS data analysis workflows. This class provides stateless helper methods for discovering files on disk, validating input locations, formatting file sizes, and performing cleanup operations. All methods are implemented as static methods allowing usage without class instantiation. The utilities support common file operations including directory creation, file pattern matching, metadata inspection, and age-based cleanup. Methods that modify the filesystem document their side effects explicitly to ensure safe usage in production environments.
    """
    
    @staticmethod
    def ensure_directory(directory: str) -> None:
        """
        Create directory if it does not already exist using idempotent filesystem operations. This safe helper wraps pathlib.Path.mkdir with parents=True and exist_ok=True to enable multiple calls without errors. The method creates all intermediate parent directories as needed when the full path does not exist. This is essential for setting up output directory structures before saving analysis results. Any filesystem errors such as permission issues will propagate as OSError exceptions from the underlying pathlib operations.

        Parameters:
            directory (str): Filesystem path to ensure exists, may be absolute or relative path string.

        Returns:
            None

        Raises:
            OSError: If the directory cannot be created due to permissions or other filesystem errors.
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def find_files(directory: str, pattern: str = "*.nc",
                   recursive: bool = False) -> List[str]:
        """
        Return sorted list of file paths matching glob pattern under specified directory. This helper searches for files using pathlib glob operations and returns paths as strings for easy consumption. The method supports both immediate directory search (glob) and recursive subdirectory search (rglob) controlled by the recursive parameter. If the directory does not exist, an empty list is returned to enable graceful handling. The returned list is always sorted for consistent ordering across multiple calls and platforms.

        Parameters:
            directory (str): Directory to search for files, if directory does not exist returns empty list.
            pattern (str): Glob pattern to match files such as "*.nc" or "diag*.nc" (default: "*.nc").
            recursive (bool): If True performs recursive search through subdirectories, if False only searches immediate directory (default: False).

        Returns:
            List[str]: Sorted list of matching file paths as strings, empty list if no matches found.
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
        Return dictionary containing basic metadata and statistics for specified file path. The returned dictionary always contains an exists key indicating file presence, with additional metadata included when the file exists. This method wraps pathlib.Path.stat to provide convenient access to file size in bytes and megabytes, modification timestamp, and creation timestamp. The metadata reflects underlying filesystem state and is useful for diagnostic logging and file validation. All timestamp values are returned as Python datetime objects for easy manipulation and formatting.

        Parameters:
            filepath (str): Path to the file to inspect for metadata extraction.

        Returns:
            Dict[str, Any]: Dictionary containing file metadata with keys: exists (bool), size (int bytes), size_mb (float megabytes), modified (datetime), created (datetime).
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
    def cleanup_files(directory: str, pattern: str = "*.tmp",
                      older_than_days: int = 7) -> int:
        """
        Remove files matching glob pattern that are older than specified age threshold based on modification time. This method performs age-based cleanup by comparing file modification timestamps against the cutoff date calculated from current time minus older_than_days. The method attempts to delete matching files and returns a count of successfully removed files for reporting. Filesystem exceptions from pathlib.Path.unlink such as permission errors will propagate to the caller for explicit handling. This utility is designed for periodic cleanup of temporary or intermediate analysis files to manage disk usage.

        Parameters:
            directory (str): Directory to search for files to delete.
            pattern (str): Glob pattern to match files for potential deletion such as "*.tmp" or "*.bak" (default: "*.tmp").
            older_than_days (int): Files with modification time older than this many days will be deleted (default: 7).

        Returns:
            int: Number of files successfully deleted from the directory.
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
        Convert byte count to human-readable size string with appropriate unit suffix. This formatting utility automatically selects the most appropriate unit (B, KB, MB, GB, TB, PB) by iteratively dividing by 1024 until the value is less than 1024. The method returns a formatted string with one decimal place precision followed by the unit abbreviation for easy reading. This is commonly used for displaying file sizes, memory usage, and data transfer amounts in user-friendly formats. The conversion follows binary units (1024 bytes per kilobyte) consistent with filesystem reporting conventions.

        Parameters:
            size_bytes (int): Size in bytes to format for human-readable display.

        Returns:
            str: Human-readable size string with one decimal place and unit suffix (e.g., "1.2 MB", "345.6 GB").
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
        Return available system memory in gigabytes using psutil library when available. This utility provides a cross-platform way to query system memory statistics for capacity planning and resource monitoring. When psutil is not installed or cannot be imported, the function returns 0.0 to indicate the metric is unavailable rather than raising an exception. The memory value represents currently available RAM that can be allocated to new processes without swapping. This information is useful for determining appropriate chunk sizes and parallelization strategies for large dataset processing.

        Returns:
            float: Available system memory in gigabytes, or 0.0 if psutil is not installed or memory information cannot be determined.
        """
        try:
            import psutil

            return psutil.virtual_memory().available / (1024**3)
        except ImportError:
            return 0.0
    
    @staticmethod
    def print_system_info() -> None:
        """
        Print concise summary of current system environment to stdout for quick diagnostics. This method outputs Python version, platform identifier, current working directory, and available memory statistics in a formatted block. The output is intended for interactive debugging and log files rather than machine parsing. Callers requiring programmatic access to individual metrics should use get_available_memory and other specific helper methods directly. The information helps diagnose environment-specific issues and verify runtime conditions before executing analysis workflows.

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
    def create_output_filename(base_name: str, time_str: str, var_name: str,
                               accum_period: str, extension: str = "png") -> str:
        """
        Create standardized output filename following consistent naming convention used throughout analysis examples. The naming pattern incorporates base name, variable identifier, accumulation period, validation time, and file extension in a structured format. This convention mirrors existing output naming used in repository examples to maintain consistency across analysis products. The format enables systematic organization and discovery of output files while encoding essential metadata in the filename. The standardized names support automated post-processing and archival workflows that rely on predictable filename patterns.

        Parameters:
            base_name (str): Base name for the file, typically the analysis or experiment name.
            time_str (str): Time stamp string to include such as "20240917T13" in compact format.
            var_name (str): Variable name or identifier such as "t2m", "rain", or "wind".
            accum_period (str): Accumulation or aggregation period identifier for temporal context.
            extension (str): File extension without dot such as "png", "pdf", or "nc" (default: "png").

        Returns:
            str: Formatted filename string following pattern "base_vartype_var_acctype_accum_valid_time_point.ext".
        """
        return f"{base_name}_vartype_{var_name}_acctype_{accum_period}_valid_{time_str}_point.{extension}"
    
    @staticmethod
    def load_config_file(config_file: str) -> MPASConfig:
        """
        Load MPASConfig object from YAML configuration file with graceful error handling and default fallback. This method attempts to read and parse the specified configuration file, returning a fully initialized MPASConfig instance with user-specified parameters. If the file cannot be found, read, or parsed for any reason, the function prints an informative error message and returns a default MPASConfig instance to allow workflows to continue. This fallback behavior enables robust operation even when configuration files are missing or malformed. The error handling prevents workflow failures while clearly communicating configuration issues to users through console output.

        Parameters:
            config_file (str): Path to YAML configuration file containing MPASConfig parameters.

        Returns:
            MPASConfig: Parsed configuration object from file, or default configuration instance if loading fails.
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
    def validate_input_files(config: MPASConfig) -> bool:
        """
        Validate that required input files and directories referenced by configuration exist and are accessible. This method performs comprehensive validation of grid file path, data directory path, and diagnostic file presence to catch configuration errors early. The function checks that the grid file exists, data directory is a valid directory (not a file), and at least one diagnostic file matching the pattern is present. When validation fails, the method prints a concise list of all detected problems to help users correct configuration issues. The boolean return value enables conditional workflow execution based on input availability.

        Parameters:
            config (MPASConfig): Configuration object containing paths to grid file, data directory, and other input specifications.

        Returns:
            bool: True if all validation checks pass (grid file exists, data directory exists and contains diagnostic files), False if any validation errors detected.
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
            data_files = FileManager.find_files(config.data_dir, "diag*.nc", recursive=False)
            if not data_files:
                diag_sub = os.path.join(config.data_dir, 'diag')
                mpasout_sub = os.path.join(config.data_dir, 'mpasout')
                data_files = FileManager.find_files(diag_sub, "diag*.nc", recursive=False)
                if not data_files:
                    data_files = FileManager.find_files(mpasout_sub, "diag*.nc", recursive=False)
            if not data_files:
                data_files = FileManager.find_files(config.data_dir, "diag*.nc", recursive=True)
            if not data_files:
                errors.append(f"No diagnostic files found in: {config.data_dir}")

        if errors:
            print("Input validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False

        return True