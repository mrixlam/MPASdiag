#!/usr/bin/env python3

"""
MPAS DateTime and Temporal Utilities

This module provides comprehensive datetime and temporal utilities for MPAS model data processing including filename datetime parsing, time coordinate extraction and validation, temporal range calculations, and time series handling. It implements the MPASDateTimeUtils class with static methods for extracting timestamps from MPAS output filenames using regex pattern matching, converting between various datetime formats (numpy.datetime64, pandas.Timestamp, Python datetime objects), validating time coordinates in xarray datasets with fallback strategies for missing or malformed time information, and computing temporal statistics like accumulation periods and time deltas. The utilities handle diverse MPAS filename conventions (diag files, history files, restart files) with flexible pattern matching, support both CF-compliant and non-standard time coordinate encodings, provide robust error handling for missing or invalid temporal metadata, and enable consistent time handling across all MPASdiag processing and visualization modules. Core capabilities include automatic time unit detection and conversion, time string formatting for plot labels and file naming, temporal subsetting and indexing operations, and integration with pandas for time series analysis suitable for operational weather diagnostics and climate model evaluation.

Classes:
    MPASDateTimeUtils: Utility class providing static methods for datetime operations on MPAS model output files and datasets.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import os
import re
import pandas as pd
import xarray as xr
from datetime import datetime
from typing import List, Tuple, Optional, Any
from .constants import DATASET_NOT_LOADED_MSG


class MPASDateTimeUtils:
    """
    DateTime utilities class for MPAS temporal operations.
    
    This class provides functionality for handling datetime parsing from filenames,
    time coordinate validation, and temporal data processing for MPAS datasets.
    """
    
    @staticmethod
    def parse_file_datetimes(diag_files: List[str], verbose: bool = True) -> List[datetime]:
        """
        Parse datetime information from MPAS diagnostic filenames using standardized naming pattern extraction. This method uses regular expressions to extract date and time components from MPAS output filenames following the standard YYYY-MM-DD_HH.MM.SS pattern commonly used in MPAS model output. The parsing handles malformed filenames gracefully by generating synthetic datetime values with hourly increments to maintain temporal ordering when actual datetime extraction fails. This robust fallback ensures that data loading operations can proceed even with inconsistently named files. If verbose mode is enabled, the method prints warning messages for unparseable filenames to alert users about potential filename issues. The extracted datetime objects are used for temporal coordinate assignment and sorting in multi-file dataset loading operations.

        Parameters:
            diag_files (List[str]): List of absolute or relative paths to MPAS diagnostic files with datetime information encoded in filenames.
            verbose (bool): Enable verbose output for debugging and warning messages about unparseable filenames (default: True).

        Returns:
            List[datetime]: List of datetime objects parsed from filenames in same order as input, with synthetic datetimes for unparseable files.
        """
        file_datetimes = []

        pattern = r'(\d{4})-(\d{2})-(\d{2})_(\d{2})\.(\d{2})\.(\d{2})'
        
        for diag_file in diag_files:
            filename = os.path.basename(diag_file)
            match = re.search(pattern, filename)
            
            if match:
                year, month, day, hour, minute, second = match.groups()
                try:
                    file_dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
                except ValueError:
                    if verbose:
                        print(f"Warning: Invalid datetime parsed from filename: {filename}")
                    file_dt = datetime(2000, 1, 1) + pd.Timedelta(hours=len(file_datetimes))
            else:
                if verbose:
                    print(f"Warning: Could not parse datetime from filename: {filename}")
                file_dt = datetime(2000, 1, 1) + pd.Timedelta(hours=len(file_datetimes))
            
            file_datetimes.append(file_dt)
        return file_datetimes

    @staticmethod
    def validate_time_parameters(dataset: Any, time_index: int, verbose: bool = False) -> Tuple[str, int, int]:
        """
        Validate time index parameter against dataset temporal dimensions with automatic bounds checking and normalization. This method performs comprehensive validation of time index requests by detecting the appropriate time dimension name used in the dataset and checking bounds against the actual time series length. The method handles both 'Time' and 'time' dimension name conventions used across different MPAS output formats. When the requested time index exceeds available time steps, the method automatically clamps to the last valid index and issues a warning if verbose mode is enabled. This automatic clamping prevents index errors while alerting users about out-of-bounds requests. The method returns all necessary information for subsequent time-based data extraction including validated indices and dimension metadata.

        Parameters:
            dataset (Any): MPAS dataset object (xarray.Dataset or ux.UxDataset) containing time information for validation.
            time_index (int): Zero-based time index to validate against dataset temporal extent.
            verbose (bool): Enable verbose output for warnings about clamped or adjusted indices (default: False).

        Returns:
            Tuple[str, int, int]: Three-element tuple containing (time_dimension_name, validated_time_index, total_time_steps) where validated_time_index is clamped to valid range.
            
        Raises:
            ValueError: If dataset is None with instruction to load dataset first.
        """
        if dataset is None:
            raise ValueError(DATASET_NOT_LOADED_MSG)
        
        time_dim = 'Time' if 'Time' in dataset.sizes else 'time'
        time_size = dataset.sizes[time_dim]
        
        if time_index >= time_size:
            if verbose:
                print(f"Warning: time_index {time_index} exceeds available times {time_size}, using last time")
            time_index = time_size - 1
        
        return time_dim, time_index, time_size

    @staticmethod
    def get_time_info(dataset: xr.Dataset, time_index: int, var_context: str = "", 
                     verbose: bool = True) -> str:
        """
        Retrieve formatted time coordinate information for diagnostic output and filename generation. This method extracts the time coordinate value at the specified index and formats it as a human-readable string suitable for plot titles, filenames, or diagnostic messages. The method handles multiple datetime formats and performs automatic conversion to pandas datetime objects when necessary. When time coordinates are unavailable or parsing fails, the method generates a fallback string using the time index to ensure robust behavior. Optional variable context can be included in verbose output to provide additional debugging information. The formatted time string follows the YYYYMMDDTHH convention for compact representation suitable for filenames while remaining human-readable.

        Parameters:
            dataset (xr.Dataset): MPAS dataset containing time coordinate information to extract and format.
            time_index (int): Zero-based index into the time dimension for which to retrieve coordinate information.
            var_context (str): Optional variable name or context string for customizing verbose diagnostic messages (default: "").
            verbose (bool): Enable verbose output for debugging and diagnostic information about time extraction (default: True).

        Returns:
            str: Formatted time string in YYYYMMDDTHH format suitable for filenames and labels, or fallback "time_{index}" string if time coordinate is unavailable.
        """
        try:
            if hasattr(dataset, 'Time') and len(dataset.Time) > time_index:
                time_value = dataset.Time.values[time_index]
                if hasattr(time_value, 'strftime'):
                    time_str = time_value.strftime('%Y%m%dT%H')
                else:
                    time_dt = pd.to_datetime(time_value)
                    time_str = time_dt.strftime('%Y%m%dT%H')
                
                if verbose:
                    context_msg = f" (using variable: {var_context})" if var_context else ""
                    print(f"Time index {time_index} corresponds to: {time_str}{context_msg}")
                
                return time_str
            else:
                if verbose:
                    context_msg = f" (time coordinate not available, using variable: {var_context})" if var_context else " (time coordinate not available)"
                    print(f"Using time index {time_index}{context_msg}")
                return f"time_{time_index}"
        except Exception as e:
            if verbose:
                context_msg = f" (could not parse time: {e}, using variable: {var_context})" if var_context else f" (could not parse time: {e})"
                print(f"Using time index {time_index}{context_msg}")
            return f"time_{time_index}"

    @staticmethod
    def get_time_range(dataset: xr.Dataset) -> Tuple[datetime, datetime]:
        """
        Extract temporal extent of loaded dataset by identifying start and end times from coordinate data. This method retrieves the first and last time coordinate values from the dataset's time dimension and converts them to standard Python datetime objects. The method performs floor rounding to the nearest second to provide clean timestamp values suitable for display and comparison operations. This temporal extent information is essential for validating time-based queries, generating dataset summaries, and ensuring temporal coverage meets analysis requirements. The method performs validation to ensure the dataset contains time coordinate information before attempting extraction. Both start and end times are returned as timezone-naive datetime objects for compatibility with standard Python datetime operations.

        Parameters:
            dataset (xr.Dataset): MPAS dataset containing time coordinate information for extent calculation.

        Returns:
            Tuple[datetime, datetime]: Two-element tuple containing (start_time, end_time) as datetime objects floor-rounded to seconds representing temporal coverage of dataset.
            
        Raises:
            ValueError: If dataset is None or doesn't contain Time coordinate with explanatory message.
        """
        if dataset is None:
            raise ValueError("Dataset cannot be None.")
        
        if 'Time' not in dataset.coords and 'Time' not in dataset.data_vars:
            raise ValueError("Dataset does not contain Time coordinate.")
        
        time_values = dataset.Time.values

        start_time = pd.to_datetime(time_values[0]).floor('s').to_pydatetime()
        end_time = pd.to_datetime(time_values[-1]).floor('s').to_pydatetime()
        
        return start_time, end_time

    @staticmethod
    def format_time_for_filename(dt: datetime, format_type: str = 'mpas') -> str:
        """
        Format datetime object into filename-safe string following specified convention for consistent file naming. This method supports multiple output format conventions including MPAS standard format with underscores and dots, ISO compact format without delimiters, and ultra-compact format with only date and hour. The MPAS format follows the YYYY-MM-DD_HH.MM.SS pattern commonly used in MPAS model output filenames. ISO format produces YYYYMMDDTHHMMSS suitable for sortable filenames with ISO 8601 convention. Compact format generates YYYYMMDDHH for minimal filename length while preserving temporal information. These formatted strings are essential for generating consistent output filenames that maintain temporal ordering and human readability. The method validates format_type parameter to prevent invalid format specifications.

        Parameters:
            dt (datetime): Python datetime object to format for use in filename generation.
            format_type (str): Format convention identifier, one of 'mpas', 'iso', or 'compact' (default: 'mpas').

        Returns:
            str: Formatted datetime string following specified convention, suitable for use in filenames with alphanumeric characters and standard delimiters only.

        Raises:
            ValueError: If format_type is not one of the supported options ('mpas', 'iso', 'compact').
        """
        if format_type == 'mpas':
            return dt.strftime('%Y-%m-%d_%H.%M.%S')
        elif format_type == 'iso':
            return dt.strftime('%Y%m%dT%H%M%S')
        elif format_type == 'compact':
            return dt.strftime('%Y%m%d%H')
        else:
            raise ValueError(f"Unknown format_type: {format_type}")

    @staticmethod
    def parse_time_from_string(time_str: str, format_patterns: Optional[List[str]] = None) -> datetime:
        """
        Parse datetime from string representation using multiple format patterns with automatic fallback. This method attempts to parse datetime strings using a comprehensive list of format patterns covering common conventions including MPAS standard format, ISO formats, and compact representations. The parser tries each pattern sequentially until successful parsing occurs or all patterns are exhausted. Default patterns include MPAS format with underscores and dots, ISO compact format, standard space-delimited format, ultra-compact format, and ISO format with colons. Users can provide custom format patterns to handle non-standard datetime string representations. This flexible parsing approach enables robust datetime extraction from various sources including filenames, metadata, and user input. The method raises a clear exception when all parsing attempts fail to facilitate debugging of format mismatches.

        Parameters:
            time_str (str): Time string to parse into datetime object, potentially in any of several common formats.
            format_patterns (Optional[List[str]]): Custom list of strftime format patterns to attempt for parsing, or None to use comprehensive default pattern list (default: None).

        Returns:
            datetime: Successfully parsed datetime object representing the time encoded in input string.
            
        Raises:
            ValueError: If time string cannot be parsed with any of the provided or default patterns, including original string in error message.
        """
        if format_patterns is None:
            format_patterns = [
                '%Y-%m-%d_%H.%M.%S',  # MPAS format
                '%Y%m%dT%H%M%S',      # ISO compact
                '%Y-%m-%d %H:%M:%S',  # Standard format
                '%Y%m%d%H',           # Compact format
                '%Y-%m-%dT%H:%M:%S',  # ISO format
            ]
        
        for pattern in format_patterns:
            try:
                return datetime.strptime(time_str, pattern)
            except ValueError:
                continue
        
        raise ValueError(f"Could not parse time string '{time_str}' with any of the provided patterns.")

    @staticmethod
    def get_time_bounds(dataset: xr.Dataset, time_index: int) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Retrieve time bounds information for specific time index if available in dataset. This method searches for time bounds variables using multiple common naming conventions including time_bnds, time_bounds, Time_bnds, and Time_bounds. Time bounds represent the start and end timestamps of time intervals or accumulation periods associated with each time coordinate value. The method extracts bounds for the specified time index and converts them to Python datetime objects for standard temporal operations. If bounds variables are not present or extraction fails, the method returns None tuple to indicate unavailable bounds information. Time bounds are particularly important for accumulated variables where the time coordinate represents the end of an accumulation period rather than an instantaneous value. This information is essential for correctly interpreting temporal meaning of data values.

        Parameters:
            dataset (xr.Dataset): MPAS dataset containing time bounds information in standard CF-compliant variables.
            time_index (int): Zero-based time index for which to retrieve start and end bound timestamps.

        Returns:
            Tuple[Optional[datetime], Optional[datetime]]: Two-element tuple containing (start_bound, end_bound) as datetime objects if available, or (None, None) if bounds variables are not present or extraction fails.
        """
        if dataset is None:
            return None, None
            
        time_bounds_vars = ['time_bnds', 'time_bounds', 'Time_bnds', 'Time_bounds']
        
        for bounds_var in time_bounds_vars:
            if bounds_var in dataset.data_vars:
                try:
                    bounds = dataset[bounds_var].values[time_index]
                    start_bound = pd.to_datetime(bounds[0]).to_pydatetime()
                    end_bound = pd.to_datetime(bounds[1]).to_pydatetime()
                    return start_bound, end_bound
                except (IndexError, ValueError):
                    continue
        
        return None, None

    @staticmethod
    def calculate_time_delta(dataset: xr.Dataset) -> pd.Timedelta:
        """
        Calculate average time delta between consecutive time steps for temporal resolution determination. This method computes time differences between all adjacent time coordinate values and returns the median difference to provide a robust estimate of dataset temporal resolution. The median statistic is preferred over mean to handle potential irregular time steps or outliers in time spacing. This temporal resolution information is essential for selecting appropriate accumulation periods, determining valid temporal offsets, and validating consistency of time series data. The method requires at least two time steps to compute meaningful time differences. Time delta calculation supports both regular and irregular time grids by using median to represent typical time spacing. The returned pandas Timedelta object supports convenient arithmetic operations and format conversions.

        Parameters:
            dataset (xr.Dataset): MPAS dataset containing time coordinate information for delta calculation.

        Returns:
            pd.Timedelta: Median time delta between consecutive time steps representing typical temporal resolution of dataset.
            
        Raises:
            ValueError: If dataset is None, doesn't contain Time coordinate, or has fewer than 2 time steps with explanatory message.
        """
        if dataset is None or 'Time' not in dataset.coords:
            raise ValueError("Dataset is None or doesn't contain Time coordinate.")
        
        time_values = pd.to_datetime(dataset.Time.values)
        
        if len(time_values) < 2:
            raise ValueError("Need at least 2 time steps to calculate time delta.")
        
        time_diffs = time_values[1:] - time_values[:-1]
        return pd.Timedelta(time_diffs.median())