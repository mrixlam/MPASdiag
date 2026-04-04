#!/usr/bin/env python3

"""
MPASdiag Core Processing Module: DateTime and Temporal Utilities

This module provides a collection of utility functions for handling date and time operations in the context of MPAS diagnostic processing. It includes methods for parsing datetime information from filenames, validating time indices against dataset temporal coverage, extracting and formatting time information for labeling and filenames, calculating temporal coverage and resolution, and handling time bounds. These utilities are designed to facilitate consistent and robust handling of temporal information across various stages of MPAS diagnostic workflows, ensuring that time-based operations are performed accurately and that users are informed about any issues related to temporal data. The module is implemented as a class with static methods to allow for easy integration and reuse across different processing scripts and functions within the MPASdiag framework. 

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
    """ Utility class for handling date and time operations in MPAS diagnostic processing. """
    
    @staticmethod
    def parse_file_datetimes(diag_files: List[str], 
                             verbose: bool = True) -> List[datetime]:
        """
        This method parses datetime information from a list of MPAS diagnostic file paths by extracting timestamps encoded in the filenames. It uses a regular expression pattern to identify and extract datetime components (year, month, day, hour, minute, second) from the filename. If a filename does not match the expected pattern or if the extracted components cannot be converted to a valid datetime object, the method generates a synthetic datetime based on a fixed start date (January 1, 2000) incremented by an hourly offset corresponding to the file's position in the input list. This approach ensures that all files receive a datetime value for consistent processing, while also providing verbose output to inform users about any issues encountered during parsing. The method returns a list of datetime objects corresponding to each input file in the same order as provided. 

        Parameters:
            diag_files (List[str]): List of file paths to MPAS diagnostic files from which to parse datetime information.
            verbose (bool): Enable verbose output for warnings about unparseable filenames and generated synthetic datetimes (default: True). 

        Returns:
            List[datetime]: List of datetime objects corresponding to each input file, parsed from filenames or generated as synthetic values when parsing fails. 
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
    def validate_time_parameters(dataset: Any, 
                                 time_index: int, 
                                 verbose: bool = False) -> Tuple[str, int, int]:
        """
        This method validates the provided time index against the temporal coverage of the dataset. It checks for the presence of a time dimension (either 'Time' or 'time') and retrieves its size to determine the total number of available time steps. If the provided time index exceeds the available range, it clamps the index to the last valid time step and optionally outputs a warning message if verbose mode is enabled. The method returns a tuple containing the name of the time dimension, the validated (and potentially clamped) time index, and the total number of time steps in the dataset. This utility ensures that subsequent processing steps that rely on valid time indices can proceed without errors due to out-of-range indices, while also providing informative feedback to users about any adjustments made to their input parameters. 

        Parameters:
            dataset (Any): MPAS dataset object containing time dimension information to validate against.
            time_index (int): Zero-based index into the time dimension to validate.
            verbose (bool): Enable verbose output for warnings about out-of-range time indices and adjustments made (default: False). 

        Returns:
            Tuple[str, int, int]: A tuple containing (time_dim_name, validated_time_index, total_time_steps) where time_dim_name is the name of the time dimension ('Time' or 'time'), validated_time_index is the potentially clamped time index that is guaranteed to be within the dataset's temporal coverage, and total_time_steps is the total number of time steps available in the dataset for reference. 
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
    def _extract_time_str(dataset: xr.Dataset, 
                          time_index: int) -> Optional[str]:
        """
        This internal method attempts to extract a formatted time string from the dataset's 'Time' coordinate at the specified time index. It checks for the presence of the 'Time' coordinate and retrieves the corresponding time value at the given index. If the time value can be successfully parsed as a datetime object, it formats it into a string representation (e.g., 'YYYYMMDDTHH') suitable for use in filenames and labels. If the time coordinate is not available or if parsing fails, it returns None to indicate that time information could not be extracted. This method is designed to be used internally by the get_time_info method to provide a consistent approach to extracting and formatting time information from the dataset while handling potential issues gracefully.

        Parameters:
            dataset (xr.Dataset): MPAS dataset containing 'Time' coordinate for extraction.
            time_index (int): Zero-based index into the time dimension to extract time information from.

        Returns:
            Optional[str]: Formatted time string (e.g., 'YYYYMMDDTHH') extracted from dataset at specified time index, or None if time coordinate is not available or parsing fails.
        """
        if not (hasattr(dataset, 'Time') and len(dataset.Time) > time_index):
            return None

        time_value = dataset.Time.values[time_index]

        if hasattr(time_value, 'strftime'):
            return time_value.strftime('%Y%m%dT%H')

        return pd.to_datetime(time_value).strftime('%Y%m%dT%H')

    @staticmethod
    def _log_time_info(verbose: bool, 
                       time_index: int, 
                       time_str: Optional[str],
                       var_context: str, 
                       error: Optional[Exception] = None) -> None:
        """
        This internal method logs time information to the console based on the provided parameters. If verbose mode is enabled, it outputs the time index and the corresponding formatted time string if available. If the time string is not available due to missing coordinates or parsing errors, it logs a warning message indicating that the time index is being used without specific time information, along with any relevant context about the variable being processed. This method centralizes the logic for logging time-related information and warnings, ensuring consistent messaging across different parts of the code that handle time extraction and formatting.

        Parameters:
            verbose (bool): Enable verbose output for time information.
            time_index (int): Zero-based index into the time dimension.
            time_str (Optional[str]): Formatted time string extracted from dataset, or None if unavailable.
            var_context (str): Optional context string describing the variable or processing step.
            error (Optional[Exception]): Optional exception encountered during time extraction.

        Returns:
            None
        """
        if not verbose:
            return

        if time_str is not None:
            ctx = f" (using variable: {var_context})" if var_context else ""
            print(f"Time index {time_index} corresponds to: {time_str}{ctx}")
        elif error is not None:
            ctx = (f" (could not parse time: {error}, using variable: {var_context})"
                   if var_context else f" (could not parse time: {error})")
            print(f"Using time index {time_index}{ctx}")
        else:
            ctx = (f" (time coordinate not available, using variable: {var_context})"
                   if var_context else " (time coordinate not available)")
            print(f"Using time index {time_index}{ctx}")

    @staticmethod
    def get_time_info(dataset: xr.Dataset,
                      time_index: int,
                      var_context: str = "",
                      verbose: bool = True) -> str:
        """
        This method retrieves and formats time information from the dataset for a specific time index to be used in labeling and filenames. It checks for the presence of a 'Time' coordinate and attempts to extract the corresponding time value at the specified index. If the time value can be successfully parsed as a datetime object, it formats it into a string representation (e.g., 'YYYYMMDDTHH') suitable for use in filenames and labels. If the time coordinate is not available or if parsing fails, it falls back to returning a string in the format 'time_{index}' to ensure that some form of time information is always provided. The method also provides verbose output to inform users about the extracted time information or any issues encountered during extraction and parsing, including the context of the variable being processed if provided. This utility helps maintain consistent and informative labeling of outputs based on temporal information, even in cases where time data may be missing or malformed. 

        Parameters:
            dataset (xr.Dataset): MPAS dataset containing time coordinate information for extraction.
            time_index (int): Zero-based index into the time dimension to extract time information from.
            var_context (str): Optional context string describing the variable or processing step for verbose output (default: "").
            verbose (bool): Enable verbose output for extracted time information and any issues encountered during extraction and parsing (default: True). 

        Returns:
            str: Formatted time string (e.g., 'YYYYMMDDTHH') extracted from dataset at specified time index, or fallback string 'time_{index}' if time coordinate is not available or parsing fails. 
        """
        try:
            time_str = MPASDateTimeUtils._extract_time_str(dataset, time_index)
            MPASDateTimeUtils._log_time_info(verbose, time_index, time_str, var_context)
            return time_str if time_str is not None else f"time_{time_index}"
        except Exception as e:
            MPASDateTimeUtils._log_time_info(verbose, time_index, None, var_context, error=e)
            return f"time_{time_index}"

    @staticmethod
    def get_time_range(dataset: xr.Dataset) -> Tuple[datetime, datetime]:
        """
        This method calculates the start and end time range of the dataset based on the 'Time' coordinate. It retrieves the first and last time values from the 'Time' coordinate, converts them to datetime objects, and floor-rounds them to seconds to represent the temporal coverage of the dataset. The method returns a tuple containing the start and end times as datetime objects. This utility is essential for understanding the overall temporal extent of the dataset, which can inform decisions about temporal subsetting, analysis, and labeling in subsequent processing steps. The method raises informative exceptions if the dataset is None or if it does not contain a 'Time' coordinate to guide users in providing valid input datasets. 

        Parameters:
            dataset (xr.Dataset): MPAS dataset containing 'Time' coordinate for which to calculate temporal coverage. 

        Returns:
            Tuple[datetime, datetime]: A tuple containing (start_time, end_time) as datetime objects representing the temporal coverage of the dataset based on the 'Time' coordinate. 
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
    def format_time_for_filename(dt: datetime, 
                                 format_type: str = 'mpas') -> str:
        """
        This method formats a datetime object into a string representation suitable for use in filenames based on a specified format convention. It supports multiple format types, including 'mpas' (e.g., 'YYYY-MM-DD_HH.MM.SS'), 'iso' (e.g., 'YYYYMMDDTHHMMSS'), and 'compact' (e.g., 'YYYYMMDDHH'). The method uses Python's strftime functionality to convert the datetime object into the desired string format, ensuring that the resulting string contains only alphanumeric characters and standard delimiters that are safe for use in filenames across different operating systems. If an unknown format type is provided, the method raises a ValueError to inform users about the valid options. This utility allows for consistent and flexible formatting of datetime information in filenames, which can enhance organization and readability of output files generated during MPAS diagnostic processing. 

        Parameters:
            dt (datetime): Datetime object to format into a string representation for filenames.
            format_type (str): Format convention to use for formatting the datetime string, with options including 'mpas', 'iso', and 'compact' (default: 'mpas'). 

        Returns:
            str: Formatted datetime string suitable for use in filenames based on the specified format convention. 
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
    def parse_time_from_string(time_str: str, 
                               format_patterns: Optional[List[str]] = None) -> datetime:
        """
        This method attempts to parse a datetime object from a given time string by trying multiple format patterns. It accepts a time string and an optional list of strftime format patterns to attempt for parsing. If no custom patterns are provided, it uses a comprehensive default list that includes common formats such as MPAS, ISO, and standard datetime representations. The method iterates through the provided patterns and tries to parse the time string using each pattern until a successful parse is achieved. If none of the patterns can parse the string, it raises a ValueError to inform users about the failure to parse the time string with the provided patterns. This utility allows for flexible and robust parsing of datetime information from various string formats that may be encountered in filenames, labels, or other sources within MPAS diagnostic processing workflows. 

        Parameters:
            time_str (str): Input string containing datetime information to be parsed.
            format_patterns (Optional[List[str]]): Optional list of strftime format patterns to try for parsing the time string. If None, a default set of common patterns will be used (default: None). 

        Returns:
            datetime: Parsed datetime object extracted from the input time string based on the provided format patterns. 
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
    def get_time_bounds(dataset: xr.Dataset, 
                        time_index: int) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        This method retrieves the start and end time bounds for a specific time index from the dataset, if available. It checks for the presence of standard CF-compliant time bounds variables (e.g., 'time_bnds', 'Time_bnds') in the dataset and attempts to extract the corresponding start and end bound timestamps for the specified time index. If the bounds variables are present and can be successfully parsed as datetime objects, it returns them as a tuple (start_bound, end_bound). If the bounds variables are not present or if extraction fails, it returns (None, None) to indicate that time bounds information is not available. This utility is important for understanding the temporal coverage of individual time steps in the dataset, which can inform decisions about temporal subsetting, analysis, and labeling in subsequent processing steps. 

        Parameters:
            dataset (xr.Dataset): MPAS dataset containing potential time bounds variables for extraction. 
            time_index (int): Zero-based index into the time dimension for which to retrieve time bounds.

        Returns:
            Tuple[Optional[datetime], Optional[datetime]]: A tuple containing (start_bound, end_bound) as datetime objects representing the time bounds for the specified time index, or (None, None) if bounds information is not available. 
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
        This method calculates the median time delta between consecutive time steps in the dataset based on the 'Time' coordinate. It retrieves the time values from the 'Time' coordinate, converts them to datetime objects, and computes the differences between consecutive time steps. The method then returns the median of these time differences as a pd.Timedelta object, which represents the typical temporal resolution of the dataset. This utility is essential for understanding the temporal spacing of data points in the dataset, which can inform decisions about temporal subsetting, analysis, and labeling in subsequent processing steps. The method raises informative exceptions if the dataset is None or if it does not contain a 'Time' coordinate, or if there are not enough time steps to calculate a meaningful time delta. 

        Parameters:
            dataset (xr.Dataset): MPAS dataset containing 'Time' coordinate for which to calculate time delta. 

        Returns:
            pd.Timedelta: Median time delta between consecutive time steps in the dataset based on the 'Time' coordinate. 
        """
        if dataset is None or 'Time' not in dataset.coords:
            raise ValueError("Dataset is None or doesn't contain Time coordinate.")
        
        time_values = pd.to_datetime(dataset.Time.values)
        
        if len(time_values) < 2:
            raise ValueError("Need at least 2 time steps to calculate time delta.")
        
        time_diffs = time_values[1:] - time_values[:-1]
        return pd.Timedelta(time_diffs.median())