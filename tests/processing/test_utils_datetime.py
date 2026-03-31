#!/usr/bin/env python3
"""
MPASdiag Test Suite: Comprehensive tests for datetime utilities in MPASdiag

This module contains a comprehensive set of unit tests for the datetime utility functions in the MPASdiag processing module. The tests cover parsing datetimes from filenames, validating time parameters in datasets, retrieving human-readable time information, extracting time ranges, formatting datetimes for filenames, and parsing datetimes from various string formats. Each test case is designed to verify correct functionality under normal conditions as well as edge cases and error scenarios. The suite ensures that the datetime utilities are robust, handle invalid inputs gracefully, and provide informative diagnostics when verbose mode is enabled.

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
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime

from mpasdiag.processing.utils_datetime import MPASDateTimeUtils

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestParseFileDatetimes:
    """ Tests for parsing datetimes from MPAS diagnostic filenames. """
    
    def test_parse_file_datetimes_valid(self: "TestParseFileDatetimes") -> None:
        """
        This test verifies that valid MPAS diagnostic filenames are correctly parsed into datetime objects. The test provides a list of filenames that follow the expected pattern (e.g., `diag.YYYY-MM-DD_HH.MM.SS.nc`) and asserts that the returned list of datetimes matches the expected values. This ensures that the parsing logic correctly extracts date and time components from well-formed filenames. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        files = [
            'diag.2024-01-01_00.00.00.nc',
            'diag.2024-01-01_06.00.00.nc',
            'diag.2024-01-01_12.00.00.nc',
        ]
        
        datetimes = MPASDateTimeUtils.parse_file_datetimes(files, verbose=False)
        
        assert len(datetimes) == pytest.approx(3)
        assert datetimes[0] == datetime(2024, 1, 1, 0, 0, 0)
        assert datetimes[1] == datetime(2024, 1, 1, 6, 0, 0)
        assert datetimes[2] == datetime(2024, 1, 1, 12, 0, 0)
    
    def test_parse_file_datetimes_invalid_filename(self: "TestParseFileDatetimes") -> None:
        """
        This test checks that the parser can handle invalid filename patterns gracefully. When the filename list contains entries that do not match expected MPAS patterns, the parser should still return a datetime list, using synthetic or fallback datetimes for unparseable names. This test asserts that a synthetic datetime is generated for the invalid entry and other valid entries are parsed correctly. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        files = [
            'diag.2024-01-01_00.00.00.nc',
            'invalid_filename.nc',
            'diag.2024-01-01_12.00.00.nc',
        ]
        
        datetimes = MPASDateTimeUtils.parse_file_datetimes(files, verbose=False)
        
        assert len(datetimes) == pytest.approx(3)
        assert datetimes[0] == datetime(2024, 1, 1, 0, 0, 0)
        assert isinstance(datetimes[1], datetime)
        assert datetimes[2] == datetime(2024, 1, 1, 12, 0, 0)
    
    def test_parse_file_datetimes_invalid_date_values(self: "TestParseFileDatetimes") -> None:
        """
        This test verifies that filenames with invalid date values (e.g., month=13) are handled without crashing. The parser should recognize that the date components are out of valid ranges and return a synthetic datetime instead. This test asserts that the returned datetime for the invalid filename is a fallback value (e.g., `datetime(2000, 1, 1, 0, 0, 0)`) rather than raising an exception. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        files = [
            'diag.2024-13-01_00.00.00.nc',  # Invalid month
        ]
        
        datetimes = MPASDateTimeUtils.parse_file_datetimes(files, verbose=False)
        
        assert len(datetimes) == pytest.approx(1)
        assert datetimes[0] == datetime(2000, 1, 1, 0, 0, 0)
    
    def test_parse_file_datetimes_verbose(self: "TestParseFileDatetimes") -> None:
        """
        This test confirms that when `verbose=True` is passed, the parser emits a warning for unparseable filenames. The test captures stdout and asserts that the expected warning message is printed when an invalid filename is encountered. It also checks that the returned datetime for the invalid entry is a synthetic fallback value. It ensures that the verbose mode provides useful diagnostics without interrupting the parsing process. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        files = ['invalid_filename.nc']
        
        from io import StringIO
        import sys

        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            datetimes = MPASDateTimeUtils.parse_file_datetimes(files, verbose=True)
            output = captured_output.getvalue()            
            assert "Warning" in output
            assert "Could not parse datetime" in output
            assert datetimes[0] == datetime(2000, 1, 1, 0, 0, 0)
        finally:
            sys.stdout = sys.__stdout__
    
    def test_parse_file_datetimes_with_path(self: "TestParseFileDatetimes") -> None:
        """
        This test verifies that the parser can extract datetimes from filenames that include directory paths. The parser should ignore the path components and correctly parse the datetime from the filename itself. This test provides a list of file paths with valid MPAS filename patterns and asserts that the returned datetimes are correct, confirming that path components do not interfere with parsing. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        files = [
            '/path/to/diag.2024-02-15_18.30.45.nc',
            '/another/path/diag.2024-02-16_00.00.00.nc',
        ]
        
        datetimes = MPASDateTimeUtils.parse_file_datetimes(files, verbose=False)
        
        assert len(datetimes) == pytest.approx(2)
        assert datetimes[0] == datetime(2024, 2, 15, 18, 30, 45)
        assert datetimes[1] == datetime(2024, 2, 16, 0, 0, 0)


class TestValidateTimeParameters:
    """ Tests for validating time parameters in xarray Datasets. """
    
    @pytest.fixture(autouse=True)
    def setup_dataset(self: "TestValidateTimeParameters") -> None:
        """
        This fixture sets up a mock xarray Dataset with a `Time` coordinate for testing time parameter validation. The dataset contains a `temperature` variable with dimensions `Time` and `nCells`, and the `Time` coordinate is populated with a range of hourly timestamps. This setup allows the test methods to validate time indices against a realistic dataset structure. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This fixture sets up instance attributes and does not
            return a value.
        """
        self.dataset = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.rand(10, 100),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=10, freq='h')}
            )
        })
    
    def test_validate_time_parameters_valid_index(self: "TestValidateTimeParameters") -> None:
        """
        This test confirms that valid time parameters are correctly validated and returned. When a valid time index is provided (e.g., 5), the function should return the correct time dimension name (`Time`), the provided index, and the total size of the time dimension (10 in this case). This test asserts that the returned values match the expected results for a well-formed dataset and valid index. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        time_dim, time_idx, time_size = MPASDateTimeUtils.validate_time_parameters(
            self.dataset, 5, verbose=False
        )
        
        assert time_dim == 'Time'
        assert time_idx == pytest.approx(5)
        assert time_size == pytest.approx(10)
    
    def test_validate_time_parameters_out_of_bounds(self: "TestValidateTimeParameters") -> None:
        """
        This test checks that when an out-of-bounds time index is provided, the function clamps the index to the last valid index and returns it without raising an exception. For example, if the dataset has 10 time steps and the requested index is 15, the function should return index 9 (the last valid index) and not raise an error. This test asserts that the returned index is correctly clamped and that the time dimension name and size are still accurate. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        time_dim, time_idx, time_size = MPASDateTimeUtils.validate_time_parameters(
            self.dataset, 15, verbose=False
        )
        
        assert time_dim == 'Time'
        assert time_idx == pytest.approx(9)
        assert time_size == pytest.approx(10)
    
    def test_validate_time_parameters_out_of_bounds_verbose(self: "TestValidateTimeParameters") -> None:
        """
        This test verifies that when an out-of-bounds time index is provided and `verbose=True`, the function emits a warning message indicating that the requested index exceeds available times and that it has been clamped to the last valid index. The test captures stdout and asserts that the expected warning message is printed, confirming that verbose mode provides useful diagnostics for out-of-range indices while still returning the clamped index correctly. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        from io import StringIO
        import sys

        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            time_dim, time_idx, time_size = MPASDateTimeUtils.validate_time_parameters(
                self.dataset, 20, verbose=True
            )
            output = captured_output.getvalue()
            
            assert "Warning" in output
            assert "exceeds available times" in output
            assert time_idx == pytest.approx(9)
        finally:
            sys.stdout = sys.__stdout__
    
    def test_validate_time_parameters_lowercase_time(self: "TestValidateTimeParameters") -> None:
        """
        This test confirms that the function can handle datasets where the time dimension is named in lowercase (e.g., `time` instead of `Time`). The function should still recognize the time dimension, validate the index, and return the correct dimension name, index, and size. This test constructs a dataset with a lowercase time dimension and asserts that the validation logic correctly identifies it and returns expected values. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        dataset_lowercase = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.rand(5, 100),
                dims=['time', 'nCells'],
                coords={'time': pd.date_range('2024-01-01', periods=5, freq='h')}
            )
        })
        
        time_dim, time_idx, time_size = MPASDateTimeUtils.validate_time_parameters(
            dataset_lowercase, 2, verbose=False
        )
        
        assert time_dim == 'time'
        assert time_idx == pytest.approx(2)
        assert time_size == pytest.approx(5)
    
    def test_validate_time_parameters_none_dataset(self: "TestValidateTimeParameters") -> None:
        """
        This test verifies that if `None` is passed as the dataset, the function raises a `ValueError` with an appropriate message indicating that the dataset is not loaded. This ensures that the function has proper error handling for missing or uninitialized datasets. The test asserts that the exception is raised and that the error message contains the expected explanation. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        with pytest.raises(ValueError) as ctx:
            MPASDateTimeUtils.validate_time_parameters(None, 0)
        
        assert "Dataset not loaded" in str(ctx.value)


class TestGetTimeInfo:
    """ Tests for retrieving human-readable time information from datasets. """
    
    @pytest.fixture(autouse=True)
    def setup_dataset(self: "TestGetTimeInfo") -> None:
        """
        This fixture sets up a mock xarray Dataset with a `Time` coordinate for testing the retrieval of human-readable time information. The dataset contains a `temperature` variable with dimensions `Time` and `nCells`, and the `Time` coordinate is populated with a range of timestamps at 6-hour intervals. This setup allows the test methods to verify that the function can extract and format time information correctly based on the provided index and dataset structure. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This fixture sets instance attributes and does not return
            a value.
        """
        self.dataset = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.rand(5, 100),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-03-15 12:00', periods=5, freq='6h')}
            )
        })
    
    def test_get_time_info_valid(self: "TestGetTimeInfo") -> None:
        """
        This test verifies that the function can successfully retrieve a human-readable time string for a valid time index. When a valid index (e.g., 0) is provided, the function should return a formatted string representing the corresponding time step (e.g., `20240315T12`). This test asserts that the returned string matches the expected format and value based on the dataset's `Time` coordinate for the given index. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        time_str = MPASDateTimeUtils.get_time_info(self.dataset, 0, verbose=False)        
        assert time_str == '20240315T12'
    
    def test_get_time_info_verbose(self: "TestGetTimeInfo") -> None:
        """
        This test confirms that when `verbose=True` is passed, the function emits informative messages about the time index and variable context. The test captures stdout and asserts that the expected informational message is printed, indicating the time index, corresponding time step, and variable context (e.g., "temperature"). It also checks that the returned time string is correct, ensuring that verbose mode provides useful diagnostics while still returning accurate time information. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        from io import StringIO
        import sys

        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            time_str = MPASDateTimeUtils.get_time_info(
                self.dataset, 2, var_context="temperature", verbose=True
            )
            output = captured_output.getvalue()
            
            assert "Time index 2 corresponds to" in output
            assert "temperature" in output
            assert time_str == '20240316T00'
        finally:
            sys.stdout = sys.__stdout__
    
    def test_get_time_info_no_time_coord(self: "TestGetTimeInfo") -> None:
        """
        This test checks that if the dataset does not contain a `Time` coordinate, the function returns a deterministic fallback label (e.g., `time_0`) instead of raising an exception. This ensures that the function can handle datasets without time information gracefully. The test constructs a dataset without a time coordinate and asserts that the returned string is the expected fallback value. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        dataset_no_time = xr.Dataset({
            'temperature': xr.DataArray(np.random.rand(100), dims=['nCells'])
        })
        
        time_str = MPASDateTimeUtils.get_time_info(dataset_no_time, 0, verbose=False)        
        assert time_str == 'time_0'
    
    def test_get_time_info_out_of_bounds(self: "TestGetTimeInfo") -> None:
        """
        This test ensures that when an out-of-range index is requested, the function returns a deterministic fallback label (e.g., `time_10`) instead of raising an exception. This allows the function to handle requests for time indices that exceed the dataset's range gracefully. The test provides an index that is beyond the available time steps and asserts that the returned string is the expected fallback value, confirming that out-of-bounds requests are managed without errors. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        time_str = MPASDateTimeUtils.get_time_info(self.dataset, 10, verbose=False)        
        assert time_str == 'time_10'
    
    def test_get_time_info_exception_handling(self: "TestGetTimeInfo") -> None:
        """
        This test verifies that if an unexpected error occurs during time information retrieval (e.g., due to malformed time data), the function handles the exception gracefully and returns a fallback label (e.g., `time_0`) instead of crashing. This ensures robustness in the face of unforeseen issues with the dataset's time information. The test constructs a dataset with a problematic time coordinate and asserts that the returned string is the expected fallback value, confirming that exceptions are managed without interrupting execution. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        dataset_bad = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.rand(3, 100),
                dims=['Time', 'nCells']
            )
        })

        dataset_bad['Time'] = xr.DataArray([None, None, None], dims=['Time'])
        
        time_str = MPASDateTimeUtils.get_time_info(dataset_bad, 0, verbose=False)
        assert time_str == 'time_0'


class TestGetTimeRange:
    """ Tests for extracting the start and end datetimes from a dataset's Time coordinate. """
    
    @pytest.fixture(autouse=True)
    def setup_dataset(self: "TestGetTimeRange") -> None:
        """
        This fixture sets up a mock xarray Dataset with a `Time` coordinate for testing the extraction of time ranges. The dataset contains a `temperature` variable with dimensions `Time` and `nCells`, and the `Time` coordinate is populated with a range of daily timestamps. This setup allows the test methods to verify that the function can correctly identify the first and last time steps in the dataset and return them as `datetime` objects. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This fixture configures instance attributes and does not
            return a value.
        """
        self.dataset = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.rand(10, 100),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=10, freq='D')}
            )
        })
    
    def test_get_time_range_valid(self: "TestGetTimeRange") -> None:
        """
        This test confirms that the function can successfully extract the start and end datetimes from a dataset's `Time` coordinate. When a valid dataset with a `Time` coordinate is provided, the function should return two `datetime` objects representing the first and last time steps in the dataset. This test asserts that the returned values are indeed `datetime` instances and that they correspond to the expected start and end times based on the dataset's `Time` coordinate. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        start_time, end_time = MPASDateTimeUtils.get_time_range(self.dataset)
        
        assert isinstance(start_time, datetime)
        assert isinstance(end_time, datetime)
        assert start_time.year == pytest.approx(2024)
        assert start_time.month == pytest.approx(1)
        assert start_time.day == pytest.approx(1)
        assert end_time.day == pytest.approx(10)
    
    def test_get_time_range_none_dataset(self: "TestGetTimeRange") -> None:
        """
        This test verifies that if `None` is passed as the dataset, the function raises a `ValueError` with an appropriate message indicating that the dataset cannot be `None`. This ensures that the function has proper error handling for missing or uninitialized datasets. The test asserts that the exception is raised and that the error message contains the expected explanation. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        with pytest.raises(ValueError) as ctx:
            MPASDateTimeUtils.get_time_range(None)  # type: ignore[arg-type]
        
        assert "Dataset cannot be None" in str(ctx.value)
    
    def test_get_time_range_no_time_coord(self: "TestGetTimeRange") -> None:
        """
        This test checks that if the dataset does not contain a `Time` coordinate, the function raises a `ValueError` with an appropriate message indicating that the dataset lacks a `Time` coordinate. This ensures that the function can handle datasets without time information gracefully by providing informative error messages. The test constructs a dataset without a time coordinate and asserts that the expected exception is raised with the correct message. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        dataset_no_time = xr.Dataset({
            'temperature': xr.DataArray(np.random.rand(100), dims=['nCells'])
        })
        
        with pytest.raises(ValueError) as ctx:
            MPASDateTimeUtils.get_time_range(dataset_no_time)
        
        assert "does not contain Time coordinate" in str(ctx.value)
    
    def test_get_time_range_single_time(self: "TestGetTimeRange") -> None:
        """
        This test verifies that if the dataset contains only a single time step, the function returns the same `datetime` for both the start and end times. This ensures that the function can handle edge cases where there is no range of time steps and still provides consistent output. The test constructs a dataset with only one time step and asserts that the returned start and end times are identical. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        dataset_single = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.rand(1, 100),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-06-15', periods=1, freq='h')}
            )
        })
        
        start_time, end_time = MPASDateTimeUtils.get_time_range(dataset_single)        
        assert start_time == end_time


class TestFormatTimeForFilename:
    """ Tests for formatting datetime objects into strings suitable for filenames. """
    
    @pytest.fixture(autouse=True)
    def setup_datetime(self: "TestFormatTimeForFilename") -> None:
        """
        This fixture sets up a known `datetime` object for testing the formatting of datetimes into strings suitable for filenames. The fixture initializes an instance attribute `self.dt` with a specific `datetime` value (e.g., July 4, 2024, at 15:30:45) that will be used across multiple test methods to verify different formatting options. This allows the tests to assert that the formatting function produces consistent and correct string representations based on the same input datetime. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This fixture sets an instance attribute and does not
            return a value.
        """
        self.dt = datetime(2024, 7, 4, 15, 30, 45)
    
    def test_format_time_mpas_format(self: "TestFormatTimeForFilename") -> None:
        """
        This test verifies that the function correctly formats a `datetime` object into the MPAS filename convention `YYYY-MM-DD_HH.MM.SS`. The test asserts that when the `format_type` is set to `'mpas'`, the returned string matches the expected format and value based on the fixture datetime. This ensures that the function can produce the specific formatting required for MPAS diagnostic filenames. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        result = MPASDateTimeUtils.format_time_for_filename(self.dt, 'mpas')        
        assert result == '2024-07-04_15.30.45'
    
    def test_format_time_iso_format(self: "TestFormatTimeForFilename") -> None:
        """
        This test confirms that the function can format a `datetime` object into a compact ISO format `YYYYMMDDTHHMMSS`. When the `format_type` is set to `'iso'`, the returned string should match this compact representation, which is commonly used for filenames that require a concise timestamp. The test asserts that the output string is correctly formatted and corresponds to the fixture datetime. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        result = MPASDateTimeUtils.format_time_for_filename(self.dt, 'iso')        
        assert result == '20240704T153045'
    
    def test_format_time_compact_format(self: "TestFormatTimeForFilename") -> None:
        """
        This test checks that the function can format a `datetime` object into a compact hour-only format `YYYYMMDDHH`. When the `format_type` is set to `'compact'`, the returned string should include only the date and hour components, which can be useful for filenames that do not require minute or second precision. The test asserts that the output string is correctly formatted and corresponds to the fixture datetime's date and hour. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        result = MPASDateTimeUtils.format_time_for_filename(self.dt, 'compact')        
        assert result == '2024070415'
    
    def test_format_time_invalid_format(self: "TestFormatTimeForFilename") -> None:
        """
        This test verifies that if an invalid `format_type` is provided, the function raises a `ValueError` with an appropriate message indicating that the format type is unknown. This ensures that the function has proper error handling for unsupported format types and provides informative feedback to the user. The test asserts that the expected exception is raised and that the error message contains the correct explanation. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        with pytest.raises(ValueError) as ctx:
            MPASDateTimeUtils.format_time_for_filename(self.dt, 'invalid')
        
        assert "Unknown format_type" in str(ctx.value)
    
    def test_format_time_default(self: "TestFormatTimeForFilename") -> None:
        """
        This test confirms that if no `format_type` is provided, the function defaults to the MPAS filename convention `YYYY-MM-DD_HH.MM.SS`. When the `format_type` parameter is omitted, the returned string should match the MPAS format, ensuring that the function has a sensible default behavior. The test asserts that the output string is correctly formatted according to the MPAS convention and corresponds to the fixture datetime. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        result = MPASDateTimeUtils.format_time_for_filename(self.dt)
        assert result == '2024-07-04_15.30.45'


class TestParseTimeFromString:
    """ Tests for parsing datetime objects from various string formats. """
    
    def test_parse_time_mpas_format(self: "TestParseTimeFromString") -> None:
        """
        This test verifies that the function can parse datetime strings in the MPAS filename convention `YYYY-MM-DD_HH.MM.SS`. The test provides a string in this format and asserts that the returned `datetime` object has the correct year, month, day, hour, minute, and second values corresponding to the input string. This ensures that the parser can handle the specific formatting used in MPAS diagnostic filenames. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        time_str = '2024-08-20_14.25.30'        
        result = MPASDateTimeUtils.parse_time_from_string(time_str)        
        assert result == datetime(2024, 8, 20, 14, 25, 30)
    
    def test_parse_time_iso_compact(self: "TestParseTimeFromString") -> None:
        """
        This test confirms that the function can parse compact ISO datetime strings in the format `YYYYMMDDTHHMMSS`. The test provides a string in this compact format and asserts that the returned `datetime` object has the correct components corresponding to the input string. This ensures that the parser can handle concise ISO formats that are commonly used for filenames. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        time_str = '20240820T142530'        
        result = MPASDateTimeUtils.parse_time_from_string(time_str)        
        assert result == datetime(2024, 8, 20, 14, 25, 30)
    
    def test_parse_time_standard_format(self: "TestParseTimeFromString") -> None:
        """
        This test verifies that the parser recognizes the common human-readable format `YYYY-MM-DD HH:MM:SS` and returns the corresponding `datetime` object. The test provides a string in this standard format and asserts that the returned `datetime` has the correct date and time components, ensuring that the parser can handle conventional timestamp formats in addition to filename-specific patterns. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        time_str = '2024-08-20 14:25:30'        
        result = MPASDateTimeUtils.parse_time_from_string(time_str)        
        assert result == datetime(2024, 8, 20, 14, 25, 30)
    
    def test_parse_time_compact_format(self: "TestParseTimeFromString") -> None:
        """
        This test checks that the parser can handle a compact format that includes only the date and hour components (e.g., `YYYYMMDDHH`) and returns a `datetime` object with minutes and seconds set to zero. The test provides a string in this compact format and asserts that the returned `datetime` has the correct year, month, day, and hour, with minutes and seconds defaulting to zero. This ensures that the parser can interpret partial datetime strings appropriately. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        time_str = '2024082014'        
        result = MPASDateTimeUtils.parse_time_from_string(time_str)        
        assert result == datetime(2024, 8, 20, 14, 0, 0)
    
    def test_parse_time_iso_with_colons(self: "TestParseTimeFromString") -> None:
        """
        This test confirms that the parser can handle ISO-like formats that include colons in the time component (e.g., `YYYY-MM-DDTHH:MM:SS`) and correctly parse them into `datetime` objects. The test provides a string in this format and asserts that the returned `datetime` has the correct date and time components, demonstrating that the parser is flexible enough to recognize variations of ISO formatting. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        time_str = '2024-08-20T14:25:30'        
        result = MPASDateTimeUtils.parse_time_from_string(time_str)        
        assert result == datetime(2024, 8, 20, 14, 25, 30)
    
    def test_parse_time_custom_patterns(self: "TestParseTimeFromString") -> None:
        """
        This test verifies that the parser can utilize custom patterns provided by the user to successfully parse datetime strings that do not match built-in formats. The test supplies a string in a custom format (e.g., `DD/MM/YYYY HH:MM`) along with the corresponding pattern and asserts that the returned `datetime` object has the correct components based on the input string. This ensures that the parser is extensible and can handle user-defined datetime formats when necessary. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        time_str = '20/08/2024 14:25'
        custom_patterns = ['%d/%m/%Y %H:%M']        
        result = MPASDateTimeUtils.parse_time_from_string(time_str, custom_patterns)        
        assert result == datetime(2024, 8, 20, 14, 25, 0)
    
    def test_parse_time_invalid_string(self: "TestParseTimeFromString") -> None:
        """
        This test checks that when an invalid time string that cannot be parsed by any of the built-in or custom patterns is provided, the function raises a `ValueError` with an appropriate message indicating that the time string could not be parsed. The test asserts that the expected exception is raised and that the error message contains the original input string for diagnostic purposes, ensuring that users receive informative feedback when parsing fails. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        time_str = 'not_a_valid_time'
        
        with pytest.raises(ValueError) as ctx:
            MPASDateTimeUtils.parse_time_from_string(time_str)
        
        assert "Could not parse time string" in str(ctx.value)
        assert time_str in str(ctx.value)


class TestGetTimeBounds:
    """ Tests for extracting time bounds from xarray Datasets, including handling of bounds variables and edge cases. """
    
    def test_get_time_bounds_valid(self: "TestGetTimeBounds") -> None:
        """
        This test verifies that the function can successfully extract the start and end time bounds from a dataset that contains a properly formatted bounds variable (e.g., `time_bnds`). The test constructs a dataset with a `Time` coordinate and an associated bounds variable, then asserts that the returned start and end bounds are `datetime` objects corresponding to the expected time intervals. This confirms that the function can correctly interpret bounds metadata when it is present. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        dataset = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.rand(3, 100),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=3, freq='h')}
            ),
            'time_bnds': xr.DataArray(
                np.array([
                    [pd.Timestamp('2024-01-01 00:00').value, pd.Timestamp('2024-01-01 01:00').value],
                    [pd.Timestamp('2024-01-01 01:00').value, pd.Timestamp('2024-01-01 02:00').value],
                    [pd.Timestamp('2024-01-01 02:00').value, pd.Timestamp('2024-01-01 03:00').value],
                ]),
                dims=['Time', 'nv']
            )
        })
        
        start_bound, end_bound = MPASDateTimeUtils.get_time_bounds(dataset, 0)
        
        assert isinstance(start_bound, datetime)
        assert isinstance(end_bound, datetime)
    
    def test_get_time_bounds_alternative_name(self: "TestGetTimeBounds") -> None:
        """
        This test confirms that the function can recognize alternative bounds variable names (e.g., `Time_bounds`) and extract the correct time bounds when the standard name (`time_bnds`) is not present. The test constructs a dataset with a `Time` coordinate and an alternative bounds variable, then asserts that the returned start and end bounds are `datetime` objects corresponding to the expected intervals. This ensures that the function is flexible in identifying bounds metadata even when it does not follow the standard naming convention. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        dataset = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.rand(2, 100),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=2, freq='h')}
            ),
            'Time_bounds': xr.DataArray(
                np.array([
                    [pd.Timestamp('2024-01-01 00:00').value, pd.Timestamp('2024-01-01 01:00').value],
                    [pd.Timestamp('2024-01-01 01:00').value, pd.Timestamp('2024-01-01 02:00').value],
                ]),
                dims=['Time', 'nv']
            )
        })
        
        start_bound, end_bound = MPASDateTimeUtils.get_time_bounds(dataset, 1)
        
        assert start_bound is not None
        assert end_bound is not None
    
    def test_get_time_bounds_no_bounds(self: "TestGetTimeBounds") -> None:
        """
        This test checks that if the dataset does not contain any bounds variable (neither `time_bnds` nor `Time_bounds`), the function returns `(None, None)` instead of raising an exception. This ensures that the function can handle datasets without bounds metadata gracefully by providing a clear fallback response. The test constructs a dataset with a `Time` coordinate but no bounds variable and asserts that the returned start and end bounds are both `None`. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        dataset = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.rand(3, 100),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=3, freq='h')}
            )
        })
        
        start_bound, end_bound = MPASDateTimeUtils.get_time_bounds(dataset, 0)
        
        assert start_bound is None
        assert end_bound is None
    
    def test_get_time_bounds_none_dataset(self: "TestGetTimeBounds") -> None:
        """
        This test verifies that if `None` is passed as the dataset, the function returns `(None, None)` instead of raising an exception. This ensures that the function has proper error handling for missing or uninitialized datasets by providing a clear fallback response. The test asserts that when `None` is provided, both the start and end bounds returned by the function are `None`, confirming that it can manage null inputs gracefully without crashing. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        start_bound, end_bound = MPASDateTimeUtils.get_time_bounds(None, 0)  # type: ignore[arg-type]
        assert start_bound is None
        assert end_bound is None
    
    def test_get_time_bounds_index_error(self: "TestGetTimeBounds") -> None:
        """
        This test checks that if an out-of-range index is requested for the time bounds, the function returns `(None, None)` instead of raising an exception. This ensures that the function can handle requests for indices that exceed the available time steps gracefully by providing a clear fallback response. The test constructs a dataset with a `Time` coordinate and bounds variable, then requests an index that is beyond the range of available time steps and asserts that both the start and end bounds returned are `None`. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        dataset = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.rand(2, 100),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=2, freq='h')}
            )
        })

        dataset['time_bnds'] = xr.DataArray(
            np.array([
                [pd.Timestamp('2024-01-01 00:00').value, pd.Timestamp('2024-01-01 01:00').value],
                [pd.Timestamp('2024-01-01 01:00').value, pd.Timestamp('2024-01-01 02:00').value],
            ]),
            dims=['Time', 'nv']
        )
        
        start_bound, end_bound = MPASDateTimeUtils.get_time_bounds(dataset, 5)
        
        assert start_bound is None
        assert end_bound is None


class TestCalculateTimeDelta:
    """ Tests for calculating the time delta between steps in a dataset's Time coordinate, including handling of regular and irregular time intervals and edge cases. """
    
    def test_calculate_time_delta_hourly(self: "TestCalculateTimeDelta") -> None:
        """
        This test verifies that the function can correctly calculate the time delta for a dataset with an hourly `Time` coordinate. The test constructs a dataset with a `Time` coordinate that advances in one-hour increments and asserts that the calculated time delta is equal to `pd.Timedelta(hours=1)`. This confirms that the function can accurately determine the time interval between steps for regularly spaced hourly data. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        dataset = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.rand(10, 100),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=10, freq='h')}
            )
        })
        
        time_delta = MPASDateTimeUtils.calculate_time_delta(dataset)        
        assert time_delta == pd.Timedelta(hours=1)
    
    def test_calculate_time_delta_daily(self: "TestCalculateTimeDelta") -> None:
        """
        This test confirms that the function can calculate the time delta for a dataset with a daily `Time` coordinate. The test constructs a dataset where the `Time` coordinate advances in one-day increments and asserts that the calculated time delta is equal to `pd.Timedelta(days=1)`. This ensures that the function can accurately determine the time interval between steps for regularly spaced daily data. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        dataset = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.rand(7, 100),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=7, freq='D')}
            )
        })
        
        time_delta = MPASDateTimeUtils.calculate_time_delta(dataset)        
        assert time_delta == pd.Timedelta(days=1)
    
    def test_calculate_time_delta_six_hourly(self: "TestCalculateTimeDelta") -> None:
        """
        This test checks that the function can calculate the time delta for a dataset with a six-hourly `Time` coordinate. The test constructs a dataset where the `Time` coordinate advances in six-hour increments and asserts that the calculated time delta is equal to `pd.Timedelta(hours=6)`. This confirms that the function can accurately determine the time interval between steps for regularly spaced six-hourly data. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        dataset = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.rand(8, 100),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=8, freq='6h')}
            )
        })
        
        time_delta = MPASDateTimeUtils.calculate_time_delta(dataset)        
        assert time_delta == pd.Timedelta(hours=6)
    
    def test_calculate_time_delta_none_dataset(self: "TestCalculateTimeDelta") -> None:
        """
        This test verifies that if `None` is passed as the dataset, the function raises a `ValueError` with an appropriate message indicating that the dataset is invalid. This ensures that the function has proper error handling for missing or uninitialized datasets by providing informative feedback to the user. The test asserts that the expected exception is raised and that the error message contains the correct explanation regarding the dataset being `None` or lacking a `Time` coordinate. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        with pytest.raises(ValueError) as ctx:
            MPASDateTimeUtils.calculate_time_delta(None)  # type: ignore[arg-type]
        
        assert "Dataset is None or doesn't contain Time coordinate" in str(ctx.value)
    
    def test_calculate_time_delta_no_time_coord(self: "TestCalculateTimeDelta") -> None:
        """
        This test checks that if the dataset does not contain a `Time` coordinate, the function raises a `ValueError` with an appropriate message indicating that the dataset is invalid. This ensures that the function can handle datasets without time information gracefully by providing informative error messages. The test constructs a dataset without a time coordinate and asserts that the expected exception is raised with the correct message regarding the absence of a `Time` coordinate.

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        dataset = xr.Dataset({
            'temperature': xr.DataArray(np.random.rand(100), dims=['nCells'])
        })
        
        with pytest.raises(ValueError) as ctx:
            MPASDateTimeUtils.calculate_time_delta(dataset)
        
        assert "Dataset is None or doesn't contain Time coordinate" in str(ctx.value)
    
    def test_calculate_time_delta_single_time(self: "TestCalculateTimeDelta") -> None:
        """
        This test verifies that if the dataset contains only a single time step, the function raises a `ValueError` with an appropriate message indicating that at least two time steps are required to calculate a time delta. This ensures that the function can handle edge cases where there is insufficient data to determine a time interval and provides informative feedback to the user. The test constructs a dataset with only one time step and asserts that the expected exception is raised with the correct message regarding the need for at least two time steps.

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        dataset = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.rand(1, 100),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=1, freq='h')}
            )
        })
        
        with pytest.raises(ValueError) as ctx:
            MPASDateTimeUtils.calculate_time_delta(dataset)
        
        assert "Need at least 2 time steps" in str(ctx.value)
    
    def test_calculate_time_delta_irregular(self: "TestCalculateTimeDelta") -> None:
        """
        This test checks that the function can calculate the time delta for a dataset with irregular time intervals by determining the most common time difference between steps. The test constructs a dataset with a `Time` coordinate that advances in irregular increments (e.g., 1 hour, 2 hours, 1 hour, 3 hours, 1 hour) and asserts that the calculated time delta corresponds to the most frequently occurring interval (in this case, 1 hour). This confirms that the function can handle datasets with non-uniform time steps and still provide a meaningful time delta based on the predominant interval. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        times = pd.to_datetime([
            '2024-01-01 00:00',
            '2024-01-01 01:00',
            '2024-01-01 03:00',
            '2024-01-01 04:00',
            '2024-01-01 07:00',
            '2024-01-01 08:00',
        ])
        
        dataset = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.rand(6, 100),
                dims=['Time', 'nCells'],
                coords={'Time': times}
            )
        })
        
        time_delta = MPASDateTimeUtils.calculate_time_delta(dataset)
        assert time_delta == pd.Timedelta(hours=1)


class TestEdgeCases:
    """ Tests for edge cases and error handling in datetime utilities, including empty inputs, zero indices, boundary datetimes, and support for different time formats in dataset coordinates. """
    
    def test_parse_file_datetimes_empty_list(self: "TestEdgeCases") -> None:
        """
        This test verifies that when an empty list of files is provided to the `parse_file_datetimes` function, it returns an empty list of datetimes without raising an exception. This ensures that the function can handle edge cases where no input files are available gracefully by providing a clear and consistent output. The test asserts that the returned list of datetimes is empty, confirming that the function behaves as expected when given an empty input. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        files = []        
        datetimes = MPASDateTimeUtils.parse_file_datetimes(files, verbose=False)        
        assert len(datetimes) == pytest.approx(0)
    
    def test_validate_time_parameters_zero_index(self: "TestEdgeCases") -> None:
        """
        This test checks that when a zero index is provided to the `validate_time_parameters` function, it correctly identifies the first time step in the dataset and returns the appropriate time index without raising an exception. This ensures that the function can handle edge cases where the user requests the initial time step and provides informative feedback if the index is valid. The test constructs a dataset with a `Time` coordinate and asserts that when index 0 is requested, the returned time index is indeed 0, confirming that the function can manage this edge case properly. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        dataset = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.rand(5, 100),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=5, freq='h')}
            )
        })
        
        time_dim, time_idx, time_size = MPASDateTimeUtils.validate_time_parameters(
            dataset, 0, verbose=False
        )
        
        assert time_idx == pytest.approx(0)
    
    def test_format_time_edge_cases(self: "TestEdgeCases") -> None:
        """
        This test verifies that the `format_time_for_filename` function can handle edge case datetimes such as midnight (00:00:00) and the end of the year (December 31st at 23:59:59) without raising exceptions and formats them correctly according to the MPAS filename convention. The test constructs `datetime` objects for these edge cases and asserts that the formatted strings match the expected output, ensuring that the function can manage boundary datetime values appropriately. 

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        dt = datetime(2024, 1, 1, 0, 0, 0)
        result = MPASDateTimeUtils.format_time_for_filename(dt, 'mpas')

        assert result == '2024-01-01_00.00.00'
        
        dt = datetime(2024, 12, 31, 23, 59, 59)
        result = MPASDateTimeUtils.format_time_for_filename(dt, 'mpas')
        assert result == '2024-12-31_23.59.59'
    
    def test_get_time_info_with_numpy_datetime64(self: "TestEdgeCases") -> None:
        """
        This test checks that the `get_time_info` function can handle datasets where the `Time` coordinate is represented as `numpy.datetime64` objects instead of `pandas.Timestamp`. The test constructs a dataset with a `Time` coordinate using `numpy.datetime64` values and asserts that the function can extract the correct time information without raising exceptions. This ensures that the function is compatible with different datetime representations commonly used in xarray datasets.

        Parameters:
            self (object): The test instance.

        Returns:
            None: This test performs assertions and does not return a value.
        """
        dataset = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.rand(3, 100),
                dims=['Time', 'nCells']
            )
        })
        
        dataset['Time'] = xr.DataArray(
            [np.datetime64('2024-01-01T00:00:00', 'ns'),
             np.datetime64('2024-01-01T06:00:00', 'ns'),
             np.datetime64('2024-01-01T12:00:00', 'ns')],
            dims=['Time']
        )
        
        time_str = MPASDateTimeUtils.get_time_info(dataset, 1, verbose=False)        
        assert time_str == '20240101T06'


if __name__ == "__main__":
    pytest.main([__file__])