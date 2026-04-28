#!/usr/bin/env python3

"""
MPASdiag Test Suite: DateTime Utilities

This module contains unit tests for the MPASDateTimeUtils class, which provides utility functions for handling datetime information in MPAS diagnostic datasets. The tests cover various scenarios for parsing datetimes from filenames, validating time parameters in datasets, extracting time strings, logging time information, and calculating time deltas. Each test verifies that the corresponding method behaves as expected under different conditions, including edge cases and error handling. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from datetime import datetime
from io import StringIO
from unittest.mock import patch

from mpasdiag.processing.utils_datetime import MPASDateTimeUtils


@pytest.fixture
def time_dataset() -> xr.Dataset:
    """
    This fixture creates a simple xarray Dataset with a 'Time' coordinate containing 5 hourly timestamps starting from 2024-01-15T06:00:00. The dataset has no data variables, only the time coordinate, which is used for testing datetime utilities that operate on datasets with a Time coordinate. The timestamps are generated using pandas date_range for convenience and consistency in formatting. 

    Parameters:
        None

    Returns:
        xr.Dataset: An xarray Dataset with a 'Time' coordinate containing 5 hourly timestamps starting from 2024-01-15T06:00:00.
    """
    times = pd.date_range("2024-01-15T06:00:00", periods=5, freq="1h")
    return xr.Dataset(coords={"Time": times})


@pytest.fixture
def lowercase_time_dataset() -> xr.Dataset:
    """
    This fixture creates a simple xarray Dataset with a 'time' coordinate (lowercase) containing 3 hourly timestamps starting from 2024-01-15T00:00:00. The dataset has no data variables, only the time coordinate, which is used for testing datetime utilities that operate on datasets with lowercase time coordinates. The timestamps are generated using pandas date_range for convenience and consistency in formatting. 

    Parameters:
        None

    Returns:
        xr.Dataset: An xarray Dataset with a 'time' coordinate containing 3 hourly timestamps starting from 2024-01-15T00:00:00.
    """
    times = pd.date_range("2024-01-15T00:00:00", periods=3, freq="1h")
    return xr.Dataset(coords={"time": times})


@pytest.fixture
def no_time_dataset() -> xr.Dataset:
    """
    This fixture creates a simple xarray Dataset that does not contain any time coordinate. The dataset has a single data variable 'temperature' with 10 values, but no coordinates related to time. This is used for testing datetime utilities that need to handle datasets without time information, ensuring that the methods can gracefully handle the absence of a Time coordinate and return appropriate results or raise errors as needed. 

    Parameters:
        None

    Returns:
        xr.Dataset: An xarray Dataset without any Time coordinate.
    """
    return xr.Dataset({"temperature": xr.DataArray(np.ones(10), dims=["nCells"])})


@pytest.fixture
def strftime_time_mock():
    """
    This fixture creates a MagicMock dataset that simulates a Time coordinate containing a single pandas Timestamp. The Time coordinate is mocked to have a length of 1 and its values return a list containing a single pd.Timestamp. This is used for testing the _extract_time_str method to ensure that it correctly uses the strftime method when the time value is a pandas Timestamp, allowing us to verify that the method can handle pandas Timestamps and produce correctly formatted time strings. 

    Parameters:
        None

    Returns:
        MagicMock: A MagicMock dataset with a Time coordinate containing a single pd.Timestamp.
    """
    from unittest.mock import MagicMock
    ts = pd.Timestamp("2024-01-15T06:00:00")
    mock_time = MagicMock()
    mock_time.__len__ = MagicMock(return_value=1)
    mock_time.values = [ts]
    mock_ds = MagicMock()
    mock_ds.Time = mock_time
    return mock_ds


class TestParseFileDatetimes:
    """Tests for MPASDateTimeUtils.parse_file_datetimes."""

    def test_valid_filenames_parsed(self: 'TestParseFileDatetimes') -> None:
        """
        This test verifies that the parse_file_datetimes method correctly parses valid datetime strings from a list of filenames. The test checks that when the filenames contain recognizable datetime formats, the method returns a list of datetime objects corresponding to the parsed datetimes, confirming that it can successfully extract and convert datetime information from filenames for use in further processing or logging.  

        Parameters:
            None

        Returns:
            None
        """
        files = [
            "/data/diag.2024-01-15_06.00.00.nc",
            "/data/diag.2024-01-15_12.00.00.nc",
        ]
        result = MPASDateTimeUtils.parse_file_datetimes(files, verbose=False)
        assert len(result) == 2
        assert result[0] == datetime(2024, 1, 15, 6, 0, 0)
        assert result[1] == datetime(2024, 1, 15, 12, 0, 0)

    def test_no_match_silent_generates_synthetic(self: 'TestParseFileDatetimes') -> None:
        """
        This test verifies that the parse_file_datetimes method generates a synthetic datetime (January 1, 2000) when it fails to find a valid datetime string in the provided filename and verbose is set to False. The test checks that when the filename does not contain any recognizable datetime format, the method returns a list containing a single datetime object set to January 1, 2000, confirming that it correctly handles the absence of valid datetime information in silent mode without raising an error. 

        Parameters:
            None

        Returns:
            None
        """
        files = ["no_date_in_this_filename.nc"]
        result = MPASDateTimeUtils.parse_file_datetimes(files, verbose=False)
        assert len(result) == 1
        assert result[0] == datetime(2000, 1, 1)

    def test_no_match_verbose_prints_warning(self: 'TestParseFileDatetimes') -> None:
        """
        This test verifies that the parse_file_datetimes method prints a warning when it fails to find a valid datetime string in the provided filename and verbose is set to True. The test checks that when the filename does not contain any recognizable datetime format, the method prints a warning message to the standard output, confirming that it correctly notifies the user of the issue in verbose mode while still returning a list containing a single datetime object set to January 1, 2000. 

        Parameters:
            None

        Returns:
            None
        """
        files = ["no_date_in_this_filename.nc"]
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            result = MPASDateTimeUtils.parse_file_datetimes(files, verbose=True)
        assert len(result) == 1
        assert "Could not parse datetime" in captured.getvalue()

    def test_invalid_datetime_match_silent_fallback(self: 'TestParseFileDatetimes') -> None:
        """
        This test verifies that the parse_file_datetimes method generates a synthetic datetime (January 1, 2000) when it encounters a filename that matches the datetime regex pattern but contains an invalid datetime value (e.g., month 13) and verbose is set to False. The test checks that when the filename contains a string that looks like a datetime but fails to convert to a valid datetime object, the method returns a list containing a single datetime object set to January 1, 2000, confirming that it correctly handles invalid datetime values in silent mode without raising an error. 

        Parameters:
            None

        Returns:
            None
        """
        files = ["/data/diag.2024-13-01_00.00.00.nc"]
        result = MPASDateTimeUtils.parse_file_datetimes(files, verbose=False)
        assert len(result) == 1
        assert result[0] == datetime(2000, 1, 1)

    def test_invalid_datetime_match_verbose_prints_warning(self: 'TestParseFileDatetimes') -> None:
        """
        This test verifies that the parse_file_datetimes method prints a warning when it encounters a filename that matches the datetime regex pattern but contains an invalid datetime value (e.g., month 13) and verbose is set to True. The test checks that when the filename contains a string that looks like a datetime but fails to convert to a valid datetime object, the method prints a warning message to the standard output, confirming that it correctly notifies the user of the issue in verbose mode while still returning a list containing a single datetime object set to January 1, 2000. 

        Parameters:
            None

        Returns:
            None
        """
        files = ["/data/diag.2024-13-01_00.00.00.nc"]
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            result = MPASDateTimeUtils.parse_file_datetimes(files, verbose=True)
        assert "Invalid datetime" in captured.getvalue()
        assert len(result) == 1


class TestValidateTimeParameters:
    """Tests for MPASDateTimeUtils.validate_time_parameters."""

    def test_none_dataset_raises(self: 'TestValidateTimeParameters') -> None:
        """
        This test verifies that the validate_time_parameters method raises a ValueError when the provided dataset is None. The test checks that when None is passed as the dataset argument, the method raises an exception with an appropriate error message, confirming that it correctly validates the input and does not allow a None dataset to be processed for time parameter validation. 

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError):
            MPASDateTimeUtils.validate_time_parameters(None, 0)

    def test_valid_index_returns_correct_tuple(self: 'TestValidateTimeParameters', 
                                               time_dataset: xr.Dataset) -> None:
        """
        This test verifies that the validate_time_parameters method returns the correct tuple containing the dimension name, index, and size when provided with a valid dataset containing a Time coordinate and a valid time index. The test checks that when the dataset has a Time coordinate and the time index is within the valid range, the method returns a tuple with the dimension name "Time", the provided index, and the size of the Time dimension, confirming that it can successfully validate time parameters for use in further processing or logging. 

        Parameters:
            time_dataset (xr.Dataset): The dataset containing the time dimension.

        Returns:
            None
        """
        dim, idx, size = MPASDateTimeUtils.validate_time_parameters(time_dataset, 2)
        assert dim == "Time"
        assert idx == 2
        assert size == 5

    def test_out_of_range_index_clamped_silent(self: 'TestValidateTimeParameters', 
                                               time_dataset: xr.Dataset) -> None:
        """
        This test verifies that the validate_time_parameters method clamps the index to the maximum valid value when the provided index is out of range and verbose is set to False. The test checks that when the time index exceeds the size of the Time dimension, the method returns a tuple with the dimension name "Time", the index clamped to the last valid index (size - 1), and the size of the Time dimension, confirming that it correctly handles out-of-range indices in silent mode without printing any warnings. 

        Parameters:
            time_dataset (xr.Dataset): The dataset containing the time dimension.

        Returns:
            None
        """
        dim, idx, size = MPASDateTimeUtils.validate_time_parameters(time_dataset, 99, verbose=False)
        assert idx == size - 1

    def test_out_of_range_index_clamped_verbose(self: 'TestValidateTimeParameters', 
                                                time_dataset: xr.Dataset) -> None:
        """
        This test verifies that the validate_time_parameters method clamps the index to the maximum valid value and prints a warning when the provided index is out of range and verbose is set to True. The test checks that when the time index exceeds the size of the Time dimension, the method returns a tuple with the dimension name "Time", the index clamped to the last valid index (size - 1), and the size of the Time dimension, while also printing a warning message to the standard output indicating that the provided index exceeds the available range, confirming that it correctly handles out-of-range indices in verbose mode with appropriate logging. 

        Parameters:
            time_dataset (xr.Dataset): The dataset containing the time dimension.

        Returns:
            None
        """
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            dim, idx, size = MPASDateTimeUtils.validate_time_parameters(time_dataset, 99, verbose=True)
        assert idx == size - 1
        assert "exceeds" in captured.getvalue()

    def test_lowercase_time_dim_used(self: 'TestValidateTimeParameters', 
                                     lowercase_time_dataset: xr.Dataset) -> None:
        """
        This test verifies that the validate_time_parameters method correctly identifies and uses a lowercase 'time' coordinate when the dataset does not contain an uppercase 'Time' coordinate. The test checks that when the dataset has a lowercase 'time' coordinate, the method returns a tuple with the dimension name "time", the provided index, and the size of the time dimension, confirming that it can successfully validate time parameters using a lowercase time coordinate for use in further processing or logging. 

        Parameters:
            lowercase_time_dataset (xr.Dataset): The dataset containing a lowercase 'time' coordinate.

        Returns:
            None
        """
        dim, idx, size = MPASDateTimeUtils.validate_time_parameters(lowercase_time_dataset, 0)
        assert dim == "time"
        assert size == 3


class TestExtractTimeStr:
    """Tests for MPASDateTimeUtils._extract_time_str."""

    def test_no_time_attr_returns_none(self: 'TestExtractTimeStr', 
                                       no_time_dataset: xr.Dataset) -> None:
        """
        This test verifies that the _extract_time_str method returns None when the provided dataset does not contain a Time coordinate. The test checks that when the dataset lacks a Time coordinate, the method returns None, confirming that it correctly handles the absence of time information and does not attempt to extract a time string when no Time coordinate is available. 

        Parameters:
            no_time_dataset (xr.Dataset): The dataset without a time attribute.

        Returns:
            None
        """
        result = MPASDateTimeUtils._extract_time_str(no_time_dataset, 0)
        assert result is None

    def test_time_index_out_of_range_returns_none(self: 'TestExtractTimeStr', 
                                                  time_dataset: xr.Dataset) -> None:
        """
        This test verifies that the _extract_time_str method returns None when the provided time index is out of range for the Time coordinate in the dataset. The test checks that when the time index exceeds the size of the Time dimension, the method returns None, confirming that it correctly handles out-of-range indices and does not attempt to extract a time string when the specified index is invalid. 

        Parameters:
            time_dataset (xr.Dataset): The dataset containing the time dimension.

        Returns:
            None
        """
        result = MPASDateTimeUtils._extract_time_str(time_dataset, 999)
        assert result is None

    def test_timestamp_uses_strftime_path(self: 'TestExtractTimeStr', 
                                          strftime_time_mock) -> None:
        """
        This test verifies that the _extract_time_str method correctly uses the strftime method to extract a time string when the time value is a pandas Timestamp. The test checks that when the Time coordinate contains a pandas Timestamp, the method calls strftime on the timestamp to format it into the expected time string, confirming that it can handle pandas Timestamps and produce correctly formatted time strings for use in filenames or logging. 

        Parameters:
            strftime_time_mock (xr.Dataset): The dataset containing a pandas Timestamp.

        Returns:
            None
        """
        time_val = strftime_time_mock.Time.values[0]
        assert hasattr(time_val, "strftime"), "Mock must yield a pd.Timestamp"
        result = MPASDateTimeUtils._extract_time_str(strftime_time_mock, 0)
        assert result == "20240115T06"

    def test_numpy_datetime64_uses_pd_to_datetime(self: 'TestExtractTimeStr', 
                                                  time_dataset: xr.Dataset) -> None:
        """
        This test verifies that the _extract_time_str method correctly uses pandas to_datetime to extract a time string when the time value is a numpy datetime64. The test checks that when the Time coordinate contains a numpy datetime64, the method converts it to a pandas Timestamp using pd.to_datetime and then formats it into the expected time string, confirming that it can handle numpy datetime64 values and produce correctly formatted time strings for use in filenames or logging. 

        Parameters:
            time_dataset (xr.Dataset): The dataset containing a numpy datetime64.

        Returns:
            None
        """
        time_val = time_dataset.Time.values[0]
        assert not hasattr(time_val, "strftime"), "Fixture must contain numpy datetime64"
        result = MPASDateTimeUtils._extract_time_str(time_dataset, 0)
        assert result == "20240115T06"


class TestLogTimeInfo:
    """Tests for MPASDateTimeUtils._log_time_info."""

    def test_silent_mode_produces_no_output(self: 'TestLogTimeInfo') -> None:
        """
        This test verifies that the _log_time_info method produces no output when verbose is set to False, regardless of the presence of a time string or context. The test checks that when verbose is False, the method does not print anything to the standard output, confirming that it correctly respects the silent mode setting and suppresses all logging output for use in filenames or logging when verbose is disabled. 

        Parameters:
            None

        Returns:
            None
        """
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            MPASDateTimeUtils._log_time_info(False, 0, "20240115T06", "precip")
        assert captured.getvalue() == ""

    def test_time_str_present_no_context(self: 'TestLogTimeInfo') -> None:
        """
        This test verifies that the _log_time_info method correctly logs the time string when it is present and no context is provided. The test checks that when a time string is available and context is an empty string, the method prints a message to the standard output that includes the time string, confirming that it can log time information for use in filenames or logging even when no additional context is provided. 

        Parameters:
            None

        Returns:
            None
        """
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            MPASDateTimeUtils._log_time_info(True, 0, "20240115T06", "")
        output = captured.getvalue()
        assert "corresponds to: 20240115T06" in output
        assert "using variable" not in output

    def test_time_str_present_with_context(self: 'TestLogTimeInfo') -> None:
        """
        This test verifies that the _log_time_info method correctly logs the time string along with the provided context when both are present. The test checks that when a time string is available and context is provided, the method prints a message to the standard output that includes both the time string and the context (e.g., variable name), confirming that it can log comprehensive time information for use in filenames or logging when additional context is available. 

        Parameters:
            None

        Returns:
            None
        """
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            MPASDateTimeUtils._log_time_info(True, 0, "20240115T06", "precip")
        assert "using variable: precip" in captured.getvalue()

    def test_error_present_no_context(self: 'TestLogTimeInfo') -> None:
        """
        This test verifies that the _log_time_info method correctly logs the error message when an error is present and no context is provided. The test checks that when an error is passed to the method and context is an empty string, the method prints a message to the standard output that includes the time index and the error message, confirming that it can log error information for use in filenames or logging even when no additional context is provided. 

        Parameters:
            None

        Returns:
            None
        """
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            MPASDateTimeUtils._log_time_info(True, 2, None, "", error=RuntimeError("oops"))
        output = captured.getvalue()
        assert "Using time index 2" in output
        assert "oops" in output
        assert "using variable" not in output

    def test_error_present_with_context(self: 'TestLogTimeInfo') -> None:
        """
        This test verifies that the _log_time_info method correctly logs the error message along with the provided context when both are present. The test checks that when an error is passed to the method and context is provided, the method prints a message to the standard output that includes the time index, the context (e.g., variable name), and the error message, confirming that it can log comprehensive error information for use in filenames or logging when additional context is available. 

        Parameters:
            None

        Returns:
            None
        """
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            MPASDateTimeUtils._log_time_info(True, 2, None, "wind", error=RuntimeError("oops"))
        output = captured.getvalue()
        assert "using variable: wind" in output
        assert "oops" in output

    def test_no_time_str_no_error_no_context(self: 'TestLogTimeInfo') -> None:
        """
        This test verifies that the _log_time_info method correctly logs the time index and a message about the time coordinate not being available when no time string, no error, and no context are provided. The test checks that when the time string is None, error is None, and context is an empty string, the method prints a message to the standard output that includes the time index and indicates that the time coordinate is not available, confirming that it can log relevant information about the time index and the absence of time information even when no additional context or error details are provided. 

        Parameters:
            None

        Returns:
            None
        """
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            MPASDateTimeUtils._log_time_info(True, 3, None, "")
        output = captured.getvalue()
        assert "Using time index 3" in output
        assert "time coordinate not available" in output
        assert "using variable" not in output

    def test_no_time_str_no_error_with_context(self: 'TestLogTimeInfo') -> None:
        """
        This test verifies that the _log_time_info method correctly logs the time index along with the provided context when no time string and no error are present. The test checks that when the time string is None, error is None, and context is provided, the method prints a message to the standard output that includes the time index and the context (e.g., variable name), confirming that it can log relevant information about the time index and context even when no specific time information or error details are available. 

        Parameters:
            None

        Returns:
            None
        """
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            MPASDateTimeUtils._log_time_info(True, 3, None, "temperature")
        assert "using variable: temperature" in captured.getvalue()


class TestGetTimeInfo:
    """Tests for MPASDateTimeUtils.get_time_info."""

    def test_exception_in_extract_returns_fallback(self: 'TestGetTimeInfo', 
                                                   time_dataset: xr.Dataset) -> None:
        """
        This test verifies that the get_time_info method correctly returns a fallback string in the format "time_{index}" when an exception occurs in the _extract_time_str method. The test checks that when _extract_time_str raises an exception, get_time_info catches it and returns a string that includes the time index, confirming that it can handle errors gracefully and provide a meaningful fallback value for use in filenames or logging when time extraction fails. 

        Parameters:
            time_dataset (xr.Dataset): The dataset containing the time dimension.

        Returns:
            None
        """
        with patch.object(
            MPASDateTimeUtils, "_extract_time_str", side_effect=RuntimeError("forced")
        ):
            result = MPASDateTimeUtils.get_time_info(time_dataset, 0, verbose=False)
        assert result == "time_0"

    def test_exception_in_extract_verbose_logs_error(self: 'TestGetTimeInfo', 
                                                     time_dataset: xr.Dataset) -> None:
        """
        This test verifies that the get_time_info method correctly logs the error message when an exception occurs in the _extract_time_str method and verbose is set to True. The test checks that when _extract_time_str raises an exception, get_time_info catches it, logs the error message to the standard output, and returns a fallback string in the format "time_{index}", confirming that it can handle errors gracefully while providing informative logging for use in filenames or logging when time extraction fails. 

        Parameters:
            time_dataset (xr.Dataset): The dataset containing the time dimension.

        Returns:
            None
        """
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            with patch.object(
                MPASDateTimeUtils, "_extract_time_str", side_effect=RuntimeError("forced")
            ):
                result = MPASDateTimeUtils.get_time_info(time_dataset, 0, verbose=True)
        assert result == "time_0"
        assert "forced" in captured.getvalue()

    def test_valid_dataset_returns_time_string(self: 'TestGetTimeInfo', 
                                               time_dataset: xr.Dataset) -> None:
        """
        This test verifies that the get_time_info method correctly returns the extracted time string when provided with a valid dataset containing a Time coordinate and a valid time index. The test checks that when the dataset has a Time coordinate and the time index is within the valid range, the method returns the expected time string (e.g., "20240115T06"), confirming that it can successfully extract and format time information for use in filenames or logging when provided with valid input. 

        Parameters:
            time_dataset (xr.Dataset): The dataset containing the time coordinate.

        Returns:
            None
        """
        result = MPASDateTimeUtils.get_time_info(time_dataset, 0, verbose=False)
        assert result == "20240115T06"

    def test_no_time_coord_returns_fallback(self: 'TestGetTimeInfo', 
                                            no_time_dataset: xr.Dataset) -> None:
        """
        This test verifies that the get_time_info method correctly returns a fallback string in the format "time_{index}" when the provided dataset does not contain a Time coordinate. The test checks that when the dataset lacks a Time coordinate, the method returns a string that includes the time index, confirming that it can handle the absence of time information gracefully and provide a meaningful fallback value for use in filenames or logging when no Time coordinate is available. 

        Parameters:
            no_time_dataset (xr.Dataset): The dataset without a time coordinate.

        Returns:
            None
        """
        result = MPASDateTimeUtils.get_time_info(no_time_dataset, 0, verbose=False)
        assert result == "time_0"


class TestGetTimeRange:
    """Tests for MPASDateTimeUtils.get_time_range."""

    def test_none_dataset_raises(self: 'TestGetTimeRange') -> None:
        """
        This test verifies that the get_time_range method raises a ValueError when the provided dataset is None. The test checks that when None is passed as the dataset argument, the method raises an exception with an appropriate error message, confirming that it correctly validates the input and does not allow a None dataset to be processed for time range extraction. 

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="cannot be None"):
            MPASDateTimeUtils.get_time_range(None)  # type: ignore[arg-type]

    def test_no_time_coord_raises(self: 'TestGetTimeRange', 
                                  no_time_dataset: xr.Dataset) -> None:
        """
        This test verifies that the get_time_range method raises a ValueError when the provided dataset does not contain a Time coordinate. The test checks that when the dataset lacks a Time coordinate, the method raises an exception with an appropriate error message indicating that a Time coordinate is required, confirming that it correctly handles the absence of time information and does not attempt to extract a time range when no Time coordinate is available. 

        Parameters:
            no_time_dataset (xr.Dataset): The dataset without a time coordinate.

        Returns:
            None
        """ 
        with pytest.raises(ValueError, match="Time coordinate"):
            MPASDateTimeUtils.get_time_range(no_time_dataset)

    def test_valid_dataset_returns_start_end(self: 'TestGetTimeRange', 
                                             time_dataset: xr.Dataset) -> None:
        """
        This test verifies that the get_time_range method correctly returns the start and end datetime objects when provided with a valid dataset containing a Time coordinate. The test checks that when the dataset has a Time coordinate, the method returns two datetime objects representing the minimum and maximum times in the Time coordinate, confirming that it can successfully extract the time range for use in further processing or logging. The test also verifies that the returned start time is less than the end time and that they match the expected values based on the input dataset. 

        Parameters:
            time_dataset (xr.Dataset): The dataset with a time coordinate.

        Returns:
            None
        """
        start, end = MPASDateTimeUtils.get_time_range(time_dataset)
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)
        assert start < end
        assert start == datetime(2024, 1, 15, 6, 0, 0)
        assert end == datetime(2024, 1, 15, 10, 0, 0)


class TestFormatTimeForFilename:
    """Tests for MPASDateTimeUtils.format_time_for_filename."""

    def setup_method(self: 'TestFormatTimeForFilename') -> None:
        """
        This setup method initializes a datetime object with a specific date and time (January 15, 2024, at 06:30:45) that will be used in the subsequent tests for the format_time_for_filename method. The datetime object is stored as an instance variable (self.dt) so that it can be easily accessed and reused across multiple test methods, ensuring consistency in the input datetime for testing different formatting options. 

        Parameters:
            None

        Returns:
            None
        """
        self.dt = datetime(2024, 1, 15, 6, 30, 45)

    def test_mpas_format(self: 'TestFormatTimeForFilename') -> None:
        """
        This test verifies that the format_time_for_filename method correctly formats a datetime object in the "mpas" format. The test checks that the method returns a string in the expected format (e.g., "2024-01-15_06.30.45"), confirming that it correctly converts the datetime object to the desired string representation for use in filenames or logging when the "mpas" format is specified. 

        Parameters:
            None

        Returns:
            None
        """
        assert MPASDateTimeUtils.format_time_for_filename(self.dt, "mpas") == "2024-01-15_06.30.45"

    def test_iso_format(self: 'TestFormatTimeForFilename') -> None:
        """
        This test verifies that the format_time_for_filename method correctly formats a datetime object in the "iso" format. The test checks that the method returns a string in the expected ISO format (e.g., "20240115T063045"), confirming that it correctly converts the datetime object to the desired string representation for use in filenames or logging when the "iso" format is specified. 

        Parameters:
            None

        Returns:
            None
        """
        assert MPASDateTimeUtils.format_time_for_filename(self.dt, "iso") == "20240115T063045"

    def test_compact_format(self: 'TestFormatTimeForFilename') -> None:
        """
        This test verifies that the format_time_for_filename method correctly formats a datetime object in the "compact" format. The test checks that the method returns a string in the expected compact format (e.g., "2024011506"), confirming that it correctly converts the datetime object to the desired string representation for use in filenames or logging when the "compact" format is specified. The test also verifies that the compact format includes only the year, month, day, and hour components of the datetime, confirming that it produces a concise string suitable for use in filenames. 

        Parameters:
            None

        Returns:
            None
        """
        assert MPASDateTimeUtils.format_time_for_filename(self.dt, "compact") == "2024011506"

    def test_unknown_format_raises(self: 'TestFormatTimeForFilename') -> None:
        """
        This test verifies that the format_time_for_filename method raises a ValueError when an unknown format type is provided. The test checks that when an invalid format type (e.g., "invalid") is passed to the method, it raises an exception with an appropriate error message indicating that the format type is unknown, confirming that it correctly validates the input and does not allow unsupported format types for formatting datetime objects for use in filenames or logging. 

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="Unknown format_type"):
            MPASDateTimeUtils.format_time_for_filename(self.dt, "invalid")


class TestParseTimeFromString:
    """Tests for MPASDateTimeUtils.parse_time_from_string."""

    def test_default_patterns_mpas_format(self: 'TestParseTimeFromString') -> None:
        """
        This test verifies that the parse_time_from_string method correctly parses a datetime string in the "mpas" format using the default patterns. The test checks that when a string in the "mpas" format (e.g., "2024-01-15_06.30.45") is provided, the method returns the expected datetime object, confirming that it can successfully parse datetime strings in the "mpas" format without requiring custom patterns. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASDateTimeUtils.parse_time_from_string("2024-01-15_06.30.45")
        assert result == datetime(2024, 1, 15, 6, 30, 45)

    def test_default_patterns_iso_compact_format(self: 'TestParseTimeFromString') -> None:
        """
        This test verifies that the parse_time_from_string method correctly parses a datetime string in the "iso" compact format using the default patterns. The test checks that when a string in the "iso" compact format (e.g., "20240115T063045") is provided, the method returns the expected datetime object, confirming that it can successfully parse datetime strings in the "iso" compact format without requiring custom patterns. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASDateTimeUtils.parse_time_from_string("20240115T063045")
        assert result == datetime(2024, 1, 15, 6, 30, 45)

    def test_default_patterns_standard_format(self: 'TestParseTimeFromString') -> None:
        """
        This test verifies that the parse_time_from_string method correctly parses a datetime string in a standard format (e.g., "2024-01-15 06:30:45") using the default patterns. The test checks that when a string in a common datetime format is provided, the method returns the expected datetime object, confirming that it can successfully parse datetime strings in standard formats without requiring custom patterns. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASDateTimeUtils.parse_time_from_string("2024-01-15 06:30:45")
        assert result == datetime(2024, 1, 15, 6, 30, 45)

    def test_custom_pattern_used(self: 'TestParseTimeFromString') -> None:
        """
        This test verifies that the parse_time_from_string method correctly uses a custom pattern to parse a datetime string when provided. The test checks that when a string is given along with a list of custom patterns (e.g., ["%d/%m/%Y"]), the method successfully parses the string according to the specified pattern and returns the expected datetime object, confirming that it can handle custom patterns for parsing datetime strings when the default patterns do not match the input format. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASDateTimeUtils.parse_time_from_string("15/01/2024", ["%d/%m/%Y"])
        assert result == datetime(2024, 1, 15)

    def test_unparseable_string_raises(self: 'TestParseTimeFromString') -> None:
        """
        This test verifies that the parse_time_from_string method raises a ValueError when the provided string cannot be parsed using any of the default or custom patterns. The test checks that when an unparseable string (e.g., "not-a-date") is given along with a list of patterns that do not match the string, the method raises an exception with an appropriate error message indicating that the string could not be parsed, confirming that it correctly handles invalid input and does not return an incorrect datetime object when parsing fails. 

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="Could not parse"):
            MPASDateTimeUtils.parse_time_from_string("not-a-date", ["%Y%m%d"])

    def test_none_patterns_uses_defaults(self: 'TestParseTimeFromString') -> None:
        """
        This test verifies that the parse_time_from_string method correctly falls back to using the default patterns when None is provided for the patterns argument. The test checks that when a valid datetime string is given and None is passed for the patterns, the method successfully parses the string using the default patterns and returns the expected datetime object, confirming that it can handle a None value for patterns by utilizing its built-in default parsing logic. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASDateTimeUtils.parse_time_from_string("2024-01-15_06.30.45", None)
        assert result == datetime(2024, 1, 15, 6, 30, 45)


class TestGetTimeBounds:
    """Tests for MPASDateTimeUtils.get_time_bounds."""

    def test_none_dataset_returns_none_none(self: 'TestGetTimeBounds') -> None:
        """
        This test verifies that the get_time_bounds method returns (None, None) when the provided dataset is None. The test checks that when None is passed as the dataset argument, the method returns a tuple containing (None, None), confirming that it correctly handles a None dataset by returning a default value indicating that time bounds are not available. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASDateTimeUtils.get_time_bounds(None, 0)  # type: ignore[arg-type]
        assert result == (None, None)

    def test_no_bounds_variable_returns_none_none(self: 'TestGetTimeBounds', 
                                                  time_dataset: xr.Dataset) -> None:
        """
        This test verifies that the get_time_bounds method returns (None, None) when the provided dataset does not contain a time_bnds or Time_bnds variable. The test checks that when a dataset with a Time coordinate but without any bounds variable is passed, the method returns a tuple containing (None, None), confirming that it correctly handles the absence of bounds information by returning a default value indicating that time bounds are not available. 

        Parameters:
            time_dataset (xr.Dataset): The dataset to test.

        Returns:
            None
        """
        result = MPASDateTimeUtils.get_time_bounds(time_dataset, 0)
        assert result == (None, None)

    def test_time_bnds_variable_returns_bounds(self: 'TestGetTimeBounds') -> None:
        """
        This test verifies that the get_time_bounds method correctly returns the start and end times when the provided dataset contains a time_bnds variable. The test checks that when a dataset with a time_bnds variable is passed, the method returns datetime objects for the start and end times, and that the start time is earlier than the end time, confirming that it correctly extracts the time bounds from a dataset with a time_bnds variable for use in further processing or logging. 

        Parameters:
            None

        Returns:
            None
        """
        bounds = np.array([
            [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")],
            [np.datetime64("2024-01-02"), np.datetime64("2024-01-03")],
        ], dtype="datetime64[ns]")
        ds = xr.Dataset({"time_bnds": xr.DataArray(bounds, dims=["time", "bnds"])})
        start, end = MPASDateTimeUtils.get_time_bounds(ds, 0)
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)
        assert start < end

    def test_Time_bnds_variable_returns_bounds(self: 'TestGetTimeBounds') -> None:
        """
        This test verifies that the get_time_bounds method correctly returns the start and end times when the provided dataset contains a Time_bnds variable. The test checks that when a dataset with a Time_bnds variable is passed, the method returns datetime objects for the start and end times, and that the start time is earlier than the end time, confirming that it correctly extracts the time bounds from a dataset with a Time_bnds variable for use in further processing or logging. 

        Parameters:
            None

        Returns:
            None
        """
        bounds = np.array([
            [np.datetime64("2024-03-01"), np.datetime64("2024-03-02")],
        ], dtype="datetime64[ns]")
        ds = xr.Dataset({"Time_bnds": xr.DataArray(bounds, dims=["Time", "bnds"])})
        start, end = MPASDateTimeUtils.get_time_bounds(ds, 0)
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)

    def test_index_error_in_bounds_returns_none_none(self: 'TestGetTimeBounds') -> None:
        """
        This test verifies that the get_time_bounds method returns (None, None) when an IndexError occurs while trying to access the time bounds for a given index. The test checks that when a dataset with a time_bnds variable is passed and the specified time index exceeds the available range, the method catches the IndexError and returns a tuple containing (None, None), confirming that it correctly handles out-of-range indices when accessing time bounds by returning a default value indicating that time bounds are not available. 

        Parameters:
            None

        Returns:
            None
        """
        bounds = np.array(
            [[np.datetime64("2024-01-01"), np.datetime64("2024-01-02")]],
            dtype="datetime64[ns]",
        )
        ds = xr.Dataset({"time_bnds": xr.DataArray(bounds, dims=["time", "bnds"])})
        result = MPASDateTimeUtils.get_time_bounds(ds, 99)
        assert result == (None, None)


class TestCalculateTimeDelta:
    """Tests for MPASDateTimeUtils.calculate_time_delta."""

    def test_none_dataset_raises(self: 'TestCalculateTimeDelta') -> None:
        """
        This test verifies that the calculate_time_delta method raises a ValueError when the provided dataset is None. The test checks that when None is passed as the dataset argument, the method raises an exception with an appropriate error message indicating that the dataset cannot be None, confirming that it correctly validates the input and does not allow a None dataset to be processed for time delta calculation. 

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError):
            MPASDateTimeUtils.calculate_time_delta(None)  # type: ignore[arg-type]

    def test_no_time_coord_raises(self: 'TestCalculateTimeDelta', 
                                  no_time_dataset: xr.Dataset) -> None:
        """
        This test verifies that the calculate_time_delta method raises a ValueError when the provided dataset does not contain a Time coordinate. The test checks that when the dataset lacks a Time coordinate, the method raises an exception with an appropriate error message indicating that a Time coordinate is required, confirming that it correctly handles the absence of time information and does not attempt to calculate a time delta when no Time coordinate is available. 

        Parameters:
            no_time_dataset (xr.Dataset): A dataset without a time coordinate.

        Returns:
            None
        """
        with pytest.raises(ValueError):
            MPASDateTimeUtils.calculate_time_delta(no_time_dataset)

    def test_single_time_step_raises(self: 'TestCalculateTimeDelta') -> None:
        """
        This test verifies that the calculate_time_delta method raises a ValueError when the provided dataset contains only a single time step. The test checks that when a dataset with a Time coordinate that has only one time value is passed, the method raises an exception with an appropriate error message indicating that at least two time steps are required to calculate a time delta, confirming that it correctly validates the input and does not attempt to calculate a time delta when there is insufficient time information available. 

        Parameters:
            None

        Returns:
            None
        """
        times = pd.date_range("2024-01-01", periods=1, freq="1h")
        ds = xr.Dataset(coords={"Time": times})
        with pytest.raises(ValueError, match="at least 2"):
            MPASDateTimeUtils.calculate_time_delta(ds)

    def test_valid_dataset_returns_correct_timedelta(self: 'TestCalculateTimeDelta') -> None:
        """
        This test verifies that the calculate_time_delta method correctly calculates and returns the time delta as a pandas Timedelta object when provided with a valid dataset containing a Time coordinate with multiple time steps. The test checks that when a dataset with a Time coordinate that has at least two time values is passed, the method returns a Timedelta object representing the difference between the first two time steps, and that the returned value matches the expected time delta based on the input dataset (e.g., 6 hours), confirming that it can successfully calculate the time delta for use in further processing or logging. 

        Parameters:
            None

        Returns:
            None
        """
        times = pd.date_range("2024-01-01", periods=5, freq="6h")
        ds = xr.Dataset(coords={"Time": times})
        delta = MPASDateTimeUtils.calculate_time_delta(ds)
        assert isinstance(delta, pd.Timedelta)
        assert delta == pd.Timedelta("6h")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
