#!/usr/bin/env python3

"""
MPASdiag Test Suite: Precipitation Diagnostics Coverage

This module contains unit tests for the PrecipitationDiagnostics class in the mpasdiag.diagnostics.precipitation module. The tests are designed to cover all lines of code in the get_accumulation_hours and compute_precipitation_difference methods, as well as any error paths and main flow logic.

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
from io import StringIO
from unittest.mock import MagicMock, patch

from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics


@pytest.fixture
def precip_ds() -> xr.Dataset:
    """
    This fixture creates a synthetic precipitation dataset with two variables, "rainnc" and "rainc", representing non-convective and convective precipitation, respectively. The dataset contains 10 time steps and 50 cells, with values generated using a random number generator for testing purposes. The time coordinate is created as a range of hourly timestamps starting from January 1, 2024. 

    Parameters:
        None

    Returns:
        xr.Dataset: A synthetic precipitation dataset.
    """
    rng = np.random.default_rng(42)
    n_t, n_c = 10, 50

    rainnc = np.cumsum(rng.uniform(0.1, 0.5, (n_t, n_c)), axis=0)
    rainc = np.cumsum(rng.uniform(0.05, 0.3, (n_t, n_c)), axis=0)

    times = pd.date_range("2024-01-01T00:00:00", periods=n_t, freq="1h")

    return xr.Dataset(
        {
            "rainnc": xr.DataArray(rainnc, dims=["Time", "nCells"]),
            "rainc": xr.DataArray(rainc, dims=["Time", "nCells"]),
        },
        coords={"Time": times},
    )


@pytest.fixture
def diag() -> PrecipitationDiagnostics:
    """
    This fixture creates an instance of the PrecipitationDiagnostics class with verbose mode disabled. It can be used in tests that do not require verbose output. 

    Parameters:
        None

    Returns:
        PrecipitationDiagnostics: An instance of the PrecipitationDiagnostics class with verbose=False. 
    """
    return PrecipitationDiagnostics(verbose=False)


@pytest.fixture
def diag_v() -> PrecipitationDiagnostics:
    """
    This fixture creates an instance of the PrecipitationDiagnostics class with verbose mode enabled. It can be used in tests that require verbose output to verify that diagnostic messages are printed as expected. 

    Parameters:
        None

    Returns:
        PrecipitationDiagnostics: An instance of the PrecipitationDiagnostics class with verbose=True. 
    """
    return PrecipitationDiagnostics(verbose=True)


class TestGetAccumulationHours:
    """ Tests for PrecipitationDiagnostics.get_accumulation_hours, covering known periods and default behavior. """

    @pytest.mark.parametrize("period,hours", [
        ("a01h", 1), ("a03h", 3), ("a06h", 6), ("a12h", 12), ("a24h", 24),
    ])
    def test_known_periods(self: 'TestGetAccumulationHours', 
                           diag: 'PrecipitationDiagnostics', 
                           period: str, 
                           hours: int) -> None:
        """
        This test verifies that the get_accumulation_hours method correctly returns the expected number of hours for known accumulation period strings. It uses parameterization to test multiple period strings and their corresponding expected hours in a single test function. 

        Parameters:
            diag (PrecipitationDiagnostics): An instance of the PrecipitationDiagnostics class.
            period (str): The accumulation period string to test (e.g., "a01h").
            hours (int): The expected number of hours corresponding to the period.  

        Returns:
            None
        """
        assert diag.get_accumulation_hours(period) == hours

    def test_unknown_period_defaults_to_24(self: 'TestGetAccumulationHours', 
                                           diag: 'PrecipitationDiagnostics') -> None:
        """
        This test verifies that the get_accumulation_hours method defaults to returning 24 hours when an unknown accumulation period string is provided. This ensures that the method has a fallback behavior for unrecognized input. 

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class.

        Returns:
            None
        """
        assert diag.get_accumulation_hours("unknown") == 24


class TestComputePrecipitationDifference:
    """ Tests for PrecipitationDiagnostics.compute_precipitation_difference, covering error handling, normal operation, and edge cases. """

    def test_none_dataset_raises(self: 'TestComputePrecipitationDifference', 
                                 diag: 'PrecipitationDiagnostics') -> None:
        """
        This test verifies that the compute_precipitation_difference method raises a ValueError when a None dataset is provided. This ensures that the method properly checks for the presence of a dataset before attempting to compute precipitation differences, preventing potential errors from null references. 

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class.

        Returns:
            None
        """
        with pytest.raises(ValueError, match="Dataset not provided"):
            diag.compute_precipitation_difference(None, 0)  # type: ignore[arg-type]

    def test_time_index_out_of_bounds_raises(self: 'TestComputePrecipitationDifference', 
                                             diag: 'PrecipitationDiagnostics', 
                                             precip_ds: xr.Dataset) -> None:
        """
        This test verifies that the compute_precipitation_difference method raises a ValueError when the specified time index exceeds the bounds of the dataset. This ensures that the method properly checks for valid time indices before attempting to access data, preventing potential index errors and ensuring robust error handling. 

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class.
            precip_ds (xr.Dataset): A dataset containing precipitation data.

        Returns:
            None
        """
        with pytest.raises(ValueError, match="exceeds dataset size"):
            diag.compute_precipitation_difference(precip_ds, 999)

    def test_normal_accumulation_rainnc(self: 'TestComputePrecipitationDifference', 
                                        diag: 'PrecipitationDiagnostics', 
                                        precip_ds: xr.Dataset) -> None:
        """
        This test verifies that the compute_precipitation_difference method correctly computes the precipitation difference for the "rainnc" variable under normal conditions. It checks that the result is a DataArray with appropriate attributes and non-negative values, which are expected for precipitation differences. This ensures that the method functions correctly under typical usage scenarios. 

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class.
            precip_ds (xr.Dataset): A dataset containing precipitation data.

        Returns:
            None
        """
        result = diag.compute_precipitation_difference(precip_ds, 3, "rainnc", "a01h")
        assert isinstance(result, xr.DataArray)
        assert result.attrs["units"] == "mm"
        assert result.attrs["accumulation_hours"] == 1
        assert np.all(result.values >= 0)

    def test_total_variable_silent(self: 'TestComputePrecipitationDifference', 
                                   diag: 'PrecipitationDiagnostics', 
                                   precip_ds: xr.Dataset) -> None:
        """
        This test verifies that the compute_precipitation_difference method correctly computes the precipitation difference for the "total" variable in silent mode. It ensures that the method returns a DataArray with non-negative values, which are expected for precipitation differences, without producing verbose output. This ensures that the method functions correctly when verbose output is not desired.

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class.
            precip_ds (xr.Dataset): A dataset containing precipitation data.

        Returns:
            None
        """
        result = diag.compute_precipitation_difference(precip_ds, 3, "total", "a01h")
        assert isinstance(result, xr.DataArray)
        assert np.all(result.values >= 0)

    def test_total_variable_verbose_triggers_diagnostic(self: 'TestComputePrecipitationDifference', 
                                                        diag_v: 'PrecipitationDiagnostics', 
                                                        precip_ds: xr.Dataset) -> None:
        """
        This test verifies that the compute_precipitation_difference method triggers diagnostic output when computing the "total" variable in verbose mode. It captures the standard output during the method call and checks that it contains information about the computation, such as range information or a message indicating that the computation is taking place. This ensures that the method provides useful diagnostic information when verbose mode is enabled.

        Parameters:
            diag_v ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class in verbose mode.
            precip_ds (xr.Dataset): A dataset containing precipitation data.

        Returns:
            None
        """
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            result = diag_v.compute_precipitation_difference(precip_ds, 3, "total", "a01h")

        assert isinstance(result, xr.DataArray)
        assert "range" in captured.getvalue().lower() or "Computing" in captured.getvalue()

    def test_time_index_zero_returns_data(self: 'TestComputePrecipitationDifference', 
                                          diag: 'PrecipitationDiagnostics', 
                                          precip_ds: xr.Dataset) -> None:
        """
        This test verifies that the compute_precipitation_difference method correctly handles the case when the time index is zero. It ensures that the method returns a DataArray for the specified variable at time index zero, which is expected to be valid since it should not require a lookback period. This test confirms that the method can handle edge cases related to time indexing without errors. 

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class.
            precip_ds (xr.Dataset): A dataset containing precipitation data.

        Returns:
            None
        """
        result = diag.compute_precipitation_difference(precip_ds, 0, "rainnc", "a01h")
        assert isinstance(result, xr.DataArray)

    def test_insufficient_lookback_returns_zero_field(self: 'TestComputePrecipitationDifference', 
                                                      diag: 'PrecipitationDiagnostics', 
                                                      precip_ds: xr.Dataset) -> None:
        """
        This test verifies that the compute_precipitation_difference method returns a zero field when the lookback period is insufficient. It ensures that the method returns a DataArray with a "note" attribute indicating the insufficient lookback, which is expected behavior when there is not enough data to compute a valid difference. This test confirms that the method can gracefully handle cases where the lookback period cannot be satisfied without producing errors. 

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class.
            precip_ds (xr.Dataset): A dataset containing precipitation data.

        Returns:
            None
        """
        result = diag.compute_precipitation_difference(precip_ds, 2, "rainnc", "a06h")
        assert isinstance(result, xr.DataArray)
        assert "note" in result.attrs

    def test_verbose_mode_prints_output(self: 'TestComputePrecipitationDifference', 
                                        diag_v: 'PrecipitationDiagnostics', 
                                        precip_ds: xr.Dataset) -> None:
        """
        This test verifies that the compute_precipitation_difference method produces verbose output when verbose mode is enabled. It captures the standard output during the method call and checks that it contains diagnostic information, such as messages about the computation process or range information. This ensures that the method provides useful feedback to the user when verbose mode is active, which can aid in understanding the computation and diagnosing potential issues. 

        Parameters:
            diag_v ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class with verbose mode enabled.
            precip_ds (xr.Dataset): A dataset containing precipitation data.

        Returns:
            None
        """
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            diag_v.compute_precipitation_difference(precip_ds, 3, "rainnc", "a01h")

        assert len(captured.getvalue()) > 0

    def test_lowercase_time_dim(self: 'TestComputePrecipitationDifference', 
                                diag: 'PrecipitationDiagnostics') -> None:
        """
        This test verifies that the compute_precipitation_difference method can handle a dataset with a lowercase time dimension name. It creates a synthetic dataset with a "time" dimension instead of "Time" and checks that the method can compute the precipitation difference without errors. This ensures that the method is flexible in handling different time dimension naming conventions, which can be common in various datasets. 

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class.

        Returns:
            None
        """
        rng = np.random.default_rng(1)
        data = np.cumsum(rng.uniform(0.1, 0.5, (5, 20)), axis=0)
        ds = xr.Dataset({"rainnc": xr.DataArray(data, dims=["time", "nCells"])})
        result = diag.compute_precipitation_difference(ds, 2, "rainnc", "a01h")
        assert isinstance(result, xr.DataArray)


class TestExtractVariableAtTime:
    """ Tests for PrecipitationDiagnostics._extract_variable_at_time, covering both xarray and uxarray paths, as well as error handling. """

    def test_xarray_path(self: 'TestExtractVariableAtTime', 
                         diag: 'PrecipitationDiagnostics', 
                         precip_ds: xr.Dataset) -> None:
        """
        This test verifies that the _extract_variable_at_time method correctly extracts a variable at a specified time index using the xarray path. It checks that the result is a DataArray with the expected shape, which confirms that the method is correctly accessing the data for the given time index and variable name. This ensures that the method functions properly when using xarray for data manipulation. 

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class.
            precip_ds (xr.Dataset): A dataset containing precipitation data.

        Returns:
            None
        """
        result = diag._extract_variable_at_time(precip_ds, "rainnc", 2, "Time", "xarray")
        assert isinstance(result, xr.DataArray)
        assert result.shape == (50,)

    def test_uxarray_path(self: 'TestExtractVariableAtTime', 
                          diag: 'PrecipitationDiagnostics', 
                          precip_ds: xr.Dataset) -> None:
        """
        This test verifies that the _extract_variable_at_time method correctly extracts a variable at a specified time index using the uxarray path. It checks that the result is a DataArray, which confirms that the method is correctly accessing the data for the given time index and variable name using the uxarray approach. This ensures that the method functions properly when using uxarray for data manipulation.

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class.
            precip_ds (xr.Dataset): A dataset containing precipitation data.

        Returns:
            None
        """
        result = diag._extract_variable_at_time(precip_ds, "rainnc", 2, "Time", "uxarray")
        assert isinstance(result, xr.DataArray)


class TestExtractSampleDataForVariable:
    """Tests for PrecipitationDiagnostics._extract_sample_data_for_variable."""

    def test_total_sums_rainc_and_rainnc(self: 'TestExtractSampleDataForVariable', 
                                         diag: 'PrecipitationDiagnostics', 
                                         precip_ds: xr.Dataset) -> None:
        """
        This test verifies that the _extract_sample_data_for_variable method correctly computes the "total" variable by summing the "rainc" and "rainnc" variables at the specified time index. It checks that the result is close to the expected sum of the two variables, confirming that the method is correctly implementing the logic for computing the total precipitation. This ensures that the method functions properly when calculating derived variables based on multiple input variables. 

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class.
            precip_ds (xr.Dataset): A dataset containing precipitation data.

        Returns:
            None
        """
        result = diag._extract_sample_data_for_variable(precip_ds, "total", 0, "Time", "xarray")

        expected = (
            precip_ds["rainc"].isel(Time=0) + precip_ds["rainnc"].isel(Time=0)
        )

        assert np.allclose(result.values, expected.values)

    def test_single_variable_returned_directly(self: 'TestExtractSampleDataForVariable', 
                                               diag: 'PrecipitationDiagnostics', 
                                               precip_ds: xr.Dataset) -> None:
        """
        This test verifies that the _extract_sample_data_for_variable method correctly returns a single variable directly when the variable name is not "total". It checks that the result is close to the expected values of the specified variable at the given time index, confirming that the method is correctly accessing and returning the data for individual variables without unnecessary processing. This ensures that the method functions properly for straightforward variable extraction cases. 

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class.
            precip_ds (xr.Dataset): A dataset containing precipitation data.

        Returns:
            None
        """
        result = diag._extract_sample_data_for_variable(precip_ds, "rainnc", 0, "Time", "xarray")
        assert np.allclose(result.values, precip_ds["rainnc"].isel(Time=0).values)


class TestHandleFirstTimeStep:
    """ Tests for PrecipitationDiagnostics._handle_first_time_step and _handle_insufficient_lookback, covering verbose output, return values, and error handling. """

    def test_time_index_zero_verbose_prints_message(self: 'TestHandleFirstTimeStep', 
                                                    diag_v: 'PrecipitationDiagnostics', 
                                                    precip_ds: xr.Dataset) -> None:
        """
        This test verifies that the _handle_time_index_zero method correctly prints a message when the time index is zero and verbose mode is enabled. It captures the standard output during the method call and checks that it contains a message indicating that the time index is zero, which is expected behavior for this edge case. This ensures that the method provides useful diagnostic information to the user when handling the first time step in verbose mode. 

        Parameters:
            diag_v ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class with verbose mode enabled.
            precip_ds (xr.Dataset): A dataset containing precipitation data.

        Returns:
            None
        """
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            result = diag_v._handle_time_index_zero(precip_ds, "Time", "rainnc", "a01h", 1, "xarray")

        assert "Time index 0" in captured.getvalue()
        assert isinstance(result, xr.DataArray)

    def test_insufficient_lookback_verbose(self: 'TestHandleFirstTimeStep', 
                                           diag_v: 'PrecipitationDiagnostics', 
                                           precip_ds: xr.Dataset) -> None:
        """
        This test verifies that the _handle_insufficient_lookback method correctly prints a warning message when verbose mode is enabled. It captures the standard output during the method call and checks that it contains a warning about insufficient lookback, which is expected behavior when there is not enough data to compute a valid difference. This ensures that the method provides useful feedback to the user about potential issues with the lookback period when verbose mode is active. 

        Parameters:
            diag_v ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class with verbose mode enabled.
            precip_ds (xr.Dataset): A dataset containing precipitation data.

        Returns:
            None
        """
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            result = diag_v._handle_insufficient_lookback(
                precip_ds, "Time", 2, "rainnc", "a06h", 6, "xarray"
            )

        output = captured.getvalue()

        assert "Warning" in output
        assert "lookback" in output.lower() or "zero" in output.lower()
        assert isinstance(result, xr.DataArray)

    def test_insufficient_lookback_silent(self: 'TestHandleFirstTimeStep', 
                                          diag: 'PrecipitationDiagnostics', 
                                          precip_ds: xr.Dataset) -> None:
        """
        This test verifies that the _handle_insufficient_lookback method returns a DataArray with a "note" attribute when verbose mode is disabled. It checks that the method returns a DataArray even when there is insufficient lookback, and that the "note" attribute is present to indicate the issue. This ensures that the method can handle insufficient lookback cases gracefully without producing verbose output, while still providing information about the situation through attributes. 

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class with verbose mode disabled.
            precip_ds (xr.Dataset): A dataset containing precipitation data.

        Returns:
            None
        """
        result = diag._handle_insufficient_lookback(
            precip_ds, "Time", 2, "rainnc", "a06h", 6, "xarray"
        )

        assert isinstance(result, xr.DataArray)
        assert result.attrs.get("note") is not None

    def test_exception_in_handle_first_time_step_raises_value_error(self: 'TestHandleFirstTimeStep', 
                                                                    diag: 'PrecipitationDiagnostics', 
                                                                    precip_ds: xr.Dataset) -> None:
        """
        This test verifies that the _handle_first_time_step method raises a ValueError when an exception occurs while handling the first time step. It simulates an error by providing a nonexistent variable name, which should trigger an exception in the method. The test checks that the exception is properly raised and that it contains a message indicating the issue with handling the first time step. This ensures that the method has robust error handling for unexpected issues during the processing of the first time step. 

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class.
            precip_ds (xr.Dataset): A dataset containing precipitation data.

        Returns:
            None
        """
        with pytest.raises(ValueError, match="Cannot handle time index"):
            diag._handle_first_time_step(
                precip_ds, "Time", 0, "NONEXISTENT", "a01h", 1, "xarray"
            )

    def test_exception_in_handle_first_time_step_verbose_prints_error(self: 'TestHandleFirstTimeStep', 
                                                                      diag_v: 'PrecipitationDiagnostics', 
                                                                      precip_ds: xr.Dataset) -> None:
        """
        This test verifies that the _handle_first_time_step method correctly prints an error message when an exception occurs while handling the first time step in verbose mode. It simulates an error by providing a nonexistent variable name, which should trigger an exception in the method. The test captures the standard output during the method call and checks that it contains an error message indicating the issue with handling the first time step, confirming that the method provides useful diagnostic information when verbose mode is enabled. This ensures that the method has robust error handling and informative output for users when issues arise during processing. 

        Parameters:
            diag_v ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class with verbose mode enabled.
            precip_ds (xr.Dataset): A dataset containing precipitation data.

        Returns:
            None
        """
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            with pytest.raises(ValueError):
                diag_v._handle_first_time_step(
                    precip_ds, "Time", 0, "NONEXISTENT", "a01h", 1, "xarray"
                )

        assert "Error handling first time step" in captured.getvalue()


class TestPrintCurrentPreviousComparison:
    """ Tests for PrecipitationDiagnostics._print_current_previous_comparison, covering normal comparisons, warnings for lower current max, and error handling during min/max extraction. """

    def test_normal_comparison_no_warning(self: 'TestPrintCurrentPreviousComparison', 
                                          diag_v: 'PrecipitationDiagnostics') -> None:
        """
        This test verifies that the _print_current_previous_comparison method correctly prints the range of current and previous data without issuing a warning when the current data's maximum is greater than or equal to the previous data's maximum. It captures the standard output during the method call and checks that it contains information about the current and previous ranges, while ensuring that no warning messages are present. This confirms that the method behaves as expected under normal comparison conditions. 

        Parameters:
            diag_v ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class with verbose mode enabled.

        Returns:
            None
        """
        current = xr.DataArray(np.array([1.0, 2.0, 3.0]))
        previous = xr.DataArray(np.array([0.5, 1.0, 1.5]))
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            diag_v._print_current_previous_comparison(current, previous, "rainnc")

        output = captured.getvalue()

        assert "Current rainnc range" in output
        assert "WARNING" not in output

    def test_current_max_less_than_previous_prints_warning(self: 'TestPrintCurrentPreviousComparison', 
                                                           diag_v: 'PrecipitationDiagnostics') -> None:
        """
        This test verifies that the _print_current_previous_comparison method correctly prints a warning message when the current data's maximum is less than the previous data's maximum. It captures the standard output during the method call and checks that it contains a warning indicating that the current maximum is lower than the previous maximum, which is expected behavior in this scenario. This ensures that the method provides useful diagnostic feedback to the user when there is a potential issue with the current data compared to previous data. 

        Parameters:
            diag_v ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class with verbose mode enabled.

        Returns:
            None
        """
        current = xr.DataArray(np.array([0.1, 0.2, 0.3]))
        previous = xr.DataArray(np.array([1.0, 2.0, 3.0]))  # larger max
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            diag_v._print_current_previous_comparison(current, previous, "test")

        assert "WARNING" in captured.getvalue()

    def test_exception_in_min_max_prints_error(self: 'TestPrintCurrentPreviousComparison', 
                                               diag_v: 'PrecipitationDiagnostics') -> None:
        """
        This test verifies that the _print_current_previous_comparison method correctly prints an error message when an exception occurs during the extraction of minimum and maximum values from the data. It simulates an error by mocking the min method of the data to raise a RuntimeError, which should trigger the exception handling in the method. The test captures the standard output during the method call and checks that it contains an error message indicating that the analysis could not be performed, confirming that the method has robust error handling for issues during min/max extraction. This ensures that users receive informative feedback when unexpected errors occur during data analysis. 

        Parameters:
            diag_v ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class with verbose mode enabled.

        Returns:
            None
        """
        mock_data = MagicMock()
        mock_data.min.side_effect = RuntimeError("forced error")
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            diag_v._print_current_previous_comparison(mock_data, mock_data, "test")

        assert "Could not analyze" in captured.getvalue()


class TestComputeResultStatistics:
    """ Tests for PrecipitationDiagnostics._compute_result_statistics, covering normal statistics computation, handling of all NaN data, and error handling during statistics computation. """

    def test_normal_data_returns_dict(self: 'TestComputeResultStatistics', 
                                      diag: 'PrecipitationDiagnostics') -> None:
        """
        This test verifies that the _compute_result_statistics method correctly computes statistics for normal data and returns a dictionary containing the expected keys. It checks that the returned dictionary includes keys for "min", "max", "mean", and "total_count", which are essential statistics for analyzing precipitation data. This ensures that the method functions properly under typical conditions and provides comprehensive statistical information about the result data. 

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class.

        Returns:
            None
        """
        data = xr.DataArray(np.array([0.0, 0.5, 1.0, 2.0, 0.0]))
        stats = diag._compute_result_statistics(data)
        assert stats is not None
        assert "min" in stats and "max" in stats and "mean" in stats
        assert stats["total_count"] == 5

    def test_all_nan_returns_none(self: 'TestComputeResultStatistics', 
                                  diag: 'PrecipitationDiagnostics') -> None:
        """
        This test verifies that the _compute_result_statistics method returns None when all data values are NaN. It checks that the method correctly identifies the lack of finite values and returns None instead of attempting to compute statistics, which would not be meaningful in this case. This ensures that the method can gracefully handle cases where the data does not contain valid values for analysis. 

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class.

        Returns:
            None
        """
        data = xr.DataArray(np.full(10, np.nan))
        result = diag._compute_result_statistics(data)
        assert result is None

    def test_exception_returns_none_and_prints(self: 'TestComputeResultStatistics', 
                                               diag: 'PrecipitationDiagnostics') -> None:
        """
        This test verifies that the _compute_result_statistics method returns None and prints an error message when an exception occurs during the computation of statistics. It simulates an error by mocking the flatten method of the data values to raise a RuntimeError, which should trigger the exception handling in the method. The test captures the standard output during the method call and checks that it contains an error message indicating that the statistics could not be computed, confirming that the method has robust error handling for unexpected issues during statistics computation. This ensures that users receive informative feedback when errors occur during data analysis. 

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class.

        Returns:
            None
        """
        mock_data = MagicMock()
        mock_data.values.flatten.side_effect = RuntimeError("forced")
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            result = diag._compute_result_statistics(mock_data)

        assert result is None
        assert "Could not compute result statistics" in captured.getvalue()


class TestPrintResultDataAnalysis:
    """ Tests for PrecipitationDiagnostics._print_result_data_analysis, covering normal data analysis, handling of all NaN data, and error handling during analysis. """

    def test_normal_data_prints_stats(self: 'TestPrintResultDataAnalysis', 
                                      diag_v: 'PrecipitationDiagnostics') -> None:
        """
        This test verifies that the _print_result_data_analysis method correctly prints statistical information when provided with normal data. It captures the standard output during the method call and checks that it contains information about the range of the data, which is expected to be part of the analysis output. This ensures that the method provides useful diagnostic information to the user when analyzing result data in verbose mode. 

        Parameters:
            diag_v ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class with verbose mode enabled.

        Returns:
            None
        """
        data = xr.DataArray(np.linspace(0, 5, 20))
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            diag_v._print_result_data_analysis(data, "rainnc")

        assert "range" in captured.getvalue().lower()

    def test_all_nan_prints_warning_and_returns(self: 'TestPrintResultDataAnalysis', 
                                                diag_v: 'PrecipitationDiagnostics') -> None:
        """
        This test verifies that the _print_result_data_analysis method correctly prints a warning message and returns immediately when all data values are NaN. It captures the standard output during the method call and checks that it contains a message indicating that there are no finite values to analyze, which is expected behavior in this scenario. This ensures that the method can gracefully handle cases where the result data does not contain valid values for analysis, while providing informative feedback to the user. 

        Parameters:
            diag_v ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class with verbose mode enabled.

        Returns:
            None
        """
        data = xr.DataArray(np.full(10, np.nan))
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            diag_v._print_result_data_analysis(data, "test")

        assert "No finite values" in captured.getvalue()


class TestAnalyzePrecipitationDiagnostics:
    """ Tests for PrecipitationDiagnostics._analyze_precipitation_diagnostics, covering behavior when verbose mode is disabled, analysis with current and previous data, analysis with result data only, and error handling during analysis. """

    def test_verbose_false_returns_immediately(self: 'TestAnalyzePrecipitationDiagnostics', 
                                               diag: 'PrecipitationDiagnostics') -> None:
        """
        This test verifies that the _analyze_precipitation_diagnostics method returns immediately without printing anything when verbose mode is disabled. It captures the standard output during the method call and checks that it is empty, confirming that the method does not perform any analysis or produce output when verbose mode is not active. This ensures that the method behaves as expected in silent mode, providing a clean output without unnecessary information. 

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class.

        Returns:
            None
        """
        current = xr.DataArray(np.array([1.0, 2.0]))
        result_data = xr.DataArray(np.array([0.5, 1.0]))
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            diag._analyze_precipitation_diagnostics(current, current, result_data, "test")

        assert captured.getvalue() == ""

    def test_with_current_and_previous_verbose(self: 'TestAnalyzePrecipitationDiagnostics', 
                                               diag_v: 'PrecipitationDiagnostics') -> None:
        """
        This test verifies that the _analyze_precipitation_diagnostics method correctly prints analysis information when both current and previous data are provided in verbose mode. It captures the standard output during the method call and checks that it contains information about the current and previous data, such as their ranges or a comparison message. This ensures that the method provides useful diagnostic information to the user when analyzing both current and previous precipitation data in verbose mode. 

        Parameters:
            diag_v ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class with verbose mode enabled.

        Returns:
            None
        """
        current = xr.DataArray(np.array([1.0, 2.0, 3.0]))
        previous = xr.DataArray(np.array([0.5, 1.0, 1.5]))
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            diag_v._analyze_precipitation_diagnostics(current, previous, var_context="rainnc")

        assert len(captured.getvalue()) > 0

    def test_with_result_data_only_verbose(self: 'TestAnalyzePrecipitationDiagnostics', 
                                           diag_v: 'PrecipitationDiagnostics') -> None:
        """
        This test verifies that the _analyze_precipitation_diagnostics method correctly prints analysis information when only result data is provided in verbose mode. It captures the standard output during the method call and checks that it contains information about the result data, such as its range or a message indicating that the analysis is being performed. This ensures that the method provides useful diagnostic information to the user when analyzing result data without current or previous comparisons in verbose mode. 

        Parameters:
            diag_v ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class with verbose mode enabled.

        Returns:
            None
        """
        result_data = xr.DataArray(np.linspace(0, 2, 10))
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            diag_v._analyze_precipitation_diagnostics(result_data=result_data, var_context="test")

        assert len(captured.getvalue()) > 0


class TestPrintTimeSliceInfo:
    """ Tests for PrecipitationDiagnostics._print_time_slice_info, covering behavior when verbose mode is disabled, printing time slice information with time coordinates, printing index information without time coordinates, and error handling during dimension access. """

    def test_verbose_false_returns_immediately(self: 'TestPrintTimeSliceInfo', 
                                               diag: 'PrecipitationDiagnostics', 
                                               precip_ds: xr.Dataset) -> None:
        """
        This test verifies that the _print_time_slice_info method returns immediately without printing anything when verbose mode is disabled. It captures the standard output during the method call and checks that it is empty, confirming that the method does not perform any analysis or produce output when verbose mode is not active. This ensures that the method behaves as expected in silent mode, providing a clean output without unnecessary information. 

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class.
            precip_ds (xr.Dataset): A dataset containing precipitation data.

        Returns:
            None
        """
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            diag._print_time_slice_info(precip_ds, 3, "rainnc", 1)

        assert captured.getvalue() == ""

    def test_with_time_coords_prints_info(self: 'TestPrintTimeSliceInfo', 
                                           diag_v: 'PrecipitationDiagnostics', 
                                           precip_ds: xr.Dataset) -> None:
        """
        This test verifies that the _print_time_slice_info method correctly prints time slice information when the dataset contains a Time coordinate and verbose mode is enabled. It captures the standard output during the method call and checks that it contains information about the time slice, such as the time value corresponding to the specified index. This ensures that the method provides useful diagnostic information to the user about the time slice being analyzed when verbose mode is active and time coordinates are available in the dataset. 

        Parameters:
            diag_v ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class with verbose mode enabled.
            precip_ds (xr.Dataset): A dataset containing precipitation data with time coordinates.

        Returns:
            None
        """
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            diag_v._print_time_slice_info(precip_ds, 3, "rainnc", 1)

        assert "Time slice info" in captured.getvalue()

    def test_without_time_coords_prints_index_info(self: 'TestPrintTimeSliceInfo', 
                                                   diag_v: 'PrecipitationDiagnostics') -> None:
        """
        This test verifies that the _print_time_slice_info method correctly prints index information when the dataset does not contain a Time coordinate and verbose mode is enabled. It creates a synthetic dataset without time coordinates and captures the standard output during the method call, checking that it contains information about the indices being accessed. This ensures that the method provides useful diagnostic information to the user about the time slice being analyzed even when time coordinates are not available in the dataset, as long as verbose mode is active. 

        Parameters:
            diag_v ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class with verbose mode enabled.

        Returns:
            None
        """
        ds_no_coord = xr.Dataset(
            {"rainnc": xr.DataArray(np.ones((5, 10)), dims=["Time", "nCells"])}
        )

        captured = StringIO()

        with patch("sys.stdout", new=captured):
            diag_v._print_time_slice_info(ds_no_coord, 3, "rainnc", 1)

        assert "indices" in captured.getvalue()

    def test_exception_in_dims_access_prints_error(self: 'TestPrintTimeSliceInfo', 
                                                   diag_v: 'PrecipitationDiagnostics') -> None:
        """
        This test verifies that the _print_time_slice_info method correctly prints an error message when an exception occurs during access to the dataset's dimensions. It simulates an error by mocking the dims attribute of the dataset to raise a RuntimeError, which should trigger the exception handling in the method. The test captures the standard output during the method call and checks that it contains an error message indicating that the time slice information could not be printed, confirming that the method has robust error handling for issues during dimension access. This ensures that users receive informative feedback when unexpected errors occur while trying to print time slice information. 

        Parameters:
            diag_v ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class with verbose mode enabled.

        Returns:
            None
        """
        mock_ds = MagicMock()
        mock_ds.dims.__contains__ = MagicMock(side_effect=RuntimeError("dims error"))
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            diag_v._print_time_slice_info(mock_ds, 1, "rainnc", 1)

        assert "Could not print time slice info" in captured.getvalue()


class TestCreatePrecipitationFieldWithAttributes:
    """ Tests for PrecipitationDiagnostics._create_precipitation_field_with_attributes, covering behavior for normal fields, insufficient data, and negative values. """

    def test_normal_field_has_correct_attrs(self: 'TestCreatePrecipitationFieldWithAttributes', 
                                            diag: 'PrecipitationDiagnostics') -> None:
        """
        This test verifies that the _create_precipitation_field_with_attributes method correctly creates a precipitation field with the expected attributes for a normal case. It checks that the resulting DataArray has the correct units, accumulation hours, and does not contain a note attribute when there is sufficient data. This ensures that the method functions properly under typical conditions and assigns appropriate metadata to the created precipitation field. 

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class.

        Returns:
            None
        """
        data = xr.DataArray(np.array([0.5, 1.0, 2.0]))
        result = diag._create_precipitation_field_with_attributes(data, "rainnc", "a01h", 1)
        assert result.attrs["units"] == "mm"
        assert result.attrs["accumulation_hours"] == 1
        assert "note" not in result.attrs

    def test_insufficient_data_flag_adds_note(self: 'TestCreatePrecipitationFieldWithAttributes', 
                                               diag: 'PrecipitationDiagnostics') -> None:
        """
        This test verifies that the _create_precipitation_field_with_attributes method correctly adds a note attribute indicating insufficient data when the is_insufficient_data flag is set to True. It checks that the resulting DataArray contains a note attribute that mentions insufficient data, and that the long_name attribute also reflects this issue. This ensures that the method provides appropriate metadata to indicate when there is not enough data to compute a valid precipitation field, which can be important for downstream analysis and interpretation. 

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class.

        Returns:
            None
        """
        data = xr.DataArray(np.zeros(5))

        result = diag._create_precipitation_field_with_attributes(
            data, "rainnc", "a06h", 6, is_insufficient_data=True
        )

        assert "note" in result.attrs
        assert "insufficient" in result.attrs["note"].lower()
        assert "insufficient" in result.attrs["long_name"].lower()

    def test_negative_values_clamped_to_zero(self: 'TestCreatePrecipitationFieldWithAttributes', 
                                              diag: 'PrecipitationDiagnostics') -> None:
        """
        This test verifies that the _create_precipitation_field_with_attributes method correctly clamps negative values to zero in the resulting precipitation field. It checks that when the input data contains negative values, the output DataArray has all values greater than or equal to zero, confirming that the method enforces non-negativity for precipitation fields. This ensures that the method produces physically meaningful results, as negative precipitation values are not valid in this context. 

        Parameters:
            diag ('PrecipitationDiagnostics'): An instance of the PrecipitationDiagnostics class.

        Returns:
            None
        """
        data = xr.DataArray(np.array([-1.0, 0.5, 2.0]))
        result = diag._create_precipitation_field_with_attributes(data, "rainnc", "a01h", 1)
        assert np.all(result.values >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
