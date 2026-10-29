#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Core Diagnostics Module: Precipitation Diagnostics

This module provides core diagnostic functions for analyzing and validating precipitation-related data from MPAS model output. It includes utilities for validating coordinate arrays, checking data quality, and summarizing statistical properties of precipitation fields. The diagnostics are designed to ensure data integrity and identify potential issues before visualization or further analysis. The module is optimized for efficiency on large unstructured mesh datasets typical of MPAS, while providing detailed diagnostic information when problems are detected. It serves as a foundational component for building more complex precipitation diagnostics and visualizations in the MPASdiag framework. It includes comprehensive validation checks for coordinate correctness, data completeness, and statistical outlier detection, helping users identify and address data quality issues in their precipitation analyses. By centralizing precipitation diagnostic functionality in this module, it promotes code reuse and simplifies the implementation of new analysis types that require precipitation-related calculations in the future. The module also includes error handling for missing or inconsistent precipitation data and provides informative messages to guide users in selecting appropriate variables for their analysis. This ensures that the processing modules can maintain robustness in handling precipitation data across all analysis types, facilitating accurate comparisons and visualizations of MPAS precipitation fields.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

# Load necessary libraries
import re

import numpy as np
import pandas as pd
import xarray as xr
from typing import Any, Optional, cast

from mpasdiag.processing.utils_logger import get_logger

logger = get_logger(__name__)


class InsufficientLookbackError(ValueError):
    """Raised when an accumulation period reaches back before the first available time step, so the accumulation cannot be computed from the loaded data."""


class PrecipitationDiagnostics:
    """Computes precipitation-related diagnostics for MPAS model output, including temporal differencing for accumulation calculations, data quality checks, and statistical summaries."""

    def __init__(self: "PrecipitationDiagnostics", verbose: bool = True) -> None:
        """
        This constructor initializes the PrecipitationDiagnostics class with an optional verbose flag that controls the level of diagnostic output during precipitation calculations. When verbose mode is enabled, the class will print detailed messages about the processing steps, any issues encountered, and statistical summaries of the precipitation data being analyzed. This allows users to gain insights into the data quality and identify potential problems during the accumulation calculation process. The verbose flag can be set to False to suppress diagnostic messages for cleaner output when running in production or when detailed diagnostics are not needed.

        Parameters:
            verbose (bool): Enable verbose output messages during precipitation calculations (default: True).

        Returns:
            None
        """
        # Enable verbose output
        self.verbose = verbose

    def compute_precipitation_difference(
        self: "PrecipitationDiagnostics",
        dataset: xr.Dataset,
        time_index: int,
        var_name: str = "rainnc",
        accum_period: str = "a01h",
        data_type: str = "xarray",
    ) -> xr.DataArray:
        """
        This function computes the precipitation accumulation for a specified variable and accumulation period by performing temporal differencing on the MPAS dataset. The start of the accumulation window is located from the dataset's time coordinate (the output at exactly accum_hours before the valid time), so the result is correct for any output interval; if the dataset has no usable time coordinate an hourly output interval is assumed and a warning is logged. If the window would begin before the first available time step (including time index 0 for any period) no accumulation can be formed and InsufficientLookbackError is raised instead of returning a placeholder field. Negative or out-of-range time indices raise ValueError. The function applies quality filters to ensure physically realistic precipitation values and annotates the resulting DataArray with CF-compliant attributes. When verbose mode is enabled, it provides detailed diagnostic output about the time slice selection, current and previous data comparisons, and statistics of the computed accumulation to help users identify potential data issues during processing.

        Parameters:
            dataset (xr.Dataset): MPAS dataset containing precipitation variables.
            time_index (int): Time index for current time step.
            var_name (str): Precipitation variable name ('rainc', 'rainnc', or 'total') (default: 'rainnc').
            accum_period (str): Accumulation period identifier (e.g., 'a01h', 'a03h', 'a06h', 'a12h', 'a24h') (default: 'a01h').
            data_type (str): Dataset type specification, either 'xarray' or 'uxarray' (default: 'xarray').

        Returns:
            xr.DataArray: Precipitation accumulation data array in millimeters with CF-compliant attributes including long_name, accumulation_period, and accumulation_hours metadata.

        Raises:
            InsufficientLookbackError: If the accumulation window starts before the first available time step or no output exists at the window start.
            ValueError: If the dataset is missing, the time index is out of range (negative indices count from the end) or the accumulation period is not of the form 'aNNh'.
        """
        # Validate dataset input and raise error if not provided
        if dataset is None:
            raise ValueError("Dataset not provided. Pass a valid MPAS dataset.")

        # Determine time dimension and size
        time_dim = "Time" if "Time" in dataset.dims else "time"
        time_size = dataset.sizes[time_dim]

        # Validate time index is within bounds of the dataset size; negative indices count from the end (Python-style)
        if time_index < -time_size or time_index >= time_size:
            raise ValueError(
                f"Time index {time_index} out of range for dataset size {time_size}"
            )
        if time_index < 0:
            time_index += time_size

        # Determine the accumulation length in hours from the period identifier
        accum_hours = self.get_accumulation_hours(accum_period)

        # Locate the start of the accumulation window from the actual time coordinate (raises InsufficientLookbackError when unavailable)
        previous_index = self._resolve_previous_index(
            dataset, time_dim, time_index, accum_hours, accum_period
        )
        time_step_diff = time_index - previous_index

        # Perform temporal differencing to compute accumulation for the specified variable and accumulation period
        if self.verbose:
            logger.info(
                "Computing %s-hour accumulation for period: %s",
                accum_hours,
                accum_period,
            )
            self._print_time_slice_info(dataset, time_index, var_name, time_step_diff)

        # Compute the accumulated precipitation based on the specified variable name, handling 'total' as a special case that sums 'rainc' and 'rainnc'
        if var_name == "total":
            accumulated_precipitation = self._compute_total_precipitation_accumulation(
                dataset, time_index, previous_index, time_dim, data_type
            )
        else:
            accumulated_precipitation = self._compute_single_variable_accumulation(
                dataset, var_name, time_index, previous_index, time_dim, data_type
            )

        # Apply quality filters and set attributes on the resulting accumulated precipitation field
        accumulated_precipitation = self._apply_precipitation_filters_and_attributes(
            accumulated_precipitation, var_name
        )

        # Update attributes to include accumulation period and hours for clarity in downstream use
        accumulated_precipitation.attrs.update(
            {
                "long_name": f"{accum_hours}-hour accumulated precipitation from {var_name}",
                "accumulation_period": accum_period,
                "accumulation_hours": accum_hours,
            }
        )

        # Attach valid-time and window-start metadata so downstream plotting annotates the correct accumulation window
        self._attach_time_metadata(
            accumulated_precipitation, dataset, time_dim, time_index, previous_index
        )

        # Provide diagnostic output about the computed accumulation, including range and spatial coverage, to help identify potential issues in the results
        if self.verbose:
            logger.info(
                "Computed %s-hour accumulated precipitation for period: %s",
                accum_hours,
                accum_period,
            )
            self._analyze_precipitation_diagnostics(
                result_data=accumulated_precipitation, var_context=var_name
            )

        # Return accumulated precipitation DataArray
        return accumulated_precipitation

    def _resolve_previous_index(
        self: "PrecipitationDiagnostics",
        dataset: xr.Dataset,
        time_dim: str,
        time_index: int,
        accum_hours: int,
        accum_period: str,
    ) -> int:
        """
        This helper function finds the time index at which an accumulation window starts, i.e. the output time exactly accum_hours before the valid time at time_index, using the dataset's time coordinate rather than assuming hourly output. When the time coordinate is missing or not interpretable as datetimes (for example undecoded numeric times or all-NaT values) it falls back to the legacy assumption of one output per hour, logging a warning, so the start index is time_index minus accum_hours. If the window would begin before the first available time step, or the output interval has no time step exactly at the window start (for example a 1-hour accumulation from 3-hourly output), InsufficientLookbackError is raised with a descriptive message so callers can skip the step instead of plotting a misleading field.

        Parameters:
            dataset (xr.Dataset): MPAS dataset containing the time coordinate.
            time_dim (str): Name of the time dimension in the dataset.
            time_index (int): Index of the valid (end) time of the accumulation.
            accum_hours (int): Accumulation length in hours.
            accum_period (str): Accumulation period identifier used in error messages.

        Returns:
            int: Index of the time step at which the accumulation window starts.
        """
        # Try to read the time coordinate as datetimes
        times = None
        if time_dim in dataset.variables or time_dim in dataset.coords:
            try:
                raw_times = dataset[time_dim].values
                if raw_times.dtype.kind in ("M", "O"):
                    times = pd.to_datetime(raw_times)
                    if times.isna().any():
                        times = None
            except (TypeError, ValueError):
                times = None

        # Without a usable time coordinate, assume hourly output and warn
        if times is None:
            logger.warning(
                "No usable time coordinate for '%s'; assuming hourly output when computing %s accumulation",
                time_dim,
                accum_period,
            )
            previous_index = time_index - accum_hours
            if previous_index < 0:
                raise InsufficientLookbackError(
                    f"Insufficient lookback for {accum_period} at time index {time_index}: "
                    f"{accum_hours} hourly steps are required but only {time_index} precede it"
                )
            return int(previous_index)

        # Compute the start time of the accumulation window from the valid time
        valid_time = times[time_index]
        start_time = valid_time - pd.Timedelta(hours=accum_hours)

        # The window cannot start before the first available time step
        if start_time < times[0]:
            raise InsufficientLookbackError(
                f"Insufficient lookback for {accum_period} at time index {time_index} "
                f"(valid {valid_time.isoformat()}): window starts at {start_time.isoformat()}, "
                f"before the first available time {times[0].isoformat()}"
            )

        # Find the output exactly at the window start (allowing one minute of tolerance)
        offsets = np.abs((times - start_time).total_seconds())
        match = int(np.argmin(offsets[: time_index + 1]))
        if offsets[match] > 60.0:
            raise InsufficientLookbackError(
                f"No output at {start_time.isoformat()} to start the {accum_period} accumulation "
                f"ending {valid_time.isoformat()}; the output interval does not divide {accum_hours} h"
            )
        return match

    def _attach_time_metadata(
        self: "PrecipitationDiagnostics",
        data: xr.DataArray,
        dataset: xr.Dataset,
        time_dim: str,
        time_index: int,
        previous_index: int,
    ) -> None:
        """
        This helper function attaches valid-time and accumulation window start metadata to the provided DataArray based on the time coordinate values from the dataset. It records the valid time of the current step as an ISO 8601 string and, if a valid previous index is provided, also records the accumulation window start time. This metadata is crucial for downstream plotting and analysis to correctly annotate the accumulation window associated with the precipitation data. The function includes error handling to ensure that any issues with accessing time metadata do not disrupt the overall processing of the precipitation data, making it a best-effort annotation that enhances the usability of the results without being a critical dependency.

        Parameters:
            data (xr.DataArray): Accumulation DataArray to annotate in place.
            dataset (xr.Dataset): Source MPAS dataset providing the time coordinate.
            time_dim (str): Name of the time dimension/coordinate in the dataset.
            time_index (int): Index of the current (valid) time step.
            previous_index (int): Index marking the start of the accumulation window; negative when undefined.

        Returns:
            None
        """
        try:
            # Resolve the time coordinate values; skip annotation if it is unavailable
            if time_dim not in dataset.variables and time_dim not in dataset.coords:
                return

            # Extract time values from the dataset for the specified time dimension
            times = dataset[time_dim].values

            # Record the valid time of the current step as an ISO 8601 string
            data.attrs["valid_time"] = pd.Timestamp(times[time_index]).isoformat()

            # Record the accumulation window start only when a valid previous index exists
            if previous_index >= 0:
                data.attrs["accumulation_start_time"] = pd.Timestamp(
                    times[previous_index]
                ).isoformat()
        except (KeyError, IndexError, ValueError, TypeError):
            # Time metadata is best-effort; never let annotation failures break the accumulation result
            return

    def get_accumulation_hours(
        self: "PrecipitationDiagnostics", accum_period: str
    ) -> int:
        """
        This helper function converts an accumulation period identifier of the form 'aNNh' (e.g., 'a01h', 'a03h', 'a06h', 'a12h', 'a24h', 'a48h') into its number of hours. Any whole number of hours from 1 to 240 is accepted (case-insensitive, optional zero padding), matching the plotter's rule. An identifier that does not match raises a ValueError instead of silently falling back to 24 hours, which would label and difference the wrong window.

        Parameters:
            accum_period (str): Accumulation period identifier following standard MPAS convention (e.g., 'a01h', 'a03h', 'a06h', 'a12h', 'a24h').

        Returns:
            int: Number of hours corresponding to the accumulation period.

        Raises:
            ValueError: If the identifier is not of the form 'aNNh' with 1 to 240 hours.
        """
        match = re.fullmatch(r"a(\d{1,3})h", str(accum_period).strip().lower())

        if match is None or not 1 <= int(match.group(1)) <= 240:
            raise ValueError(
                f"Unknown accumulation period '{accum_period}'. Use the form 'aNNh' "
                "with 1 to 240 hours, e.g. 'a01h', 'a03h', 'a06h', 'a12h', 'a24h'"
            )

        return int(match.group(1))

    def _extract_variable_at_time(
        self: "PrecipitationDiagnostics",
        dataset: xr.Dataset,
        var_name: str,
        time_index: int,
        time_dim: str,
        data_type: str,
    ) -> xr.DataArray:
        """
        This helper function extracts a specified variable at a given time index from the dataset, handling both 'xarray' and 'uxarray' data access styles. It checks for the presence of a compute method (e.g., for Dask arrays) and computes the data if necessary to ensure that the returned DataArray is in memory and ready for processing. By centralizing the variable extraction logic, this function promotes code reuse and ensures consistent handling of variable access across different dataset types and processing steps in the precipitation accumulation calculations.

        Parameters:
            dataset (xr.Dataset): Source MPAS dataset.
            var_name (str): Name of the variable to extract.
            time_index (int): Time index to select.
            time_dim (str): Name of the time dimension in `dataset`.
            data_type (str): Dataset access style, either 'xarray' or 'uxarray'.

        Returns:
            xr.DataArray: DataArray corresponding to the requested variable and time index.
        """
        if data_type == "uxarray":
            # For uxarray, .isel is not supported, so we use direct indexing
            data = dataset[var_name][time_index]
        else:
            # For xarray, we can use .isel to select the time index
            data = dataset[var_name].isel({time_dim: time_index})

        # Compute the data if it is a Dask array
        if hasattr(data, "compute"):
            data = cast(Any, data).compute()

        # Return the extracted variable data as a DataArray
        return data

    def _extract_precipitation_pair(
        self: "PrecipitationDiagnostics",
        dataset: xr.Dataset,
        var_name: str,
        current_index: int,
        previous_index: int,
        time_dim: str,
        data_type: str,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        This helper function extracts the current and previous time slices for a specified precipitation variable from the dataset, handling both 'xarray' and 'uxarray' access styles. It returns a tuple containing the current and previous DataArrays for the variable, which can then be used for temporal differencing to compute accumulations. By centralizing this extraction logic, it promotes code reuse and ensures consistent handling of variable access across different processing steps in the precipitation diagnostics. This function also allows for easy extension in the future if additional variables or access patterns need to be supported in the precipitation accumulation calculations.

        Parameters:
            dataset (xr.Dataset): Source MPAS dataset.
            var_name (str): Name of the precipitation variable.
            current_index (int): Index of the current time slice.
            previous_index (int): Index of the previous time slice used for differencing.
            time_dim (str): Name of the time dimension.
            data_type (str): Dataset access style, either 'xarray' or 'uxarray'.

        Returns:
            tuple[xr.DataArray, xr.DataArray]: (current_data, previous_data) DataArrays.
        """
        # Extract current time slice for the specified variable
        current_data = self._extract_variable_at_time(
            dataset, var_name, current_index, time_dim, data_type
        )

        # Extract previous time slice for the same variable
        previous_data = self._extract_variable_at_time(
            dataset, var_name, previous_index, time_dim, data_type
        )

        # Return the current and previous data as a tuple
        return current_data, previous_data

    def _compute_total_precipitation_accumulation(
        self: "PrecipitationDiagnostics",
        dataset: xr.Dataset,
        current_index: int,
        previous_index: int,
        time_dim: str,
        data_type: str,
    ) -> xr.DataArray:
        """
        This function computes the total precipitation accumulation by summing the contributions from both convective (rainc) and non-convective (rainnc) precipitation variables. It extracts the current and previous time slices for both variables, performs the necessary summations to get the total precipitation for each time slice, and then computes the accumulation by differencing the current total from the previous total. When verbose mode is enabled, it provides diagnostic output comparing the current and previous totals to help identify any potential issues in the data or processing steps. This function centralizes the logic for computing total precipitation accumulation, making it easier to maintain and extend in the future if additional precipitation components need to be included in the total calculation.

        Parameters:
            dataset (xr.Dataset): Source MPAS dataset.
            current_index (int): Index of the current time slice.
            previous_index (int): Index of the previous time slice used for differencing.
            time_dim (str): Name of the time dimension.
            data_type (str): Dataset access style, either 'xarray' or 'uxarray'.

        Returns:
            xr.DataArray: Accumulated total precipitation (current - previous) as a DataArray.
        """
        # Extract current and previous time slices for the convective precipitation variable (rainc) using a helper function to ensure consistent handling of variables
        current_rainc, previous_rainc = self._extract_precipitation_pair(
            dataset, "rainc", current_index, previous_index, time_dim, data_type
        )

        # Extract current and previous time slices for the non-convective precipitation variable (rainnc) using the same helper function to ensure consistent handling of variables
        current_rainnc, previous_rainnc = self._extract_precipitation_pair(
            dataset, "rainnc", current_index, previous_index, time_dim, data_type
        )

        # Calculate current total precipitation by summing the current convective and non-convective components
        current_total = current_rainc + current_rainnc

        # Calculate previous total precipitation by summing the previous convective and non-convective components
        previous_total = previous_rainc + previous_rainnc

        # Provide diagnostic output comparing current and previous total precipitation to identify potential issues in the data or processing steps
        if self.verbose:
            self._analyze_precipitation_diagnostics(
                current_total, previous_total, var_context="total"
            )

        # Return the difference between current and previous total precipitation as the accumulated value
        return current_total - previous_total

    def _compute_single_variable_accumulation(
        self: "PrecipitationDiagnostics",
        dataset: xr.Dataset,
        var_name: str,
        current_index: int,
        previous_index: int,
        time_dim: str,
        data_type: str,
    ) -> xr.DataArray:
        """
        This function computes the precipitation accumulation for a single specified variable by performing temporal differencing on the current and previous time slices extracted from the dataset. It handles both 'xarray' and 'uxarray' access styles and returns the accumulated precipitation as a DataArray. When verbose mode is enabled, it provides diagnostic output comparing the current and previous data to help identify any potential issues in the data or processing steps. This function centralizes the logic for computing single-variable precipitation accumulation, making it easier to maintain and extend in the future if additional variables or access patterns need to be supported in the precipitation diagnostics.

        Parameters:
            dataset (xr.Dataset): Source MPAS dataset.
            var_name (str): Precipitation variable name to process.
            current_index (int): Index of the current time slice.
            previous_index (int): Index of the previous time slice used for differencing.
            time_dim (str): Name of the time dimension.
            data_type (str): Dataset access style, either 'xarray' or 'uxarray'.

        Returns:
            xr.DataArray: Accumulated precipitation as DataArray (current - previous).
        """
        # Extract current and previous time slices for the specified variable using a helper function to ensure consistent handling of variables
        current_data, previous_data = self._extract_precipitation_pair(
            dataset, var_name, current_index, previous_index, time_dim, data_type
        )

        # Provide diagnostic output comparing current and previous data for the specified variable to identify potential issues in the data or processing steps
        if self.verbose:
            self._analyze_precipitation_diagnostics(
                current_data, previous_data, var_context=var_name
            )

        # Return the difference between current and previous data as the accumulated value for the specified variable
        return current_data - previous_data

    def _extract_sample_data_for_variable(
        self: "PrecipitationDiagnostics",
        dataset: xr.Dataset,
        var_name: str,
        time_index: int,
        time_dim: str,
        data_type: str,
    ) -> xr.DataArray:
        """
        This helper function extracts a sample data slice for a specified variable at a given time index, which is used in cases where the accumulation calculation cannot be performed due to insufficient lookback (e.g., at the first time step). It handles both 'xarray' and 'uxarray' access styles and ensures that the returned DataArray is in memory and ready for processing. This function centralizes the logic for extracting sample data for variables, making it easier to maintain and extend in the future if additional variables or access patterns need to be supported in the precipitation diagnostics. By providing a consistent way to extract sample data, it helps ensure that the handling of edge cases in the accumulation calculations is robust and informative for downstream users.

        Parameters:
            dataset (xr.Dataset): Source MPAS dataset.
            var_name (str): Variable name or 'total' for combined precipitation.
            time_index (int): Time index to extract sample from.
            time_dim (str): Name of the time dimension.
            data_type (str): Dataset access style, either 'xarray' or 'uxarray'.

        Returns:
            xr.DataArray: Sample DataArray used for first-step or insufficient lookback handling.
        """
        if var_name == "total":
            # Extract convective precipitation component for the specified time index
            rainc = self._extract_variable_at_time(
                dataset, "rainc", time_index, time_dim, data_type
            )

            # Extract non-convective precipitation component for the same time index
            rainnc = self._extract_variable_at_time(
                dataset, "rainnc", time_index, time_dim, data_type
            )

            # Return the sum of rainc and rainnc as the sample data for 'total' precipitation
            return rainc + rainnc
        else:
            # For a single variable, simply extract it at the specified time index
            return self._extract_variable_at_time(
                dataset, var_name, time_index, time_dim, data_type
            )

    def _create_precipitation_field_with_attributes(
        self: "PrecipitationDiagnostics",
        data: xr.DataArray,
        var_name: str,
        accum_period: str,
        accum_hours: int,
        is_insufficient_data: bool = False,
    ) -> xr.DataArray:
        """
        This helper function applies quality filters to the precipitation data and sets CF-compliant attributes, including a descriptive long_name that indicates the accumulation period and variable context. If the data is flagged as having insufficient lookback, it annotates the long_name accordingly to inform downstream users of the data limitation. This centralizes the logic for filtering and annotating precipitation fields, ensuring consistent metadata across different cases in the accumulation calculations. By providing clear and informative attributes, it helps users understand the context of the precipitation data they are working with, especially in cases where the accumulation calculation could not be performed due to insufficient historical data.

        Parameters:
            data (xr.DataArray): Raw precipitation data to filter and annotate.
            var_name (str): Variable context used to build descriptive long_name.
            accum_period (str): Accumulation period identifier (e.g., 'a01h').
            accum_hours (int): Number of accumulation hours.
            is_insufficient_data (bool): Flag indicating insufficient lookback (default: False).

        Returns:
            xr.DataArray: Filtered and metadata-annotated precipitation DataArray.
        """
        # Apply quality filters: NaN is preserved, negative differences (resets) and absurdly large values become NaN
        filtered_data = self._mask_invalid_precipitation(data, var_name)

        if is_insufficient_data:
            # If there is insufficient data for accumulation, indicate this in the long_name for clarity in downstream use
            long_name = f"{accum_hours}-hour accumulated precipitation from {var_name} (insufficient data)"
        else:
            # Construct a long_name that clearly indicates the accumulation period and variable context for clarity in downstream use
            long_name = f"{accum_hours}-hour accumulated precipitation from {var_name}"

        # Set attributes including units, standard_name, long_name, and accumulation metadata for clarity in downstream use
        filtered_data.attrs.update(
            {
                "units": "mm",
                "standard_name": "precipitation",
                "long_name": long_name,
                "accumulation_period": accum_period,
                "accumulation_hours": accum_hours,
            }
        )

        # Flag the data with an attribute if it is generated due to insufficient lookback, so that downstream code can handle it appropriately
        if is_insufficient_data:
            filtered_data.attrs["note"] = (
                f"Insufficient historical data for {accum_hours}-hour accumulation"
            )

        # Return the filtered and annotated precipitation DataArray
        return filtered_data

    def _mask_invalid_precipitation(
        self: "PrecipitationDiagnostics", data: xr.DataArray, var_context: str = ""
    ) -> xr.DataArray:
        """
        This helper function applies the physical-realism quality filter to a precipitation accumulation field without hiding problems. Missing values (NaN) are preserved as NaN. Differences more negative than -1e-6 mm, which indicate a bucket or restart reset or corrupted data, are set to NaN rather than clamped to zero, and a warning with the count of affected points is logged. Tiny negative values between -1e-6 and 0 mm are treated as floating-point round-off and set to zero. Values at or above 1e5 mm (and infinities) fail the upper sanity check and are set to NaN, also with a logged count. The input attributes are preserved.

        Parameters:
            data (xr.DataArray): Precipitation accumulation field in millimeters.
            var_context (str): Variable context used in the warning messages (default: "").

        Returns:
            xr.DataArray: Filtered field with invalid points set to NaN.
        """
        # Identify points that are genuinely negative (beyond round-off) and points that exceed the upper sanity threshold
        negative = data < -1.0e-6
        too_large = (data >= 1.0e5) | np.isinf(data)

        # Log the number of rejected points so that resets are not silently ignored
        n_negative = int(negative.sum())
        n_large = int(too_large.sum())
        label = var_context if var_context else "precipitation"
        if n_negative:
            logger.warning(
                "%s: %d point(s) with negative accumulation (bucket/restart reset?) set to NaN",
                label,
                n_negative,
            )
        if n_large:
            logger.warning("%s: %d point(s) >= 1e5 mm set to NaN", label, n_large)

        # Set invalid points to NaN, clamp round-off negatives to zero, and keep NaN as NaN
        filtered = data.where(~(negative | too_large))
        filtered = filtered.where(~((filtered < 0) & (filtered >= -1.0e-6)), 0.0)
        filtered.attrs = dict(data.attrs)

        # Return the filtered field
        return filtered

    def _apply_precipitation_filters_and_attributes(
        self: "PrecipitationDiagnostics", data: xr.DataArray, var_context: str = ""
    ) -> xr.DataArray:
        """
        This helper function applies quality filters to the precipitation data to ensure that values are physically realistic (e.g., non-negative and below an upper threshold). It also sets CF-compliant attributes, including units in millimeters, standard_name as 'precipitation', and a long_name that incorporates the variable context for clarity. By centralizing this logic, it promotes code reuse and ensures consistent filtering and metadata annotation across different steps in the precipitation diagnostics. This function is used both for handling edge cases (e.g., first time step) and for processing the results of temporal differencing to ensure that all precipitation DataArrays returned by the diagnostics have appropriate quality control and metadata.

        Parameters:
            data (xr.DataArray): Precipitation data array to be filtered and attributed.
            var_context (str): Variable context string for customizing the long_name attribute (default: "").

        Returns:
            xr.DataArray: Filtered precipitation data array with updated CF-compliant attributes including units in millimeters, standard_name as 'precipitation', and customized long_name.
        """
        # Apply quality filters: NaN is preserved, negative differences (resets) and absurdly large values become NaN
        data = self._mask_invalid_precipitation(data, var_context)

        # Construct a long_name that incorporates the variable context for clarity in downstream use
        long_name = (
            f"hourly accumulated precipitation from {var_context}"
            if var_context
            else "hourly accumulated precipitation"
        )

        # Set attributes to ensure the resulting DataArray is well-described and compliant with CF conventions
        data.attrs.update(
            {
                "units": "mm",
                "standard_name": "precipitation",
                "long_name": long_name,
            }
        )

        # Return the filtered and attributed precipitation DataArray
        return data

    def _extract_min_max_from_data(
        self: "PrecipitationDiagnostics", data: Any
    ) -> tuple[float, float]:
        """
        This helper function takes an array-like object (such as an xarray.DataArray or a numpy array) and extracts the minimum and maximum values, returning them as Python floats. This is used in diagnostic comparisons to summarize the range of values in the current and previous time slices of precipitation data, helping to identify potential issues such as decreasing cumulative values that may indicate data loading or differencing problems. By centralizing this logic, it promotes code reuse and ensures consistent handling of min/max extraction across different diagnostic functions in the precipitation analysis.

        Parameters:
            data (Any): Array-like object (xarray.DataArray or numpy array) supporting min()/max().

        Returns:
            tuple[float, float]: (minimum_value, maximum_value) as Python floats.
        """
        # Return the minimum and maximum values from the data
        return float(data.min()), float(data.max())

    def _print_current_previous_comparison(
        self: "PrecipitationDiagnostics",
        current_data: Any,
        previous_data: Any,
        var_context: str,
    ) -> None:
        """
        This function compares the current and previous time slice data for a specified variable context by extracting their minimum and maximum values and printing them in a formatted manner. It also checks if the current maximum is less than the previous maximum, which could indicate a potential issue with data loading or differencing, and prints a warning message if this condition is met. This diagnostic comparison helps users identify potential problems in the precipitation data before further processing or visualization steps. By centralizing this logic, it promotes code reuse and ensures consistent diagnostic output across different variables and contexts in the precipitation analysis.

        Parameters:
            current_data (Any): Current time slice data array.
            previous_data (Any): Previous time slice data array.
            var_context (str): Context label used in printed messages.

        Returns:
            None: Only prints messages; does not modify data.
        """
        try:
            # Extract minimum and maximum values from the current and previous data arrays for comparison
            current_min, current_max = self._extract_min_max_from_data(current_data)
            previous_min, previous_max = self._extract_min_max_from_data(previous_data)

            # Format the variable context for labeling the output messages
            variable_label = var_context if var_context else "precipitation"

            # Print the current range of values for the specified variable context
            logger.debug(
                "Current %s range: %.2f to %.2f mm",
                variable_label,
                current_min,
                current_max,
            )

            # Print the previous range for comparison
            logger.debug(
                "Previous %s range: %.2f to %.2f mm",
                variable_label,
                previous_min,
                previous_max,
            )

            # Check if the current maximum is less than the previous maximum
            if current_max < previous_max:
                logger.warning(
                    "Current max (%.2f) < Previous max (%.2f) - possible data loading issue",
                    current_max,
                    previous_max,
                )
        except Exception as e:
            logger.warning("Could not analyze current/previous data: %s", e)

    def _compute_result_statistics(
        self: "PrecipitationDiagnostics", result_data: Any
    ) -> Optional[dict[str, Any]]:
        """
        This function computes key statistics from the result data array, including minimum, maximum, mean, count of non-zero precipitation points, total count of finite values, and the percentage of non-zero points. It handles cases where there may be no finite values in the data by returning None and printing a warning message. This statistical summary provides insights into the range and spatial coverage of precipitation in the computed results, which can help identify potential issues such as excessive zero values or unrealistic ranges in the accumulation calculations. By centralizing this logic, it promotes code reuse and ensures consistent statistical analysis across different result data arrays in the precipitation diagnostics.

        Parameters:
            result_data (Any): Result DataArray or array-like object to analyze.

        Returns:
            Optional[dict[str, Any]]: Dictionary with keys 'min', 'max', 'mean', 'nonzero_count', 'total_count', 'nonzero_percentage', or None if no finite values available.
        """
        try:
            # Flatten the data values and filter out non-finite values to ensure accurate statistics
            data_values = result_data.values.flatten()
            finite_values = data_values[np.isfinite(data_values)]

            # If there are no finite values, return None to indicate that statistics cannot be computed
            if len(finite_values) == 0:
                return None

            # Compute minimum, maximum, and mean values from the finite data for summary statistics
            data_min = float(np.nanmin(finite_values))
            data_max = float(np.nanmax(finite_values))
            data_mean = np.nanmean(finite_values)

            # Count the number of points with precipitation greater than a small threshold (e.g., 0.01 mm)
            nonzero_count = int(np.sum(finite_values > 0.01))
            total_count = len(finite_values)

            # Calculate the percentage of non-zero precipitation points relative to the total number of finite points
            nonzero_percentage = 100 * nonzero_count / total_count

            # Return a dictionary containing the computed statistics for the result data array
            return {
                "min": data_min,
                "max": data_max,
                "mean": data_mean,
                "nonzero_count": nonzero_count,
                "total_count": total_count,
                "nonzero_percentage": nonzero_percentage,
            }
        # Catch any exceptions that occur during the computation of statistics
        except Exception as e:
            logger.warning("Could not compute result statistics: %s", e)
            return None

    def _print_result_data_analysis(
        self: "PrecipitationDiagnostics", result_data: Any, var_context: str
    ) -> None:
        """
        This function prints a comprehensive analysis of the result data array, including the range (min/max), mean, and spatial coverage of non-zero precipitation points. It uses the statistics computed by the `_compute_result_statistics` function to provide insights into the characteristics of the computed precipitation accumulation, which can help identify potential issues such as unrealistic ranges or excessive zero values. If there are no finite values in the result data, it prints a warning message instead. By centralizing this logic, it promotes code reuse and ensures consistent diagnostic output across different result data arrays in the precipitation analysis.

        Parameters:
            result_data (Any): Result DataArray or array-like object to analyze.
            var_context (str): Context label used in printed messages.

        Returns:
            None: Only prints messages; does not modify data.
        """
        # Compute key statistics from the result data array for analysis
        result_stats = self._compute_result_statistics(result_data)

        # Warn if there are no finite values in the result data
        if result_stats is None:
            logger.warning("No finite values found in result data")
            return

        # Format the variable context for labeling the output messages
        variable_label = var_context if var_context else "precipitation"

        # Print the range of values in the result data to help identify potential issues
        logger.info(
            "Result %s range: %.3f to %.3f mm",
            variable_label,
            result_stats["min"],
            result_stats["max"],
        )

        # Print the mean value of the result data for additional insight
        logger.info(
            "Result %s mean: %.3f mm",
            variable_label,
            result_stats["mean"],
        )

        # Print the count and percentage of points with non-zero precipitation
        logger.info(
            "Points with precipitation > 0.01 mm: %s/%s (%.1f%%)",
            f"{result_stats['nonzero_count']:,}",
            f"{result_stats['total_count']:,}",
            result_stats["nonzero_percentage"],
        )

    def _analyze_precipitation_diagnostics(
        self: "PrecipitationDiagnostics",
        current_data: Any = None,
        previous_data: Any = None,
        result_data: Any = None,
        var_context: str = "",
    ) -> None:
        """
        This function provides a comprehensive diagnostic analysis of the precipitation data by comparing the current and previous time slices (if provided) and analyzing the resulting accumulated precipitation data. It prints the range of values for both current and previous data to help identify potential issues in the temporal differencing process, such as decreasing cumulative values that may indicate data loading problems. It also computes and prints key statistics from the result data, including range, mean, and spatial coverage of non-zero precipitation points, to provide insights into the characteristics of the computed accumulation. All diagnostic output is controlled by the verbose flag and includes optional variable context labeling for clarity. By centralizing this logic, it promotes code reuse and ensures consistent diagnostic analysis across different steps in the precipitation calculations.

        Parameters:
            current_data (Any): Data array for the current time slice, typically xarray DataArray or numpy array (default: None).
            previous_data (Any): Data array for the previous time slice used for accumulation differencing (default: None).
            result_data (Any): Resulting precipitation data array after differencing or filtering operations (default: None).
            var_context (str): Variable name or context string used for labeling diagnostic output messages (default: "").

        Returns:
            None
        """
        # If verbose mode is not enabled, skip all diagnostic output
        if not self.verbose:
            return

        # If both current and previous data are provided, print a comparison of their ranges
        if current_data is not None and previous_data is not None:
            self._print_current_previous_comparison(
                current_data, previous_data, var_context
            )

        # If result data is provided, compute and print statistics
        if result_data is not None:
            self._print_result_data_analysis(result_data, var_context)

    def _print_time_slice_info(
        self: "PrecipitationDiagnostics",
        dataset: xr.Dataset,
        time_index: int,
        var_context: str = "",
        time_step_diff: int = 1,
    ) -> None:
        """
        This function prints diagnostic information about the current and previous time slices being used in the accumulation calculation, including their actual time values if available. It checks for the presence of a time coordinate in the dataset and retrieves the corresponding time values for the current and previous indices. This information can help users understand the temporal context of the data being processed and identify any potential issues with time indexing or data loading. By centralizing this logic, it promotes code reuse and ensures consistent diagnostic output regarding time slice information across different steps in the precipitation diagnostics.

        Parameters:
            dataset (xr.Dataset): MPAS dataset containing time coordinate information.
            time_index (int): Current time index being processed for accumulation calculation.
            var_context (str): Variable context string for customizing diagnostic message labels (default: "").
            time_step_diff (int): Time step difference used for accumulation period lookback (default: 1).

        Returns:
            None
        """
        # If verbose mode is not enabled, skip printing time slice information
        if not self.verbose:
            return

        try:
            # Determine the name of the time dimension
            time_dim = "Time" if "Time" in dataset.dims else "time"

            if time_dim in dataset.coords and hasattr(dataset[time_dim], "values"):
                # Retrieve the actual time values for the current and previous time indices
                current_time = dataset[time_dim].values[time_index]
                previous_time = dataset[time_dim].values[time_index - time_step_diff]

                # Format the variable context for labeling the output messages
                variable_label = f" for {var_context}" if var_context else ""
                logger.debug("Time slice info%s:", variable_label)
                logger.debug("  Current time (index %d): %s", time_index, current_time)
                logger.debug(
                    "  Previous time (index %d): %s",
                    time_index - time_step_diff,
                    previous_time,
                )
                logger.debug("  Time step difference: %d steps", time_step_diff)
            else:
                logger.debug(
                    "Time slice: indices %d -> %d",
                    time_index - time_step_diff,
                    time_index,
                )

        except Exception as e:
            logger.warning("Could not print time slice info: %s", e)
