#!/usr/bin/env python3

"""
MPAS Precipitation Diagnostics

This module provides specialized functionality for precipitation analysis and calculations from MPAS model output data including accumulation computations, temporal differencing, and statistical diagnostics. It includes methods for computing precipitation accumulation over various time periods (hourly, daily, custom intervals), handling unit conversions between model output and display units, extracting convective and non-convective precipitation components, and performing data validation with NaN filtering. The module is designed to integrate seamlessly with the broader MPASdiag framework, leveraging existing data processing and visualization utilities for meteorological applications. Core capabilities include temporal differencing for deriving instantaneous rates from cumulative totals, accumulation period calculations with automatic time coordinate handling, and robust error handling for missing or invalid precipitation data.

Classes:
    PrecipitationDiagnostics: Specialized class for performing precipitation-specific diagnostics and calculations from MPAS datasets.
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import numpy as np
import xarray as xr
from typing import Any, Optional, Union, cast


class PrecipitationDiagnostics:
    """
    This class provides methods for precipitation accumulation calculations, temporal differencing, and precipitation analysis from MPAS diagnostic files.
    """
    
    def __init__(self, verbose: bool = True) -> None:
        """
        Initialize the precipitation diagnostics helper for analyzing MPAS model precipitation output. This constructor sets up the precipitation diagnostics instance with configurable verbosity for controlling console output during precipitation calculations and analysis. The verbose parameter allows users to enable or disable detailed diagnostic messages during accumulation computations, temporal differencing operations, and precipitation statistics. This initialization prepares the instance for subsequent precipitation analysis operations on MPAS diagnostic datasets. The class supports both convective and non-convective precipitation analysis with automatic unit handling and data validation.

        Parameters:
            verbose (bool): Enable verbose output messages during precipitation calculations (default: True).

        Returns:
            None
        """
        self.verbose = verbose
    
    def compute_precipitation_difference(self, dataset: xr.Dataset, time_index: int, 
                                       var_name: str = 'rainnc', accum_period: str = 'a01h',
                                       data_type: str = 'xarray') -> xr.DataArray:
        """
        The function computes temporal differences between current and prior cumulative fields according to the requested accumulation interval and supports convective (`rainc`), non-convective (`rainnc`) and combined (`total`) variables. Special-case handling is provided for the first time step or when there is insufficient lookback by returning actual file data or a zero-filled field. Diagnostic output is produced when `verbose` is enabled to aid troubleshooting.

        Parameters:
            dataset (xr.Dataset): MPAS dataset containing precipitation variables.
            time_index (int): Time index for current time step.
            var_name (str): Precipitation variable name ('rainc', 'rainnc', or 'total') (default: 'rainnc').
            accum_period (str): Accumulation period identifier (e.g., 'a01h', 'a03h', 'a06h', 'a12h', 'a24h') (default: 'a01h').
            data_type (str): Dataset type specification, either 'xarray' or 'uxarray' (default: 'xarray').

        Returns:
            xr.DataArray: Precipitation accumulation data array in millimeters with CF-compliant attributes including long_name, accumulation_period, and accumulation_hours metadata.
        """
        if dataset is None:
            raise ValueError("Dataset not provided. Pass a valid MPAS dataset.")
        
        time_dim = 'Time' if 'Time' in dataset.dims else 'time'
        time_size = dataset.sizes[time_dim]
        
        if time_index >= time_size:
            raise ValueError(f"Time index {time_index} exceeds dataset size {time_size}")
        
        accum_hours = self.get_accumulation_hours(accum_period)
        time_step_diff = accum_hours
        
        if time_index < time_step_diff:
            return self._handle_first_time_step(dataset, time_dim, time_index, var_name, accum_period, accum_hours, data_type)
        
        if self.verbose:
            print(f"Computing {accum_hours}-hour accumulation for period: {accum_period}")
            self._print_time_slice_info(dataset, time_index, var_name, time_step_diff)
        
        # Compute accumulation based on variable type
        previous_index = time_index - time_step_diff

        if var_name == 'total':
            accum_precip = self._compute_total_precipitation_accumulation(
                dataset, time_index, previous_index, time_dim, data_type
            )
        else:
            accum_precip = self._compute_single_variable_accumulation(
                dataset, var_name, time_index, previous_index, time_dim, data_type
            )
        
        # Apply filters and set attributes
        accum_precip = self._apply_precipitation_filters_and_attributes(accum_precip, var_name)
        
        accum_precip.attrs.update({
            'long_name': f'{accum_hours}-hour accumulated precipitation from {var_name}',
            'accumulation_period': accum_period,
            'accumulation_hours': accum_hours
        })
        
        if self.verbose:
            print(f"Computed {accum_hours}-hour accumulated precipitation for period: {accum_period}")
            self._analyze_precipitation_diagnostics(result_data=accum_precip, var_context=var_name)
        
        return accum_precip
    
    def get_accumulation_hours(self, accum_period: str) -> int:
        """
        Extract accumulation hours from standardized accumulation period string identifier. This method provides a mapping from string-based accumulation period codes to integer hour values for precipitation accumulation calculations. The mapping supports standard forecast accumulation periods including 1-hour, 3-hour, 6-hour, 12-hour, and 24-hour intervals commonly used in MPAS diagnostic output. If an unrecognized period identifier is provided, the method defaults to 24 hours to ensure robust behavior. This utility function is used internally by precipitation differencing methods to determine the appropriate temporal offset for computing accumulation periods.

        Parameters:
            accum_period (str): Accumulation period identifier following standard MPAS convention (e.g., 'a01h', 'a03h', 'a06h', 'a12h', 'a24h').

        Returns:
            int: Number of hours corresponding to the accumulation period, with default value of 24 hours if identifier is not recognized.
        """
        accum_hours_map = {'a01h': 1, 'a03h': 3, 'a06h': 6, 'a12h': 12, 'a24h': 24}
        return accum_hours_map.get(accum_period, 24)

    def _extract_variable_at_time(self, dataset: xr.Dataset, var_name: str, 
                                  time_index: int, time_dim: str, 
                                  data_type: str) -> xr.DataArray:
        """
        This helper centralizes access to variables for both 'xarray' and 'uxarray' access patterns and ensures that any lazy-backed objects are computed before being returned. It preserves DataArray metadata and returns a concrete xarray.DataArray ready for arithmetic and inspection.

        Parameters:
            dataset (xr.Dataset): Source MPAS dataset.
            var_name (str): Name of the variable to extract.
            time_index (int): Time index to select.
            time_dim (str): Name of the time dimension in `dataset`.
            data_type (str): Dataset access style, either 'xarray' or 'uxarray'.

        Returns:
            xr.DataArray: DataArray corresponding to the requested variable and time index.
        """
        if data_type == 'uxarray':
            data = dataset[var_name][time_index]
        else:
            data = dataset[var_name].isel({time_dim: time_index})
        
        if hasattr(data, 'compute'):
            data = cast(Any, data).compute()
        
        return data

    def _extract_precipitation_pair(self, dataset: xr.Dataset, var_name: str,
                                   current_index: int, previous_index: int,
                                   time_dim: str, data_type: str) -> tuple[xr.DataArray, xr.DataArray]:
        """
        This helper calls `_extract_variable_at_time` for both indices and returns the pair as a tuple in the order (current, previous). It is designed to simplify accumulation differencing code by centralizing pair extraction and ensuring consistent compute behavior for lazy-backed arrays.

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
        current_data = self._extract_variable_at_time(dataset, var_name, current_index, time_dim, data_type)
        previous_data = self._extract_variable_at_time(dataset, var_name, previous_index, time_dim, data_type)
        return current_data, previous_data

    def _compute_total_precipitation_accumulation(self, dataset: xr.Dataset, 
                                                 current_index: int, previous_index: int,
                                                 time_dim: str, data_type: str) -> xr.DataArray:
        """
        The routine extracts `rainc` and `rainnc` at the supplied current and previous indices, forms total fields for each time slice, and returns the temporal difference (current_total - previous_total). When verbose mode is enabled diagnostic comparisons are printed to help identify data issues.

        Parameters:
            dataset (xr.Dataset): Source MPAS dataset.
            current_index (int): Index of the current time slice.
            previous_index (int): Index of the previous time slice used for differencing.
            time_dim (str): Name of the time dimension.
            data_type (str): Dataset access style, either 'xarray' or 'uxarray'.

        Returns:
            xr.DataArray: Accumulated total precipitation (current - previous) as a DataArray.
        """
        current_rainc, previous_rainc = self._extract_precipitation_pair(
            dataset, 'rainc', current_index, previous_index, time_dim, data_type
        )
        current_rainnc, previous_rainnc = self._extract_precipitation_pair(
            dataset, 'rainnc', current_index, previous_index, time_dim, data_type
        )
        
        current_total = current_rainc + current_rainnc
        previous_total = previous_rainc + previous_rainnc
        
        if self.verbose:
            self._analyze_precipitation_diagnostics(current_total, previous_total, var_context='total')
        
        return current_total - previous_total

    def _compute_single_variable_accumulation(self, dataset: xr.Dataset, var_name: str,
                                             current_index: int, previous_index: int,
                                             time_dim: str, data_type: str) -> xr.DataArray:
        """
        The function extracts the current and previous time slices for `var_name` and returns their difference. When verbose mode is enabled it will print diagnostic comparisons between the two slices to assist debugging.

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
        current_data, previous_data = self._extract_precipitation_pair(
            dataset, var_name, current_index, previous_index, time_dim, data_type
        )
        
        if self.verbose:
            self._analyze_precipitation_diagnostics(current_data, previous_data, var_context=var_name)
        
        return current_data - previous_data
    
    def _extract_sample_data_for_variable(self, dataset: xr.Dataset, var_name: str,
                                         time_index: int, time_dim: str, 
                                         data_type: str) -> xr.DataArray:
        """
        This helper is used when the accumulation lookback is insufficient (e.g., first time step). For `var_name == 'total'` it composes `rainc + rainnc`, otherwise it simply returns the variable at `time_index`.

        Parameters:
            dataset (xr.Dataset): Source MPAS dataset.
            var_name (str): Variable name or 'total' for combined precipitation.
            time_index (int): Time index to extract sample from.
            time_dim (str): Name of the time dimension.
            data_type (str): Dataset access style, either 'xarray' or 'uxarray'.

        Returns:
            xr.DataArray: Sample DataArray used for first-step or insufficient lookback handling.
        """
        if var_name == 'total':
            rainc = self._extract_variable_at_time(dataset, 'rainc', time_index, time_dim, data_type)
            rainnc = self._extract_variable_at_time(dataset, 'rainnc', time_index, time_dim, data_type)
            return rainc + rainnc
        else:
            return self._extract_variable_at_time(dataset, var_name, time_index, time_dim, data_type)

    def _create_precipitation_field_with_attributes(self, data: xr.DataArray, var_name: str,
                                                   accum_period: str, accum_hours: int,
                                                   is_insufficient_data: bool = False) -> xr.DataArray:
        """
        Negative values are set to zero and extreme sentinel values are removed. The returned DataArray is annotated with `units`, `standard_name`, `long_name`, `accumulation_period`, and `accumulation_hours`. When `is_insufficient_data` is True an explanatory `note` attribute is added.

        Parameters:
            data (xr.DataArray): Raw precipitation data to filter and annotate.
            var_name (str): Variable context used to build descriptive long_name.
            accum_period (str): Accumulation period identifier (e.g., 'a01h').
            accum_hours (int): Number of accumulation hours.
            is_insufficient_data (bool): Flag indicating insufficient lookback (default: False).

        Returns:
            xr.DataArray: Filtered and metadata-annotated precipitation DataArray.
        """
        # Apply quality filters
        filtered_data = data.where(data >= 0, 0)
        filtered_data = filtered_data.where(filtered_data < 1e5, 0)
        
        # Build long name based on data availability
        if is_insufficient_data:
            long_name = f'{accum_hours}-hour accumulated precipitation from {var_name} (insufficient data)'
        else:
            long_name = f'{accum_hours}-hour accumulated precipitation from {var_name}'
        
        # Set CF-compliant attributes
        filtered_data.attrs.update({
            'units': 'mm',
            'standard_name': 'precipitation',
            'long_name': long_name,
            'accumulation_period': accum_period,
            'accumulation_hours': accum_hours,
        })
        
        # Add note for insufficient data case
        if is_insufficient_data:
            filtered_data.attrs['note'] = f'Insufficient historical data for {accum_hours}-hour accumulation'
        
        return filtered_data

    def _handle_time_index_zero(self, dataset: xr.Dataset, time_dim: str, 
                               var_name: str, accum_period: str, 
                               accum_hours: int, data_type: str) -> xr.DataArray:
        """
        For time index zero the function returns the actual file data (not a differenced field) with filters and attributes applied. This preserves observed behavior at model start times while maintaining consistent metadata for downstream users.

        Parameters:
            dataset (xr.Dataset): Source MPAS dataset.
            time_dim (str): Name of the time dimension.
            var_name (str): Precipitation variable to extract.
            accum_period (str): Accumulation period identifier.
            accum_hours (int): Number of hours for the accumulation period.
            data_type (str): Dataset access style, either 'xarray' or 'uxarray'.

        Returns:
            xr.DataArray: DataArray representing the precipitation at time index 0 with attributes.
        """
        if self.verbose:
            print(f"Time index 0 requested - using actual data from file, variable: {var_name}")
        
        sample_data = self._extract_sample_data_for_variable(dataset, var_name, 0, time_dim, data_type)
        return self._create_precipitation_field_with_attributes(
            sample_data, var_name, accum_period, accum_hours, is_insufficient_data=False
        )

    def _handle_insufficient_lookback(self, dataset: xr.Dataset, time_dim: str,
                                     time_index: int, var_name: str, 
                                     accum_period: str, accum_hours: int,
                                     data_type: str) -> xr.DataArray:
        """
        The routine constructs a zero field matching the requested variable shape and annotates it to indicate insufficient lookback, which helps downstream code differentiate real zero precipitation from missing historical information.

        Parameters:
            dataset (xr.Dataset): Source MPAS dataset.
            time_dim (str): Name of the time dimension.
            time_index (int): Current time index being processed.
            var_name (str): Precipitation variable name.
            accum_period (str): Accumulation period identifier.
            accum_hours (int): Number of hours for the accumulation period.
            data_type (str): Dataset access style, either 'xarray' or 'uxarray'.

        Returns:
            xr.DataArray: Zero-filled precipitation DataArray with metadata indicating insufficient data.
        """
        if self.verbose:
            print(f"Warning: Time index {time_index} < required {accum_hours}-hour lookback")
            print("Creating zero precipitation field for insufficient data")
        
        sample_data = self._extract_sample_data_for_variable(dataset, var_name, time_index, time_dim, data_type)
        zero_precip = sample_data * 0.0
        
        return self._create_precipitation_field_with_attributes(
            zero_precip, var_name, accum_period, accum_hours, is_insufficient_data=True
        )

    def _handle_first_time_step(self, dataset: xr.Dataset, time_dim: str, time_index: int,
                               var_name: str, accum_period: str, accum_hours: int, 
                               data_type: str) -> xr.DataArray:
        """
        Handle the special case of time index 0 or insufficient lookback data for precipitation accumulation calculations. This method provides appropriate fallback behavior when the requested time index does not have sufficient historical data for the specified accumulation period. For time index 0, the method returns the actual precipitation data from the file with proper filtering and attribute assignment. For insufficient lookback cases, the method generates a zero precipitation field with appropriate metadata indicating the data limitation. The method supports both single precipitation variables and total precipitation combining convective and non-convective components. Error handling ensures robust behavior with informative diagnostic messages when verbose mode is enabled.

        Parameters:
            dataset (xr.Dataset): MPAS dataset containing precipitation variables.
            time_dim (str): Name of the time dimension in the dataset.
            time_index (int): Current time index being processed.
            var_name (str): Precipitation variable name ('rainc', 'rainnc', or 'total').
            accum_period (str): Accumulation period identifier (e.g., 'a01h', 'a24h').
            accum_hours (int): Number of hours for the accumulation period.
            data_type (str): Dataset type specification, either 'xarray' or 'uxarray'.

        Returns:
            xr.DataArray: Precipitation data array at the specified time index with appropriate filtering and metadata, or zero precipitation field with explanatory note attribute if insufficient data is available.
        """
        try:
            if time_index == 0:
                return self._handle_time_index_zero(
                    dataset, time_dim, var_name, accum_period, accum_hours, data_type
                )
            else:
                return self._handle_insufficient_lookback(
                    dataset, time_dim, time_index, var_name, accum_period, accum_hours, data_type
                )
        except Exception as e:
            if self.verbose:
                print(f"Error handling first time step: {e}")
            raise ValueError(f"Cannot handle time index {time_index} for variable {var_name}: {e}")
    
    def _apply_precipitation_filters_and_attributes(self, data: xr.DataArray, var_context: str = "") -> xr.DataArray:
        """
        Apply standard precipitation data quality filters and set CF-compliant metadata attributes. This method performs two primary filtering operations to ensure physically realistic precipitation values by removing negative values and capping unrealistically large values. The method sets negative values to zero to handle numerical artifacts from cumulative differencing operations. Values exceeding 100,000 mm are also set to zero to filter out missing data flags or computational errors. After filtering, the method adds standardized CF-compliant attributes including units, standard_name, and long_name for consistent metadata across precipitation datasets. The long_name is customized based on the provided variable context to maintain traceability of precipitation type.

        Parameters:
            data (xr.DataArray): Precipitation data array to be filtered and attributed.
            var_context (str): Variable context string for customizing the long_name attribute (default: "").

        Returns:
            xr.DataArray: Filtered precipitation data array with updated CF-compliant attributes including units in millimeters, standard_name as 'precipitation', and customized long_name.
        """
        data = data.where(data >= 0, 0) 
        data = data.where(data < 1e5, 0) 
        
        long_name = f'hourly accumulated precipitation from {var_context}' if var_context else 'hourly accumulated precipitation'

        data.attrs.update({
            'units': 'mm',
            'standard_name': 'precipitation',
            'long_name': long_name,
        })
        
        return data
    
    def _extract_min_max_from_data(self, data: Any) -> tuple[float, float]:
        """
        The helper attempts to convert the provided object into numeric min/max floats suitable for printing. It centralizes error handling for cases where the input may not expose typical numpy-like min/max semantics.

        Parameters:
            data (Any): Array-like object (xarray.DataArray or numpy array) supporting min()/max().

        Returns:
            tuple[float, float]: (minimum_value, maximum_value) as Python floats.
        """
        return float(data.min()), float(data.max())

    def _print_current_previous_comparison(self, current_data: Any, previous_data: Any, 
                                          var_context: str) -> None:
        """
        The function extracts min/max for both inputs and prints ranges with a warning if the current maximum is less than the previous maximum which can indicate data loading or differencing issues. It is a diagnostic convenience invoked when verbose mode is enabled.

        Parameters:
            current_data (Any): Current time slice data array.
            previous_data (Any): Previous time slice data array.
            var_context (str): Context label used in printed messages.

        Returns:
            None: Only prints messages; does not modify data.
        """
        try:
            curr_min, curr_max = self._extract_min_max_from_data(current_data)
            prev_min, prev_max = self._extract_min_max_from_data(previous_data)
            
            var_label = var_context if var_context else "precipitation"
            print(f"Current {var_label} range: {curr_min:.2f} to {curr_max:.2f} mm")
            print(f"Previous {var_label} range: {prev_min:.2f} to {prev_max:.2f} mm")
            
            if curr_max < prev_max:
                print(f"WARNING: Current max ({curr_max:.2f}) < Previous max ({prev_max:.2f}) - possible data loading issue!")
        except Exception as e:
            print(f"Could not analyze current/previous data: {e}")

    def _compute_result_statistics(self, result_data: Any) -> Optional[dict[str, Any]]:
        """
        The routine extracts finite values, computes min/max/mean, counts points exceeding a small threshold, and returns these statistics in a dictionary suitable for printing or programmatic consumption. If no finite values are present the function returns None.

        Parameters:
            result_data (Any): Result DataArray or array-like object to analyze.

        Returns:
            Optional[dict[str, Any]]: Dictionary with keys 'min', 'max', 'mean', 'nonzero_count', 'total_count', 'nonzero_percentage', or None if no finite values available.
        """
        try:
            data_values = result_data.values.flatten()
            finite_values = data_values[np.isfinite(data_values)]
            
            if len(finite_values) == 0:
                return None
            
            data_min = np.nanmin(finite_values)
            data_max = np.nanmax(finite_values)
            data_mean = np.nanmean(finite_values)
            nonzero_count = np.sum(finite_values > 0.01)
            total_count = len(finite_values)
            nonzero_percentage = 100 * nonzero_count / total_count
            
            return {
                'min': data_min,
                'max': data_max,
                'mean': data_mean,
                'nonzero_count': nonzero_count,
                'total_count': total_count,
                'nonzero_percentage': nonzero_percentage
            }
        except Exception as e:
            print(f"Could not compute result statistics: {e}")
            return None

    def _print_result_data_analysis(self, result_data: Any, var_context: str) -> None:
        """
        This function formats the output from `_compute_result_statistics` and prints range, mean, and spatial coverage of non-zero precipitation points with percentages. It is intended for diagnostic logging when verbose mode is enabled.

        Parameters:
            result_data (Any): Result DataArray or array-like object to analyze.
            var_context (str): Context label used in printed messages.

        Returns:
            None: Only prints messages; does not modify data.
        """
        stats = self._compute_result_statistics(result_data)
        
        if stats is None:
            print("Warning: No finite values found in result data")
            return
        
        var_label = var_context if var_context else "precipitation"
        print(f"Result {var_label} range: {stats['min']:.3f} to {stats['max']:.3f} mm")
        print(f"Result {var_label} mean: {stats['mean']:.3f} mm")
        print(f"Points with precipitation > 0.01 mm: {stats['nonzero_count']:,}/{stats['total_count']:,} ({stats['nonzero_percentage']:.1f}%)")

    def _analyze_precipitation_diagnostics(self, current_data: Any = None, previous_data: Any = None, 
                                         result_data: Any = None, var_context: str = "") -> None:
        """
        Print comprehensive diagnostic summaries for precipitation data inputs and computed results. This method provides detailed statistical analysis of current and previous time slice data along with accumulation results to help identify data loading issues, differencing problems, or numerical artifacts. For current and previous data pairs, the method compares min/max ranges and flags potential issues such as decreasing cumulative values. For result data, the method computes statistics including range, mean, and spatial coverage of non-zero precipitation. All diagnostic output respects the verbose flag and includes clear labeling based on variable context. This diagnostic tool is essential for validating precipitation calculations and troubleshooting data quality issues during MPAS output processing.

        Parameters:
            current_data (Any): Data array for the current time slice, typically xarray DataArray or numpy array (default: None).
            previous_data (Any): Data array for the previous time slice used for accumulation differencing (default: None).
            result_data (Any): Resulting precipitation data array after differencing or filtering operations (default: None).
            var_context (str): Variable name or context string used for labeling diagnostic output messages (default: "").

        Returns:
            None
        """
        if not self.verbose:
            return
        
        if current_data is not None and previous_data is not None:
            self._print_current_previous_comparison(current_data, previous_data, var_context)
        
        if result_data is not None:
            self._print_result_data_analysis(result_data, var_context)
    
    def _print_time_slice_info(self, dataset: xr.Dataset, time_index: int, 
                              var_context: str = "", time_step_diff: int = 1) -> None:
        """
        Print detailed diagnostic information about time slice selection for precipitation accumulation calculations. This method provides visibility into the temporal differencing process by displaying the current and previous time coordinates along with the time step offset. The output helps users verify correct time slice selection and diagnose issues related to accumulation period calculations. If time coordinate values are available, the method displays actual timestamps for both current and previous time steps along with the temporal separation. For datasets without explicit time coordinates, the method falls back to displaying index-based information. All diagnostic output respects the verbose flag and includes optional variable context labeling for clarity.

        Parameters:
            dataset (xr.Dataset): MPAS dataset containing time coordinate information.
            time_index (int): Current time index being processed for accumulation calculation.
            var_context (str): Variable context string for customizing diagnostic message labels (default: "").
            time_step_diff (int): Time step difference used for accumulation period lookback (default: 1).

        Returns:
            None
        """
        if not self.verbose:
            return
        
        try:
            time_dim = 'Time' if 'Time' in dataset.dims else 'time'
            
            if time_dim in dataset.coords and hasattr(dataset[time_dim], 'values'):
                current_time = dataset[time_dim].values[time_index]
                previous_time = dataset[time_dim].values[time_index - time_step_diff]
                
                var_label = f" for {var_context}" if var_context else ""
                print(f"Time slice info{var_label}:")
                print(f"  Current time (index {time_index}): {current_time}")
                print(f"  Previous time (index {time_index - time_step_diff}): {previous_time}")
                print(f"  Time step difference: {time_step_diff} steps")
            else:
                print(f"Time slice: indices {time_index - time_step_diff} -> {time_index}")
                
        except Exception as e:
            print(f"Could not print time slice info: {e}")