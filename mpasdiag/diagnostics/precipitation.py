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
    Specialized diagnostics for precipitation calculations from MPAS data.
    
    This class provides methods for precipitation accumulation calculations,
    temporal differencing, and precipitation analysis from MPAS diagnostic files.
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
        Compute precipitation accumulation from cumulative precipitation data for specified accumulation period. This method calculates precipitation by computing the temporal difference between current and previous cumulative values based on the specified accumulation period. The method handles multiple precipitation variable types including convective (rainc), non-convective (rainnc), and total precipitation with automatic unit conversion and data validation. Special handling is provided for the first time step and insufficient lookback periods with appropriate zero-field generation. The computation includes comprehensive diagnostic output when verbose mode is enabled, showing time slice information and data range validation. The method supports both xarray and uxarray dataset types with flexible time dimension detection.

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
        
        if var_name == 'total':
            if data_type == 'uxarray':
                current_rainc = dataset['rainc'][time_index]
                current_rainnc = dataset['rainnc'][time_index]
                previous_rainc = dataset['rainc'][time_index - time_step_diff]
                previous_rainnc = dataset['rainnc'][time_index - time_step_diff]
            else:
                current_rainc = dataset['rainc'].isel({time_dim: time_index})
                current_rainnc = dataset['rainnc'].isel({time_dim: time_index})
                previous_rainc = dataset['rainc'].isel({time_dim: time_index - time_step_diff})
                previous_rainnc = dataset['rainnc'].isel({time_dim: time_index - time_step_diff})
            
            if hasattr(current_rainc, 'compute'):
                current_rainc = cast(Any, current_rainc).compute()
                current_rainnc = cast(Any, current_rainnc).compute()
                previous_rainc = cast(Any, previous_rainc).compute()
                previous_rainnc = cast(Any, previous_rainnc).compute()
            
            current_total = current_rainc + current_rainnc
            previous_total = previous_rainc + previous_rainnc
            
            if self.verbose:
                self._analyze_precipitation_diagnostics(current_total, previous_total, var_context=var_name)
            
            accum_precip = current_total - previous_total
            
        else:
            if data_type == 'uxarray':
                current_data = dataset[var_name][time_index]
                previous_data = dataset[var_name][time_index - time_step_diff]
            else:
                current_data = dataset[var_name].isel({time_dim: time_index})
                previous_data = dataset[var_name].isel({time_dim: time_index - time_step_diff})
            
            if hasattr(current_data, 'compute'):
                current_data = cast(Any, current_data).compute()
                previous_data = cast(Any, previous_data).compute()
            
            if self.verbose:
                self._analyze_precipitation_diagnostics(current_data, previous_data, var_context=var_name)
            
            accum_precip = current_data - previous_data
        
        accum_precip = self._apply_precipitation_filters_and_attributes(accum_precip, var_name)
        
        if hasattr(accum_precip, 'attrs'):
            accum_precip.attrs['long_name'] = f'{accum_hours}-hour accumulated precipitation from {var_name}'
            accum_precip.attrs['accumulation_period'] = accum_period
            accum_precip.attrs['accumulation_hours'] = accum_hours
        
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
            if var_name == 'total':
                if data_type == 'uxarray':
                    rainc_data = dataset['rainc'][time_index]
                    rainnc_data = dataset['rainnc'][time_index]
                else:
                    rainc_data = dataset['rainc'].isel({time_dim: time_index})
                    rainnc_data = dataset['rainnc'].isel({time_dim: time_index})
                sample_data = rainc_data + rainnc_data
            else:
                if data_type == 'uxarray':
                    sample_data = dataset[var_name][time_index]
                else:
                    sample_data = dataset[var_name].isel({time_dim: time_index})
            
            if hasattr(sample_data, 'compute'):
                sample_data = cast(Any, sample_data).compute()
            
            if time_index == 0:
                if self.verbose:
                    print(f"Time index 0 requested - using actual data from file, variable: {var_name}")
                
                hourly_precip_at_time = sample_data.where(sample_data >= 0, 0)
                hourly_precip_at_time = hourly_precip_at_time.where(hourly_precip_at_time < 1e5, 0)
                
                long_name = f'{accum_hours}-hour accumulated precipitation from {var_name}'
                hourly_precip_at_time.attrs.update({
                    'units': 'mm',
                    'standard_name': 'precipitation',
                    'long_name': long_name,
                    'accumulation_period': accum_period,
                    'accumulation_hours': accum_hours,
                })
                
                return hourly_precip_at_time
            else:
                if self.verbose:
                    print(f"Warning: Time index {time_index} < required {accum_hours}-hour lookback")
                    print("Creating zero precipitation field for insufficient data")
                
                zero_precip = sample_data * 0.0  
                zero_precip.attrs.update({
                    'units': 'mm',
                    'standard_name': 'precipitation',
                    'long_name': f'{accum_hours}-hour accumulated precipitation from {var_name} (insufficient data)',
                    'accumulation_period': accum_period,
                    'accumulation_hours': accum_hours,
                    'note': f'Insufficient historical data for {accum_hours}-hour accumulation at time index {time_index}'
                })
                return zero_precip
                
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
            try:
                curr_min, curr_max = float(current_data.min()), float(current_data.max())
                prev_min, prev_max = float(previous_data.min()), float(previous_data.max())
                
                var_label = var_context if var_context else "precipitation"
                print(f"Current {var_label} range: {curr_min:.2f} to {curr_max:.2f} mm")
                print(f"Previous {var_label} range: {prev_min:.2f} to {prev_max:.2f} mm")
                
                if curr_max < prev_max:
                    print(f"WARNING: Current max ({curr_max:.2f}) < Previous max ({prev_max:.2f}) - possible data loading issue!")
                    
            except Exception as e:
                print(f"Could not analyze current/previous data: {e}")
        
        if result_data is not None:
            try:
                data_values = result_data.values.flatten()
                finite_values = data_values[np.isfinite(data_values)]
                
                if len(finite_values) > 0:
                    data_min, data_max = np.nanmin(finite_values), np.nanmax(finite_values)
                    data_mean = np.nanmean(finite_values)
                    nonzero_count = np.sum(finite_values > 0.01) 
                    
                    var_label = var_context if var_context else "precipitation"
                    print(f"Result {var_label} range: {data_min:.3f} to {data_max:.3f} mm")
                    print(f"Result {var_label} mean: {data_mean:.3f} mm")
                    print(f"Points with precipitation > 0.01 mm: {nonzero_count:,}/{len(finite_values):,} ({100*nonzero_count/len(finite_values):.1f}%)")
                else:
                    print("Warning: No finite values found in result data")
                    
            except Exception as e:
                print(f"Could not analyze result data: {e}")
    
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