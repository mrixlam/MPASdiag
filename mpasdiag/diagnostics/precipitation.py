#!/usr/bin/env python3

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
import numpy as np
import xarray as xr
from typing import Any, Optional, cast


class PrecipitationDiagnostics:
    """ Computes precipitation-related diagnostics for MPAS model output, including temporal differencing for accumulation calculations, data quality checks, and statistical summaries. """
    
    def __init__(self: "PrecipitationDiagnostics", 
                 verbose: bool = True) -> None:
        """
        This constructor initializes the PrecipitationDiagnostics class with an optional verbose flag that controls the level of diagnostic output during precipitation calculations. When verbose mode is enabled, the class will print detailed messages about the processing steps, any issues encountered, and statistical summaries of the precipitation data being analyzed. This allows users to gain insights into the data quality and identify potential problems during the accumulation calculation process. The verbose flag can be set to False to suppress diagnostic messages for cleaner output when running in production or when detailed diagnostics are not needed. 

        Parameters:
            verbose (bool): Enable verbose output messages during precipitation calculations (default: True).

        Returns:
            None
        """
        # Enable verbose output
        self.verbose = verbose
    
    def compute_precipitation_difference(self: "PrecipitationDiagnostics",
                                         dataset: xr.Dataset, 
                                         time_index: int, 
                                         var_name: str = 'rainnc', 
                                         accum_period: str = 'a01h', 
                                         data_type: str = 'xarray') -> xr.DataArray:
        """
        This function computes the precipitation accumulation for a specified variable and accumulation period by performing temporal differencing on the MPAS dataset. It handles edge cases such as the first time step where differencing is not possible by returning the actual data with appropriate metadata. The function applies quality filters to ensure physically realistic precipitation values and annotates the resulting DataArray with CF-compliant attributes. When verbose mode is enabled, it provides detailed diagnostic output about the time slice selection, current and previous data comparisons, and statistics of the computed accumulation to help users identify potential data issues during processing. 

        Parameters:
            dataset (xr.Dataset): MPAS dataset containing precipitation variables.
            time_index (int): Time index for current time step.
            var_name (str): Precipitation variable name ('rainc', 'rainnc', or 'total') (default: 'rainnc').
            accum_period (str): Accumulation period identifier (e.g., 'a01h', 'a03h', 'a06h', 'a12h', 'a24h') (default: 'a01h').
            data_type (str): Dataset type specification, either 'xarray' or 'uxarray' (default: 'xarray').

        Returns:
            xr.DataArray: Precipitation accumulation data array in millimeters with CF-compliant attributes including long_name, accumulation_period, and accumulation_hours metadata.
        """
        # Validate dataset input and raise error if not provided
        if dataset is None:
            raise ValueError("Dataset not provided. Pass a valid MPAS dataset.")
        
        # Determine time dimension and size
        time_dim = 'Time' if 'Time' in dataset.dims else 'time'
        time_size = dataset.sizes[time_dim]
        
        # Validate time index is within bounds of the dataset size
        if time_index >= time_size:
            raise ValueError(f"Time index {time_index} exceeds dataset size {time_size}")
        
        # Validate variable name and raise error if not recognized 
        accum_hours = self.get_accumulation_hours(accum_period)
        time_step_diff = accum_hours
        
        # Handle edge case for first time step where differencing is not possible
        if time_index < time_step_diff:
            return self._handle_first_time_step(dataset, time_dim, time_index, var_name, accum_period, accum_hours, data_type)
        
        # Perform temporal differencing to compute accumulation for the specified variable and accumulation period
        if self.verbose:
            print(f"Computing {accum_hours}-hour accumulation for period: {accum_period}")
            self._print_time_slice_info(dataset, time_index, var_name, time_step_diff)
        
        # Calculate previous time index for differencing
        previous_index = time_index - time_step_diff

        # Compute the accumulated precipitation based on the specified variable name, handling 'total' as a special case that sums 'rainc' and 'rainnc'
        if var_name == 'total':
            accum_precip = self._compute_total_precipitation_accumulation(
                dataset, time_index, previous_index, time_dim, data_type
            )
        else:
            accum_precip = self._compute_single_variable_accumulation(
                dataset, var_name, time_index, previous_index, time_dim, data_type
            )
        
        # Apply quality filters and set attributes on the resulting accumulated precipitation field 
        accum_precip = self._apply_precipitation_filters_and_attributes(accum_precip, var_name)
        
        # Update attributes to include accumulation period and hours for clarity in downstream use
        accum_precip.attrs.update({
            'long_name': f'{accum_hours}-hour accumulated precipitation from {var_name}',
            'accumulation_period': accum_period,
            'accumulation_hours': accum_hours
        })
        
        # Provide diagnostic output about the computed accumulation, including range and spatial coverage, to help identify potential issues in the results
        if self.verbose:
            print(f"Computed {accum_hours}-hour accumulated precipitation for period: {accum_period}")
            self._analyze_precipitation_diagnostics(result_data=accum_precip, var_context=var_name)
        
        # Return accumulated precipitation DataArray
        return accum_precip
    
    def get_accumulation_hours(self: "PrecipitationDiagnostics", 
                               accum_period: str) -> int:
        """
        This helper function maps standard MPAS accumulation period identifiers (e.g., 'a01h', 'a03h', 'a06h', 'a12h', 'a24h') to their corresponding number of hours. If the provided identifier is not recognized, it defaults to 24 hours. This centralizes the mapping logic for accumulation periods and allows for easy extension in the future if additional periods are needed. 

        Parameters:
            accum_period (str): Accumulation period identifier following standard MPAS convention (e.g., 'a01h', 'a03h', 'a06h', 'a12h', 'a24h').

        Returns:
            int: Number of hours corresponding to the accumulation period, with default value of 24 hours if identifier is not recognized.
        """
        # Define mapping of accumulation period identifiers to hours
        accum_hours_map = {'a01h': 1, 'a03h': 3, 'a06h': 6, 'a12h': 12, 'a24h': 24}

        # Return accumulation hours (default to 24 if identifier is not recognized)
        return accum_hours_map.get(accum_period, 24)

    def _extract_variable_at_time(self: "PrecipitationDiagnostics", 
                                  dataset: xr.Dataset, 
                                  var_name: str, 
                                  time_index: int, 
                                  time_dim: str, 
                                  data_type: str) -> xr.DataArray:
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
        if data_type == 'uxarray':
            # For uxarray, .isel is not supported, so we use direct indexing
            data = dataset[var_name][time_index]
        else:
            # For xarray, we can use .isel to select the time index
            data = dataset[var_name].isel({time_dim: time_index})
        
        # Compute the data if it is a Dask array 
        if hasattr(data, 'compute'):
            data = cast(Any, data).compute()
        
        # Return the extracted variable data as a DataArray
        return data

    def _extract_precipitation_pair(self: "PrecipitationDiagnostics", 
                                    dataset: xr.Dataset, 
                                    var_name: str, 
                                    current_index: int, 
                                    previous_index: int, 
                                    time_dim: str, 
                                    data_type: str) -> tuple[xr.DataArray, xr.DataArray]:
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
        current_data = self._extract_variable_at_time(dataset, var_name, current_index, time_dim, data_type)

        # Extract previous time slice for the same variable
        previous_data = self._extract_variable_at_time(dataset, var_name, previous_index, time_dim, data_type)

        # Return the current and previous data as a tuple 
        return current_data, previous_data

    def _compute_total_precipitation_accumulation(self: "PrecipitationDiagnostics", 
                                                  dataset: xr.Dataset, 
                                                  current_index: int, 
                                                  previous_index: int, 
                                                  time_dim: str, 
                                                  data_type: str) -> xr.DataArray:
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
            dataset, 'rainc', current_index, previous_index, time_dim, data_type
        )
        
        # Extract current and previous time slices for the non-convective precipitation variable (rainnc) using the same helper function to ensure consistent handling of variables
        current_rainnc, previous_rainnc = self._extract_precipitation_pair(
            dataset, 'rainnc', current_index, previous_index, time_dim, data_type
        )
        
        # Calculate current total precipitation by summing the current convective and non-convective components 
        current_total = current_rainc + current_rainnc

        # Calculate previous total precipitation by summing the previous convective and non-convective components
        previous_total = previous_rainc + previous_rainnc
        
        # Provide diagnostic output comparing current and previous total precipitation to identify potential issues in the data or processing steps
        if self.verbose:
            self._analyze_precipitation_diagnostics(current_total, previous_total, var_context='total')
        
        # Return the difference between current and previous total precipitation as the accumulated value
        return current_total - previous_total

    def _compute_single_variable_accumulation(self: "PrecipitationDiagnostics", 
                                              dataset: xr.Dataset, 
                                              var_name: str, 
                                              current_index: int, 
                                              previous_index: int, 
                                              time_dim: str, 
                                              data_type: str) -> xr.DataArray:
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
            self._analyze_precipitation_diagnostics(current_data, previous_data, var_context=var_name)
        
        # Return the difference between current and previous data as the accumulated value for the specified variable
        return current_data - previous_data
    
    def _extract_sample_data_for_variable(self: "PrecipitationDiagnostics", 
                                          dataset: xr.Dataset, 
                                          var_name: str, 
                                          time_index: int, 
                                          time_dim: str, 
                                          data_type: str) -> xr.DataArray:
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
        if var_name == 'total':
            # Extract convective precipitation component for the specified time index 
            rainc = self._extract_variable_at_time(dataset, 'rainc', time_index, time_dim, data_type)

            # Extract non-convective precipitation component for the same time index
            rainnc = self._extract_variable_at_time(dataset, 'rainnc', time_index, time_dim, data_type)

            # Return the sum of rainc and rainnc as the sample data for 'total' precipitation
            return rainc + rainnc
        else:
            # For a single variable, simply extract it at the specified time index
            return self._extract_variable_at_time(dataset, var_name, time_index, time_dim, data_type)

    def _create_precipitation_field_with_attributes(self: "PrecipitationDiagnostics", 
                                                    data: xr.DataArray, 
                                                    var_name: str, 
                                                    accum_period: str, 
                                                    accum_hours: int, 
                                                    is_insufficient_data: bool = False) -> xr.DataArray:
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
        # Apply quality filters to ensure precipitation values are non-negative and below an upper threshold to maintain physical realism
        filtered_data = data.where(data >= 0, 0)
        filtered_data = filtered_data.where(filtered_data < 1e5, 0)
        
        if is_insufficient_data:
            # If there is insufficient data for accumulation, indicate this in the long_name for clarity in downstream use
            long_name = f'{accum_hours}-hour accumulated precipitation from {var_name} (insufficient data)'
        else:
            # Construct a long_name that clearly indicates the accumulation period and variable context for clarity in downstream use
            long_name = f'{accum_hours}-hour accumulated precipitation from {var_name}'
        
        # Set attributes including units, standard_name, long_name, and accumulation metadata for clarity in downstream use
        filtered_data.attrs.update({
            'units': 'mm',
            'standard_name': 'precipitation',
            'long_name': long_name,
            'accumulation_period': accum_period,
            'accumulation_hours': accum_hours,
        })
        
        # Flag the data with an attribute if it is generated due to insufficient lookback, so that downstream code can handle it appropriately 
        if is_insufficient_data:
            filtered_data.attrs['note'] = f'Insufficient historical data for {accum_hours}-hour accumulation'
        
        # Return the filtered and annotated precipitation DataArray
        return filtered_data

    def _handle_time_index_zero(self: "PrecipitationDiagnostics", 
                                dataset: xr.Dataset, 
                                time_dim: str, 
                                var_name: str, 
                                accum_period: str, 
                                accum_hours: int, 
                                data_type: str) -> xr.DataArray:
        """
        This routine handles the special case of time index 0 for precipitation accumulation calculations, where temporal differencing cannot be performed due to the lack of a previous time slice. In this case, it returns the actual precipitation data from the file for the specified variable and accumulation period, applying appropriate filtering and metadata annotation to indicate that this is the initial time step. This allows downstream code to still have access to valid precipitation data at time index 0 while clearly indicating that it is not an accumulated value. By centralizing this logic in a dedicated function, it promotes code reuse and ensures consistent handling of the first time step across different variables and accumulation periods in the precipitation diagnostics. 

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
        # For verbose output, indicate that time index 0 is being handled with actual data from the file 
        if self.verbose:
            print(f"Time index 0 requested - using actual data from file, variable: {var_name}")
        
        # Extract the sample data for the specified variable at time index 0 
        sample_data = self._extract_sample_data_for_variable(dataset, var_name, 0, time_dim, data_type)

        # Return the sample data with appropriate filtering and attributes
        return self._create_precipitation_field_with_attributes(
            sample_data, var_name, accum_period, accum_hours, is_insufficient_data=False
        )

    def _handle_insufficient_lookback(self: "PrecipitationDiagnostics", 
                                      dataset: xr.Dataset, 
                                      time_dim: str, 
                                      time_index: int, 
                                      var_name: str, 
                                      accum_period: str, 
                                      accum_hours: int, 
                                      data_type: str) -> xr.DataArray:
        """
        This routine handles cases where the requested time index does not have sufficient historical data for the specified accumulation period (e.g., time index < required lookback hours). In such cases, it generates a zero-filled precipitation field with appropriate metadata indicating that the accumulation calculation could not be performed due to insufficient data. This allows downstream code to still have a valid DataArray to work with while clearly communicating the limitation of the data for that time step. By centralizing this logic in a dedicated function, it promotes code reuse and ensures consistent handling of insufficient lookback cases across different variables and accumulation periods in the precipitation diagnostics. 

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
        # For verbose output, indicate that there is insufficient lookback for the requested accumulation period
        if self.verbose:
            print(f"Warning: Time index {time_index} < required {accum_hours}-hour lookback")
            print("Creating zero precipitation field for insufficient data")
        
        # Generate a zero-filled precipitation field with the same shape and coordinates as the sample data for the specified variable at the current time index
        sample_data = self._extract_sample_data_for_variable(dataset, var_name, time_index, time_dim, data_type)
        zero_precip = sample_data * 0.0
        
        # Return a zero-filled precipitation field with attributes indicating insufficient data for accumulation calculation
        return self._create_precipitation_field_with_attributes(
            zero_precip, var_name, accum_period, accum_hours, is_insufficient_data=True
        )

    def _handle_first_time_step(self: "PrecipitationDiagnostics", 
                                dataset: xr.Dataset, 
                                time_dim: str, 
                                time_index: int, 
                                var_name: str, 
                                accum_period: str, 
                                accum_hours: int, 
                                data_type: str) -> xr.DataArray:
        """
        This routine handles the special case of the first time step (time index 0) for precipitation accumulation calculations, where temporal differencing cannot be performed due to the lack of a previous time slice. It extracts the actual precipitation data for the specified variable at time index 0, applies quality filters to ensure physically realistic values, and annotates the resulting DataArray with CF-compliant attributes that indicate this is the initial time step. This allows downstream code to still have access to valid precipitation data at time index 0 while clearly indicating that it is not an accumulated value. By centralizing this logic in a dedicated function, it promotes code reuse and ensures consistent handling of the first time step across different variables and accumulation periods in the precipitation diagnostics. 

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
            # Handle the first time step by extracting the actual data for the specified variable 
            if time_index == 0:
                return self._handle_time_index_zero(
                    dataset, time_dim, var_name, accum_period, accum_hours, data_type
                )
            # Handle cases where time index is greater than 0 but still does not have sufficient lookback for the requested accumulation period
            else:
                return self._handle_insufficient_lookback(
                    dataset, time_dim, time_index, var_name, accum_period, accum_hours, data_type
                )
        # Catch any exceptions that occur during the handling of the first time step 
        except Exception as e:
            if self.verbose:
                print(f"Error handling first time step: {e}")
            raise ValueError(f"Cannot handle time index {time_index} for variable {var_name}: {e}")
    
    def _apply_precipitation_filters_and_attributes(self: "PrecipitationDiagnostics", 
                                                    data: xr.DataArray, 
                                                    var_context: str = "") -> xr.DataArray:
        """
        This helper function applies quality filters to the precipitation data to ensure that values are physically realistic (e.g., non-negative and below an upper threshold). It also sets CF-compliant attributes, including units in millimeters, standard_name as 'precipitation', and a long_name that incorporates the variable context for clarity. By centralizing this logic, it promotes code reuse and ensures consistent filtering and metadata annotation across different steps in the precipitation diagnostics. This function is used both for handling edge cases (e.g., first time step) and for processing the results of temporal differencing to ensure that all precipitation DataArrays returned by the diagnostics have appropriate quality control and metadata. 

        Parameters:
            data (xr.DataArray): Precipitation data array to be filtered and attributed.
            var_context (str): Variable context string for customizing the long_name attribute (default: "").

        Returns:
            xr.DataArray: Filtered precipitation data array with updated CF-compliant attributes including units in millimeters, standard_name as 'precipitation', and customized long_name.
        """
        # Apply quality filters to ensure precipitation values are non-negative and below an upper threshold to maintain physical realism
        data = data.where(data >= 0, 0) 
        data = data.where(data < 1e5, 0) 
        
        # Construct a long_name that incorporates the variable context for clarity in downstream use
        long_name = f'hourly accumulated precipitation from {var_context}' if var_context else 'hourly accumulated precipitation'

        # Set attributes to ensure the resulting DataArray is well-described and compliant with CF conventions
        data.attrs.update({
            'units': 'mm',
            'standard_name': 'precipitation',
            'long_name': long_name,
        })
        
        # Return the filtered and attributed precipitation DataArray
        return data
    
    def _extract_min_max_from_data(self: "PrecipitationDiagnostics", 
                                   data: Any) -> tuple[float, float]:
        """
        This helper function takes an array-like object (such as an xarray.DataArray or a numpy array) and extracts the minimum and maximum values, returning them as Python floats. This is used in diagnostic comparisons to summarize the range of values in the current and previous time slices of precipitation data, helping to identify potential issues such as decreasing cumulative values that may indicate data loading or differencing problems. By centralizing this logic, it promotes code reuse and ensures consistent handling of min/max extraction across different diagnostic functions in the precipitation analysis. 

        Parameters:
            data (Any): Array-like object (xarray.DataArray or numpy array) supporting min()/max().

        Returns:
            tuple[float, float]: (minimum_value, maximum_value) as Python floats.
        """
        # Return the minimum and maximum values from the data
        return float(data.min()), float(data.max())

    def _print_current_previous_comparison(self: "PrecipitationDiagnostics", 
                                           current_data: Any, 
                                           previous_data: Any, 
                                           var_context: str) -> None:
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
            curr_min, curr_max = self._extract_min_max_from_data(current_data)
            prev_min, prev_max = self._extract_min_max_from_data(previous_data)
            
            # Format the variable context for labeling the output messages
            var_label = var_context if var_context else "precipitation"
            print(f"Current {var_label} range: {curr_min:.2f} to {curr_max:.2f} mm")
            print(f"Previous {var_label} range: {prev_min:.2f} to {prev_max:.2f} mm")
            
            # Check if the current maximum is less than the previous maximum
            if curr_max < prev_max:
                print(f"WARNING: Current max ({curr_max:.2f}) < Previous max ({prev_max:.2f}) - possible data loading issue!")
        except Exception as e:
            print(f"Could not analyze current/previous data: {e}")

    def _compute_result_statistics(self: "PrecipitationDiagnostics", 
                                   result_data: Any) -> Optional[dict[str, Any]]:
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
            data_min = np.nanmin(finite_values)
            data_max = np.nanmax(finite_values)
            data_mean = np.nanmean(finite_values)

            # Count the number of points with precipitation greater than a small threshold (e.g., 0.01 mm)
            nonzero_count = np.sum(finite_values > 0.01)
            total_count = len(finite_values)

            # Calculate the percentage of non-zero precipitation points relative to the total number of finite points
            nonzero_percentage = 100 * nonzero_count / total_count
            
            # Return a dictionary containing the computed statistics for the result data array
            return {
                'min': data_min,
                'max': data_max,
                'mean': data_mean,
                'nonzero_count': nonzero_count,
                'total_count': total_count,
                'nonzero_percentage': nonzero_percentage
            }
        # Catch any exceptions that occur during the computation of statistics
        except Exception as e:
            print(f"Could not compute result statistics: {e}")
            return None

    def _print_result_data_analysis(self: "PrecipitationDiagnostics", 
                                    result_data: Any, 
                                    var_context: str) -> None:
        """
        This function prints a comprehensive analysis of the result data array, including the range (min/max), mean, and spatial coverage of non-zero precipitation points. It uses the statistics computed by the `_compute_result_statistics` function to provide insights into the characteristics of the computed precipitation accumulation, which can help identify potential issues such as unrealistic ranges or excessive zero values. If there are no finite values in the result data, it prints a warning message instead. By centralizing this logic, it promotes code reuse and ensures consistent diagnostic output across different result data arrays in the precipitation analysis. 

        Parameters:
            result_data (Any): Result DataArray or array-like object to analyze.
            var_context (str): Context label used in printed messages.

        Returns:
            None: Only prints messages; does not modify data.
        """
        # Compute key statistics from the result data array for analysis
        stats = self._compute_result_statistics(result_data)
        
        # Warn if there are no finite values in the result data
        if stats is None:
            print("Warning: No finite values found in result data")
            return
        
        # Format the variable context for labeling the output messages
        var_label = var_context if var_context else "precipitation"
        print(f"Result {var_label} range: {stats['min']:.3f} to {stats['max']:.3f} mm")
        print(f"Result {var_label} mean: {stats['mean']:.3f} mm")
        print(f"Points with precipitation > 0.01 mm: {stats['nonzero_count']:,}/{stats['total_count']:,} ({stats['nonzero_percentage']:.1f}%)")

    def _analyze_precipitation_diagnostics(self: "PrecipitationDiagnostics", 
                                           current_data: Any = None, 
                                           previous_data: Any = None, 
                                           result_data: Any = None, 
                                           var_context: str = "") -> None:
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
            self._print_current_previous_comparison(current_data, previous_data, var_context)
        
        # If result data is provided, compute and print statistics 
        if result_data is not None:
            self._print_result_data_analysis(result_data, var_context)
    
    def _print_time_slice_info(self: "PrecipitationDiagnostics", 
                               dataset: xr.Dataset, 
                               time_index: int, 
                               var_context: str = "", 
                               time_step_diff: int = 1) -> None:
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
            time_dim = 'Time' if 'Time' in dataset.dims else 'time'
            
            if time_dim in dataset.coords and hasattr(dataset[time_dim], 'values'):
                # Retrieve the actual time values for the current and previous time indices
                current_time = dataset[time_dim].values[time_index]
                previous_time = dataset[time_dim].values[time_index - time_step_diff]
                
                # Format the variable context for labeling the output messages
                var_label = f" for {var_context}" if var_context else ""
                print(f"Time slice info{var_label}:")
                print(f"  Current time (index {time_index}): {current_time}")
                print(f"  Previous time (index {time_index - time_step_diff}): {previous_time}")
                print(f"  Time step difference: {time_step_diff} steps")
            else:
                print(f"Time slice: indices {time_index - time_step_diff} -> {time_index}")
        
        # Catch any exceptions that occur while trying to print time slice information
        except Exception as e:
            print(f"Could not print time slice info: {e}")