#!/usr/bin/env python3

"""
MPAS Data Processing Module

This module provides comprehensive functionality for reading, processing, and analyzing
MPAS (Model for Prediction Across Scales) unstructured mesh model output data.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Last Modified: 2025-10-06

Features:
    - Load MPAS diagnostic files with lazy loading for memory efficiency
    - Process unstructured grid data using UXarray and xarray
    - Compute temporal differences for precipitation analysis
    - Handle multiple precipitation variables (rainc, rainnc, total)
    - Comprehensive data validation and diagnostics
    - Support for batch processing of time series data
"""

import os
import re
import sys
import glob
import warnings
from datetime import datetime
from typing import List, Tuple, Any, Optional, Dict, Union

import numpy as np
import pandas as pd
import xarray as xr
import uxarray as ux

warnings.filterwarnings('ignore', message='The specified chunks separate the stored chunks.*')
warnings.filterwarnings('ignore', message='invalid value encountered in create_collection')
warnings.filterwarnings('ignore', message='.*Shapely.*')
warnings.filterwarnings('ignore', category=UserWarning, module='cartopy')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='shapely')
warnings.filterwarnings('ignore', category=UserWarning, message='.*chunks.*degrade performance.*')


class MPASDataProcessor:
    """
    Main class for processing MPAS model output data.
    
    This class provides methods for loading, processing, and analyzing MPAS unstructured
    mesh data with support for lazy loading, temporal analysis, and precipitation processing.
    """
    
    def __init__(self, grid_file: str, verbose: bool = True):
        """
        Initialize the MPAS data processor.

        Parameters:
            grid_file (str): Path to MPAS grid file containing mesh information.
            verbose (bool): Enable verbose output for debugging and diagnostics.

        Returns:
            None
        """
        self.grid_file = grid_file
        self.verbose = verbose
        self.dataset = None
        self.data_type = None
        
        if not os.path.exists(grid_file):
            raise FileNotFoundError(f"Grid file not found: {grid_file}")
    
    def find_diagnostic_files(self, data_dir: str) -> List[str]:
        """
        Find and validate diagnostic files in the specified directory.

        Parameters:
            data_dir (str): Directory containing diagnostic files.

        Returns:
            List[str]: Sorted list of diagnostic files.

        Raises:
            FileNotFoundError: If no diagnostic files are found.
            ValueError: If insufficient files for temporal analysis.
        """
        diag_pattern = os.path.join(data_dir, "diag*.nc")
        diag_files = sorted(glob.glob(diag_pattern))
        
        if not diag_files:
            raise FileNotFoundError(f"No diagnostic files found matching pattern: {diag_pattern}")
        
        if len(diag_files) < 2:
            raise ValueError(f"Insufficient files for temporal analysis. Found {len(diag_files)}, need at least 2.")
        
        if self.verbose:
            print(f"\nFound {len(diag_files)} diagnostic files:")
            for i, f in enumerate(diag_files[:5]):  
                print(f"  {i+1}: {os.path.basename(f)}")

            if len(diag_files) > 5:
                print(f"  ... and {len(diag_files) - 5} more files")
        
        return diag_files
    
    def parse_file_datetimes(self, diag_files: List[str]) -> List[datetime]:
        """
        Parse datetime information from diagnostic filenames.

        Parameters:
            diag_files (List[str]): List of diagnostic file paths.

        Returns:
            List[datetime]: List of datetime objects parsed from filenames.
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
                    if self.verbose:
                        print(f"Warning: Invalid datetime parsed from filename: {filename}")
                    file_dt = datetime(2000, 1, 1) + pd.Timedelta(hours=len(file_datetimes))
            else:
                if self.verbose:
                    print(f"Warning: Could not parse datetime from filename: {filename}")
                file_dt = datetime(2000, 1, 1) + pd.Timedelta(hours=len(file_datetimes))
            
            file_datetimes.append(file_dt)
        return file_datetimes
    
    def add_spatial_coordinates(self, combined_ds: xr.Dataset) -> xr.Dataset:
        """
        Add spatial coordinates from MPAS grid file to the dataset.

        This function adds coordinate variables from the MPAS grid file to eliminate
        "Dimensions without coordinates" warnings and improve dataset usability.

        Parameters:
            combined_ds (xr.Dataset): Combined xarray dataset with time series data.

        Returns:
            xr.Dataset: Dataset with added coordinate variables.
        """
        try:
            grid_file_ds = xr.open_dataset(self.grid_file)
            if self.verbose:
                print(f"\nGrid file loaded successfully with variables: \n{list(grid_file_ds.variables.keys())}\n")
            
            coords_to_add = {}
            data_vars_to_add = {}
            
            if 'lonCell' in grid_file_ds.variables and 'nCells' in combined_ds.dims:
                coords_to_add['nCells'] = ('nCells', np.arange(combined_ds.sizes['nCells'])) 
                if self.verbose:
                    print(f"Added nCells index coordinate for nCells dimension ({combined_ds.sizes['nCells']} values)")
            
            if 'lonVertex' in grid_file_ds.variables and 'nVertices' in combined_ds.dims:
                coords_to_add['nVertices'] = ('nVertices', np.arange(combined_ds.sizes['nVertices']))  
                if self.verbose:
                    print(f"Added nVertices index coordinate for nVertices dimension ({combined_ds.sizes['nVertices']} values)")

            if 'nIsoLevelsT' in combined_ds.dims:
                coords_to_add['nIsoLevelsT'] = ('nIsoLevelsT', np.arange(combined_ds.sizes['nIsoLevelsT']))
                if self.verbose:
                    print(f"Added nIsoLevelsT index coordinate for nIsoLevelsT dimension ({combined_ds.sizes['nIsoLevelsT']} values)")

            if 'nIsoLevelsZ' in combined_ds.dims:
                coords_to_add['nIsoLevelsZ'] = ('nIsoLevelsZ', np.arange(combined_ds.sizes['nIsoLevelsZ']))
                if self.verbose:
                    print(f"Added nIsoLevelsZ index coordinate for nIsoLevelsZ dimension ({combined_ds.sizes['nIsoLevelsZ']} values)\n")
            
            spatial_vars = ['latCell', 'lonCell', 'latVertex', 'lonVertex']

            for var_name in spatial_vars:
                if var_name in grid_file_ds.variables and var_name not in combined_ds.data_vars:
                    var_data = grid_file_ds[var_name]
                    data_vars_to_add[var_name] = var_data
                    if self.verbose:
                        print(f"Added spatial coordinate variable: {var_name}")
            
            if coords_to_add:
                combined_ds = combined_ds.assign_coords(coords_to_add)
                if self.verbose:
                    print(f"\nSuccessfully added {len(coords_to_add)} coordinate variables")
                
            if data_vars_to_add:
                for var_name, var_data in data_vars_to_add.items():
                    combined_ds[var_name] = var_data
                if self.verbose:
                    print(f"Successfully added {len(data_vars_to_add)} spatial variables")
                    print("\nUpdated dataset coordinates:", list(combined_ds.coords.keys()))
            else:
                if self.verbose:
                    print("No additional coordinate variables found to add")
                
            grid_file_ds.close()
            
        except Exception as coord_error:
            if self.verbose:
                print(f"Warning: Could not add spatial coordinates: {coord_error}")
                print("Continuing without additional coordinates...")
        
        return combined_ds
    
    def load_data(self, data_dir: str, use_pure_xarray: bool = False, 
                  reference_file: str = "") -> Tuple[Any, str]:
        """
        Load MPAS data from multiple diagnostic files with lazy loading.

        Parameters:
            data_dir (str): Directory containing diagnostic files.
            use_pure_xarray (bool): If True, use pure xarray instead of UXarray.
            reference_file (str): Optional reference file for time ordering.

        Returns:
            Tuple[Any, str]: Dataset object and data type identifier ('xarray' or 'uxarray').
        """
        diag_files = self.find_diagnostic_files(data_dir)
        file_datetimes = self.parse_file_datetimes(diag_files)
        
        try:
            combined_ds = xr.open_mfdataset(
                diag_files,
                combine='nested',
                concat_dim='Time',
                chunks={'Time': 1, 'n_face': 100000},  
                parallel=False  
            )
            
            combined_ds = combined_ds.assign_coords(Time=pd.to_datetime(file_datetimes))
            combined_ds = combined_ds.sortby('Time')        
            combined_ds = self.add_spatial_coordinates(combined_ds)
            
            if self.verbose:
                print("\nDataset structure:")
                print(combined_ds)
            
            if use_pure_xarray:
                if self.verbose:
                    print(f"\nSuccessfully loaded {len(diag_files)} files with pure xarray (lazy)")
                    print(f"Combined dataset time dimension: {combined_ds.Time.shape}")
                    print(f"\nTime range: {combined_ds.Time.values[0]} to {combined_ds.Time.values[-1]}")
                    print("Memory usage: Dataset uses chunked/lazy arrays")
                    print("Data loaded as: pure xarray")
                self.dataset = combined_ds
                self.data_type = 'xarray'
                return combined_ds, 'xarray'
            else:
                grid_ds = ux.open_dataset(self.grid_file, diag_files[0])
                grid_info = grid_ds.uxgrid

                if grid_info is None:
                    raise ValueError("Could not extract uxgrid from grid dataset")

                final_ds = ux.UxDataset(combined_ds, uxgrid=grid_info)
                if self.verbose:
                    print(f"\nSuccessfully loaded {len(diag_files)} files with UXarray (lazy)")
                    print(f"Combined dataset time dimension: {final_ds.Time.shape}")
                    print(f"\nTime range: {final_ds.Time.values[0]} to {final_ds.Time.values[-1]}")
                self.dataset = final_ds
                self.data_type = 'uxarray'
                return final_ds, 'uxarray'
                
        except Exception as e:
            if self.verbose:
                print(f"Primary loading failed: {e}")
                print("Trying xarray fallback...")
            
            try:
                combined_ds = xr.open_mfdataset(
                    diag_files,
                    combine='nested', 
                    concat_dim='Time',
                    chunks={'Time': 1},
                    parallel=False
                )
                
                combined_ds = combined_ds.assign_coords(Time=pd.to_datetime(file_datetimes))
                combined_ds = combined_ds.sortby('Time')
                
                if self.verbose:
                    print(f"Successfully loaded {len(diag_files)} files with xarray (lazy)")
                    print(f"Combined dataset time dimension: {combined_ds.Time.shape}")
                    print(f"Time range: {combined_ds.Time.values[0]} to {combined_ds.Time.values[-1]}")
                    print("Memory usage: Dataset uses chunked/lazy arrays")
                
                self.dataset = combined_ds
                self.data_type = 'xarray'
                return combined_ds, 'xarray'
                
            except Exception as e2:
                if self.verbose:
                    print(f"Xarray fallback also failed: {e2}")
                try:
                    return self._load_single_file_fallback(reference_file, diag_files)
                except Exception as e3:
                    print(f"All loading strategies failed: {e3}")
                    sys.exit(1)
    
    def _load_single_file_fallback(self, reference_file: str, diag_files: List[str]) -> Tuple[Any, str]:
        """
        Fallback loading strategy using a single file when multi-file loading fails.

        Parameters:
            reference_file (str): Optional reference file path.
            diag_files (List[str]): List of diagnostic files.

        Returns:
            Tuple[Any, str]: Dataset and data type identifier ('xarray' or 'uxarray').
        """
        if self.verbose:
            print("Falling back to single-file loading (limited functionality)...")
        
        if reference_file and os.path.exists(reference_file):
            try:
                ds = ux.open_dataset(self.grid_file, reference_file)
                if self.verbose:
                    print(f"Loaded single file: {reference_file}")
                self.dataset = ds
                self.data_type = 'uxarray'
                return ds, 'uxarray'
            except:
                ds = xr.open_dataset(reference_file)
                if self.verbose:
                    print(f"Loaded single file with xarray: {reference_file}")
                self.dataset = ds
                self.data_type = 'xarray'
                return ds, 'xarray'
        else:
            try:
                ds = ux.open_dataset(self.grid_file, diag_files[0])
                if self.verbose:
                    print(f"Loaded first file: {diag_files[0]}")
                self.dataset = ds
                self.data_type = 'uxarray'
                return ds, 'uxarray'
            except:
                ds = xr.open_dataset(diag_files[0])
                if self.verbose:
                    print(f"Loaded first file with xarray: {diag_files[0]}")
                self.dataset = ds
                self.data_type = 'xarray'
                return ds, 'xarray'
    
    def validate_time_parameters(self, time_index: int) -> Tuple[str, int, int]:
        """
        Validate time parameters and return normalized values.

        Parameters:
            time_index (int): Requested time index.

        Returns:
            Tuple[str, int, int]: (time_dimension_name, validated_time_index, total_time_steps).
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        time_dim = 'Time' if 'Time' in self.dataset.dims else 'time'
        time_size = self.dataset.sizes[time_dim]
        
        if time_index >= time_size:
            if self.verbose:
                print(f"Warning: time_index {time_index} exceeds available times {time_size}, using last time")
            time_index = time_size - 1
        
        return time_dim, time_index, time_size
    
    def get_time_info(self, time_index: int, var_context: str = "") -> str:
        """
        Get time coordinate information for diagnostics.

        Parameters:
            time_index (int): Time index to describe.
            var_context (str): Additional context for variable-specific messages.

        Returns:
            str: Formatted time string.
        """
        try:
            if hasattr(self.dataset, 'Time') and len(self.dataset.Time) > time_index:
                time_value = self.dataset.Time.values[time_index]
                if hasattr(time_value, 'strftime'):
                    time_str = time_value.strftime('%Y%m%dT%H')
                else:
                    time_dt = pd.to_datetime(time_value)
                    time_str = time_dt.strftime('%Y%m%dT%H')
                
                if self.verbose:
                    context_msg = f" (using variable: {var_context})" if var_context else ""
                    print(f"Time index {time_index} corresponds to: {time_str}{context_msg}")
                
                return time_str
            else:
                if self.verbose:
                    context_msg = f" (time coordinate not available, using variable: {var_context})" if var_context else " (time coordinate not available)"
                    print(f"Using time index {time_index}{context_msg}")
                return f"time_{time_index}"
        except Exception as e:
            if self.verbose:
                context_msg = f" (could not parse time: {e}, using variable: {var_context})" if var_context else f" (could not parse time: {e})"
                print(f"Using time index {time_index}{context_msg}")
            return f"time_{time_index}"
    
    def compute_precipitation_difference(self, time_index: int, var_name: str = 'rainnc', accum_period: str = 'a01h') -> xr.DataArray:
        """
        Compute precipitation from cumulative precipitation data for specified accumulation period.

        Parameters:
            time_index (int): Time index for current time step.
            var_name (str): Precipitation variable name ('rainc', 'rainnc', or 'total').
            accum_period (str): Accumulation period (e.g., 'a01h', 'a03h', 'a06h', 'a12h', 'a24h').

        Returns:
            xr.DataArray: Precipitation data for the specified accumulation period.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        from .utils import get_accumulation_hours
        
        time_dim, time_index, time_size = self.validate_time_parameters(time_index)
        
        accum_hours = get_accumulation_hours(accum_period)
        time_step_diff = accum_hours
        
        if time_index < time_step_diff:
            if time_index == 0:
                return self._handle_first_time_step(time_dim, var_name, accum_period, accum_hours)
            else:
                if self.verbose:
                    print(f"Warning: Time index {time_index} < required {accum_hours}-hour lookback ({time_step_diff} steps)")
                    print(f"Skipping time index {time_index} - insufficient data for {accum_hours}-hour accumulation")
                
                try:
                    sample_data = self.dataset[var_name].isel({time_dim: time_index})
                    if hasattr(sample_data, 'compute'):
                        sample_data = sample_data.compute()
                    
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
                        print(f"Error creating zero precipitation field: {e}")
                    raise ValueError(f"Cannot compute {accum_hours}-hour accumulation for time index {time_index}: insufficient data")
        
        if self.verbose:
            print(f"DEBUG: Computing {accum_hours}-hour accumulation (period: {accum_period})")
            print(f"DEBUG: time_index = {time_index}, time_step_diff = {time_step_diff}")
            self._print_time_slice_info(time_index, var_name, time_step_diff)
        
        if var_name == 'total':
            current_rainc = self.dataset['rainc'].isel({time_dim: time_index})
            current_rainnc = self.dataset['rainnc'].isel({time_dim: time_index})
            previous_rainc = self.dataset['rainc'].isel({time_dim: time_index - time_step_diff})
            previous_rainnc = self.dataset['rainnc'].isel({time_dim: time_index - time_step_diff})
            
            if hasattr(current_rainc, 'compute'):
                current_rainc = current_rainc.compute()
                current_rainnc = current_rainnc.compute()
                previous_rainc = previous_rainc.compute()
                previous_rainnc = previous_rainnc.compute()
            
            current_total = current_rainc + current_rainnc
            previous_total = previous_rainc + previous_rainnc
            
            if self.verbose:
                self._analyze_precipitation_diagnostics(current_total, previous_total, var_context=var_name)
            
            accum_precip = current_total - previous_total
            
        else:
            current_data = self.dataset[var_name].isel({time_dim: time_index})
            previous_data = self.dataset[var_name].isel({time_dim: time_index - time_step_diff})
            
            if hasattr(current_data, 'compute'):
                current_data = current_data.compute()
                previous_data = previous_data.compute()
            
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
    
    def _handle_first_time_step(self, time_dim: str, var_name: str, accum_period: str = 'a01h', accum_hours: int = 1) -> xr.DataArray:
        """
        Handle the special case of time index 0.

        Parameters:
            time_dim (str): Time dimension name.
            var_name (str): Variable name.
            accum_period (str): Accumulation period.
            accum_hours (int): Number of hours for accumulation.

        Returns:
            xr.DataArray: Data at time index 0.
        """
        try:
            if var_name == 'total':
                rainc_data = self.dataset['rainc'].isel({time_dim: 0})
                rainnc_data = self.dataset['rainnc'].isel({time_dim: 0})
                sample_data = rainc_data + rainnc_data
            else:
                sample_data = self.dataset[var_name].isel({time_dim: 0})
            
            if hasattr(sample_data, 'compute'):
                sample_data = sample_data.compute()
            
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
            
            if self.verbose:
                data_values = hourly_precip_at_time.values.flatten()
                valid_data = data_values[np.isfinite(data_values)]
                if len(valid_data) > 0:
                    data_min, data_max = np.nanmin(valid_data), np.nanmax(valid_data)
                    print(f"Data range at time 0: {data_min:.3f} to {data_max:.3f} mm")
            
            return hourly_precip_at_time
            
        except Exception as e:
            if self.verbose:
                print(f"Time index 0 requested - could not access data from file ({e}), creating zeros, variable: {var_name}")
            
            if var_name == 'total':
                reference_var = 'rainc' if 'rainc' in self.dataset else 'rainnc'
            else:
                reference_var = var_name
                
            try:
                sample_data = self.dataset[reference_var].isel({time_dim: 0})
            except Exception:
                raise ValueError(f"Cannot access variable '{reference_var}' at time index 0")
        
            hourly_precip_at_time = xr.zeros_like(sample_data)
            long_name = f'hourly accumulated precipitation from {var_name}'

            hourly_precip_at_time.attrs.update({
                'units': 'mm',
                'standard_name': 'precipitation',
                'long_name': long_name,
            })
            return hourly_precip_at_time
    
    def _print_time_slice_info(self, time_index: int, var_context: str = "", time_step_diff: int = 1) -> None:
        """
        Print information about the time slices being loaded.

        Parameters:
            time_index (int): Time index being processed.
            var_context (str): Variable context for messaging.
            time_step_diff (int): Time step difference for accumulation.

        Returns:
            None
        """
        prev_time_index = time_index - time_step_diff
        context_msg = f" for {var_context} difference calculation" if var_context else " for difference calculation"
        print(f"\nLoading only time slices {prev_time_index} and {time_index}{context_msg}\n")
        
        try:
            if hasattr(self.dataset, 'Time') and len(self.dataset.Time) > time_index:
                prev_time_value = self.dataset.Time.values[prev_time_index]
                prev_time_dt = pd.to_datetime(prev_time_value)
                prev_time_str = prev_time_dt.strftime('%Y%m%dT%H')
                
                curr_time_value = self.dataset.Time.values[time_index]
                curr_time_dt = pd.to_datetime(curr_time_value)
                curr_time_str = curr_time_dt.strftime('%Y%m%dT%H')
                
                print(f"Previous time slice: {prev_time_str} (index {prev_time_index})")
                print(f"Current time slice: {curr_time_str} (index {time_index})")

                if var_context:
                    print(f"\n............... Computing {var_context} difference ...............\n")
            else:
                print(f"Using time indices {prev_time_index} and {time_index} (time coordinates not available)")
        except Exception as e:
            print(f"Using time indices {prev_time_index} and {time_index} (could not parse times: {e})")
    
    def _apply_precipitation_filters_and_attributes(self, data: xr.DataArray, var_context: str = "") -> xr.DataArray:
        """
        Apply standard precipitation data filters and set attributes.

        Parameters:
            data (xr.DataArray): Precipitation data array.
            var_context (str): Variable context for long_name.

        Returns:
            xr.DataArray: Filtered data with updated attributes.
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
        Perform comprehensive diagnostic analysis of precipitation data.

        Parameters:
            current_data (Any): Current time-step data.
            previous_data (Any): Previous time-step data.
            result_data (Any): Computed difference/result data.
            var_context (str): Variable context for messaging.

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
                print(f"Previous {var_label} range: {prev_min:.2f} to {prev_max:.2f} mm\n")
                
                if curr_max < prev_max:
                    print(f"WARNING: Current max ({curr_max:.2f}) < Previous max ({prev_max:.2f}) - possible data loading issue!")
                
            except Exception as debug_error:
                print(f"Debug info extraction failed: {debug_error}")
        
        if result_data is not None:
            try:
                precip_min = float(result_data.min())
                precip_max = float(result_data.max())
                var_label = var_context if var_context else "precipitation"

                print(f"\nHourly {var_label} difference range: {precip_min:.2f} to {precip_max:.2f} mm/hour")
                
                hourly_values = result_data.values.flatten()
                hourly_valid = hourly_values[np.isfinite(hourly_values)]
                above_50mm = hourly_valid[hourly_valid > 50.0]
                total_valid_points = len(hourly_valid)
                above_50mm_count = len(above_50mm)
                
                if total_valid_points > 0:
                    percentage_above_50mm = 100 * above_50mm_count / total_valid_points
                    print(f"Areas above 50mm/hour threshold: {above_50mm_count}/{total_valid_points} ({percentage_above_50mm:.2f}%)")
                else:
                    print("No valid hourly precipitation points found")
                    
            except Exception as analysis_error:
                print(f"Hourly precipitation analysis failed: {analysis_error}")
    
    def get_available_variables(self) -> List[str]:
        """
        Get list of available variables in the loaded dataset.

        Parameters:
            None

        Returns:
            List[str]: List of variable names.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        return list(self.dataset.data_vars.keys())
    
    def get_time_range(self) -> Tuple[datetime, datetime]:
        """
        Get the time range of the loaded dataset.

        Parameters:
            None

        Returns:
            Tuple[datetime, datetime]: Start and end times.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        time_values = self.dataset.Time.values
        start_time = pd.to_datetime(time_values[0]).floor('s').to_pydatetime()
        end_time = pd.to_datetime(time_values[-1]).floor('s').to_pydatetime()
        
        return start_time, end_time
    
    def extract_spatial_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract spatial coordinates from the dataset.

        Parameters:
            None

        Returns:
            Tuple[np.ndarray, np.ndarray]: Longitude and latitude arrays.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        lon_names = ['lonCell', 'longitude', 'lon']
        lat_names = ['latCell', 'latitude', 'lat']
        
        lon_coords = lat_coords = None
        
        for name in lon_names:
            if name in self.dataset.coords or name in self.dataset.data_vars:
                lon_coords = self.dataset[name].values
                break
                
        for name in lat_names:
            if name in self.dataset.coords or name in self.dataset.data_vars:
                lat_coords = self.dataset[name].values
                break
                
        if lon_coords is None or lat_coords is None:
            available_vars = list(self.dataset.coords.keys()) + list(self.dataset.data_vars.keys())
            raise ValueError(f"Could not find spatial coordinates. Available variables: {available_vars}")
        
        if np.nanmax(np.abs(lat_coords)) <= np.pi:
            lat_coords = lat_coords * 180.0 / np.pi
            lon_coords = lon_coords * 180.0 / np.pi
        
        lon_coords = lon_coords.ravel()
        lat_coords = lat_coords.ravel()
        
        lon_coords = ((lon_coords + 180) % 360) - 180
        
        return lon_coords, lat_coords
    
    def filter_by_spatial_extent(self, data: xr.DataArray, 
                                 lon_min: float, lon_max: float, 
                                 lat_min: float, lat_max: float) -> Tuple[xr.DataArray, np.ndarray]:
        """
        Filter data by spatial extent.

        Parameters:
            data (xr.DataArray): Data to filter.
            lon_min (float): Minimum longitude bound.
            lon_max (float): Maximum longitude bound.
            lat_min (float): Minimum latitude bound.
            lat_max (float): Maximum latitude bound.

        Returns:
            Tuple[xr.DataArray, np.ndarray]: Filtered data and spatial mask.
        """
        lon, lat = self.extract_spatial_coordinates()
        
        mask = ((lon >= lon_min) & (lon <= lon_max) & 
                (lat >= lat_min) & (lat <= lat_max))
        
        if 'nCells' in data.dims:
            filtered_data = data.where(mask)
        else:
            filtered_data = data
        
        return filtered_data, mask
    
    def get_variable_data(self, var_name: str, time_index: int = 0) -> xr.DataArray:
        """
        Extract data for any MPAS variable at a specific time index.

        Parameters:
            var_name (str): Variable name to extract.
            time_index (int): Time index to extract data from.

        Returns:
            xr.DataArray: Variable data at specified time index.

        Raises:
            ValueError: If variable not found or time index invalid.
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_data() first.")
        
        if var_name not in self.dataset.data_vars:
            available_vars = list(self.dataset.data_vars.keys())
            raise ValueError(f"Variable '{var_name}' not found. Available variables: {available_vars}")
        
        time_dim, validated_time_index, time_size = self.validate_time_parameters(time_index)
        
        if self.verbose:
            print(f"Extracting {var_name} data at time index {validated_time_index}")
        
        var_data = self.dataset[var_name].isel({time_dim: validated_time_index})
        
        if hasattr(var_data, 'compute'):
            var_data = var_data.compute()
        
        if self.verbose:
            if hasattr(var_data, 'values'):
                data_values = var_data.values.flatten()
                finite_values = data_values[np.isfinite(data_values)]
                if len(finite_values) > 0:
                    print(f"Variable {var_name} range: {finite_values.min():.3f} to {finite_values.max():.3f}")
                    
                    if hasattr(var_data, 'attrs') and 'units' in var_data.attrs:
                        print(f"Units: {var_data.attrs['units']}")
                else:
                    print(f"Warning: No finite values found for {var_name}")
        
        return var_data

    def get_wind_components(self, u_variable: str, v_variable: str, time_index: int = 0) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Extract U and V wind components from the dataset.

        Parameters:
            u_variable (str): Name of U-component wind variable (e.g., 'u10', 'u850').
            v_variable (str): Name of V-component wind variable (e.g., 'v10', 'v850').
            time_index (int): Time index to extract (default: 0).

        Returns:
            Tuple[xr.DataArray, xr.DataArray]: U and V wind component data arrays.

        Raises:
            ValueError: If wind variables are not found in dataset.
            IndexError: If time_index is out of range.
        """
        if self.dataset is None:
            raise RuntimeError("No dataset loaded. Call load_data() first.")
        
        print(f"Extracting wind components {u_variable}, {v_variable} at time index {time_index}")
        
        available_vars = list(self.dataset.data_vars.keys())
        missing_vars = []
        
        if u_variable not in available_vars:
            missing_vars.append(u_variable)
        if v_variable not in available_vars:
            missing_vars.append(v_variable)
            
        if missing_vars:
            raise ValueError(f"Wind variables {missing_vars} not found in dataset. Available variables: {available_vars[:20]}...")
        
        time_dim = self.dataset.sizes.get('Time', 0)
        if time_index >= time_dim:
            raise IndexError(f"Time index {time_index} out of range. Dataset has {time_dim} time steps.")
        
        try:
            u_data = self.dataset[u_variable].isel(Time=time_index)
            v_data = self.dataset[v_variable].isel(Time=time_index)
            
            if hasattr(u_data.data, 'compute'):
                u_data = u_data.compute()

            if hasattr(v_data.data, 'compute'):
                v_data = v_data.compute()
            
            u_min, u_max = float(u_data.min()), float(u_data.max())
            v_min, v_max = float(v_data.min()), float(v_data.max())
            
            wind_speed = np.sqrt(u_data**2 + v_data**2)
            wind_min, wind_max = float(wind_speed.min()), float(wind_speed.max())
            
            print(f"Wind component {u_variable} range: {u_min:.2f} to {u_max:.2f} m/s")
            print(f"Wind component {v_variable} range: {v_min:.2f} to {v_max:.2f} m/s") 
            print(f"Wind speed range: {wind_min:.2f} to {wind_max:.2f} m/s")
            
            u_units = getattr(u_data, 'units', 'm s^{-1}')
            v_units = getattr(v_data, 'units', 'm s^{-1}')

            print(f"Units: {u_units}")
            
            if u_units != v_units:
                print(f"Warning: U and V components have different units: {u_units} vs {v_units}")
            
            return u_data, v_data
            
        except KeyError as e:
            raise ValueError(f"Error accessing wind variables: {e}")
        except Exception as e:
            raise RuntimeError(f"Error extracting wind components: {e}")


def validate_geographic_extent(extent: Tuple[float, float, float, float]) -> bool:
    """
    Validate if geographic extent coordinates are within valid ranges and properly ordered.

    Parameters:
        extent (Tuple[float, float, float, float]): Geographic extent as (lon_min, lon_max, lat_min, lat_max).

    Returns:
        bool: True if extent is valid, False otherwise.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    return (
        -180.0 <= lon_min <= 180.0 and -180.0 <= lon_max <= 180.0
        and -90.0 <= lat_min <= 90.0 and -90.0 <= lat_max <= 90.0
        and lon_max > lon_min and lat_max > lat_min
    )


def normalize_longitude(lon: np.ndarray) -> np.ndarray:
    """
    Normalize longitude array to [-180, 180].

    Parameters:
        lon (np.ndarray): Longitude array to normalize.

    Returns:
        np.ndarray: Normalized longitude array.
    """
    lon = np.asarray(lon)
    lon = ((lon + 180) % 360) - 180
    return lon


def get_accumulation_hours(accum_period: str) -> int:
    """
    Get accumulation hours from accumulation period string.

    Parameters:
        accum_period (str): Accumulation period identifier (e.g., 'a01h', 'a24h').

    Returns:
        int: Number of hours for the accumulation period.
    """
    accum_hours_map = {'a01h': 1, 'a03h': 3, 'a06h': 6, 'a12h': 12, 'a24h': 24}
    return accum_hours_map.get(accum_period, 24)
