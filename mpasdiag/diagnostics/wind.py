#!/usr/bin/env python3

"""
MPAS Wind Diagnostics

This module provides specialized functionality for wind analysis and calculations from MPAS model output data including wind component extraction, speed and direction computations, and vector manipulations. It includes methods for computing wind speed from u/v components using vectorized operations, calculating wind direction with meteorological conventions (direction from which wind blows), extracting zonal and meridional wind components at various pressure levels or model layers, and performing wind-related statistical diagnostics. The module is designed to integrate seamlessly with the broader MPASdiag framework, leveraging existing data processing utilities for handling MPAS unstructured mesh data and 3D atmospheric fields. Core capabilities include robust handling of missing data with NaN filtering, unit-aware calculations, coordinate transformations for rotated wind components, and efficient numpy-based vectorized operations for high-performance wind field analysis suitable for operational weather diagnostics and atmospheric dynamics research.

Classes:
    WindDiagnostics: Specialized class for performing wind-specific diagnostics and calculations from MPAS datasets.
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import numpy as np
import xarray as xr
from typing import Tuple, Union, Optional, Any, cast

WIND_SPEED_UNITS = 'm s^{-1}'


class WindDiagnostics:
    """
    Specialized diagnostics for wind calculations from MPAS data.
    
    This class provides methods for wind component analysis, wind speed calculations,
    and wind direction computations from MPAS model output.
    """
    
    def __init__(self, verbose: bool = True) -> None:
        """
        Initialize the WindDiagnostics class for analyzing wind data from MPAS model output.
        
        This constructor sets up the wind diagnostics instance with configurable verbosity for controlling console output during wind calculations. The verbose parameter allows users to enable or disable detailed diagnostic messages during wind component extraction, speed calculations, and direction computations. This initialization prepares the instance for subsequent wind analysis operations on MPAS datasets.
        
        Parameters:
            verbose (bool): Enable verbose output messages during wind calculations (default: True).
        
        Returns:
            None
        """
        self.verbose = verbose
    
    def compute_wind_speed(self, u_component: xr.DataArray, v_component: xr.DataArray) -> xr.DataArray:
        """
        Compute horizontal wind speed magnitude from U and V wind components using vector magnitude calculation. This method calculates the Euclidean norm of the horizontal wind vector by taking the square root of the sum of squared U and V components. The computation preserves xarray metadata and attributes while applying numpy's efficient mathematical operations. The resulting wind speed data array includes proper units, standard name, and long name attributes following CF conventions. If verbose mode is enabled, the method prints diagnostic information about the input component ranges and output speed statistics.

        Parameters:
            u_component (xr.DataArray): U (zonal) wind component in meters per second.
            v_component (xr.DataArray): V (meridional) wind component in meters per second.

        Returns:
            xr.DataArray: Horizontal wind speed magnitude in meters per second with CF-compliant attributes.
        """
        wind_speed = xr.apply_ufunc(
            np.sqrt,
            u_component**2 + v_component**2,
            keep_attrs=True
        )
        wind_speed.attrs.update({
            'units': WIND_SPEED_UNITS,
            'standard_name': 'wind_speed',
            'long_name': 'horizontal wind speed',
        })
        
        if self.verbose:
            u_min, u_max = float(u_component.min()), float(u_component.max())
            v_min, v_max = float(v_component.min()), float(v_component.max())
            speed_min, speed_max = float(wind_speed.min()), float(wind_speed.max())
            speed_mean = float(wind_speed.mean())
            
            print(f"Wind component U range: {u_min:.2f} to {u_max:.2f} m/s")
            print(f"Wind component V range: {v_min:.2f} to {v_max:.2f} m/s")
            print(f"Wind speed range: {speed_min:.2f} to {speed_max:.2f} m/s")
            print(f"Wind speed mean: {speed_mean:.2f} m/s")
        
        return wind_speed
    
    def compute_wind_direction(self, u_component: xr.DataArray, v_component: xr.DataArray, 
                             degrees: bool = True) -> xr.DataArray:
        """
        Compute meteorological wind direction from U and V components following standard conventions. This method calculates the direction from which the wind is blowing using arctan2 to determine the angle from the wind vector components. The result follows the meteorological convention where 0 degrees represents wind from the north, 90 degrees from the east, 180 degrees from the south, and 270 degrees from the west. The calculation preserves xarray metadata and can return results in either degrees or radians based on the degrees parameter. If verbose mode is enabled, the method prints diagnostic information about the computed direction range and mean values.

        Parameters:
            u_component (xr.DataArray): U (zonal) wind component in meters per second.
            v_component (xr.DataArray): V (meridional) wind component in meters per second.
            degrees (bool): Return direction in degrees if True, radians if False (default: True).

        Returns:
            xr.DataArray: Wind direction following meteorological convention where direction indicates where wind is coming from, with 0=North, 90=East, 180=South, 270=West for degrees or 0=North, π/2=East, π=South, 3π/2=West for radians.
        """
        direction_rad = xr.apply_ufunc(
            np.arctan2,
            v_component,
            u_component,
            keep_attrs=True
        )
        direction_rad = direction_rad + np.pi
        
        if degrees:
            direction_deg = xr.apply_ufunc(
                np.rad2deg,
                direction_rad,
                keep_attrs=True
            )
            direction_deg = direction_deg % 360
            
            direction_deg.attrs.update({
                'units': 'degrees',
                'standard_name': 'wind_from_direction',
                'long_name': 'wind direction (meteorological convention)',
                'note': 'Direction wind is coming from, 0=North, 90=East, 180=South, 270=West'
            })
            
            if self.verbose:
                dir_min, dir_max = float(direction_deg.min()), float(direction_deg.max())
                dir_mean = float(direction_deg.mean())
                print(f"Wind direction range: {dir_min:.1f} to {dir_max:.1f} degrees")
                print(f"Wind direction mean: {dir_mean:.1f} degrees")
            
            return direction_deg
        else:
            direction_rad = direction_rad % (2 * np.pi)
            
            direction_rad.attrs.update({
                'units': 'radians',
                'standard_name': 'wind_from_direction',
                'long_name': 'wind direction (meteorological convention)',
                'note': 'Direction wind is coming from, 0=North, π/2=East, π=South, 3π/2=West'
            })
            
            if self.verbose:
                dir_min, dir_max = float(direction_rad.min()), float(direction_rad.max())
                dir_mean = float(direction_rad.mean())
                print(f"Wind direction range: {dir_min:.3f} to {dir_max:.3f} radians")
                print(f"Wind direction mean: {dir_mean:.3f} radians")
            
            return direction_rad
    
    def analyze_wind_components(self, u_component: xr.DataArray, v_component: xr.DataArray, 
                               w_component: Optional[xr.DataArray] = None) -> dict:
        """
        Perform comprehensive statistical analysis of wind components and compute derived quantities. This method extracts minimum, maximum, mean, and standard deviation values for each wind component and calculates derived quantities including horizontal wind speed and wind direction. If a vertical wind component is provided, the method also computes full 3D wind speed statistics. The results are organized in a dictionary structure with separate entries for each component and derived field. If verbose mode is enabled, the method prints a formatted summary of all computed statistics to the console.

        Parameters:
            u_component (xr.DataArray): U (zonal) wind component in meters per second.
            v_component (xr.DataArray): V (meridional) wind component in meters per second.
            w_component (Optional[xr.DataArray]): W (vertical) wind component in meters per second (default: None).

        Returns:
            dict: Dictionary containing comprehensive wind analysis results with keys 'u_component', 'v_component', 'horizontal_speed', 'direction', and optionally 'w_component' and 'total_speed' if vertical component is provided, each containing min, max, mean, std, and units.
        """
        analysis = {}
        
        analysis['u_component'] = {
            'min': float(u_component.min()),
            'max': float(u_component.max()),
            'mean': float(u_component.mean()),
            'std': float(u_component.std()),
            'units': u_component.attrs.get('units', WIND_SPEED_UNITS)
        }
        
        analysis['v_component'] = {
            'min': float(v_component.min()),
            'max': float(v_component.max()),
            'mean': float(v_component.mean()),
            'std': float(v_component.std()),
            'units': v_component.attrs.get('units', WIND_SPEED_UNITS)
        }
        
        wind_speed = self.compute_wind_speed(u_component, v_component)
        analysis['horizontal_speed'] = {
            'min': float(wind_speed.min()),
            'max': float(wind_speed.max()),
            'mean': float(wind_speed.mean()),
            'std': float(wind_speed.std()),
            'units': WIND_SPEED_UNITS
        }
        
        wind_direction = self.compute_wind_direction(u_component, v_component, degrees=True)
        analysis['direction'] = {
            'min': float(wind_direction.min()),
            'max': float(wind_direction.max()),
            'mean': float(wind_direction.mean()),
            'std': float(wind_direction.std()),
            'units': 'degrees'
        }
        
        if w_component is not None:
            analysis['w_component'] = {
                'min': float(w_component.min()),
                'max': float(w_component.max()),
                'mean': float(w_component.mean()),
                'std': float(w_component.std()),
                'units': w_component.attrs.get('units', WIND_SPEED_UNITS)
            }
            
            wind_speed_3d = np.sqrt(u_component**2 + v_component**2 + w_component**2)
            analysis['total_speed'] = {
                'min': float(wind_speed_3d.min()),
                'max': float(wind_speed_3d.max()),
                'mean': float(wind_speed_3d.mean()),
                'std': float(wind_speed_3d.std()),
                'units': WIND_SPEED_UNITS
            }
        
        if self.verbose:
            print("Wind Component Analysis:")
            print(f"  U component: {analysis['u_component']['min']:.2f} to {analysis['u_component']['max']:.2f} m/s (mean: {analysis['u_component']['mean']:.2f})")
            print(f"  V component: {analysis['v_component']['min']:.2f} to {analysis['v_component']['max']:.2f} m/s (mean: {analysis['v_component']['mean']:.2f})")
            print(f"  Horizontal speed: {analysis['horizontal_speed']['min']:.2f} to {analysis['horizontal_speed']['max']:.2f} m/s (mean: {analysis['horizontal_speed']['mean']:.2f})")
            print(f"  Direction: {analysis['direction']['min']:.1f} to {analysis['direction']['max']:.1f} degrees (mean: {analysis['direction']['mean']:.1f})")
            
            if w_component is not None:
                print(f"  W component: {analysis['w_component']['min']:.2f} to {analysis['w_component']['max']:.2f} m/s (mean: {analysis['w_component']['mean']:.2f})")
                print(f"  Total 3D speed: {analysis['total_speed']['min']:.2f} to {analysis['total_speed']['max']:.2f} m/s (mean: {analysis['total_speed']['mean']:.2f})")
        
        return analysis
    
    def compute_wind_shear(self, u_upper: xr.DataArray, v_upper: xr.DataArray,
                          u_lower: xr.DataArray, v_lower: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Compute wind shear magnitude and direction between two vertical levels. This method calculates the vector difference between upper and lower level wind components and derives the shear magnitude using Euclidean distance. The shear direction is computed following meteorological conventions to indicate the direction toward which the shear vector points. The computation preserves xarray metadata and adds appropriate CF-compliant attributes to the output data arrays. If verbose mode is enabled, the method prints diagnostic information about the computed shear magnitude range and mean values.

        Parameters:
            u_upper (xr.DataArray): U component at upper level in meters per second.
            v_upper (xr.DataArray): V component at upper level in meters per second.
            u_lower (xr.DataArray): U component at lower level in meters per second.
            v_lower (xr.DataArray): V component at lower level in meters per second.

        Returns:
            Tuple[xr.DataArray, xr.DataArray]: Two-element tuple containing (shear_magnitude, shear_direction) where magnitude is in meters per second and direction follows meteorological convention in degrees.
        """
        du = u_upper - u_lower
        dv = v_upper - v_lower
        
        shear_magnitude = xr.apply_ufunc(
            np.sqrt,
            du**2 + dv**2,
            keep_attrs=True
        )
        shear_magnitude.attrs.update({
            'units': WIND_SPEED_UNITS,
            'standard_name': 'wind_shear_magnitude',
            'long_name': 'wind shear magnitude',
        })
        
        shear_direction = self.compute_wind_direction(du, dv, degrees=True)
        shear_direction.attrs['long_name'] = 'wind shear direction'
        
        if self.verbose:
            shear_min, shear_max = float(shear_magnitude.min()), float(shear_magnitude.max())
            shear_mean = float(shear_magnitude.mean())
            print(f"Wind shear magnitude range: {shear_min:.2f} to {shear_max:.2f} m/s")
            print(f"Wind shear magnitude mean: {shear_mean:.2f} m/s")
        
        return shear_magnitude, shear_direction

    def get_3d_wind_components(self, dataset: xr.Dataset, u_variable: str, v_variable: str, 
                               w_variable: str = 'w', level: Union[str, int, float] = 0, 
                               time_index: int = 0, data_type: str = 'xarray') -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        Extract U, V, and W wind components from 3D MPAS atmospheric dataset at specified vertical level and time. This method handles flexible level specification including integer model levels, pressure levels in Pascals, or string identifiers like 'surface' and 'top'. The extraction includes proper error handling for missing variables and automatic fallback to zero vertical velocity if W component extraction fails. The method validates time indices and vertical level bounds, and if verbose mode is enabled, prints diagnostic information about the extraction process and resulting data ranges. The extracted data arrays preserve all CF-compliant metadata and include additional attributes for the selected level and level index.

        Parameters:
            dataset (xr.Dataset): MPAS 3D dataset containing wind variables.
            u_variable (str): Name of U-component wind variable (e.g., 'uReconstructZonal').
            v_variable (str): Name of V-component wind variable (e.g., 'vReconstructMeridional').
            w_variable (str): Name of W-component wind variable (default: 'w').
            level (Union[str, int, float]): Vertical level specification as model level index, pressure in Pa, or 'surface'/'top'.
            time_index (int): Time index to extract from dataset (default: 0).
            data_type (str): Dataset type specification, either 'xarray' or 'uxarray' (default: 'xarray').

        Returns:
            Tuple[xr.DataArray, xr.DataArray, xr.DataArray]: Three-element tuple containing (u_component, v_component, w_component) as xarray DataArrays in meters per second with CF-compliant attributes and selected level metadata.

        Raises:
            ValueError: If wind variables are not found in dataset or if level specification is invalid.
        """
        from mpasdiag.processing.utils_datetime import MPASDateTimeUtils
        
        if self.verbose:
            print(f"Extracting 3D wind components {u_variable}, {v_variable}, {w_variable} at level {level}, time index {time_index}")
        
        def get_3d_variable_at_level(var_name: str, level_spec: Union[str, int, float], 
                                     time_idx: int) -> xr.DataArray:
            """Extract 3D variable data at specific level and time."""
            if var_name not in dataset.data_vars:
                raise ValueError(f"Variable '{var_name}' not found in dataset")
            
            var_dims = dataset[var_name].dims
            if 'nVertLevels' not in var_dims and 'nVertLevelsP1' not in var_dims:
                raise ValueError(f"Variable '{var_name}' is not a 3D atmospheric variable")
            
            time_dim, validated_time_index, _ = MPASDateTimeUtils.validate_time_parameters(dataset, time_idx, self.verbose)
            
            if isinstance(level_spec, int):
                level_idx = level_spec
                vertical_dim = 'nVertLevels' if 'nVertLevels' in var_dims else 'nVertLevelsP1'
                max_levels = dataset.sizes.get(vertical_dim, 0)
                if level_idx >= max_levels:
                    raise ValueError(f"Model level {level_idx} exceeds available levels {max_levels}")
                    
            elif isinstance(level_spec, float):
                if 'pressure_p' in dataset and 'pressure_base' in dataset:
                    pressure_p = dataset['pressure_p'].isel({time_dim: validated_time_index})
                    pressure_base = dataset['pressure_base'].isel({time_dim: validated_time_index})
                    total_pressure = pressure_p + pressure_base
                    
                    mean_pressure = total_pressure.mean(dim='nCells')
                    pressure_diff = np.abs(mean_pressure - level_spec)
                    level_idx = int(pressure_diff.argmin())
                    
                    if self.verbose:
                        target_p = mean_pressure.isel(nVertLevels=level_idx).values
                        print(f"Requested pressure: {level_spec:.1f} Pa, using level {level_idx}: {target_p:.1f} Pa")
                else:
                    raise ValueError("Cannot find pressure level - pressure data not available")
                    
            elif isinstance(level_spec, str):
                if level_spec.lower() == 'surface':
                    level_idx = 0
                elif level_spec.lower() == 'top':
                    vertical_dim = 'nVertLevels' if 'nVertLevels' in var_dims else 'nVertLevelsP1'
                    level_idx = dataset.sizes.get(vertical_dim, 1) - 1
                else:
                    raise ValueError(f"Unknown level specification: {level_spec}")
            else:
                raise ValueError(f"Invalid level specification: {level_spec}")
            
            vertical_dim = 'nVertLevels' if 'nVertLevels' in var_dims else 'nVertLevelsP1'
            
            if data_type == 'uxarray' and hasattr(dataset, '__getitem__'):
                var_data = dataset[var_name][validated_time_index].isel({vertical_dim: level_idx})
            else:
                var_data = dataset[var_name].isel({time_dim: validated_time_index, vertical_dim: level_idx})
            
            if hasattr(var_data, 'compute'):
                var_data = cast(Any, var_data).compute()
            
            if hasattr(var_data, 'attrs'):
                var_data.attrs['selected_level'] = level_spec
                var_data.attrs['level_index'] = level_idx
            
            return var_data
        
        try:
            u_data = get_3d_variable_at_level(u_variable, level, time_index)
            v_data = get_3d_variable_at_level(v_variable, level, time_index)
            
            try:
                w_data = get_3d_variable_at_level(w_variable, level, time_index)
            except (ValueError, IndexError) as e:
                if self.verbose:
                    print(f"Warning: Could not extract {w_variable} at level {level}: {e}")
                    print("Setting W component to zero...")
                w_data = xr.zeros_like(u_data)
                w_data.attrs['units'] = WIND_SPEED_UNITS
                w_data.attrs['long_name'] = f'Zero vertical velocity (could not extract {w_variable})'
            
            wind_speed = np.sqrt(u_data**2 + v_data**2)
            
            u_min, u_max = float(u_data.min()), float(u_data.max())
            v_min, v_max = float(v_data.min()), float(v_data.max())
            w_min, w_max = float(w_data.min()), float(w_data.max())
            wind_min, wind_max = float(wind_speed.min()), float(wind_speed.max())
            
            if self.verbose:
                print(f"Wind component {u_variable} range: {u_min:.2f} to {u_max:.2f} m/s")
                print(f"Wind component {v_variable} range: {v_min:.2f} to {v_max:.2f} m/s") 
                print(f"Wind component {w_variable} range: {w_min:.2f} to {w_max:.2f} m/s")
                print(f"Horizontal wind speed range: {wind_min:.2f} to {wind_max:.2f} m/s")
                
                u_units = u_data.attrs.get('units', WIND_SPEED_UNITS)
                print(f"Units: {u_units}")
            
            return u_data, v_data, w_data
            
        except ValueError as e:
            available_vars = list(dataset.data_vars.keys())
            missing_vars = []
            if u_variable not in available_vars:
                missing_vars.append(u_variable)
            if v_variable not in available_vars:
                missing_vars.append(v_variable)
            if w_variable not in available_vars:
                missing_vars.append(w_variable)
            
            if missing_vars:
                raise ValueError(f"3D wind variables {missing_vars} not found in dataset. Available variables: {available_vars[:20]}...")
            else:
                raise e

    def get_2d_wind_components(self, dataset: xr.Dataset, u_variable: str, v_variable: str, 
                               time_index: int = 0, data_type: str = 'xarray') -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Extract U and V wind components from 2D MPAS diagnostic dataset at specified time index. This method handles extraction of surface or constant-level wind components from MPAS diagnostic files with comprehensive error checking and validation. The extraction process validates variable availability, checks time index bounds, and supports both xarray and uxarray dataset types. If verbose mode is enabled, the method prints diagnostic information including component ranges, computed wind speed range, and unit verification with warnings for unit mismatches. The method raises specific exceptions for missing variables, invalid time indices, or dataset access errors to facilitate debugging.

        Parameters:
            dataset (xr.Dataset): MPAS 2D dataset containing wind variables.
            u_variable (str): Name of U-component wind variable (e.g., 'u10', 'u850').
            v_variable (str): Name of V-component wind variable (e.g., 'v10', 'v850').
            time_index (int): Time index to extract from dataset (default: 0).
            data_type (str): Dataset type specification, either 'xarray' or 'uxarray' (default: 'xarray').

        Returns:
            Tuple[xr.DataArray, xr.DataArray]: Two-element tuple containing (u_component, v_component) as xarray DataArrays with preserved CF-compliant metadata and computed values.

        Raises:
            ValueError: If wind variables are not found in dataset.
            IndexError: If time_index is out of range.
            RuntimeError: If dataset is None or other extraction errors occur.
        """
        from mpasdiag.processing.utils_datetime import MPASDateTimeUtils
        
        if dataset is None:
            raise RuntimeError("No dataset provided for wind component extraction.")
        
        if self.verbose:
            print(f"Extracting wind components {u_variable}, {v_variable} at time index {time_index}")
        
        available_vars = list(dataset.data_vars.keys())
        missing_vars = []
        
        if u_variable not in available_vars:
            missing_vars.append(u_variable)

        if v_variable not in available_vars:
            missing_vars.append(v_variable)
            
        if missing_vars:
            raise ValueError(f"Wind variables {missing_vars} not found in dataset. Available variables: {available_vars[:20]}...")
        
        time_dim, validated_time_index, time_size = MPASDateTimeUtils.validate_time_parameters(dataset, time_index, self.verbose)
        
        try:
            if data_type == 'uxarray':
                u_data = dataset[u_variable][validated_time_index]
                v_data = dataset[v_variable][validated_time_index]
            else:
                u_data = dataset[u_variable].isel({time_dim: validated_time_index})
                v_data = dataset[v_variable].isel({time_dim: validated_time_index})
            
            if hasattr(u_data, 'compute'):
                u_data = cast(Any, u_data).compute()

            if hasattr(v_data, 'compute'):
                v_data = cast(Any, v_data).compute()
            
            u_min, u_max = float(u_data.min()), float(u_data.max())
            v_min, v_max = float(v_data.min()), float(v_data.max())
            
            wind_speed = np.sqrt(u_data**2 + v_data**2)
            wind_min, wind_max = float(wind_speed.min()), float(wind_speed.max())
            
            if self.verbose:
                print(f"Wind component {u_variable} range: {u_min:.2f} to {u_max:.2f} m/s")
                print(f"Wind component {v_variable} range: {v_min:.2f} to {v_max:.2f} m/s") 
                print(f"Wind speed range: {wind_min:.2f} to {wind_max:.2f} m/s")
                
                u_units = u_data.attrs.get('units', WIND_SPEED_UNITS)
                v_units = v_data.attrs.get('units', WIND_SPEED_UNITS)

                print(f"Units: {u_units}")
                
                if u_units != v_units:
                    print(f"Warning: U and V components have different units: {u_units} vs {v_units}")
            
            return u_data, v_data
            
        except KeyError as e:
            raise ValueError(f"Error accessing wind variables: {e}")
        except Exception as e:
            raise RuntimeError(f"Error extracting 2D wind components: {e}")