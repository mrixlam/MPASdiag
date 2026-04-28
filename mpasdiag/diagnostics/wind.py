#!/usr/bin/env python3

"""
MPASdiag Core Diagnostics Module: Wind Diagnostics

This module defines the WindDiagnostics class, which provides methods for computing various diagnostics related to wind fields in MPAS model output. The class includes functions to calculate horizontal wind speed, wind direction, wind shear, and perform comprehensive analysis of wind components. It also includes helper methods for validating 3D variables, determining vertical dimensions, and extracting variable slices at specified levels. The diagnostics are designed to be flexible and informative, with options for verbose output to assist with debugging and understanding the data. The module relies on xarray for data handling and numpy for numerical operations, and it is structured to integrate seamlessly with the overall MPASdiag processing framework. 
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import numpy as np
import xarray as xr
from typing import Tuple, Union, Optional, Any, cast, Dict

from mpasdiag.processing.constants import WIND_SPEED_UNITS


class WindDiagnostics:
    """ Computes diagnostics related to wind fields in MPAS model output, including wind speed, direction, shear, and component analysis. """
    
    def __init__(self: 'WindDiagnostics', 
                 verbose: bool = True) -> None:
        """
        This constructor initializes the WindDiagnostics class with an optional verbose flag that enables detailed output messages during wind calculations. The verbose mode provides insights into the ranges and means of computed wind diagnostics, which can be helpful for identifying potential issues with the input data or understanding the characteristics of the wind fields being analyzed. By default, verbose mode is enabled, but it can be turned off if a quieter output is preferred during processing. 

        Parameters:
            verbose (bool): Enable verbose output messages during wind calculations (default: True).

        Returns:
            None: This constructor does not return a value.
        """
        self.verbose = verbose
    
    def compute_wind_speed(self: 'WindDiagnostics', 
                           u_component: xr.DataArray, 
                           v_component: xr.DataArray) -> xr.DataArray:
        """
        This function computes the horizontal wind speed magnitude from the U and V wind components using the Pythagorean theorem. The wind speed is calculated as the square root of the sum of squared U and V components. The resulting DataArray includes CF-compliant metadata for units, standard name, and long name to ensure that it is properly described for downstream analysis and visualization. If verbose mode is enabled, it prints the range and mean of the computed wind speed to help identify potential issues with the input data or calculation. 

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
            'long_name': 'Horizontal Wind Speed',
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
    
    def compute_wind_direction(self: 'WindDiagnostics', 
                               u_component: xr.DataArray, 
                               v_component: xr.DataArray, 
                               degrees: bool = True) -> xr.DataArray:
        """
        This function computes the wind direction from the U and V wind components using the arctangent function. The direction is calculated in meteorological convention, where 0 degrees corresponds to wind coming from the north, 90 degrees from the east, 180 degrees from the south, and 270 degrees from the west. The function can return the direction in either degrees or radians based on the `degrees` parameter. The resulting DataArray includes CF-compliant metadata for units, standard name, long name, and a note describing the convention used for wind direction. If verbose mode is enabled, it prints the range and mean of the computed wind direction to help identify potential issues with the input data or calculation. 

        Parameters:
            u_component (xr.DataArray): U (zonal) wind component in meters per second.
            v_component (xr.DataArray): V (meridional) wind component in meters per second.
            degrees (bool): Return wind direction in degrees if True, or radians if False (default: True). 

        Returns:
            xr.DataArray: Wind direction in degrees or radians with CF-compliant attributes. 
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
                'long_name': 'Wind Direction (meteorological convention)',
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
                'long_name': 'Wind Direction (meteorological convention)',
                'note': 'Direction wind is coming from, 0=North, π/2=East, π=South, 3π/2=West'
            })
            
            if self.verbose:
                dir_min, dir_max = float(direction_rad.min()), float(direction_rad.max())
                dir_mean = float(direction_rad.mean())
                print(f"Wind direction range: {dir_min:.3f} to {dir_max:.3f} radians")
                print(f"Wind direction mean: {dir_mean:.3f} radians")
            
            return direction_rad
    
    def analyze_wind_components(self: 'WindDiagnostics', 
                                u_component: xr.DataArray, 
                                v_component: xr.DataArray, 
                                w_component: Optional[xr.DataArray] = None) -> Dict[str, Any]:
        """
        This function performs a comprehensive analysis of the U, V, and optionally W wind components by calculating summary statistics (minimum, maximum, mean, standard deviation) for each component, as well as the horizontal wind speed and wind direction derived from U and V. If the W component is provided, it also calculates the total 3D wind speed. The results are returned in a dictionary format that includes the computed statistics and units for each variable. If verbose mode is enabled, it prints the ranges and means of the components, horizontal speed, and direction to help identify potential issues with the input data or calculations. This function provides a convenient way to summarize key characteristics of the wind fields for diagnostic purposes. 

        Parameters:
            u_component (xr.DataArray): U (zonal) wind component in meters per second.
            v_component (xr.DataArray): V (meridional) wind component in meters per second.
            w_component (Optional[xr.DataArray]): W (vertical) wind component in meters per second (optional).

        Returns:
            Dict[str, Any]: A dictionary containing summary statistics and units for U, V, W (if provided), horizontal wind speed, and wind direction.
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
    
    def compute_wind_shear(self: 'WindDiagnostics', 
                           u_upper: xr.DataArray, 
                           v_upper: xr.DataArray, 
                           u_lower: xr.DataArray, 
                           v_lower: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        This function computes the wind shear magnitude and direction between two vertical levels using the U and V wind components at the upper and lower levels. The shear magnitude is calculated as the square root of the sum of squared differences in U and V components between the two levels. The shear direction is computed using the arctangent function to determine the direction of the shear vector, following meteorological convention. Both the shear magnitude and direction are returned as xarray DataArrays with CF-compliant attributes for units, standard name, and long name. If verbose mode is enabled, it prints the range and mean of the computed wind shear magnitude to help identify potential issues with the input data or calculation. 

        Parameters:
            u_upper (xr.DataArray): U component at the upper level in meters per second.
            v_upper (xr.DataArray): V component at the upper level in meters per second.
            u_lower (xr.DataArray): U component at the lower level in meters per second.
            v_lower (xr.DataArray): V component at the lower level in meters per second. 

        Returns:
            Tuple[xr.DataArray, xr.DataArray]: A tuple containing the wind shear magnitude (in m/s) and shear direction (in degrees) as xarray DataArrays with CF-compliant attributes.
        """
        delta_u = u_upper - u_lower
        delta_v = v_upper - v_lower

        shear_magnitude = xr.apply_ufunc(
            np.sqrt,
            delta_u**2 + delta_v**2,
            keep_attrs=True
        )
        shear_magnitude.attrs.update({
            'units': WIND_SPEED_UNITS,
            'standard_name': 'wind_shear_magnitude',
            'long_name': 'wind shear magnitude',
        })

        shear_direction = self.compute_wind_direction(delta_u, delta_v, degrees=True)
        shear_direction.attrs['long_name'] = 'wind shear direction'
        
        if self.verbose:
            shear_min, shear_max = float(shear_magnitude.min()), float(shear_magnitude.max())
            shear_mean = float(shear_magnitude.mean())
            print(f"Wind shear magnitude range: {shear_min:.2f} to {shear_max:.2f} m/s")
            print(f"Wind shear magnitude mean: {shear_mean:.2f} m/s")
        
        return shear_magnitude, shear_direction

    def _validate_3d_variable(self: 'WindDiagnostics', 
                              dataset: xr.Dataset, 
                              var_name: str) -> None:
        """
        This helper function checks whether the specified variable in the dataset is a 3D atmospheric variable by verifying that it contains either 'nVertLevels' or 'nVertLevelsP1' as one of its dimensions. If the variable is not found in the dataset or does not have the required vertical dimension, the function raises a ValueError with an appropriate message. This validation step is crucial to ensure that subsequent calculations that rely on vertical indexing are performed on compatible variables, thereby preventing errors and ensuring the integrity of the diagnostics. 

        Parameters:
            dataset (xr.Dataset): Input dataset containing the variable.
            var_name (str): Variable name to validate.

        Returns:
            None: This function does not return a value but raises an error if validation fails.
        """
        if var_name not in dataset.data_vars:
            raise ValueError(f"Variable '{var_name}' not found in dataset")
        
        var_dims = dataset[var_name].dims

        if 'nVertLevels' not in var_dims and 'nVertLevelsP1' not in var_dims:
            raise ValueError(f"Variable '{var_name}' is not a 3D atmospheric variable")

    def _get_vertical_dimension(self: 'WindDiagnostics', 
                                dataset: xr.Dataset, 
                                var_name: str) -> str:
        """
        This helper function determines the name of the vertical dimension for a given variable in the dataset by checking for the presence of 'nVertLevels' or 'nVertLevelsP1' in the variable's dimensions. It returns the name of the vertical dimension that is found, which can then be used for indexing and slicing operations in subsequent calculations. This function assumes that the variable has already been validated as a 3D atmospheric variable using the `_validate_3d_variable` method, ensuring that one of the expected vertical dimensions is present. 

        Parameters:
            dataset (xr.Dataset): Input dataset containing the variable.
            var_name (str): Variable name for which to determine the vertical dimension.

        Returns:
            str: The name of the vertical dimension ('nVertLevels' or 'nVertLevelsP1').
        """
        var_dims = dataset[var_name].dims
        return 'nVertLevels' if 'nVertLevels' in var_dims else 'nVertLevelsP1'

    def _compute_level_index_from_pressure(self: 'WindDiagnostics', 
                                           dataset: xr.Dataset, 
                                           pressure_level: float, 
                                           time_dim: str, 
                                           time_index: int) -> int:
        """
        This helper function computes the vertical level index corresponding to a specified pressure level in Pascals by using the pressure diagnostics available in the dataset. It first checks for the presence of 'pressure_p' and 'pressure_base' variables, which are necessary to calculate the total pressure at each vertical level. The function then calculates the mean pressure across all cells for each vertical level and finds the level index where the mean pressure is closest to the specified target pressure level. If verbose mode is enabled, it prints the requested pressure and the actual pressure at the selected level to help verify that the correct level has been identified. This function allows users to specify vertical levels based on physical pressure values rather than model level indices, providing greater flexibility in selecting levels for diagnostics. 

        Parameters:
            dataset (xr.Dataset): Input dataset containing pressure diagnostics.
            pressure_level (float): Target pressure level in Pascals for which to find the corresponding vertical index.
            time_dim (str): Name of the time dimension in the dataset.
            time_index (int): Time index to use for selecting the pressure data.

        Returns:
            int: The vertical level index corresponding to the specified pressure level.
        """
        if 'pressure_p' not in dataset or 'pressure_base' not in dataset:
            raise ValueError("Cannot find pressure level - pressure data not available")
        
        pressure_p = dataset['pressure_p'].isel({time_dim: time_index})
        pressure_base = dataset['pressure_base'].isel({time_dim: time_index})
        total_pressure = pressure_p + pressure_base
        
        mean_pressure = total_pressure.mean(dim='nCells')
        pressure_diff = np.abs(mean_pressure - pressure_level)
        level_idx = int(pressure_diff.argmin())
        
        if self.verbose:
            actual_pressure_at_level = mean_pressure.isel(nVertLevels=level_idx).values
            print(f"Requested pressure: {pressure_level:.1f} Pa, using level {level_idx}: {actual_pressure_at_level:.1f} Pa")
        
        return level_idx

    def _compute_level_index(self: 'WindDiagnostics', 
                             dataset: xr.Dataset, 
                             var_name: str, 
                             level_spec: Union[str, int, float], 
                             time_dim: str, 
                             time_index: int) -> int:
        """
        This helper function computes the vertical level index based on the provided specification, which can be an integer index, a pressure value in Pascals, or a string identifier such as 'surface' or 'top'. If the specification is an integer, it checks that the index is within the available levels in the dataset. If it is a float, it calls the `_compute_level_index_from_pressure` method to find the corresponding level index based on pressure diagnostics. If it is a string, it checks for 'surface' (which corresponds to index 0) and 'top' (which corresponds to the maximum vertical level index). If the specification does not match any of these formats, it raises a ValueError. This function provides a flexible way to specify vertical levels for diagnostics, allowing users to choose levels based on their preferred method of identification. 

        Parameters:
            dataset (xr.Dataset): Input dataset containing the variable and pressure diagnostics.
            var_name (str): Name of the variable for which to compute the level index.
            level_spec (Union[str, int, float]): Specification of the vertical level (integer index, pressure in Pa, or 'surface'/'top').
            time_dim (str): Name of the time dimension in the dataset.
            time_index (int): Time index to use for selecting pressure data if needed.

        Returns:
            int: The computed vertical level index based on the provided specification.
        """
        vertical_dim = self._get_vertical_dimension(dataset, var_name)
        
        if isinstance(level_spec, int):
            level_idx = level_spec
            max_levels = dataset.sizes.get(vertical_dim, 0)
            if level_idx >= max_levels:
                raise ValueError(f"Model level {level_idx} exceeds available levels {max_levels}")
            return level_idx
            
        elif isinstance(level_spec, float):
            return self._compute_level_index_from_pressure(dataset, level_spec, time_dim, time_index)
            
        elif isinstance(level_spec, str):
            if level_spec.lower() == 'surface':
                return 0
            elif level_spec.lower() == 'top':
                return dataset.sizes.get(vertical_dim, 1) - 1
            else:
                raise ValueError(f"Unknown level specification: {level_spec}")
        else:
            raise ValueError(f"Invalid level specification: {level_spec}")

    def _extract_variable_slice(self: 'WindDiagnostics', 
                                dataset: xr.Dataset, 
                                var_name: str, 
                                time_dim: str, 
                                time_index: int, 
                                vertical_dim: str, 
                                level_idx: int, 
                                data_type: str) -> xr.DataArray:
        """
        This helper function extracts a slice of the specified variable from the dataset at the given time index and vertical level index. It handles both 'xarray' and 'uxarray' dataset access styles by checking for the presence of the `__getitem__` method when using 'uxarray'. The function uses the `isel` method to select the appropriate time and vertical level indices, and it ensures that any lazy computations are executed by calling `compute()` if the resulting DataArray has that method. The extracted variable slice is returned as an xarray DataArray, ready for further analysis and diagnostics. 

        Parameters:
            dataset (xr.Dataset): Input dataset containing the variable.
            var_name (str): Name of the variable to extract.
            time_dim (str): Name of the time dimension in the dataset.
            time_index (int): Time index to select.
            vertical_dim (str): Name of the vertical dimension in the dataset.
            level_idx (int): Vertical level index to select.
            data_type (str): Dataset access style, either 'xarray' or 'uxarray'. 

        Returns:
            xr.DataArray: The extracted variable slice at the specified time and vertical level, ready for analysis.
        """
        if data_type == 'uxarray' and hasattr(dataset, '__getitem__'):
            var_data = dataset[var_name][time_index].isel({vertical_dim: level_idx})
        else:
            var_data = dataset[var_name].isel({time_dim: time_index, vertical_dim: level_idx})
        
        if hasattr(var_data, 'compute'):
            var_data = cast(Any, var_data).compute()
        
        return var_data

    def get_3d_variable_at_level(self: 'WindDiagnostics', 
                                 dataset: xr.Dataset, 
                                 var_name: str, 
                                 level: Union[str, int, float], 
                                 time_index: int, 
                                 data_type: str = 'xarray') -> xr.DataArray:
        """
        This function retrieves a 3D variable from the dataset at a specified vertical level and time index, with support for both 'xarray' and 'uxarray' dataset access styles. It first validates that the variable is a 3D atmospheric variable using the `_validate_3d_variable` method. Then it determines the time dimension and validates the time index using the `MPASDateTimeUtils.validate_time_parameters` method. Next, it computes the vertical level index based on the provided level specification (which can be an integer index, pressure in Pascals, or 'surface'/'top') using the `_compute_level_index` method. Finally, it extracts the variable slice at the specified time and vertical level using the `_extract_variable_slice` method and returns it as an xarray DataArray with metadata attributes indicating the selected level and index. This function provides a flexible and robust way to access 3D variables at specific levels for wind diagnostics and other analyses. 

        Parameters:
            dataset (xr.Dataset): Input dataset containing the variable.
            var_name (str): Name of the variable to retrieve.
            level (Union[str, int, float]): Specification of the vertical level for retrieval (integer index, pressure in Pa, or 'surface'/'top').
            time_index (int): Time index to select for retrieval.
            data_type (str): Dataset access style, either 'xarray' or 'uxarray' (default: 'xarray'). 

        Returns:
            xr.DataArray: The extracted variable slice at the specified time and vertical level, with metadata attributes for the selected level and index. 
        """
        from mpasdiag.processing.utils_datetime import MPASDateTimeUtils
        
        self._validate_3d_variable(dataset, var_name)
        
        time_dim, validated_time_index, _ = MPASDateTimeUtils.validate_time_parameters(
            dataset, time_index, self.verbose
        )
        
        level_idx = self._compute_level_index(dataset, var_name, level, time_dim, validated_time_index)
        vertical_dim = self._get_vertical_dimension(dataset, var_name)
        
        var_data = self._extract_variable_slice(
            dataset, var_name, time_dim, validated_time_index, 
            vertical_dim, level_idx, data_type
        )
        
        var_data.attrs['selected_level'] = level
        var_data.attrs['level_index'] = level_idx
        
        return var_data

    def _extract_w_component_with_fallback(self: 'WindDiagnostics', 
                                           dataset: xr.Dataset, 
                                           w_variable: str, 
                                           level: Union[str, int, float], 
                                           time_index: int, 
                                           data_type: str, 
                                           u_data: xr.DataArray) -> xr.DataArray:
        """
        This function attempts to extract the W (vertical) wind component from the dataset at a specified vertical level and time index. If the extraction is successful, it returns the W component as an xarray DataArray. However, if the extraction fails due to missing variables, invalid level specifications, or other issues, it catches the exception and creates a fallback W component filled with zeros that has the same shape and metadata as the U component. The fallback W component is assigned appropriate attributes to indicate that it is a zero vertical velocity and includes a note about the failed extraction. This approach ensures that downstream diagnostics can still be performed even if the W component is not available in the dataset, while providing clear information about the reason for using the fallback data. 

        Parameters:
            dataset (xr.Dataset): Input dataset containing the wind variables.
            w_variable (str): Name of the W-component variable to extract.
            level (Union[str, int, float]): Specification of the vertical level for extraction (integer index, pressure in Pa, or 'surface'/'top').
            time_index (int): Time index to select for extraction.
            data_type (str): Dataset access style, either 'xarray' or 'uxarray'.
            u_data (xr.DataArray): The U component DataArray used to create a zero-filled fallback if W extraction fails. 

        Returns:
            xr.DataArray: The extracted W component if successful, or a zero-filled fallback DataArray with appropriate attributes if extraction fails. 
        """
        try:
            return self.get_3d_variable_at_level(dataset, w_variable, level, time_index, data_type)
        except (ValueError, IndexError) as e:
            if self.verbose:
                print(f"Warning: Could not extract {w_variable} at level {level}: {e}")
                print("Setting W component to zero...")
            w_data = xr.zeros_like(u_data)
            w_data.attrs['units'] = WIND_SPEED_UNITS
            w_data.attrs['long_name'] = f'Zero vertical velocity (could not extract {w_variable})'
            return w_data

    def _print_wind_component_diagnostics(self: 'WindDiagnostics', 
                                          u_data: xr.DataArray, 
                                          v_data: xr.DataArray, 
                                          w_data: xr.DataArray, 
                                          u_variable: str, 
                                          v_variable: str, 
                                          w_variable: str) -> None:
        """
        This helper function prints diagnostic information about the extracted U, V, and W wind components, including their ranges and units. It calculates the horizontal wind speed from the U and V components and also prints its range. The function checks if verbose mode is enabled before printing any information, allowing users to control the level of output during processing. This diagnostic output can help identify potential issues with the extracted wind components, such as unexpected ranges or missing data, and provides context for understanding the characteristics of the wind fields being analyzed. 

        Parameters:
            u_data (xr.DataArray): Extracted U component DataArray.
            v_data (xr.DataArray): Extracted V component DataArray.
            w_data (xr.DataArray): Extracted W component DataArray (or fallback).
            u_variable (str): Name of the U component variable.
            v_variable (str): Name of the V component variable.
            w_variable (str): Name of the W component variable.

        Returns:
            None: This function does not return a value but prints diagnostic information to the console if verbose mode is enabled. 
        """
        if not self.verbose:
            return
        
        wind_speed = np.sqrt(u_data**2 + v_data**2)
        
        u_min, u_max = float(u_data.min()), float(u_data.max())
        v_min, v_max = float(v_data.min()), float(v_data.max())
        w_min, w_max = float(w_data.min()), float(w_data.max())
        wind_min, wind_max = float(wind_speed.min()), float(wind_speed.max())
        
        print(f"Wind component {u_variable} range: {u_min:.2f} to {u_max:.2f} m/s")
        print(f"Wind component {v_variable} range: {v_min:.2f} to {v_max:.2f} m/s") 
        print(f"Wind component {w_variable} range: {w_min:.2f} to {w_max:.2f} m/s")
        print(f"Horizontal wind speed range: {wind_min:.2f} to {wind_max:.2f} m/s")
        
        u_units = u_data.attrs.get('units', WIND_SPEED_UNITS)
        print(f"Units: {u_units}")

    def get_3d_wind_components(self: 'WindDiagnostics', 
                               dataset: xr.Dataset, 
                               u_variable: str, 
                               v_variable: str, 
                               w_variable: str = 'w', 
                               level: Union[str, int, float] = 0, 
                               time_index: int = 0, 
                               data_type: str = 'xarray') -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        This function extracts the U, V, and W wind components from a 3D MPAS dataset at a specified vertical level and time index. It validates the presence of the specified variables in the dataset, checks that the time index is within bounds, and retrieves the variable slices for each component using the `get_3d_variable_at_level` method. If the W component cannot be extracted due to missing variables or invalid level specifications, it creates a fallback W component filled with zeros using the `_extract_w_component_with_fallback` method. The function includes diagnostic print statements that provide information about the ranges of the extracted components and their units if verbose mode is enabled. The resulting U, V, and W components are returned as xarray DataArrays with CF-compliant attributes and metadata for the selected level. If any issues arise during extraction (e.g., missing variables, out-of-range time index), it raises appropriate exceptions with informative messages to help diagnose the problem. 

        Parameters:
            dataset (xr.Dataset): Input dataset containing the wind variables.
            u_variable (str): Name of the U-component variable to extract.
            v_variable (str): Name of the V-component variable to extract.
            w_variable (str): Name of the W-component variable to extract (default: 'w').
            level (Union[str, int, float]): Specification of the vertical level for extraction (integer index, pressure in Pa, or 'surface'/'top') (default: 0).
            time_index (int): Time index to select for extraction (default: 0).
            data_type (str): Dataset access style, either 'xarray' or 'uxarray' (default: 'xarray'). 

        Returns:
            Tuple[xr.DataArray, xr.DataArray, xr.DataArray]: A tuple containing the extracted U, V, and W components as xarray DataArrays with CF-compliant attributes and metadata for the selected level.
        """
        if self.verbose:
            print(f"Extracting 3D wind components {u_variable}, {v_variable}, {w_variable} at level {level}, time index {time_index}")
        
        try:
            u_data = self.get_3d_variable_at_level(dataset, u_variable, level, time_index, data_type)
            v_data = self.get_3d_variable_at_level(dataset, v_variable, level, time_index, data_type)
            w_data = self._extract_w_component_with_fallback(dataset, w_variable, level, time_index, data_type, u_data)
            
            self._print_wind_component_diagnostics(u_data, v_data, w_data, u_variable, v_variable, w_variable)
            
            return u_data, v_data, w_data
            
        except ValueError as e:
            available_vars = list(dataset.data_vars.keys())
            missing_vars = [var for var in [u_variable, v_variable, w_variable] 
                           if var not in available_vars]
            
            if missing_vars:
                raise ValueError(f"3D wind variables {missing_vars} not found in dataset. Available variables: {available_vars[:20]}...")
            else:
                raise e

    def get_2d_wind_components(self: 'WindDiagnostics', 
                               dataset: xr.Dataset, 
                               u_variable: str, 
                               v_variable: str, 
                               time_index: int = 0, 
                               data_type: str = 'xarray') -> Tuple[xr.DataArray, xr.DataArray]:
        """
        This function extracts the U and V wind components from a 2D MPAS dataset at a specified time index. It validates the presence of the specified variables in the dataset and checks that the time index is within bounds. The function retrieves the variable slices for U and V components using either direct indexing for 'uxarray' datasets or the `isel` method for 'xarray' datasets. It ensures that any lazy computations are executed by calling `compute()` if necessary. The function includes diagnostic print statements that provide information about the ranges of the extracted components and their units if verbose mode is enabled. The resulting U and V components are returned as xarray DataArrays with CF-compliant attributes. If any issues arise during extraction (e.g., missing variables, out-of-range time index), it raises appropriate exceptions with informative messages to help diagnose the problem. 

        Parameters:
            dataset (xr.Dataset): Input dataset containing the wind variables.
            u_variable (str): Name of the U-component variable to extract.
            v_variable (str): Name of the V-component variable to extract.
            time_index (int): Time index to select for extraction (default: 0).
            data_type (str): Dataset access style, either 'xarray' or 'uxarray' (default: 'xarray'). 

        Returns:
            Tuple[xr.DataArray, xr.DataArray]: A tuple containing the extracted U and V components as xarray DataArrays with CF-compliant attributes. 
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