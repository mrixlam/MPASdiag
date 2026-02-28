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
from typing import Tuple, Union, Optional, Any, cast, Dict

WIND_SPEED_UNITS = 'm s^{-1}'


class WindDiagnostics:
    """
    This class provides methods for wind component analysis, wind speed calculations, and wind direction computations from MPAS model output.
    """
    
    def __init__(self, verbose: bool = True) -> None:
        """
        This constructor configures the diagnostics object with verbosity controls used across wind processing methods. Verbose mode enables runtime diagnostic printing useful during debugging and interactive use. The instance is lightweight and stateless aside from the `verbose` attribute.

        Parameters:
            verbose (bool): Enable verbose output messages during wind calculations (default: True).

        Returns:
            None: This constructor does not return a value.
        """
        self.verbose = verbose
    
    def compute_wind_speed(self, u_component: xr.DataArray, v_component: xr.DataArray) -> xr.DataArray:
        """
        This function computes the Euclidean magnitude of the horizontal wind vector (sqrt(u^2 + v^2)) and preserves xarray attributes on the result. It sets CF-like metadata including `units`, `standard_name`, and `long_name`. When verbose is enabled the method prints summary statistics for inputs and the resulting speed.

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
        The function computes the angle from the wind vector using arctan2 and converts it to the meteorological convention (direction the wind is coming from). Results may be returned in degrees (default) or radians depending on the `degrees` flag. Xarray attributes are preserved and a descriptive `note` attribute is added. If verbose is enabled, summary statistics are printed.

        Parameters:
            u_component (xr.DataArray): U (zonal) wind component in meters per second.
            v_component (xr.DataArray): V (meridional) wind component in meters per second.
            degrees (bool): Return direction in degrees if True, radians if False (default: True).

        Returns:
            xr.DataArray: Wind direction with CF-like metadata. In degrees: 0=North, 90=East, 180=South, 270=West; in radians: 0=North, π/2=East, π=South, 3π/2=West.
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
                               w_component: Optional[xr.DataArray] = None) -> Dict[str, Any]:
        """
        This routine computes min, max, mean, and standard deviation for U and V components and derives horizontal wind speed and meteorological direction. If a vertical component `w_component` is supplied, full 3D speed statistics are computed and included in the result. The output is a dictionary keyed by variable name containing numeric summaries and units.

        Parameters:
            u_component (xr.DataArray): U (zonal) wind component in meters per second.
            v_component (xr.DataArray): V (meridional) wind component in meters per second.
            w_component (Optional[xr.DataArray]): W (vertical) wind component in meters per second (default: None).

        Returns:
            Dict[str, Any]: Dictionary of analysis results. Keys include 'u_component', 'v_component', 'horizontal_speed', 'direction', and optionally 'w_component' and 'total_speed'. Each entry contains summary statistics (`min`, `max`, `mean`, `std`) and `units`.
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
        The method computes the vector difference (upper - lower) for U and V components, derives the Euclidean shear magnitude, and computes the meteorological shear direction in degrees. Both outputs preserve xarray metadata and include CF-like attributes for units and descriptive names.

        Parameters:
            u_upper (xr.DataArray): U component at upper level in meters per second.
            v_upper (xr.DataArray): V component at upper level in meters per second.
            u_lower (xr.DataArray): U component at lower level in meters per second.
            v_lower (xr.DataArray): V component at lower level in meters per second.

        Returns:
            Tuple[xr.DataArray, xr.DataArray]: (shear_magnitude, shear_direction) where magnitude is in meters per second and direction is in degrees using meteorological convention.
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

    def _validate_3d_variable(self, dataset: xr.Dataset, var_name: str) -> None:
        """
        Checks that the variable name is present in `dataset.data_vars` and that it contains a recognized vertical dimension ('nVertLevels' or 'nVertLevelsP1'). If either check fails, a ValueError is raised. 

        Parameters:
            dataset (xr.Dataset): Input xarray dataset containing variables.
            var_name (str): Variable name to validate.

        Returns:
            None: Raises ValueError if validation fails.
        """
        if var_name not in dataset.data_vars:
            raise ValueError(f"Variable '{var_name}' not found in dataset")
        
        var_dims = dataset[var_name].dims

        if 'nVertLevels' not in var_dims and 'nVertLevelsP1' not in var_dims:
            raise ValueError(f"Variable '{var_name}' is not a 3D atmospheric variable")

    def _get_vertical_dimension(self, dataset: xr.Dataset, var_name: str) -> str:
        """
        The function inspects the variable dimensions and returns 'nVertLevels' when present, otherwise 'nVertLevelsP1'.

        Parameters:
            dataset (xr.Dataset): Input dataset containing the variable.
            var_name (str): Variable name to inspect.

        Returns:
            str: The vertical dimension name ('nVertLevels' or 'nVertLevelsP1').
        """
        var_dims = dataset[var_name].dims
        return 'nVertLevels' if 'nVertLevels' in var_dims else 'nVertLevelsP1'

    def _compute_level_index_from_pressure(self, dataset: xr.Dataset, pressure_level: float, 
                                          time_dim: str, time_index: int) -> int:
        """
        This helper reads `pressure_p` and `pressure_base` from the dataset, forms the full pressure profile, computes the mean pressure over cells, and finds the vertical index whose mean pressure is closest to the requested value. If verbose is enabled, it prints the requested pressure and the actual pressure at the selected level. 

        Parameters:
            dataset (xr.Dataset): Dataset containing pressure diagnostics.
            pressure_level (float): Requested pressure level in Pascals.
            time_dim (str): Time dimension name in the dataset.
            time_index (int): Time index to select.

        Returns:
            int: Vertical level index corresponding to the nearest pressure level.
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
            target_p = mean_pressure.isel(nVertLevels=level_idx).values
            print(f"Requested pressure: {pressure_level:.1f} Pa, using level {level_idx}: {target_p:.1f} Pa")
        
        return level_idx

    def _compute_level_index(self, dataset: xr.Dataset, var_name: str, 
                            level_spec: Union[str, int, float],
                            time_dim: str, time_index: int) -> int:
        """
        Accepted `level_spec` formats are integer model levels, float pressure values (in Pa) which are resolved via pressure diagnostics, or string identifiers 'surface' and 'top'. Validation against available levels is performed and a ValueError is raised for invalid specifications.

        Parameters:
            dataset (xr.Dataset): Dataset containing the variable and vertical dims.
            var_name (str): Variable name used to determine vertical dimension.
            level_spec (Union[str, int, float]): Level specifier (index, pressure in Pa, or 'surface'/'top').
            time_dim (str): Time dimension name.
            time_index (int): Time index to use for pressure-based lookup.

        Returns:
            int: Resolved model vertical level index.
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

    def _extract_variable_slice(self, dataset: xr.Dataset, var_name: str, 
                               time_dim: str, time_index: int, 
                               vertical_dim: str, level_idx: int,
                               data_type: str) -> xr.DataArray:
        """
        Supports both 'xarray' and 'uxarray' style access patterns and calls `.compute()` when the returned object exposes that method. The returned DataArray preserves attributes and is annotated with selected level info by the calling functions.

        Parameters:
            dataset (xr.Dataset): Source dataset containing the variable.
            var_name (str): Name of the variable to extract.
            time_dim (str): Name of the time dimension.
            time_index (int): Time index to select.
            vertical_dim (str): Name of the vertical dimension.
            level_idx (int): Vertical level index to select.
            data_type (str): Dataset access style, either 'xarray' or 'uxarray'.

        Returns:
            xr.DataArray: Extracted DataArray for the requested time and level.
        """
        if data_type == 'uxarray' and hasattr(dataset, '__getitem__'):
            var_data = dataset[var_name][time_index].isel({vertical_dim: level_idx})
        else:
            var_data = dataset[var_name].isel({time_dim: time_index, vertical_dim: level_idx})
        
        if hasattr(var_data, 'compute'):
            var_data = cast(Any, var_data).compute()
        
        return var_data

    def get_3d_variable_at_level(self, dataset: xr.Dataset, var_name: str, 
                                level: Union[str, int, float], 
                                time_index: int, data_type: str = 'xarray') -> xr.DataArray:
        """
        The function accepts integer model levels, float pressure values (Pa), or string identifiers ('surface', 'top') for `level`. It validates that the target variable is 3D, resolves the correct vertical index, and returns a CF-compliant xarray DataArray annotated with `selected_level` and `level_index` attributes.

        Parameters:
            dataset (xr.Dataset): MPAS 3D dataset containing the variable.
            var_name (str): Name of the variable to extract.
            level (Union[str, int, float]): Vertical level (index, pressure Pa, or 'surface'/'top').
            time_index (int): Time index to extract from dataset.
            data_type (str): Dataset type specification, either 'xarray' or 'uxarray' (default: 'xarray').

        Returns:
            xr.DataArray: Extracted variable data at the specified level and time with metadata attributes `selected_level` and `level_index`.

        Raises:
            ValueError: If the variable is not found or is not 3D.
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

    def _extract_w_component_with_fallback(self, dataset: xr.Dataset, w_variable: str,
                                          level: Union[str, int, float], time_index: int,
                                          data_type: str, u_data: xr.DataArray) -> xr.DataArray:
        """
        If extraction of `w_variable` fails (missing variable or index errors), the function returns a zero-filled DataArray matching `u_data` shape and assigns descriptive metadata explaining the fallback.

        Parameters:
            dataset (xr.Dataset): Source dataset to extract from.
            w_variable (str): Name of the vertical velocity variable.
            level (Union[str, int, float]): Vertical level specification.
            time_index (int): Time index to extract.
            data_type (str): Dataset access style ('xarray' or 'uxarray').
            u_data (xr.DataArray): Reference DataArray used to shape the fallback zeros.

        Returns:
            xr.DataArray: Extracted W component or zero-filled fallback with metadata.
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

    def _print_wind_component_diagnostics(self, u_data: xr.DataArray, v_data: xr.DataArray,
                                         w_data: xr.DataArray, u_variable: str, 
                                         v_variable: str, w_variable: str) -> None:
        """
        Displays min/max ranges for U, V, and W components and horizontal wind speed summary. Also prints the units attribute extracted from `u_data`.

        Parameters:
            u_data (xr.DataArray): Extracted U-component DataArray.
            v_data (xr.DataArray): Extracted V-component DataArray.
            w_data (xr.DataArray): Extracted W-component DataArray.
            u_variable (str): Name of the U variable used for messaging.
            v_variable (str): Name of the V variable used for messaging.
            w_variable (str): Name of the W variable used for messaging.

        Returns:
            None: Diagnostic printing only.
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